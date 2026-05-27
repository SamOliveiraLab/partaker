import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Union, Sequence, Optional

import dask.array as da
import nd2
import numpy as np
from pubsub import pub

from partaker.analysis.segmentation.segmentation_cache import SegmentationCache
from partaker.analysis.segmentation.segmentation_models import SegmentationModels
from partaker.analysis.segmentation.segmentation_service import SegmentationService
from partaker.utils.registration import register_images, ShiftedImage_2D_numba


"""
Singleton, implements data access for the time-lapse experiments.

Supports both ND2 and TIFF (including OME-TIFF) inputs. All file formats are
normalized to a 5D dask array of shape (T, P, C, Y, X) before being stored.
"""


_TIFF_EXTENSIONS = (
    ".tif", ".tiff",
    ".ome.tif", ".ome.tiff",
    ".ome.btf", ".ome.tf2", ".ome.tf8",
)


def _is_tiff_path(path: str) -> bool:
    return path.lower().endswith(_TIFF_EXTENSIONS)


def _is_nd2_path(path: str) -> bool:
    return path.lower().endswith(".nd2")


class ImageData:
    _instance: Optional["ImageData"] = None
    _lock = threading.Lock()

    def __init__(self, data, path, is_image=True, channel_n=0):
        self.data = data
        self.image_filename = path
        self.processed_images = []
        self.is_image = is_image
        self.registration_offsets: Optional[np.ndarray] = (
            None  # If we are doing registration
        )
        self.crop_coordinates = None
        self.channel_n = channel_n

        # Optional, set by load_tiff_directory:
        self._memmap_file: Optional[str] = None
        self._tiff_file_map: Optional[dict] = None
        self._tiff_mode: Optional[str] = None
        self.voxel_size = None

        # Initialize segmentation components
        self.segmentation_cache = SegmentationCache(data)
        self.segmentation_service = SegmentationService(
            cache=self.segmentation_cache,
            models=SegmentationModels(),
            data_getter=self.get,
        )

        pub.subscribe(self.on_crop_selected, "crop_selected")
        pub.subscribe(self.on_crop_reset, "crop_reset")

    # ── Backward-compat aliases ─────────────────────────────────────
    # Older code referred to `is_nd2`/`nd2_filename`. Keep both surfaces
    # working so the rest of the app can be migrated incrementally.
    @property
    def is_nd2(self) -> bool:
        """Deprecated alias for `is_image`."""
        return self.is_image

    @is_nd2.setter
    def is_nd2(self, value: bool) -> None:
        self.is_image = value

    @property
    def nd2_filename(self):
        """Deprecated alias for `image_filename`."""
        return self.image_filename

    @nd2_filename.setter
    def nd2_filename(self, value) -> None:
        self.image_filename = value

    # ── Singleton plumbing ───────────────────────────────────────────
    @classmethod
    def get_instance(cls) -> Optional["ImageData"]:
        """Get the current singleton instance"""
        with cls._lock:
            return cls._instance

    @classmethod
    def create_instance(cls, data, path, is_image=True, channel_n=0) -> "ImageData":
        """Create/replace the singleton instance.

        Reads voxel size from the first source file when possible.
        """
        with cls._lock:
            # Clean up old instance if it exists
            if cls._instance is not None:
                cls._instance._cleanup()

            # Normalize first path (path may be a list).
            first_path = path[0] if isinstance(path, (list, tuple)) else path
            first_path = str(first_path)

            voxel = None
            try:
                if _is_nd2_path(first_path):
                    with nd2.ND2File(first_path) as f:
                        voxel = f.voxel_size()
                elif _is_tiff_path(first_path):
                    import tifffile
                    with tifffile.TiffFile(first_path) as tif:
                        # Best-effort: only attempt OME metadata; pixel sizes
                        # are not used elsewhere yet.
                        if tif.is_ome and tif.ome_metadata:
                            voxel = None
            except Exception:
                # Voxel-size discovery is non-critical.
                voxel = None

            instance = cls(data, path, is_image, channel_n)
            instance.voxel_size = voxel
            cls._instance = instance

        pub.sendMessage("image_data_loaded", image_data=cls._instance)
        return cls._instance

    def on_crop_selected(self, coords: list):
        self.crop_coordinates = coords
        pub.sendMessage("image_data_loaded", image_data=self)

    def on_crop_reset(self):
        self.crop_coordinates = None
        pub.sendMessage("image_data_loaded", image_data=self)

    def _cleanup(self):
        # Remove temp memmap file from a previous TIFF directory load, if any.
        memmap_file = getattr(self, "_memmap_file", None)
        if memmap_file and os.path.exists(memmap_file):
            try:
                os.remove(memmap_file)
            except OSError:
                pass

    def get_channel_n(self):
        return self.channel_n

    def get(self, t, p, c):
        """Helper method to retrieve raw images"""
        if len(self.data.shape) == 5:  # has channel
            raw_image = self.data[t, p, c]
        elif len(self.data.shape) == 4:  # no channel
            raw_image = self.data[t, p]
        else:
            print(f"Unusual data format: {len(self.data.shape)} dimensions")
            if len(self.data.shape) >= 3:
                raw_image = self.data[t, p]
            else:
                raw_image = self.data[t]

        if hasattr(raw_image, "compute"):
            raw_image = raw_image.compute()

        if self.crop_coordinates is not None:
            x, y, width, height = self.crop_coordinates
            raw_image = raw_image[y : y + height, x : x + width]

        if self.registration_offsets is not None:
            _x, _y = self.registration_offsets[t]
            raw_image = ShiftedImage_2D_numba(raw_image, _x, _y)

        return raw_image

    def do_registration_p(self, p: int, c: int = 0):
        """
        Runs the image registration at a specific position. Afterward, all images
        will receive the transformation.
        """
        image_series = self.data[:, p, c].compute()
        image_series = ((image_series / 65535) * 255).astype(np.uint8)

        s = time.time()
        res = register_images(image_series)
        e = time.time()
        print("Image registration took {:.2f} seconds".format(e - s))

        self.registration_offsets = res.offsets

    # ── Loaders ──────────────────────────────────────────────────────
    @staticmethod
    def normalize_axes(arr, axes: str):
        """Adjust array/axes string to canonical 5D shape (T, P, C, Y, X)."""
        axes = axes.upper()
        if "T" not in axes:
            arr = np.expand_dims(arr, axis=0)
            axes = "T" + axes
        if "P" not in axes:
            arr = np.expand_dims(arr, axis=1)
            axes = axes.replace("T", "TP")
        if "C" not in axes:
            arr = np.expand_dims(arr, axis=2)
            axes = axes.replace("TP", "TPC")
        return arr, axes

    @classmethod
    def load_nd2(cls, file_paths: Union[str, Sequence[str]], import_mode: Optional[str] = None):
        """
        Load one or more ND2 or TIFF files into a unified 5D dask array.

        Verifies that channel count and Y/X match across files. For ND2 inputs
        (which are already 5D), the P dimension is cropped to the smallest
        found and arrays are concatenated along time. For single TIFF inputs
        or when `import_mode` is "Directory"/"batch_tiff"/"stacked_tiff", the
        first array is used as-is.
        """
        if isinstance(file_paths, str):
            file_paths = [file_paths]

        arrays = []
        p_dims = []
        channels = height = width = None

        for path in file_paths:
            if _is_nd2_path(path):
                arr = nd2.imread(path, dask=True)
                with nd2.ND2File(path) as f:
                    axes = "".join(f.sizes)
            elif _is_tiff_path(path):
                import tifffile
                store = tifffile.imread(path, aszarr=True)
                arr = da.from_zarr(store)
                with tifffile.TiffFile(path) as tif:
                    series = tif.series[0]
                    axes = series.axes
            else:
                raise ValueError(
                    f"Unsupported image format for {path}"
                )

            arr, axes = cls.normalize_axes(arr, axes)
            shape = arr.shape

            if len(shape) != 5:
                raise ValueError(
                    f"File {path} has shape {shape}; expected (T, P, C, Y, X) after normalization"
                )

            T, P, C, Y, X = shape

            if channels is None:
                channels, height, width = C, Y, X
            else:
                if C != channels:
                    raise ValueError(f"{path}: channels {C} != {channels}")
                if Y != height:
                    raise ValueError(f"{path}: height {Y} != {height}")
                if X != width:
                    raise ValueError(f"{path}: width {X} != {width}")

            p_dims.append(P)
            arrays.append(arr)

        if import_mode in ("Directory", "batch_tiff", "stacked_tiff"):
            # The TIFF directory loader has already produced a unified array;
            # we shouldn't actually reach this branch in normal use because
            # `load_tiff_directory` is called instead — but be defensive.
            full_data = arrays[0]
        else:
            if len(arrays) > 1:
                min_p = min(p_dims)
                cropped = [arr[:, :min_p, :, :, :] for arr in arrays]
                full_data = da.concatenate(cropped, axis=0)
                print(f"Cropped P to {min_p}. Final array shape: {full_data.shape}")
            else:
                full_data = arrays[0]

        print(f"Loaded {len(file_paths)} file(s). Final array shape: {full_data.shape}")

        inst = cls.create_instance(data=full_data, path=file_paths, is_image=True)
        inst.channel_n = channels
        return inst

    # Back-compat alias for callers that explicitly load TIFFs.
    @classmethod
    def load_image(cls, file_paths, import_mode: Optional[str] = None):
        return cls.load_nd2(file_paths, import_mode=import_mode)

    @classmethod
    def load_tiff_directory(cls, file_map: dict, mode: str, progress_callback=None):
        """
        Load a TIFF directory into a 5D (T, P, C, Y, X) array.

        Args:
            file_map: Mapping of (P, T, C) tuples to TIFF paths. For stacked
                mode the T component is None and each file holds a stack of
                pages along T.
            mode: "batch_tiff" or "stacked_tiff".
            progress_callback: Optional callable receiving an integer percentage.
        """
        import tifffile

        # First valid path drives shape detection.
        first_path = None
        for v in file_map.values():
            if v is not None:
                first_path = v
                break
        if first_path is None:
            raise ValueError("No valid TIFF paths found in file_map")

        positions = sorted({k[0] for k in file_map})
        channels = sorted({k[2] for k in file_map})

        if mode == "batch_tiff":
            # Use only observed timepoints from filenames (supports sparse series).
            times = sorted({k[1] for k in file_map if k[1] is not None})
        else:
            # Stacked: number of frames is the page count of the first stack.
            first_path = next(v for v in file_map.values() if v is not None)
            with tifffile.TiffFile(first_path) as tif:
                times = list(range(len(tif.pages)))

        T, P, C = len(times), len(positions), len(channels)

        def _to_2d(frame):
            """Reduce a raw TIFF frame to a 2D (Y, X) array."""
            if frame.ndim == 2:
                return frame
            if frame.ndim == 3 and frame.shape[2] in (3, 4):
                return frame[:, :, 0].copy()
            if frame.ndim == 3:
                return frame[0]
            if frame.ndim == 4 and frame.shape[-1] in (3, 4):
                return frame[0, :, :, 0].copy()
            return frame.reshape(frame.shape[-2], frame.shape[-1])

        # Determine frame size and the widest dtype across the dataset.
        sample = _to_2d(tifffile.imread(first_path))
        Y, X = sample.shape
        widest_dtype = sample.dtype
        for path in file_map.values():
            if path is not None and path != first_path:
                probe = _to_2d(tifffile.imread(path))
                if probe.dtype.itemsize > widest_dtype.itemsize:
                    widest_dtype = probe.dtype
                break  # one extra probe is enough

        print(f"TIFF directory: detected frame size {Y}x{X}, dtype={widest_dtype}")

        total = max(1, len(positions) * len(channels) * len(times))
        current = 0
        missing_frames = []
        missing_stacks = []

        filename = f"temp_{uuid.uuid4().hex}.dat"
        data = np.memmap(
            filename,
            dtype=widest_dtype,
            mode="w+",
            shape=(T, P, C, Y, X),
        )

        for pi, p in enumerate(positions):
            for ci, c in enumerate(channels):
                if mode == "batch_tiff":
                    for ti, t in enumerate(times):
                        path = file_map.get((p, t, c))
                        if path is None:
                            missing_frames.append((p, t, c))
                        else:
                            frame = _to_2d(tifffile.imread(path))
                            data[ti, pi, ci] = frame
                        current += 1
                        if progress_callback:
                            progress = int((current / total) * 100)
                            progress_callback(progress)
                else:  # stacked_tiff
                    path = file_map.get((p, None, c))
                    if path is None:
                        missing_stacks.append((p, c))
                        current += T
                        if progress_callback:
                            progress_callback(min(int((current / total) * 100), 100))
                        continue

                    with tifffile.TiffFile(path) as tif:
                        for ti in range(T):
                            frame = _to_2d(tif.pages[ti].asarray())
                            data[ti, pi, ci] = frame
                            current += 1
                            if progress_callback:
                                progress = int((current / total) * 100)
                                progress_callback(progress)

        if missing_frames:
            sample_str = ", ".join(
                [f"(p={p}, t={t}, c={c})" for p, t, c in missing_frames[:8]]
            )
            suffix = " ..." if len(missing_frames) > 8 else ""
            print(
                f"Warning: {len(missing_frames)} missing frame(s) were filled with zeros. "
                f"Examples: {sample_str}{suffix}"
            )

        if missing_stacks:
            sample_str = ", ".join([f"(p={p}, c={c})" for p, c in missing_stacks[:8]])
            suffix = " ..." if len(missing_stacks) > 8 else ""
            print(
                f"Warning: {len(missing_stacks)} missing stack(s) were filled with zeros. "
                f"Examples: {sample_str}{suffix}"
            )

        dask_arr = da.from_array(data, chunks=(1, 1, 1, Y, X))
        inst = cls.create_instance(
            data=dask_arr,
            path=list(file_map.values()),
            is_image=True,
            channel_n=C,
        )
        inst._memmap_file = filename
        # JSON-friendly key form so we can round-trip via save_to_disk/load_from_disk.
        inst._tiff_file_map = {f"{k[0]},{k[1]},{k[2]}": v for k, v in file_map.items()}
        inst._tiff_mode = mode

        return inst

    # ── Persistence ──────────────────────────────────────────────────
    def save_to_disk(self, filename: str):
        """Save image-data state to a project folder.

        We don't write the raw pixels — the source files are kept on disk —
        only metadata describing how to reload them.
        """
        base_dir = Path(filename)
        os.makedirs(base_dir, exist_ok=True)

        if self.segmentation_cache is not None:
            cache_path = base_dir / "segmentation_cache.h5"
            self.segmentation_cache.save(str(cache_path))

        container_data = {
            "image_filename": self.image_filename,
            "is_image": self.is_image,
            # Back-compat keys for older readers.
            "nd2_filename": self.image_filename,
            "is_nd2": self.is_image,
        }

        if getattr(self, "_tiff_file_map", None):
            container_data["tiff_file_map"] = self._tiff_file_map
            container_data["tiff_mode"] = self._tiff_mode

        with open(base_dir / "image_data.json", "w") as f:
            json.dump(container_data, f)

    @staticmethod
    def _relink_paths(paths, search_dirs):
        """Try to resolve missing file paths by looking in `search_dirs`."""
        if isinstance(paths, str):
            paths_list = [paths]
            was_str = True
        else:
            paths_list = list(paths)
            was_str = False

        resolved = []
        for p in paths_list:
            if os.path.exists(p):
                resolved.append(p)
                continue
            basename = os.path.basename(p)
            found = False
            for d in search_dirs:
                candidate = os.path.join(d, basename)
                if os.path.exists(candidate):
                    resolved.append(candidate)
                    found = True
                    break
            if not found:
                return None
        return resolved[0] if was_str else resolved

    @classmethod
    def load_from_disk(cls, filename, relink_root=None):
        """Load image-data state from a project folder."""
        base_dir = Path(filename)

        meta_path = base_dir / "image_data.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata file not found in {filename}")

        with open(meta_path, "r") as f:
            meta_json = json.load(f)

        # Accept both the new key and the legacy one.
        image_filename = meta_json.get("image_filename")
        if image_filename is None:
            image_filename = meta_json.get("nd2_filename")
        is_image = meta_json.get("is_image")
        if is_image is None:
            is_image = meta_json.get("is_nd2", True)

        files_ok = (
            isinstance(image_filename, str) and os.path.exists(image_filename)
        ) or (
            isinstance(image_filename, list)
            and all(os.path.exists(_fname) for _fname in image_filename)
        )

        if not files_ok:
            search_dirs = [str(base_dir)]
            if relink_root:
                search_dirs.append(relink_root)
            relinked = cls._relink_paths(image_filename, search_dirs)
            if relinked is not None:
                image_filename = relinked
                files_ok = True

        if not files_ok:
            raise FileNotFoundError(f"Image file not found: {image_filename}")

        # TIFF directory import?
        tiff_file_map_raw = meta_json.get("tiff_file_map")
        tiff_mode = meta_json.get("tiff_mode")

        if tiff_file_map_raw and tiff_mode:
            file_map = {}
            for key_str, path in tiff_file_map_raw.items():
                parts = key_str.split(",")
                p = int(parts[0])
                t = int(parts[1]) if parts[1] != "None" else None
                c = int(parts[2])
                file_map[(p, t, c)] = path
            image_data = cls.load_tiff_directory(file_map, tiff_mode)
        else:
            image_data = cls.load_nd2(image_filename)

        # Restore segmentation cache if file exists.
        cache_path = base_dir / "segmentation_cache.h5"
        if cache_path.exists():
            image_data.segmentation_cache = SegmentationCache.load(
                str(cache_path), image_data.data
            )
            image_data.segmentation_service = SegmentationService(
                cache=image_data.segmentation_cache,
                models=SegmentationModels(),
                data_getter=image_data.get,
            )

        return image_data
