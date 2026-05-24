"""
segmentation_cache.py

Purely a key-value store for uint16 labeled segmentation frames.

Contract
--------
- Keys:   (time, position) integer tuples
- Values: uint16 ndarray, shape (H, W), pixel value = cell label (0 = background)
- Segmentation is computed lazily on first access via SegmentationModels.
- All post-processing is the responsibility of SegmentationService.
"""

from __future__ import annotations

import json
import os
from itertools import product

import h5py
import numpy as np

from .segmentation_models import SegmentationModels


class SegmentationCache:
    """
    Lazy per-model cache of uint16 labeled segmentation frames.

    The cache is keyed exclusively by (time, position). Segmentation is
    always performed on a single designated phase-contrast channel
    (``phc_channel``), whose index is fixed at construction time.

    Usage
    -----
    cache.with_model("unet")[(time, position)]  ->  np.ndarray uint16 (H, W)
    """

    def __init__(self, nd2_data, phc_channel: int = 0) -> None:
        """
        Args:
            nd2_data:    ND2 dataset object exposing .shape, .ndim, and
                         array indexing that returns dask/numpy frames.
            phc_channel: Channel index of the phase-contrast image used
                         for segmentation (default 0).
        """
        self.nd2_data = nd2_data
        self.phc_channel = phc_channel
        # {model_name: (array[T, P, H, W], set_of_computed_(t,p)_tuples)}
        self._store: dict[str, tuple[np.ndarray, set[tuple]]] = {}
        self._active_model: str | None = None

    # ------------------------------------------------------------------
    # Model selection
    # ------------------------------------------------------------------

    def with_model(self, model_name: str) -> "SegmentationCache":
        """Select the active model, allocating backing storage if needed."""
        self._active_model = model_name
        if model_name not in self._store:
            self._store[model_name] = (
                np.zeros(self._spatial_shape, dtype=np.uint16),
                set(),
            )
        return self

    # ------------------------------------------------------------------
    # Item access  (keys are always (time, position))
    # ------------------------------------------------------------------

    def __getitem__(self, key: tuple) -> np.ndarray:
        if self._active_model is None:
            raise RuntimeError("Call with_model() before accessing the cache.")

        array, computed = self._store[self._active_model]
        t, p = self._parse_tp(key)

        if (t, p) not in computed:
            self._compute_and_store(array, computed, t, p)

        return array[t, p]  # shape (H, W)

    def __setitem__(self, key: tuple, value: np.ndarray) -> None:
        """Allow SegmentationService to write a post-processed frame back."""
        if self._active_model is None:
            raise RuntimeError("Call with_model() before writing to the cache.")
        array, computed = self._store[self._active_model]
        t, p = self._parse_tp(key)
        array[t, p] = value
        computed.add((t, p))

    def is_computed(self, model_name: str, key: tuple) -> bool:
        """Return True if the frame has already been segmented and stored."""
        if model_name not in self._store:
            return False
        _, computed = self._store[model_name]
        t, p = self._parse_tp(key)
        return (t, p) in computed

    # ------------------------------------------------------------------
    # Lazy computation
    # ------------------------------------------------------------------

    def _compute_and_store(
        self,
        array: np.ndarray,
        computed: set[tuple],
        t: int,
        p: int,
    ) -> None:
        """Segment one (time, position) frame and write it into the backing array."""
        frame = self._read_phc_frame(t, p)

        labeled = SegmentationModels().segment_images([frame], mode=self._active_model)[
            0
        ]  # uint16, shape (H, W)

        array[t, p] = labeled
        computed.add((t, p))

    def _read_phc_frame(self, t: int, p: int) -> np.ndarray:
        """Read the phase-contrast frame at (t, p) from nd2_data."""
        frame = self.nd2_data[t, p, self.phc_channel]
        if hasattr(frame, "compute"):
            frame = frame.compute()
        return frame

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, file_path: str) -> None:
        """Persist all cached segmentations to an HDF5 file."""
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

        with h5py.File(file_path, "w") as f:
            f.attrs["active_model"] = self._active_model or ""
            f.attrs["phc_channel"] = self.phc_channel

            store_group = f.create_group("store")
            for model_name, (array, computed) in self._store.items():
                g = store_group.create_group(model_name)
                g.create_dataset("array", data=array, compression="gzip")
                g.attrs["computed"] = json.dumps([list(idx) for idx in computed])

            if self.nd2_data is not None:
                meta = f.create_group("nd2_metadata")
                if hasattr(self.nd2_data, "filename"):
                    meta.attrs["filename"] = str(self.nd2_data.filename)
                if hasattr(self.nd2_data, "shape"):
                    meta.attrs["shape"] = json.dumps(list(self.nd2_data.shape))

    @classmethod
    def load(cls, file_path: str, nd2_data=None) -> "SegmentationCache":
        """Restore a cache from an HDF5 file."""
        with h5py.File(file_path, "r") as f:
            phc_channel = int(f.attrs.get("phc_channel", 0))
            cache = cls(nd2_data, phc_channel=phc_channel)
            cache._active_model = f.attrs.get("active_model") or None

            if "store" not in f:
                return cache

            for model_name, g in f["store"].items():
                array = g["array"][:]
                computed: set[tuple] = set()
                try:
                    for entry in json.loads(g.attrs.get("computed", "[]")):
                        computed.add(tuple(int(i) for i in entry))
                except Exception as exc:
                    print(
                        f"[SegmentationCache] Could not parse indices for {model_name!r}: {exc}"
                    )

                cache._store[model_name] = (array, computed)
                print(
                    f"[SegmentationCache] Loaded {len(computed)} frame(s) for {model_name!r}"
                )

        return cache

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_tp(key) -> tuple[int, int]:
        """Extract (time, position) as plain ints from various key forms."""
        if isinstance(key, tuple):
            if len(key) >= 2:
                return int(key[0]), int(key[1])
            return int(key[0]), 0
        return int(key), 0

    @property
    def _spatial_shape(self) -> tuple:
        """Shape of the backing array: (T, P, H, W)."""
        full = self.nd2_data.shape  # e.g. (T, P, C, H, W)
        # Always (T, P, H, W) — drop channel dimension(s), keep spatial tail
        return (*full[:2], *full[-2:])

    @property
    def shape(self) -> tuple:
        return self.nd2_data.shape

    @property
    def ndim(self) -> int:
        return self.nd2_data.ndim - 2
