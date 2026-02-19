"""
Create ground-truth segmentation masks for benchmark using Napari.

Workflow:
  1. Load ND2 and select 5 frames (spread across T)
  2. Optionally pre-fill labels from a Partaker model (bact_phase_cp3)
  3. Edit in Napari (paint, erase, add cells)
  4. Save labels on close → gt_T{t}_P{p}_C{c}.tif

Usage:
  partaker-create-gt <nd2_path> [--output-dir DIR] [--n-frames 5] [--prefill bact_phase_cp3]
  python scripts/create_ground_truth.py <nd2_path> ...

Requires: pip install napari tifffile  (or uv sync --extra ground-truth)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import nd2

try:
    import napari
    import tifffile
except ImportError:
    napari = tifffile = None


def load_nd2_frames(nd2_path: str, frame_indices: list, p: int = 0, c: int = 0) -> tuple:
    """Load frames from ND2. Returns (frames_array, frame_indices)."""
    nd2_path = str(nd2_path)

    # Try 1: imread (or dask)
    for use_dask in (False, True):
        try:
            arr = nd2.imread(nd2_path, dask=use_dask)
            if arr.ndim < 5:
                raise ValueError(f"Expected ND2 shape (T,P,C,Y,X), got {arr.shape}")
            frames = []
            for (t, _, _) in frame_indices:
                f = arr[t, p, c]
                if hasattr(f, "compute"):
                    f = f.compute()
                frames.append(np.asarray(f).squeeze())
            return np.stack(frames), frame_indices
        except (ValueError, OSError):
            continue

    # Try 2: ND2File.to_dask() and read only needed frames (file must stay open)
    try:
        with nd2.ND2File(nd2_path) as f:
            arr = f.to_dask()
            frames = []
            for (t, _, _) in frame_indices:
                frame = arr[t, p, c]
                if hasattr(frame, "compute"):
                    frame = frame.compute()
                frames.append(np.asarray(frame).squeeze())
            return np.stack(frames), frame_indices
    except (ValueError, OSError, TypeError):
        pass

    raise ValueError(
        "Could not read this ND2 (metadata shape mismatch). Workarounds:\n"
        "  1. Use a different ND2 from the same microscope that loads in Partaker.\n"
        "  2. Export 5 frames to TIFF (e.g. in ImageJ/Fiji), then run with a folder of TIFFs.\n"
        "  3. Open an issue at https://github.com/tlambert03/nd2 with the file details."
    )


def select_frame_indices(T: int, n_frames: int = 5, p: int = 0, c: int = 0) -> list:
    """Select n_frames evenly spread across T."""
    n = min(n_frames, T)
    if n <= 1:
        return [(0, p, c)]
    t_idx = np.linspace(0, T - 1, n, dtype=int)
    return [(int(t), p, c) for t in t_idx]


def run_prefill_segmentation(frames: np.ndarray, model_id: str) -> np.ndarray:
    """Run Partaker model on frames, return label masks (T, Y, X)."""
    from nd2_analyzer.analysis.segmentation.segmentation_models import SegmentationModels

    sm = SegmentationModels()
    out = []
    for i in range(frames.shape[0]):
        masks = sm.segment_images([frames[i]], mode=model_id, preprocess=True)
        out.append(masks[0])
    return np.array(out, dtype=np.int32)


def main() -> int:
    if napari is None or tifffile is None:
        print("Error: napari and tifffile are required. Install with:")
        print("  pip install napari tifffile")
        print("  or: uv sync --extra ground-truth")
        return 1

    parser = argparse.ArgumentParser(
        description="Create ground-truth masks with Napari for Partaker benchmark"
    )
    parser.add_argument("nd2_path", type=str, help="Path to ND2 file")
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: <nd2_dir>/ground_truth)",
    )
    parser.add_argument(
        "-n", "--n-frames",
        type=int,
        default=5,
        help="Number of frames to annotate (default: 5)",
    )
    parser.add_argument(
        "-p", "--position",
        type=int,
        default=0,
        help="Position index (default: 0)",
    )
    parser.add_argument(
        "-c", "--channel",
        type=int,
        default=0,
        help="Channel index (default: 0)",
    )
    parser.add_argument(
        "--prefill",
        type=str,
        default="bact_phase_cp",
        metavar="MODEL",
        help="Pre-fill labels from this model (default: bact_phase_cp; use bact_phase_cp3 if you have it from cellpose.org). Use 'none' to start blank.",
    )
    args = parser.parse_args()

    nd2_path = Path(args.nd2_path)
    if not nd2_path.exists():
        print(f"File not found: {nd2_path}")
        return 1

    output_dir = Path(args.output_dir) if args.output_dir else nd2_path.parent / "ground_truth"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Get T from ND2 (avoid full imread reshape if file has layout quirks)
    try:
        arr = nd2.imread(str(nd2_path))
        T = arr.shape[0]
    except (ValueError, OSError):
        with nd2.ND2File(str(nd2_path)) as f:
            T = f.sizes.get("T", f.shape[0])
    frame_indices = select_frame_indices(T, args.n_frames, args.position, args.channel)
    print(f"Frames to annotate: {frame_indices}")

    frames, _ = load_nd2_frames(str(nd2_path), frame_indices, args.position, args.channel)
    print(f"Loaded {frames.shape[0]} frames, shape {frames[0].shape}")

    labels = None
    if args.prefill.lower() != "none":
        print(f"Pre-filling labels from {args.prefill}...")
        labels = run_prefill_segmentation(frames, args.prefill)
        print("Done. Edit in Napari, then close the window to save.")

    def save_on_close(event=None):
        try:
            for layer in viewer.layers:
                if isinstance(layer, napari.layers.Labels):
                    lbl = layer.data
                    for i, idx in enumerate(frame_indices):
                        if i < lbl.shape[0]:
                            t, p, c = idx
                            path = output_dir / f"gt_T{t}_P{p}_C{c}.tif"
                            tifffile.imwrite(path, lbl[i].astype(np.uint16))
                    print(f"Saved {len(frame_indices)} ground-truth masks to {output_dir}")
                    break
        except Exception as e:
            print(f"Error saving labels: {e}")

    viewer = napari.Viewer(title="Create ground truth — edit labels, then close to save")
    viewer.add_image(frames, name="frames")

    if labels is not None:
        viewer.add_labels(labels, name="cells")
    else:
        viewer.add_labels(np.zeros_like(frames, dtype=np.int32), name="cells")

    viewer.events.closing.connect(save_on_close)

    napari.run()
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
