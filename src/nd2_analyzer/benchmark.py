"""
CLI entry point for Partaker segmentation model benchmark.

Usage:
    partaker-benchmark <nd2_path> [--frames 0,0,0 1,0,0]
    python -m nd2_analyzer.benchmark <nd2_path>

Runs without loading the GUI (Qt), so it works in headless/SSH environments.
"""

import argparse
import sys

# Import benchmark first (no Qt). Avoid importing nd2_analyzer package root
# which eagerly loads Qt.
from nd2_analyzer.analysis.segmentation.segmentation_benchmark import (
    benchmark_from_nd2,
    load_ground_truth_from_dir,
    print_benchmark_report,
)

import nd2


def main():
    from nd2_analyzer.analysis.segmentation.segmentation_benchmark import DEFAULT_N_FRAMES

    parser = argparse.ArgumentParser(
        description="Run Partaker segmentation model benchmark (speed & memory per frame)"
    )
    parser.add_argument(
        "nd2_path",
        type=str,
        help="Path to ND2 file (or first file if directory)",
    )
    parser.add_argument(
        "-n",
        "--n-frames",
        type=int,
        default=DEFAULT_N_FRAMES,
        help=f"Number of representative frames (default {DEFAULT_N_FRAMES})",
    )
    parser.add_argument(
        "--frames",
        type=str,
        default=None,
        help="Explicit frames as t,p,c (e.g. 0,0,0 5,0,0). Overrides -n.",
    )
    parser.add_argument(
        "-p", "--position",
        type=int,
        default=0,
        help="Position index (default 0)",
    )
    parser.add_argument(
        "-c", "--channel",
        type=int,
        default=0,
        help="Channel index (default 0)",
    )
    parser.add_argument(
        "--save-overlays",
        type=str,
        metavar="DIR",
        help="Save overlay images to DIR for paper figures",
    )
    parser.add_argument(
        "--ground-truth-dir",
        type=str,
        metavar="DIR",
        help="Directory with gt_T*_P*_C*.tif masks (from create_ground_truth.py) for IoU/precision/recall",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip warmup run before timing",
    )
    args = parser.parse_args()

    frame_indices = None
    if args.frames:
        frame_pairs = [s.strip() for s in args.frames.split()]
        frame_indices = []
        for s in frame_pairs:
            parts = [int(x) for x in s.split(",")]
            if len(parts) != 3:
                print(f"Invalid frame format: {s} (expected t,p,c)", file=sys.stderr)
                sys.exit(1)
            frame_indices.append(tuple(parts))

    try:
        arr = nd2.imread(args.nd2_path, dask=True)
    except Exception as e:
        print(f"Failed to load ND2: {e}", file=sys.stderr)
        sys.exit(1)

    if arr.ndim < 5:
        print("ND2 must have shape (T, P, C, Y, X)", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {args.nd2_path}")
    print(f"Shape: {arr.shape} (T, P, C, Y, X)")
    n_frames = len(frame_indices) if frame_indices else args.n_frames
    print(f"Running benchmark on {n_frames} frame(s)...")
    if args.save_overlays:
        print(f"Saving overlays to {args.save_overlays}")
    print()

    ground_truth_masks = None
    if args.ground_truth_dir:
        # Resolve frame_indices before loading GT
        if frame_indices is None:
            from nd2_analyzer.analysis.segmentation.segmentation_benchmark import (
                select_representative_frames,
            )
            frame_indices = select_representative_frames(arr, args.n_frames, args.position, args.channel)
        ground_truth_masks = load_ground_truth_from_dir(args.ground_truth_dir, frame_indices)
        print(f"Loaded {len(ground_truth_masks)} ground-truth masks from {args.ground_truth_dir}")

    data = benchmark_from_nd2(
        arr,
        frame_indices=frame_indices,
        n_frames=args.n_frames,
        p=args.position,
        c=args.channel,
        warmup=not args.no_warmup,
        save_overlays_dir=args.save_overlays,
        ground_truth_masks=ground_truth_masks,
    )
    print_benchmark_report(data)
    return 0


if __name__ == "__main__":
    sys.exit(main())
