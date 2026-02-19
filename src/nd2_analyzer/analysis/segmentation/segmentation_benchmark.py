"""
Segmentation model benchmark for Partaker paper and reproducibility.

Runs all Partaker segmentation models on 5–10 representative frames, reports:
- Time per frame (s) — mean ± std across frames
- Peak memory (MB) — tracemalloc / resource.getrusage
- Cell count (labels) per model
- Optional: IoU, precision, recall vs ground truth masks

Use for Layer 2 (Partaker-specific benchmarks) in the paper.
"""

import os
import resource
import sys
import time
import tracemalloc
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from skimage.measure import label as sk_label

from .segmentation_models import SegmentationModels


# Default number of representative frames for paper benchmarks
DEFAULT_N_FRAMES = 5

# Human-readable names for paper/console
MODEL_DISPLAY_NAMES = {
    SegmentationModels.OMNIPOSE_BACT_PHASE: "bact_phase_omni",
    SegmentationModels.OMNIPOSE_BACT_FLUOR: "bact_fluor_omni",
    SegmentationModels.CELLPOSE_BACT_PHASE: "bact_phase_cp3",
    SegmentationModels.CELLPOSE_BACT_FLUOR: "bact_fluor_cp3",
    SegmentationModels.UNET: "unet",
    SegmentationModels.CELLPOSE: "deepbacs_cp3",
}


def _get_memory_mb() -> float:
    """Return current process memory usage in MB (fallback when tracemalloc unused)."""
    try:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return rss / (1024 * 1024)
        return rss / 1024
    except Exception:
        return 0.0


def _compute_iou_precision_recall(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict:
    """
    Compute IoU, precision, recall between predicted and ground-truth labeled masks.
    Treats each as binary (foreground > 0).
    """
    pred_bin = (pred_mask > 0).astype(np.uint8)
    gt_bin = (gt_mask > 0).astype(np.uint8)
    intersection = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()
    pred_pos = pred_bin.sum()
    gt_pos = gt_bin.sum()
    iou = intersection / union if union > 0 else 0.0
    precision = intersection / pred_pos if pred_pos > 0 else 0.0
    recall = intersection / gt_pos if gt_pos > 0 else 0.0
    return {"iou": float(iou), "precision": float(precision), "recall": float(recall)}


def select_representative_frames(
    nd2_data,
    n_frames: int = DEFAULT_N_FRAMES,
    p: int = 0,
    c: int = 0,
) -> List[Tuple[int, int, int]]:
    """
    Select n_frames spread across the T dimension for variety
    (early, mid, late frames; normal rods, dense packing, etc.).

    Parameters
    ----------
    nd2_data : array-like
        Shape (T, P, C, Y, X) or compatible
    n_frames : int
        Number of frames to select (default 5)
    p, c : int
        Position and channel index to use

    Returns
    -------
    list of (t, p, c)
    """
    shape = nd2_data.shape
    T = shape[0]
    n = min(n_frames, T)
    if n <= 1:
        return [(0, p, c)]
    # Spread evenly across T (include first, last, and intermediates)
    t_indices = np.linspace(0, T - 1, n, dtype=int)
    return [(int(t), p, c) for t in t_indices]


def _save_overlay(raw: np.ndarray, seg: np.ndarray, path: str) -> None:
    """Save overlay image (green boundaries on raw) for paper figures."""
    import cv2
    from skimage.segmentation import find_boundaries

    raw_rgb = np.stack([raw] * 3, axis=-1) if raw.ndim == 2 else cv2.cvtColor(raw, cv2.COLOR_GRAY2RGB)
    boundaries = find_boundaries(seg, mode="inner")
    overlay = raw_rgb.copy()
    overlay[boundaries] = [0, 255, 0]
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    cv2.imwrite(path, cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_RGB2BGR))


def _get_models_to_run(skip_unet_if_no_weights: bool = True) -> List[str]:
    """Return list of model IDs to benchmark."""
    models = list(SegmentationModels.available_models)
    if skip_unet_if_no_weights and "UNET_WEIGHTS" not in os.environ:
        models = [m for m in models if m != SegmentationModels.UNET]
    return models


def run_benchmark(
    frame: np.ndarray,
    frame_idx: Tuple[int, int, int] = (0, 0, 0),
    models: Optional[List[str]] = None,
    warmup: bool = True,
    n_runs: int = 1,
    use_tracemalloc: bool = True,
    ground_truth: Optional[np.ndarray] = None,
    save_overlays_dir: Optional[str] = None,
) -> dict:
    """
    Run segmentation benchmark on a single frame.

    Uses time.perf_counter() for timing and tracemalloc for peak memory.
    """
    if models is None:
        models = _get_models_to_run()

    sm = SegmentationModels()
    results = {}

    for model_id in models:
        display_name = MODEL_DISPLAY_NAMES.get(model_id, model_id)

        if use_tracemalloc:
            try:
                tracemalloc.start()
            except Exception:
                pass

        times = []
        n_labels = 0
        success = False
        err_msg = None
        last_mask = None
        iou_metrics = None

        try:
            if warmup:
                sm.segment_images([frame.copy()], mode=model_id, preprocess=True)

            for _ in range(n_runs):
                t0 = time.perf_counter()
                out = sm.segment_images([frame.copy()], mode=model_id, preprocess=True)
                t1 = time.perf_counter()
                times.append(t1 - t0)
                last_mask = out[0]
                n_labels = int(sk_label(last_mask > 0).max())

            if ground_truth is not None and last_mask is not None:
                iou_metrics = _compute_iou_precision_recall(last_mask, ground_truth)

            if save_overlays_dir and last_mask is not None:
                t, p, c = frame_idx
                path = os.path.join(
                    save_overlays_dir,
                    f"T{t}_P{p}_C{c}_{display_name}.png",
                )
                _save_overlay(frame, last_mask, path)

            success = True
        except Exception as e:
            err_msg = str(e)

        if use_tracemalloc:
            try:
                current, peak = tracemalloc.get_traced_memory()
                mem_peak_mb = peak / (1024 * 1024)
                tracemalloc.stop()
            except Exception:
                mem_peak_mb = _get_memory_mb()
        else:
            mem_peak_mb = _get_memory_mb()

        results[model_id] = {
            "display_name": display_name,
            "time_per_frame_s": float(np.mean(times)) if times else 0.0,
            "time_std_s": float(np.std(times)) if len(times) > 1 else None,
            "memory_peak_mb": mem_peak_mb,
            "n_labels": n_labels,
            "success": success,
            "error": err_msg,
            "iou": iou_metrics,
        }

    return {
        "frame_shape": frame.shape,
        "frame_idx": frame_idx,
        "results": results,
    }


def print_benchmark_report(
    data: dict,
    title: str = "Partaker Segmentation Model Benchmark",
) -> None:
    """Print a formatted benchmark report to stdout (paper-ready)."""
    frame_indices = data.get("frame_indices") or ([data.get("frame_idx")] if data.get("frame_idx") else [(0, 0, 0)])
    if not frame_indices or frame_indices[0] is None:
        frame_indices = [(0, 0, 0)]
    n_frames = len(frame_indices)
    t, p, c = frame_indices[0]
    h, w = data["frame_shape"]

    lines = [
        "",
        "=" * 80,
        title,
        "=" * 80,
        f"Frames: {n_frames} (T spread) at P={p}, C={c}  |  Shape: {h}×{w} px",
        f"Frame indices: {frame_indices}",
        "-" * 80,
    ]

    # Build header: Model, Time, Memory, Labels, [IoU/Prec/Recall], Status
    header = f"{'Model':<22} {'Time/frame (s)':>14} {'Peak mem (MB)':>14} {'Labels':>8}"
    has_iou = any(r.get("iou") for r in data["results"].values())
    if has_iou:
        header += f" {'IoU':>8} {'Prec':>8} {'Recall':>8}"
    header += "  Status"
    lines.append(header)
    lines.append("-" * 80)

    for model_id, r in data["results"].items():
        name = r["display_name"]
        t_s = r["time_per_frame_s"]
        t_str = f"{t_s:.3f}"
        if r.get("time_std_s") is not None and r["time_std_s"] > 0:
            t_str += f" ± {r['time_std_s']:.3f}"
        mem = r.get("memory_peak_mb", 0)
        mem_str = f"{mem:.1f}"
        n_lbl = r.get("n_labels", 0)
        row = f"{name:<22} {t_str:>14} {mem_str:>14} {n_lbl:>8}"
        if has_iou and r.get("iou"):
            m = r["iou"]
            row += f" {m['iou']:>8.3f} {m['precision']:>8.3f} {m['recall']:>8.3f}"
        elif has_iou:
            row += " " * (8 + 8 + 8 + 2)
        status = "OK" if r["success"] else (r.get("error", "?")[:28] + ".." if r.get("error") else "FAIL")
        row += f"  {status}"
        lines.append(row)

    lines.extend(["-" * 80, ""])
    print("\n".join(lines))


def benchmark_from_nd2(
    nd2_data,
    frame_indices: Optional[List[Tuple[int, int, int]]] = None,
    models: Optional[List[str]] = None,
    n_frames: int = DEFAULT_N_FRAMES,
    p: int = 0,
    c: int = 0,
    warmup: bool = True,
    use_tracemalloc: bool = True,
    ground_truth_masks: Optional[dict] = None,
    save_overlays_dir: Optional[str] = None,
) -> dict:
    """
    Run benchmark on multiple frames from nd2_data.

    Parameters
    ----------
    nd2_data : array-like
        Shape (T, P, C, Y, X)
    frame_indices : list of (t,p,c), optional
        Explicit frame list. If None, uses select_representative_frames(n_frames, p, c).
    models : list of str, optional
    n_frames : int
        Used when frame_indices is None (default 5)
    p, c : int
        Position and channel when auto-selecting frames
    warmup : bool
    use_tracemalloc : bool
    ground_truth_masks : dict, optional
        {(t,p,c): mask_array} for IoU/precision/recall
    save_overlays_dir : str, optional
        If set, save overlay images per model per frame for figures

    Returns
    -------
    dict
        Aggregated results (mean time, max memory across frames)
    """
    if frame_indices is None:
        frame_indices = select_representative_frames(nd2_data, n_frames=n_frames, p=p, c=c)

    ground_truth_masks = ground_truth_masks or {}

    all_results = []
    for idx in frame_indices:
        t, p_i, c_i = idx
        frame = nd2_data[t, p_i, c_i]
        if hasattr(frame, "compute"):
            frame = frame.compute()
        frame = np.asarray(frame).squeeze()
        if frame.ndim != 2:
            frame = frame[0] if frame.shape[0] == 1 else frame
        gt = ground_truth_masks.get(idx)
        data = run_benchmark(
            frame,
            frame_idx=idx,
            models=models,
            warmup=warmup,
            use_tracemalloc=use_tracemalloc,
            ground_truth=gt,
            save_overlays_dir=save_overlays_dir,
        )
        all_results.append(data)

    first = all_results[0]
    agg = {
        "frame_shape": first["frame_shape"],
        "frame_indices": frame_indices,
        "results": {},
    }
    for model_id in first["results"]:
        times = [r["results"][model_id]["time_per_frame_s"] for r in all_results]
        mems = [r["results"][model_id].get("memory_peak_mb", 0) for r in all_results]
        labels_list = [r["results"][model_id].get("n_labels", 0) for r in all_results]
        ok = all(r["results"][model_id]["success"] for r in all_results)
        iou_list = [r["results"][model_id].get("iou") for r in all_results if r["results"][model_id].get("iou")]
        iou_agg = None
        if iou_list:
            iou_agg = {
                "iou": float(np.mean([m["iou"] for m in iou_list])),
                "precision": float(np.mean([m["precision"] for m in iou_list])),
                "recall": float(np.mean([m["recall"] for m in iou_list])),
            }
        agg["results"][model_id] = {
            **first["results"][model_id],
            "time_per_frame_s": float(np.mean(times)),
            "time_std_s": float(np.std(times)) if len(times) > 1 else None,
            "memory_peak_mb": float(np.max(mems)),
            "n_labels": int(np.mean(labels_list)),
            "success": ok,
            "iou": iou_agg,
        }
    return agg


def load_ground_truth_from_dir(
    dir_path: str,
    frame_indices: List[Tuple[int, int, int]],
) -> dict:
    """
    Load ground-truth masks from a directory of gt_T{t}_P{p}_C{c}.tif files
    (created by scripts/create_ground_truth.py).

    Returns
    -------
    dict
        {(t, p, c): mask_array} for use with benchmark_from_nd2(ground_truth_masks=...)
    """
    try:
        import tifffile
    except ImportError:
        raise ImportError("tifffile required: pip install tifffile")

    result = {}
    for t, p, c in frame_indices:
        path = Path(dir_path) / f"gt_T{t}_P{p}_C{c}.tif"
        if path.exists():
            result[(t, p, c)] = tifffile.imread(str(path))
    return result
