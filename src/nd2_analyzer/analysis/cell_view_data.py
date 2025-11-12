"""
Simple module to reorganize tracking + morphology data into cell-based structure.
"""

import polars as pl
import numpy as np


def create_cell_based_dataset(tracks, metrics_df):
    """
    Merge cell trajectories with morphology measurements.

    Parameters:
    -----------
    tracks : list of dict
        Each dict has: {"ID", "x", "y", "t", "parent", "children"}
    metrics_df : polars.DataFrame
        Columns: time, position, cell_id, area, major_axis_length,
                 minor_axis_length, morphology_class, etc.

    Returns:
    --------
    dict : {cell_id: {timepoints, x, y, length, width, area, states, lifespan, fate}}
    """
    print(f"\n🔄 Starting cell-based data reorganization...")
    print(f"   Input: {len(tracks)} tracks")
    print(f"   Input: {metrics_df.height} morphology measurements")

    cell_data = {}
    skipped = 0

    for i, track in enumerate(tracks):
        cell_id = track["ID"]

        if (i + 1) % 100 == 0:
            print(f"   Processing cell {i+1}/{len(tracks)}...")

        # Get all morphology data for this cell
        cell_metrics = metrics_df.filter(
            pl.col("cell_id") == cell_id
        ).sort("time")

        # Skip if no morphology data found
        if cell_metrics.is_empty():
            skipped += 1
            continue

        # Determine what happened to this cell
        fate = _determine_fate(track)

        # Store everything
        cell_data[cell_id] = {
            # Trajectory
            "timepoints": track["t"],
            "x": track["x"],
            "y": track["y"],

            # Morphology time series
            "length": cell_metrics["major_axis_length"].to_list(),
            "width": cell_metrics["minor_axis_length"].to_list(),
            "area": cell_metrics["area"].to_list(),
            "aspect_ratio": cell_metrics["aspect_ratio"].to_list(),
            "states": cell_metrics["morphology_class"].to_list(),

            # Life stats
            "lifespan": len(track["t"]),
            "fate": fate,

            # Lineage
            "parent": track.get("parent"),
            "children": track.get("children", [])
        }

    print(f"\n✅ Reorganization complete!")
    print(f"   Successfully merged: {len(cell_data)} cells")
    print(f"   Skipped (no morphology data): {skipped} cells")

    return cell_data


def _determine_fate(track):
    """Figure out what happened to the cell."""

    # If it has children, it divided
    if track.get("children") and len(track["children"]) > 0:
        return "divided"

    # If track is very short, likely left field of view or segmentation error
    if len(track["t"]) < 5:
        return "left_fov"

    # If we have more sophisticated detection later, add here
    # For now, assume still alive at end
    return "alive_at_end"


def filter_cells_by_lifespan(cell_data, min_frames=10):
    """
    Keep only cells that lived long enough to analyze.

    Parameters:
    -----------
    cell_data : dict
        Output from create_cell_based_dataset
    min_frames : int
        Minimum number of timepoints required

    Returns:
    --------
    dict : Filtered cell_data
    """
    print(f"\n🔍 Filtering cells by lifespan (min: {min_frames} frames)...")
    print(f"   Before filtering: {len(cell_data)} cells")

    filtered = {
        cell_id: data
        for cell_id, data in cell_data.items()
        if data["lifespan"] >= min_frames
    }

    print(f"   After filtering: {len(filtered)} cells")
    print(f"   Removed: {len(cell_data) - len(filtered)} cells")

    return filtered


def get_summary_stats(cell_data):
    """
    Get quick statistics about your cells.

    Returns:
    --------
    dict : Summary statistics
    """
    print(f"\n📊 Calculating summary statistics...")

    lifespans = [data["lifespan"] for data in cell_data.values()]
    fates = [data["fate"] for data in cell_data.values()]

    # Count fates
    fate_counts = {}
    for fate in fates:
        fate_counts[fate] = fate_counts.get(fate, 0) + 1

    stats = {
        "total_cells": len(cell_data),
        "avg_lifespan": np.mean(lifespans),
        "median_lifespan": np.median(lifespans),
        "max_lifespan": np.max(lifespans),
        "min_lifespan": np.min(lifespans),
        "fate_distribution": fate_counts
    }

    print(f"   Total cells: {stats['total_cells']}")
    print(f"   Average lifespan: {stats['avg_lifespan']:.1f} frames")
    print(f"   Median lifespan: {stats['median_lifespan']:.1f} frames")
    print(f"   Longest lived: {stats['max_lifespan']} frames")
    print(f"   Fate distribution:")
    for fate, count in fate_counts.items():
        print(f"      {fate}: {count} cells ({100*count/stats['total_cells']:.1f}%)")

    return stats


# Quick example usage:
"""
from nd2_analyzer.analysis.cell_view_data import create_cell_based_dataset, get_summary_stats, filter_cells_by_lifespan

# Merge your data
cell_data = create_cell_based_dataset(tracks, metrics_service.df)

# Get stats
stats = get_summary_stats(cell_data)

# Filter out short-lived cells
cell_data_filtered = filter_cells_by_lifespan(cell_data, min_frames=20)

# Look at one cell
cell_5 = cell_data[5]
print(f"\nCell 5 lived for {cell_5['lifespan']} frames and {cell_5['fate']}")
print(f"States over time: {cell_5['states']}")
"""
