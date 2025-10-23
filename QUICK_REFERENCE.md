# Cell View Analysis - Quick Reference Guide

## Status At-A-Glance

```
READINESS ASSESSMENT: 60-70% Complete
┌─────────────────────────────────────────────────────────────────┐
│                                                                   │
│  Component                    Status      Readiness    Priority   │
│  ─────────────────────────────────────────────────────────────   │
│  Cell Tracking (btrack)       ✓ EXISTS    READY        HIGH      │
│  Morphology Metrics           ✓ EXISTS    PARTIAL      HIGH      │
│  COMSOL Integration           ✗ MISSING   REQUIRED     CRITICAL  │
│  Linking Cell-Morph           ✗ MISSING   IN BRANCH    CRITICAL  │
│  Time Series Structure        ✓ EXISTS    READY        MEDIUM    │
│  Data Export/Format           ✓ EXISTS    READY        MEDIUM    │
│  Visualization                ✓ EXISTS    READY        MEDIUM    │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Critical Files for Cell View Analysis

### Already Implemented & Working
```
/src/nd2_analyzer/analysis/tracking/tracking.py      (2075 lines)
  ├─ track_cells()                  - Main tracking function
  ├─ calculate_cell_motility()      - Motility metrics
  ├─ tracks_to_dataframe()          - Export function
  └─ visualizations                 - 5+ plot functions

/src/nd2_analyzer/analysis/morphology/morphology.py
  ├─ extract_individual_cells()     - Cell extraction
  ├─ classify_morphology()          - Health classification
  └─ 10+ computed features          - Area, solidity, etc.

/src/nd2_analyzer/analysis/metrics_service.py
  ├─ MetricsService (Singleton)     - Metrics storage
  ├─ Polars DataFrame               - Primary storage
  └─ query_optimized()              - Query interface
```

### Partially Implemented (In Progress)
```
BRANCH: map-cell-id-to-track-id-for-morphology
  Status: Incomplete, not merged
  Priority: CRITICAL
  Impact: Cannot link morphology to individual tracks without this
```

### Must Build for Cell View Analysis
```
NEW MODULE: ComsiolDataLoader
  - Read FEM results (VTK/HDF5/CSV format)
  - Parse velocity, pressure, shear stress fields
  - Map to microscopy coordinate system

NEW MODULE: FieldInterpolator
  - Spatial interpolation (velocity at cell position)
  - Temporal interpolation (if FEM has multiple snapshots)
  - Handle coordinate transformations

NEW MODULE: CellEnvironmentMapper
  - Link cell positions to field values
  - Query environment at each cell time point
  - Extend track dictionary with env data
```

## Data Flow Diagram

```
ND2 Image File
    ↓
ImageData Singleton (TPCYX format)
    ├─→ Segmentation Service
    │   └─→ U-Net/Cellpose/CellSAM
    │       └─→ Labeled Frames
    │
    ├─→ Metrics Service (Polars DataFrame)
    │   ├─ Morphology metrics per (time, position)
    │   ├─ Area, solidity, circularity, etc.
    │   └─ ISSUE: Not linked to cell ID!
    │
    └─→ BayesianTracker (btrack)
        ├─ Convert to objects
        ├─ Run tracking algorithm
        ├─ Generate track dictionary
        ├─ ID, x, y, t, parent, children
        ├─ Lineage information
        └─ Can export to DataFrame/CSV

[MISSING] COMSOL FEM Results
    ├─ Velocity field (Vx, Vy, Vz)
    ├─ Pressure field
    └─ Shear stress field
        ↓
    [TO BUILD] FieldInterpolator
        ↓
    [TO BUILD] CellEnvironmentMapper
        ↓
    Extend Track Dictionary
        ├─ env_velocity
        ├─ env_pressure
        ├─ env_shear_stress
        └─ env_velocity_components
```

## Core Data Structures

### Track Dictionary (Currently Available)
```python
track = {
    "ID": 42,                           # Cell ID (persistent)
    "x": np.array([100, 102, 101, ...]), # X coordinates
    "y": np.array([200, 198, 199, ...]), # Y coordinates  
    "t": np.array([0, 1, 2, ...]),      # Time points
    "parent": 15,                       # Parent cell ID (or None)
    "children": [43, 44],               # Daughter cell IDs
}
```

### Morphology Data (Currently Available - But Unlinked)
```python
# Stored in MetricsService per (time, position):
metrics = {
    "area": 150.5,
    "major_axis_length": 20.3,
    "minor_axis_length": 10.1,
    "orientation": 0.45,
    "solidity": 0.95,
    "circularity": 0.72,
    "perimeter": 65.2,
    "aspect_ratio": 2.0,
    "equivalent_diameter": 13.8,
}
```

### Track with Environment (To Be Built)
```python
track = {
    "ID": 42,
    "x": [...], "y": [...], "t": [...],
    "parent": 15, "children": [43, 44],
    
    # TO ADD:
    "env_velocity": [...],              # Magnitude at each t
    "env_velocity_x": [...],            # X component
    "env_velocity_y": [...],            # Y component
    "env_pressure": [...],              # Pressure at each t
    "env_shear_stress": [...],          # Shear at each t
}
```

## Quick Start: What Works Now

### 1. Load and Track Cells
```python
from nd2_analyzer.analysis.tracking.tracking import track_cells
import numpy as np

# Get segmented images (from segmentation pipeline)
segmented = np.array(...)  # Shape: (time, height, width)

# Track them
tracks, graph = track_cells(segmented)
print(f"Found {len(tracks)} tracks")
```

### 2. Export Tracking to DataFrame
```python
from nd2_analyzer.analysis.tracking.tracking import tracks_to_dataframe

df = tracks_to_dataframe(tracks)
df.to_csv("cell_tracking.csv", index=False)
# Columns: ID, t, x, y, z, parent, root, state, generation, features...
```

### 3. Analyze Motility
```python
from nd2_analyzer.analysis.tracking.tracking import enhanced_motility_index

metrics = enhanced_motility_index(tracks, chamber_dimensions=(1392, 1040))
print(f"Population avg motility: {metrics['population_avg_motility']:.1f}")
```

### 4. Visualize Lineage
```python
from nd2_analyzer.analysis.tracking.tracking import visualize_lineage_tree

visualize_lineage_tree(tracks, output_path="lineage.png")
```

## What's Missing - Build This Next

### 1. Link Morphology to Tracks
**Current code:** branch `map-cell-id-to-track-id-for-morphology`
**What it needs:**
- Function to map segmentation label → track ID
- Per-cell morphology time series (morphology values at track['t'])
- Extend track dict with morphology fields

**Impact:** WITHOUT THIS - Cannot analyze cell morphology over time!

### 2. Load COMSOL Data
**Input:** FEM simulation output (velocity, pressure, stress)
**Output:** Spatial field data interpolable at any (x, y) point
**Required libraries:** scipy.interpolate for spatial/temporal interpolation

```python
# Pseudo-code of what to build:
class ComsiolDataLoader:
    def load_velocity_field(comsol_file):
        # Parse COMSOL output
        # Return: grid points, velocity vectors (Vx, Vy, Vz)
        pass
    
    def load_pressure_field(comsol_file):
        # Similar to velocity
        pass
```

### 3. Interpolate Fields to Cell Positions
**Input:** Cell position (x, y) at time t
**Output:** Velocity, pressure, shear at that position

```python
# Pseudo-code:
class FieldInterpolator:
    def __init__(self, velocity_grid, pressure_grid):
        # Setup scipy RegularGridInterpolator objects
        pass
    
    def velocity_at_point(self, x, y, t):
        # Interpolate (or extrapolate if on boundary)
        return vx, vy, vz, magnitude
    
    def pressure_at_point(self, x, y, t):
        return pressure_value
```

### 4. Map Cells to Environment
**Input:** Tracks + field interpolators
**Output:** Extended tracks with environment data

```python
# Pseudo-code:
def add_environment_to_tracks(tracks, field_interpolator, pixels_to_um=0.07):
    for track in tracks:
        env_velocity = []
        env_pressure = []
        
        for i, (x, y, t) in enumerate(zip(track['x'], track['y'], track['t'])):
            # Convert pixels to physical coordinates
            x_um = x * pixels_to_um
            y_um = y * pixels_to_um
            
            # Query environment
            vx, vy, vz, vmag = field_interpolator.velocity_at_point(x_um, y_um, t)
            p = field_interpolator.pressure_at_point(x_um, y_um, t)
            
            env_velocity.append(vmag)
            env_pressure.append(p)
        
        track['env_velocity'] = np.array(env_velocity)
        track['env_pressure'] = np.array(env_pressure)
    
    return tracks
```

## Testing & Validation

### Existing Test Notebook
**Path:** `/notebooks/example_tracking_pipeline-features(1).ipynb`
**Coverage:**
- ✓ Segmentation → Tracking pipeline
- ✓ Track statistics and lineage
- ✓ Lineage tree visualization
- ✓ Export to DataFrame & CSV
- ✗ Morphology linking (not tested)
- ✗ Environmental data (N/A - not implemented)

### Test Dataset
```
Input:  37 frames, 1040x1392 pixels
Output: 2822 tracks detected
        323 cell division events
        670 long-term tracks (≥15 frames)
```

## Dependencies Already Installed

```
✓ btrack (0.6.5)              - Cell tracking
✓ polars (>= 1.30.0)          - Data storage
✓ pandas (>= 2.2.3)           - Data export
✓ numpy                        - Computation
✓ scipy (1.11.4)              - Interpolation (ready to use!)
✓ matplotlib / seaborn        - Visualization
✓ scikit-image                - Image processing
✓ opencv-python               - Video export
✓ tensorflow-macos (2.15)     - Segmentation
✓ cellpose / omnipose         - Alternative segmentation
```

NO ADDITIONAL INSTALLATIONS NEEDED - scipy for interpolation is already there!

## Recommended Timeline

| Phase | Task | Duration | Dependencies |
|-------|------|----------|--------------|
| 1 | Complete cell-morphology linking | 1 week | None |
| 2 | Build COMSOL loader | 1 week | COMSOL file format spec |
| 3 | Build field interpolator | 1 week | Velocity/pressure fields |
| 4 | Cell-environment mapper | 3-5 days | Coord system definition |
| 5 | Analysis & visualization | 1-2 weeks | All above |
| **Total** | | **3-4 weeks** | |

---

## File Locations - COPY THIS PATH

**Full Report:**
`/Applications/Oliveira Lab Projects/nd2-analyzer/PARTAKER/partaker/CELL_VIEW_ANALYSIS_REPORT.md`

**Key Source Files:**
```
/Applications/Oliveira Lab Projects/nd2-analyzer/PARTAKER/partaker/src/nd2_analyzer/analysis/tracking/tracking.py
/Applications/Oliveira Lab Projects/nd2-analyzer/PARTAKER/partaker/src/nd2_analyzer/analysis/morphology/morphology.py
/Applications/Oliveira Lab Projects/nd2-analyzer/PARTAKER/partaker/src/nd2_analyzer/analysis/metrics_service.py
/Applications/Oliveira Lab Projects/nd2-analyzer/PARTAKER/partaker/src/nd2_analyzer/data/image_data.py
/Applications/Oliveira Lab Projects/nd2-analyzer/PARTAKER/partaker/src/nd2_analyzer/ui/widgets/tracking_widget.py
```

---

**ASSESSMENT COMPLETE** - See detailed report for full analysis.
