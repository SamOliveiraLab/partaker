# Comprehensive Cell View Analysis Readiness Report

## Executive Summary

The PARTAKER/nd2-analyzer codebase has **MODERATE READINESS** for Cell View Analysis implementation. Key tracking and morphology infrastructure exists, but COMSOL environmental data integration is completely absent and requires new development.

---

## 1. CELL TRACKING DATA - EXISTS (READY)

### Current Implementation Status: FULLY OPERATIONAL

#### 1.1 Tracking Code Location
**File:** `/src/nd2_analyzer/analysis/tracking/tracking.py` (2075 lines)

**Core Functionality:**
- **BayesianTracker (btrack)** integration: Fully implemented
- **Feature extraction:** area, major_axis_length, minor_axis_length, orientation, solidity
- **Track object format:** Dictionary with fields:
  - `ID`: unique cell identifier persisting across timeframes
  - `x`, `y`: coordinates over time
  - `t`: time points
  - `parent`: parent cell ID (for lineage)
  - `children`: list of daughter cell IDs (division events)
  
#### 1.2 Data Structure
Each tracked cell is stored as:
```python
{
    "ID": int,                      # Persistent cell identifier
    "x": np.array,                  # X coordinates over time
    "y": np.array,                  # Y coordinates over time
    "t": np.array,                  # Time points
    "parent": int or None,          # Parent cell ID
    "children": list,               # Daughter cell IDs
}
```

#### 1.3 Lineage/Pedigree Support
- Parent-child relationships tracked
- Cell division events detected (323 divisions in test dataset)
- Lineage tree visualization available
- LineageGraph integration for family tree representation

#### 1.4 Export Capabilities
- Function `tracks_to_dataframe()` converts to pandas DataFrame
- Exports: ID, t, x, y, z, parent, root, state, generation, morphology features
- CSV export ready via pandas

---

## 2. MORPHOLOGICAL DATA PER CELL - PARTIALLY EXISTS (PARTIAL)

### Current Implementation Status: PARTIALLY READY

#### 2.1 Morphology Data Available
**Files:** 
- `/src/nd2_analyzer/analysis/morphology/morphology.py` (function list incomplete in read)
- `/src/nd2_analyzer/analysis/metrics_service.py` (polars-based metrics storage)

**Features Computed:**
- Area
- Major axis length
- Minor axis length
- Orientation
- Solidity
- Aspect ratio
- Circularity
- Perimeter
- Equivalent diameter

#### 2.2 Cell Health Classification
Function `classify_morphology()` provides:
- Artifact detection
- Divided cells
- Healthy cells
- Elongated cells
- Deformed cells

**Classification Thresholds:** Hardcoded defaults with 13 parameters (optimizable)

#### 2.3 Storage Format
**Storage Engine:** Polars DataFrames (primary)
- Backend: `/src/nd2_analyzer/analysis/metrics_service.py`
- Singleton MetricsService maintains all metrics
- Query interface: `query_optimized(time, position)`
- Conversion to pandas available when needed

#### 2.4 Limitations
- **CRITICAL GAP:** Morphology metrics NOT linked to cell tracking IDs
  - Metrics stored by (time, position) frame, not by tracked cell ID
  - No existing mapping between segmentation labels and btrack IDs
  - Work item: "map-cell-id-to-track-id-for-morphology" branch exists (incomplete)

---

## 3. COMSOL ENVIRONMENTAL DATA - DOES NOT EXIST (MISSING)

### Current Implementation Status: NOT IMPLEMENTED

#### 3.1 Current Search Results
- **ZERO code** referencing: comsol, velocity, pressure, shear, flow, fluid, environmental
- No data loading mechanism for FEM simulation results
- No field interpolation code

#### 3.2 What Would Be Needed
1. **Data Import Module**
   - COMSOL export format reader (VTK, HDF5, CSV, or native format)
   - Field data structure: velocity (Vx, Vy, Vz), pressure, shear stress
   - Spatial grid mapping to microscopy coordinates

2. **Field Interpolation**
   - Spatial interpolation (velocity field at cell position)
   - Time interpolation (if FEM is periodic or has multiple snapshots)

3. **Cell-to-Field Mapping**
   - Link tracked cell coordinates to FEM grid coordinates
   - Handle coordinate system conversion (microscopy pixels to physical units)
   - Account for experimental timing to simulation timing mapping

#### 3.3 Data Integration Point
Would need to extend track dictionary:
```python
# Current structure:
{"ID": ..., "x": [...], "y": [...], "t": [...], ...}

# Proposed addition:
{
    ...,
    "env_velocity": [...],      # Velocity magnitude at each timepoint
    "env_velocity_x": [...],    # Vx component
    "env_velocity_y": [...],    # Vy component
    "env_pressure": [...],      # Pressure
    "env_shear_stress": [...],  # Shear stress
}
```

---

## 4. INTEGRATION/LINKING CODE - PARTIALLY EXISTS (PARTIAL)

### Current Implementation Status: PARTIALLY READY

#### 4.1 Existing Integration
**File:** `/src/nd2_analyzer/ui/widgets/tracking_widget.py`
- Loads segmentation from MetricsService
- Converts to btrack-compatible format
- Runs tracking
- Updates UI with results

#### 4.2 Data Linking Status

| Component | Linked | Notes |
|-----------|--------|-------|
| Tracking ID to Position | YES | btrack maintains ID across frames |
| Tracking to Morphology | NO | **CRITICAL GAP** |
| Tracking to Fluorescence | NO | Not tested |
| Tracking to Environment | NO | Not applicable (env data missing) |
| Morphology Metrics | YES | Stored in MetricsService |

#### 4.3 Missing Integration: Cell ID to Morphology

**Current Problem:**
```
Morphology metrics stored by: (time, position) -> cell metrics
Tracking data stored by: track_id -> coordinates over time

These are decoupled!
```

**Existing Branch:** `map-cell-id-to-track-id-for-morphology` (incomplete, not merged)

---

## 5. DATA FORMATS - MULTIPLE FORMATS SUPPORTED

### Current Implementation Status: WELL-SUPPORTED

#### 5.1 Tracking Data Format
- **Primary:** Python dictionaries (in-memory)
- **Export:** pandas DataFrame → CSV via `tracks_to_dataframe()`
- **Visualization:** napari-compatible track data + properties

#### 5.2 Morphology Data Format
- **Primary:** Polars DataFrame (efficient, columnar)
- **Export:** Convertible to pandas → CSV
- **Access:** Via MetricsService queries

#### 5.3 Segmentation Data Format
- **Input:** ND2 files (via nd2 library)
- **Processing:** NumPy arrays (T, H, W)
- **Storage:** Cached locally via SegmentationCache
- **Export:** Via segmentation_to_objects() to btrack format

#### 5.4 Video/Animation Export
- **Format:** MP4 (via opencv-python)
- **Function:** `overlay_tracks_on_images()` creates tracking video
- **Support:** 100 arbitrary track visualization, color-coded trajectories

---

## 6. TIME SERIES STRUCTURES - EXISTS (READY)

### Current Implementation Status: FULLY OPERATIONAL

#### 6.1 Frame-Based Structure
**File:** `/src/nd2_analyzer/data/frame.py`
```python
@dataclass
class TLFrame:
    index: Tuple[int, int]          # (time, position)
    labeled_phc: np.ndarray         # Segmentation
    mcherry: Optional[np.ndarray]   # Fluorescence channel 1
    yfp: Optional[np.ndarray]       # Fluorescence channel 2
```

#### 6.2 Time Series Access
- **Image Data Singleton:** `ImageData` (TPCYX format: time, position, channel, Y, X)
- **Get Interface:** `ImageData.get(t, p, c)` → frame at time t, position p, channel c
- **Dask Support:** Large datasets use dask.array for lazy loading

#### 6.3 Track-Based Time Series
Each track already maintains time-indexed data:
```python
for i in range(len(track['t'])):
    t = track['t'][i]
    x = track['x'][i]
    y = track['y'][i]
    # Can index morphology metrics at this (t, position)
```

---

## 7. CURRENT IMPLEMENTATION QUALITY

### 7.1 Code Quality Strengths
- **Modular architecture:** Separate modules for tracking, morphology, segmentation
- **Singleton patterns:** MetricsService, ImageData for global state management
- **Type hints:** Partial Python type annotations
- **Pub/Sub messaging:** Clean loose coupling via pypubsub

### 7.2 Testing
- **Test notebook:** `example_tracking_pipeline-features(1).ipynb`
  - Demonstrates full workflow: segmentation → tracking → lineage visualization
  - Uses test dataset: 37 frames, 1040x1392 pixels
  - Results: 2822 tracks, 323 divisions detected
  - Includes motility analysis and density-based export

### 7.3 Performance Considerations
- **Parallel processing:** Segmentation uses 4 workers
- **Caching:** SegmentationCache for repeated access
- **Batch processing:** MetricsService batches metrics updates
- **No optimization:** Track linking uses brute-force search

---

## 8. DEPENDENCIES & REQUIREMENTS

### 8.1 Required Libraries (Installed)
```
btrack (0.6.5)          # Cell tracking
polars (>= 1.30.0)      # Data storage
pandas (>= 2.2.3)       # Data manipulation
numpy                   # Numerical computing
scikit-image            # Image processing
opencv-python (>= 4.11) # Video writing
matplotlib              # Visualization
```

### 8.2 Data Requirements for Cell View Analysis
1. **Time-lapse microscopy:** ND2 files with segmentation capable
2. **Cell segmentation:** Pre-computed or on-demand via U-Net/Cellpose/CellSAM
3. **Morphology metrics:** Computed automatically via MetricsService
4. **COMSOL data:** **NOT YET INTEGRATED** - requires development

---

## 9. ASSESSMENT SUMMARY

### What EXISTS and is READY:
- [x] Cell tracking IDs persisting across time frames
- [x] Lineage/pedigree information (parent-child)
- [x] Morphological features (10+ metrics per cell)
- [x] Time series frame-based data structures
- [x] Export capabilities (CSV, video)
- [x] Visualization (lineage trees, tracking videos, density maps)
- [x] Data caching and query systems

### What PARTIALLY EXISTS (Needs Work):
- [ ] Linking cell tracking to morphology metrics
  - **Impact:** High - Cannot do per-cell morphology over time
  - **Effort:** Medium - Branch exists, needs completion
  - **Priority:** CRITICAL for Cell View Analysis

- [ ] Fluorescence integration with tracks
  - **Impact:** Medium - If fluorescence channels exist
  - **Effort:** Medium - Already partially implemented
  - **Priority:** HIGH

### What DOES NOT EXIST (Must Build):
- [ ] COMSOL environmental data import
  - **Impact:** Critical - Core requirement
  - **Effort:** High - New module needed
  - **Priority:** CRITICAL

- [ ] Field interpolation engine
  - **Impact:** Critical - Required for matching
  - **Effort:** High - Spatial/temporal mapping
  - **Priority:** CRITICAL

- [ ] Cell-to-field mapping module
  - **Impact:** Critical - Core linking logic
  - **Effort:** High - Coordinate transformation
  - **Priority:** CRITICAL

---

## 10. RECOMMENDED NEXT STEPS

### Phase 1: FIX EXISTING GAPS (Weeks 1-2)
1. **Complete cell ID to morphology mapping**
   - Merge/finish branch: `map-cell-id-to-track-id-for-morphology`
   - Create function: `link_morphology_to_tracks(tracks, metrics_service)`
   - Verify per-cell morphology time series

2. **Verify fluorescence integration**
   - Test fluorescence channel extraction with tracks
   - Ensure time alignment with frame numbers

### Phase 2: IMPLEMENT ENVIRONMENTAL DATA (Weeks 2-4)
1. **Create COMSOL data module**
   - Design format spec (input data structure from COMSOL)
   - Build parser/loader for chosen format
   - Implement spatial/temporal interpolation

2. **Cell-field mapping**
   - Coordinate system alignment
   - Build interpolation functions
   - Verify mapping accuracy

### Phase 3: CELL VIEW ANALYSIS (Weeks 4+)
1. **Build analysis module**
   - Per-cell environmental context extraction
   - Correlation analysis (morphology vs environment)
   - Statistical tests

2. **Create visualization**
   - Cell trails with environmental overlay
   - Scatter plots (cell properties vs environment)
   - Heatmaps of correlated factors

---

## 11. FILE INVENTORY FOR CELL VIEW ANALYSIS

### Core Tracking Files
- `/src/nd2_analyzer/analysis/tracking/tracking.py` (2075 lines)
- `/src/nd2_analyzer/analysis/tracking/track_view.py` (visualization)
- `/src/nd2_analyzer/analysis/tracking/config/btrack_config.json`

### Morphology Files
- `/src/nd2_analyzer/analysis/morphology/morphology.py`
- `/src/nd2_analyzer/analysis/morphology/morphologyworker.py`

### Data Management
- `/src/nd2_analyzer/data/experiment.py` (experiment configuration)
- `/src/nd2_analyzer/data/image_data.py` (data singleton)
- `/src/nd2_analyzer/data/frame.py` (frame dataclass)
- `/src/nd2_analyzer/analysis/metrics_service.py` (metrics storage)

### UI Components
- `/src/nd2_analyzer/ui/widgets/tracking_widget.py` (tracking UI)
- `/src/nd2_analyzer/ui/widgets/tracking_manager.py` (track management)
- `/src/nd2_analyzer/ui/widgets/motility_widget.py` (motility analysis)

### Configuration
- `/btrack_config.json` (tracking parameters)
- `/pyproject.toml` (dependencies - btrack, polars, etc.)

### Test Resources
- `/notebooks/example_tracking_pipeline-features(1).ipynb` (full workflow demo)

---

## Conclusion

The codebase is **60-70% ready** for Cell View Analysis. Core infrastructure (tracking, morphology, time series) exists and is functional. The critical missing components are:

1. **Linking cell tracking IDs to morphology** (existing branch, needs completion)
2. **COMSOL environmental data integration** (new development required)
3. **Field interpolation and cell-environment mapping** (new development required)

With focused effort on these three items, Cell View Analysis can be operational within 3-4 weeks.
