import polars as pl
import numpy as np
from skimage.measure import regionprops
from pubsub import pub
from typing import Optional, Dict, Tuple
import logging
from nd2_analyzer.analysis.morphology import classify_morphology
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

import time
import functools
import logging
from typing import Dict, Any, Callable

# Performance logger setup
perf_logger = logging.getLogger('performance')
perf_logger.setLevel(logging.INFO)

def timing_decorator(func_name: str = None):
    """Decorator to measure and log function execution time"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                end_time = time.perf_counter()
                execution_time = end_time - start_time
                
                name = func_name or func.__name__
                perf_logger.info(f"{name}: {execution_time:.4f}s")
                
                return result
            except Exception as e:
                end_time = time.perf_counter()
                execution_time = end_time - start_time
                name = func_name or func.__name__
                perf_logger.error(f"{name}: {execution_time:.4f}s (FAILED: {e})")
                raise
        return wrapper
    return decorator

class TimingContext:
    """Context manager for timing code blocks"""
    def __init__(self, operation_name: str):
        self.name = operation_name
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.perf_counter()
        execution_time = end_time - self.start_time
        if exc_type is None:
            perf_logger.info(f"{self.name}: {execution_time:.4f}s")
        else:
            perf_logger.error(f"{self.name}: {execution_time:.4f}s (FAILED: {exc_val})")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('MetricsService')

class MetricsService:
    """
    High-performance metrics service using Polars for efficient data operations.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            logger.info("Creating OptimizedMetricsService singleton instance")
            cls._instance = super(MetricsService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not getattr(self, '_initialized', False):
            logger.info("Initializing OptimizedMetricsService")
            
            # Use Polars DataFrame as primary storage
            self.df = pl.DataFrame()
            
            # Fast lookup cache for pending fluorescence updates
            self._pending_metrics = defaultdict(dict)  # {(t,p): {cell_id: metrics_dict}}
            
            # Segmentation cache for fluorescence processing
            self._segmentation_cache = {}  # {(t,p): labeled_image}
            
            # Batch processing configuration
            self._batch_size = 1000
            self._pending_count = 0
            
            pub.subscribe(self.on_image_ready, "image_ready")
            self._initialized = True

    @timing_decorator("on_image_ready")
    def on_image_ready(self, image: np.ndarray, time, position, channel, mode):
        """Process images with optimized data handling."""
        
        # Handle fluorescence images
        if mode == "normal" and channel != 0:
            self._process_fluorescence_optimized(image, time, position, channel)
            return

        # Process segmentation
        if mode not in ["segmented", "labeled"]:
            return

        if not isinstance(image, np.ndarray):
            image = np.array(image)

        num_features = np.unique(image).shape[0]
        if num_features == 0:
            return

        logger.info(f"Processing {num_features} cells at T={time}, P={position}")

        # Cache segmentation for fluorescence analysis
        self._segmentation_cache[(time, position)] = image
        
        # Calculate metrics efficiently
        with TimingContext("calculate_metrics_optimized"):
            self._calculate_metrics_optimized(image, time, position, channel)
        
        # Request fluorescence data
        for chan in range(1, 3):
            pub.sendMessage("raw_image_request", time=time, position=position, channel=chan)

    @timing_decorator("calculate_metrics_optimized")
    def _calculate_metrics_optimized(self, labeled_image, time, position, channel):
        """Calculate metrics and store in optimized structure."""
        props = regionprops(labeled_image)
        
        # Prepare batch data for efficient DataFrame operations
        batch_data = []
        
        for prop in props:
            cell_id = prop.label
            
            # Calculate derived metrics
            circularity = (4 * np.pi * prop.area) / (prop.perimeter**2) if prop.perimeter > 0 else 0
            aspect_ratio = prop.major_axis_length / prop.minor_axis_length if prop.minor_axis_length > 0 else 1.0
            y1, x1, y2, x2 = prop.bbox
            
            metrics_dict = {
                "area": prop.area,
                "perimeter": prop.perimeter,
                "aspect_ratio": aspect_ratio,
                "circularity": circularity,
                "solidity": prop.solidity,
                "equivalent_diameter": prop.equivalent_diameter,
                "orientation": prop.orientation
            }
            
            morphology_class = classify_morphology(metrics_dict)
            
            # Create row data
            row_data = {
                "time": time,
                "position": position,
                "cell_id": cell_id,
                "channel": channel,
                "area": prop.area,
                "perimeter": prop.perimeter,
                "eccentricity": prop.eccentricity,
                "major_axis_length": prop.major_axis_length,
                "minor_axis_length": prop.minor_axis_length,
                "centroid_y": prop.centroid[0],
                "centroid_x": prop.centroid[1],
                "aspect_ratio": aspect_ratio,
                "circularity": circularity,
                "solidity": prop.solidity,
                "equivalent_diameter": prop.equivalent_diameter,
                "orientation": prop.orientation,
                "morphology_class": morphology_class,
                "y1": y1, "x1": x1, "y2": y2, "x2": x2,
                "fluo_mcherry": None,
                "fluo_yfp": None
            }
            
            batch_data.append(row_data)
            
            # Store in pending cache for fast fluorescence updates
            self._pending_metrics[(time, position)][cell_id] = row_data

        # Batch insert into DataFrame
        if batch_data:
            new_df = pl.DataFrame(batch_data)
            self.df = pl.concat([self.df, new_df], how="vertical") if not self.df.is_empty() else new_df
            self._pending_count += len(batch_data)

    @timing_decorator("process_fluorescence")
    def _process_fluorescence_optimized(self, image, time, position, channel):
        """Optimized fluorescence processing using fast lookups."""
        cache_key = (time, position)
        
        if cache_key not in self._segmentation_cache:
            logger.warning(f"No segmentation for T={time}, P={position}, C={channel}")
            return
            
        if cache_key not in self._pending_metrics:
            logger.warning(f"No pending metrics for T={time}, P={position}, C={channel}")
            return

        labeled_image = self._segmentation_cache[cache_key]
        pending_cells = self._pending_metrics[cache_key]
        
        if not isinstance(image, np.ndarray):
            image = np.array(image)

        # Calculate background
        background_mask = labeled_image == 0
        background_intensity = np.mean(image[background_mask]) if np.any(background_mask) else 0
        
        fluo_column = "fluo_mcherry" if channel == 1 else "fluo_yfp"
        
        # Update fluorescence values efficiently
        updates = []
        for cell_id, metrics in pending_cells.items():
            cell_mask = labeled_image == cell_id
            if np.any(cell_mask):
                cell_fluorescence = np.mean(image[cell_mask])
                fluorescence_value = cell_fluorescence if cell_fluorescence > background_intensity else 0
                
                # Store update for batch processing
                updates.append({
                    "time": time,
                    "position": position, 
                    "cell_id": cell_id,
                    fluo_column: fluorescence_value
                })

        # Batch update DataFrame using efficient Polars operations
        if updates:
            self._batch_update_fluorescence(updates, fluo_column)

    @timing_decorator("batch_update_fluorescence")
    def _batch_update_fluorescence(self, updates, fluo_column):
        """Efficiently update fluorescence values using Polars joins."""
        if not updates:
            return
            
        # Create update DataFrame
        update_df = pl.DataFrame(updates)
        
        # Use efficient join operation instead of row-by-row updates
        self.df = self.df.join(
            update_df,
            on=["time", "position", "cell_id"],
            how="left",
            suffix="_update"
        ).with_columns([
            pl.when(pl.col(f"{fluo_column}_update").is_not_null())
            .then(pl.col(f"{fluo_column}_update"))
            .otherwise(pl.col(fluo_column))
            .alias(fluo_column)
        ]).drop(f"{fluo_column}_update")

    @timing_decorator("query_optimized")
    def query_optimized(self, time: Optional[int] = None, position: Optional[int] = None, 
                       cell_id: Optional[int] = None) -> pl.DataFrame:
        """Fast querying using Polars' optimized operations."""
        if self.df.is_empty():
            return pl.DataFrame()

        # Build filter conditions efficiently
        conditions = []
        if time is not None:
            conditions.append(pl.col("time") == time)
        if position is not None:
            conditions.append(pl.col("position") == position)
        if cell_id is not None:
            conditions.append(pl.col("cell_id") == cell_id)
        
        if conditions:
            # Combine conditions with logical AND
            combined_condition = conditions[0]
            for condition in conditions[1:]:
                combined_condition = combined_condition & condition
            return self.df.filter(combined_condition)
        
        return self.df

    def get_cell_timeseries(self, position: int, cell_id: int) -> pl.DataFrame:
        """Get time series data for a specific cell efficiently."""
        return self.df.filter(
            (pl.col("position") == position) & (pl.col("cell_id") == cell_id)
        ).sort("time")

    def get_position_summary(self, position: int) -> pl.DataFrame:
        """Get summary statistics for a position."""
        return self.df.filter(pl.col("position") == position).group_by("time").agg([
            pl.count("cell_id").alias("cell_count"),
            pl.mean("area").alias("mean_area"),
            pl.mean("fluo_mcherry").alias("mean_mcherry"),
            pl.mean("fluo_yfp").alias("mean_yfp")
        ]).sort("time")

    def save_optimized(self, folder_path: str):
        """Save data in efficient Parquet format."""
        if not self.df.is_empty():
            import os
            parquet_path = os.path.join(folder_path, "metrics_data.parquet")
            self.df.write_parquet(parquet_path)
            logger.info(f"Saved {self.df.height} rows to {parquet_path}")

    def load_optimized(self, folder_path: str):
        """Load data from Parquet format."""
        import os
        parquet_path = os.path.join(folder_path, "metrics_data.parquet")
        if os.path.exists(parquet_path):
            self.df = pl.read_parquet(parquet_path)
            logger.info(f"Loaded {self.df.height} rows from {parquet_path}")
