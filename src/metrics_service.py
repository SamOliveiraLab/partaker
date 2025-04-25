import polars as pl
import numpy as np
from skimage.measure import regionprops
from pubsub import pub
from typing import Optional, Dict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('MetricsService')

class MetricsService:
    """
    Service that tracks cell metrics across time and position.
    Listens to image_ready messages from SegmentationService and calculates metrics.
    
    This class is implemented as a singleton to ensure only one instance exists
    throughout the application.
    """
    # Class variable to store the singleton instance
    _instance = None
    
    def __new__(cls):
        # If no instance exists, create one
        if cls._instance is None:
            logger.info("Creating MetricsService singleton instance")
            cls._instance = super(MetricsService, cls).__new__(cls)
            # Initialize instance attributes
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        # Only initialize once
        if not getattr(self, '_initialized', False):
            logger.info("Initializing MetricsService")
            # Initialize an empty polars DataFrame
            self.df = pl.DataFrame()
            self._data = []  # Temporary storage for rows before creating DataFrame
            self._segmentation_cache = {}
            # Subscribe only to image_ready messages
            pub.subscribe(self.on_image_ready, "image_ready")
            
            # Mark as initialized
            self._initialized = True
    
    def on_image_ready(self, image, time, position, channel, mode):
        """
        Process segmented images to extract cell metrics.
        
        Args:
            image: The image data (numpy array)
            time: Time point of the image
            position: Position of the image
            channel: Channel of the image
            mode: Display mode of the image
        """
        # Handle raw images for fluorescence analysis
        if mode == "normal":
            self._process_fluorescence_image(image, time, position, channel)
            return
            
        # Only process segmentation-related modes
        if mode not in ["segmented", "labeled"]:
            return
        
        # Convert image to numpy array if needed
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        
        # For segmented images, we need to label the connected components
        if mode == "segmented":
            from scipy.ndimage import label
            labeled_image, num_features = label(image > 0)
        else:  # mode == "labeled"
            # For labeled images, use the labels directly
            labeled_image = image
            num_features = np.max(labeled_image) if labeled_image.size > 0 else 0
        
        if num_features == 0:
            logger.info(f"No cells found in image at T={time}, P={position}, C={channel}")
            return
        
        logger.info(f"Analyzing {num_features} cells in image at T={time}, P={position}, C={channel}")
        
        # Store the labeled image for later fluorescence analysis
        cache_key = (time, position, channel)
        self._segmentation_cache[cache_key] = labeled_image
        
        # Calculate metrics for the segmented cells
        self._calculate_metrics(labeled_image, time, position, channel)
        
        # Request raw images for fluorescence analysis
        # We need to request the raw image for the current channel
        pub.sendMessage("raw_image_request",
                       time=time,
                       position=position,
                       channel=channel)
    
    def _process_fluorescence_image(self, image, time, position, channel):
        """
        Process raw fluorescence images using stored segmentation labels.
        
        Args:
            image: The raw fluorescence image
            time: Time point
            position: Position
            channel: Channel
        """
        cache_key = (time, position, channel)
        
        # Check if we have segmentation labels for this image
        if cache_key not in self._segmentation_cache:
            logger.warning(f"No segmentation labels found for T={time}, P={position}, C={channel}")
            return
        
        # Get the labeled image from cache
        labeled_image = self._segmentation_cache[cache_key]
        
        # Find metrics for this time/position/channel in our data
        matching_metrics = [
            (i, m) for i, m in enumerate(self._data) 
            if m["time"] == time and m["position"] == position and m["channel"] == channel
        ]
        
        if not matching_metrics:
            logger.warning(f"No metrics found for T={time}, P={position}, C={channel}")
            return
        
        # Convert image to numpy array if needed
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        
        # Calculate background intensity (areas with no cells)
        background_mask = labeled_image == 0
        background_intensity = np.mean(image[background_mask]) if np.any(background_mask) else 0
        
        # Update fluorescence metrics for each cell
        for idx, metrics in matching_metrics:
            cell_id = metrics["cell_id"]
            cell_mask = labeled_image == cell_id
            
            if np.any(cell_mask):
                cell_pixels = image[cell_mask]
                metrics["mean_intensity"] = np.mean(cell_pixels)
                metrics["max_intensity"] = np.max(cell_pixels)
                metrics["min_intensity"] = np.min(cell_pixels)
                metrics["std_intensity"] = np.std(cell_pixels)
                metrics["integrated_intensity"] = np.sum(cell_pixels)
                metrics["background_intensity"] = background_intensity
                metrics["normalized_intensity"] = metrics["mean_intensity"] / background_intensity if background_intensity > 0 else None
                
                # Log the fluorescence metrics
                self._log_fluorescence_metrics(metrics)
                
                # Update the metrics in our data
                self._data[idx] = metrics
        
        # Update the DataFrame
        self._update_dataframe()
        
        # Clean up the cache to prevent memory leaks
        del self._segmentation_cache[cache_key]
    
    def _calculate_metrics(self, labeled_image, time, position, channel):
        """
        Calculate basic shape metrics for all cells in a labeled image.
        
        Args:
            labeled_image: Image with labeled regions
            time: Time point
            position: Position
            channel: Channel
        """
        # Use regionprops to calculate metrics for each labeled region
        props = regionprops(labeled_image)
        
        # For each cell, compute metrics and add to data
        for prop in props:
            cell_id = prop.label
            
            # Calculate basic shape metrics
            metrics = {
                "position": position,
                "time": time,
                "cell_id": cell_id,
                "channel": channel,
                "area": prop.area,
                "perimeter": prop.perimeter,
                "eccentricity": prop.eccentricity,
                "major_axis_length": prop.major_axis_length,
                "minor_axis_length": prop.minor_axis_length,
                "centroid_y": prop.centroid[0],
                "centroid_x": prop.centroid[1],
                "aspect_ratio": prop.major_axis_length / prop.minor_axis_length if prop.minor_axis_length > 0 else None,
                # Initialize fluorescence metrics to None - will be filled later
                "mean_intensity": None,
                "max_intensity": None,
                "min_intensity": None,
                "std_intensity": None,
                "integrated_intensity": None,
                "background_intensity": None,
                "normalized_intensity": None
            }
            
            # Log the shape metrics
            self._log_shape_metrics(metrics)
            
            # Add to data collection
            self._data.append(metrics)

    def _log_shape_metrics(self, metrics):
        """
        Log shape metrics for a cell.
        
        Args:
            metrics: Dictionary of cell metrics
        """
        logger.info(f"Shape metrics - T:{metrics['time']} P:{metrics['position']} Cell:{metrics['cell_id']} C:{metrics['channel']}")
        logger.info(f"  Area: {metrics['area']} pixels")
        logger.info(f"  Perimeter: {metrics['perimeter']:.2f}")
        logger.info(f"  Eccentricity: {metrics['eccentricity']:.2f}")
        logger.info(f"  Aspect Ratio: {metrics['aspect_ratio']:.2f}" if metrics['aspect_ratio'] else "  Aspect Ratio: None")
    
    def _log_fluorescence_metrics(self, metrics):
        """
        Log fluorescence metrics for a cell.
        
        Args:
            metrics: Dictionary of cell metrics
        """
        logger.info(f"Fluorescence metrics - T:{metrics['time']} P:{metrics['position']} Cell:{metrics['cell_id']} C:{metrics['channel']}")
        logger.info(f"  Mean Intensity: {metrics['mean_intensity']:.2f}")
        logger.info(f"  Max Intensity: {metrics['max_intensity']:.2f}")
        logger.info(f"  Integrated Intensity: {metrics['integrated_intensity']:.2f}")
        logger.info(f"  Background Intensity: {metrics['background_intensity']:.2f}")
        logger.info(f"  Normalized Intensity: {metrics['normalized_intensity']:.2f}" if metrics['normalized_intensity'] else "  Normalized Intensity: None")
    
    def _update_dataframe(self):
        """Update the Polars DataFrame with collected data"""
        if not self._data:
            return
            
        # Create or update DataFrame
        self.df = pl.DataFrame(self._data)
        
        # Log summary of the updated DataFrame
        logger.info(f"Updated DataFrame with {len(self._data)} rows")
        logger.info(f"DataFrame now has {self.df.height} rows and {self.df.width} columns")
    
    def query(self, position: Optional[int] = None, 
              time: Optional[int] = None, 
              cell_id: Optional[int] = None,
              channel: Optional[int] = None) -> pl.DataFrame:
        """
        Query the metrics DataFrame with optional filters.
        
        Args:
            position: Filter by position
            time: Filter by time
            cell_id: Filter by cell ID
            channel: Filter by channel
            
        Returns:
            Filtered Polars DataFrame
        """
        if self.df.is_empty():
            return pl.DataFrame()
            
        df = self.df
        
        # Apply filters
        if position is not None:
            df = df.filter(pl.col("position") == position)
        if time is not None:
            df = df.filter(pl.col("time") == time)
        if cell_id is not None:
            df = df.filter(pl.col("cell_id") == cell_id)
        if channel is not None:
            df = df.filter(pl.col("channel") == channel)
            
        return df
    
    def clear(self):
        """Clear all collected data"""
        self._data = []
        self.df = pl.DataFrame()
        logger.info("Cleared all metrics data")
