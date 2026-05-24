import logging
import os
from collections import defaultdict
from typing import Optional

import numpy as np
import polars as pl
from pubsub import pub
from skimage.measure import regionprops

from nd2_analyzer.analysis.morphology.morphology import classify_morphology
from nd2_analyzer.data.frame import TLFrame
from nd2_analyzer.data.image_data import ImageData
from nd2_analyzer.utils import timing_decorator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MetricsService")


class MetricsService:
    _instance = None
    DEFAULT_MODEL = "unet"

    def __new__(cls):
        if cls._instance is None:
            logger.info("Creating OptimizedMetricsService singleton instance")
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not getattr(self, "_initialized", False):
            logger.info("Initializing OptimizedMetricsService")

            # One metrics table per segmentation model
            self.tables: dict[str, pl.DataFrame] = {}

            # Kept for future use if needed
            self._pending_metrics = defaultdict(dict)
            self._segmentation_cache = {}

            self._batch_size = 1000
            self._pending_count = 0

            pub.subscribe(self.compute_metrics_at_frame, "frame_segmented")
            self._initialized = True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _normalize_model(self, model: Optional[str]) -> str:
        """Normalize missing/legacy model names to a stable key."""
        return model or self.DEFAULT_MODEL

    def _get_table(self, model: str) -> pl.DataFrame:
        """Return the table for one model, creating an empty one if needed."""
        model = self._normalize_model(model)
        if model not in self.tables:
            self.tables[model] = pl.DataFrame()
        return self.tables[model]

    def _set_table(self, model: str, df: pl.DataFrame) -> None:
        """Replace the table for one model."""
        self.tables[self._normalize_model(model)] = df

    # ------------------------------------------------------------------
    # Frame ingestion
    # ------------------------------------------------------------------

    @timing_decorator("compute_metrics_at_frame")
    def compute_metrics_at_frame(
        self,
        labeled_frame: np.ndarray,
        time: int,
        position: int,
        channel: int,
        model: str,
    ) -> None:
        """
        Compute metrics for a segmented frame and store them in the table
        corresponding to the segmentation model that produced it.
        """
        model = self._normalize_model(model)

        chan_n = ImageData.get_instance().channel_n
        mcherry_frame = None
        yfp_frame = None

        if chan_n == 3:
            mcherry_frame = ImageData.get_instance().get(time, position, 1)
            yfp_frame = ImageData.get_instance().get(time, position, 2)
        elif chan_n == 2:
            mcherry_frame = ImageData.get_instance().get(time, position, 1)

        curr_analysis_frame = TLFrame(
            index=(time, position),
            labeled_phc=labeled_frame,
            mcherry=mcherry_frame,
            yfp=yfp_frame,
        )

        batch_metrics = self.calculate_cell_metrics(curr_analysis_frame)
        self.update_frame_metrics(model=model, batch_data=batch_metrics)

    @timing_decorator("replace_frame_metrics")
    def update_frame_metrics(self, model: str, batch_data: list[dict]) -> None:
        """
        Replace metrics for one frame within one model-specific table.

        Deletes existing rows for the same (time, position) in the selected
        model table, then appends the new batch.
        """
        if not batch_data:
            return

        model = self._normalize_model(model)
        table = self._get_table(model)
        new_df = pl.DataFrame(batch_data)

        if table.is_empty():
            self._set_table(model, new_df)
            return

        first_row = batch_data[0]
        time_point = first_row["time"]
        position = first_row["position"]

        table = table.filter(
            ~((pl.col("time") == time_point) & (pl.col("position") == position))
        )

        table = pl.concat([table, new_df], how="vertical")
        self._set_table(model, table)

    # ------------------------------------------------------------------
    # Metrics calculation
    # ------------------------------------------------------------------

    @classmethod
    def calculate_cell_metrics(cls, frame: TLFrame) -> list[dict]:
        """
        Compute per-cell metrics from one labeled frame.

        Returns:
            A list of row dictionaries ready to insert into a model-specific table.
        """
        labeled = frame.labeled_phc.astype(np.int32, copy=False)
        cells = regionprops(labeled)
        batch_data: list[dict] = []

        mcherry_shape = frame.mcherry.shape if frame.mcherry is not None else None
        print(f"shape phc {labeled.shape} mcherry {mcherry_shape}")

        back_fluo_mcherry = (
            frame.mcherry[labeled == 0].mean() if frame.mcherry is not None else -1
        )
        back_fluo_yfp = frame.yfp[labeled == 0].mean() if frame.yfp is not None else -1

        max_back_fluo = max(back_fluo_mcherry, back_fluo_yfp)
        has_fluorescence = max_back_fluo >= 0.01

        logger.info(
            "[calculate_cell_metrics] time=%s position=%s cells=%s",
            frame.index[0],
            frame.index[1],
            len(cells),
        )

        for cell in cells:
            cell_id = int(cell.label)

            circularity = (
                round((4 * np.pi * cell.area) / (cell.perimeter**2), 4)
                if cell.perimeter > 0
                else 0.0
            )
            aspect_ratio = (
                round(cell.major_axis_length / cell.minor_axis_length, 4)
                if cell.minor_axis_length > 0
                else 1.0
            )
            y1, x1, y2, x2 = cell.bbox

            metrics_dict = {
                "area": cell.area,
                "perimeter": cell.perimeter,
                "aspect_ratio": aspect_ratio,
                "circularity": circularity,
                "solidity": cell.solidity,
                "equivalent_diameter": cell.equivalent_diameter,
                "orientation": cell.orientation,
            }

            morphology_class = classify_morphology(metrics_dict)

            fluorescence_channel = -1
            fluorescence_level = 0.0

            if has_fluorescence:
                if back_fluo_mcherry != -1 and back_fluo_yfp == -1:
                    mcherry_fluo = round(frame.mcherry[labeled == cell_id].mean(), 4)
                    fluorescence_channel = 1
                    fluorescence_level = mcherry_fluo

                elif back_fluo_mcherry != -1 and back_fluo_yfp != -1:
                    mcherry_fluo = round(frame.mcherry[labeled == cell_id].mean(), 4)
                    yfp_fluo = round(frame.yfp[labeled == cell_id].mean(), 4)

                    if (mcherry_fluo / back_fluo_mcherry) > (yfp_fluo / back_fluo_yfp):
                        fluorescence_channel = 1
                        fluorescence_level = mcherry_fluo
                    else:
                        fluorescence_channel = 2
                        fluorescence_level = yfp_fluo

            row_data = {
                "time": frame.index[0],
                "position": frame.index[1],
                "cell_id": cell_id,
                "area": cell.area,
                "perimeter": cell.perimeter,
                "eccentricity": round(cell.eccentricity, 4),
                "major_axis_length": round(cell.major_axis_length, 4),
                "minor_axis_length": round(cell.minor_axis_length, 4),
                "centroid_y": round(cell.centroid[0], 4),
                "centroid_x": round(cell.centroid[1], 4),
                "aspect_ratio": aspect_ratio,
                "circularity": circularity,
                "solidity": round(cell.solidity, 4),
                "equivalent_diameter": round(cell.equivalent_diameter, 4),
                "orientation": round(cell.orientation, 4),
                "morphology_class": morphology_class,
                "y1": y1,
                "x1": x1,
                "y2": y2,
                "x2": x2,
                "fluorescence_channel": fluorescence_channel,
                "fluo_level": fluorescence_level,
            }
            batch_data.append(row_data)

        return batch_data

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @timing_decorator("query_optimized")
    def query_optimized(
        self,
        model: str,
        time: Optional[int] = None,
        position: Optional[int] = None,
        cell_id: Optional[int] = None,
    ) -> pl.DataFrame:
        """
        Query metrics from one model-specific table.

        Callers must now specify the segmentation model whose metrics they want.
        """
        model = self._normalize_model(model)
        table = self._get_table(model)

        logger.info(f"Model {model}, table {table}")

        if table.is_empty():
            return pl.DataFrame()

        conditions = []
        if time is not None:
            conditions.append(pl.col("time") == time)
        if position is not None:
            conditions.append(pl.col("position") == position)
        if cell_id is not None:
            conditions.append(pl.col("cell_id") == cell_id)

        if not conditions:
            return table

        combined = conditions[0]
        for condition in conditions[1:]:
            combined = combined & condition

        return table.filter(combined)

    @timing_decorator("batch_update_fluorescence")
    def _batch_update_fluorescence(
        self,
        model: str,
        updates: list[dict],
        fluo_column: str,
    ) -> None:
        """Update fluorescence values within one model-specific table."""
        if not updates:
            return

        model = self._normalize_model(model)
        table = self._get_table(model)
        if table.is_empty():
            return

        update_df = pl.DataFrame(updates)

        table = (
            table.join(
                update_df,
                on=["time", "position", "cell_id"],
                how="left",
                suffix="_update",
            )
            .with_columns(
                pl.when(pl.col(f"{fluo_column}_update").is_not_null())
                .then(pl.col(f"{fluo_column}_update"))
                .otherwise(pl.col(fluo_column))
                .alias(fluo_column)
            )
            .drop(f"{fluo_column}_update")
        )

        self._set_table(model, table)

    def get_cell_timeseries(
        self,
        model: str,
        position: int,
        cell_id: int,
    ) -> pl.DataFrame:
        """Get time-series data for one cell from one model-specific table."""
        model = self._normalize_model(model)
        table = self._get_table(model)

        return table.filter(
            (pl.col("position") == position) & (pl.col("cell_id") == cell_id)
        ).sort("time")

    def get_position_summary(
        self,
        model: str,
        position: int,
    ) -> pl.DataFrame:
        """Get summary statistics for one position from one model-specific table."""
        model = self._normalize_model(model)
        table = self._get_table(model)

        return (
            table.filter(pl.col("position") == position)
            .group_by("time")
            .agg(
                [
                    pl.len().alias("cell_count"),
                    pl.mean("area").alias("mean_area"),
                    pl.mean("fluo_level").alias("mean_fluo_level"),
                ]
            )
            .sort("time")
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_optimized(self, folder_path: str) -> None:
        """
        Save each model table as its own Parquet file.

        Example:
            metrics_unet.parquet
            metrics_omnipose_bact_phase.parquet
        """
        os.makedirs(folder_path, exist_ok=True)

        for model, table in self.tables.items():
            if table.is_empty():
                continue
            filename = f"metrics_{model}.parquet"
            parquet_path = os.path.join(folder_path, filename)
            table.write_parquet(parquet_path)
            logger.info(
                "Saved %s rows for model %s to %s", table.height, model, parquet_path
            )

    def load_optimized(self, folder_path: str) -> None:
        """Load all model-specific metrics parquet files from a folder."""
        if not os.path.isdir(folder_path):
            return

        loaded_tables: dict[str, pl.DataFrame] = {}

        for filename in os.listdir(folder_path):
            if not filename.startswith("metrics_") or not filename.endswith(".parquet"):
                continue

            model = filename[len("metrics_") : -len(".parquet")]
            parquet_path = os.path.join(folder_path, filename)
            loaded_tables[model] = pl.read_parquet(parquet_path)
            logger.info(
                "Loaded %s rows for model %s from %s",
                loaded_tables[model].height,
                model,
                parquet_path,
            )

        self.tables = loaded_tables
