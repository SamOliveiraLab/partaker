import btrack
from btrack import datasets
import numpy as np

from cachier import cachier
import datetime

@cachier(stale_after=datetime.timedelta(days=3))
def track_cells(segmentation):
    FEATURES = [
        "area", 
        "major_axis_length",
        "minor_axis_length", 
        "orientation", 
        "solidity"
    ]

    objects = btrack.utils.segmentation_to_objects(
        segmentation, 
        properties=tuple(FEATURES), 
        num_workers=4,  # parallelise this
    )

    # initialise a tracker session using a context manager
    with btrack.BayesianTracker() as tracker:

        # configure the tracker using a config file
        tracker.configure('btrack_config.json')
        tracker.max_search_radius = 50
        tracker.tracking_updates = ["MOTION", "VISUAL"]
        tracker.features = FEATURES
            
        # append the objects to be tracked
        tracker.append(objects)

        # set the volume (Z axis volume limits default to [-1e5, 1e5] for 2D data)
        tracker.volume = ((0, 512), (0, 512))

        # track them (in interactive mode)
        tracker.track_interactive(step_size=100)

        # generate hypotheses and run the global optimizer
        # tracker.optimize()

        # store the data in an HDF5 file
        # tracker.export('tracks.h5', obj_type='obj_type_1')

        # get the tracks as a python list
        tracks = tracker.tracks

        # optional: get the data in a format for napari
        # data, properties, graph = tracker.to_napari()

    return tracks