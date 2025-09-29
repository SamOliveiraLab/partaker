import json
import os
from typing import List, Dict

from nd2 import ND2File


# TODO: put experiment parameters, basically medium type

class Experiment:
    """
    Represents a time-lapse microscopy experiment.
    
    Attributes:
        name (str): Name of the experiment
        nd2_files (List[str]): List of paths to ND2 files
        phc_interval (float): Time step between frames in seconds
        rpu_values (Dict[str, float]): Dictionary of RPU values
    """

    def __init__(self, name: str, nd2_files: List[str], interval: float,
                 rpu_values: Dict[str, float] = None):
        """
        Initialize an experiment.
        
        Args:
            name: Name of the experiment
            nd2_files: List of paths to ND2 files
            interval: Time step between frames in seconds
            rpu_values: Dictionary of RPU values (optional)
        """
        self.name = name
        self.nd2_files = []
        self.phc_interval = interval
        self.rpu_values = rpu_values or {}
        self.base_shape = ()

        for _file in nd2_files:
            self.add_nd2_file(_file)

    def add_nd2_file(self, file_path: str) -> None:
        """
        Add a new ND2 file to the experiment.
        
        Args:
            file_path: Path to the ND2 file
            
        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If the file cannot be opened as an ND2 file or if its shape is incompatible
        """
        # Check if file already exists in the list
        if file_path in self.nd2_files:
            return  # File already added, nothing to do

        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist.")

        try:
            with ND2File(file_path) as reader:

                shape = reader.shape

                if len(self.base_shape) == 0:
                    self.base_shape = shape

                elif self.nd2_files:
                    try:
                        # "compatibility" means it can be concatenated on the first axis
                        if len(shape) != len(self.base_shape):
                            raise ValueError(
                                f"File {file_path} has different dimensions ({len(shape)}) than existing files ({len(self.base_shape)}).")

                        if shape[1:] != self.base_shape[1:]:
                            raise ValueError(
                                f"File {file_path} shape {shape} is not compatible with existing files shape {self.base_shape}.")

                    except Exception as e:
                        raise ValueError(f"Error checking compatibility: {str(e)}")

                self.nd2_files.append(file_path)

        except Exception as e:
            raise ValueError(f"Error opening ND2 file {file_path}: {str(e)}")

    def save(self, folder_path: str) -> None:
        """
        Save experiment configuration to a JSON file.
        
        Args:
            folder_path: Path to save the configuration
        """
        config = {
            'name': self.name,
            'nd2_files': self.nd2_files,
            'interval': self.phc_interval,
            'rpu_values': self.rpu_values
        }
        import os
        file_path = os.path.join(folder_path, "metrics_data.parquet")
        with open(file_path, 'w') as f:
            json.dump(config, f, indent=4)

    @classmethod
    def load(cls, folder_path: str) -> 'Experiment':
        """
        Load experiment configuration from a JSON file.
        
        Args:
            folder_path: Path to the configuration file
            
        Returns:
            An Experiment instance
        """
        import os
        file_path = os.path.join(folder_path, "metrics_data.parquet")
        with open(file_path, 'r') as f:
            config = json.load(f)

        return cls(
            name=config['name'],
            nd2_files=config['nd2_files'],
            interval=config['interval'],
            rpu_values=config['rpu_values']
        )
