"""
Data storage engine for loading and saving parquet files.
"""
import pandas as pd
from pathlib import Path
from typing import Dict, Any


class DataStorageEngine:
    """
    Storage engine for saving and loading train/val/test datasets in parquet format.

    Handles conversion between pandas Series and DataFrame formats for targets.
    """

    def __init__(self, storage_dir: str = "./data/processed/ready_data"):
        """
        Initialize storage engine.

        Args:
            storage_dir: Directory path for storing parquet files
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        print(f"Storage Engine initialized at: {self.storage_dir}")

    def save_dataset(self, data_dict: Dict[str, Any]) -> None:
        """
        Save dataset dictionary to parquet files.

        Converts Series to DataFrame with 'target' column name for consistency.

        Args:
            data_dict: Dictionary with keys like X_train, y_train, metadata, etc.
        """
        print("\n--- Saving X/y/metadata to Disk (Parquet) ---")
        for key, data in data_dict.items():
            file_path = self.storage_dir / f"{key}.parquet"

            # Convert Series to DataFrame for parquet compatibility
            if isinstance(data, pd.Series):
                data = data.to_frame(name='target')

            if isinstance(data, pd.DataFrame):
                print(f"Saving {key} ({data.shape})...")
                data.to_parquet(file_path, engine='pyarrow', compression='snappy')

        print("Save Complete.")

    def load_dataset(self) -> Dict[str, Any]:
        """
        Load all parquet files from storage directory.

        Automatically converts target files (y_*) back to Series.

        Returns:
            Dictionary with all loaded data

        Raises:
            FileNotFoundError: If no parquet files found in storage directory
        """
        print("\n--- Loading Data from Disk ---")
        loaded_dict = {}
        files = list(self.storage_dir.glob("*.parquet"))

        if not files:
            raise FileNotFoundError(f"No parquet files found in {self.storage_dir}")

        for file_path in files:
            key = file_path.stem
            print(f"Loading {key}...")
            df = pd.read_parquet(file_path)

            # Convert target files back to Series
            if key.startswith('y_'):
                loaded_dict[key] = df.iloc[:, 0]
            else:
                loaded_dict[key] = df

        return loaded_dict
