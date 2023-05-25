import os

import pandas as pd

_local_dir = os.path.dirname(os.path.realpath(__file__))


def load(dataset_name: str = "housing") -> pd.DataFrame:
    """Loads the pre-defined dataset for testing purposes

    Args:
        dataset_name (str, optional): Name of the dataset. Defaults to "housing".

    Returns:
        pd.DataFrame: Dataframe containing the dataset
    """
    if dataset_name == "housing":
        path = os.path.join(_local_dir, "housing.xlsx")
        data_frame = pd.read_excel(path)
    else:
        raise ValueError(f"Dataset {dataset_name} not found")
    return data_frame
