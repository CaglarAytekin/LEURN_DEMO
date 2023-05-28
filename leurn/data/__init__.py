import os
import pkgutil

import pandas as pd


def load_data(dataset_name: str = "housing") -> pd.DataFrame:
    """Loads the pre-defined dataset for testing purposes

    Args:
        dataset_name (str, optional): Name of the dataset. Defaults to "housing".

    Returns:
        pd.DataFrame: Dataframe containing the dataset
    """
    if dataset_name == "housing":
        data_frame = pd.read_excel(pkgutil.get_data(__name__, "housing.xlsx"))
    elif dataset_name == "iris":
        from sklearn.datasets import load_iris

        return load_iris(as_frame=True).frame
    else:
        raise ValueError(f"Dataset {dataset_name} not found")
    return data_frame
