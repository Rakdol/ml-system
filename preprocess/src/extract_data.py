import pickle
import json
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


def save_to_csv(
    data: Union[np.array, pd.DataFrame],
    destination: str,
    name_prefix: str,
    header: Optional[str],
) -> pd.DataFrame:
    save_dest = Path(destination)
    filename_format = f"housing_{name_prefix}.csv"
    csv_path = save_dest / filename_format
    df = pd.DataFrame(data, columns=header.split(","))
    df.to_csv(csv_path, index=False)
    return df


ScalerType = Union[StandardScaler, MinMaxScaler, RobustScaler]


def save_scaler(
    data: np.array,
    scaler_output_destination: str,
    scaler_name: str,
    header: Optional[str],
) -> ScalerType:

    if scaler_name == "standard":
        scaler = StandardScaler()
    elif scaler_name == "minmax":
        scaler = MinMaxScaler()
    elif scaler_name == "robust":
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()

    scaler.fit(data)
    scaler_dir = Path(scaler_output_destination)
    filename = f"{scaler_name}_scaler.pkl"
    scaler_path = str(scaler_dir / filename)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    if header is not None:
        headers = {i: col for i, col in enumerate(header.split(","))}
        header_filename = "data_feature_names.json"
        header_path = scaler_dir / header_filename
        with open(header_path, "w") as f:
            json.dump(headers, f)
    return scaler
