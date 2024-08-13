import os

from distutils.dir_util import copy_tree
from argparse import ArgumentParser, RawTextHelpFormatter

import mlflow
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

from src.configurations import PreprocessConfigurations
from src.extract_data import save_to_csv, save_scaler


def main():
    parser = ArgumentParser(
        description="Make dataset",
        formatter_class=RawTextHelpFormatter,
    )

    parser.add_argument(
        "--data",
        type=str,
        default="housing",
        help="california housing",
    )
    parser.add_argument(
        "--downstream",
        type=str,
        default="/opt/housing/preprocess/",
        help="downstream directory",
    )
    parser.add_argument(
        "--cached_data_id",
        type=str,
        default="",
        help="previous run id for cache",
    )

    args = parser.parse_args()

    downstream_directory = args.downstream

    if args.cached_data_id:
        cached_artifact_directory = os.path.join(
            "/tmp/mlruns/0",
            args.cached_data_id,
            "artifacts/downstream_directory",
        )
        copy_tree(
            cached_artifact_directory,
            downstream_directory,
        )
    else:
        train_output_destination = os.path.join(
            downstream_directory,
            "train",
        )

        valid_output_destination = os.path.join(
            downstream_directory,
            "valid",
        )
        test_output_destination = os.path.join(
            downstream_directory,
            "test",
        )

        scaler_output_destination = os.path.join(
            downstream_directory,
            "scaler",
        )

        os.makedirs(downstream_directory, exist_ok=True)
        os.makedirs(train_output_destination, exist_ok=True)
        os.makedirs(valid_output_destination, exist_ok=True)
        os.makedirs(test_output_destination, exist_ok=True)
        os.makedirs(scaler_output_destination, exist_ok=True)

        housing = fetch_california_housing()

        X_train_full, X_test, y_train_full, y_test = train_test_split(
            housing.data, housing.target.reshape(-1, 1), random_state=42
        )
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train_full, y_train_full, random_state=42
        )

        train_data = np.c_[X_train, y_train].astype("float32")
        valid_data = np.c_[X_valid, y_valid].astype("float32")
        test_data = np.c_[X_test, y_test].astype("float32")
        header_cols = (
            PreprocessConfigurations.feature_names
            + PreprocessConfigurations.target_names
        )
        header = ",".join(header_cols)

        train_data = save_to_csv(train_data, train_output_destination, "train", header)
        valid_data = save_to_csv(valid_data, valid_output_destination, "valid", header)
        test_data = save_to_csv(test_data, test_output_destination, "test", header)
        scaler = save_scaler(
            train_data,
            scaler_output_destination,
            PreprocessConfigurations.scaler,
            header,
        )

        mlflow.log_artifacts(downstream_directory, artifact_path="downstream_directory")


if __name__ == "__main__":
    main()
