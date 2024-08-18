import os
from logging import getLogger

from src.constants import CONSTANTS, PLATFORM_ENUM

logger = getLogger(__name__)


class PlatformConfigurations:
    platform = os.getenv("PLATFORM", PLATFORM_ENUM.DOCKER.value)
    if not PLATFORM_ENUM.has_value(platform):
        raise ValueError(
            f"PLATFORM must be one of {[v.value for v in PLATFORM_ENUM.__members__.values()]}"
        )


class TrainConfigurations(object):
    feature_names = [
        "MedInc",
        "HouseAge",
        "AveRooms",
        "AveBedrms",
        "Population",
        "AveOccup",
        "Latitude",
        "Longitude",
    ]
    target_names = ["MedHouseVal"]
    scaler = "standard"

    train_prefix = "train"
    train_file_name = "housing_train.csv"
    valid_prefix = "valid"
    valid_file_name = "housing_valid.csv"
    test_prefix = "test"
    test_file_name = "housing_test.csv"
    scaler_prefix = "scaler"
    scaler_name = "standard_scaler.pkl"
