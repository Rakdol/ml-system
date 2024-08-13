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


class PreprocessConfigurations(object):
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
    n_inputs = 8
    scaler = "standard"