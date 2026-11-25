# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Typed configuration models and legacy static-data compatibility helpers."""

from .loader import config_from_yaml, load_config, to_static_data
from .schema import ExperimentConfig

__all__ = ["ExperimentConfig", "config_from_yaml", "load_config", "to_static_data"]
