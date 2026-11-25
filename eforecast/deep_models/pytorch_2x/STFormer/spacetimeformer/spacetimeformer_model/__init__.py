from . import nn
try:
    from .spacetimeformer_model import Spacetimeformer_Forecaster
except Exception:
    Spacetimeformer_Forecaster = None
