# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Optional publication context shared with inference, without importing training services."""
from contextvars import ContextVar

current_stage = ContextVar('eforecast_stage_workspace', default=None)
