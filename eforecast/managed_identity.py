# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Which managed run, task, attempt and executor a process belongs to, read from its environment.

This is a *contract*, not a service: the supervising host puts four variables and a report
directory into the environment it launches with, and anything downstream can read them. It holds
no state and imports nothing.

It exists as its own module because the neutral side needs it. `eforecast.config.run_reporting`
owns reporting -- writing events, checking cancellation -- and names A09's allocator and executor
machinery to do so. A container launcher that only wants to label a container with the run it
belongs to should not have to reach through all of that: the online image is built without
`eforecast/training/`, and `tests/test_online.py` reads the import graph rather than trusting that
a function-local import never fires.

`run_reporting.context` re-exports `context` so existing callers are unaffected and there is one
definition of the variable names.
"""
from __future__ import annotations

import os

#: The four identifiers a managed process inherits. All four, or none: a partial identity names
#: no attempt and would label a container with a lie.
CONTEXT_ENV = {key: 'EFORECAST_MANAGED_' + key.upper() for key in
               ('run_id', 'task_id', 'attempt_id', 'executor_id')}
#: Where a managed process writes its events. Its absence means nothing is collecting them, so
#: the process is not managed however many identifiers are set.
REPORT_DIR = 'EFORECAST_RUN_REPORT_DIR'


def context():
    """The managed identity, or ``None`` when this process is not part of a managed run."""
    values = {key: os.environ.get(name) for key, name in CONTEXT_ENV.items()}
    return values if all(values.values()) and os.environ.get(REPORT_DIR) else None
