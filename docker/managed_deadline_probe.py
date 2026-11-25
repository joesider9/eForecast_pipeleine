# eForecast project author: George Sideratos and Claude
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Qualification-only worker that stays active until its container is terminated.

``smoke_managed_deadlines.py`` copies this module beside the script its Job runs, in the shared
folder, as ``smoke_managed_kubernetes.py`` does with ``managed_compute_probe.py``.
"""
import json
import os
from pathlib import Path
import time


def hold(payload, context):
    Path(payload['marker']).write_text(json.dumps({'run_id': os.environ['EFORECAST_RUN_ID'],
        'pid': os.getpid(), 'pod_uid': os.environ['EFORECAST_POD_UID']}))
    time.sleep(600)
    raise AssertionError('deadline probe must be terminated before returning')


def main():
    """Start one worker that holds, and wait for it: the Pod's deadline is what ends both."""
    from eforecast.config.model_compute import model_workers
    run_id = os.environ['EFORECAST_RUN_ID']
    workers = model_workers()
    request = workers.request(run_id, run_id + '-deadline', run_id + '-attempt', 1)
    marker = Path(os.environ['EFORECAST_WORKSPACE']).parent / (run_id + '-active-compute.json')
    handle = workers.prepare(request, 'managed_deadline_probe:hold', {'marker': str(marker)},
                             execution={'device': 'cpu', 'threads': 1})
    workers.wait(handle, timeout=300)
    raise AssertionError('deadline root must not survive its Pod')
