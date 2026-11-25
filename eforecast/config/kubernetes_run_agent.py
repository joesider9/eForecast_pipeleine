# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""One Pod, one accepted managed run. No broker consumption or model policy lives here."""
from __future__ import annotations

import logging
import os
import signal
import sys
import time


def main(argv=None):
    from .kubernetes_runs import attach_pod, _run, KubernetesRunError
    from .run_store import configured_store
    from .run_coordinator import Coordinator
    from .run_model import CancelRequest, identity
    from .service import Workspace

    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1 or args[0] != os.environ.get('EFORECAST_KUBE_RUN_ID'):
        raise KubernetesRunError('the Job must name its assigned run')
    logging.basicConfig(level=logging.INFO)
    workspace = Workspace(os.environ['EFORECAST_WORKSPACE'], admin=True)
    store = configured_store(workspace.root)
    run = attach_pod(store, args[0], os.environ['EFORECAST_KUBE_DISPATCH_ID'], os.environ['EFORECAST_POD_UID'])
    coordinator = Coordinator(workspace, store=store, workers=2, batch=32, max_owned=2)
    stopped = []
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda sig, _: stopped.append(sig))
    try:
        while True:
            if stopped:
                stopped.clear()
                current = store.get(run.owner_id, run.case_id, run.id)
                if not current.cancel and current.state != 'terminal':
                    store.request_cancel(run.owner_id, run.case_id, run.id,
                        CancelRequest(identity(), time.time(), 'kubernetes', 'Pod termination requested'))
            coordinator.tick()
            current = store.get(run.owner_id, run.case_id, run.id)
            if current.state == 'terminal' and current.cleanup in ('complete', 'not_required'):
                return 0 if current.outcome.kind == 'success' else 1
            time.sleep(.2)
    finally:
        coordinator.close()


if __name__ == '__main__':
    raise SystemExit(main())
