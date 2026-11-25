# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Serve the configuration web UI (API + the built ``webui/dist``) with uvicorn.

    python -m eforecast.config.serve                       # http://0.0.0.0:8000, authentication on
    python -m eforecast.config.serve --port 8080 --no-auth  # single local user
    python -m eforecast.config.serve --workspace /media/smartrue/HHD1/George/eforecast_workspace

Environment (the flags above set the same variables for the process):
    EFORECAST_WORKSPACE        cases, users.json, run records      (required; a known lab workstation has a default)
    EFORECAST_AUTH             on | off
    EFORECAST_SECRET           cookie-signing secret (else <workspace>/.secret is created once)
    EFORECAST_BROWSE_ROOTS     colon-separated folders the file browser may show
    EFORECAST_PYTHON           interpreter for the pipeline runs (default: this one)
    EFORECAST_OPTUNA_STORAGE   Optuna storage URL of the training runs (default: journal files)
    EFORECAST_OPTUNA_DASHBOARD_URL, EFORECAST_GRAFANA_URL, EFORECAST_METRICS_URL   links of the Studies page
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="0.0.0.0", help="interface to bind (0.0.0.0 = every interface, 127.0.0.1 = this machine only)")
    parser.add_argument("--port", type=int, default=8000, help="TCP port of the web UI")
    parser.add_argument("--workspace", default=None, help="EFORECAST_WORKSPACE")
    parser.add_argument("--no-auth", action="store_true", help="EFORECAST_AUTH=off: one local user, no login")
    parser.add_argument("--reload", action="store_true", help="restart on code changes (development)")
    parser.add_argument("--log-level", default="info")
    parser.add_argument('--coordinator', choices=('auto', 'external', 'embedded'),
                        default=os.environ.get('EFORECAST_RUN_COORDINATOR', 'auto'),
                        help='auto starts an independent coordinator; external uses a separately supervised service')
    args = parser.parse_args(argv)
    if args.workspace:
        os.environ["EFORECAST_WORKSPACE"] = args.workspace
    if args.no_auth:
        os.environ["EFORECAST_AUTH"] = "off"
    os.environ['EFORECAST_RUN_COORDINATOR'] = args.coordinator
    dist = Path(__file__).resolve().parents[2] / "webui" / "dist" / "index.html"
    if not dist.is_file():
        print(f"warning: {dist} is missing; run `npm run build` in webui/ (the API works, the pages do not)", file=sys.stderr)
    from eforecast.config.service import WorkspaceNotConfigured, workspace_root
    try:
        workspace_root()
    except WorkspaceNotConfigured as error:
        # Before anything is created: a colleague starting the service on their own machine
        # gets the one thing to set, not a permission error from somewhere inside a request.
        print(f"error: {error}", file=sys.stderr)
        return 2
    from eforecast.config import auth
    if auth.auth_enabled() and not auth.load_users():
        print(f"warning: no user in {auth.users_file()}; create the first (admin) with "
              f"`{sys.executable} -m eforecast.config.auth add <name>`", file=sys.stderr)
    import uvicorn
    if args.coordinator == 'auto':
        from eforecast.config.run_coordinator import ensure_service
        from eforecast.config.service import Workspace
        ensure_service(Workspace())
    uvicorn.run("eforecast.config.api:create_app", factory=True, host=args.host, port=args.port, reload=args.reload,
                log_level=args.log_level)
    return 0


if __name__ == "__main__":
    sys.exit(main())
