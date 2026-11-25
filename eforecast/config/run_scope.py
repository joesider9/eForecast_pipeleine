# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Shared durable workspace/tenant/case identity without constructing an executor."""
import os

from .run_model import pinned


def resolve(workspace, store, case, *, cache=None):
    folder = workspace.case_dir(case)
    root = os.environ.get('EFORECAST_RUN_WORKSPACE_ID') or str(workspace.root.resolve())
    meta = workspace._meta(folder)
    tenant = workspace._tenant_of(folder, meta)
    locator = meta.get('id') or folder.relative_to(workspace.root).as_posix()
    key = pinned({'workspace': root, 'tenant': tenant, 'locator': locator})
    if cache is not None and key in cache:
        return cache[key]
    owner = store.identity('owner', pinned({'workspace': root, 'tenant': tenant}))
    case_id = store.identity('case', pinned({'owner': owner, 'locator': locator}))
    if cache is not None:
        cache[key] = owner, case_id
    return owner, case_id
