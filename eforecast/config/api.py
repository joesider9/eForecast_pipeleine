# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""FastAPI application over the configuration service.

    uvicorn eforecast.config.api:app --host 0.0.0.0 --port 8000

Routes under ``/api``; the built web UI (``webui/dist``) is served at ``/`` when it exists.
Authentication and per-user workspaces are in eforecast.config.auth and config.workspace.
"""
from __future__ import annotations

import threading
import asyncio
from contextlib import asynccontextmanager
import logging

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ValidationError

from eforecast.config import auth, service, tenants
from eforecast.config.options import ui_options
from eforecast.config import graph_service
from eforecast.config.layer_catalogue import catalogue
from eforecast.config.family_presets import family_preset
from eforecast.config import runs as runs_module, studies as studies_module
from eforecast.config import result_summary as results_module
from eforecast.config import support_report
from eforecast.config.service import ServiceError, Workspace
from eforecast.config import case_revision

router = APIRouter(prefix="/api")


class Login(BaseModel):
    name: str
    password: str


class GraphRequest(BaseModel):
    document: dict[str, Any]
    tag: str
    method: str | None = None


class Experiments(BaseModel):
    experiments: dict[str, Any]


class RunRequest(BaseModel):
    script: str
    options: dict[str, Any] = {}
    client_key: str | None = None
    expected_revision: str | None = None
    # Omitted stays omitted: a request accepted before this field existed hashes without it, and
    # inserting a default here would turn its retry into a different request (R10-01). The UI
    # always sends its chosen action; the preview alone falls back to 'continue'.
    action: str | None = None
    # Of the throughput profiles this pipeline once had. Still accepted, and ignored, so that a
    # page built before they went can submit a run.
    expected_profile: str | None = None
    # The answer to what the preview showed: this run may replace datasets or trained models
    # that were built from another configuration. Omitted stays omitted, as ``action`` does.
    accept_changes: bool | None = None


class Document(BaseModel):
    document: dict[str, Any]
    path: str | None = None
    expected_revision: str | None = None
    # A13-04/06: the case being checked, so it is not reported as clashing with itself.
    case: str | None = None
    # The ``digest`` of the consequences the caller was shown and accepted for this save
    # (``change_consequences``): what makes a later run build datasets or train families again.
    accepted_consequences: str | None = None


class VarImpRequest(BaseModel):
    document: dict[str, Any]
    rule_sets: Any = None               # the value being edited; None reads the stored one


class NewCase(BaseModel):
    name: str
    document: dict[str, Any] | None = None
    copy_from: str | None = None
    tenant: str | None = None        # which tenant owns it; the caller's own by default


class Sections(BaseModel):
    sections: dict[str, dict[str, Any]]
    replace: list[str] = []
    expected_revision: str | None = None
    accepted_consequences: str | None = None


def _errors(exc: ValidationError) -> list[dict[str, Any]]:
    return [{"path": ".".join(str(item) for item in error["loc"]), "message": error["msg"], "type": error["type"]}
            for error in exc.errors()]


def _validation_response(exc: ValidationError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": "validation failed", "errors": _errors(exc)})


def current_user(request: Request) -> str | None:
    if not auth.auth_enabled():
        return None
    return auth.read_token(request.cookies.get(auth.COOKIE))


def workspace(request: Request | None = None) -> Workspace:
    """The workspace as seen by the caller: single-user when auth is off, else the user's own cases plus shared ones."""
    if request is None or not auth.auth_enabled():
        return Workspace()
    user = current_user(request)
    return Workspace(user=user, admin=auth.is_admin(user))


def system_workspace() -> Workspace:
    """Every case of every user (the run manager and the studies reader work for all users)."""
    return Workspace(admin=True)


def require_admin(request: Request) -> str:
    user = current_user(request)
    if auth.auth_enabled() and not auth.is_admin(user):
        raise HTTPException(status_code=403, detail="administrators only")
    return user or "local"


def readable(request: Request, path: str) -> str:
    """A caller-supplied path this caller may read, or 400.

    The template endpoints used to accept any path inside the browse roots, and the workspace root
    was one of those: a logged-in user could read another user's case configuration, and
    ``module-templates`` would *execute* a file of their choosing as the server user. A path is now
    checked against what this caller owns or may see.
    """
    try:
        return str(workspace(request).may_read(path))
    except ServiceError as error:
        # One answer for "outside the allowed folders" and "another user's case": a refusal that
        # does not say which of the two it was.
        raise HTTPException(status_code=400, detail=str(error)) from error


@dataclass(frozen=True)
class CaseScope:
    """A named case this caller is allowed to touch, and in what capacity.

    Obtaining one is the *only* way a route reaches a case: a route that forgets it has no case to
    work with, and ``test_tenant_isolation`` enumerates the routes to make sure none does.
    """
    ws: Workspace
    name: str
    tenant: str | None
    role: str | None


def _scope(request: Request, name: str) -> CaseScope:
    ws = workspace(request)
    if not ws.exists(name):
        # the same answer as for a name nobody ever used: 404, never 403, and never a hint of whose
        raise HTTPException(status_code=404, detail=f"case {name!r} does not exist")
    folder = ws.case_dir(name)
    return CaseScope(ws=ws, name=name, tenant=ws.tenant_of(name), role=ws.role_in(ws.tenant_of(name)))


def readable_case(request: Request, name: str) -> CaseScope:
    """A case of a tenant this caller belongs to, or one shared with it. 404 otherwise."""
    return _scope(request, name)


def writable_case(request: Request, name: str) -> CaseScope:
    """... which this caller may also use: edit it, run it. 403 for a case only shared with it."""
    scope = _scope(request, name)
    require_writable(scope.ws, name)
    return scope


def writable_or_new_case(request: Request, name: str) -> CaseScope:
    """A case to write: one this caller may use, or a name still free in its own tenant.

    A name another tenant already uses answers the 404 of a name that does not exist, so a save
    cannot be used to discover what other tenants call their cases.
    """
    ws = workspace(request)
    if ws.exists(name):
        return writable_case(request, name)
    if ws.taken(name):
        raise HTTPException(status_code=404, detail=f"case {name!r} does not exist")
    return CaseScope(ws=ws, name=name, tenant=ws.tenant, role=ws.role_in(ws.tenant))


def managed_case(request: Request, name: str) -> CaseScope:
    """... and dispose of: delete it, share it. A tenant ``member`` may not; 403."""
    scope = _scope(request, name)
    try:
        scope.ws.check_manageable(name)
    except ServiceError as error:
        raise HTTPException(status_code=404 if "does not exist" in str(error) else 403,
                            detail=str(error)) from error
    return scope


@dataclass(frozen=True)
class TenantScope:
    """A tenant this caller is a member of, and its role in it."""
    tenant: str
    role: str
    record: dict[str, Any]
    user: str | None


def tenant_scope(request: Request, tenant: str) -> TenantScope:
    """A tenant this caller belongs to. A tenant it does not belong to does not exist for it."""
    user = current_user(request)
    role = tenants.role_in(tenant, user) if auth.auth_enabled() else "owner"
    found = tenants.record(tenant, user) if auth.auth_enabled() else tenants.personal(tenant)
    if role is None or found is None:
        raise HTTPException(status_code=404, detail=f"tenant {tenant!r} does not exist")
    return TenantScope(tenant=tenant, role=role, record=found, user=user)


def owned_tenant(request: Request, tenant: str) -> TenantScope:
    """... whose members and lifetime are the owner's to change. 403 for an admin or a member."""
    scope = tenant_scope(request, tenant)
    if scope.role not in tenants.MANAGE_TENANT:
        raise HTTPException(status_code=403, detail=f"tenant {tenant!r}: only its owner may do that "
                                                    f"(you are {scope.role})")
    return scope


def require_writable(ws: Workspace, name: str) -> None:
    try:
        ws.check_writable(name)
    except ServiceError as error:
        raise HTTPException(status_code=404 if "does not exist" in str(error) else 403, detail=str(error)) from error


# ---------------------------------------------------------------------------- session

@router.post("/login")
def post_login(body: Login, response: Response) -> dict[str, Any]:
    if not auth.check_password(body.name, body.password):
        raise HTTPException(status_code=401, detail="wrong user name or password")
    response.set_cookie(auth.COOKIE, auth.make_token(body.name), httponly=True, samesite="lax",
                        max_age=auth.SESSION_SECONDS)
    return {"user": body.name}


@router.post("/logout")
def post_logout(response: Response) -> dict[str, Any]:
    response.delete_cookie(auth.COOKIE)
    return {"user": None}


@router.get("/me")
def get_me(request: Request) -> dict[str, Any]:
    if not auth.auth_enabled():
        return {"user": "local", "auth": False, "role": "admin", "tenants": {}, "tenant": None}
    user = auth.read_token(request.cookies.get(auth.COOKIE))
    mine = tenants.of(user) if user else {}
    return {"user": user, "auth": True, "role": auth.role_of(user), "tenants": mine, "tenant": user}


class NewUser(BaseModel):
    name: str
    password: str
    role: str = "user"


class Password(BaseModel):
    password: str
    current: str | None = None


@router.get("/users")
def list_users(request: Request) -> list[dict[str, str]]:
    require_admin(request)
    return auth.list_users()


@router.post("/users", status_code=201)
def create_user(request: Request, body: NewUser) -> dict[str, str]:
    require_admin(request)
    try:
        return auth.add_user(body.name, body.password, body.role)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.put("/users/{name}/role")
def put_role(request: Request, name: str, body: dict[str, str]) -> dict[str, str]:
    me = require_admin(request)
    if name == me and body.get("role") != "admin":
        raise HTTPException(status_code=400, detail="you cannot demote yourself")
    try:
        auth.set_role(name, body.get("role", ""))
    except KeyError as error:
        raise HTTPException(status_code=404, detail=f"user {name} not found") from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {"name": name, "role": body["role"]}


@router.put("/users/{name}/password")
def put_password(request: Request, name: str, body: Password) -> dict[str, str]:
    me = current_user(request) or "local"
    if name != me:
        require_admin(request)
    elif auth.auth_enabled() and not auth.check_password(me, body.current or ""):
        raise HTTPException(status_code=403, detail="current password is wrong")
    try:
        return auth.add_user(name, body.password)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.delete("/users/{name}", status_code=204)
def delete_user(request: Request, name: str) -> Response:
    me = require_admin(request)
    if name == me:
        raise HTTPException(status_code=400, detail="you cannot remove yourself")
    auth.remove_user(name)              # the memberships go with the account
    return Response(status_code=204)


# ---------------------------------------------------------------------------- schema, defaults, presets

@router.get("/schema")
def get_schema() -> dict[str, Any]:
    return service.schema()


@router.get("/options")
def get_options() -> dict[str, Any]:
    return ui_options()


@router.get("/doctor")
def get_doctor(request: Request, case: str | None = None, refresh: bool = False) -> dict[str, Any]:
    """Whether this installation is working, for the page rather than for a terminal.

    The same checks `bin/eforecast doctor` runs, read from the cached reading so the page
    answers immediately (measured: 1.6 s to take one, 0.2 ms to read it). It is the only
    surface this had outside a shell, which is why nearly every refusal used to need a
    maintainer: the findings already say whose each one is, so a colleague can see the ones
    that are theirs and how many are not.

    `refresh=true` forces a new reading; a caller who is not a maintainer cannot, because the
    reading costs seconds of this machine's time.
    """
    from eforecast.config import health
    if case is not None:
        readable_case(request, case)                    # a case is only named by someone entitled to it
    admin = not auth.auth_enabled() or auth.is_admin(current_user(request))
    record = _guard(lambda: health.current(case=case, force=bool(refresh) and admin))
    return health.for_user(record, admin=admin)


@router.get("/defaults")
def get_defaults(docker: bool = False, backend: str = "TORCH") -> dict[str, Any]:
    return service.defaults(docker=docker, backend=backend)


@router.get("/presets/variables")
def get_variable_presets(problem_type: str, horizon: int = 0, horizon_type: str = "day_ahead",
                         ts_resolution: str = "H", timezone: str = "UTC") -> list[dict[str, Any]]:
    try:
        return service.variable_presets(problem_type, horizon, horizon_type, ts_resolution, timezone)
    except (ValueError, KeyError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


# ---------------------------------------------------------------------------- documents

@router.post("/validate")
def post_validate(body: Document) -> dict[str, Any]:
    return service.validate(body.document)


@router.post("/preview")
def post_preview(body: Document):
    try:
        static_data = service.preview_static_data(body.document, body.path)
    except ValidationError as exc:
        return _validation_response(exc)
    except Exception as error:  # noqa: BLE001 - the loader may fail on the content (paths, modules)
        raise HTTPException(status_code=400, detail=f"cannot build static_data: {error}") from error
    return {"static_data": _jsonable(static_data), "derived": service.derived_facts(body.document)}


@router.post("/derived")
def post_derived(body: Document) -> dict[str, Any]:
    return service.derived_facts(body.document)


@router.post("/apply-data-types")
def post_apply_data_types(body: Document) -> dict[str, Any]:
    return service.apply_data_types(body.document)


@router.post("/sync")
def post_sync(body: Document) -> dict[str, Any]:
    return service.sync_document(body.document)


@router.post("/readiness")
def post_readiness(body: Document, request: Request) -> dict[str, Any]:
    """Is this case ready to start a long run, and if not, what exactly is wrong?

    A bounded read: the index column of the target file under a row cap, one directory listing
    per weather archive, a spec lookup for each selected method, and the case's own existing
    reports. Nothing is trained, fitted, downloaded or launched, and no source or artifact is
    written. The caller gets ``inputs_digest`` so it can tell whether the verdict still
    describes the document on screen.
    """
    from eforecast.config import readiness
    try:
        # The sibling model paths come from *this caller's* workspace, so a finding can never
        # name a case the reader is not entitled to see.
        try:
            siblings = readiness.case_model_paths(workspace(request))
        except Exception:  # noqa: BLE001 - a readiness check must not fail over a sibling scan
            siblings = None
        return readiness.report(body.document, path=body.path, case=body.case, siblings=siblings)
    except Exception as error:  # noqa: BLE001 - a readiness check must never be the thing that breaks
        raise HTTPException(status_code=400, detail=f"cannot check readiness: {error}") from error


# ---------------------------------------------------------------------------- tenants

class NewTenant(BaseModel):
    id: str
    title: str | None = None


class MemberRole(BaseModel):
    role: str = "member"


@router.get("/tenants")
def list_tenants(request: Request) -> list[dict[str, Any]]:
    """The tenants this caller belongs to. A tenant it does not belong to is not listed."""
    user = current_user(request)
    if not auth.auth_enabled():
        return []
    return [{**(tenants.record(tenant, user) or {}), "role": role} for tenant, role in tenants.of(user).items()]


@router.post("/tenants", status_code=201)
def create_tenant(request: Request, body: NewTenant) -> dict[str, Any]:
    """A tenant whose cases its members share. Its creator is its owner; no platform role needed."""
    user = current_user(request)
    if not auth.auth_enabled():
        raise HTTPException(status_code=400, detail="tenants need authentication (EFORECAST_AUTH=on)")
    if body.id == user or auth.load_users().get(body.id) is not None:
        raise HTTPException(status_code=400, detail=f"{body.id!r} is an account name: "
                                                    "a personal tenant already has it")
    try:
        return {**tenants.create(body.id, user, body.title), "role": "owner"}
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.get("/tenants/{tenant}")
def get_tenant(scope: TenantScope = Depends(tenant_scope)) -> dict[str, Any]:
    return {**scope.record, "role": scope.role}


@router.delete("/tenants/{tenant}", status_code=204)
def delete_tenant(tenant: str, scope: TenantScope = Depends(owned_tenant)) -> Response:
    """Forget a shared tenant. Its cases stay on disk; this route never deletes data."""
    if scope.record.get("personal"):
        raise HTTPException(status_code=400, detail="a personal tenant goes with its account")
    tenants.remove(tenant)
    return Response(status_code=204)


@router.put("/tenants/{tenant}/members/{name}")
def put_member(tenant: str, name: str, body: MemberRole,
               scope: TenantScope = Depends(owned_tenant)) -> dict[str, str]:
    """Invite an account, or change its role. It takes effect on that account's next request."""
    if scope.record.get("personal"):
        raise HTTPException(status_code=400, detail="a personal tenant has one member")
    if auth.load_users().get(name) is None:
        raise HTTPException(status_code=404, detail=f"user {name} not found")
    try:
        return tenants.set_member(tenant, name, body.role)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.delete("/tenants/{tenant}/members/{name}", status_code=204)
def delete_member(tenant: str, name: str, scope: TenantScope = Depends(owned_tenant)) -> Response:
    if scope.record.get("personal"):
        raise HTTPException(status_code=400, detail="a personal tenant has one member")
    try:
        tenants.remove_member(tenant, name)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return Response(status_code=204)


# ---------------------------------------------------------------------------- cases of the workspace

@router.get("/cases")
def list_cases(request: Request) -> list[dict[str, Any]]:
    return workspace(request).list()


@router.post("/cases", status_code=201)
def create_case(request: Request, body: NewCase):
    try:
        ws = workspace(request)
        with case_revision.locked(ws, body.name):
            result = ws.create(body.name, body.document, body.copy_from, tenant=body.tenant)
    except ServiceError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except ValidationError as exc:
        return _validation_response(exc)
    return result


@router.get("/cases/{name}")
def get_case(name: str, scope: CaseScope = Depends(readable_case)) -> dict[str, Any]:
    ws = scope.ws
    try:
        with case_revision.locked(ws, name, shared=True):  # readers share; only a writer excludes them
            document = ws.load(name)
    except ServiceError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return {"name": name, "document": document, "validation": service.validate(document),
            "revision": case_revision.revision(document)}


def check_case_revision(ws: Workspace, name: str, expected: str | None):
    if expected is None:  # existing API/CLI clients keep their established save behavior
        return
    current = ws.load(name) if ws.exists(name) else None
    current_revision = case_revision.revision(current) if current is not None else None
    if expected != current_revision:
        return JSONResponse(status_code=409, content={
            "detail": "The saved case changed. Review the current version before saving.",
            "document": current, "revision": current_revision,
            "validation": service.validate(current) if current is not None else None})


def saved_case_response(ws: Workspace, name: str, result):
    document = ws.load(name)
    return {**result, "document": document, "revision": case_revision.revision(document),
            "validation": service.validate(document)}


#: 428 Precondition Required: the save is refused until the caller sends back the digest of
#: what it was shown. Not 409 -- nothing conflicts, and the page treats that as another's edit.
CONFIRMATION_REQUIRED = 428


def consequences_of(ws: Workspace, name: str, document: dict[str, Any]) -> dict[str, Any]:
    """What saving ``document`` as the case ``name`` would do to what that case has built."""
    from eforecast.config import change_consequences
    saved = ws.load(name) if ws.exists(name) else None
    return change_consequences.report(ws.case_path(name), saved, document, case=name)


def check_consequences(ws: Workspace, name: str, document: dict[str, Any], expected: str | None,
                       accepted: str | None):
    """Refuse a save that makes built work be built again, until that was read and accepted.

    A case builds one model, and saving another configuration over the one it was built from
    is what makes the next run recreate datasets or train a family from the beginning. So the
    caller is shown exactly that, and the save goes through when it sends back the digest of
    what it was shown -- of *this* change against *this* model: a list accepted for another
    change, or before another run changed what is built, accepts nothing.

    Asked of the callers that take part in the revision protocol, which is the editor. A
    client that sends neither field keeps its established save, as it does for revisions; what
    its change replaces is still asked of whoever starts the run that replaces it.
    """
    if expected is None and accepted is None:
        return None
    try:
        found = consequences_of(ws, name, document)
    except Exception as error:                                  # noqa: BLE001 -- a check never blocks a save by failing
        logging.getLogger(__name__).warning('consequences of saving %s could not be worked out (%r)', name, error)
        return None
    if not found['consequential'] or accepted == found['digest']:
        return None
    return JSONResponse(status_code=CONFIRMATION_REQUIRED, content={
        "detail": "Saving this changes what the case's datasets or models were built from. Read what "
                  "will be built again and accept it to save.",
        "consequences": found})


@router.put("/cases/{name}")
def put_case(name: str, body: Document, scope: CaseScope = Depends(writable_or_new_case)):
    ws = scope.ws
    try:
        with case_revision.locked(ws, name):
            conflict = check_case_revision(ws, name, body.expected_revision)
            if conflict is not None:
                return conflict
            unaccepted = check_consequences(ws, name, body.document, body.expected_revision,
                                            body.accepted_consequences)
            if unaccepted is not None:
                return unaccepted
            return saved_case_response(ws, name, ws.save(name, body.document))
    except ServiceError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except ValidationError as exc:
        return _validation_response(exc)


@router.patch("/cases/{name}")
def patch_case(name: str, body: Sections, scope: CaseScope = Depends(writable_case)):
    """Save some sections of a case (the editors' per-step save); partial documents stay partial."""
    ws = scope.ws
    try:
        with case_revision.locked(ws, name):
            conflict = check_case_revision(ws, name, body.expected_revision)
            if conflict is not None:
                return conflict
            merged = service.merge_sections(ws.case_path(name), body.sections, tuple(body.replace))
            unaccepted = check_consequences(ws, name, merged, body.expected_revision, body.accepted_consequences)
            if unaccepted is not None:
                return unaccepted
            result = service.save_sections(ws.case_path(name), body.sections, tuple(body.replace))
            return saved_case_response(ws, name, result)
    except ValidationError as exc:
        return _validation_response(exc)


@router.post("/cases/{name}/consequences")
def post_consequences(name: str, body: Document, scope: CaseScope = Depends(readable_case)) -> dict[str, Any]:
    """What saving this draft would do to what the case has built, before it is saved.

    Which datasets a later run builds again, which families it trains from the beginning,
    and the settings whose change makes it so. A read: two configurations are resolved and the
    case's receipts and study journals inspected; nothing is saved, written or started.
    """
    ws = scope.ws
    try:
        with case_revision.locked(ws, name, shared=True):
            return consequences_of(ws, name, body.document)
    except ServiceError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:                                  # noqa: BLE001 -- said, not a 500
        raise HTTPException(status_code=400, detail=f'cannot work out the consequences: {error}') from error


@router.delete("/cases/{name}", status_code=204)
def delete_case(name: str, scope: CaseScope = Depends(managed_case)) -> None:
    try:
        with case_revision.locked(scope.ws, name):
            scope.ws.delete(name)
    except ServiceError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


class Share(BaseModel):
    shared: bool


@router.patch("/cases/{name}/share")
def patch_share(name: str, body: Share, scope: CaseScope = Depends(managed_case)) -> dict[str, Any]:
    with case_revision.locked(scope.ws, name):
        return scope.ws.share(name, body.shared)


class FamilyPresetRequest(BaseModel):
    name: str
    document: dict[str, Any] = {}


@router.post("/presets/family")
def post_family_preset(body: FamilyPresetRequest) -> dict[str, Any]:
    """The starting parameters of a method family, data options resolved against the document."""
    return family_preset(body.name, body.document)


# ---------------------------------------------------------------------------- experiment graph

@router.get("/layers")
def get_layers() -> dict[str, Any]:
    return catalogue()


@router.post("/graph/normalize")
def post_normalize(body: Experiments) -> dict[str, Any]:
    try:
        return graph_service.normalize_experiments(body.experiments)
    except ServiceError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.get("/graph/preset-templates")
def get_preset_templates(request: Request, source: str = "repo", path: str | None = None) -> dict[str, Any]:
    try:
        return graph_service.preset_templates(source, readable(request, path) if path else None)
    except ServiceError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.get("/graph/module-templates")
def get_module_templates(request: Request, path: str) -> dict[str, Any]:
    # This endpoint *runs* the file it is given. It may only ever be a module of a case this
    # caller owns, or the repository's own templates.
    try:
        return graph_service.module_templates(readable(request, path))
    except ServiceError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"cannot load {path}: {error}") from error


@router.post("/clustering/resolve-variables")
def post_resolve_rbf_variables(body: VarImpRequest) -> dict[str, Any]:
    """What every ``rbf_var_imp`` entry selects: the kind, the matched columns and whether it looks
    intended.  Resolved against the built row dataset when there is one, else against the variables
    the case declares."""
    from eforecast.config.clustering_variables import resolve
    try:
        return resolve(body.document, body.rule_sets)
    except ServiceError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.post("/graph/shapes")
def post_shapes(body: Document) -> dict[str, Any]:
    return graph_service.data_shapes(body.document)


@router.post("/graph/dry-build")
def post_dry_build(body: GraphRequest) -> dict[str, Any]:
    from eforecast.config.compute_workers import graph_preview
    from eforecast.training.command_line.resource_reporting import ResourceUnavailable
    try:
        return graph_preview(body.document, body.tag, body.method)
    except (ResourceUnavailable, TimeoutError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except ServiceError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


# ---------------------------------------------------------------------------- runs and studies

_run_manager: runs_module.RunManager | None = None
_run_manager_lock = threading.Lock()


def run_manager() -> runs_module.RunManager:
    global _run_manager
    with _run_manager_lock:
        if _run_manager is None or _run_manager.workspace.root != system_workspace().root:
            embedded = os.environ.get('EFORECAST_RUN_COORDINATOR', 'auto') == 'embedded'
            _run_manager = runs_module.RunManager(system_workspace(), coordinated=not embedded)
            if embedded:
                _run_manager.recover_pending()
        return _run_manager


@router.get("/scripts")
def get_scripts() -> dict[str, Any]:
    return runs_module.scripts_catalogue()


@router.get("/cases/{name}/runs")
def list_runs(name: str, scope: CaseScope = Depends(readable_case)) -> list[dict[str, Any]]:
    # An empty list for a case the caller cannot see said "no runs" where it should have said
    # "no such case": every case-scoped route answers the same way about a case that is not theirs.
    return _guard(lambda: run_manager().list(name))


@router.post("/cases/{name}/runs", status_code=201)
def start_run(name: str, body: RunRequest, scope: CaseScope = Depends(writable_case)) -> dict[str, Any]:
    def submit():
        manager = run_manager()
        record = manager.submit(name, body.script, body.options, client_key=body.client_key,
            expected_revision=body.expected_revision, submitted_by=scope.ws.user, action=body.action,
            accept_changes=bool(body.accept_changes))
        manager.enqueue(name, record['id'])
        return record
    return _guard(submit, conflict=('already has a running',))


@router.post("/cases/{name}/runs/preview")
def preview_run(name: str, body: RunRequest, scope: CaseScope = Depends(readable_case)) -> dict[str, Any]:
    return _guard(lambda: run_manager().preview(name, body.script, body.options,
                                               action=body.action or 'continue',
                                               expected_revision=body.expected_revision))


@router.get("/cases/{name}/runs/{run_id}")
def get_run(name: str, run_id: str, scope: CaseScope = Depends(readable_case)) -> dict[str, Any]:
    return _guard(lambda: run_manager().get(name, run_id))


@router.get("/cases/{name}/runs/{run_id}/children")
def run_children(name: str, run_id: str, after: str = '', limit: int = Query(100, ge=1, le=256),
                 scope: CaseScope = Depends(readable_case)) -> dict[str, Any]:
    def read():
        manager = run_manager()
        run = manager.get(name, run_id)
        if run.get('read_only'):
            return {'items': [], 'next': None}
        return manager.store.children(run['owner_id'], run['case_id'], run_id, after=after, limit=limit)
    return _guard(read)


@router.get("/cases/{name}/runs/{run_id}/log")
def get_run_log(name: str, run_id: str, offset: int = 0, limit: int = Query(262144, ge=1),
                scope: CaseScope = Depends(readable_case)) -> dict[str, Any]:
    return _guard(lambda: run_manager().log(name, run_id, offset, min(limit, 4 * 1024 * 1024)))


@router.get("/cases/{name}/runs/{run_id}/support")
def run_support_report(name: str, run_id: str, text: bool = False,
                       scope: CaseScope = Depends(readable_case)):
    """A14-03: one sanitized report a colleague can hand a maintainer.

    Assembled from what this installation already knows -- the durable run record, the A13
    doctor and a bounded tail of this run's own log. The doctor is the slow part and the
    report is still useful without it, so a doctor that cannot run becomes a stated absence
    rather than a failed request.
    """
    ws = scope.ws

    def gather():
        manager = run_manager()
        run = manager.get(name, run_id)
        try:
            revision = case_revision.revision(ws.load(name))
        except Exception:                               # noqa: BLE001 -- the run still matters
            revision = None
        try:
            log = manager.log(name, run_id, 0, support_report.LOG_BYTES * 4)
        except Exception:                               # noqa: BLE001
            log = None
        try:
            from eforecast.config import doctor as doctor_module
            health = doctor_module.report(timeout=2.0)
        except Exception as error:                      # noqa: BLE001
            health = {'verdict': None, 'counts': None,
                      'findings': [{'check': 'doctor', 'severity': 'attention',
                                    'finding': f'the doctor could not run ({error})',
                                    'remedy': 'run bin/eforecast doctor on the machine'}]}
        return support_report.assemble(case=name, run=run, revision=revision, doctor=health,
                                       log=log, links=studies_module.links())

    report = _guard(gather)
    if text:
        return Response(support_report.render(report), media_type='text/plain; charset=utf-8',
                        headers={'Content-Disposition':
                                 f'attachment; filename="{name}-{run_id}-support.txt"'})
    return report


@router.get("/cases/{name}/runs/{run_id}/log/download")
def download_run_log(name: str, run_id: str, scope: CaseScope = Depends(readable_case)):
    record = _guard(lambda: run_manager().get(name, run_id))
    path = Path(record['log'])
    if not path.is_file():
        raise HTTPException(status_code=404, detail="The run log is unavailable")
    return FileResponse(path, media_type='text/plain', filename=f'{name}-{run_id}.log')


@router.post("/cases/{name}/runs/{run_id}/stop")
def stop_run(name: str, run_id: str, scope: CaseScope = Depends(writable_case)) -> dict[str, Any]:
    return _guard(lambda: run_manager().stop(name, run_id))


class Resolution(BaseModel):
    reason: str


@router.post("/cases/{name}/runs/{run_id}/resolve")
def resolve_run(name: str, run_id: str, body: Resolution,
                scope: CaseScope = Depends(managed_case)) -> dict[str, Any]:
    """End a run whose execution or cleanup can never be confirmed, and free its case.

    The decision belongs to whoever owns the tenant, not to any member who can start runs:
    it releases a case that is being held because work may still be alive somewhere.
    """
    return _guard(lambda: run_manager().resolve(name, run_id, actor=scope.ws.user or 'local',
                                                reason=body.reason))


@router.delete("/cases/{name}/runs/{run_id}", status_code=204)
def delete_run(name: str, run_id: str, scope: CaseScope = Depends(writable_case)) -> Response:
    _guard(lambda: run_manager().delete(name, run_id))
    return Response(status_code=204)


class Recovery(BaseModel):
    apply: bool = False


@router.post("/cases/{name}/stages/recover")
def recover_stages(name: str, body: Recovery, scope: CaseScope = Depends(managed_case)) -> list[dict[str, Any]]:
    """What an interrupted publication left, and -- with ``apply`` -- the receipt it can put back.

    Restoring a receipt changes what the case's artifacts claim to be, so it belongs to whoever
    owns the tenant rather than to anyone who may start a run. It refuses while a publication is
    running, and it never deletes a half-promoted payload.
    """
    return _guard(lambda: run_manager().recover_stages(name, apply=body.apply))


@router.get("/cases/{name}/results")
def case_results(name: str, scope: CaseScope = Depends(readable_case)) -> dict[str, Any]:
    """A14-02: what the recorded artifacts say, each number labelled with the question it answers.

    The durable state is optional here. A results page that cannot be opened because the run
    store is down tells a colleague nothing; one that says the approved version could not be
    read tells them exactly what they are missing.
    """
    ws = scope.ws
    try:
        manager = run_manager()
        store, runs = manager.store, lambda: manager.list(name)
    except Exception:                                   # noqa: BLE001 -- the page degrades
        store, runs = None, None
    from .serving_scope import for_case
    return _guard(lambda: results_module.summary(ws.load(name), str(ws.case_path(name)),
        store=store, runs=runs, name=name,
        case_id=for_case(store, ws, name)['case_id'] if store else None))


@router.get("/cases/{name}/studies")
def get_studies(name: str, load: bool = True, scope: CaseScope = Depends(readable_case)) -> dict[str, Any]:
    ws = scope.ws
    return _guard(lambda: studies_module.studies(ws.load(name), str(ws.case_path(name)), load=load))


@router.get("/cases/{name}/studies/{cluster}/{method}")
def get_trials(name: str, cluster: str, method: str, limit: int = 500,
               scope: CaseScope = Depends(readable_case)) -> dict[str, Any]:
    ws = scope.ws
    return _guard(lambda: studies_module.trials(ws.load(name), cluster, method, str(ws.case_path(name)), limit=limit))


def _guard(fn, conflict: tuple[str, ...] = ()):
    try:
        return fn()
    except runs_module.ChangesNotAccepted as error:
        # Not an error of the request: the run would replace datasets or trained models that
        # were built from another configuration, and that is accepted by a person first. The
        # preview travels with the refusal, so the page can show what is asked at once.
        from fastapi.encoders import jsonable_encoder
        return JSONResponse(status_code=CONFIRMATION_REQUIRED, content={
            'detail': str(error), 'preview': jsonable_encoder(_jsonable(error.preview))})
    except runs_module.RunConflict as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except runs_module.StoreUnavailable as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except ServiceError as error:
        text = str(error)
        status = 409 if any(c in text for c in conflict) else (404 if "not found" in text or "does not exist" in text else 400)
        raise HTTPException(status_code=status, detail=text) from error


# ---------------------------------------------------------------------------- file system

@router.get("/browse")
def get_browse(request: Request, path: str | None = Query(default=None),
               mode: str = Query(default="dir")) -> dict[str, Any]:
    try:
        return service.browse(path, mode, caller=workspace(request))
    except ServiceError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


def _jsonable(value: Any) -> Any:
    """static_data holds sets, tuples, numpy scalars and Paths; make it JSON."""
    import numpy as np
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def create_app() -> FastAPI:
    auth.check_mode()        # a production installation does not start without authentication
    @asynccontextmanager
    async def lifespan(app):
        if os.environ.get('EFORECAST_RUN_COORDINATOR', 'auto') == 'auto':
            from eforecast.config.run_coordinator import ensure_service
            try:
                await asyncio.to_thread(ensure_service, system_workspace())
            except Exception:
                logging.getLogger(__name__).exception('coordinator startup unavailable; accepted work remains durable')
        yield  # the detached coordinator and its workers survive API shutdown/reload
    app = FastAPI(title="eforecast configuration", version="0.1", lifespan=lifespan)

    @app.middleware("http")
    async def require_session(request: Request, call_next):
        path = request.url.path
        open_paths = {"/api/login", "/api/me"}
        if path.startswith("/api/") and path not in open_paths and auth.auth_enabled():
            if auth.read_token(request.cookies.get(auth.COOKIE)) is None:
                return JSONResponse(status_code=401, content={"detail": "not logged in"})
        return await call_next(request)

    @app.exception_handler(case_revision.CaseBusy)
    def case_busy(request: Request, error: case_revision.CaseBusy) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": str(error)}, headers={"Retry-After": "1"})

    app.include_router(router)
    dist = Path(os.environ.get("EFORECAST_WEBUI_DIST", Path(__file__).resolve().parents[2] / "webui" / "dist"))
    if dist.is_dir():
        from fastapi.responses import FileResponse
        from fastapi.staticfiles import StaticFiles
        app.mount("/assets", StaticFiles(directory=str(dist / "assets")), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str):
            # a real file of the build is served as is; every other path is a client-side route -> index.html
            if path.startswith("api/") or path == "api":
                raise HTTPException(status_code=404, detail="unknown API route")
            candidate = (dist / path).resolve()
            if path and candidate.is_file() and dist.resolve() in candidate.parents:
                return FileResponse(candidate)
            return FileResponse(dist / "index.html")
    return app


app = create_app()
