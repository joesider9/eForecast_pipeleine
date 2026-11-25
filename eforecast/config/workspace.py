# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Where this installation keeps its cases, and who may look at them.

The other half of what ``service`` used to be, and the half almost everything wants: 60 places
import ``Workspace`` and 7 ``workspace_root``. Keeping it with the document editing operations
meant that a module needing to know where a case lives also imported pydantic, the schema and
the option-sync rules.

Cases live under one root (``EFORECAST_WORKSPACE``), one directory per case holding the part
files and ``main.yaml``. Browsing the file system is restricted to ``EFORECAST_BROWSE_ROOTS``
(colon separated; default: the workspace root, ``/media`` and the home directory).

One edge leaves this module for the document layer: saving a case validates it, so ``Workspace``
calls ``case_document.save_case``. Nothing in ``case_document`` points back here.
"""
from __future__ import annotations

import copy
import json
import os
import re
import shutil
import uuid
from pathlib import Path
from typing import Any

from eforecast.config.case_document import SECTION_MODELS, save_case
from eforecast.config.config_files import (config_paths, existing_parts, load_document, missing_sections,
                                           save_parts, validation_note)
from eforecast.config.errors import ServiceError, WorkspaceNotConfigured
from eforecast.config.machine_defaults import machine_defaults

LAB_WORKSPACES = (("/media/sider/data", "/media/sider/data//eforecast_workspace"),
                  ("/media/smartrue/HHD1", "/media/smartrue/HHD1/George/eforecast_workspace"))


def _default_workspace() -> str | None:
    """The workspace of a known lab workstation, or ``None`` on any other machine.

    This used to fall back to ``"/"``.  Nothing then failed at startup: the service simply
    adopted the filesystem root as its workspace and went on to create ``users.json``,
    ``tenants.json``, ``.secret`` and every case directory directly in ``/``.  On a colleague's
    own machine that is a baffling permission error far from its cause, and where it is
    permitted it is worse than an error.  A machine that is not recognised now has no default
    and must say where its workspace is.
    """
    return next((workspace for marker, workspace in LAB_WORKSPACES if os.path.exists(marker)), None)


DEFAULT_WORKSPACE = _default_workspace()
CASE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


# ---------------------------------------------------------------------------- workspace and browsing

WORKSPACE_REMEDY = ("Set EFORECAST_WORKSPACE to a directory this account can write, for example\n"
                    "    EFORECAST_WORKSPACE=$HOME/eforecast_workspace\n"
                    "or pass it to the service: python -m eforecast.config.serve --workspace "
                    "$HOME/eforecast_workspace")


def workspace_root() -> Path:
    """Where users, tenants and cases live.

    The explicit ``EFORECAST_WORKSPACE`` wins, then a known lab workstation's own path. The
    filesystem root is refused from either source: it is never a workspace, and accepting it
    turns a missing setting into files written into ``/``.
    """
    configured = os.environ.get("EFORECAST_WORKSPACE") or None
    raw = configured or DEFAULT_WORKSPACE
    if not raw:
        raise WorkspaceNotConfigured(
            "No workspace directory is configured and this machine is not a known lab "
            "workstation, so there is nowhere to keep users, tenants and cases.\n" + WORKSPACE_REMEDY)
    root = Path(raw).expanduser()
    if root == Path(root.anchor):
        source = "EFORECAST_WORKSPACE" if configured else "the built-in default"
        raise WorkspaceNotConfigured(
            f"{source} names the filesystem root {str(root)!r}, which is not a workspace: the "
            "service would create users.json, tenants.json, .secret and every case directory "
            "there.\n" + WORKSPACE_REMEDY)
    return root


def browse_roots(caller: "Workspace | None" = None) -> list[Path]:
    """Where this caller may look.

    The workspace root used to be a browse root for everybody, so any logged-in user could list
    ``<workspace>/users`` and read the names and paths of every other user's cases. A caller who
    is not an administrator now gets **its own** case directories instead of the whole workspace,
    beside the data roots -- the target files and NWP folders a case needs are outside the
    workspace, and browsing those is the point of this endpoint.
    """
    raw = os.environ.get("EFORECAST_BROWSE_ROOTS")
    configured = [Path(item).expanduser() for item in raw.split(":") if item] if raw else None
    if caller is None or caller.admin or caller.user is None:
        roots = configured if configured is not None else [workspace_root(), Path("/media"), Path.home()]
        return [root.resolve() for root in roots]
    workspace = workspace_root().resolve()
    outside = [root.resolve() for root in (configured or [Path("/media"), Path.home()])
               if not _within(root.resolve(), workspace)]
    return [*(folder for folder in caller.own_roots() if folder.exists()), *outside]


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def inside_roots(path: str | Path, roots: list[Path] | None = None) -> Path:
    """The resolved path, or ServiceError when it is outside every allowed root."""
    resolved = Path(path).expanduser().resolve()
    for root in roots or browse_roots():
        try:
            resolved.relative_to(root)
            return resolved
        except ValueError:
            continue
    raise ServiceError(f"{path} is outside the allowed folders ({', '.join(str(r) for r in (roots or browse_roots()))})")


def browse(path: str | None = None, mode: str = "dir", caller: "Workspace | None" = None) -> dict[str, Any]:
    """Entries of a folder inside the allowed roots: ``{'path', 'parent', 'dirs', 'files', 'roots'}``.

    ``mode`` 'dir' lists folders only, 'file' lists folders and files. ``caller`` decides which
    roots those are: a user is not shown the workspace of the other users."""
    roots = browse_roots(caller)
    if not path:
        return {"path": None, "parent": None, "dirs": [str(r) for r in roots if r.exists()], "files": [],
                "roots": [str(r) for r in roots]}
    base = inside_roots(path, roots)
    if not base.is_dir():
        raise ServiceError(f"{path} is not a folder")
    dirs, files = [], []
    try:
        for entry in sorted(base.iterdir(), key=lambda e: e.name.lower()):
            if entry.name.startswith("."):
                continue
            (dirs if entry.is_dir() else files).append(entry.name)
    except OSError as error:
        raise ServiceError(f"cannot read {path}: {error}") from error
    parent = None if any(base == r for r in roots) else str(base.parent)
    return {"path": str(base), "parent": parent, "dirs": dirs, "files": files if mode == "file" else [],
            "roots": [str(r) for r in roots]}


TENANT_ROLES = ("owner", "admin", "member")      # a member may use the tenant's cases
MANAGING_ROLES = ("owner", "admin")              # ... delete or share one


class Workspace:
    """The cases of the web UI.

    Single-user mode (``user`` None): one directory per case under ``<root>/cases``.
    Multi-user mode: every case belongs to a **tenant**, and a caller reaches it only through a
    membership in that tenant (``eforecast.config.tenants``). An account's personal tenant is its
    own name and its directory is ``<root>/users/<user>/cases``, the folder it already had; a
    shared tenant keeps its cases under ``<root>/tenants/<id>/cases``. A case whose ``.case.json``
    says ``shared`` stays visible to everybody, read-only -- the installation-wide flag that
    predates tenants. An admin (or ``all=True``, used by the run manager) sees and writes every
    case, including the single-user directory. Case names are unique across the workspace, so
    ``/api/cases/{name}`` needs no owner.

    ``tenant`` is where a *new* case goes; it defaults to the caller's personal tenant.
    """

    META = ".case.json"

    def __init__(self, root: str | Path | None = None, user: str | None = None, admin: bool = False,
                 tenant: str | None = None):
        self.root = Path(root).expanduser() if root else workspace_root()
        self.user = user
        self.admin = admin
        self._cached_memberships: dict[str, str] | None = None
        self.tenant = tenant or user
        self.cases_dir = self.tenant_dir(self.tenant) if user else (self.root / "cases")

    # ------------------------------------------------------------- tenants
    def tenant_dir(self, tenant: str | None) -> Path:
        """Where a tenant's cases live: an account's own folder, or a shared tenant's folder."""
        if tenant is None:
            return self.root / "cases"
        return (self.root / "users" / tenant / "cases") if self._personal(tenant) \
            else (self.root / "tenants" / tenant / "cases")

    def _personal(self, tenant: str) -> bool:
        """A tenant with no record in the store is an account's own: that is what makes the
        upgrade free of migration -- ``users/<name>/cases`` is already the personal tenant."""
        from eforecast.config import tenants as tenant_store
        return tenant_store.load().get(tenant) is None

    def _memberships(self) -> dict[str, str]:
        """The caller's tenants and roles, read from the store once per Workspace -- and a
        Workspace is built per request, so a removed membership is gone on the next request. A
        membership is never carried in the session cookie."""
        if self._cached_memberships is None:
            from eforecast.config import tenants as tenant_store
            self._cached_memberships = tenant_store.of(self.user)
        return self._cached_memberships

    def memberships(self) -> dict[str, str]:
        return dict(self._memberships())

    def role_in(self, tenant: str | None) -> str | None:
        """The caller's role in a tenant: ``owner``/``admin``/``member``, or None for a stranger."""
        if self.user is None:
            return "owner"
        return self._memberships().get(tenant or "")

    def own_roots(self) -> list[Path]:
        """The folders this caller may look in: the case folder of every tenant it belongs to, and
        each case shared with it by a tenant it does not belong to."""
        folders = [folder.resolve() for folder in
                   (self.tenant_dir(tenant) for tenant in self._memberships()) if folder.exists()]
        for folder in self._all_case_dirs():
            if folder.parent.resolve() not in folders and self._visible(folder, self._meta(folder)):
                folders.append(folder.resolve())
        return folders

    def may_read(self, path: str | Path) -> Path:
        """A path this caller is allowed to read, or ``ServiceError``.

        Reading a file is allowed inside a case this caller may see, or anywhere the browse roots
        of this caller permit. It is **not** allowed merely because the path is somewhere under the
        workspace: that is how one user came to read another user's configuration through the
        template endpoints.
        """
        resolved = Path(path).expanduser().resolve()
        for folder in self.own_roots():
            if _within(resolved, folder):
                return resolved
        workspace = self.root.resolve()
        for root in browse_roots(self):
            if _within(resolved, root) and not _within(resolved, workspace):
                return resolved
        raise ServiceError(f"{path} is not yours to read")

    def _check_name(self, name: str) -> str:
        if not CASE_NAME.match(name or ""):
            raise ServiceError(f"case name {name!r}: letters, digits, '_', '-', '.' only (max 64), starting with a letter or digit")
        return name

    # ------------------------------------------------------------- where cases live
    def _all_case_dirs(self) -> list[Path]:
        """Every case directory: the single-user one, every account's, and every shared tenant's."""
        dirs: list[Path] = []
        for cases_dir in [self.root / "cases", *sorted((self.root / "users").glob("*/cases")),
                          *sorted((self.root / "tenants").glob("*/cases"))]:
            if cases_dir.is_dir():
                dirs.extend(sorted((e for e in cases_dir.iterdir() if e.is_dir() and not e.name.startswith(".")), key=lambda e: e.name.lower()))
        return dirs

    def _meta(self, folder: Path) -> dict[str, Any]:
        path = folder / self.META
        if path.is_file():
            try:
                return json.loads(path.read_text())
            except ValueError:
                pass
        return {"owner": self._owner_of(folder), "shared": False}

    def _owner_of(self, folder: Path) -> str | None:
        """The account a case folder sits under, for a case whose metadata is missing."""
        return folder.parent.parent.name if folder.parent.parent.parent.name == "users" \
            and folder.parent.name == "cases" else None

    def _tenant_of(self, folder: Path, meta: dict[str, Any]) -> str | None:
        """The tenant that owns a case. A case stored before tenants existed has none written, so
        it is read from where it sits: a shared tenant's folder, or its owner's personal tenant.
        A stored value is normalised, never rejected."""
        if meta.get("tenant"):
            return str(meta["tenant"])
        if folder.parent.parent.parent.name == "tenants" and folder.parent.name == "cases":
            return folder.parent.parent.name
        return meta.get("owner") or self._owner_of(folder)

    def _write_meta(self, folder: Path, meta: dict[str, Any]) -> None:
        (folder / self.META).write_text(json.dumps(meta, indent=1))

    def _visible(self, folder: Path, meta: dict[str, Any]) -> bool:
        if self.admin or self.user is None:
            return True
        return self.role_in(self._tenant_of(folder, meta)) is not None or bool(meta.get("shared"))

    def _writable(self, folder: Path, meta: dict[str, Any]) -> bool:
        """Any member of the owning tenant may use a case: read it, edit it and run it."""
        if self.admin or self.user is None:
            return True
        return self.role_in(self._tenant_of(folder, meta)) in TENANT_ROLES

    def _manageable(self, folder: Path, meta: dict[str, Any]) -> bool:
        """Deleting or sharing a case is the owning tenant's owner's and admins' to do."""
        if self.admin or self.user is None:
            return True
        return self.role_in(self._tenant_of(folder, meta)) in MANAGING_ROLES

    def _find(self, name: str) -> Path | None:
        self._check_name(name)
        own = self.cases_dir / name
        if own.is_dir():
            return own
        for folder in self._all_case_dirs():
            if folder.name == name:
                return folder
        return None

    def case_dir(self, name: str) -> Path:
        """The directory of an existing case visible to this user, or where a new one would go."""
        found = self._find(name)
        if found is None:
            return self.cases_dir / name
        if not self._visible(found, self._meta(found)):
            raise ServiceError(f"case {name!r} does not exist")
        return found

    def case_path(self, name: str) -> Path:
        """The path handed to config_files: the case directory (parts + main.yaml)."""
        return self.case_dir(name)

    def tenant_of(self, name: str) -> str | None:
        """The tenant that owns a case visible to this caller."""
        found = self._find(name)
        return None if found is None else self._tenant_of(found, self._meta(found))

    def taken(self, name: str) -> bool:
        """Whether this installation already has a case of that name, visible to this caller or not.

        Case names are unique across the workspace by design, so a name somebody else uses cannot
        be reused -- and the caller must not be told whose it is.
        """
        return self._find(name) is not None

    def exists(self, name: str) -> bool:
        found = self._find(name)
        return found is not None and self._visible(found, self._meta(found))

    def can_write(self, name: str) -> bool:
        found = self._find(name)
        return found is not None and self._visible(found, self._meta(found)) and self._writable(found, self._meta(found))

    def check_writable(self, name: str) -> None:
        if not self.exists(name):
            raise ServiceError(f"case {name!r} does not exist")
        if not self.can_write(name):
            found = self._find(name)
            raise ServiceError(f"case {name!r} is read-only for you: it belongs to "
                               f"{self._tenant_of(found, self._meta(found))}")

    def check_manageable(self, name: str) -> None:
        """Deleting or sharing a case: a ``member`` of the tenant may use it but not dispose of it."""
        self.check_writable(name)
        found = self._find(name)
        if not self._manageable(found, self._meta(found)):
            tenant = self._tenant_of(found, self._meta(found))
            raise ServiceError(f"case {name!r}: deleting or sharing a case of tenant {tenant!r} is "
                               f"for its owner and admins; you are a member")

    def share(self, name: str, shared: bool) -> dict[str, Any]:
        self.check_manageable(name)
        folder = self._find(name)
        meta = self._meta(folder)
        meta["shared"] = bool(shared)
        self._write_meta(folder, meta)
        return meta

    def list(self) -> list[dict[str, Any]]:
        cases = []
        for entry in self._all_case_dirs():
            meta = self._meta(entry)
            if not self._visible(entry, meta):
                continue
            main, split_dir = config_paths(entry)
            document = load_document(entry) if (main.is_file() or existing_parts(split_dir)) else {}
            project = document.get("project") or {}
            cases.append({"name": entry.name, "project": project.get("name"), "type": project.get("type"),
                          "horizon_type": project.get("horizon_type"), "complete": not missing_sections(document),
                          "parts": [part.stem for part in existing_parts(split_dir)],
                          "modified": max([p.stat().st_mtime for p in existing_parts(split_dir)] + [0.0]),
                          "owner": meta.get("owner"), "tenant": self._tenant_of(entry, meta),
                          "role": self.role_in(self._tenant_of(entry, meta)),
                          "shared": bool(meta.get("shared")), "writable": self._writable(entry, meta),
                          "manageable": self._manageable(entry, meta)})
        return cases

    def load(self, name: str) -> dict[str, Any]:
        if not self.exists(name):
            raise ServiceError(f"case {name!r} does not exist")
        return load_document(self.case_path(name))

    def save(self, name: str, document: dict[str, Any]) -> dict[str, Any]:
        new_case = self._find(name) is None
        if not new_case:
            self.check_writable(name)
        folder = self.case_dir(name)
        folder.mkdir(parents=True, exist_ok=True)
        if new_case:
            self._write_meta(folder, {"id": uuid.uuid4().hex, "owner": self.user,
                                      "tenant": self.tenant if self.user else None, "shared": False})
        elif not (folder / self.META).exists() and self.user:
            self._write_meta(folder, {"owner": self.user, "tenant": self.tenant, "shared": False})
        return save_case(document, self.case_path(name))

    def create(self, name: str, document: dict[str, Any] | None = None, copy_from: str | None = None,
               tenant: str | None = None) -> dict[str, Any]:
        """A new case from a document, from another case, or from the machine defaults (partial documents are
        written as parts without full validation, so a case can be built up step by step).

        ``tenant`` is the tenant that will own it; the default is the caller's own. A caller that
        is not a member of the named tenant is told the tenant does not exist."""
        self._check_name(name)
        if self._find(name) is not None:
            raise ServiceError(f"case {name!r} exists already" + ("" if self.exists(name) else " (another user owns that name)"))
        owning = tenant or self.tenant
        if self.user is not None and owning != self.tenant:
            if self.role_in(owning) not in TENANT_ROLES:
                raise ServiceError(f"tenant {owning!r} does not exist")
        target = (self.tenant_dir(owning) if self.user else self.cases_dir) / name
        if copy_from:
            document = self._copied(copy_from, name)
        if document is None:
            document = {key: value for key, value in machine_defaults().items() if key in SECTION_MODELS}
        target.mkdir(parents=True)
        self._write_meta(target, {"id": uuid.uuid4().hex, "owner": self.user,
                                  "tenant": owning if self.user else None, "shared": False})
        if missing_sections(document):
            written = save_parts(document, target)
            return {"main": None, "parts": [str(p) for p in written], "note": validation_note(document)}
        return save_case(document, target)

    def _copied(self, copy_from: str, name: str) -> dict[str, Any]:
        """The source's settings, under an identity of the new case's own.

        ``project.name`` is not decoration: the model store path is
        ``<group>/<type>/<project name>/<horizon type>/model_ver<N>`` and ``_id`` is the project
        name too. A plain deep copy therefore produced a second case addressing the **same**
        model folder as its source -- so the copy read the source's datasets and would overwrite
        its frozen masks, its global scaler fits and its trained models, while appearing to be a
        new site. Copying ``pv_park`` really did yield
        ``.../pv/Aggiti/long_term/model_ver1`` twice.

        The new case therefore starts under its own name. The site group of the header is where a
        colleague puts the real site name afterwards, which moves the folder again -- never onto
        the source. Everything else is preserved exactly, including fields this version does not
        know about.
        """
        document = copy.deepcopy(self.load(copy_from))
        project = document.get("project")
        if isinstance(project, dict) and project.get("name"):
            project["name"] = name
        return document

    def delete(self, name: str) -> None:
        self.check_manageable(name)
        shutil.rmtree(self.case_dir(name))
