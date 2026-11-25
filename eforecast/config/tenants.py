# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Tenants and memberships: who shares cases with whom.

A **tenant** owns cases; an account reaches a case only through a membership in the tenant that
owns it. Every account has a *personal* tenant of its own name, whose directory is the
``<workspace>/users/<name>/cases`` folder it already used, so nothing moves on disk and an
installation that never heard of tenants keeps working. A shared tenant, created by its owner,
keeps its cases under ``<workspace>/tenants/<id>/cases``.

The store is ``<workspace>/tenants.json``::

    {"acme": {"title": "Acme forecasting", "members": {"alice": "owner", "bob": "member"},
              "personal": false, "created": 1760000000.0}}

Roles, three of them, as the plan asks:

``owner``   members, the tenant itself, deleting and sharing a case -- everything;
``admin``   everything operational: create, edit, run, delete and share the tenant's cases;
``member``  use: read the tenant's cases, edit them and start runs, but not delete or share them.

A membership is read from this file on **every request**. It is never carried in a cookie, so a
removal takes effect on the next request.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

from eforecast.config.workspace import workspace_root

ROLES = ("owner", "admin", "member")
MANAGE_CASES = ("owner", "admin")          # delete or share a case
MANAGE_TENANT = ("owner",)                 # members, title, removal
TENANT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,31}$")


def tenants_file() -> Path:
    return workspace_root() / "tenants.json"


def load() -> dict[str, dict]:
    path = tenants_file()
    if not path.exists():
        return {}
    try:
        with path.open(encoding="utf-8") as stream:
            stored = json.load(stream)
    except ValueError:
        return {}
    return stored if isinstance(stored, dict) else {}


def save(tenants: dict[str, dict]) -> None:
    path = tenants_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        json.dump(tenants, stream, indent=1)
    os.replace(tmp, path)
    os.chmod(path, 0o600)


def check_id(tenant: str) -> str:
    if not TENANT_ID.match(tenant or ""):
        raise ValueError("tenant id: letters, digits, '_', '-' and '.', 32 characters at most")
    return tenant


def personal(name: str) -> dict:
    """The personal tenant of an account, as a record. It needs no file: it is the account."""
    return {"id": name, "title": name, "members": {name: "owner"}, "personal": True}


def record(tenant: str, name: str | None = None) -> dict | None:
    """A tenant, personal or shared. ``name`` is the caller, for whom its own id is personal."""
    stored = load().get(tenant)
    if stored is not None:
        return {"id": tenant, "title": stored.get("title") or tenant, "personal": False,
                "members": dict(stored.get("members") or {}), "created": stored.get("created")}
    if tenant == name or (name is None and tenant):
        return personal(tenant)
    return None


def role_in(tenant: str, name: str | None) -> str | None:
    """The caller's role in a tenant, or None when it is not a member -- the authorization answer.

    A stored tenant answers from its members alone. Only a tenant with no record is personal, and
    then its name is its only member's: the order matters, because a shared tenant must never be
    owned by whoever happens to hold an account of the same name.
    """
    if name is None:
        return None
    stored = load().get(tenant)
    if stored is not None:
        return (stored.get("members") or {}).get(name)
    return "owner" if tenant == name else None


def of(name: str | None) -> dict[str, str]:
    """Every tenant this account belongs to, as ``{tenant_id: role}``, its own first."""
    if name is None:
        return {}
    stored_tenants = load()
    mine = {} if name in stored_tenants else {name: "owner"}
    for tenant, stored in sorted(stored_tenants.items()):
        role = (stored.get("members") or {}).get(name)
        if role:
            mine[tenant] = role
    return mine


def is_tenant(name: str) -> bool:
    """Whether a shared tenant already holds this id -- an account may not take it afterwards."""
    return name in load()


def create(tenant: str, owner: str, title: str | None = None) -> dict:
    """A shared tenant with one owner. A personal tenant is never written here."""
    check_id(tenant)
    tenants = load()
    if tenant in tenants:
        raise ValueError(f"tenant {tenant!r} exists already")
    tenants[tenant] = {"title": title or tenant, "members": {owner: "owner"},
                       "created": time.time()}
    save(tenants)
    return record(tenant) or {}


def set_member(tenant: str, name: str, role: str) -> dict:
    if role not in ROLES:
        raise ValueError(f"role must be one of {', '.join(ROLES)}")
    tenants = load()
    if tenant not in tenants:
        raise KeyError(tenant)
    tenants[tenant].setdefault("members", {})[name] = role
    save(tenants)
    return {"tenant": tenant, "name": name, "role": role}


def remove_member(tenant: str, name: str) -> None:
    tenants = load()
    if tenant not in tenants:
        raise KeyError(tenant)
    members = tenants[tenant].setdefault("members", {})
    if members.get(name) == "owner" and [r for r in members.values() if r == "owner"] == ["owner"]:
        raise ValueError("a tenant cannot lose its last owner")
    members.pop(name, None)
    save(tenants)


def remove(tenant: str) -> None:
    """Forget a shared tenant. Its case directory is left on disk: data is never deleted here."""
    tenants = load()
    if tenant not in tenants:
        raise KeyError(tenant)
    tenants.pop(tenant)
    save(tenants)


def forget_account(name: str) -> None:
    """Drop an account from every tenant it belonged to, when the account itself is removed."""
    tenants = load()
    changed = False
    for stored in tenants.values():
        if (stored.get("members") or {}).pop(name, None) is not None:
            changed = True
    if changed:
        save(tenants)
