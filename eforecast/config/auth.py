# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Users and sessions of the configuration web UI.

Users live in ``<workspace>/users.json`` as ``{name: {"salt", "hash", "role"}}`` (PBKDF2-SHA256; role
``admin`` or ``user``; on a workstation the first user created is an admin, in
``EFORECAST_MODE=production`` nobody becomes one without ``--admin``).  Every user owns the cases under
``<workspace>/users/<name>/cases`` and sees the cases other users marked as shared, read-only.
Admins see and edit every case and manage the users.  A login
sets a signed cookie (``itsdangerous``); every ``/api`` route except ``/api/login`` needs it.
``EFORECAST_AUTH=off`` disables the check (single-user desktop use, tests).

    python -m eforecast.config.auth add <name> [--admin]   # prompts for the password
    python -m eforecast.config.auth list
    python -m eforecast.config.auth remove <name>
"""
from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import re
import secrets
import sys
from pathlib import Path
from typing import Any

from itsdangerous import BadSignature, URLSafeTimedSerializer

from eforecast.config.service import workspace_root

COOKIE = "eforecast_session"
SESSION_SECONDS = 12 * 3600
ITERATIONS = 200_000


def users_file() -> Path:
    return workspace_root() / "users.json"


def load_users() -> dict[str, dict[str, str]]:
    path = users_file()
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def save_users(users: dict[str, dict[str, str]]) -> None:
    path = users_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(users, stream, indent=1)
    os.chmod(path, 0o600)


def _hash(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), ITERATIONS).hex()


USER_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,31}$")


def add_user(name: str, password: str, role: str | None = None) -> dict[str, str]:
    """Create or reset a user; the first user of the file becomes admin unless a role is given."""
    if not USER_NAME.match(name or ""):
        raise ValueError("user name: letters, digits, '_', '-' and '.', 32 characters at most")
    if len(password or "") < 6:
        raise ValueError("password: at least 6 characters")
    from eforecast.config import tenants           # deferred: tenants reads the same workspace
    if name not in load_users() and tenants.is_tenant(name):
        # an account of a shared tenant's name would otherwise look like that tenant's owner
        raise ValueError(f"{name!r} is the name of a tenant: choose another account name")
    users = load_users()
    if role is None:
        # "the first account wins" is a workstation convenience and a hosted vulnerability: in
        # production an administrator is only ever created by asking for one explicitly.
        role = users.get(name, {}).get("role") or \
            ("user" if production() or users else "admin")
    if role not in ("admin", "user"):
        raise ValueError("role must be admin or user")
    salt = secrets.token_hex(16)
    users[name] = {"salt": salt, "hash": _hash(password, salt), "role": role}
    save_users(users)
    return {"name": name, "role": role}


def set_role(name: str, role: str) -> None:
    users = load_users()
    if name not in users:
        raise KeyError(name)
    if role not in ("admin", "user"):
        raise ValueError("role must be admin or user")
    users[name]["role"] = role
    save_users(users)


def role_of(name: str | None) -> str | None:
    if name is None:
        return None
    if not auth_enabled():
        return "admin"
    user = load_users().get(name)
    return (user.get("role") or "user") if user else None


def is_admin(name: str | None) -> bool:
    return role_of(name) == "admin"


def list_users() -> list[dict[str, str]]:
    return [{"name": n, "role": u.get("role") or "user"} for n, u in sorted(load_users().items())]


def remove_user(name: str) -> None:
    """Remove an account and every membership it held: an account's access has one place to end."""
    from eforecast.config import tenants           # deferred: tenants reads the same workspace
    users = load_users()
    users.pop(name, None)
    save_users(users)
    tenants.forget_account(name)


def check_password(name: str, password: str) -> bool:
    user = load_users().get(name)
    return bool(user) and secrets.compare_digest(_hash(password, user["salt"]), user["hash"])


def auth_enabled() -> bool:
    return os.environ.get("EFORECAST_AUTH", "on").lower() not in {"off", "0", "false", "no"}


def mode() -> str:
    """``production`` for a hosted installation, ``workstation`` (the default) for the owner's own."""
    return os.environ.get("EFORECAST_MODE", "workstation").strip().lower() or "workstation"


def production() -> bool:
    return mode() == "production"


def check_mode() -> None:
    """Refuse to start a production installation with the authentication turned off.

    A startup check, not a request-time one: the failure has to be loud, immediate and impossible
    to reach by a request. ``EFORECAST_AUTH=off`` stays the single-owner workstation default.
    """
    if production() and not auth_enabled():
        raise RuntimeError("EFORECAST_AUTH=off is refused when EFORECAST_MODE=production")
    if production() and not os.environ.get("EFORECAST_SECRET"):
        raise RuntimeError("EFORECAST_MODE=production needs EFORECAST_SECRET: a cookie secret that "
                           "survives a restart and is shared by every process")


#: A16-10: the cookie secret's name in the installation's one secret source.
SESSION_SECRET = "session_secret"


def _secret() -> str:
    """The cookie-signing secret, from one place.

    Resolution order, and why: ``EFORECAST_SECRET`` first, so an existing deployment that
    already injects one keeps working unchanged; then the A16-10 secret source, which is where a
    rotation writes; then the historical ``<workspace>/.secret``, which is adopted rather than
    broken. Only when nothing exists at all is one generated, and it is generated into the
    source so there is a single place to rotate next time.
    """
    from eforecast.config import secrets_source
    env = os.environ.get("EFORECAST_SECRET")
    if env:
        return env
    try:
        return secrets_source.resolve(SESSION_SECRET).reveal()
    except secrets_source.SecretMissing:
        pass
    legacy = workspace_root() / ".secret"
    if legacy.exists():
        return legacy.read_text().strip()
    secrets_source.store(SESSION_SECRET, secrets.token_hex(32))
    return secrets_source.resolve(SESSION_SECRET).reveal()


def make_token(name: str) -> str:
    return URLSafeTimedSerializer(_secret(), salt="session").dumps({"user": name, "epoch": session_epoch(name)})


def session_epoch(name: str | None) -> str | None:
    """What a session is bound to besides the name: the credential it was issued against.

    A cookie used to carry a name and a signature and nothing else, so it outlived the account:
    a removed user, or one whose password had just been reset, kept working until the twelve hours
    ran out. The epoch changes whenever the stored credential does.

    A14-04 adds a second reason for it to change: ``revoke_sessions`` sets a nonce, which ends
    every existing session without touching the password -- what decommissioning a machine
    needs, since the machine holds a cookie and the person keeps their account. The nonce
    joins the digest only when it exists, so adopting this does not log anyone out.
    """
    user = load_users().get(name or "")
    if not user:
        return None
    body = f"{user.get('salt')}:{user.get('hash')}"
    nonce = user.get("sessions")
    if nonce:
        body = f"{body}:{nonce}"
    return hashlib.sha256(body.encode()).hexdigest()[:16]


def revoke_sessions(name: str) -> str:
    """End every session for one account, on every machine, without changing its password.

    Per *account*, not per machine: a cookie records who it belongs to, not where it was
    issued, so there is nothing to revoke one machine by. For decommissioning that is the
    safe direction -- it logs the retired machine out, and the person signs in again
    everywhere else with the password they already have.
    """
    users = load_users()
    if name not in users:
        raise ValueError(f"no such user: {name}")
    users[name]["sessions"] = secrets.token_hex(8)
    save_users(users)
    return users[name]["sessions"]


def read_token(token: str | None) -> str | None:
    """The user this cookie belongs to, if that user still exists with that credential."""
    if not token:
        return None
    try:
        data = URLSafeTimedSerializer(_secret(), salt="session").loads(token, max_age=SESSION_SECONDS)
    except BadSignature:
        return None
    name = data.get("user")
    if not auth_enabled():
        return name
    epoch = session_epoch(name)
    if epoch is None or data.get("epoch") != epoch:
        # The account is gone, suspended or its password changed: this session is over. Membership
        # is never read once at login and trusted afterwards.
        return None
    return name


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command")
    parser.set_defaults(command="add")
    add = sub.add_parser("add"); add.add_argument("name"); add.add_argument("--password")
    parser.set_defaults(name='guest')
    parser.set_defaults(password='111111')
    add.add_argument("--admin", action="store_true", help="administrator (sees and edits every case, manages users)")
    parser.set_defaults(admin=False)
    rem = sub.add_parser("remove"); rem.add_argument("name")
    rev = sub.add_parser("revoke", help="end every session for an account, keeping its password")
    rev.add_argument("name")
    role = sub.add_parser("role"); role.add_argument("name"); role.add_argument("role", choices=["admin", "user"])
    sub.add_parser("list")
    args = parser.parse_args(argv)
    if args.command == "add":
        password = args.password or getpass.getpass(f"password for {args.name}: ")
        info = add_user(args.name, password, "admin" if args.admin else None)
        print(f"user {args.name} ({info['role']}) saved in {users_file()}")
    elif args.command == "remove":
        remove_user(args.name); print(f"user {args.name} removed")
    elif args.command == "role":
        set_role(args.name, args.role); print(f"user {args.name} is now {args.role}")
    elif args.command == "revoke":
        revoke_sessions(args.name)
        print(f"every session for {args.name} has ended; the password is unchanged")
    else:
        print("\n".join(f"{u['name']}  {u['role']}" for u in list_users()) or "(no users)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
