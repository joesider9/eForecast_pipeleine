# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""A16-10: one place each credential lives, per deployment shape.

The lab has already shipped a credential by accident: the OpenWeather key was tracked in git,
pushed to three branches and baked into the running worker image. That happened with one secret
and one image. A16 adds a registry, a broker, a database, an object store and AWS, so the count
grows in every direction, and a secret in a layer or a queue message cannot be recalled by
deleting it afterwards.

So a credential has exactly one authoritative source per deployment shape, and everything else
carries a **reference** instead of a value:

====================  ==========================================================================
local                 one file per secret under ``EFORECAST_SECRETS_DIR``, default
                      ``~/.config/eforecast/secrets``, mode 0600, outside the repository
cluster               the same per-file layout, mounted read-only from a Kubernetes Secret, so
                      the resolver below is unchanged -- specified now, qualified by A16-05
aws                   scoped workload identity where the service supports it, else a secrets
                      manager reference resolved into the same layout -- qualified by A16-09
====================  ==========================================================================

A mounted copy is a delivery mechanism, not a second source: nothing edits it in place.

``reference(name)`` is what may appear in a case, a run record, a broker message, a support
report or an evidence file. ``resolve(name)`` is the only thing that returns a value, and it is
never logged, never put in a finding and never written into a record. ``Secret`` exists so that
a value cannot be printed by accident: its ``repr`` and ``str`` are redacted, and reading it is
an explicit ``.reveal()``.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

#: Deployment shapes. ``local`` is implemented; the others are specified and not yet qualified.
SHAPES = ('local', 'cluster', 'aws')

#: A secret name is a stable reference, so it must be boring.
NAME = re.compile(r'^[a-z][a-z0-9_]{1,62}$')

#: What a reference looks like wherever a record needs to say *which* credential, not what it is.
SCHEME = 'eforecast-secret'


class SecretMissing(RuntimeError):
    """The credential is not in its authoritative source, and nothing will guess one."""


class Secret:
    """A value that does not print itself.

    Every accident this task exists to prevent starts with a value reaching somewhere it was
    never meant to be -- a log line, an exception message, a JSON record. Wrapping it means the
    careless paths are safe and only an explicit ``reveal()`` is not.
    """

    __slots__ = ('_value', 'name')

    def __init__(self, name: str, value: str):
        self.name = name
        self._value = value

    def reveal(self) -> str:
        return self._value

    def __repr__(self) -> str:
        return f'<Secret {self.name} redacted>'

    __str__ = __repr__

    def __format__(self, spec) -> str:
        return self.__repr__()

    def __bool__(self) -> bool:
        return bool(self._value)

    def __len__(self) -> int:
        return len(self._value)


def shape() -> str:
    """Which deployment this process is part of."""
    value = str(os.environ.get('EFORECAST_DEPLOYMENT', 'local')).lower()
    return value if value in SHAPES else 'local'


def directory() -> Path:
    """The one authoritative source for this shape: a directory of one file per secret.

    The cluster shape mounts a Kubernetes Secret at this path, so the resolver does not change
    between deployments -- which is the point of naming the source rather than the mechanism.
    """
    configured = os.environ.get('EFORECAST_SECRETS_DIR')
    if configured:
        return Path(configured).expanduser()
    base = Path(os.environ.get('XDG_CONFIG_HOME', Path.home() / '.config'))
    return base / 'eforecast' / 'secrets'


def reference(name: str) -> str:
    """The non-secret name a record may carry: which credential, never what it is."""
    if not NAME.match(name or ''):
        raise ValueError('a secret name is lowercase letters, digits and underscores')
    return f'{SCHEME}:{shape()}:{name}'


def parse(value: str) -> str | None:
    """The secret name inside a reference, or ``None`` when it is not one."""
    parts = str(value or '').split(':')
    return parts[2] if len(parts) == 3 and parts[0] == SCHEME and NAME.match(parts[2]) else None


def path_of(name: str) -> Path:
    if not NAME.match(name or ''):
        raise ValueError('a secret name is lowercase letters, digits and underscores')
    return directory() / name


def present(name: str) -> bool:
    return path_of(name).is_file()


def resolve(name: str, *, environment: str | None = None) -> Secret:
    """The credential, from its one source. Raises rather than returning a guess.

    ``environment`` names a variable that may carry the value instead, for the existing
    deployments that already inject one that way. It is read first precisely so adopting this
    module never requires moving a working secret on the same day.
    """
    if environment:
        injected = os.environ.get(environment)
        if injected:
            return Secret(name, injected)
    path = path_of(name)
    try:
        value = path.read_text(encoding='utf-8').strip()
    except FileNotFoundError:
        raise SecretMissing(
            f'{name} is not in this installation\'s secret source.\n'
            f'Put it in {path} with mode 0600 (and nowhere else):\n'
            f'    install -m 600 /dev/null {path} && $EDITOR {path}\n'
            f'The repository, an image layer, a case file and a run record are not places for it.'
        ) from None
    except OSError as error:
        raise SecretMissing(f'{name} could not be read from {path}: {error}') from None
    if not value:
        raise SecretMissing(f'{name} exists at {path} but is empty')
    return Secret(name, value)


def store(name: str, value: str) -> Path:
    """Write a credential into its one source, readable only by this account.

    Atomic on purpose. The first version truncated the file and then wrote, so a write that
    failed part way -- a full disk, a killed process -- left an **empty** file where the working
    credential had been: a rotation that fails would have destroyed the thing it was replacing,
    which is the one outcome a rotation must never have. It now writes a restricted temporary
    file beside the target and renames it, so the previous value survives any failure before the
    rename and no reader ever sees a half-written one.

    It is deliberately the only writer: a rotation updates the source and consumers reload from
    it -- no code edit, no image rebuild, no case change.
    """
    path = path_of(name)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.exists() and not os.access(path, os.W_OK):
        raise PermissionError(f'{path} is not writable by this account; a rotation cannot '
                              'replace it, and the existing credential is left untouched')
    import tempfile
    # Create restricted, then write: never a window where the value is world-readable.
    descriptor, name = tempfile.mkstemp(prefix=f'.{path.name}.', dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return path


def audit() -> list[dict[str, object]]:
    """What is in the source, how it is protected and when it last changed. No values."""
    import stat
    folder = directory()
    rows = []
    if not folder.is_dir():
        return rows
    for path in sorted(folder.iterdir()):
        if not path.is_file() or not NAME.match(path.name):
            continue
        info = path.stat()
        mode = stat.S_IMODE(info.st_mode)
        rows.append({'name': path.name, 'reference': reference(path.name),
                     'mode': oct(mode), 'restricted': mode & 0o077 == 0,
                     'bytes': info.st_size, 'changed_at': int(info.st_mtime)})
    return rows


# ---------------------------------------------------------------------------- is this a secret?

#: File names that should never be inside an image, a build context or a record.
SECRET_SHAPED_NAME = re.compile(
    r'(^|[._-])(secret|secrets|credential|credentials|password|passwd|token|apikey|api_key)([._-]|$)'
    r'|(^|[._/-])id_(rsa|dsa|ecdsa|ed25519)$'
    r'|\.(pem|p12|pfx|jks|keystore)$'
    r'|(^|[._-])key\.json$'
    r'|^\.env(\..+)?$', re.IGNORECASE)

#: Formats distinctive enough to recognise on sight. Deliberately **not** a generic
#: "long hex string" rule: this repository is full of sha256 artifact identities, plan ids and
#: content digests, so such a rule would report thousands of false positives and be switched off
#: within a day. What is here either carries an issuer's own prefix or is a container format.
SECRET_SHAPED_CONTENT = (
    ('private key block', re.compile(r'-----BEGIN [A-Z ]*PRIVATE KEY-----')),
    ('aws access key id', re.compile(r'\b(?:AKIA|ASIA)[0-9A-Z]{16}\b')),
    ('github token', re.compile(r'\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36,}\b|\bgithub_pat_[A-Za-z0-9_]{22,}\b')),
    ('slack token', re.compile(r'\bxox[abposr]-[A-Za-z0-9-]{10,}\b')),
    ('hugging face token', re.compile(r'\bhf_[A-Za-z0-9]{30,}\b')),
    ('openai key', re.compile(r'\bsk-[A-Za-z0-9]{20,}\b')),
    ('password in a url', re.compile(r'\b[a-z][a-z0-9+.-]*://[^\s:/@]+:[^\s:/@]+@')),
)


#: A file of source code or documentation is not a credential, whatever it is called. The
#: content rules are what catch a key embedded in source. Judging by name alone made the
#: worker-image guard fail on A16-10's own module, `secrets_source.py`, the first time an
#: image carried it -- a false positive that would have been read as a leak.
SOURCE_SUFFIXES = frozenset({'.py', '.pyi', '.ts', '.tsx', '.js', '.jsx', '.md', '.rst'})


def secret_shaped_name(path) -> bool:
    """Would a file with this name be a credential? Judged on the name alone.

    Data extensions stay in scope: `my_secret.txt` and `openweather_key.json` are exactly
    what this must catch, so only code and prose are exempt.
    """
    name = Path(str(path)).name
    if name == '.env.example':
        return False                                            # the documented template
    if Path(name).suffix.lower() in SOURCE_SUFFIXES:
        return False
    return bool(SECRET_SHAPED_NAME.search(name))


def known_values() -> list[tuple[str, str]]:
    """``(label, value)`` for every credential this installation actually holds.

    The honest way to ask "does this output leak a secret?" is to compare against the real
    values rather than guess at their shape. Nothing here is logged or returned to a caller that
    did not already have access to the files.
    """
    found = []
    folder = directory()
    if folder.is_dir():
        for path in sorted(folder.iterdir()):
            if path.is_file() and NAME.match(path.name):
                try:
                    value = path.read_text(encoding='utf-8').strip()
                except OSError:
                    continue
                if len(value) >= 8:
                    found.append((path.name, value))
    for label, extra in (('session_secret_legacy', _legacy_secret()),):
        if extra and len(extra) >= 8:
            found.append((label, extra))
    return found


def _legacy_secret() -> str | None:
    try:
        from eforecast.config.service import workspace_root
        path = workspace_root() / '.secret'
        return path.read_text(encoding='utf-8').strip() if path.is_file() else None
    except Exception:                                           # noqa: BLE001
        return None


def leaks(text, *, values=None) -> list[str]:
    """The credentials and credential-shaped values present in ``text``.

    Returns labels, never the values themselves, so a failure message can name what leaked
    without repeating it.
    """
    body = text if isinstance(text, str) else str(text)
    found = []
    for label, value in (values if values is not None else known_values()):
        if value and value in body:
            found.append(f'{label} (exact value)')
    for label, pattern in SECRET_SHAPED_CONTENT:
        if pattern.search(body):
            found.append(label)
    return sorted(set(found))


def redact(text, *, values=None) -> str:
    """``text`` with every known credential replaced. For diagnostics that must still be useful."""
    body = text if isinstance(text, str) else str(text)
    for _, value in (values if values is not None else known_values()):
        if value and len(value) >= 8:
            body = body.replace(value, '<redacted>')
    for _, pattern in SECRET_SHAPED_CONTENT:
        body = pattern.sub('<redacted>', body)
    return body


# ------------------------------------------------------------------- command line

def main(argv=None) -> int:
    """Store and inspect credentials from one place.

        python -m eforecast.config.secrets_source set nixtla_api_key
        python -m eforecast.config.secrets_source list
        python -m eforecast.config.secrets_source where nixtla_api_key

    ``set`` reads the value from a prompt, or from standard input when one is piped. It is
    never taken from the command line: an argument would be recorded in shell history, in
    the process list and in any terminal capture, which is three more places than this
    module exists to allow.
    """
    import argparse
    import getpass
    import sys

    parser = argparse.ArgumentParser(prog='eforecast.config.secrets_source',
                                     description='the one place a credential is kept')
    commands = parser.add_subparsers(dest='command', required=True)
    store_it = commands.add_parser('set', help='store or replace one credential')
    store_it.add_argument('name')
    commands.add_parser('list', help='what is stored, with no values')
    locate = commands.add_parser('where', help='the file one credential would be read from')
    locate.add_argument('name')
    arguments = parser.parse_args(argv)

    # The name is checked before anything is read, so a typo never costs the person a
    # credential they have already typed -- and an environment variable spelling is the
    # likely typo, because the doctor names both and only one of them belongs here.
    name = getattr(arguments, 'name', None)
    if name is not None and not NAME.match(name):
        print(f'{name!r} is not a secret name: use lowercase letters, digits and '
              f'underscores.', file=sys.stderr)
        if NAME.match(name.lower()):
            print(f'{name.lower()!r} is probably what you want. The uppercase spelling is '
                  f'the environment variable the model reads; the secret itself is stored '
                  f'under the lowercase name.', file=sys.stderr)
        return 2

    if arguments.command == 'list':
        rows = audit()
        if not rows:
            print(f'nothing is stored in {directory()}')
            return 0
        width = max(len(str(row.get('name', ''))) for row in rows)
        for row in rows:
            detail = ', '.join(f'{key}={row[key]}' for key in sorted(row) if key != 'name')
            print(f'{str(row.get("name", "")):{width}}  {detail}')
        return 0

    if arguments.command == 'where':
        print(path_of(arguments.name))
        return 0

    if sys.stdin.isatty():
        value = getpass.getpass(f'value for {arguments.name} (not echoed): ')
        again = getpass.getpass('again: ')
        if value != again:
            print('the two entries differ; nothing was written', file=sys.stderr)
            return 2
    else:
        value = sys.stdin.read()
    value = value.strip()
    if not value:
        print('an empty value is not a credential; nothing was written', file=sys.stderr)
        return 2
    path = store(arguments.name, value)
    # the value is never echoed back, not even truncated
    print(f'{arguments.name} stored in {path} (mode 0600)')
    for label in leaks(str(path)):
        print(f'warning: the path itself looks like {label}', file=sys.stderr)
    return 0


if __name__ == '__main__':
    import sys as _sys
    _sys.exit(main())
