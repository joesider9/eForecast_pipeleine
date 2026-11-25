# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""The installation's health, taken once and read instantly -- A14/§4.1 and §4.2.

`bin/eforecast doctor` is the authority and stays read-only. It also takes seconds: it inspects
images, scrapes a port, walks the filesystem and asks the store. Two callers cannot pay that:

* the **web UI**, which needs an answer while a colleague is looking at the page. Until now the
  installation's health had no surface outside a terminal, and of the findings on this machine
  34 of 35 were addressed to the maintainer -- so a colleague meeting a refusal had no way to
  see why;
* a **run submission**, which should record what the installation looked like when the run was
  accepted. That stamp cannot be taken inside the submission transaction: it holds the case
  claim, and a multi-second health check there would serialize every other submission behind
  it.

So a reading is taken once, written here, and read by both. A reader never waits for a refresh
somebody else is doing, and a submission never starts one: a stale answer labelled stale is
more useful to both than a slow one, and far more useful than none.
"""
from __future__ import annotations

import contextlib
import fcntl
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

MAX_AGE = 120.0          # how old a reading may be before a reader refreshes it
LOCK_STALE = 120.0       # a refresh that has not finished in this long is abandoned
FINDINGS = 48            # a bounded record: the doctor has ~35 checks today
TEXT = 500               # per sentence
FIELDS = ('check', 'field', 'severity', 'checked', 'finding', 'consequence', 'remedy', 'who')
VERSION = 1


def _slug(case):
    kept = ''.join(character if character.isalnum() or character in '._-' else '-'
                   for character in str(case))[:64]
    return kept or 'case'


def run_state(path=None, *, case=None) -> Path:
    """Where this installation keeps state the service and the commands share.

    Keyed off `EFORECAST_WORKSPACE`, exactly as `image_runtime.witness_file()` is, so a
    reading taken in a terminal lands where the service reads it -- and the reverse. Writing
    to the wrong one of those two files is the A13-02 trap, and it is worth only making once.

    One file per scope. A reading that also checked a case answers a different question from
    the installation-wide one, so sharing a file would make two callers asking about different
    scopes overwrite each other's reading and re-run the doctor on every single request.
    """
    base = Path(path) if path is not None else None
    if base is None:
        workspace = os.environ.get('EFORECAST_WORKSPACE')
        root = (Path(workspace) / '.run-state' if workspace
                else Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache')) / 'eforecast')
        base = root / 'health.json'
    return base if case is None else base.with_name(f'{base.stem}-{_slug(case)}{base.suffix}')


def _bounded(value):
    from eforecast.config import secrets_source
    if isinstance(value, str):
        return secrets_source.redact(value[:TEXT])
    return value


def record_of(report, *, case=None, duration=None, now=None):
    """A doctor report as the bounded, redacted reading this module stores."""
    findings = report.get('findings') or []
    kept = [{key: _bounded(item.get(key)) for key in FIELDS if item.get(key) is not None}
            for item in findings[:FINDINGS]]
    return {'version': VERSION, 'checked_at': float(now if now is not None else time.time()),
            'verdict': report.get('verdict'), 'counts': report.get('counts') or {},
            'case': case, 'findings': kept, 'examined': len(findings),
            'duration_s': None if duration is None else round(float(duration), 2)}


def snapshot(*, case=None, timeout=3.0, path=None, now=None, reporter=None) -> dict[str, Any]:
    """Take a reading and write it where the UI and new runs can read it."""
    from eforecast.config import doctor
    started = time.monotonic()
    report = (reporter or doctor.report)(case=case, timeout=timeout)
    record = record_of(report, case=case, duration=time.monotonic() - started, now=now)
    write(record, path=path, case=case)
    return record


def write(record, *, path=None, case=None) -> Path:
    """Atomically, because a reader must never see half a reading."""
    target = run_state(path, case=case)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=str(target.parent), prefix='.health-')
    try:
        with os.fdopen(handle, 'w') as stream:
            json.dump(record, stream, indent=1, default=str)
        os.replace(temporary, target)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temporary)
        raise
    return target


def latest(*, path=None, case=None, now=None, max_age=MAX_AGE) -> dict[str, Any] | None:
    """The reading on disk, with its age, or None when none was ever taken."""
    target = run_state(path, case=case)
    try:
        record = json.loads(target.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(record, dict) or record.get('version') != VERSION:
        return None
    age = max(0.0, float(now if now is not None else time.time()) - float(record.get('checked_at') or 0))
    return {**record, 'age_s': round(age, 1), 'stale': age > max_age, 'source': str(target)}


@contextlib.contextmanager
def _refreshing(path=None, case=None):
    """A non-blocking claim on the refresh. Yields False when somebody else holds it."""
    lock = run_state(path, case=case).with_suffix('.lock')
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open('a') as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            # Abandon a claim nothing is finishing, so one killed refresh cannot freeze the
            # reading for everyone; otherwise leave the holder to it and read what is there.
            try:
                held = time.time() - lock.stat().st_mtime
            except OSError:
                held = 0.0
            yield held > LOCK_STALE
            return
        try:
            os.utime(lock, None)
            yield True
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def current(*, case=None, max_age=MAX_AGE, timeout=3.0, path=None, force=False,
            now=None, reporter=None) -> dict[str, Any]:
    """The freshest reading this caller can get without waiting on anybody else.

    Fresh enough -- or being refreshed by someone else -- means the reading on disk. Otherwise
    this caller takes one. A refresh that fails returns the old reading, labelled stale, with
    the reason: a health page that goes blank when the health check breaks is the one outcome
    worse than a stale number.
    """
    existing = latest(path=path, case=case, now=now, max_age=max_age)
    if existing is not None and not force and not existing['stale']:
        return existing
    with _refreshing(path, case) as mine:
        if not mine:
            return existing or {'version': VERSION, 'verdict': None, 'findings': [], 'counts': {},
                                'checked_at': None, 'age_s': None, 'stale': True, 'examined': 0,
                                'case': case, 'unavailable': 'a reading is being taken now'}
        try:
            return {**snapshot(case=case, timeout=timeout, path=path, now=now, reporter=reporter),
                    'age_s': 0.0, 'stale': False}
        except Exception as error:                                          # noqa: BLE001
            reason = f'the health check could not run ({type(error).__name__}: {error})'
            if existing is not None:
                return {**existing, 'unavailable': _bounded(reason)}
            return {'version': VERSION, 'verdict': None, 'findings': [], 'counts': {},
                    'checked_at': None, 'age_s': None, 'stale': True, 'examined': 0,
                    'case': case, 'unavailable': _bounded(reason)}


def summary(*, path=None, now=None, max_age=MAX_AGE) -> dict[str, Any]:
    # Always the installation-wide reading: what a run records is about the machine that
    # accepted it, not about whichever case somebody happened to look at last.
    """What a run records about the installation that accepted it.

    Check names and a verdict, never the sentences: this lives in every run's metadata, so it
    has to stay small, and the sentences are in the reading itself. `absent` and `age_s` are
    part of the answer -- "nobody had looked" and "it was fine" are different facts, and a run
    that failed an hour after a two-day-old reading should not look explained.
    """
    record = latest(path=path, now=now, max_age=max_age)
    if record is None:
        return {'verdict': None, 'absent': True}
    needing = {}
    for item in record['findings']:
        if item.get('severity') and item['severity'] != 'ok':
            needing.setdefault(item['severity'], []).append(item.get('check'))
    return {'verdict': record.get('verdict'), 'checked_at': record.get('checked_at'),
            'age_s': record.get('age_s'), 'stale': record.get('stale'),
            'checks': {severity: sorted(set(names))[:FINDINGS]
                       for severity, names in sorted(needing.items())}}


def for_user(record, *, admin: bool) -> dict[str, Any]:
    """The same reading, as much of it as this caller is entitled to see.

    A maintainer sees every finding. Everyone else sees the verdict, the counts and the
    findings addressed to them, plus how many are the maintainer's -- enough to know whether
    the installation is the problem and whether it is theirs to fix, without publishing this
    machine's paths, ports and image ids to every member of every tenant.
    """
    if admin:
        return {**record, 'scope': 'maintainer'}
    mine = [item for item in record.get('findings') or () if item.get('who') == 'colleague']
    others = [item for item in record.get('findings') or ()
              if item.get('who') != 'colleague' and item.get('severity') != 'ok']
    return {**{key: value for key, value in record.items() if key not in {'findings', 'source'}},
            'findings': mine, 'scope': 'colleague', 'maintainer_findings': len(others)}
