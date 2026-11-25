# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Trusted runtime metadata for an exact immutable worker image.

A profile records the environment it was qualified in. When the work will run in a
container, the host cannot answer "does this profile match?" by inspecting itself: the
container has its own Python, libraries and code. Checking the host instead produces
two wrong answers -- a host-qualified profile passes here and then fails inside, and an
image-qualified profile fails here even though it would match inside.

So the image states its own runtime once, by running the very same
``profile_runtime.hardware`` inside itself, and that statement is recorded against the
image's immutable id. The host then validates a profile against the recorded witness for
the image it is about to launch, while A08 continues to check physical capacity
separately. The worker still rechecks its own runtime before loading a target, so a
witness that no longer describes the image cannot activate anything.

Recording runs a container and is therefore an explicit operator action, never something
a run does implicitly.
"""
import json
import os
from pathlib import Path
import subprocess
import time

TRIAL_ENTRYPOINT = 'eforecast.training.command_line.trial_runner:main'
# Everything an image owns. The rest of the hardware key -- the CPU and the physical
# GPU -- belongs to the host that will run the container, not to the image.
IMAGE_FIELDS = ('python', 'backend', 'libraries', 'code', 'image')


def witness_file():
    override = os.environ.get('EFORECAST_IMAGE_WITNESS_FILE')
    if override:
        return Path(override)
    workspace = os.environ.get('EFORECAST_WORKSPACE')
    base = Path(workspace) / '.run-state' if workspace else Path(
        os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache')) / 'eforecast'
    return base / 'image-runtime.json'


def image_id(reference):
    """The immutable id of an image reference, or None when it cannot be read."""
    try:
        out = subprocess.run(['docker', 'image', 'inspect', '--format', '{{.Id}}', reference],
                             check=True, capture_output=True, text=True, timeout=30)
    except Exception:                                           # noqa: BLE001
        return None
    return out.stdout.strip() or None


def venue(entrypoint, method=None):
    """The image a piece of work will actually run in, or None for this host.

    Trials run in a container exactly when the orchestrator was configured with a worker
    image; every other target runs on this host in the ordinary supervised way.

    ``method`` names a model family when the caller has one, and a family routed to its own
    model image (A16-03) is a different venue with a different runtime -- so a profile
    approved for one must never match the other. Routing is consulted without its coherence
    check: whether the image is stale is decided at dispatch, and asking here would turn a
    venue question into a release question.
    """
    image = os.environ.get('EFORECAST_WORKER_IMAGE')
    if image and entrypoint.split(':', 1)[0] == TRIAL_ENTRYPOINT.split(':', 1)[0]:
        return image
    if method:
        try:
            from eforecast import runtime_routing
            decision = runtime_routing.route(method, require_coherent=False)
        except Exception:                                           # noqa: BLE001
            return None            # the venue cannot be established; the caller's own checks apply
        if decision['where'] == runtime_routing.IMAGE:
            return decision['reference']
    return None


def emit(entrypoint=TRIAL_ENTRYPOINT, device_uuid=None):
    """The runtime statement, produced inside the image it describes."""
    from eforecast.profile_runtime import hardware, source
    return {'version': 1, 'entrypoint': entrypoint,
            'implementation': source(entrypoint),
            'hardware': hardware(entrypoint, device_uuid)}


def load(reference, entrypoint=TRIAL_ENTRYPOINT, *, path=None):
    """The recorded witness for the reference's current id, or None.

    A witness is bound to an immutable id. If the tag now points at a different image,
    or nothing was recorded, this returns None and the caller must not activate.
    """
    identity = image_id(reference)
    if identity is None:
        return None
    store = Path(path) if path else witness_file()
    if not store.is_file():
        return None
    try:
        records = json.loads(store.read_text())
    except (OSError, ValueError):
        return None
    record = (records.get('images') or {}).get(identity, {}).get(entrypoint)
    if not isinstance(record, dict) or record.get('version') != 1:
        return None
    if record.get('hardware', {}).get('image') not in (identity, None):
        return None
    return {**record, 'image_id': identity, 'reference': reference}


def record(reference, entrypoint=TRIAL_ENTRYPOINT, *, path=None, runner=None):
    """Run the image once and record what it says about itself."""
    identity = image_id(reference)
    if identity is None:
        raise RuntimeError(f'cannot read the immutable id of image {reference!r}')
    # The worker image sets its own ENTRYPOINT, so the interpreter is named explicitly
    # and the repository is deliberately not mounted: the image must describe its own code.
    argv = runner or ['docker', 'run', '--rm', '--entrypoint', 'python',
                      '-e', f'EFORECAST_IMAGE_ID={identity}', reference,
                      '-m', 'eforecast.image_runtime', 'emit', '--entrypoint', entrypoint]
    out = subprocess.run(argv, capture_output=True, text=True, timeout=600)
    if out.returncode != 0:
        detail = (out.stderr or out.stdout).strip().splitlines()[-1:] or ['no output']
        raise RuntimeError(
            f'image {reference} cannot state its runtime ({detail[0][:200]}). An image must '
            'carry eforecast.image_runtime to be witnessed; rebuild it from this revision.')
    lines = [line for line in out.stdout.strip().splitlines() if line.startswith('{')]
    if not lines:
        raise RuntimeError('image returned no runtime statement')
    statement = json.loads(lines[-1])
    if statement.get('version') != 1 or statement.get('entrypoint') != entrypoint:
        raise RuntimeError('image returned an unusable runtime statement')
    if statement['hardware'].get('image') != identity:
        raise RuntimeError('image runtime statement does not name the image it ran in')
    store = Path(path) if path else witness_file()
    store.parent.mkdir(parents=True, exist_ok=True)
    try:
        records = json.loads(store.read_text()) if store.is_file() else {}
    except (OSError, ValueError):
        records = {}
    records.setdefault('version', 1)
    images = records.setdefault('images', {})
    images.setdefault(identity, {})[entrypoint] = {**statement, 'reference': reference,
                                                   'recorded_at': time.time()}
    temporary = store.with_suffix(store.suffix + '.tmp')
    temporary.write_text(json.dumps(records, indent=1, sort_keys=True))
    os.replace(temporary, store)
    return {**statement, 'image_id': identity, 'reference': reference}


def expected(witness, host_hardware):
    """What the container will report: the image's own fields over this host's."""
    return {**host_hardware, **{k: witness['hardware'][k] for k in IMAGE_FIELDS}}


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    for name in ('emit', 'record', 'show'):
        child = sub.add_parser(name)
        child.add_argument('--entrypoint', default=TRIAL_ENTRYPOINT)
        if name != 'emit':
            child.add_argument('--image', required=True)
            child.add_argument('--file')
    args = parser.parse_args(argv)
    if args.action == 'emit':
        print(json.dumps(emit(args.entrypoint)))
        return 0
    if args.action == 'record':
        value = record(args.image, args.entrypoint, path=args.file)
        print(json.dumps({'image': value['reference'], 'image_id': value['image_id'],
                          'entrypoint': value['entrypoint'],
                          'python': value['hardware']['python'],
                          'backend': value['hardware']['backend'],
                          'file': str(Path(args.file) if args.file else witness_file())}, indent=1))
        return 0
    value = load(args.image, args.entrypoint, path=args.file)
    print(json.dumps(value, indent=1, sort_keys=True) if value else
          json.dumps({'image': args.image, 'witness': None,
                      'reason': 'no recorded runtime witness for this image id'}, indent=1))
    return 0 if value else 1


if __name__ == '__main__':
    raise SystemExit(main())
