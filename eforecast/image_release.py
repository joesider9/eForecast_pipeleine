# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Is every image in the set built from the code it is supposed to be? -- A16-11.

With one worker image a code change meant one rebuild, and it was still missed once: the
`a12-followup` candidate predated three fixes, and a trial launched from it would have
produced TimesFM backcasts rather than forecasts.
A16-02 replaced that one image with a set. The chance that every member carries the current
code falls with each image added, and the symptom is a wrong number, not an error.

So a release is a manifest of *immutable* identities and content digests, and one command
compares the images on this host against it, per image and per reason. Deliberately:

* **Content, not timestamps or tags.** A matching tag proves nothing; the manifest records
  the image id a reference resolved to, so a moved tag is a difference.
* **Rebuilding bytes and refetching weights are different answers.** A checkpoint revision
  that moved does not mean the source changed, and reporting it as one would send a
  maintainer to rebuild an image that is already correct.
* **A fresh witness cannot cure a content mismatch.** The witness answers "has this exact
  id been qualified"; it says nothing about whether the id was built from today's code, so
  the two are compared independently.
* **Per member, never one red light for the set.** A case needs some of these images; an
  unrelated optional family being stale must not block it.

Nothing here pulls, builds, re-witnesses or loads a model. It reads files, asks Docker what
it already has, and reads the local checkpoint cache.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from functools import lru_cache

from eforecast import image_runtime

VERSION = 1

#: Where captured releases live. In the repository, so a release is versioned with the code
#: it describes and a stored run can still find the release it was accepted with.
RELEASES = Path('docker/releases')

#: Files copied into *both* model images. Editing one marks every consumer stale, which is
#: the dependency this task exists to track.
SHARED_ADAPTER = (
    'eforecast/__init__.py',
    'eforecast/runtime_contract.py',
    'eforecast/image_runtime.py',
    'eforecast/profile_runtime.py',
    'eforecast/profile_policy.py',
    'eforecast/runtime_adapters/__init__.py',
    'eforecast/runtime_adapters/__main__.py',
    'eforecast/runtime_adapters/transport.py',
    'eforecast/runtime_adapters/checkpoints.py',
)

#: Each required image, what builds it, and how it is witnessed. `trees` are whole
#: directories copied into the image; they are digested through the build's own
#: .dockerignore, so a file the build excludes does not report a false difference.
RUNTIMES = {
    'foundation': {
        'reference': 'eforecast-foundation:a16-02',
        'recipe': 'docker/Dockerfile.foundation',
        'lock': 'docker/requirements-foundation.txt',
        'files': SHARED_ADAPTER + ('eforecast/runtime_adapters/foundation.py',),
        'trees': ('eforecast/runtime_adapters',),
        'entrypoint': 'eforecast.runtime_adapters.__main__:run',
    },
    'lag_llama': {
        'reference': 'eforecast-lagllama:a16-02',
        'recipe': 'docker/Dockerfile.lagllama',
        'lock': 'docker/requirements-lagllama.txt',
        'files': SHARED_ADAPTER + ('eforecast/runtime_adapters/lag_llama.py',),
        'trees': ('eforecast/runtime_adapters',),
        'entrypoint': 'eforecast.runtime_adapters.__main__:run',
    },
    'worker': {
        'reference': 'eforecast-worker:latest',
        'recipe': 'docker/Dockerfile.worker',
        'lock': 'docker/requirements-worker.txt',
        'files': ('run_datasets.py', 'run_training.py', 'run_combine.py', 'run_model_script.py'),
        'trees': ('eforecast', 'configuration'),
        'entrypoint': image_runtime.TRIAL_ENTRYPOINT,
    },
    'online': {
        'reference': 'eforecast-online:step02',
        'recipe': 'eforecast_online/Dockerfile',
        'lock': 'eforecast_online/requirements-online.txt',
        'files': ('run_model_online.py', 'run_model_script.py',
                  'configuration/experiments.py', 'configuration/experiment_template.py'),
        'trees': ('eforecast', 'eforecast_online'),
        # ENTRYPOINT is ["python", "-m", "eforecast_online"], whose __main__ calls this
        'entrypoint': 'eforecast_online.cli:main',
    },
}

def witness_candidates():
    """Every file a witness could have been recorded into on this machine.

    ``image_runtime.witness_file()`` keys off ``EFORECAST_WORKSPACE``, so a witness recorded
    in an ordinary terminal lands where the service never looks -- and the reverse. That is
    the A13-02 trap, and the first run of this command walked straight into it: it reported
    `eforecast-worker:latest` as never witnessed while the service's own file had witnessed
    it all along. Reporting "no witness" for one that exists in the other file is a false
    finding, and a false finding here sends a maintainer to rebuild a correct image.
    """
    from eforecast.config import service

    served = image_runtime.witness_file()
    candidates = [served,
                  Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache'))
                  / 'eforecast' / 'image-runtime.json']
    for _, workspace in getattr(service, 'LAB_WORKSPACES', ()):
        candidates.append(Path(workspace) / '.run-state' / 'image-runtime.json')
    seen, unique = set(), []
    for path in candidates:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def witness_elsewhere(identity, entrypoint):
    """A file other than the one this configuration reads that has witnessed this id."""
    served = image_runtime.witness_file()
    for path in witness_candidates():
        if path == served or not path.is_file():
            continue
        try:
            records = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            continue
        entry = (records.get('images') or {}).get(identity) or {}
        if entrypoint in entry:
            return path
    return None


def host_implementation(entrypoint):
    """The digest of the entrypoint module as it stands in this tree, or None.

    A witness records the same digest computed *inside* the image, so comparing the two is
    the one piece of evidence that ties an image's contents to the working tree. Everything
    else here compares the tree against the recorded baseline, which cannot notice an image
    that was built before that baseline and never retagged.

    It covers one file, not the whole tree, so it is a partial check -- see the limits in
    docs/administrator.md section 3.
    """
    from eforecast import profile_runtime
    try:
        return profile_runtime.source(entrypoint)
    except (ValueError, OSError):
        return None             # an entrypoint outside the eforecast package has no digest here


#: Reasons that can only be answered by building image bytes again.
REBUILD = ('recipe', 'lock', 'base', 'source', 'implementation')
#: A weight that moved is not a source change; saying so would send someone to rebuild an
#: image that is already correct.
REFETCH = ('checkpoint',)
#: The id is present and matches, but nothing this configuration reads has qualified it.
#: `witness_elsewhere` is the A13-02 case: a witness exists, in a file nothing here reads.
REQUALIFY = ('witness', 'witness_elsewhere')

COHERENT, STALE, MISSING, UNKNOWN = 'coherent', 'stale', 'missing', 'unknown'

#: Asking Docker what images it holds costs a subprocess. The doctor may spend one; readiness
#: answers a web request and deliberately does not (its existing checks ask Docker only when
#: EFORECAST_WORKER_IMAGE says to). So the image id and the witness can be left unexamined --
#: recorded as NOT_CHECKED, never as a match, because unavailable evidence is not agreement.
NOT_CHECKED = 'not checked'
SOURCE_INPUTS = ('recipe', 'lock', 'base', 'source')
IMAGE_INPUTS = ('image', 'witness')

#: Paths that never enter any image and must not be digested.
ALWAYS_SKIP = re.compile(r'(^|/)(__pycache__|\.git|node_modules)(/|$)|\.pyc$')

#: How many differing paths a report names before it summarises the rest.
NAMED_PATHS = 8


class ReleaseError(RuntimeError):
    """The release cannot be read or captured."""


# ----------------------------------------------------------------- content digests

def _sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def file_digest(path, *, root=None):
    """The digest of one file, or None when it is not there."""
    full = Path(root or '.') / path
    try:
        return _sha_bytes(full.read_bytes())
    except (OSError, IsADirectoryError):
        return None


def _compile(pattern):
    """A .dockerignore pattern as a regex. `**` spans separators, `*` stays inside one."""
    out, index = [], 0
    while index < len(pattern):
        if pattern.startswith('**', index):
            out.append('.*')
            index += 2
        elif pattern[index] == '*':
            out.append('[^/]*')
            index += 1
        elif pattern[index] == '?':
            out.append('[^/]')
            index += 1
        else:
            out.append(re.escape(pattern[index]))
            index += 1
    return re.compile('^' + ''.join(out) + '$')


def ignore_rules(recipe, *, root=None):
    """The build's own exclusion rules, in order, from `<dockerfile>.dockerignore`."""
    path = Path(root or '.') / (str(recipe) + '.dockerignore')
    if not path.is_file():
        return []
    rules = []
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        negated = line.startswith('!')
        body = line[1:] if negated else line
        rules.append((negated, _compile(body.rstrip('/'))))
    return rules


def excluded(relative, rules):
    """Docker's semantics: the last rule that matches the path or one of its parents wins."""
    if not rules:
        return False
    parts = relative.split('/')
    candidates = [relative] + ['/'.join(parts[:n]) for n in range(1, len(parts))]
    decision = False
    for negated, regex in rules:
        if any(regex.match(candidate) for candidate in candidates):
            decision = not negated
    return decision


def tree_files(tree, rules, *, root=None):
    """Every file under a copied directory that the build actually includes."""
    base = Path(root or '.')
    found = {}
    for directory, subdirectories, names in os.walk(base / tree):
        subdirectories[:] = [name for name in subdirectories if not ALWAYS_SKIP.search(name)]
        for name in names:
            full = Path(directory) / name
            relative = str(full.relative_to(base))
            if ALWAYS_SKIP.search(relative) or excluded(relative, rules):
                continue
            found[relative] = full
    return found


def source_digests(runtime, *, root=None):
    """Every source path that enters this image, with its digest.

    Keyed by repository path so a difference can be named, not merely counted.
    """
    spec = RUNTIMES[runtime]
    rules = ignore_rules(spec['recipe'], root=root)
    digests = {}
    for path in spec['files']:
        digest = file_digest(path, root=root)
        if digest is not None:
            digests[path] = digest
    for tree in spec['trees']:
        for relative, full in tree_files(tree, rules, root=root).items():
            try:
                digests[relative] = _sha_bytes(full.read_bytes())
            except OSError:
                continue
    return digests


def rollup(digests):
    """One digest over a set of (path, digest) pairs, order-independent."""
    body = '\n'.join(f'{path}\t{digest}' for path, digest in sorted(digests.items()))
    return _sha_bytes(body.encode())


def base_of(recipe, *, root=None):
    """The base this recipe pins, from its own text. No registry is consulted."""
    try:
        text = (Path(root or '.') / recipe).read_text(encoding='utf-8')
    except OSError:
        return {'reference': None, 'digest': None, 'pinned': False}
    found = re.search(r'^FROM\s+(\S+)', text, re.MULTILINE)
    if not found:
        return {'reference': None, 'digest': None, 'pinned': False}
    reference = found.group(1)
    digest = reference.split('@', 1)[1] if '@' in reference else None
    return {'reference': reference, 'digest': digest, 'pinned': digest is not None}


# ----------------------------------------------------------------- checkpoints

def cached_checkpoints(cache=None):
    """Checkpoint repositories in the local cache, with the revision each ref points at.

    Read from the cache directory, not from Hugging Face: this must work with no network.
    """
    from eforecast import runtime_launcher

    if cache is None:
        base = Path(os.environ.get(
            'EFORECAST_RUNTIME_CACHE', Path.home() / '.cache' / 'eforecast' / 'runtimes'))
        cache = base / 'checkpoints' / 'hub'
    cache = Path(cache)
    found = {}
    if not cache.is_dir():
        return found
    for entry in sorted(cache.glob('models--*')):
        repo = entry.name[len('models--'):].replace('--', '/')
        refs = entry / 'refs'
        revision = None
        if refs.is_dir():
            for ref in sorted(refs.iterdir()):
                try:
                    revision = ref.read_text(encoding='utf-8').strip()
                    break
                except OSError:
                    continue
        if revision is None:
            snapshots = sorted((entry / 'snapshots').glob('*')) if (entry / 'snapshots').is_dir() else []
            revision = snapshots[0].name if snapshots else None
        found[repo] = revision
    return found


# ----------------------------------------------------------------- capture and check

def image_sources(identity, paths, workdir):
    """Read source hashes from an immutable local image, never model code or credentials.

    No network, mounts or model imports; bounded CPU, memory and wall time. Caching
    by immutable id avoids starting another probe on every preview of that image.
    Failure is unknown evidence, never an implicit match.
    """
    try:
        return _image_sources(identity, paths, workdir)
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


@lru_cache(maxsize=16)
def _image_sources(identity, paths, workdir):
    program = ('import json,hashlib,pathlib,sys; '
               'paths=json.load(sys.stdin); '
               'print(json.dumps({p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() '
               'if pathlib.Path(p).is_file() else None for p in paths}))')
    result = subprocess.run(
        ['docker', 'run', '--rm', '-i', '--network', 'none', '--read-only',
         '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges', '--cpus', '1',
         '--memory', '128m', '--pids-limit', '32', '--workdir', workdir,
         '--entrypoint', 'python', identity, '-c', program],
        input=json.dumps(paths), capture_output=True, text=True, timeout=20, check=True)
    values = json.loads(result.stdout)
    if (not isinstance(values, dict) or set(values) != set(paths)
            or any(v is not None and (not isinstance(v, str) or not re.fullmatch('[0-9a-f]{64}', v))
                   for v in values.values())):
        raise ValueError('invalid image source observation')
    return values


def observe(runtime, *, root=None, witness_path=None, ask_docker=True, reference=None):
    """What this host can say about one runtime right now, with no network.

    ``ask_docker`` false skips the image id and the witness, so the whole thing runs in
    process. What is skipped is reported as such rather than as a match.

    ``reference`` names a specific image instead of the runtime's usual tag. Without it there is
    no way to prepare a release without first moving the tag production runs from, which is a
    bad reason to retag anything: a release should be able to describe images that exist under a
    dated tag and leave the floating one where it is.
    """
    spec = RUNTIMES[runtime]
    reference = reference or spec['reference']
    digests = source_digests(runtime, root=root)
    if not ask_docker:
        identity = witness = NOT_CHECKED
    else:
        identity = image_runtime.image_id(reference)
        witness = (image_runtime.load(reference, entrypoint=spec['entrypoint'], path=witness_path)
                   if identity is not None else None)
    return {
        'reference': reference,
        'image': identity,
        'recipe': {'path': spec['recipe'], 'sha256': file_digest(spec['recipe'], root=root)},
        'lock': {'path': spec['lock'], 'sha256': file_digest(spec['lock'], root=root)},
        'base': base_of(spec['recipe'], root=root),
        'source': {'digest': rollup(digests), 'files': digests},
        'image_source': (NOT_CHECKED if not ask_docker else
                         image_sources(identity, tuple(sorted(digests)),
                                       '/app' if runtime == 'online' else '/opt/eforecast')
                         if identity else None),
        'entrypoint': spec['entrypoint'],
        'implementation': {
            'host': host_implementation(spec['entrypoint']),
            'image': (NOT_CHECKED if witness is NOT_CHECKED
                      else (witness or {}).get('implementation')),
        },
        'witness': (NOT_CHECKED if witness is NOT_CHECKED else
                    None if witness is None else
                    {'recorded': True, 'image': witness.get('image_id')}),
    }


def unqualified(observation):
    """Intrinsic defects in a member, as opposed to drift from a recorded baseline.

    These matter at capture time because capture records *what is*: recording a member that
    has no witness, or whose recipe does not pin its base, publishes that defect as the
    baseline and every later check then calls it coherent. Drift is relative; these are not.
    """
    problems = []
    if observation['image'] in (None, NOT_CHECKED):
        problems.append(f'{observation["reference"]} is not present on this host')
    elif not observation['witness']:
        problems.append(f'{observation["reference"]} has no witness for '
                        f'{observation["image"][:19]}... under {observation["entrypoint"]}')
    if not (observation['base'] or {}).get('pinned'):
        problems.append(f'{observation["recipe"]["path"]} does not pin its base by digest')
    if observation.get('image_source') in (None, NOT_CHECKED):
        problems.append('image source contents could not be verified')
    elif observation['image_source'] != observation['source']['files']:
        problems.append('image source contents differ from this release tree; rebuild the image')
    return problems


def capture(release, *, root=None, runtimes=None, witness_path=None, checkpoints=None,
            references=None,
            require=()):
    """A release manifest describing the current state. A maintainer action, never implicit.

    `require` names the members that must qualify for this to be publishable. A member that
    is recorded but not required is kept with its defects listed, so the manifest is honest
    about which of its members were fit to depend on.
    """
    chosen = sorted(runtimes or RUNTIMES)
    unknown = [name for name in chosen if name not in RUNTIMES] + [
        name for name in require if name not in chosen]
    if unknown:
        raise ReleaseError(f'not a known or captured runtime: {", ".join(sorted(set(unknown)))}')
    observed = {name: observe(name, root=root, witness_path=witness_path,
                              reference=(references or {}).get(name))
                for name in chosen}
    defects = {name: unqualified(entry) for name, entry in observed.items()}
    refused = {name: defects[name] for name in require if defects[name]}
    if refused:
        detail = '; '.join(f'{name}: {", ".join(items)}' for name, items in sorted(refused.items()))
        raise ReleaseError(
            f'release {release!r} was not written: a required member does not qualify -- '
            f'{detail}. Fix the member, or drop it from --require and depend on it knowingly.')
    return {
        'version': VERSION,
        'release': release,
        'required': sorted(require),
        'runtimes': observed,
        'unqualified': {name: items for name, items in sorted(defects.items()) if items},
        'checkpoints': cached_checkpoints(checkpoints),
    }


def write(manifest, *, directory=None):
    directory = Path(directory or RELEASES)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'{manifest["release"]}.json'
    temporary = path.with_suffix('.json.partial')
    temporary.write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding='utf-8')
    os.replace(temporary, path)
    return path


def read(release, *, directory=None):
    path = Path(directory or RELEASES) / f'{release}.json'
    try:
        manifest = json.loads(path.read_text(encoding='utf-8'))
    except FileNotFoundError:
        raise ReleaseError(
            f'no release named {release!r} in {Path(directory or RELEASES)}. Capture one: '
            f'python -m eforecast.image_release capture --release {release}') from None
    except (OSError, ValueError) as error:
        raise ReleaseError(f'release {release!r} cannot be read: {error}') from None
    if manifest.get('version') != VERSION:
        raise ReleaseError(f'release {release!r} is version {manifest.get("version")}, not {VERSION}')
    return manifest


def _named(paths):
    listed = sorted(paths)[:NAMED_PATHS]
    rest = len(paths) - len(listed)
    return listed + ([f'and {rest} more'] if rest > 0 else [])


def differences(recorded, current):
    """Every differing input between a recorded runtime and what is here now.

    Each entry is (reason, detail). The reasons are deliberately separate so the report can
    say *what* to do: rebuild, refetch a weight, or record a witness.
    """
    found = []
    if current['image'] is NOT_CHECKED:
        pass                    # neither compared nor assumed equal; see NOT_CHECKED
    elif current['image'] is None:
        found.append(('image', f'{current["reference"]} is not present on this host'))
    elif recorded.get('image') != current['image']:
        found.append(('image', f'{current["reference"]} now resolves to '
                               f'{(current["image"] or "nothing")[:19]}..., recorded as '
                               f'{(recorded.get("image") or "nothing")[:19]}... -- the tag moved'))
    for key in ('recipe', 'lock'):
        was, now = (recorded.get(key) or {}).get('sha256'), (current[key] or {}).get('sha256')
        if now is None:
            found.append((key, f'{current[key]["path"]} is missing'))
        elif was != now:
            found.append((key, f'{current[key]["path"]} changed'))
    was, now = recorded.get('base') or {}, current['base']
    if not now.get('pinned'):
        found.append(('base', f'the recipe does not pin its base by digest '
                              f'({now.get("reference") or "no FROM line"}); a moved tag would '
                              f'change the build without changing this file'))
    elif was.get('digest') != now.get('digest'):
        found.append(('base', f'the pinned base changed from '
                              f'{(was.get("digest") or "unpinned")[:19]}... to '
                              f'{now["digest"][:19]}...'))
    before = (recorded.get('source') or {}).get('files') or {}
    after = current['source']['files']
    changed = [p for p in set(before) & set(after) if before[p] != after[p]]
    added = sorted(set(after) - set(before))
    removed = sorted(set(before) - set(after))
    if changed:
        found.append(('source', f'{len(changed)} file(s) changed: {", ".join(_named(changed))}'))
    if added:
        found.append(('source', f'{len(added)} file(s) added: {", ".join(_named(added))}'))
    if removed:
        found.append(('source', f'{len(removed)} file(s) no longer present: {", ".join(_named(removed))}'))
    # The entrypoint module as the image reports it, against this tree. This is the only
    # check that looks inside the image rather than at the inputs beside it.
    inside = (current.get('implementation') or {})
    image_source = current.get('image_source')
    if current['image'] not in (None, NOT_CHECKED):
        if image_source is None:
            found.append(('implementation', 'image source contents could not be verified'))
        elif image_source != current['source']['files']:
            changed = sorted(path for path, value in current['source']['files'].items()
                             if image_source.get(path) != value)
            found.append(('implementation', 'image has different or missing source: ' + ', '.join(_named(changed))))
    if (inside.get('image') not in (None, NOT_CHECKED) and inside.get('host')
            and inside['image'] != inside['host']):
        found.append(('implementation',
                      f'the image reports {current["entrypoint"]} as '
                      f'{inside["image"][:16]}..., this tree has {inside["host"][:16]}... -- '
                      f'the image was built from different code'))

    # Checked independently of everything above: a witness says this exact id was qualified,
    # not that the id was built from today's code. Recording one cannot cure a mismatch.
    if (current['image'] not in (None, NOT_CHECKED)
            and current['witness'] is None):
        other = witness_elsewhere(current['image'], current['entrypoint'])
        if other is not None:
            found.append(('witness_elsewhere',
                          f'{current["image"][:19]}... is witnessed in {other}, but this '
                          f'configuration reads {image_runtime.witness_file()} -- the '
                          f'recording went to a file nothing here looks at'))
        else:
            found.append(('witness', f'no runtime witness is recorded for '
                                     f'{current["image"][:19]}... under {current["entrypoint"]}'))
    return found


def check(release, *, root=None, runtimes=None, directory=None, witness_path=None,
          checkpoints=None, ask_docker=True, references=None):
    """Compare the image set against a release, per image and per reason.

    ``ask_docker`` false compares only the build inputs that are files, which needs no
    subprocess. Each runtime then reports which inputs were examined, so a caller cannot
    mistake "the source matches" for "this image is the right one".
    """
    manifest = read(release, directory=directory)
    recorded = manifest.get('runtimes') or {}
    chosen = sorted(runtimes) if runtimes else sorted(recorded)
    report = {'version': VERSION, 'release': release, 'runtimes': {}}

    for name in chosen:
        if name not in RUNTIMES:
            report['runtimes'][name] = {'status': UNKNOWN, 'reasons': [
                {'reason': 'unknown', 'detail': f'{name} is not a runtime this build knows'}]}
            continue
        if name not in recorded:
            report['runtimes'][name] = {'status': UNKNOWN, 'reasons': [
                {'reason': 'unrecorded',
                 'detail': f'release {release!r} does not describe {name}; it cannot be '
                           f'declared coherent against it'}]}
            continue
        # A release describes particular images, so it is checked against the reference it
        # recorded -- not against whatever the floating tag points at today. An explicit
        # `references` entry still wins, for asking "would *this* image satisfy that release".
        reference = (references or {}).get(name) or recorded[name].get('reference')
        current = observe(name, root=root, witness_path=witness_path, ask_docker=ask_docker,
                          reference=reference)
        found = differences(recorded[name], current)
        absent = any(reason == 'image' and 'not present' in detail for reason, detail in found)
        report['runtimes'][name] = {
            'status': COHERENT if not found else (MISSING if absent else STALE),
            'image': current['image'],
            'reference': current['reference'],
            'reasons': [{'reason': reason, 'detail': detail} for reason, detail in found],
            'rebuild': sorted({r for r, _ in found} & set(REBUILD)),
            'refetch': sorted({r for r, _ in found} & set(REFETCH)),
            'requalify': sorted({r for r, _ in found} & set(REQUALIFY)),
            'checked': list(SOURCE_INPUTS if not ask_docker
                            else SOURCE_INPUTS + IMAGE_INPUTS),
        }

    # Checkpoints are compared once, for the set: they are stored outside every image, and a
    # revision that moved is a refetch, never a rebuild.
    here = cached_checkpoints(checkpoints)
    was = manifest.get('checkpoints') or {}
    moved = {repo: (was[repo], here.get(repo)) for repo in was if was[repo] != here.get(repo)}
    report['checkpoints'] = {
        'recorded': was, 'present': here,
        'moved': [{'repository': repo, 'recorded': before, 'present': now,
                   'detail': (f'{repo} is not in the local cache' if now is None else
                              f'{repo} is at {str(now)[:12]}, recorded at {str(before)[:12]}')}
                  for repo, (before, now) in sorted(moved.items())],
    }
    return report


def activatable(report, runtimes):
    """Whether a run needing these runtimes may start, and why not when it may not.

    Required members are the case's; an unrelated stale family is not this run's problem.
    """
    blocking = []
    for name in sorted(set(runtimes)):
        entry = (report.get('runtimes') or {}).get(name)
        if entry is None:
            blocking.append(f'{name}: not checked against release {report.get("release")!r}')
        elif entry['status'] != COHERENT:
            reasons = ', '.join(sorted({item['reason'] for item in entry['reasons']}))
            blocking.append(f'{name}: {entry["status"]} ({reasons})')
    return (not blocking), blocking


# ----------------------------------------------------------------- command line

ACTION = {'recipe': 'rebuild the image', 'lock': 'rebuild the image',
          'base': 'rebuild the image', 'source': 'rebuild the image',
          'implementation': 'rebuild the image',
          'image': 'build or retag the image', 'checkpoint': 'refetch the checkpoint',
          'witness': 'record a witness',
          'witness_elsewhere': 'record it where this configuration reads',
          'unknown': 'nothing: unrecognised runtime',
          'unrecorded': 'capture a release that describes it'}

BUILD_COMMAND = {
    'foundation': 'docker build -f docker/Dockerfile.foundation -t eforecast-foundation:a16-02 .',
    'lag_llama': 'docker build -f docker/Dockerfile.lagllama -t eforecast-lagllama:a16-02 .',
    'worker': 'docker build -f docker/Dockerfile.worker -t eforecast-worker:latest .',
    'online': 'docker build -f eforecast_online/Dockerfile -t eforecast-online:step02 .',
}


def render(report, *, verbose=False):
    """One block per image, because a single verdict for the set hides the stale member."""
    lines = [f'release {report["release"]}']
    order = {STALE: 0, MISSING: 1, UNKNOWN: 2, COHERENT: 3}
    for name in sorted(report['runtimes'], key=lambda n: (order.get(
            report['runtimes'][n]['status'], 9), n)):
        entry = report['runtimes'][name]
        head = f'  {name:12s} {entry["status"]}'
        if entry['status'] == COHERENT and not verbose:
            lines.append(head)
            continue
        lines.append(head + (f'  ({entry.get("reference")})' if entry.get('reference') else ''))
        for item in entry['reasons']:
            lines.append(f'      - {item["reason"]}: {item["detail"]}')
        if entry.get('rebuild'):
            lines.append(f'      -> {BUILD_COMMAND.get(name, "rebuild the image")}')
        if entry.get('requalify') and not entry.get('rebuild'):
            # only worth saying on its own: after a rebuild the witness follows anyway, and
            # recording one over stale bytes would be the wrong thing to suggest
            workspace = os.environ.get('EFORECAST_WORKSPACE')
            prefix = f'EFORECAST_WORKSPACE={workspace} ' if workspace else ''
            lines.append(f'      -> {prefix}python -m eforecast.image_runtime record --image '
                         f'{entry.get("reference")} --entrypoint {RUNTIMES[name]["entrypoint"]}')
    moved = report.get('checkpoints', {}).get('moved') or []
    if moved:
        lines.append('  checkpoints  moved (a weight, not source: refetch, do not rebuild)')
        for item in moved:
            lines.append(f'      - {item["detail"]}')
    elif verbose:
        lines.append(f'  checkpoints  {len(report.get("checkpoints", {}).get("recorded") or {})} '
                     f'recorded, all present')
    return '\n'.join(lines)


def main(argv=None):
    """`capture` records what is here; `check` compares what is here against a release.

        python -m eforecast.image_release capture --release a16-02
        python -m eforecast.image_release check --release a16-02 [--runtime foundation ...]

    Neither pulls, builds, re-witnesses nor loads a model. Rebuilding stays a maintainer
    action; this command only ever says what differs.
    """
    import argparse
    import sys

    parser = argparse.ArgumentParser(prog='eforecast.image_release',
                                     description='image-set coherence against a release')
    commands = parser.add_subparsers(dest='command', required=True)
    for name, help_text in (('capture', 'record a release manifest from the current state'),
                            ('check', 'compare the image set against a release')):
        sub = commands.add_parser(name, help=help_text)
        sub.add_argument('--release', required=True)
        sub.add_argument('--directory', default=None)
        sub.add_argument('--runtime', action='append', dest='runtimes', default=None,
                         help='limit to these runtimes; repeat the flag')
        sub.add_argument('--json', action='store_true')
        sub.add_argument('--reference', action='append', dest='references', default=None,
                         metavar='RUNTIME=IMAGE',
                         help='use this image for a runtime instead of its usual tag, so a '
                              'release can describe a dated build without moving the tag '
                              'production runs from; repeat the flag')
        if name == 'check':
            sub.add_argument('--verbose', action='store_true')
        else:
            sub.add_argument('--require', action='append', dest='require', default=None,
                             help='members that must qualify or nothing is written; repeat')
    arguments = parser.parse_args(argv)
    references = {}
    for item in arguments.references or ():
        runtime, _, image = item.partition('=')
        if not image or runtime not in RUNTIMES:
            parser.error(f'--reference expects RUNTIME=IMAGE with a known runtime, got {item!r}')
        references[runtime] = image

    try:
        if arguments.command == 'capture':
            manifest = capture(arguments.release, runtimes=arguments.runtimes,
                               require=tuple(arguments.require or ()),
                               references=references or None)
            path = write(manifest, directory=arguments.directory)
            if arguments.json:
                print(json.dumps(manifest, indent=1, sort_keys=True))
            else:
                print(f'captured {len(manifest["runtimes"])} runtime(s) and '
                      f'{len(manifest["checkpoints"])} checkpoint(s) into {path}')
                for name, entry in sorted(manifest['runtimes'].items()):
                    state = (entry['image'] or 'not present on this host')[:19]
                    mark = ('required' if name in manifest['required']
                            else 'recorded with defects' if manifest['unqualified'].get(name)
                            else 'recorded')
                    print(f'  {name:12s} {state:21s} {mark}')
                    for item in manifest['unqualified'].get(name, []):
                        print(f'      ! {item}')
            return 0

        report = check(arguments.release, runtimes=arguments.runtimes,
                       directory=arguments.directory, references=references or None)
    except ReleaseError as error:
        print(f'{error}', file=sys.stderr)
        return 3
    if arguments.json:
        print(json.dumps(report, indent=1, sort_keys=True))
    else:
        print(render(report, verbose=arguments.verbose))
    states = {entry['status'] for entry in report['runtimes'].values()}
    if states - {COHERENT}:
        return 1
    return 0


if __name__ == '__main__':
    import sys as _sys
    _sys.exit(main())


# ----------------------------------------------------------------- selection

def available(directory=None):
    """The releases captured in this installation, newest name last."""
    directory = Path(directory or RELEASES)
    if not directory.is_dir():
        return []
    return sorted(path.stem for path in directory.glob('*.json'))


def selected(directory=None):
    """The release this installation runs against, or None when that is not established.

    Guessing would be worse than saying nothing: declaring an image coherent against a
    release nobody chose is exactly the false assurance this task exists to remove.
    """
    name = os.environ.get('EFORECAST_RELEASE')
    if name:
        return name
    names = available(directory)
    return names[0] if len(names) == 1 else None


def required_for(document):
    """The runtimes a case actually needs, so an unrelated family cannot block it.

    Only the model images: whether the worker or online image is required depends on how the
    run will execute, which is A16-03's dispatch decision rather than the case's content.
    """
    from eforecast.runtime_adapters import adapter_name
    from eforecast import runtime_launcher

    selected_methods = [name for name, on in
                        ((document or {}).get('methods', {}).get('methods_cpu') or {}).items() if on]
    needed = set()
    for method in selected_methods:
        served = runtime_launcher.CASE_METHODS.get(method)
        if served is None:
            continue
        try:
            needed.add(adapter_name(served))
        except Exception:                                               # noqa: BLE001
            continue
    return sorted(needed)
