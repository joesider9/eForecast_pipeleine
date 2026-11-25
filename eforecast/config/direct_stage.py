# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""One case, one writer, straight into the case's own folders.

The installation serves many people, one at a time. Nothing a stage writes has to be hidden
from a second run, because there is no second run: so a dataset stage takes the case's lock,
writes its files where they belong -- each one under a temporary name, renamed when it is whole
-- and records a receipt. There is no private attempt, no copy of the model tree, nothing to
promote afterwards, nothing reserved or counted, and nothing left behind by a run that died
except the files it finished.

Two things a single sequential user needs:

* one writer per case, with a refusal that names who holds it and how its owner stops it
  (``stop_mine``; ``python -m eforecast.config.direct_stage stop-mine <case model folder>``);
* the stage receipt, which is what lets the next run reuse a finished stage.

Whether the disk has room is the operating system's to say: a stage that runs out fails with
that error, and what it had written before is what the next run of the stage rebuilds.

Every stage of every script takes this route, however the run was started. Training and
combination launch model workers; those publish their verified results into the case's model
tree themselves, and the stage holds the case and records the receipt.
"""
from contextlib import ExitStack
from contextvars import ContextVar
from copy import deepcopy
import getpass
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import time
import uuid

from eforecast.common_utils import file_stamps, lineage
from eforecast.datasets.data_preprocessing.partition_report import atomic_json
from eforecast.folder_lock import Busy as AllocatorBusy, folder_lock as file_critical_section
from . import stage_artifacts as artifacts

log = logging.getLogger(__name__)


def whoami():
    """Whoever runs this process, as the record of who holds a case names them.

    ``EFORECAST_USER`` first: a run that executes somewhere else on a person's behalf -- a Job
    on a cluster -- is told who that person is. Then the login name. A container that runs under
    its submitter's uid has no password entry for that uid, and ``getpass.getuser()`` then
    raises instead of answering; the record exists to name somebody in a message, and must
    never be what stops a stage. It did: every stage of a run on the cluster venue ended with
    ``KeyError: 'getpwuid(): uid not found: 1000'`` before it had read anything.
    """
    named = os.environ.get('EFORECAST_USER')
    if named:
        return named
    try:
        return getpass.getuser()
    except Exception:                                # noqa: BLE001 -- KeyError from getpwuid, OSError without a login
        return f'uid {os.getuid()}'

#: Stages whose producer decides whether to compute by finding its output, so a rebuild -- asked
#: for, or owed after an interrupted run -- starts without it. Extraction keeps its archive (one
#: file per day, each replaced whole, with its own flag for which days to redo) and the scalers
#: keep their own key-checked cache and ``force``.
PRESENCE_DECIDES = frozenset({'nwp', 'images', 'target', 'row', 'lstm', 'split', 'clustering', 'features'})
HOLDER = 'holder.json'
PREVIOUS = artifacts.PREVIOUS
_active = ContextVar('eforecast_direct_stage', default=None)


def active():
    """The direct stage this thread is inside, or None."""
    return _active.get()


class DirectStage:
    def __init__(self, static, plan, key):
        self.static = static                 # the case's own paths: nothing is rebased
        self.plan, self.key = deepcopy(plan), key
        from . import run_reporting
        self.context = run_reporting.context()
        if self.context:
            self.uid = hashlib.sha256(
                f"{self.context['run_id']}|{self.context['attempt_id']}|{key}".encode()).hexdigest()[:32]
        else:
            self.uid = uuid.uuid4().hex
        self.observation = None
        self.previous = []
        self.base = self.view = Path(static['path_model']).absolute()
        self.validated_payloads = None       # digests a worker checked under its own grant

    def rebase(self, value, *, canonical=False):
        return value

    @staticmethod
    def check():
        """What the workers a stage launches ask of it: whether the run was told to stop."""
        from .run_reporting import check_cancelled
        check_cancelled()

    def _folders(self):
        """What this stage writes into: the case's model tree, and the shared weather archive."""
        folders = [Path(self.static['path_model'])]
        if self.key == 'extract':
            folders.append(Path(self.static['path_group_nwp']))
        if self.key == 'images':
            folders.append(Path(self.static['path_image']))
        return folders

    def _lock(self, stack):
        for folder in sorted({artifacts.directory(self.static),
                              *(path / artifacts.DIRECTORY for path in self._folders()[1:])}):
            holder = Path(folder) / HOLDER
            try:
                stack.enter_context(file_critical_section(folder, timeout=2))
            except AllocatorBusy:
                try:
                    who = json.loads(holder.read_text())
                    since = time.strftime('%H:%M:%S', time.localtime(who['started']))
                    detail = (f"stage {who['stage']} of {who['case']}, started {since} by "
                              f"{who['user']} (pid {who['pid']})")
                except (OSError, ValueError, KeyError):
                    detail = 'a run that did not record itself'
                raise AllocatorBusy(
                    f'{self.static.get("_id", "this case")} is being written by another run: {detail}. '
                    f'One run writes a case at a time; wait for it to finish, or, if it is yours, stop it: '
                    f'python -m eforecast.config.direct_stage stop-mine {self.static["path_model"]}') from None
            atomic_json(holder, {'pid': os.getpid(), 'user': whoami(), 'stage': self.key,
                                 'case': self.static.get('_id'), 'started': time.time()})
            stack.callback(holder.unlink, missing_ok=True)

    def __enter__(self):
        self.stack = ExitStack()
        self.stack.__enter__()
        try:
            self.stack.enter_context(artifacts.verification_boundary())
            self._lock(self.stack)
            from .run_inputs import identity
            ledger = artifacts.directory(self.static) / 'writer.json'
            previous = artifacts._read(ledger) if ledger.exists() else {}
            # A run the coordinator started is recognised by these names when it collects the
            # stage: the receipt has to say which run, task and attempt produced it.
            ctx = self.context
            task, attempt = (None, None)
            if ctx:
                from .stage_workflow import stage_ids
                task, attempt = stage_ids(ctx['run_id'], self.key)
            self.producer = {**(ctx or {}), 'stage_task_id': task, 'stage_attempt_id': attempt,
                             'attempt_uid': self.uid, 'fence': int(previous.get('fence', 0)) + 1,
                             'effective_revision': (os.environ.get('EFORECAST_STAGE_EFFECTIVE_REVISION') if ctx else None)
                             or identity(self.static, self.plan['script'], self.plan['options'])}
            atomic_json(ledger, self.producer)
            if self.key == 'extract':
                from eforecast.nwp_extraction import refill
                refill.before_inputs(self.static)
            self.inputs = artifacts.input_records(self.static, self.plan, self.key)
            if 'image_source' in self.inputs:
                # The processed pixels this stage was given are part of what its receipt says
                # it read, so that pixels cached later for another case are not mistaken for it.
                processed = (Path(self.static['path_image']) / 'processed').absolute()
                self.plan['products']['image_source']['observed_processed'] = [
                    r['path'] for r in self.inputs['image_source']['files']
                    if Path(r['path']).is_relative_to(processed)]
            options = self.plan['options']
            rebuilding = ((self.plan.get('resume') or {}).get(self.key) == 'rebuild' or
                          (options.get('recreate_extraction') if self.key == 'extract' else options.get('recreate')))
            # A stage that was interrupted left this marker, and not every file it writes is
            # renamed into place when whole: what it left may be half of something. So the next
            # run of that stage does the work again instead of trusting what it finds.
            self.marker = artifacts.running_marker(self.static, self.key)
            if self.marker.exists() and artifacts.finished_despite_marker(self.static, self.key):
                # Killed after its receipt and before its tidying: that run finished. What it
                # had set aside is let go now, as it would have been a moment later.
                for path in self._own_files(anywhere=True):
                    for file in (path, Path(str(path) + lineage.SIDECAR)):
                        Path(str(file) + PREVIOUS).unlink(missing_ok=True)
                self.marker.unlink()
            interrupted = self.marker.exists()
            if interrupted:
                # Only the stages below lose what they had written. The others check their own
                # output: extract keeps the days it finished, training the trials.
                log.warning('stage %s did not finish last time; %s', self.key,
                            'its outputs are rebuilt' if self.key in PRESENCE_DECIDES else
                            'it checks what that run left and does what is missing')
                self.marker.unlink()        # this run answers for the stage from here
            if (rebuilding or interrupted) and self.key in PRESENCE_DECIDES:
                self._remove_outputs()
            # What the stage finds already in place, so that afterwards it can be told apart
            # from what the stage made: a file it did not touch keeps the provenance it had.
            self.old = artifacts.inspect_stage(self.static, self.key)
            self.before = {str(path): self._stamp(path) for path in self._own_files(anywhere=True)}
            atomic_json(self.marker, {'pid': os.getpid(), 'started': time.time(), 'attempt_uid': self.uid})
            token = _active.set(self)
            self.stack.callback(_active.reset, token)
            if self.plan.get('script') != 'datasets':
                # Training and combination hand their work to model workers, which ask the
                # current stage whether the case is already held (it is) and whether to stop.
                from eforecast.common_utils.stage_context import current_stage
                held = current_stage.set(self)
                self.stack.callback(current_stage.reset, held)
            from .stage_workspace import Observation
            self.observation = Observation(self)
            return self.observation
        except BaseException:
            self.stack.__exit__(*__import__('sys').exc_info())
            raise

    @staticmethod
    def _stamp(path):
        stat = Path(path).stat()
        return stat.st_size, stat.st_mtime_ns, stat.st_ino

    def _own_files(self, *, anywhere=False):
        """This stage's product files that exist now: inside the case's model tree, or anywhere."""
        base = Path(self.static['path_model']).absolute()
        for name, product in self.plan['products'].items():
            if product.get('producer') != self.key:
                continue
            try:
                paths = artifacts.product_paths(self.static, self.plan, name, verify=False)
            except Exception:           # noqa: BLE001 -- a product that cannot be listed has no files yet
                continue
            for path in paths:
                path = Path(path).absolute()
                if path.is_file() and (anywhere or path.is_relative_to(base)):
                    yield path

    def _guard(self):
        """A stage whose inputs moved while it worked describes something that no longer exists."""
        current = artifacts.input_records(self.static, self.plan, self.key)
        if {k: v['identity'] for k, v in current.items()} != {k: v['identity'] for k, v in self.inputs.items()}:
            raise artifacts.IncompleteStage('stage inputs changed during execution; nothing was recorded')
        for name, before in self.inputs.items():
            product = self.plan['products'].get(name, {})
            if product.get('producer') is None or name in {'extracted_nwp', 'image_pixels'}:
                # File by file, as stamps: a file a sync client marked as uploaded while the
                # stage worked moved its change time and nothing else (``file_stamps``).
                was, now = before['files'], current[name]['files']
                if len(was) != len(now) or any(not file_stamps.same(artifacts.stamp_of(a), artifacts.stamp_of(b))
                                               for a, b in zip(was, now)):
                    raise artifacts.IncompleteStage('stage inputs changed during execution; nothing was recorded')

    def _discard_new(self):
        """A stage that failed leaves nothing under a real name that it made on the way."""
        if self.key not in PRESENCE_DECIDES or not hasattr(self, 'before'):
            return
        for path in list(self._own_files()):
            if self.before.get(str(path)) != self._stamp(path):
                path.unlink()
                Path(str(path) + lineage.SIDECAR).unlink(missing_ok=True)

    def _remove_outputs(self):
        """Set the stage's outputs aside so that it computes them, and can have them back.

        The producer looks for its output to decide whether to work, so the output must not be
        there. It is renamed rather than deleted: a rebuild that fails puts the previous dataset
        back, and only one that succeeds lets it go.
        """
        for name, product in self.plan['products'].items():
            if product.get('producer') != self.key:
                continue
            base = Path(self.static['path_model']).absolute()
            for path in artifacts.product_paths(self.static, self.plan, name, verify=False):
                if not Path(path).absolute().is_relative_to(base):
                    continue        # a cache shared with other cases is never set aside for one
                for file in (Path(path), Path(str(path) + lineage.SIDECAR)):
                    aside = Path(str(file) + PREVIOUS)
                    if aside.is_file():
                        # A run killed outright never restored it: that is the last good copy,
                        # and whatever has the real name now is what the killed run left.
                        os.replace(aside, file)
                    if file.is_file():
                        log.info('rebuild: setting aside %s', file)
                        os.replace(file, str(file) + PREVIOUS)
                        self.previous.append(file)

    def _restore(self):
        self._discard_new()
        for file in self.previous:
            aside = Path(str(file) + PREVIOUS)
            if aside.is_file():
                os.replace(aside, file)             # over whatever the failed attempt left there
                log.warning('stage %s failed; the previous %s is back in place', self.key, file.name)
        if self.previous:
            # What has the real names again is what the last receipt describes, and the receipt
            # check still catches anything else that moved; nothing here is half of a rebuild.
            self.marker.unlink(missing_ok=True)

    def finish(self):
        self._guard()
        validated = self.validated_payloads
        outputs = artifacts.output_records(self.static, self.plan, self.key,
                                          **({'validated': validated} if validated else {}))
        predicted = getattr(self.observation, 'expected_predictions', None)
        if predicted is not None:
            import joblib
            required = {str(Path(record['path']).parent) for record in outputs['predictions']
                        if Path(record['path']).name == 'predictions_eval.pickle'}
            if {str(Path(path).absolute()) for path in predicted} != required:
                raise artifacts.IncompleteStage('prediction did not return every required model output')
            for path, expected in predicted.items():
                if Path(path).parent.name == 'GRAPH':
                    continue  # GRAPH returns a configured view of its existing export, without rewriting it
                if not joblib.load(Path(path) / 'predictions_eval.pickle').equals(expected):
                    raise artifacts.IncompleteStage('prediction values were not fully published')
        unchanged = all(self.before.get(record['path']) == self._stamp(record['path'])
                        for records in outputs.values() for record in records)
        for records in outputs.values():
            for record in records:
                lineage.publish_identity(record['path'], sha256=record['sha256'])
        transform_file = Path(self.static['path_model']) / 'transformers.pickle'
        if transform_file.is_file():
            lineage.publish_identity(str(transform_file), sha256=artifacts.content_digest(transform_file))
        # Output that was already there, with nothing to say who made it or from what, is
        # inspected and recorded as exactly that; this stage did not make it and cannot vouch.
        legacy = self.plan.get('completion_version') != 1 or (
            unchanged and self.old['status'] == 'legacy' and self.key != 'training')
        if self.key == 'training' and not self.plan['options'].get('prepare_only'):
            # A trial that does not say which datasets, partition and scalers it was fitted on
            # cannot be given that binding afterwards by the stage that found it.
            from eforecast.common_utils import artifact
            from eforecast.common_utils.image_assets import trial_uses_pixels
            for records in outputs.values():
                for record in records:
                    if Path(record['path']).name != 'net_weights.pickle':
                        continue
                    inputs = (artifact.read(str(Path(record['path']).parent)) or {}).get('depends_on') or {}
                    if (not inputs.get('datasets') or not inputs.get('partition') or not inputs.get('scalers') or
                            any(not item.get('sha256') for item in inputs.get('datasets', {}).values()) or
                            not (inputs.get('scalers') or {}).get('sha256') or
                            (trial_uses_pixels(self.static, Path(record['path']).parent) and not inputs.get('image_pixels'))):
                        legacy = True
        receipt = artifacts.completion(self.static, self.key, self.plan, self.producer, self.inputs,
                                       outputs, legacy=legacy)
        if unchanged and self.old.get('id'):
            receipt = artifacts.seal({**receipt, 'reused_from': {
                k: self.old[k] for k in ('id', 'producer') if k in self.old}})
        receipt = artifacts.publish_receipt(self.static, receipt)
        if legacy:
            log.warning('%s reuses inspected legacy output; original input/producer provenance is unknown', self.key)
        for file in self.previous:
            Path(str(file) + PREVIOUS).unlink(missing_ok=True)
        self.marker.unlink(missing_ok=True)
        self.observation['completion'] = {'id': receipt['id'], 'producer': self.producer,
                                         'status': 'legacy' if legacy else 'verified'}
        # The words the run page and its readers already know for these two outcomes.
        self.observation['detail'] = ('legacy output inspected; historical provenance remains unknown' if legacy
                                      else 'required stage outputs and input content verified')

    def __exit__(self, kind, error, tb):
        try:
            if kind is None and self.observation['kind'] in {'success', 'nothing_to_do'}:
                self.finish()
            elif kind is None:
                self._restore()
                self.marker.unlink(missing_ok=True)     # left without failing and without output
        except BaseException:
            self._restore()
            raise
        else:
            if kind is not None:
                self._restore()
        finally:
            self.stack.__exit__(kind, error, tb)


def holders(path_model):
    """Who is writing the case whose model folder this is: one record a held folder, newest first."""
    found = []
    for holder in sorted(Path(path_model).glob(f'**/{artifacts.DIRECTORY}/{HOLDER}')) if Path(path_model).is_dir() else []:
        try:
            found.append({**json.loads(holder.read_text()), 'folder': str(holder.parent)})
        except (OSError, ValueError):
            continue
    return sorted(found, key=lambda who: -float(who.get('started') or 0))


def stop_mine(path_model, *, patience=10.):
    """Stop the run that is writing this case, if it is the caller's own. Returns what was done.

    The one action a person can take on their own work without anybody's help: the process
    that holds the case is asked to stop (SIGTERM), which it does the way it does for Ctrl-C --
    its workers stop with it and the case's lock is given back by the kernel. Somebody else's
    run is never touched: the process has to belong to the calling user.
    """
    import signal
    records = holders(path_model)
    if not records:
        return {'stopped': False, 'reason': 'nothing is writing this case'}
    who = records[0]
    pid = int(who['pid'])
    try:
        owner = os.stat(f'/proc/{pid}').st_uid
    except OSError:
        return {'stopped': False, 'reason': f'the run that held it (pid {pid}) is gone; start again'}
    if owner != os.getuid():
        return {'stopped': False, 'reason': f"it is {who.get('user')}'s run (pid {pid}), not yours"}
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + patience
    while time.monotonic() < deadline and os.path.exists(f'/proc/{pid}'):
        time.sleep(.2)
    gone = not os.path.exists(f'/proc/{pid}')
    return {'stopped': True, 'pid': pid, 'stage': who.get('stage'), 'exited': gone,
            'reason': f"stage {who.get('stage')} of {who.get('case')} was asked to stop" +
                      ('' if gone else f'; it has not exited after {patience:g} s and is still stopping')}


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description='Who is writing a case, and stopping your own run of it.')
    parser.add_argument('command', choices=['status', 'stop-mine'])
    parser.add_argument('path_model', help='the model folder of the case (path_model)')
    args = parser.parse_args(argv)
    if args.command == 'status':
        records = holders(args.path_model)
        for who in records:
            since = time.strftime('%H:%M:%S', time.localtime(who['started']))
            print(f"stage {who['stage']} of {who['case']}, started {since} by {who['user']} (pid {who['pid']})")
        if not records:
            print('nothing is writing this case')
        return 0
    result = stop_mine(args.path_model)
    print(result['reason'])
    return 0 if result['stopped'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
