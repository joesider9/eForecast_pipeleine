# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Durable UI runs with detached execution and exclusive case ownership.

New records use the transactional run store; existing ``<case>/runs/<id>.json`` history is
read-only. Logs remain in ``<id>.log``. A run whose process
is gone without a recorded exit is reported as ``lost``.  Arguments are validated against an
allow-list per script and passed as an argv (no shell).
"""
from __future__ import annotations

import codecs
from copy import deepcopy

import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

from eforecast.config import case_revision, run_inputs, run_submission
from eforecast.config.executors import LocalExecutor
from eforecast.config.run_model import (Attempt, CancelRequest, ExecutorHandle, Outcome, Run, Task,
                                        identity, pinned)
from eforecast.config.run_store import (LegacyJSONHistory, RunConflict, RunNotFound, RunStore,
                                        StoreUnavailable, configured_store)
from eforecast.config.service import ServiceError, Workspace
from eforecast.config.stage_resume import ACTIONS as RESUME_ACTIONS
from eforecast.config.capabilities import CPU_METHODS, CLUSTER_METHODS, GLOBAL_METHODS
from eforecast.config.pipeline_stages import (ALL_DATASET_STAGES as DATASET_STAGES, COMBINE_STAGES,
                                              DATASET_DEFAULT_STAGES, COMBINE_DEFAULT_STAGES)

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
LOG_LEVELS = ("DEBUG", "INFO", "WARNING")

# script -> (file, {option: (kind, allowed values or None)})
SCRIPTS: dict[str, dict[str, Any]] = {
    "datasets": {"file": "run_datasets.py", "defaults": {"stages": list(DATASET_DEFAULT_STAGES)}, "options": {
        "stages": ("list", DATASET_STAGES + ("datasets", "all")), "recreate": ("flag", None), "recreate_extraction": ("flag", None),
        "serial": ("flag", None), "dry_run": ("flag", None)}},
    "training": {"file": "run_training.py", "options": {
        "cpu_study_device": ("str", None), "cpu_workers": ("int", None), "cpu_study_threads": ("int", None), "cpu": ("flag", None), "cpu_scope": ("str", ("global", "clusters", "both")), "cpu_method": ("str", CPU_METHODS), "refit": ("flag", None), "gpus": ("ints", None), "n_jobs": ("int", None),
        "global_method": ("str", tuple(m for m in GLOBAL_METHODS if m != 'GRAPH' and m not in CPU_METHODS)), "cluster_method": ("str", CLUSTER_METHODS), "docker": ("flag", None), "prepare_only": ("flag", None),
        "halve_batch": ("flag", None), "storage": ("str", None), "dry_run": ("flag", None)}},
    "combine": {"file": "run_combine.py", "defaults": {"stages": list(COMBINE_DEFAULT_STAGES)}, "options": {"stages": ("list", COMBINE_STAGES + ("all",)), "refit": ("flag", None), "dry_run": ("flag", None)}},
}

for _spec in SCRIPTS.values():
    _spec['options'].update(model_cache_mode=('str', ('prefer-local', 'online', 'offline', 'prewarmed')))

def python_executable(environment=None) -> str:
    from eforecast.runtime_registry import effective_python
    return effective_python(environment)


def build_argv(script: str, options: dict[str, Any] | None) -> list[str]:
    """The command line for a script from the UI options; unknown options and values are refused."""
    if script not in SCRIPTS:
        raise ServiceError(f"unknown script {script!r}; one of {', '.join(SCRIPTS)}")
    spec = SCRIPTS[script]["options"]
    argv: list[str] = []
    for key, value in run_inputs.current_options(options).items():
        if key == "log_level":
            if value not in LOG_LEVELS:
                raise ServiceError(f"log_level must be one of {LOG_LEVELS}")
            argv += ["--log-level", str(value)]
            continue
        if key not in spec:
            raise ServiceError(f"{script}: unknown option {key!r}")
        kind, allowed = spec[key]
        flag = "--" + key.replace("_", "-")
        if value is None or value == "":
            continue
        if kind == 'bool':
            if type(value) is not bool:raise ServiceError(f'{script}: {key} must be a boolean')
            argv.append(flag if value else '--no-'+key.replace('_','-'))
        elif kind == "flag":
            if value is False:
                continue
            if value is not True:
                raise ServiceError(f"{script}: {key} is a flag")
            argv.append(flag)
        elif kind == "list":
            values = [str(v) for v in (value if isinstance(value, list) else [value])]
            if not values:
                raise ServiceError(f"{script}: {key} must not be empty")
            bad = [v for v in values if v not in allowed]
            if bad:
                raise ServiceError(f"{script}: {key} cannot contain {bad}")
            argv += [flag.replace("--stages", "--stage"), *values]
        elif kind == "ints":
            values = value if isinstance(value, list) else [value]
            if not values or not all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in values):
                raise ServiceError(f"{script}: {key} must be nonnegative GPU ids")
            if len(set(values)) != len(values):
                raise ServiceError(f"{script}: {key} must not contain duplicate GPU ids")
            argv += [flag, *[str(v) for v in values]]
        elif kind == "int":
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ServiceError(f"{script}: {key} must be a positive integer")
            argv += [flag, str(value)]
        else:
            if key == 'cpu_study_device' and not isinstance(value, str):
                raise ServiceError('training: cpu_study_device must be a string')
            text = str(value)
            if allowed and text not in allowed:
                raise ServiceError(f"{script}: {key} must be one of {', '.join(allowed)}")
            if any(c in text for c in "\n\r\0"):
                raise ServiceError(f"{script}: {key} contains control characters")
            argv += [flag, text]
    return argv


def _alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:  # a zombie of ours counts as finished
        with open(f"/proc/{pid}/stat") as fh:
            return fh.read().split(")")[-1].split()[0] != "Z"
    except OSError:
        return True


class ChangesNotAccepted(ServiceError):
    """A run that would replace built work for a changed configuration, submitted without the answer.

    Raised before anything is accepted or recorded: there is no run to fail. ``preview`` is
    what the run would do, which is what the caller shows to whoever has to answer.
    """

    def __init__(self, message, preview):
        super().__init__(message)
        self.preview = preview


class RunManager:
    """Coordinates durable inputs and executor observations; Popen is only a reaping cache."""

    def __init__(self, workspace: Workspace | None = None, *, store: RunStore | None = None, executor=None,
                 coordinated=False):
        self.workspace = workspace or Workspace()
        self.store = store if store is not None else configured_store(self.workspace.root)
        self._scope_cache: dict[str, tuple[str, str]] = {}
        self._processes: dict[str, subprocess.Popen] = {}
        self.executor = executor if executor is not None else LocalExecutor(workspace=self.workspace.root)
        self.coordinated = coordinated
        self._preparer = ThreadPoolExecutor(max_workers=1, thread_name_prefix='eforecast-prepare')
        self._pending_jobs: set[str] = set()
        self._jobs_lock = threading.Lock()
        self._parser_warmup = run_inputs.warm_parsers()

    # ---------------------------------------------------------------- records
    def runs_dir(self, case: str) -> Path:
        return self.workspace.case_dir(case) / "runs"

    def _record_path(self, case: str, run_id: str) -> Path:
        if not run_id.replace("-", "").isalnum():
            raise ServiceError(f"bad run id {run_id!r}")
        return self.runs_dir(case) / f"{run_id}.json"

    def _scope(self, case: str) -> tuple[str, str]:
        from .run_scope import resolve
        return resolve(self.workspace, self.store, case, cache=self._scope_cache)

    def _project(self, run: Run) -> dict[str, Any]:
        from .stage_workflow import project as stages
        record = json.loads(run.metadata_json)
        legacy_pid_only = False
        if run.snapshot_json is not None:
            record.update(json.loads(run.snapshot_json))
        if run.tasks:
            attempt = run.tasks[0].attempts[-1] if run.tasks[0].attempts else None
            if attempt and attempt.executor:
                # not the seal: the launch intent is reported to the UI, the supervisor's
                # receipt secret stays in the durable record
                record.update({key: value for key, value in json.loads(attempt.executor.identity_json).items()
                               if key != 'seal'})
                record['executor_adapter'] = attempt.executor.adapter
                legacy_pid_only = attempt.executor.adapter == 'local_process'
            if attempt and attempt.execution_json:
                record.update(json.loads(attempt.execution_json))
        status = 'running'
        if run.state == 'unknown':
            status = 'lost'
        elif run.outcome is not None:
            status = {'success': 'finished', 'failure': 'failed', 'cancelled': 'stopped',
                      'partial': 'failed', 'nothing_to_do': 'finished'}[run.outcome.kind]
        request = json.loads(run.request_json)
        record.pop('profile_selection', None)      # of a run accepted when profiles existed
        state = run.state
        if self.coordinated and legacy_pid_only and state != 'terminal':
            state, status = 'unknown', 'lost'  # read-time normalization; the old PID is not proof of ownership
        error = run.report_error
        if not error and self.coordinated and legacy_pid_only and run.state != 'terminal':
            error = 'Legacy PID ownership is unverified; case ownership is retained for reconciliation.'
        elif not error and run.outcome and run.outcome.kind == 'failure' and record.get('pid') is None:
            error = run.outcome.detail
        elif not error and run.state == 'unknown' and record.get('pid') is None:
            error = 'Launch status is unconfirmed; case ownership is retained pending executor recovery.'
        return {**record, 'id': run.id, 'owner_id': run.owner_id, 'case_id': run.case_id,
                'revision': run.revision, 'storage': 'durable', 'state': state,
                'status': status, 'returncode': run.outcome.returncode if run.outcome else None,
                'finished': run.outcome.observed_at if run.outcome else None,
                'stop_requested': run.cancel is not None, 'cleanup': run.cleanup,
                'outcome': json.loads(run.to_json())['outcome'],
                'submission_key': request.get('client_key'), 'request_hash': request.get('request_hash'),
                'submission_key_scope': request.get('client_key_scope', 'tenant') if request.get('client_key') else None,
                'accepted_case_revision': request.get('case_revision'), 'accepted': run.created_at,
                'deleted': run.deleted,
                'report_error': run.report_error,
                'worker_result': json.loads(run.result_json) if run.result_json else None,
                'collection': json.loads(run.collection_json) if run.collection_json else None,
                'cleanup_detail': json.loads(run.cleanup_json) if run.cleanup_json else None,
                'stage_plan': json.loads(run.snapshot_json).get('stage_plan') if run.snapshot_json else None,
                'stages': stages(run),
                'preparation_error': error}

    def _create(self, record: dict[str, Any]) -> dict[str, Any]:
        owner, case_id = self._scope(record['case'])
        request = pinned({key: record[key] for key in ('script', 'options', 'command')})
        handle = ExecutorHandle(identity(), 'local_process', pinned({'pid': record['pid']}))
        attempt = Attempt(identity(), 0, state='running', executor=handle)
        task = Task(identity(), f"entrypoint:{record['script']}", 'case', request,
                    state='running', attempts=(attempt,))
        metadata = {key: value for key, value in record.items() if key not in
                    {'status', 'returncode', 'finished', 'stop_requested', 'resources'}}
        snapshot = {key: value for key, value in record.items() if key.startswith(
            ('config', 'input_', 'normalized_', 'effective_'))}
        run = Run(record['id'], owner, case_id, request, record['started'],
                  snapshot_json=pinned(snapshot), metadata_json=pinned(metadata),
                  state='running', tasks=(task,))
        return self._project(self.store.create(run))

    def _write(self, record: dict[str, Any]) -> dict[str, Any]:
        owner, case_id = self._scope(record['case'])
        old = self.store.get(owner, case_id, record['id'])
        if old.revision != record['revision']:
            raise RunConflict('run revision changed; reload before transitioning')
        cancel = old.cancel
        if record.get('stop_requested') and cancel is None:
            cancel = CancelRequest(identity(), time.time(), owner, 'UI stop')
        state, outcome = old.state, old.outcome
        if record['status'] == 'lost':
            state = 'unknown'  # an absent PID is not an observed exit status
        elif record['status'] != 'running':
            state = 'terminal'
            kind = {'finished': 'success', 'failed': 'failure', 'stopped': 'cancelled'}[record['status']]
            outcome = Outcome(kind, record['finished'], record.get('returncode'), 'UI process observation')
        tasks = tuple(replace(task, state=state, outcome=outcome,
                    attempts=tuple(replace(attempt, state=state, outcome=outcome) for attempt in task.attempts))
                      for task in old.tasks)
        cleanup = old.cleanup
        if cancel is not None and cleanup == 'not_required':
            cleanup = 'pending'
        return self._project(self.store.compare_and_set(
            replace(old, state=state, outcome=outcome, cancel=cancel, tasks=tasks, cleanup=cleanup), old.revision))

    def _refresh(self, record: dict[str, Any]) -> dict[str, Any]:
        """Bring a record up to date with the process state; persists exit codes of runs we started."""
        for cache in (self._processes, getattr(self.executor, 'processes', {})):
            for key, proc in list(cache.items()):
                if proc.poll() is not None:
                    cache.pop(key, None)
        if record.get('storage') == 'legacy_json':
            return record
        resource_path = self.runs_dir(record['case']) / 'resources' / f"{record['id']}.json"
        try:
            snapshot = json.loads(resource_path.read_text())
            if not isinstance(snapshot, dict):
                raise ValueError('invalid resource snapshot')
            record['resources'] = snapshot
        except FileNotFoundError:
            record.pop('resources', None)
        except (OSError, ValueError):
            record['resources'] = {'error': 'Resource snapshot unreadable'}
        if record.get('resources'):
            from .resource_views import project
            record['resources']['compute'] = project(record['resources'])
        if self.coordinated or record.get('dispatch_route', {}).get('venue') == 'broker':
            return record  # HTTP reads never dispatch or settle an independently coordinated run
        if record.get('executor_adapter') == self.executor.adapter:
            owner, case_id = self._scope(record['case'])
            run = self.store.get(owner, case_id, record['id'])
            if run.state != 'terminal':
                run = self._observe_executor(run)
            if run.state == 'terminal' and run.cleanup not in {'complete', 'not_required'}:
                from eforecast.config.run_cleanup import step
                run = step(self, run)
            record.update(self._project(run))
            return record
        if record.get('state') in {'pending', 'preparing', 'queued'}:
            return record
        if record.get('status') != 'running' and not (record.get('state') == 'unknown'
                                                    and record['id'] in self._processes):
            return record
        proc = self._processes.get(record["id"])
        if proc is not None:
            code = proc.poll()
            if code is None:
                return record
            cancelled = record.get('stop_requested') and code in {-signal.SIGTERM, -signal.SIGKILL}
            record.update(status='finished' if code == 0 else ('stopped' if cancelled else 'failed'),
                          returncode=code, finished=time.time())
            try:
                updated = self._write(record)
            except RunConflict:
                return self.get(record['case'], record['id'])
            self._processes.pop(record["id"], None)
            record.update(updated)
            return record
        # A numeric-only historical handle proves neither ownership nor liveness, even when
        # the number currently exists. Keep the original record and normalize its state.
        if proc is None:
            record.update(status="lost")
            try:
                record.update(self._write(record))
            except RunConflict:
                return self.get(record['case'], record['id'])
        return record

    def list(self, case: str) -> list[dict[str, Any]]:
        owner, case_id = self._scope(case)
        durable = self.store.list(owner, case_id, include_deleted=True)
        known = {run.id for run in durable}
        records = [self._refresh(self._project(run)) for run in durable if not run.deleted]
        records.extend(self._refresh(record) for record in LegacyJSONHistory(self.runs_dir(case)).list()
                       if record['id'] not in known)
        return sorted(records, key=lambda r: r.get("started", 0) or 0, reverse=True)

    def get(self, case: str, run_id: str) -> dict[str, Any]:
        self._record_path(case, run_id)  # validate path-facing identity, including legacy lookup
        owner, case_id = self._scope(case)
        try:
            run = self.store.get(owner, case_id, run_id)
        except RunNotFound:
            return self._refresh(LegacyJSONHistory(self.runs_dir(case)).get(run_id))
        if run.deleted:
            raise RunNotFound(f'run {run_id!r} not found')
        return self._refresh(self._project(run))

    def running(self, case: str) -> list[dict[str, Any]]:
        return [r for r in self.list(case) if r["status"] == "running"]

    # ---------------------------------------------------------------- frozen inputs
    def inputs_dir(self, case: str, run_id: str) -> Path:
        return self.runs_dir(case) / "inputs" / run_id

    def freeze_inputs(self, case: str, run_id: str, script: str = 'datasets', argv=None, env=None) -> dict[str, Any]:
        """Archive coherent raw files and pin resolved experiment data plus parsed execution options."""
        source = self.workspace.case_path(case)
        options = run_inputs.options_for(script, ['--config', str(source), *(argv or [])], env)
        return run_inputs.capture(source, self.inputs_dir(case, run_id) / case, script, options,
                                  lock=case_revision.locked(self.workspace, case, shared=True), environment=env)

    def prune_inputs(self, case: str, keep: tuple[str, ...] = ()) -> int:
        """Remove the frozen configurations of this case's runs that have ended; how many went.

        A run needs its copy until it has ended and been collected. After that the copy is
        nobody's input: what the model was built from is kept once, in the model's folder
        (``model_configuration``). The copies of command-line runs go by ``run_inputs.prune``,
        which leaves alone whatever a live process still runs from.
        """
        root = self.runs_dir(case) / 'inputs'
        if not root.is_dir():
            return 0
        owner, case_id = self._scope(case)
        known = {run.id: run for run in self.store.list(owner, case_id, include_deleted=True)}
        removed = 0
        for entry in sorted(root.iterdir()):
            run = known.get(entry.name)
            if not entry.is_dir() or entry.name in keep or run is None:
                continue                # not a folder of a run this store knows: not ours to judge
            if run.state != 'terminal' or run.cleanup not in {'not_required', 'complete'}:
                continue
            shutil.rmtree(entry, ignore_errors=True)
            removed += not entry.exists()
        return removed + run_inputs.prune(root, keep=[root / name for name in keep])

    # ---------------------------------------------------------------- lifecycle
    def preview(self, case, script, options=None, *, expected_revision=None, action='continue'):
        """Resolve the saved case through the same catalogue as preparation and CLI dry-run.

        ``action`` -- continue, new or rebuild -- adds the resume plan: what a run would reuse,
        what must be hashed first, what would be rebuilt and which artifacts that would replace.
        ``None`` is the action a run submitted without one is prepared with: what its own
        options say (``recreate``, ``refit``), else the case's.
        """
        import tempfile
        from .stage_catalogue import resolve
        argv = build_argv(script, options)
        source = self.workspace.case_path(case)
        effective = run_inputs.options_for(script, ['--config', str(source), *argv])
        with case_revision.locked(self.workspace, case, shared=True):
            document = self.workspace.load(case)
            from eforecast.runtime_registry import resolve as resolve_runtime, require_available, ContractError
            try:
                runtime_binding = require_available(resolve_runtime(document.get('methods') or {}),
                                                    python=python_executable(), check_source=True)
            except ContractError as error:
                raise ServiceError(str(error)) from error
            revision = case_revision.revision(document)
            if expected_revision is not None and revision != expected_revision:
                raise RunConflict('case revision changed; reload the saved case before previewing')
            files = run_submission.archive(source)
        with tempfile.TemporaryDirectory(prefix='eforecast-stage-preview-') as temporary:
            folder = Path(temporary) / case
            run_submission.restore(files, folder)
            try:
                static = run_inputs.resolve(folder, source, document, runtime_binding=runtime_binding)
                from eforecast.init.initialize import initializer
                from .stage_resume import decide, requested_action
                # The configuration as a run has it when it plans: with its folders and with the
                # area of each weather provider, which is part of what a dataset stage is built
                # from. The plan used to be made one step earlier, so the fingerprint it compared
                # with the receipts lacked the areas and every dataset of a case that reads
                # weather looked as if its configuration had changed.
                prepared = initializer(static, dry_run=True)
                plan = resolve(prepared, script, effective)
                # What the families' inputs were called by the earlier formula, from the
                # configurations of this case's runs that can still be read. A preview learns
                # for itself and writes nothing.
                from . import model_configuration
                model_configuration.learn(prepared, self.runs_dir(case) / 'inputs')
                resume = (decide(prepared, plan, action=action or requested_action(prepared, plan['options']))
                          if plan.get('completion_version') == 1 else None)
            except Exception as error:
                return {'case_revision': revision, 'status': 'unresolved',
                        'reason': f'{type(error).__name__}: {error}', 'stage_plan': None, 'resume': None}
        # 'ready' is a claim about artifacts this preview did not hash. When a payload's metadata
        # moved since its receipt, the honest answer is that it must be verified first -- the stage
        # boundary will hash it, and a planner must not treat it as a stage it may skip.
        status = 'blocked' if plan['blocked'] else 'verify' if plan.get('unverified') else 'ready'
        result = {'case_revision': revision, 'status': status, 'stage_plan': plan, 'resume': resume}
        if resume is not None:
            # What this run would replace because the case no longer says what it was built
            # from, which is what a person accepts before it starts -- and, where the model
            # keeps the configuration it was built from, the settings that changed since.
            from .stage_resume import changes
            result['changes'] = changes(resume)
            if result['changes']:
                try:
                    from . import model_configuration
                    result['since'] = model_configuration.since(plan['path_model'], document)
                except Exception as error:                      # noqa: BLE001 -- a reference, never a refusal
                    log.debug('what changed since the model was built is unavailable (%r)', error)
        if script == 'training' and effective.get('cpu'):
            from .resource_contract import ResourcePreview, CONTROL_KEYS
            from .schema import JobsConfig
            bounded = run_inputs.apply_overrides(static, script, effective)
            controls = {key: bounded.get(key, JobsConfig.model_fields[key].default) for key in CONTROL_KEYS}
            result['resource_controls'] = ResourcePreview(controls=controls).model_dump(mode='json')
        return result

    def recover_stages(self, case: str, *, apply: bool = False) -> list[dict[str, Any]]:
        """Interrupted publications of this case, and optionally the receipts they can restore."""
        from eforecast.init.initialize import initializer
        from .stage_resume import recover
        if not self.workspace.exists(case):
            raise ServiceError(f'case {case} does not exist')
        source = self.workspace.case_path(case)
        with case_revision.locked(self.workspace, case, shared=True):
            static = initializer(run_inputs.normalize(source, source), dry_run=True)
        return recover(static, apply=apply)

    def submit(self, case: str, script: str, options: dict[str, Any] | None = None,
               env: dict[str, str] | None = None, *, client_key: str | None = None,
               expected_revision: str | None = None, submitted_by: str | None = None,
               action: str | None = None, accept_changes: bool = False) -> dict[str, Any]:
        """Commit identity, immutable input bytes and exclusive case ownership before any spawn.

        Client keys are tenant-wide within a workspace/store namespace. Retrying must retain
        expected_revision (including its omission); submitted_by records the original caller.
        ``action`` is the Continue / New revision / Rebuild choice the caller previewed: it is
        part of the request, so a retry of the same key must carry the same one, and preparation
        plans with it instead of re-deriving a choice from the case's own settings.

        ``accept_changes`` is the caller's answer to what the preview showed: the configuration
        is not the one the case's datasets or models were built from, and this run may replace
        them. Without it such a run is refused when it is prepared, before anything is launched
        (``stage_resume.changes``). It is part of the request too, and only when it is given:
        a request made before the question existed still replays as the request it was.
        """
        if action is not None and action not in RESUME_ACTIONS:
            raise ServiceError(f"action must be one of {', '.join(RESUME_ACTIONS)}")
        if not self.workspace.exists(case):
            raise ServiceError(f"case {case} does not exist")
        key = identity() if client_key is None else client_key
        if not isinstance(key, str) or not key.strip() or len(key) > 200 or '\x00' in key:
            raise ServiceError('submission key must contain 1–200 characters')
        if env is not None and any(not isinstance(k, str) or not isinstance(v, str) or
                not k or '=' in k or '\x00' in k or '\x00' in v for k, v in env.items()):
            raise ServiceError('execution environment overrides must have valid string keys and values')
        # Canonical argv normalizes suppressed flags, null/empty fields and scalar/list forms.
        options = deepcopy(run_inputs.current_options(options))
        argv = build_argv(script, options)
        canonical_argv = build_argv(script, dict(sorted(options.items())))
        request_hash = run_submission.digest(pinned({'script': script, 'argv': canonical_argv,
            'expected_revision': expected_revision, 'environment_overrides': env or {},
            **({'action': action} if action else {}),
            **({'accept_changes': True} if accept_changes else {})}).encode())
        owner, case_id = self._scope(case)
        previous = self.store.submission(owner, case_id, key, request_hash)
        if previous is not None:
            return self._project(previous)
        if not accept_changes:
            self.refuse_unaccepted(case, script, options, action, expected_revision)
        # Collect Popen outcomes before the transaction, never signal or wait under a DB lock.
        self.list(case)
        for legacy in LegacyJSONHistory(self.runs_dir(case)).list():
            if legacy.get('status') == 'running':
                raise RunConflict(f'case already has a running legacy run ({legacy["id"]})')
        run_id = time.strftime('%Y%m%d-%H%M%S') + '-' + identity()
        folder = self.runs_dir(case)
        source = self.workspace.case_path(case)
        environment = {**os.environ, **(env or {})}
        effective_options = run_inputs.options_for(script, ['--config', str(source), *argv], environment)
        with case_revision.locked(self.workspace, case, shared=True):
            if self._scope(case) != (owner, case_id):
                raise RunConflict('case identity changed; reload before submitting')
            # Check retries again after waiting for a case reader/writer. An accepted retry
            # must not be rejected merely because the live configuration has since changed.
            previous = self.store.submission(owner, case_id, key, request_hash)
            if previous is not None:
                return self._project(previous)
            document = self.workspace.load(case)
            revision = case_revision.revision(document)
            if expected_revision is not None and revision != expected_revision:
                raise RunConflict('case revision changed; reload the saved case before submitting')
            from eforecast.runtime_registry import resolve as resolve_runtime, require_available, ContractError
            try:
                runtime_binding = require_available(resolve_runtime(document.get('methods') or {}, environment),
                                                    python=python_executable(environment), check_source=True)
            except ContractError as error:
                raise ServiceError(str(error)) from error
            input_error = None
            try:
                files = run_submission.archive(source)
                guards = run_submission.experiment_sources(document, source)
                methods = document.get('methods') or {}
                if not isinstance(methods, dict):
                    methods = {}
                transfer = methods.get('transfer_learning_from') or {}
                if isinstance(methods, dict) and isinstance(transfer, dict) and methods.get('transfer_learning') and transfer.get('configuration'):
                    base = Path(transfer['configuration']).expanduser().resolve()
                    guards[str(base)] = run_submission.digest(base.read_bytes())
            except OSError as error:
                files, guards = {}, {}
                input_error = f'{type(error).__name__}: {error}'
            from .kubernetes_runs import configured_target, code_digest
            managed_code = code_digest(REPO_ROOT) if configured_target() is not None else None
            request = {'client_key': key, 'client_key_scope': 'tenant',
                'request_hash': request_hash, 'script': script, 'argv': argv, 'action': action,
                **({'accept_changes': True} if accept_changes else {}),
                'expected_revision': expected_revision, 'case_revision': revision,
                'source': str(source), 'workspace': str(self.workspace.root.resolve()),
                'python': python_executable(environment), 'runtime_binding': runtime_binding,
                'entrypoint': str(REPO_ROOT / SCRIPTS[script]['file']),
                'entrypoint_sha256': run_submission.digest((REPO_ROOT / SCRIPTS[script]['file']).read_bytes()),
                **({'code_digest': managed_code} if managed_code else {}),
                'effective_options': run_inputs.encode(effective_options),
                'environment': {**{name: environment.get(name) for name in run_submission.EXECUTION_ENV}, **(env or {})},
                'files': files, 'input_read_error': input_error,
                'experiment_sources': guards,
                'submitted_by': submitted_by or 'local'}
            metadata = {'case': case, 'script': script, 'options': options or {}, 'command': [],
                'pid': None, 'started': time.time(), 'log': str(folder / f'{run_id}.log'),
                'config_source': str(source), 'input_snapshot_status': 'pending'}
            # The store stamps the route in the same transaction as acceptance. A failed
            # route read must never become an implicit local submission.
            # What the installation looked like when this run was accepted. Read from the
            # cached reading, never taken here: a submission must not wait on docker, and
            # this is inside the window that holds the case claim. A run that fails forty
            # minutes later then arrives with the explanation already attached, and an
            # absent or stale reading says so rather than implying a healthy machine.
            try:
                from eforecast.config import health
                metadata['health'] = health.summary()
            except Exception as error:                                      # noqa: BLE001
                log.debug('health summary unavailable for run %s: %r', run_id, error)
            attempt = Attempt(identity(), 0)
            task = Task(identity(), f'entrypoint:{script}', 'case', pinned({'script': script, 'argv': argv}),
                        attempts=(attempt,))
            run = Run(run_id, owner, case_id, pinned(request), metadata['started'],
                      metadata_json=pinned(metadata), tasks=(task,))
            try:
                run, _ = self.store.submit(run, key, request_hash)
            except StoreUnavailable:
                accepted = self.store.submission(owner, case_id, key, request_hash)
                if accepted is None:
                    raise
                run = accepted  # committed acceptance survived its lost acknowledgement
        return self._project(run)

    def refuse_unaccepted(self, case, script, options, action, expected_revision):
        """Refuse, before anything is accepted, a run that replaces built work for a changed configuration.

        The datasets or the trained models of the case were built from other settings than the
        case has now, and this run would replace them. That is asked of a person with the list
        in front of them, so a request without the answer is turned away here with the list --
        no run is recorded, nothing has to be cleaned up, and the same request with
        ``accept_changes`` is a new request.

        Preparation asks again with hashed evidence (``prepare``), because what is built can
        change between the two. A plan that cannot be made here refuses nothing: preparation
        will say why it cannot.
        """
        from .stage_resume import ASKED_ABOUT, describe_changes
        if script not in ASKED_ABOUT:
            return
        try:
            preview = self.preview(case, script, options, expected_revision=expected_revision, action=action)
        except Exception as error:                              # noqa: BLE001 -- see the docstring
            # Also a revision that moved or a store that is away: the submission itself meets
            # the same condition a moment later and answers it in its own words.
            log.debug('no plan for %s %s before submission (%r); preparation decides', case, script, error)
            return
        if not preview.get('changes'):
            return
        from .document_changes import describe
        settings = [describe(change) for change in (preview.get('since') or {}).get('changes', ())]
        raise ChangesNotAccepted(
            describe_changes(preview['changes'], case, settings) + '\nNothing was changed and no run was started. '
            'Preview this run, read what it replaces and accept it there; or put the configuration back.', preview)

    def start(self, case: str, script: str, options: dict[str, Any] | None = None,
              env: dict[str, str] | None = None, *, client_key=None, expected_revision=None, action=None,
              accept_changes=False):
        """Synchronous compatibility for programmatic callers; HTTP uses submit + enqueue."""
        record = self.submit(case, script, options, env, client_key=client_key,
                             expected_revision=expected_revision, action=action, accept_changes=accept_changes)
        self.prepare(case, record['id'])
        return self.get(case, record['id'])

    def enqueue(self, case, run_id):
        if self.coordinated:
            if os.environ.get('EFORECAST_RUN_COORDINATOR', 'auto') == 'auto':
                from eforecast.config.run_coordinator import ensure_service
                try:
                    ensure_service(self.workspace)
                except Exception:
                    log.exception('coordinator startup unavailable; accepted run %s remains pending', run_id)
            return  # the independent coordinator discovers the committed case claim
        with self._jobs_lock:
            if run_id in self._pending_jobs:
                return
            self._pending_jobs.add(run_id)

        def work():
            try:
                self.prepare(case, run_id)
            except Exception:
                log.exception('run %s preparation remains durable for recovery', run_id)
            finally:
                with self._jobs_lock:
                    self._pending_jobs.discard(run_id)

        try:
            self._preparer.submit(work)
        except RuntimeError:
            with self._jobs_lock:
                self._pending_jobs.discard(run_id)
            log.exception('run %s accepted but local preparer unavailable; restart/recover pending work', run_id)

    def recover_pending(self):
        """Startup discovery hook. An independent periodic coordinator follows in A09-04."""
        try:
            pending = self.store.pending()
        except StoreUnavailable:
            log.exception('pending-run scan unavailable; accepted work retained for recovery')
            return
        for run in pending:
            request = json.loads(run.request_json)
            if request.get('workspace') == str(self.workspace.root.resolve()):
                self.enqueue(json.loads(run.metadata_json)['case'], run.id)

    @staticmethod
    def _execution_progress(run, state, *, outcome=None, executor=None, execution=None):
        from .stage_workflow import execution_progress
        return execution_progress(run, state, outcome=outcome, executor=executor, execution=execution)

    def prepare(self, case, run_id):
        from dataclasses import asdict
        from types import SimpleNamespace
        from eforecast.training.command_line import liveness
        from eforecast.training.command_line.liveness import ProcessIdentity, process_identity

        owner, case_id = self._scope(case)
        run = self.store.get(owner, case_id, run_id)
        from .kubernetes_runs import is_broker, in_assigned_pod
        if is_broker(run) and not in_assigned_pod(self.store, run):
            from .kubernetes_runs import target_of
            if target_of(run) is None:
                detail = 'Broker/Kubernetes execution is not connected to this historical intent; ownership retained'
                if run.report_error != detail:
                    self.store.compare_and_set(replace(run, report_error=detail), run.revision)
            return  # the durable outbox/Job owns preparation; never fall back to a host process
        if self._executor_handle(run) is not None and run.state in {'queued', 'running', 'unknown'}:
            self._observe_executor(run, launch=True)
            return
        if run.state not in {'pending', 'preparing', 'queued'} or run.deleted:
            return
        # A run is prepared and started when it is asked for: there is no slot to wait for and
        # nothing is set aside for it. What keeps two runs off one case is the case's own lock.
        if run.state != 'queued':
            if run.preparation_json:
                preparation = json.loads(run.preparation_json)
                prior = ProcessIdentity(**preparation['owner'])
                observation = SimpleNamespace(process=None, request=SimpleNamespace(owner=prior))
                fenced_preparer = preparation.get('controller') is not None and run.controller_json is not None and (
                    getattr(self.store, 'controller_json', None) == run.controller_json)
                if not fenced_preparer and liveness.observe(observation)[0] != 'dead':
                    return
            boot = liveness.boot_id()
            token = identity()
            preparer = pinned({'token': token, 'owner': asdict(process_identity(
                os.getpid(), liveness.host_id(boot), boot)), 'started_at': time.time(),
                'controller': run.controller_json})
            try:
                run = self.store.compare_and_set(replace(self._execution_progress(run, 'preparing'),
                    preparation_json=preparer), run.revision)
            except RunConflict:
                return
            except StoreUnavailable:
                observed = self.store.get(owner, case_id, run_id)
                if observed.state != 'preparing' or observed.preparation_json != preparer:
                    raise
                run = observed  # exact committed token, not a new preparation attempt
            request = json.loads(run.request_json)
            try:
                from .kubernetes_runs import worker_command
                accepted_entrypoint = worker_command(run)[1] if is_broker(run) else request['entrypoint']
                if request.get('input_read_error'):
                    raise ServiceError(request['input_read_error'])
                run_submission.verify_sources(request['experiment_sources'])
                frozen = run_inputs.capture(request['source'],
                    self.inputs_dir(case, run_id) / token / case, request['script'],
                    run_inputs.decode(request['effective_options']), environment=run_submission.environment(request['environment']),
                    accepted_files=request['files'], runtime_binding=request.get('runtime_binding'))
                run_submission.verify_sources(request['experiment_sources'])
                if not frozen['config_frozen']:
                    raise ServiceError(f"cannot prepare accepted configuration: {frozen.get('config_error')}")
                if ((request.get('runtime_binding') or {}).get('selection')
                        and frozen['input_snapshot_status'] != 'pinned'):
                    raise ServiceError('Accepted runtime selection could not be pinned; dispatch was refused.')
                if frozen.get('config_revision') != request['case_revision']:
                    raise ServiceError('accepted files do not match the accepted case revision; submit a new request')
                # A degraded normalization still executes the accepted case bytes, with the
                # original experiment anchor supplied explicitly to the managed CLI path.
                if frozen['input_snapshot_status'] == 'degraded':
                    frozen['config'] = frozen['config_raw']
                if (Path(accepted_entrypoint).resolve() == REPO_ROOT / f"run_{request['script']}.py"
                        and frozen['input_snapshot_status'] == 'pinned'):
                    from .stage_catalogue import resolve
                    from .stage_resume import decide, pin
                    from .stage_workflow import bind
                    from eforecast.init.initialize import initializer
                    # Planned on the configuration as the run itself will have it -- folders and
                    # weather areas included (see ``preview``) -- so that the service and the
                    # worker compare the same fingerprint with the same receipts.
                    static = prepared = initializer(run_inputs.load(frozen['config']), dry_run=True)
                    plan = resolve(prepared, request['script'],
                                   run_inputs.decode(request['effective_options']))
                    # What this run may skip is decided here, from hashed evidence, and pinned:
                    # the worker is given permission, never the decision. It hashes again at the
                    # stage boundary, so a payload that changes in between is still rebuilt.
                    if plan.get('completion_version') == 1:
                        from .stage_resume import requested_action
                        chosen = request.get('action') or requested_action(prepared, plan['options'])
                        # Before anything is decided with them, and before the copies of the
                        # runs that ended are removed below: the names the earlier formula gave
                        # the families' inputs are learned from those copies and kept.
                        from . import model_configuration
                        model_configuration.remember(prepared, self.runs_dir(case) / 'inputs')
                        decided = decide(prepared, plan, verify=True, action=chosen)
                        # Datasets or trained models that exist were built from other settings
                        # than the case has now, and this run would replace them. That is asked
                        # of a person, with the list in front of them; a request that carries no
                        # answer is refused here, where nothing has been launched yet.
                        from .stage_resume import changes, describe_changes, settings_changed
                        replaced = changes(decided)
                        if replaced and not request.get('accept_changes'):
                            raise ServiceError(
                                describe_changes(replaced, case, settings_changed(plan['path_model'], frozen['config']))
                                + '\nNothing was changed. Preview this run, read what it replaces and accept '
                                'it there; or put the configuration back.')
                        pin(plan, decided)
                    frozen['stage_plan'] = plan
                    run = bind(run, plan, frozen.get('effective_revision'))
                run = self.store.compare_and_set(replace(self._execution_progress(run, 'queued'),
                    snapshot_json=pinned(frozen)), run.revision)
                try:
                    # This run has its frozen configuration; those of the runs that ended go.
                    self.prune_inputs(case, keep=(run_id,))
                except Exception as error:                              # noqa: BLE001 -- tidying fails no run
                    log.warning('earlier frozen configurations of %s were not removed (%s)', case, error)
            except RunConflict:
                return  # cancelled or replaced while preparing; never overwrite the successor
            except StoreUnavailable:
                observed = self.store.get(owner, case_id, run_id)
                if observed.state != 'queued' or observed.snapshot_json != pinned(frozen):
                    raise
                run = observed  # queued snapshot committed; a lost reply is not failed preparation
            except Exception as error:
                failed = self._execution_progress(run, 'terminal', outcome=Outcome('failure', time.time(), None,
                    f'preparation failed: {type(error).__name__}: {error}'))
                self.store.compare_and_set(failed, run.revision)
                return
        self._launch(run)

    def _launch(self, run):
        request, metadata = json.loads(run.request_json), json.loads(run.metadata_json)
        frozen = json.loads(run.snapshot_json)
        try:
            from .kubernetes_runs import require_execution_authority
            require_execution_authority(self.store, run)
            if frozen.get('stage_plan'):
                from .stage_catalogue import require_ready, resolve
                require_ready(frozen['stage_plan'])
                # On the configuration as the run has it, like the plan it is compared with
                # (``preview``): without the weather areas every dataset stage of a case that
                # reads weather was found "configuration changed" here, and the run of the
                # service ended before it was launched.
                from eforecast.init.initialize import initializer
                current = resolve(initializer(run_inputs.load(frozen['config']), dry_run=True), request['script'],
                                  run_inputs.decode(request['effective_options']),
                                  completion_version=frozen['stage_plan'].get('completion_version'))
                if current['identity'] != frozen['stage_plan']['identity']:
                    raise ServiceError('stage catalogue differs from the prepared contract')
                current['resume'] = frozen['stage_plan'].get('resume', {})
                require_ready(current)
            if self._scope(metadata['case']) != (run.owner_id, run.case_id):
                raise ServiceError('case identity changed before launch')
            run_submission.verify_sources(request['experiment_sources'])
            config = frozen['config']
            from .kubernetes_runs import is_broker, worker_command
            python, entrypoint = worker_command(run) if is_broker(run) else (request['python'], request['entrypoint'])
            cmd = [python, entrypoint, '--config', config, *request['argv']]
            environment = {**run_submission.environment(request['environment']), 'PYTHONUNBUFFERED': '1',
                'EFORECAST_RUN_RESULT_REQUIRED': '1' if Path(entrypoint).resolve() ==
                    REPO_ROOT / f"run_{request['script']}.py" else '0',
                'EFORECAST_CASE': config, 'EFORECAST_RUN_INPUT_CONFIG': str(Path(config).resolve()),
                'EFORECAST_RUN_INPUT_STATUS': frozen['input_snapshot_status'],
                'EFORECAST_RUN_INPUT_SOURCE': request['source'],
                'EFORECAST_RUN_INPUT_GUARD': pinned(request['experiment_sources']),
                'EFORECAST_RESOURCE_STATUS_FILE': str(self.runs_dir(metadata['case']) / 'resources' / f'{run.id}.json'),
                'EFORECAST_RUN_ID': run.id, 'EFORECAST_TASK_ID': run.tasks[0].id,
                'EFORECAST_ATTEMPT_ID': run.tasks[0].attempts[-1].id}
            if is_broker(run):
                environment['EFORECAST_PYTHON'] = python
                environment['PYTHONPATH'] = json.loads(run.metadata_json)['dispatch_route']['execution']['source_root']
            environment.pop('EFORECAST_PROFILE_SELECTION', None)    # of a version that had profiles
            environment.pop('EFORECAST_STAGE_PLAN_ID', None)
            environment.pop('EFORECAST_STAGE_COMPLETION_VERSION', None)
            environment.pop('EFORECAST_STAGE_EFFECTIVE_REVISION', None)
            # Left over from a version that kept a ledger of runs; a run inherits neither.
            environment.pop('EFORECAST_DISPATCH', None)
            environment.pop('EFORECAST_STAGE_STORAGE', None)
            environment.pop('EFORECAST_STUDY_ACTION', None)
            environment.pop('EFORECAST_STUDY_REVISION', None)
            if request['script'] == 'training':
                accepted_resume = (frozen.get('stage_plan') or {}).get('resume_plan') or {}
                if accepted_resume.get('action') in RESUME_ACTIONS:
                    environment['EFORECAST_STUDY_ACTION'] = accepted_resume['action']
                    if accepted_resume['action'] == 'new' and (accepted_resume.get('revision') or {}).get('training'):
                        environment['EFORECAST_STUDY_REVISION'] = accepted_resume['revision']['training']
            environment.pop('EFORECAST_STAGE_RESUME', None)
            # The submitter's answer to what the preview showed, or no answer at all: a run
            # never inherits the one somebody gave the service's own environment.
            environment.pop('EFORECAST_ACCEPT_CHANGES', None)
            if request.get('accept_changes'):
                environment['EFORECAST_ACCEPT_CHANGES'] = '1'
            if frozen.get('stage_plan'):
                # The permissions this run was accepted with travel with it. The worker may
                # narrow them -- rebuild what the plan hoped to reuse -- but it may not grant
                # itself a reuse the collector never approved.
                environment['EFORECAST_STAGE_RESUME'] = pinned(frozen['stage_plan'].get('resume') or {})
                environment['EFORECAST_STAGE_PLAN_ID'] = frozen['stage_plan']['identity']
                environment['EFORECAST_STAGE_COMPLETION_VERSION'] = str(frozen['stage_plan'].get('completion_version', 0))
                environment['EFORECAST_STAGE_EFFECTIVE_REVISION'] = frozen.get('effective_revision', '')
            effective = run_inputs.decode(request['effective_options'])
            if request['script'] == 'training':
                environment['EFORECAST_OPTUNA_STORAGE'] = effective.get('storage') or ''
                environment['EFORECAST_METRICS_PORT'] = str(effective['metrics_port'])
                environment.update({key: value or '' for key, value in effective['worker_environment'].items()})
            handle = self.executor.plan(run.id, run.tasks[0].id, run.tasks[0].attempts[-1].id,
                                        cmd, environment, REPO_ROOT, metadata['log'])
        except Exception as error:
            failed = self._execution_progress(run, 'terminal', outcome=Outcome('failure', time.time(), None,
                f'launch preparation failed: {type(error).__name__}: {error}'))
            self.store.compare_and_set(failed, run.revision)
            return
        # Pin the complete intent before calling an executor. Discovery after a lost DB reply
        # uses this same identity; it never allocates a replacement launch token.
        try:
            run = self.store.compare_and_set(self._execution_progress(run, 'queued', executor=handle), run.revision)
        except RunConflict:
            return
        except StoreUnavailable:
            observed = self.store.get(run.owner_id, run.case_id, run.id)
            if self._executor_handle(observed) != handle:
                raise
            run = observed
        self._observe_executor(run, launch=True)

    def _executor_handle(self, run):
        if run.tasks and run.tasks[0].attempts:
            handle = run.tasks[0].attempts[-1].executor
            if handle is not None and handle.adapter == self.executor.adapter:
                return handle
        return None

    def _observe_executor(self, run, *, launch=False):
        from .kubernetes_runs import is_broker, in_assigned_pod
        if is_broker(run) and not in_assigned_pod(self.store, run):
            return run  # a host reader cannot observe or settle native PIDs from a Pod
        handle = self._executor_handle(run)
        observed = self.executor.collect(handle)
        intent = json.loads(handle.identity_json)
        if intent.get('report_directory'):
            from eforecast.config import run_reporting
            context = {'run_id': run.id, 'task_id': run.tasks[0].id,
                       'attempt_id': run.tasks[0].attempts[-1].id, 'executor_id': handle.id}
            try:
                events, offset, at_end, partial = run_reporting.read_batch(
                    intent['report_directory'], run.report_offset, context)
                if events:
                    run = self.store.collect_reports(run, events, offset)
                if partial and observed.state == 'finished':
                    raise ValueError('worker ended with an unfinished report journal record')
                if not at_end:
                    if run.state in {'queued', 'unknown'} and observed.state in {'running', 'finished'}:
                        run = self.store.compare_and_set(self._execution_progress(run, 'running'), run.revision)
                    return run  # bounded collection continues on the next tick
            except RunConflict:
                return self.store.get(run.owner_id, run.case_id, run.id)
            except StoreUnavailable:
                raise
            except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
                after = self._execution_progress(run, 'unknown') if run.state != 'terminal' else run
                return self.store.compare_and_set(replace(after, report_error=f'report collection: {error}'), run.revision)
        if launch and observed.state == 'absent' and run.cancel is None:
            # The adapter must recheck absence inside its atomic launch boundary.
            check = getattr(self.store, 'check_ownership', None)
            if check is not None:
                check()
            from .kubernetes_runs import require_execution_authority
            try:
                require_execution_authority(self.store, run)
            except ServiceError as error:
                # Preserve this historical queued intent without launching it on a different
                # venue, and make the recovery refusal visible in the ordinary run panel.
                if run.report_error == str(error):
                    return run
                return self.store.compare_and_set(replace(run, report_error=str(error)), run.revision)
            observed = self.executor.submit(handle)
            if intent.get('report_directory') and observed.state in {'running', 'finished'}:
                # A fast worker can publish its result and exit during submit's handshake.
                # Read that journal after observing the exit, before classifying completion.
                return self._observe_executor(self.store.get(run.owner_id, run.case_id, run.id))
        proc = getattr(self.executor, 'processes', {}).get(handle.id)
        if proc is not None:
            self._processes[run.id] = proc
        prior = run.tasks[0].attempts[-1].execution_json
        execution = prior
        if prior is None and json.loads(observed.evidence).get('worker'):
            execution = observed.evidence
        if observed.state == 'finished':
            if run.state == 'terminal':
                return run
            cancelled = run.cancel is not None and (observed.returncode in {-signal.SIGTERM, -signal.SIGKILL}
                                                  or observed.detail == 'cancelled_before_spawn')
            kind = 'cancelled' if cancelled else 'success' if observed.returncode == 0 else 'failure'
            detail = observed.detail
            if run.result_json:
                structured = json.loads(run.result_json)
                kind, detail = structured['kind'], structured['detail']
                if observed.returncode not in (0, None) and kind in {'success', 'nothing_to_do', 'partial'}:
                    kind, detail = 'failure', f'worker exit {observed.returncode} contradicts {kind}: {detail}'
            elif intent.get('result_required') and observed.returncode == 0:
                kind, detail = 'failure', 'worker exited zero without its required structured result'
            stage_tasks = [task for task in run.tasks[1:] if 'catalogue_version' in json.loads(task.request_json)]
            if kind in {'success', 'nothing_to_do'} and any(task.outcome is None or
                    task.outcome.kind not in {'success', 'nothing_to_do'} for task in stage_tasks):
                kind, detail = 'failure', 'entry point ended without successful observations for its declared stages'
            children = self.store.child_summary(run)
            if children['unrooted']:
                return self.store.compare_and_set(replace(run, report_error='unrooted child execution records'), run.revision)
            cleanup = 'pending' if children['unresolved'] or handle.adapter == 'local_supervised' else run.cleanup
            receipt = pinned({'executor_id': handle.id, 'returncode': observed.returncode,
                'finished_at': observed.finished_at, 'evidence': json.loads(observed.evidence),
                'source': 'structured_worker' if run.result_json else 'process_exit', 'kind': kind})
            after = replace(self._execution_progress(run, 'terminal', execution=execution,
                outcome=Outcome(kind, observed.finished_at, observed.returncode, detail)),
                collection_json=receipt, cleanup=cleanup, report_error=None)
        else:
            if not launch and run.state == 'queued' and observed.state in {'absent', 'unknown'}:
                return run  # publishing/starting a supervisor is still accepted queued work
            state = 'running' if observed.state == 'running' else 'unknown'
            after = self._execution_progress(run, state, execution=execution)
        if after == run:
            return run
        try:
            updated = self.store.compare_and_set(after, run.revision)
        except RunConflict:
            return self.store.get(run.owner_id, run.case_id, run.id)
        if updated.state == 'terminal':
            proc = self._processes.get(run.id)
            if proc is not None and proc.poll() is not None:
                self._processes.pop(run.id, None)
        return updated

    def stop(self, case: str, run_id: str, grace: float = 10.0) -> dict[str, Any]:
        record = self.get(case, run_id)
        if record.get('read_only'):
            raise ServiceError('legacy run history is read-only; verify worker ownership before taking action')
        owner, case_id = self._scope(case)
        run = self.store.get(owner, case_id, run_id)
        from .kubernetes_runs import is_broker
        if self.coordinated or is_broker(run):
            return self._project(self.store.request_cancel(owner, case_id, run_id,
                CancelRequest(identity(), time.time(), owner, 'UI stop')))
        if record.get('state') in {'pending', 'preparing', 'queued'} and self._executor_handle(run) is None:
            if run.revision != record['revision']:
                raise RunConflict('run changed before cancellation; reload before retrying')
            cancelled = replace(self._execution_progress(run, 'terminal', outcome=Outcome('cancelled', time.time(),
                None, 'cancelled before launch')), cancel=CancelRequest(identity(), time.time(), owner, 'UI stop'))
            cancelled = self.store.compare_and_set(cancelled, run.revision)
            return self._project(cancelled)
        owner, case_id = self._scope(case)
        run = self.store.get(owner, case_id, run_id)
        handle = self._executor_handle(run)
        if handle is not None and run.state != 'terminal':
            if run.cancel is None:
                run = self.store.request_cancel(owner, case_id, run_id,
                    CancelRequest(identity(), time.time(), owner, 'UI stop'))
            from eforecast.config.run_cleanup import step
            deadline = time.monotonic() + grace + 5
            while time.monotonic() < deadline:
                run = step(self, run, grace=grace)
                if run.cleanup in {'complete', 'unknown'}:
                    return self._project(run)
                time.sleep(.05)
            return self._project(run)
        if record["status"] != "running":
            return record
        # Old numeric-only records have no safe signal target. Normalize to unknown and
        # retain case ownership; never kill an unrelated process that reused the number.
        record.update(status='lost', stop_requested=True)
        return self._write(record)

    def resolve(self, case: str, run_id: str, *, actor: str | None = None, reason: str = '') -> dict[str, Any]:
        """End a run that can no longer end itself, with a person as the evidence.

        A supervisor lost with its host, or a report journal that cannot be read, leaves a run
        whose execution or cleanup can never be confirmed. That state is deliberate -- the case
        claim is held so nothing starts beside work that may still be alive -- but until now it
        was also permanent: such a run cannot be deleted and its case accepts nothing new.

        This invents no exit code and no successful cleanup. It records who decided, when, why,
        and what was still unconfirmed, and it refuses while the system can still make progress
        by itself or while the execution answers as alive.
        """
        record = self.get(case, run_id)
        if record.get('read_only'):
            raise ServiceError('legacy run history is read-only; verify worker ownership before taking action')
        owner, case_id = self._scope(case)
        run = self.store.get(owner, case_id, run_id)
        from .kubernetes_runs import is_broker, assignment
        bound = assignment(self.store, run.id) if is_broker(run) else None
        if bound and bound['state'] != 'complete':
            raise ServiceError('Kubernetes still owns this attempt; confirm its Pod and allocator cleanup before resolving it')
        handle = self._executor_handle(run)
        if handle is not None:
            observed = self.executor.discover(handle)
            if observed.state in {'running', 'created'}:
                raise ServiceError('the execution is alive; stop the run before resolving it')
            if observed.state == 'finished' and run.state != 'terminal':
                raise ServiceError('this run has a completion receipt; let the coordinator collect it')
        if run.state != 'unknown' and run.cleanup != 'unknown':
            raise ServiceError('only a run whose execution or cleanup cannot be confirmed needs an '
                               'operator decision; this one can still finish on its own')
        # A decision needs a decider's words, and it is asked for only once the run is one that
        # really needs deciding: the record keeps them instead of an invented phrase.
        if not isinstance(reason, str) or not reason.strip():
            raise ServiceError('a resolution needs a reason: say what was checked and why no worker '
                               'can still write')
        reason = reason.strip()
        latest = self.store.get(owner, case_id, run_id)
        plan = json.loads(latest.cleanup_json) if latest.cleanup_json else {}
        decided_at = time.time()
        plan.update(phase='complete', resolved_by=actor or 'local', resolved_at=decided_at,
                    resolution=reason, unconfirmed=plan.get('issues') or ['execution evidence was lost'])
        detail = f'resolved by {actor or "local"} without confirmed evidence: {reason}'
        # Cleanup that never had anything to do, or already finished, is not advanced: 'not_required'
        # -> 'complete' is not a transition the aggregate allows, and inventing one would say that
        # something was cleaned up (R10-03).
        cleanup = 'complete' if latest.cleanup in {'pending', 'running', 'unknown'} else latest.cleanup
        resolved = replace(latest, cleanup=cleanup, cleanup_json=pinned(plan))
        if latest.state != 'terminal':
            # Not an observed exit: a failure whose detail says exactly whose decision it was.
            resolved = replace(self._execution_progress(resolved, 'terminal',
                outcome=Outcome('failure', decided_at, None, detail)),
                cleanup=cleanup, cleanup_json=pinned(plan))
        return self._project(self.store.compare_and_set(resolved, latest.revision))

    def delete(self, case: str, run_id: str) -> None:
        record = self.get(case, run_id)
        if record.get('read_only'):
            raise ServiceError('legacy run history is read-only; an explicit migration is required to change it')
        if record["status"] == "running":
            raise ServiceError("stop the run before deleting it")
        if record.get('state') == 'unknown' or record.get('cleanup') not in {None, 'not_required', 'complete'}:
            raise ServiceError('resolve run ownership and cleanup before deleting it')
        owner, case_id = self._scope(case)
        run = self.store.get(owner, case_id, run_id)
        if run.revision != record['revision']:
            raise RunConflict('run revision changed; reload before deleting')
        self.store.compare_and_set(replace(run, deleted=True), record['revision'])
        Path(record.get("log", "")).unlink(missing_ok=True)
        (self.runs_dir(case) / 'resources' / f'{run_id}.json').unlink(missing_ok=True)
        shutil.rmtree(self.inputs_dir(case, run_id), ignore_errors=True)

    def log(self, case: str, run_id: str, offset: int = 0, limit: int = 256 * 1024) -> dict[str, Any]:
        """The log from ``offset`` (bytes), at most ``limit`` bytes; ``next`` is the offset to poll from."""
        record = self.get(case, run_id)
        path = Path(record["log"])
        limit = max(1, min(int(limit), 4 * 1024 * 1024))
        if not path.exists() and record.get('storage') == 'durable' and record.get('pid') is None:
            text = record.get('preparation_error') or ''
            return {'text': text, 'offset': 0, 'next': 0, 'size': 0, 'status': record['status'],
                    'truncated': False, 'generation': f'prelaunch:{run_id}'}
        with open(path, "rb") as fh:
            stat = os.fstat(fh.fileno())
            size = stat.st_size
            if offset < 0 or offset > size:
                offset = max(0, size - limit)
            fh.seek(offset)
            chunk = fh.read(limit)
        decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
        text = decoder.decode(chunk, final=offset + len(chunk) >= size)
        pending = len(decoder.getstate()[0])
        if pending and pending < len(chunk):
            chunk = chunk[:-pending]  # the next page starts at this character's first byte
        elif pending:
            text = chunk.decode('utf-8', errors='replace')  # preserve progress for a tiny caller-supplied limit
        return {"text": text, "offset": offset, "next": offset + len(chunk), "size": size,
                "status": record["status"], "truncated": offset + len(chunk) < size,
                "generation": f'{stat.st_dev}:{stat.st_ino}'}


def scripts_catalogue() -> dict[str, Any]:
    """The options of every script for the UI."""
    return {name: {"file": spec["file"], "defaults": spec.get('defaults', {}), "options": {k: {"kind": v[0], "choices": list(v[1]) if v[1] else None} for k, v in spec["options"].items()}}
            for name, spec in SCRIPTS.items()}
