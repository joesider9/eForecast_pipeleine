# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Managed whole-run handoff: transactional outbox -> Job -> existing A09 coordinator.

Stages remain within their accepted entry-point supervisor. A Job hosts one dedicated
coordinator, not a second stage scheduler. Its Pod UID is attached exactly once before
preparation; a replacement Pod cannot adopt the same run. All outcomes and artifacts still
pass the ordinary A09/A08 publication contracts. Loss without confirmed Pod termination
retains ownership and cleanup as unknown.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import time

from .executors import digest, location
from .run_model import CancelRequest, ExecutorHandle, Outcome, Run, identity, pinned
from .service import ServiceError
from . import dispatch_outbox as outbox

ENV = 'EFORECAST_KUBE_RUNS_FILE'
TARGET = 'kubernetes-run'
SCHEMA = (
    '''CREATE TABLE IF NOT EXISTS eforecast_kubernetes_runs (
       namespace TEXT NOT NULL,dispatch_id TEXT NOT NULL,run_id TEXT NOT NULL,handle TEXT NOT NULL,
       state TEXT NOT NULL,pod_uid TEXT,agent TEXT,created DOUBLE PRECISION NOT NULL,
       updated DOUBLE PRECISION NOT NULL,detail TEXT NOT NULL,
       PRIMARY KEY(namespace,dispatch_id),UNIQUE(namespace,run_id))''',
    '''CREATE INDEX IF NOT EXISTS eforecast_kubernetes_runs_active
       ON eforecast_kubernetes_runs(namespace,state,created,dispatch_id)''',
)


class KubernetesRunError(ServiceError):
    pass


class PlacementBusy(KubernetesRunError):
    pass


def validate_target(value):
    from .kubernetes_executor import immutable_image
    if not isinstance(value, dict) or value.get('version') != 1:
        raise KubernetesRunError('managed Kubernetes target must be a version 1 object')
    required = {'version', 'image', 'server', 'namespace', 'shared_root', 'workspace', 'state_root',
                'source_root', 'python', 'service_account', 'connection_secret', 'resources',
                'node_selector', 'pending_deadline', 'active_deadline', 'max_jobs', 'volume'}
    if set(value) != required:
        raise KubernetesRunError('managed Kubernetes target fields differ: ' + ', '.join(sorted(set(value) ^ required)))
    immutable_image(value['image'])
    if not isinstance(value['server'], str) or not value['server'].startswith('https://'):
        raise KubernetesRunError('managed Kubernetes needs an explicit TLS API server')
    for field in ('shared_root', 'workspace', 'state_root', 'source_root', 'python'):
        if not isinstance(value[field], str) or not Path(value[field]).is_absolute() or '..' in Path(value[field]).parts:
            raise KubernetesRunError(f'{field} must be an absolute canonical path')
    shared = Path(value['shared_root'])
    if shared == Path('/') or any(not Path(value[key]).is_relative_to(shared) for key in ('workspace', 'state_root')):
        raise KubernetesRunError('workspace and state_root must belong to one explicit shared root, never /')
    if Path(value['state_root']).is_relative_to(Path(value['workspace'])):
        raise KubernetesRunError('execution state must be outside the mutable case workspace')
    if not isinstance(value['node_selector'], dict) or not value['node_selector']:
        raise KubernetesRunError('explicit placement is required for the qualified shared-storage scope')
    for key in ('max_jobs', 'pending_deadline', 'active_deadline'):
        if type(value[key]) is not int or value[key] <= 0:
            raise KubernetesRunError(f'{key} must be a bounded positive integer')
    if value['max_jobs'] > 32 or value['pending_deadline'] > value['active_deadline']:
        raise KubernetesRunError('invalid managed Job bounds')
    resources = value['resources']
    if (not isinstance(resources, dict) or set(resources) != {'cpu_threads', 'ram_mb', 'gpu_slots'}
            or any(type(resources[key]) is not int or resources[key] <= 0 for key in ('cpu_threads', 'ram_mb'))
            or type(resources['gpu_slots']) is not int or resources['gpu_slots'] not in (0, 1)):
        raise KubernetesRunError('managed Jobs need positive CPU/RAM and zero or one whole GPU')
    volume = value['volume']
    if not isinstance(volume, dict) or len(volume) != 1 or next(iter(volume)) not in ('hostPath', 'persistentVolumeClaim'):
        raise KubernetesRunError('choose an explicit shared hostPath or PVC binding')
    if 'hostPath' in volume:
        if volume['hostPath'] != {'path': str(shared), 'type': 'Directory'}:
            raise KubernetesRunError('hostPath must match the shared root and already exist')
        if set(value['node_selector']) != {'kubernetes.io/hostname'}:
            raise KubernetesRunError('hostPath qualification is restricted to one explicitly named node')
    for field in ('namespace', 'service_account', 'connection_secret'):
        if not isinstance(value[field], str) or not value[field] or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-.' for c in value[field]):
            raise KubernetesRunError(f'{field} must name a Kubernetes object')
    return value


def code_digest(root):
    """Content identity of the Python package/entry points copied into the worker image."""
    root = Path(root)
    files = sorted((root/'eforecast').rglob('*.py'))
    files += [root/name for name in ('run_datasets.py', 'run_training.py', 'run_combine.py')]
    return digest({str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in files if '__pycache__' not in path.parts})


def configured_target():
    path = os.environ.get(ENV)
    if not path:
        return None
    try:
        return validate_target(json.loads(Path(path).read_text()))
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise KubernetesRunError(f'managed Kubernetes configuration is unusable: {error}') from error


def target_of(run):
    return json.loads(run.metadata_json).get('dispatch_route', {}).get('execution')


def is_broker(run):
    return json.loads(run.metadata_json).get('dispatch_route', {}).get('venue') == 'broker'


def message(run):
    target = validate_target(target_of(run))
    attempt = run.tasks[0].attempts[-1]
    return outbox.message(run_id=run.id, task_id=run.tasks[0].id, attempt_id=attempt.id,
        generation=attempt.number, runtime='worker-run', target=TARGET,
        input_digest=digest({'request': json.loads(run.request_json), 'target': target}),
        requirements=target['resources'], deadline=target['pending_deadline'])


def enqueue(store, run, *, cursor):
    if not is_broker(run):
        return
    request = json.loads(run.request_json)
    target = validate_target(target_of(run))
    if store.placeholder != '%s':
        raise KubernetesRunError('managed Kubernetes requires PostgreSQL run and allocator stores')
    if request.get('workspace') != target['workspace']:
        raise KubernetesRunError('case workspace differs from the pinned Kubernetes shared volume')
    if not request.get('entrypoint_sha256') or not request.get('code_digest'):
        raise KubernetesRunError('managed Kubernetes needs a pinned entry-point digest; submit a new run')
    if request.get('environment', {}).get('EFORECAST_WORKER_IMAGE') or json.loads(run.metadata_json).get('options', {}).get('docker'):
        raise KubernetesRunError('nested host Docker is not available inside a managed Job')
    binding = request.get('runtime_binding') or {}
    if any(binding.get('definitions', {}).get(name, {}).get('kind') == 'container'
           for name in binding.get('selection', {}).values()):
        raise KubernetesRunError('this managed Job route does not launch nested Docker model runtimes; qualify their Kubernetes worker adapter first')
    outbox.enqueue(store, message(run), cursor=cursor)


def _run(store, run_id, cursor, *, lock=False):
    suffix = ' FOR UPDATE' if lock and store.placeholder == '%s' else ''
    store._execute(cursor, 'SELECT payload FROM eforecast_runs WHERE namespace=? AND id=?' + suffix,
                   (store.namespace, run_id))
    row = cursor.fetchone()
    if row is None:
        raise outbox.Obsolete('the managed run no longer exists')
    return Run.from_json(row[0])


def assignment(store, run_id, *, cursor=None):
    from contextlib import nullcontext
    with nullcontext(cursor) if cursor is not None else store._transaction(write=False) as handle:
        store._execute(handle, '''SELECT dispatch_id,handle,state,pod_uid,agent,created,updated,detail
            FROM eforecast_kubernetes_runs WHERE namespace=? AND run_id=?''', (store.namespace, run_id))
        row = handle.fetchone()
    if row is None:
        return None
    record = dict(zip(('dispatch_id', 'handle', 'state', 'pod_uid', 'agent', 'created', 'updated', 'detail'), row))
    record['handle'] = ExecutorHandle(**json.loads(record['handle']))
    record['agent'] = json.loads(record['agent']) if record['agent'] else None
    return record


def _save(store, run_id, **fields):
    with store._transaction() as cursor:
        store._execute(cursor, 'UPDATE eforecast_kubernetes_runs SET ' + ','.join(k+'=?' for k in fields)
            + ',updated=? WHERE namespace=? AND run_id=?', (*fields.values(), time.time(), store.namespace, run_id))


def in_assigned_pod(store, run):
    dispatch_id, uid = os.environ.get('EFORECAST_KUBE_DISPATCH_ID'), os.environ.get('EFORECAST_POD_UID')
    if not dispatch_id or not uid:
        return False
    bound = assignment(store, run.id)
    return bool(bound and bound['dispatch_id'] == dispatch_id and bound['pod_uid'] == uid
                and bound['state'] == 'running' and os.environ.get('EFORECAST_KUBE_RUN_ID') == run.id)


def require_execution_authority(store, run):
    if is_broker(run):
        if not in_assigned_pod(store, run):
            raise KubernetesRunError('Broker/Kubernetes execution is not connected to this process; only the assigned Pod may launch')
    else:
        from .dispatch_route import require_executable
        require_executable('local')


def attach_pod(store, run_id, dispatch_id, uid):
    """One Pod UID, before preparation. A duplicate never acquires execution authority."""
    if not uid:
        raise KubernetesRunError('a managed Job must identify its Pod UID')
    with store._transaction() as cursor:
        run = _run(store, run_id, cursor, lock=True)
        bound = assignment(store, run_id, cursor=cursor)
        if (not bound or bound['dispatch_id'] != dispatch_id or bound['state'] in ('complete', 'unknown')
                or run.cancel or run.state == 'terminal'):
            raise KubernetesRunError('dispatch is cancelled, completed or no longer available')
        if bound['pod_uid'] and bound['pod_uid'] != uid:
            raise KubernetesRunError('a different Pod already owns this attempt; replacement refused')
        here = location()
        store._execute(cursor, '''UPDATE eforecast_kubernetes_runs SET pod_uid=?,agent=?,state='running',updated=?
            WHERE namespace=? AND run_id=?''', (uid, pinned(here), time.time(), store.namespace, run_id))
    return run


def worker_command(run):
    """Map the accepted entry point into the pinned image; verify its bytes before executing."""
    target = validate_target(target_of(run))
    request = json.loads(run.request_json)
    if code_digest(target['source_root']) != request.get('code_digest'):
        raise KubernetesRunError('worker image Python source differs from the accepted source; rebuild and qualify it')
    entry = Path(target['source_root']) / Path(request['entrypoint']).name
    if entry.name not in ('run_datasets.py', 'run_training.py', 'run_combine.py'):
        # Explicit synthetic qualification programs may live on the shared volume.
        entry = Path(request['entrypoint'])
        if not entry.is_relative_to(Path(target['shared_root'])):
            raise KubernetesRunError('entry point is outside the pinned image/shared input boundary')
    if hashlib.sha256(entry.read_bytes()).hexdigest() != request.get('entrypoint_sha256'):
        raise KubernetesRunError('image entry point differs from the accepted code; qualify the image before retrying')
    return target['python'], str(entry)


class Bridge:
    def __init__(self, store, workspace, *, cluster=None, target=None, connection=None):
        from .kubernetes_api import Cluster
        self.store, self.workspace = store, workspace
        self.target = validate_target(target or configured_target())
        self.cluster = cluster or Cluster.configured(timeout=3)
        if (self.cluster.namespace != self.target['namespace'] or
                getattr(self.cluster, 'server', self.target['server']) != self.target['server']):
            raise KubernetesRunError('configured cluster differs from the managed execution target')
        self.connection = connection
        self.cursor = None
        self.dispatch_cursor = None

    def executor(self, target):
        from .kubernetes_executor import KubernetesExecutor
        return KubernetesExecutor(self.cluster, Path(target['state_root'])/'kubernetes', image=target['image'],
                                  service_account=target['service_account'])

    def launch(self, store, document, *, cursor):
        """Validate under the run lock and persist the Job intent in the outbox claim transaction."""
        run = _run(store, document['identity']['run_id'], cursor, lock=True)
        if (run.cancel or run.state != 'pending' or not is_broker(run)
                or document != message(run) or run.tasks[0].attempts[-1].executor is not None):
            raise outbox.Obsolete('run, generation, venue or accepted inputs moved before claim')
        target = validate_target(target_of(run))
        if digest(target) != digest(self.target):
            raise PlacementBusy('this dispatcher does not own the accepted target; keep its message queued')
        # Namespace-wide short claim lock bounds all launchers, including absent first rows.
        if store.placeholder == '%s':
            key = int.from_bytes(hashlib.sha256(('kubernetes-cap:' + store.namespace).encode()).digest()[:8], 'big', signed=True)
            store._execute(cursor, 'SELECT pg_advisory_xact_lock(?)', (key,))
        store._execute(cursor, "SELECT COUNT(*) FROM eforecast_kubernetes_runs WHERE namespace=? AND state<>'complete'", (store.namespace,))
        if cursor.fetchone()[0] >= target['max_jobs']:
            raise PlacementBusy('the bounded managed Job queue is full')
        request = json.loads(run.request_json)
        environment = {'EFORECAST_WORKSPACE': target['workspace'], 'EFORECAST_RUN_STORE': 'postgres',
            'EFORECAST_RUN_STORE_NAMESPACE': store.namespace,
            'EFORECAST_RUN_EXECUTOR_DIR': str(Path(target['state_root'])/'pods'/run.id),
            'EFORECAST_KUBE_RUN_ID': run.id, 'EFORECAST_KUBE_DISPATCH_ID': document['dispatch_id'],
            'EFORECAST_RUN_WORKSPACE_ID': os.environ.get('EFORECAST_RUN_WORKSPACE_ID') or request['workspace'],
            'EFORECAST_ALLOCATOR_BACKEND': 'postgres', 'EFORECAST_PYTHON': target['python'],
            'PYTHONUNBUFFERED': '1', 'EFORECAST_HOME': target['source_root'],
            'PYTHONPATH': target['source_root'],
            # The Pod runs under the coordinator's uid, which has no name inside the image. Whoever
            # submitted the run is who holds the case while it executes (``direct_stage.whoami``).
            'EFORECAST_USER': str(json.loads(run.metadata_json).get('submitted_by') or 'local')}
        handle = self.executor(target).plan_job(run_id=run.id, task_id=run.tasks[0].id,
            attempt_id=run.tasks[0].attempts[-1].id,
            argv=[target['python'], '-m', 'eforecast.config.kubernetes_run_agent', run.id],
            environment=environment, resources=target['resources'],
            volumes=[{'name':'managed', 'path':target['shared_root'], 'volume':target['volume']}],
            active_deadline=target['active_deadline'], pending_deadline=target['pending_deadline'],
            node_selector=target['node_selector'], uid=os.getuid(), gid=os.getgid())
        # Connection material is a namespace-scoped Secret, never a broker payload or run record.
        intent = json.loads(handle.identity_json)
        path = Path(intent['directory'])/'kubernetes.json'
        manifest = json.loads(path.read_text())
        from .kubernetes_retention import FINALIZER, CONTRACT
        manifest['manifest']['spec']['template']['metadata']['finalizers'] = [FINALIZER]
        pod = manifest['manifest']['spec']['template']['spec']
        pod['containers'][0]['envFrom'] = [{'secretRef': {'name': target['connection_secret']}}]
        pod['terminationGracePeriodSeconds'] = 45
        from .executors import atomic_json
        atomic_json(path, manifest)
        handle = replace(handle, identity_json=pinned({**intent,
            'termination_retention': CONTRACT, 'manifest_digest': digest(manifest['manifest'])}))
        now = time.time()
        store._execute(cursor, '''INSERT INTO eforecast_kubernetes_runs
            (namespace,dispatch_id,run_id,handle,state,pod_uid,agent,created,updated,detail)
            VALUES(?,?,?,?,'planned',NULL,NULL,?,?,'durable launch intent')''',
            (store.namespace, document['dispatch_id'], run.id, pinned(asdict(handle)), now, now))

    def _report(self, run, detail):
        prefix = 'Kubernetes: '
        # Do not overwrite an independently reported artifact/cleanup failure.
        if run.report_error and not run.report_error.startswith(prefix):
            return
        detail = prefix+detail if detail else None
        if run.report_error != detail:
            self.store.compare_and_set(replace(run, report_error=detail), run.revision)

    def tick(self):
        from . import dispatch_broker as broker
        from .run_store import RunConflict
        page = self.store.active_page(after=self.cursor, limit=16)
        self.cursor = (page[-1].created_at, page[-1].id) if page else None
        for run in page:
            if not is_broker(run) or target_of(run) != self.target:
                continue
            bound = assignment(self.store, run.id)
            if bound is None:
                if run.cancel:
                    self._unlaunched_cancel(run)
                elif run.state == 'pending':
                    row = outbox.get(self.store, message(run)['dispatch_id'])
                    if row is None:
                        with self.store._transaction() as cursor:
                            current = _run(self.store, run.id, cursor, lock=True)
                            if not current.cancel and current.state == 'pending':
                                enqueue(self.store, current, cursor=cursor)
                    elif row['state'] == outbox.PUBLISHED and time.time()-row['updated_at'] > 30:
                        outbox.reconcile(self.store, [message(run)])
        published = broker.drain(self.store, limit=8, connection=self.connection, target=TARGET)
        try:
            received = broker.dispatch(self.store, limit=2, connection=self.connection, routes=('managed',),
                            launch=self.launch, timeout=3)
            waiting = published.get('waiting') or received.get('waiting')
            if waiting:
                for run in page:
                    if is_broker(run) and target_of(run) == self.target and assignment(self.store, run.id) is None:
                        try:
                            self._report(run, waiting)
                        except RunConflict:
                            pass
        except PlacementBusy:
            pass  # Unacknowledged delivery is requeued on connection close.
        # Include terminal runs until their Job/placement has been retired.
        with self.store._transaction(write=False) as cursor:
            self.store._execute(cursor, """SELECT run_id FROM eforecast_kubernetes_runs
                WHERE namespace=? AND state<>'complete' AND dispatch_id>? ORDER BY dispatch_id LIMIT 16""",
                (self.store.namespace, self.dispatch_cursor or ''))
            ids = [row[0] for row in cursor.fetchall()]
        self.dispatch_cursor = None
        for run_id in ids:
            try:
                with self.store._transaction(write=False) as cursor:
                    run = _run(self.store, run_id, cursor)
                bound = assignment(self.store, run_id)
                self.dispatch_cursor = bound['dispatch_id']
                self.reconcile(run, bound)
            except RunConflict:
                continue  # Independent Pod coordinator or cancellation advanced the aggregate.

    def _unlaunched_cancel(self, run):
        from .runs import RunManager
        with self.store._transaction() as cursor:
            current = _run(self.store, run.id, cursor, lock=True)
            if assignment(self.store, run.id, cursor=cursor) is not None:
                return
            if current.cancel and current.state != 'terminal':
                ended = RunManager._execution_progress(current, 'terminal',
                    outcome=Outcome('cancelled', time.time(), None, 'cancelled before Kubernetes handoff'))
                self.store._cas(cursor, replace(ended, cleanup='complete'), current.revision)
                outbox.record_failure(self.store, message(current)['dispatch_id'], outbox.OBSOLETE,
                                      'cancelled before handoff', cursor=cursor)

    def reconcile(self, run, bound):
        executor = self.executor(target_of(run))
        observed = executor.collect(bound['handle'])
        if bound['pod_uid'] and observed.state == 'created' and not json.loads(observed.evidence).get('pods'):
            from .executors import Observation
            observed = Observation('unknown', observed.evidence,
                detail='assigned Pod is missing and its termination evidence was not retained')
        expired = (time.time()-bound['created'] >= self.target['pending_deadline'] and not bound['pod_uid'])
        if expired and not run.cancel and run.state != 'terminal':
            run = self.store.request_cancel(run.owner_id, run.case_id, run.id,
                CancelRequest(identity(), time.time(), 'kubernetes', 'placement deadline exceeded: '+observed.detail))
        if observed.state == 'absent' and not run.cancel and run.state != 'terminal':
            observed = executor.submit(bound['handle'])
        if run.cancel and run.state != 'terminal':
            if not bound['pod_uid'] and observed.state != 'finished':
                # Attachment and cancellation take the same run lock. Re-read after the
                # cancellation commit: a Pod that won before Stop must use terminated recovery.
                with self.store._transaction() as cursor:
                    current = _run(self.store, run.id, cursor, lock=True)
                    latest = assignment(self.store, run.id, cursor=cursor)
                    unattached = (current.cancel and not latest['pod_uid'] and
                                  not current.tasks[0].attempts[-1].executor)
                if not unattached:
                    return
                observed = executor.cancel(bound['handle'])
                if observed.state == 'unknown':
                    self._report(run, observed.detail + '; cancellation pending, ownership retained')
                    return
                try:
                    from .kubernetes_retention import release_unattached
                    release_unattached(executor, bound['handle'])
                except (ValueError, OSError, KeyError, TypeError, RuntimeError) as error:
                    self._report(run, f'Pod evidence cleanup pending: {error}')
                    return
                observed = executor.discover(bound['handle'])
                if observed.state == 'absent':
                    return self._finish_unstarted(run, bound)
            # An attached agent reads the durable cancellation and reaps its children first.
            elif observed.state != 'finished' and time.time()-run.cancel.requested_at > 45:
                self._report(run, 'cancellation pending; waiting for descendant cleanup or the bounded Job deadline')
                return  # retain the Pod and its exit evidence; deletion is not proof of death
        if observed.state == 'finished':
            if bound['pod_uid'] and json.loads(observed.evidence).get('pod_uid') != bound['pod_uid']:
                self._report(run, 'termination evidence belongs to another Pod; ownership retained')
                return
            if run.state != 'terminal' or run.cleanup not in ('complete', 'not_required'):
                from .kubernetes_run_recovery import recover_terminated
                run = recover_terminated(self.store, run, bound, observed)
            if run.state == 'terminal' and run.cleanup in ('complete', 'not_required'):
                try:
                    from .kubernetes_retention import release_completed
                    release_completed(executor, bound['handle'], expected_uid=bound['pod_uid'])
                except (ValueError, OSError, KeyError, TypeError, RuntimeError) as error:
                    self._report(run, f'Pod evidence cleanup pending: {error}')
                    return
                self._report(run, '')
                self._remove_job(executor, bound)
                self._complete(run, bound, 'managed outcome and cleanup recorded')
                return
        if observed.state == 'unknown':
            self._report(run, observed.detail + '; ownership retained, no replacement launched')
        elif observed.state == 'created':
            self._report(run, observed.detail)
        elif observed.state == 'running' and run.report_error and run.report_error.startswith('Kubernetes: '):
            self._report(run, '')

    def _finish_unstarted(self, run, bound):
        from .runs import RunManager
        with self.store._transaction() as cursor:
            current = _run(self.store, run.id, cursor, lock=True)
            latest = assignment(self.store, run.id, cursor=cursor)
            if latest['pod_uid'] or current.tasks[0].attempts[-1].executor:
                return  # Attachment won; the agent must perform cleanup.
            reason = current.cancel.reason if current.cancel else 'Kubernetes launch cancelled'
            ended = RunManager._execution_progress(current, 'terminal', outcome=Outcome('cancelled', time.time(), None, reason))
            ended = self.store._cas(cursor, replace(ended, cleanup='complete', report_error=None), current.revision)
        self._complete(ended, bound, reason)

    def _remove_job(self, executor, bound):
        from .kubernetes_api import ApiError
        intent = json.loads(bound['handle'].identity_json)
        try:
            job = self.cluster.read_job(intent['name'])
            metadata = job['metadata']
            created = executor._recorded(intent, 'kubernetes_created.json') or {}
            if (any(metadata.get('labels', {}).get(key) != value for key, value in intent['labels'].items())
                    or (created.get('job_uid') and created['job_uid'] != metadata.get('uid'))):
                raise KubernetesRunError('refusing cleanup of a foreign Job')
            self.cluster.delete_job(intent['name'], uid=metadata['uid'])
        except ApiError as error:
            if not error.absent:
                raise

    def _complete(self, run, bound, detail):
        with self.store._transaction() as cursor:
            self.store._execute(cursor, """UPDATE eforecast_kubernetes_runs SET state='complete',updated=?,detail=?
                WHERE namespace=? AND run_id=?""", (time.time(), detail, self.store.namespace, run.id))
            self.store._execute(cursor, """UPDATE eforecast_dispatch_outbox SET state=?,failure=?,updated_at=?
                WHERE namespace=? AND dispatch_id=? AND state=?""",
                (outbox.COMPLETE, None,
                 time.time(), self.store.namespace, bound['dispatch_id'], outbox.CLAIMED))
