# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""A16-05: the Kubernetes worker executor, behind the same interface as the local and Docker ones.

Kubernetes places Pods. It does not decide whether model work may start -- A08 does, on the
physical host, after placement has revealed which node and which card the Pod actually got. So
the Job this executor creates does **not** launch a model: it launches
``eforecast.kubernetes_bootstrap``, which resolves the real device, acquires the matching A08
grant, and only then execs the neutral worker. Pod bootstrap is coordination, not permission.

Three things this executor refuses to assume:

* **That ``parallelism: 1`` means once.** Kubernetes documents that the same program may start
  twice even then, so exactly-once is not bought from the platform. The Job is created with
  ``backoffLimit: 0`` and ``restartPolicy: Never`` to stop the platform retrying behind our back,
  and discovery treats *two started Pods* as `unknown` rather than picking a winner. The thing
  that actually makes one attempt run once is the A08 grant the bootstrap must hold.
* **That a missing Job means nothing ran.** A Job that we recorded creating and now cannot find
  is loss, not permission to run again -- the same rule the Docker adapter follows.
* **That a succeeded Job means verified work.** A Job's success is a process exit code. Receipts
  and fences still decide acceptance, exactly as for a local process.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import asdict
from pathlib import Path

from eforecast.config.executors import (Observation, atomic_json, digest, location, locked,
                                        same_location)
from eforecast.config.kubernetes_api import (ApiError, Cluster, ClusterUnavailable,
                                             CORE as CORE_PODS)
from eforecast.config.run_model import ExecutorHandle, identity, pinned

#: Ownership is expressed in labels so discovery never depends on a name we might reuse, and so
#: an operator can see whose work a Pod is with `kubectl get pods -L`.
LABEL_RUN = 'eforecast.io/run'
LABEL_TASK = 'eforecast.io/task'
LABEL_ATTEMPT = 'eforecast.io/attempt'
LABEL_EXECUTOR = 'eforecast.io/executor'
LABEL_COMPONENT = 'eforecast.io/component'

#: Canonical paths inside every worker container, so an intent is readable on any node.
INPUTS = '/eforecast/inputs'          # immutable accepted inputs, read-only
SCRATCH = '/eforecast/scratch'        # owned, quota'd, per-attempt
RESULTS = '/eforecast/results'        # lab persistent storage, published through A16-03 staging

#: Bounded queueing: a pilot that can enqueue unlimited Jobs has moved the backlog from our
#: durable state, where A09/A12 can reason about it, into the cluster, where they cannot.
MAX_QUEUED_JOBS = 32
DEFAULT_ACTIVE_DEADLINE = 6 * 3600
DEFAULT_PENDING_DEADLINE = 600

_NAME = re.compile(r'^[a-z0-9]([-a-z0-9]*[a-z0-9])?$')
#: `repository@sha256:<64 hex>`. A tag is a label the registry may move between the moment a
#: Job is recorded and the moment a node pulls it, so it is not an identity (R27-13).
_IMMUTABLE_IMAGE = re.compile(r'^[^@\s]+@sha256:[0-9a-f]{64}$')


def immutable_image(reference, *, allow_tag=False):
    """`(reference, pinned)` for a launch intent, or a refusal naming how to pin it.

    Resolving a tag is deliberately not done here. The release machinery already records which
    image a runtime is (`eforecast.image_release`), and normalising a tag against the witnessed
    release is an explicit step its caller takes before planning the Job -- so the intent that
    reaches the cluster says exactly which bytes it meant, and this module never guesses.

    `allow_tag` is the legacy path, for a cluster with no registry to pin against: a
    side-loaded k3s pilot image has no manifest digest to name. It is a declared choice of the
    caller, and the intent records `image_pinned: false`, so a launch that used a label rather
    than an identity says so in the durable record instead of looking the same as one that did.
    """
    if not reference:
        raise ValueError('a worker Job needs an explicitly pinned image')
    if _IMMUTABLE_IMAGE.match(str(reference)):
        return str(reference), True
    if allow_tag:
        return str(reference), False
    raise ValueError(
        f'{reference!r} is a tag, not an image identity: a registry may move it between '
        'recording this Job and pulling it. Pass repository@sha256:<digest>, resolving the '
        'tag first (eforecast.image_release records the digest each runtime was witnessed '
        'at), or construct the executor with allow_mutable_image=True to declare that this '
        'cluster has no registry to pin against.')


def job_name(run_id, task_id, attempt_id):
    """A deterministic, DNS-safe name derived from the attempt.

    Deterministic on purpose: a create whose response was lost is answered `409 AlreadyExists`
    on the next try, which is evidence our own earlier create succeeded rather than a reason to
    make a second Job. The digest keeps it inside 63 characters whatever the ids look like.
    """
    token = digest({'run': run_id, 'task': task_id, 'attempt': attempt_id})[:32]
    name = f'eforecast-{token}'
    if not _NAME.match(name) or len(name) > 63:
        raise ValueError('derived Job name is not a valid DNS label')
    return name


def label_value(value):
    """Labels are matched exactly, so a value that cannot be a label must not be silently cut."""
    if not isinstance(value, str) or len(value) > 63 or not re.match(
            r'^[A-Za-z0-9]([-A-Za-z0-9_.]*[A-Za-z0-9])?$', value):
        raise ValueError(f'{value!r} cannot be a Kubernetes label value')
    return value


def wait_reason(pod):
    """Why a Pod has not started yet, in words an operator can act on (A14).

    The three waits this pilot actually produces are an unschedulable Pod, an image that cannot
    be pulled, and a volume that cannot be attached or mounted. Each needs a different action,
    so each says which it is rather than "pending".
    """
    status = pod.get('status') or {}
    for condition in status.get('conditions') or []:
        if condition.get('type') == 'PodScheduled' and condition.get('status') == 'False':
            reason = condition.get('reason') or 'Unschedulable'
            return (f'no node can take this Pod ({reason}): '
                    f'{condition.get("message") or "no detail given"}')
    for container in (status.get('containerStatuses') or []) + \
                     (status.get('initContainerStatuses') or []):
        waiting = (container.get('state') or {}).get('waiting')
        if not waiting:
            continue
        reason = waiting.get('reason') or ''
        if 'Image' in reason:
            return (f'the worker image cannot be pulled ({reason}): '
                    f'{waiting.get("message") or "check the registry and its credential"}')
        if reason:
            return f'container is waiting ({reason}): {waiting.get("message") or ""}'.strip()
    for event in status.get('_events') or []:                       # supplied by callers if read
        if event.get('reason') in {'FailedAttachVolume', 'FailedMount'}:
            return f'storage is not available ({event["reason"]}): {event.get("message") or ""}'
    phase = status.get('phase') or 'Pending'
    return f'Pod is {phase.lower()} and has not started a container yet'


class KubernetesExecutor:
    """Submit/discover/status/cancel/collect for a Kubernetes Job, one Job per attempt."""

    adapter = 'kubernetes_job'

    def __init__(self, cluster=None, directory=None, *, image=None, service_account=None,
                 workspace=None, allow_mutable_image=False):
        from eforecast.config.executors import LocalExecutor
        self.cluster = cluster
        self.local = LocalExecutor(directory, workspace=workspace)
        self.image = image
        self.service_account = service_account or 'eforecast-worker'
        #: A cluster with no registry to pin against -- the side-loaded k3s pilot -- declares
        #: this, and every intent it plans records that its image was a label (R27-13).
        self.allow_mutable_image = bool(allow_mutable_image)

    # ------------------------------------------------------------------ intent

    def plan_job(self, *, run_id, task_id, attempt_id, argv, environment, resources,
                 volumes, image=None, active_deadline=DEFAULT_ACTIVE_DEADLINE,
                 pending_deadline=DEFAULT_PENDING_DEADLINE, node_selector=None,
                 uid=None, gid=None):
        """Record the launch intent, and the exact manifest, before anything is created.

        The handle is immutable: the manifest it carries is what `submit` sends, so a create that
        is retried after a lost response sends byte-identical content under the same name.
        """
        name = job_name(run_id, task_id, attempt_id)
        token = identity()
        labels = {LABEL_RUN: label_value(run_id), LABEL_TASK: label_value(task_id),
                  LABEL_ATTEMPT: label_value(attempt_id), LABEL_EXECUTOR: label_value(token),
                  LABEL_COMPONENT: 'worker'}
        folder = self.local.directory / token
        folder.mkdir(parents=True, mode=0o700)
        manifest = self.manifest(
            name=name, labels=labels, argv=argv, environment=environment, resources=resources,
            volumes=volumes, image=image or self.image, active_deadline=active_deadline,
            node_selector=node_selector, uid=uid, gid=gid)
        atomic_json(folder / 'kubernetes.json', {'manifest': manifest})
        _, pinned_image = immutable_image(image or self.image,
                                          allow_tag=self.allow_mutable_image)
        return ExecutorHandle(token, self.adapter, pinned({
            'version': 1, 'directory': str(folder), 'location': location(), 'name': name,
            'image': str(image or self.image), 'image_pinned': pinned_image,
            'labels': labels, 'namespace': self._namespace(), 'selector': self._selector(labels),
            'server': getattr(self.cluster, 'server', None),
            'pending_deadline': float(pending_deadline),
            'active_deadline': float(active_deadline)}))

    def manifest(self, *, name, labels, argv, environment, resources, volumes, image,
                 active_deadline, node_selector=None, uid=None, gid=None):
        """The Job, with every platform retry turned off and every path canonical."""
        image, _ = immutable_image(image, allow_tag=self.allow_mutable_image)
        gpus = int(resources.get('gpu_slots') or 0)
        requests = {'cpu': str(resources['cpu_threads']),
                    'memory': f"{int(resources['ram_mb'])}Mi"}
        if gpus:
            # Whole cards only in the pilot; sharing and multi-node claims are A16-07.
            requests['nvidia.com/gpu'] = str(gpus)
        container = {
            'name': 'worker',
            'image': image,
            'imagePullPolicy': 'IfNotPresent',
            'command': list(argv),
            'env': [{'name': key, 'value': str(value)} for key, value in
                    sorted(environment.items())] + [
                # Placement facts the bootstrap needs, from the API rather than from guesswork.
                {'name': 'EFORECAST_NODE_NAME',
                 'valueFrom': {'fieldRef': {'fieldPath': 'spec.nodeName'}}},
                {'name': 'EFORECAST_POD_NAME',
                 'valueFrom': {'fieldRef': {'fieldPath': 'metadata.name'}}},
                {'name': 'EFORECAST_POD_UID',
                 'valueFrom': {'fieldRef': {'fieldPath': 'metadata.uid'}}}],
            # requests == limits: the kubelet's view of this Pod and A08's grant for it are the
            # same numbers, so capacity is not counted twice and not quietly exceeded.
            'resources': {'requests': dict(requests), 'limits': dict(requests)},
            # `sub_path` mounts one file from a volume, which is how a pilot delivers a single
            # module into the image's package without replacing the directory around it.
            'volumeMounts': [{k: v for k, v in (
                ('name', entry['name']), ('mountPath', entry['path']),
                ('readOnly', bool(entry.get('read_only'))),
                ('subPath', entry.get('sub_path'))) if v is not None} for entry in volumes],
            'securityContext': {
                'allowPrivilegeEscalation': False,
                'readOnlyRootFilesystem': False,        # torch compiles into its own cache dirs
                'capabilities': {'drop': ['ALL']}},
        }
        spec = {
            'serviceAccountName': self.service_account,
            'restartPolicy': 'Never',                   # no hidden restart loop
            'containers': [container],
            # One volume per distinct name: several mounts may share a volume via sub_path.
            'volumes': list({entry['name']: entry['volume'] | {'name': entry['name']}
                             for entry in volumes}.values()),
            'securityContext': {k: v for k, v in (
                ('runAsUser', uid), ('runAsGroup', gid), ('fsGroup', gid)) if v is not None},
            'automountServiceAccountToken': False,      # the worker is not a cluster client
            'enableServiceLinks': False,
        }
        if node_selector:
            spec['nodeSelector'] = dict(node_selector)
        return {
            'apiVersion': 'batch/v1', 'kind': 'Job',
            'metadata': {'name': name, 'labels': dict(labels)},
            'spec': {
                'parallelism': 1, 'completions': 1,
                # Zero platform retries: this application owns retry. A replacement Pod would be
                # a second attempt with the predecessor's identity, which the fence must refuse
                # anyway -- better that it never exists.
                'backoffLimit': 0,
                'activeDeadlineSeconds': int(active_deadline),
                'ttlSecondsAfterFinished': None,        # we delete after collecting the receipt
                'template': {'metadata': {'labels': dict(labels)}, 'spec': spec}},
        }

    # ------------------------------------------------------------------ helpers

    def _namespace(self):
        return self.cluster.namespace if self.cluster is not None else ''

    @staticmethod
    def _selector(labels):
        return ','.join(f'{key}={labels[key]}' for key in
                        (LABEL_RUN, LABEL_TASK, LABEL_ATTEMPT, LABEL_EXECUTOR))

    def _connected(self, intent):
        if self.cluster is None:
            return Observation('unknown', detail='no Kubernetes cluster is configured')
        if intent.get('namespace') and intent['namespace'] != self.cluster.namespace:
            return Observation('unknown',
                               detail='this intent belongs to another namespace')
        if intent.get('server') and intent['server'] != getattr(self.cluster, 'server', None):
            return Observation('unknown', detail='this intent belongs to another Kubernetes API server')
        return None

    @staticmethod
    def _recorded(intent, name):
        path = Path(intent['directory']) / name
        return json.loads(path.read_text()) if path.exists() else None

    # ------------------------------------------------------------------ discovery

    def discover(self, handle):
        intent = json.loads(handle.identity_json)
        try:
            saved = self._recorded(intent, 'kubernetes_completion.json')
            if saved is not None:
                if saved.pop('token') != handle.id or saved.pop('intent_digest') != digest(intent):
                    raise ValueError('foreign Kubernetes completion receipt')
                observation = Observation(**saved)
                if (observation.state != 'finished' or type(observation.returncode) is not int or
                        type(observation.finished_at) not in (float, int) or
                        not math.isfinite(observation.finished_at) or observation.finished_at <= 0):
                    raise ValueError('invalid Kubernetes completion receipt')
                return observation
            return self._inspect(handle, intent)
        except (OSError, ValueError, TypeError, KeyError) as error:
            return Observation('unknown', detail=f'Kubernetes receipt unavailable: {error}')

    def _inspect(self, handle, intent):
        refusal = self._connected(intent)
        if refusal is not None:
            return refusal
        job_missing = False
        try:
            job = self.cluster.read_job(intent['name'])
        except ApiError as error:
            if error.absent:
                if self._recorded(intent, 'kubernetes_cancelled.json'):
                    # We removed it ourselves. Reporting loss here would leave every cancelled
                    # attempt permanently `unknown`, which recovery would keep reopening.
                    return Observation('absent', detail='the Job was cancelled and removed')
                if self._recorded(intent, 'kubernetes_created.json'):
                    from .kubernetes_retention import CONTRACT
                    created = self._recorded(intent, 'kubernetes_created.json')
                    if intent.get('termination_retention') == CONTRACT and created.get('job_uid'):
                        # A background Job deletion may leave our finalizer-protected Pod.
                        # Only that exact owner's terminal container evidence can recover it.
                        job_missing = True
                        job = {'metadata': {'uid': created['job_uid'], 'labels': intent['labels']}}
                    else:
                        # Loss, not permission. Something removed a Job we know we created.
                        return Observation('unknown', detail='a previously created Job is missing')
                else:
                    return Observation('absent')
            else:
                return Observation('unknown', detail=f'Kubernetes refused: {error}')
        except ClusterUnavailable as error:
            return Observation('unknown', detail=f'cluster unavailable: {error}')

        uid = ((job.get('metadata') or {}).get('uid') or '')
        created = self._recorded(intent, 'kubernetes_created.json')
        if created and created.get('job_uid') and created['job_uid'] != uid:
            # The name is ours but the object is not: a different Job now holds it.
            return Observation('unknown', detail='the Job name belongs to a different object')
        labels = (job.get('metadata') or {}).get('labels') or {}
        if any(labels.get(key) != value for key, value in intent['labels'].items()):
            return Observation('unknown', detail='the Job carries foreign ownership labels')

        try:
            pods = (self.cluster.list_pods(intent['selector']) or {}).get('items') or []
        except ApiError as error:
            return Observation('unknown', detail=f'Pod discovery refused: {error}')
        except ClusterUnavailable as error:
            return Observation('unknown', detail=f'cluster unavailable: {error}')

        evidence = {'job_uid': uid, 'name': intent['name'], 'labels': intent['labels'],
                    'pods': sorted((pod.get('metadata') or {}).get('uid', '') for pod in pods)}
        if intent.get('termination_retention') and len(pods) > 1:
            return Observation('unknown', pinned(evidence),
                               detail='multiple Pods for one managed attempt; evidence retained')
        started = [pod for pod in pods
                   if (pod.get('status') or {}).get('phase') in
                   {'Running', 'Succeeded', 'Failed'}]
        if len(started) > 1:
            # Kubernetes allows this even at parallelism 1. We will not choose a winner: the A08
            # grant did, and whichever Pod does not hold it has produced nothing acceptable.
            return Observation('unknown', pinned(evidence),
                               detail=f'{len(started)} Pods started for one attempt; '
                                      'the A08 fence decides which, if any, holds authority')
        if not pods:
            if job_missing or any(c.get('type') in ('Complete', 'Failed') and c.get('status') == 'True'
                                  for c in job.get('status', {}).get('conditions', [])):
                return Observation('unknown', pinned(evidence),
                    detail='Job ended or disappeared without retained Pod termination evidence')
            return Observation('created', pinned(evidence),
                               detail='the Job exists and has not produced a Pod yet')
        pod = started[0] if started else pods[0]
        status = pod.get('status') or {}
        phase = status.get('phase')
        evidence['pod_uid'] = (pod.get('metadata') or {}).get('uid', '')
        evidence['pod_name'] = (pod.get('metadata') or {}).get('name', '')
        evidence['node'] = (pod.get('spec') or {}).get('nodeName') or ''
        if phase in {'Succeeded', 'Failed'}:
            from .kubernetes_retention import CONTRACT, owned_pod, terminated_worker
            if intent.get('termination_retention') == CONTRACT:
                try:
                    owned_pod(pod, intent, uid)
                    terminated_worker(pod)
                except ValueError as error:
                    return Observation('unknown', pinned(evidence), detail=str(error))
            terminated = next(((container.get('state') or {}).get('terminated')
                               for container in status.get('containerStatuses') or []
                               if (container.get('state') or {}).get('terminated')), None)
            if not terminated or type(terminated.get('exitCode')) is not int:
                return Observation('unknown', pinned(evidence),
                                   detail='the Pod ended without a container exit status')
            observed = _timestamp(terminated.get('finishedAt'))
            if observed is None:
                return Observation('unknown', pinned(evidence),
                                   detail='the Pod completion time is unverified')
            return Observation('finished', pinned(evidence), terminated['exitCode'], observed,
                               f'Kubernetes {phase}')
        if job_missing:
            return Observation('unknown', pinned(evidence), detail='Job disappeared but its Pod is not terminated')
        if phase == 'Running':
            return Observation('running', pinned(evidence))
        return Observation('created', pinned(evidence), detail=wait_reason(pod))

    # ------------------------------------------------------------------ the interface

    def submit(self, handle):
        intent = json.loads(handle.identity_json)
        if not same_location(intent):
            return Observation('unknown', detail='cannot launch from another execution location')
        refusal = self._connected(intent)
        if refusal is not None:
            return refusal
        try:
            with locked(intent['directory']) as descriptor:
                if descriptor is None:
                    return Observation('unknown', detail='Kubernetes dispatch already in progress')
                if (self._recorded(intent, 'kubernetes_cancel_intent.json') or
                        self._recorded(intent, 'kubernetes_cancelled.json')):
                    return Observation('absent', detail='Kubernetes launch authority was cancelled')
                observed = self._inspect(handle, intent)
                if observed.state != 'absent':
                    return observed
                if not self.accepting():
                    # Advisory until now: the cap was computed and nothing consulted it, so a
                    # full queue still got one more Job. Two dispatchers racing can still each
                    # pass this check -- settling that belongs at the durable claim, which is
                    # the cutover's job (R27-12/A16-08) -- but a single dispatcher no longer
                    # floods a queue it can see is full.
                    return Observation('absent', detail=(
                        f'the cluster already has {MAX_QUEUED_JOBS} unfinished Jobs of ours, '
                        'so this one was not created; it stays eligible and is launched when '
                        'the queue drains'))
                manifest = json.loads(
                    (Path(intent['directory']) / 'kubernetes.json').read_text())['manifest']
                if intent.get('manifest_digest') and digest(manifest) != intent['manifest_digest']:
                    raise ValueError('Kubernetes manifest differs from its durable intent')
                # The create is attempted once. If its response is lost, the next pass gets 409
                # and reads the Job that our own earlier create made.
                try:
                    job = self.cluster.create_job(manifest)
                except ApiError as error:
                    if not error.already_exists:
                        return Observation('unknown', detail=f'Job creation refused: {error}')
                    job = None
                except ClusterUnavailable as error:
                    # Unknown, not failed: a create may have been accepted. Record the attempt so
                    # a later `absent` is read as loss rather than as never-created.
                    atomic_json(Path(intent['directory']) / 'kubernetes_created.json',
                                {'token': handle.id, 'job_uid': ''})
                    return Observation('unknown',
                                       detail=f'Job creation unacknowledged: {error}')
                uid = ((job or {}).get('metadata') or {}).get('uid') or ''
                atomic_json(Path(intent['directory']) / 'kubernetes_created.json',
                            {'token': handle.id, 'job_uid': uid})
                return self._inspect(handle, intent)
        except (OSError, ValueError, KeyError) as error:
            return Observation('unknown', detail=f'Kubernetes launch unavailable: {error}')

    def status(self, handle):
        return self.discover(handle)

    def cancel(self, handle, *, force=False):
        intent = json.loads(handle.identity_json)
        refusal = self._connected(intent)
        if refusal is not None:
            return refusal
        try:
            with locked(intent['directory']) as descriptor:
                if descriptor is None:
                    return Observation('unknown', detail='Kubernetes dispatch in progress; cancellation pending')
                # Serialize against create, including the not-yet-created case. A stale
                # dispatcher must not create an orphan after the run has been retired.
                atomic_json(Path(intent['directory']) / 'kubernetes_cancel_intent.json',
                            {'token': handle.id, 'force': bool(force)})
                return self._cancel_locked(handle, intent, force=force)
        except (OSError, ValueError, KeyError) as error:
            return Observation('unknown', detail=f'Kubernetes cancellation unavailable: {error}')

    def _cancel_locked(self, handle, intent, *, force):
        observed = self.discover(handle)
        if observed.state in {'created', 'running', 'unknown'}:
            try:
                job = self.cluster.read_job(intent['name'])
                metadata = job.get('metadata') or {}
                created = self._recorded(intent, 'kubernetes_created.json') or {}
                if (not metadata.get('uid') or any((metadata.get('labels') or {}).get(k) != v
                        for k, v in intent['labels'].items()) or
                        (created.get('job_uid') and created['job_uid'] != metadata['uid'])):
                    return Observation('unknown', detail='refusing to cancel a Job with foreign ownership')
                # Launch cancellation is not death evidence. Only an exactly owned
                # deletion request gets the historical removal receipt.
                atomic_json(Path(intent['directory']) / 'kubernetes_cancelled.json',
                            {'token': handle.id, 'force': bool(force)})
                self.cluster.delete_job(intent['name'], grace_seconds=0 if force else 30,
                                        uid=metadata['uid'])
            except ApiError as error:
                if not error.absent:
                    return Observation('unknown', observed.evidence,
                                       detail=f'cancellation refused: {error}')
            except ClusterUnavailable as error:
                return Observation('unknown', observed.evidence,
                                   detail=f'cancellation unconfirmed: {error}')
        return self.discover(handle)

    def collect(self, handle):
        observed = self.discover(handle)
        if observed.state == 'finished':
            intent = json.loads(handle.identity_json)
            try:
                atomic_json(Path(intent['directory']) / 'kubernetes_completion.json',
                            {'token': handle.id, 'intent_digest': digest(intent),
                             **asdict(observed)})
            except OSError as error:
                return Observation('unknown', observed.evidence,
                                   detail=f'receipt publication failed: {error}')
        return observed

    # ------------------------------------------------------------------ bounds

    def queued(self):
        """How many of our Jobs are not finished, so a caller can stop before flooding."""
        if self.cluster is None:
            return 0
        try:
            job_list = self.cluster.request(
                'GET', '/apis/batch/v1/namespaces/{namespace}/jobs',
                query={'labelSelector': f'{LABEL_COMPONENT}=worker'})
        except (ApiError, ClusterUnavailable):
            return MAX_QUEUED_JOBS           # unknown is treated as full, never as room
        active = 0
        for job in (job_list or {}).get('items') or []:
            status = job.get('status') or {}
            if not (status.get('succeeded') or status.get('failed')):
                active += 1
        return active

    def accepting(self):
        return self.queued() < MAX_QUEUED_JOBS


def _timestamp(value):
    """A Kubernetes RFC3339 instant as a positive epoch float, or None if it is unusable."""
    if not isinstance(value, str) or not value:
        return None
    from datetime import datetime
    try:
        moment = datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp()
    except (ValueError, OSError, OverflowError):
        return None
    return moment if math.isfinite(moment) and moment > 0 else None


def pod_inspector(cluster):
    """A liveness inspector for A08, keyed by Pod UID rather than by PID.

    `liveness.container_liveness` expects `inspect(id)` to return `None` when the thing is gone,
    a running-ish state name when it is alive, `'unknown'` when the state cannot be read, and any
    other name when it is finished. Its default asks `docker inspect`, which knows nothing about
    a Pod -- so a Kubernetes attempt needs this one, passed through
    `liveness.probe(inspect=pod_inspector(cluster))`.

    Why not the PID: a Pod's process is numbered in the Pod's own namespace (PID 1, measured), so
    from the host it names init. A08 then answers "unknown" and holds the capacity forever, which
    is safe and useless. The Pod UID is the identity that outlives the Pod and that the cluster
    can still be asked about.
    """
    def inspect(pod_uid):
        try:
            # The Cluster object is already scoped to one namespace, and only our own worker
            # Pods are considered: a liveness answer about someone else's Pod is not an answer.
            found = cluster.request('GET', CORE_PODS,
                                    query={'labelSelector': f'{LABEL_COMPONENT}=worker'})
        except ClusterUnavailable as error:
            # Unreachable is not gone. Raising makes `container_liveness` answer 'unknown', so
            # the capacity stays held rather than being handed to someone else.
            raise RuntimeError(f'the cluster API could not be asked about {pod_uid}: {error}')
        for pod in (found or {}).get('items') or []:
            if ((pod.get('metadata') or {}).get('uid') or '') != pod_uid:
                continue
            phase = (pod.get('status') or {}).get('phase') or 'unknown'
            if phase in {'Pending', 'Running'}:
                return 'running'
            if phase in {'Succeeded', 'Failed'}:
                return 'exited'
            return 'unknown'
        # The cluster answered and does not have it: the Pod is gone, so the grant may be reaped.
        return None

    return inspect
