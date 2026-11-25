# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Keep managed Pod evidence until durable outcome/cleanup, including coordinator downtime.

The finalizer retains the API object, not execution: kubelet still stops its containers.
No timeout removes it without evidence. The existing bounded assignment queue retains the
slot until reconciliation completes. Workers hold no Kubernetes credential.
"""
import json

from .kubernetes_api import ApiError

FINALIZER = 'eforecast.io/termination-evidence'
CONTRACT = 'pod-finalizer-v1'


def owned_pod(pod, intent, job_uid):
    meta = pod.get('metadata') or {}
    if (not job_uid or not meta.get('uid') or not meta.get('name') or
            any(meta.get('labels', {}).get(k) != v for k, v in intent['labels'].items()) or
            not any(ref.get('uid') == job_uid and ref.get('name') == intent['name'] and
                    ref.get('kind') == 'Job' and ref.get('apiVersion') == 'batch/v1' and
                    ref.get('controller') is True for ref in meta.get('ownerReferences', []))):
        raise ValueError('Pod identity, labels or owning Job differ from the durable intent')
    return meta


def terminated_worker(pod):
    """A terminal phase alone (e.g. node loss) is not a container termination receipt."""
    spec, status = pod.get('spec') or {}, pod.get('status') or {}
    if (status.get('phase') not in ('Succeeded', 'Failed') or
            [c.get('name') for c in spec.get('containers', [])] != ['worker'] or
            spec.get('initContainers') or spec.get('ephemeralContainers')):
        raise ValueError('managed Pod must have exactly one terminated worker and no sidecars')
    containers = status.get('containerStatuses') or []
    if len(containers) != 1 or containers[0].get('name') != 'worker':
        raise ValueError('managed Pod lacks the exact worker exit status')
    value = containers[0].get('state', {}).get('terminated') or {}
    if type(value.get('exitCode')) is not int or not value.get('finishedAt'):
        raise ValueError('managed worker termination is unverified')
    return value


def _release(cluster, pod):
    meta = pod['metadata'];finalizers = meta.get('finalizers') or []
    if FINALIZER in finalizers:
        cluster.release_pod_finalizer(meta['name'], uid=meta['uid'],
            resource_version=meta.get('resourceVersion'), finalizers=finalizers, finalizer=FINALIZER)


def release_completed(executor, handle, *, expected_uid=None):
    """Caller has committed terminal outcome and cleanup; a durable receipt is also required."""
    intent = json.loads(handle.identity_json)
    if intent.get('termination_retention') != CONTRACT:
        return  # Legacy intents retain their reading; never rewrite accepted manifests.
    saved = executor._recorded(intent, 'kubernetes_completion.json')
    if saved is None:
        raise ValueError('retain Pod evidence: termination receipt has not been persisted')
    observed = executor.discover(handle)  # validates token, intent digest and terminal shape
    if observed.state != 'finished':
        raise ValueError('retain Pod evidence: ' + observed.detail)
    evidence = json.loads(observed.evidence)
    if expected_uid and evidence.get('pod_uid') != expected_uid:
        raise ValueError('termination receipt does not name the assigned Pod')
    if not evidence.get('pod_name') or not evidence.get('pod_uid'):
        raise ValueError('termination receipt lacks exact Pod identity')
    try:
        pod = executor.cluster.read_pod(evidence['pod_name'])
    except ApiError as error:
        if error.absent:
            return  # A previously persisted exit survives a crash after finalizer removal.
        raise
    meta = owned_pod(pod, intent, evidence['job_uid'])
    if meta['uid'] != evidence['pod_uid']:
        raise ValueError('Pod name now belongs to another UID; evidence retained')
    terminated_worker(pod)
    _release(executor.cluster, pod)


def release_unattached(executor, handle):
    """Only after run-lock proof that durable cancellation closed all attachment authority.

No root executor or Pod binding may exist. This is not death inferred from a missing Pod:
no worker ever acquired permission to execute. Deletion is requested before releasing keys,
so the Job cannot create new children after this bounded inventory.
"""
    intent = json.loads(handle.identity_json)
    if intent.get('termination_retention') != CONTRACT:
        return
    created = executor._recorded(intent, 'kubernetes_created.json') or {}
    job_uid = created.get('job_uid')
    try:
        job = executor.cluster.read_job(intent['name'])
        meta = job['metadata']
        if (any(meta.get('labels', {}).get(k) != v for k, v in intent['labels'].items()) or
                (job_uid and meta.get('uid') != job_uid) or not meta.get('deletionTimestamp')):
            raise ValueError('unattached Job deletion is not confirmed; evidence retained')
        job_uid = meta['uid']
    except ApiError as error:
        if not error.absent:
            raise
    pods = executor.cluster.list_pods(intent['selector']).get('items') or []
    if len(pods) > 32:
        raise ValueError('unattached Pod inventory exceeds the cleanup bound')
    for pod in pods:
        owned_pod(pod, intent, job_uid)
    for pod in pods:
        _release(executor.cluster, pod)
