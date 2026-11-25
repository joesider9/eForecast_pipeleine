# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""A minimal namespace-scoped Kubernetes client for worker execution and evidence retention.

Written against the standard library on purpose. The official client is a large dependency, and
the lab's cluster decision is deliberately still open (A16-05) -- adding it to the interpreter
that runs production forecasting, for a cluster that may never be installed, is a cost with no
benefit yet. What we need is five REST endpoints on a stable, versioned API:

    create a Job, read a Job, delete a Job, list the Pods of a Job, read a Pod.

Two deliberate absences:

* **No retry on create.** A retried create whose first response was merely lost would be a second
  Job. Instead the Job's name is derived from the attempt, so a repeated create is answered with
  ``409 AlreadyExists`` -- which is not an error but the evidence that our own earlier create
  succeeded. That is how a lost create response is recovered rather than duplicated.
* **No way to disable TLS verification.** A worker executor talks to the API server with a
  ServiceAccount token; a token sent to an unverified endpoint is a token given away.
"""
from __future__ import annotations

import json
import os
import socket
import ssl
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

SERVICE_ACCOUNT = Path('/var/run/secrets/kubernetes.io/serviceaccount')
#: Namespace-scoped, so the executor's credential cannot reach the rest of the cluster.
BATCH = '/apis/batch/v1/namespaces/{namespace}/jobs'
CORE = '/api/v1/namespaces/{namespace}/pods'
TIMEOUT = 15.0
MAX_RESPONSE_BYTES = 8 * 1024 * 1024


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        # Never forward a ServiceAccount credential to a login or proxy redirect.
        return None


class ClusterUnavailable(RuntimeError):
    """The API server could not be reached or answered something unusable.

    Never conflated with "the Job is not there": an unreachable API server means *unknown*, and
    unknown must not become permission to launch the work again.
    """


class ApiError(RuntimeError):
    """The API server refused, with the status it gave."""

    def __init__(self, status, reason, detail=''):
        super().__init__(f'{status} {reason}: {detail}'.strip())
        self.status, self.reason, self.detail = status, reason, detail

    @property
    def already_exists(self):
        return self.status == 409 and self.reason == 'AlreadyExists'

    @property
    def absent(self):
        return self.status == 404


class Cluster:
    """One namespace on one API server, with one token."""

    def __init__(self, server, token, *, namespace, ca_file=None, timeout=TIMEOUT):
        if not isinstance(server, str) or not server.startswith('https://'):
            raise ClusterUnavailable('the API server must be an https:// endpoint')
        if not token or not isinstance(token, str):
            raise ClusterUnavailable('a namespace-scoped ServiceAccount token is required')
        if not namespace or not isinstance(namespace, str):
            raise ClusterUnavailable('an explicit namespace is required')
        self.server, self.namespace, self.timeout = server.rstrip('/'), namespace, timeout
        self._token = token
        context = ssl.create_default_context(cafile=ca_file) if ca_file else \
            ssl.create_default_context()
        context.check_hostname = True
        context.verify_mode = ssl.CERT_REQUIRED
        self._context = context
        self._opener = urllib.request.build_opener(NoRedirect(), urllib.request.HTTPSHandler(context=context))

    # ------------------------------------------------------------------ construction

    @classmethod
    def in_cluster(cls, *, timeout=TIMEOUT):
        """The credential a Pod is given, which is the only one a worker executor should have."""
        try:
            token = (SERVICE_ACCOUNT / 'token').read_text().strip()
            namespace = (SERVICE_ACCOUNT / 'namespace').read_text().strip()
        except OSError as error:
            raise ClusterUnavailable(f'no in-cluster ServiceAccount: {error}') from None
        host = os.environ.get('KUBERNETES_SERVICE_HOST')
        port = os.environ.get('KUBERNETES_SERVICE_PORT', '443')
        if not host:
            raise ClusterUnavailable('KUBERNETES_SERVICE_HOST is not set; this is not a Pod')
        return cls(f'https://{host}:{port}', token, namespace=namespace,
                   ca_file=str(SERVICE_ACCOUNT / 'ca.crt'), timeout=timeout)

    @classmethod
    def configured(cls, *, timeout=TIMEOUT):
        """An explicitly configured endpoint, for a coordinator that runs outside the cluster.

        Deliberately not kubeconfig: that file can carry client certificates and exec plugins
        that fetch credentials by running programs, which is a much larger trust surface than a
        pilot needs. Three values, named, from the operator's own environment.
        """
        server = os.environ.get('EFORECAST_KUBE_SERVER')
        namespace = os.environ.get('EFORECAST_KUBE_NAMESPACE')
        ca_file = os.environ.get('EFORECAST_KUBE_CA') or None
        token = None
        reference = os.environ.get('EFORECAST_KUBE_TOKEN_FILE')
        if reference:
            try:
                token = Path(reference).read_text().strip()
            except OSError as error:
                raise ClusterUnavailable(f'token file unreadable: {error}') from None
        else:
            from . import secrets_source
            try:
                token = secrets_source.resolve('kube_token').reveal()
            except secrets_source.SecretMissing:
                token = None
        if not (server and namespace and token):
            raise ClusterUnavailable(
                'set EFORECAST_KUBE_SERVER, EFORECAST_KUBE_NAMESPACE and either '
                'EFORECAST_KUBE_TOKEN_FILE or the `kube_token` secret '
                '(bin/eforecast secret set kube_token)')
        return cls(server, token, namespace=namespace, ca_file=ca_file, timeout=timeout)

    @classmethod
    def available(cls, *, timeout=TIMEOUT):
        """The in-cluster credential if we are a Pod, otherwise a configured one, or None."""
        for build in (cls.in_cluster, cls.configured):
            try:
                return build(timeout=timeout)
            except ClusterUnavailable:
                continue
        return None

    # ------------------------------------------------------------------ transport

    def request(self, method, path, body=None, *, query=None, content_type='application/json'):
        url = self.server + path.format(namespace=self.namespace)
        if query:
            url += '?' + urllib.parse.urlencode(query)
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(url, data=data, method=method)
        request.add_header('Authorization', 'Bearer ' + self._token)
        request.add_header('Accept', 'application/json')
        if data is not None:
            request.add_header('Content-Type', content_type)
        try:
            with self._opener.open(request, timeout=self.timeout) as answer:
                raw = answer.read(MAX_RESPONSE_BYTES + 1)
            if len(raw) > MAX_RESPONSE_BYTES:
                raise ValueError('API response exceeds its size bound')
            document = json.loads(raw or b'{}')
            if not isinstance(document, dict):
                raise ValueError('API response is not an object')
            return document
        except urllib.error.HTTPError as error:
            raw = error.read(MAX_RESPONSE_BYTES + 1)
            try:
                document = json.loads(raw or b'{}')
            except ValueError:
                document = {}
            if not isinstance(document, dict):
                document = {}
            raise ApiError(error.code, document.get('reason', ''),
                           document.get('message', '')) from None
        except (urllib.error.URLError, socket.timeout, ssl.SSLError, OSError,
                ValueError) as error:
            # Unreachable, not absent. The caller must treat this as unknown.
            raise ClusterUnavailable(f'API server unreachable: {error}') from None

    # ------------------------------------------------------------------ the five calls

    def create_job(self, manifest):
        return self.request('POST', BATCH, manifest)

    def read_job(self, name):
        return self.request('GET', BATCH + '/' + name)

    def delete_job(self, name, *, grace_seconds=30, uid=None):
        # Foreground propagation so the Job is gone only once its Pods are, which is what makes
        # "cancelled and cleaned up" an observation rather than a hope.
        return self.request('DELETE', BATCH + '/' + name, {
            'apiVersion': 'meta/v1', 'kind': 'DeleteOptions',
            'propagationPolicy': 'Foreground', 'gracePeriodSeconds': int(grace_seconds),
            **({'preconditions': {'uid': uid}} if uid else {})})

    def list_pods(self, selector):
        return self.request('GET', CORE, query={'labelSelector': selector})

    def read_pod(self, name):
        return self.request('GET', CORE + '/' + name)

    def release_pod_finalizer(self, name, *, uid, resource_version, finalizers, finalizer):
        """Remove only our key, conditional on the exact object and observed revision."""
        if not uid or not resource_version or finalizer not in finalizers:
            raise ValueError('finalizer release needs exact Pod identity and revision')
        return self.request('PATCH', CORE + '/' + name, [
            {'op': 'test', 'path': '/metadata/uid', 'value': uid},
            {'op': 'test', 'path': '/metadata/resourceVersion', 'value': resource_version},
            {'op': 'test', 'path': '/metadata/finalizers', 'value': finalizers},
            {'op': 'remove', 'path': '/metadata/finalizers/' + str(finalizers.index(finalizer))},
        ], content_type='application/json-patch+json')

    def version(self):
        """Used only to record what the pilot ran against."""
        return self.request('GET', '/version')
