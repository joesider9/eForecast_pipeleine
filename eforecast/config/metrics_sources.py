# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Read-only, bounded adapters for the fleet endpoint's live infrastructure sources."""
import base64
import json
import math
import os
import time
import urllib.parse
import urllib.request

from . import dispatch_broker as broker
from .kubernetes_api import Cluster, BATCH, CORE


def broker_depths(endpoint, *, url=None, timeout=2):
    """Read queue totals through RabbitMQ management; credentials use the broker source.

    An explicit management endpoint is necessary: a container's mapped management
    port cannot be inferred from its AMQP port. Never put credentials in that URL.
    """
    target = urllib.parse.urlsplit(endpoint)
    if target.scheme not in ('http', 'https') or not target.hostname or target.username or target.password:
        raise ValueError('broker management endpoint must be an http(s) URL without credentials')
    address = urllib.parse.urlsplit(broker.broker_url(url))
    username = urllib.parse.unquote(address.username or '')
    password = urllib.parse.unquote(address.password or '')
    credential = base64.b64encode(f'{username}:{password}'.encode()).decode()
    vhost = urllib.parse.unquote(address.path[1:]) or '/'
    values = {}
    # A management API must not forward Authorization to a login/proxy redirect.
    opener = urllib.request.build_opener(NoRedirect())
    for route in broker.ROUTES:
        path = '/api/queues/' + urllib.parse.quote(vhost, safe='') + '/' + urllib.parse.quote(broker.queue_of(route), safe='')
        request = urllib.request.Request(endpoint.rstrip('/') + path,
                                         headers={'Authorization': 'Basic ' + credential, 'Accept': 'application/json'})
        try:
            with opener.open(request, timeout=timeout) as response:
                raw = response.read(65537)
            if len(raw) > 65536:
                raise ValueError('queue observation exceeds its size bound')
            document = json.loads(raw)
            counts = (document['messages'], document['messages_unacknowledged'])
            if any(type(n) is not int or n < 0 for n in counts):
                raise ValueError('queue counts are not nonnegative integers')
        except Exception as error:
            raise RuntimeError(f'broker queue observation failed ({type(error).__name__})') from None
        values[route] = counts
    return values


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        return None


def cluster_counts(cluster, *, now=None, limit=500):
    """Count unfinished managed Jobs and the oldest Pending Pod in one namespace."""
    from .kubernetes_executor import _timestamp
    query = {'labelSelector': 'eforecast.io/component=worker', 'limit': limit}
    jobs = cluster.request('GET', BATCH, query=query)
    pods = cluster.request('GET', CORE, query=query)
    for document in (jobs, pods):
        if (not isinstance(document, dict) or not isinstance(document.get('items'), list)
                or len(document['items']) > limit or (document.get('metadata') or {}).get('continue')):
            raise ValueError('cluster observation is incomplete; partial counts withheld')
    active = sum(not any(c.get('type') in ('Complete', 'Failed') and c.get('status') == 'True'
                         for c in (job.get('status') or {}).get('conditions', [])) for job in jobs['items'])
    longest = 0.0
    moment = time.time() if now is None else now
    for pod in pods['items']:
        if (pod.get('status') or {}).get('phase') == 'Pending':
            created = _timestamp((pod.get('metadata') or {}).get('creationTimestamp'))
            if created is None or not math.isfinite(created):
                raise ValueError('pending Pod has no valid creation timestamp')
            longest = max(longest, moment - created)
    return active, longest


def configured():
    sources = {}
    endpoint = os.environ.get('EFORECAST_BROKER_MANAGEMENT_URL')
    if endpoint:
        sources['broker_depths'] = lambda: broker_depths(endpoint)
    cluster = Cluster.available(timeout=2)
    if cluster is not None:
        sources['cluster'] = lambda: cluster_counts(cluster)
    return sources
