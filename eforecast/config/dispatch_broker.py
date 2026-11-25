# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""A16-04: the RabbitMQ half of dispatch -- a confirmed publisher and a short dispatcher.

The durable record is the outbox in :mod:`eforecast.config.dispatch_outbox`, written in the same
transaction that made a task eligible. Nothing here is allowed to be the only record of
unfinished work: the broker is a transport, and after queue loss the store is reconciled against
it (``dispatch_outbox.reconcile``) rather than the other way round.

Two boundaries, protected by two different mechanisms:

* **Publisher confirms** tell us the broker accepted and persisted a message. A crash before the
  publish leaves the row ``pending``, which the next drain republishes under the same derived
  identity. A crash after acceptance but before ``mark_published`` also leaves it ``pending``, so
  the message may be delivered twice -- which is why the consumer is idempotent rather than
  careful.
* **Consumer acknowledgements** tell the broker we are done with a delivery. The dispatcher
  commits its claim to PostgreSQL *before* it acknowledges, so a crash between the two redelivers
  a dispatch that is already claimed, and the claim says so. It never holds a delivery open for
  the hours a fitting takes: it records the durable launch intent and acknowledges, and from that
  commit on A09 recovery owns the unfinished launch.

The queue is a quorum queue so that a message survives a broker restart and, deployed across
members, a lost one. A single-host pilot demonstrates persistence, not availability.
"""
from __future__ import annotations

import json
import os
import socket
import time

from . import dispatch_outbox as outbox

EXCHANGE = 'eforecast.dispatch'
# One queue per execution venue, not per case or per trial: a queue per case would make the
# broker's topology a function of user data, and fairness between cases is A09/A12's decision
# over the outbox, not something queue ordering is allowed to express.
ROUTES = ('local', 'remote', 'managed')
QUEUE = 'eforecast.dispatch.{route}'
# A bounded drain and a bounded prefetch. The publisher is not a backlog pump and the dispatcher
# holds no more deliveries than it can claim promptly.
DRAIN_LIMIT = 64
PREFETCH = 8
ENVIRONMENT = 'EFORECAST_BROKER_URL'


class BrokerUnavailable(RuntimeError):
    """The broker could not be reached. The dispatch stays visible as a wait, never as a loss."""


def route_of(document) -> str:
    route = ('managed' if document['target'] == 'kubernetes-run' else
             document['target'] if document['target'] in ROUTES else 'remote')
    return route


def queue_of(route: str) -> str:
    if route not in ROUTES:
        raise outbox.OutboxError(f'{route!r} is not a configured dispatch route')
    return QUEUE.format(route=route)


def broker_url(url=None) -> str:
    """The broker URL, from the caller, the environment, or the operator's secret store.

    Never from a message and never from a run: a dispatch says what to run, not where to connect.
    """
    if url:
        return url
    from . import secrets_source
    try:
        # A16-10's source, which already reads the environment variable first, so adopting the
        # secret store never required moving a working deployment's URL on the same day.
        return secrets_source.resolve('broker_url', environment=ENVIRONMENT).reveal()
    except secrets_source.SecretMissing as error:
        raise BrokerUnavailable(
            f'no broker configured: set {ENVIRONMENT}, or store the secret `broker_url` '
            f'with `bin/eforecast secret set broker_url`.\n{error}') from None


def _pika():
    try:
        import pika
    except ImportError as error:                                        # pragma: no cover
        raise BrokerUnavailable('pika is not installed in this interpreter') from error
    return pika


def declare(channel) -> None:
    """The durable topology. Idempotent, and asserted by both sides before they use it."""
    pika = _pika()
    channel.exchange_declare(exchange=EXCHANGE, exchange_type='direct', durable=True)
    for route in ROUTES:
        channel.queue_declare(queue=queue_of(route), durable=True,
                              arguments={'x-queue-type': 'quorum'})
        channel.queue_bind(queue=queue_of(route), exchange=EXCHANGE, routing_key=route)
    del pika


class Connection:
    """One blocking connection with publisher confirms turned on."""

    def __init__(self, url=None, *, timeout=10.0, name=None):
        self.url, self.timeout = broker_url(url), timeout
        self.name = name or f'eforecast@{socket.gethostname()}:{os.getpid()}'
        self.connection = self.channel = None
        self.published = 0

    def __enter__(self):
        pika = _pika()
        try:
            parameters = pika.URLParameters(self.url)
            parameters.socket_timeout = self.timeout
            parameters.blocked_connection_timeout = self.timeout
            parameters.client_properties = {'connection_name': self.name}
            self.connection = pika.BlockingConnection(parameters)
            self.channel = self.connection.channel()
            self.channel.confirm_delivery()
            declare(self.channel)
        except Exception as error:                      # pika raises a family of them
            self.close()
            raise BrokerUnavailable(f'broker unreachable ({type(error).__name__}); check the broker connection') from error
        return self

    def __exit__(self, *_):
        self.close()
        return False

    def close(self):
        for handle in (self.channel, self.connection):
            try:
                if handle is not None and handle.is_open:
                    handle.close()
            except Exception:                           # closing is best effort
                pass
        self.channel = self.connection = None

    # ------------------------------------------------------------------ publishing

    def publish(self, document) -> str:
        """Publish one validated dispatch, persistently, and return its confirmation token.

        Raises :class:`dispatch_outbox.OutboxError` for an unroutable message -- a dispatch for a
        venue this broker has no queue for is a classification, not a retry -- and
        :class:`BrokerUnavailable` for a nack or a lost connection, which is a wait.
        """
        pika = _pika()
        outbox.validate(document)
        route = route_of(document)
        body = outbox.canonical(document)
        properties = pika.BasicProperties(
            delivery_mode=2,                            # persistent, so a restart keeps it
            content_type='application/json',
            message_id=document['dispatch_id'],
            app_id='eforecast.dispatch',
            type=f'dispatch.v{outbox.VERSION}')
        try:
            self.channel.basic_publish(exchange=EXCHANGE, routing_key=route, body=body,
                                       properties=properties, mandatory=True)
        except pika.exceptions.UnroutableError as error:
            raise outbox.OutboxError(f'no queue is bound for route {route!r}') from error
        except Exception as error:
            raise BrokerUnavailable(f'publish was not confirmed ({type(error).__name__})') from error
        self.published += 1
        return f'{self.name}/{route}#{self.published}'


def drain(store, *, url=None, limit=DRAIN_LIMIT, connection=None, target=None):
    """Publish the pending outbox, bounded, recording each confirmation as it arrives.

    Ordering within the batch is the outbox's (oldest first), so a slow case cannot be starved by
    a busy one. A crash anywhere in this loop is safe: whatever was not marked published is
    republished next time under the same identity.
    """
    report = {'published': [], 'unroutable': [], 'waiting': None}
    rows = outbox.pending(store, limit=limit, target=target)
    if not rows:
        return report
    held = connection is None
    try:
        link = connection if connection is not None else Connection(url).__enter__()
    except BrokerUnavailable as error:
        report['waiting'] = str(error)                  # a visible wait, not a failed dispatch
        return report
    try:
        for row in rows:
            document = row['payload']
            try:
                token = link.publish(document)
            except outbox.OutboxError as error:
                outbox.record_failure(store, row['dispatch_id'], outbox.UNROUTABLE, str(error))
                report['unroutable'].append(row['dispatch_id'])
                continue
            except BrokerUnavailable as error:
                report['waiting'] = str(error)
                break
            outbox.mark_published(store, row['dispatch_id'], token)
            report['published'].append(row['dispatch_id'])
    finally:
        if held:
            link.close()
    return report


# ---------------------------------------------------------------------- consuming


def dispatch(store, *, url=None, limit=PREFETCH, prefetch=PREFETCH, timeout=5.0,
             recheck=None, launch=None, consumer=None, connection=None, routes=('local', 'remote')):
    """The short dispatcher: claim durably, then acknowledge. Bounded in messages and in time.

    ``recheck(store, document)`` re-reads the task the message names and returns a reason to
    refuse, or ``None`` to go ahead. Cancellation, a superseded generation, an unobserved
    dependency and an unsupported runtime are all reasons; they are recorded no-ops, never
    retries, because redelivering them would not change the answer.

    ``launch(store, document)`` records the durable launch intent *inside* the claim, so the
    acknowledgement that follows can never be the thing that owned the work. It must not start a
    process, hold a GPU or fit anything: A08 admits resources at execution on the real host.
    """
    if not callable(launch):
        raise outbox.OutboxError('dispatcher has no durable launch adapter; leave deliveries queued '
                                'until the coordinator handoff is configured')
    pika = _pika()
    import hashlib
    host = hashlib.sha256(socket.gethostname().encode()).hexdigest()[:12]
    consumer = consumer or f'eforecast-{host}-{os.getpid()}'
    recheck = recheck or (lambda _store, _document: None)
    report = {'claimed': [], 'repeated': [], 'obsolete': [], 'malformed': [], 'refused': [],
              'waiting': None}
    if type(limit) is not int or not 1 <= limit <= 1024:
        raise outbox.OutboxError(f'bounded dispatch expected 1..1024, got {limit!r}')
    held = connection is None
    try:
        link = connection if connection is not None else Connection(url, name=consumer).__enter__()
    except BrokerUnavailable as error:
        report['waiting'] = str(error)
        return report
    try:
        link.channel.basic_qos(prefetch_count=max(1, min(prefetch, limit)))
        taken = 0
        deadline = time.monotonic() + timeout
        for route in routes:
            queue_of(route)  # validate before consuming
            while taken < limit and time.monotonic() < deadline:
                try:
                    method, properties, body = link.channel.basic_get(queue_of(route),
                                                                     auto_ack=False)
                except Exception as error:
                    report['waiting'] = f'delivery interrupted: {error}'
                    return report
                if method is None:
                    break
                taken += 1
                _settle(store, link, method, body, consumer, recheck, launch, report)
        del pika, properties
    finally:
        if held:
            link.close()
    return report


def _settle(store, link, method, body, consumer, recheck, launch, report):
    """One delivery, settled before the acknowledgement that releases it."""
    try:
        document = json.loads(body)
        outbox.validate(document)
    except (ValueError, TypeError, outbox.OutboxError) as error:
        # Malformed is not a retry: the same bytes will not become valid. Record it where a
        # maintainer will see it, then take it off the queue.
        report['malformed'].append(str(error))
        _remember_malformed(store, body, str(error))
        link.channel.basic_ack(method.delivery_tag)
        return
    identity = document['dispatch_id']
    reason = recheck(store, document)
    if reason is not None:
        outbox.record_failure(store, identity, outbox.OBSOLETE, reason)
        report['refused'].append((identity, reason))
        link.channel.basic_ack(method.delivery_tag)
        return
    try:
        state, row = outbox.claim(store, identity, consumer, launch=launch, document=document)
    except outbox.Obsolete as error:
        outbox.record_failure(store, identity, outbox.OBSOLETE, str(error))
        report['obsolete'].append(str(error))
        link.channel.basic_ack(method.delivery_tag)
        return
    if state == outbox.CLAIMED:
        report['claimed'].append(identity)
    else:
        # Already handed off -- a redelivery, or the coordinator got there first. A no-op, and
        # the acknowledgement is still ours to give.
        report['repeated'].append((identity, row['claimed_by']))
    link.channel.basic_ack(method.delivery_tag)


def _remember_malformed(store, body, reason):
    """A persistent failed-delivery record for a message that is not a dispatch.

    It has no identity we can trust, so it is recorded under a digest of its own bytes: an
    operator can see how many arrived and what they looked like without a dead-letter queue
    becoming the only trace of them.
    """
    import hashlib
    digest = hashlib.sha256(body if isinstance(body, bytes) else str(body).encode()).hexdigest()
    # If the store is down leave the delivery unacknowledged. Losing both the
    # delivery and its classification would make the broker the only durable record.
    outbox.record_undeliverable(store, digest, reason, len(body or b''))
