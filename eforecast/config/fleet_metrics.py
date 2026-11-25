# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""A16-06: Prometheus exposition of durable application state.

Three rules shape everything here.

**An unavailable metric is not zero activity.** If the outbox cannot be read, this does not emit
``eforecast_dispatch_pending 0`` -- that is a lie a dashboard would draw as a calm flat line. It
emits ``eforecast_dispatch_up 0`` and *no* backlog series at all, so a panel shows "no data" and
an alert can fire on the source being down rather than on a reassuring zero.

**Label dimensions are closed sets.** Every label value comes from an enumeration declared in
this module and checked against the modules that own it, so the number of series a running
installation can produce is a product of small constants, computable in a test. Run ids, task
ids, attempt ids and dispatch ids are **never** labels: they go in the drill-down link and in the
durable record, which is where an operator who needs one actually looks.

**It renders, it does not serve.** The same stdlib text format the training orchestrator already
exposes, built as a string. Whatever serves it -- the orchestrator's existing endpoint, the web
UI, a one-shot CLI -- is not this module's business, and there is no new dependency.
"""
from __future__ import annotations

import time

#: The states a dispatch row can be in. Checked against `dispatch_outbox` below.
STATES = ('pending', 'published', 'claimed', 'failed')
#: Why a delivery failed. Also from `dispatch_outbox`.
CLASSIFICATIONS = ('malformed', 'unsupported', 'unroutable', 'obsolete', 'exhausted')
#: Where work runs. From `image_release.RUNTIMES`.
RUNTIMES = ('worker', 'online', 'foundation', 'lag_llama')
#: Execution venues. From `dispatch_broker.ROUTES`, and `managed` is not optional there:
#: `kubernetes_runs` dispatches on it (`routes=('managed',)`), so a queue exists for it and a
#: backlog on it would otherwise be invisible here.
ROUTES = ('local', 'remote', 'managed')
#: What a run is waiting for, in the A14 vocabulary that the UI already speaks.
WAITS = ('dispatch', 'worker', 'image', 'staging', 'admission', 'gpu', 'cpu', 'disk', 'store',
         'compute', 'supervisor_slot', 'stage_slot', 'artifact_lock', 'unknown')

#: Sources that can be independently up or down. One `_up` gauge each.
SOURCES = ('dispatch', 'runs', 'cluster', 'broker', 'storage')
#: The sources an installation has only when it uses that venue: a broker for the `broker` route,
#: a Kubernetes cluster for `managed`. A workstation has neither. Their `_up` stays 0 there, as
#: for any source that cannot be read, and `eforecast_source_configured` says why, so that an
#: alert or a panel can leave out what this installation never meant to have.
OPTIONAL_SOURCES = ('broker', 'cluster')


class Series:
    """A metric family: a name, a type, a help string, and the labels it is allowed to carry."""

    __slots__ = ('name', 'kind', 'help', 'labels')

    def __init__(self, name, kind, help_text, labels=()):
        self.name, self.kind, self.help, self.labels = name, kind, help_text, tuple(labels)


FAMILIES = (
    Series('eforecast_source_up', 'gauge',
           'Whether a metric source could be read at all. 0 means unknown, never zero activity.',
           ('source',)),
    Series('eforecast_source_configured', 'gauge',
           'Whether this installation is meant to have the source. 0 only for a broker or a '
           'cluster it does not use; its source_up 0 is then no outage.',
           ('source',)),
    Series('eforecast_dispatch_rows', 'gauge',
           'Dispatch records by state. Absent entirely when the outbox cannot be read.',
           ('state',)),
    Series('eforecast_dispatch_oldest_age_seconds', 'gauge',
           'Age of the oldest dispatch still waiting to be published.'),
    Series('eforecast_dispatch_failures', 'gauge',
           'Dispatch records that failed, by classification.', ('classification',)),
    Series('eforecast_dispatch_undeliverable', 'gauge',
           'Deliveries that were not dispatches at all, counted by digest.'),
    Series('eforecast_dispatch_attempts_total', 'counter',
           'Publication attempts recorded against dispatch records.'),
    Series('eforecast_runs_waiting', 'gauge',
           'Runs waiting, by what they are waiting for.', ('wait',)),
    Series('eforecast_runs_active', 'gauge', 'Runs currently executing.'),
    Series('eforecast_completions_verified_total', 'counter',
           'Completions accepted with a verified receipt, by runtime.', ('runtime',)),
    Series('eforecast_cleanup_pending', 'gauge',
           'Runs whose cleanup has been recorded as owed but not observed.'),
    Series('eforecast_queue_messages', 'gauge',
           'Messages the broker holds, by route.', ('route',)),
    Series('eforecast_queue_unacknowledged', 'gauge',
           'Deliveries the broker is holding open, by route. A short dispatcher keeps this at 0.',
           ('route',)),
    Series('eforecast_cluster_jobs', 'gauge',
           'Worker Jobs in the cluster that have not finished.'),
    Series('eforecast_cluster_pods_pending_seconds', 'gauge',
           'How long the longest-pending worker Pod has been waiting for a node, image or volume.'),
    Series('eforecast_storage_free_bytes', 'gauge',
           'Free space on the workspace filesystem. Emitted only when it could be measured.'),
    Series('eforecast_storage_total_bytes', 'gauge',
           'Size of the workspace filesystem, so a ratio can be alerted on without a constant.'),
)

BY_NAME = {family.name: family for family in FAMILIES}


class MetricError(ValueError):
    """A series was asked to carry something this exposition will not emit."""


def _check_label(family, labels):
    allowed = {
        'source': SOURCES, 'state': STATES, 'classification': CLASSIFICATIONS,
        'runtime': RUNTIMES, 'route': ROUTES, 'wait': WAITS,
    }
    if tuple(sorted(labels)) != tuple(sorted(family.labels)):
        raise MetricError(f'{family.name} takes labels {family.labels}, got {tuple(labels)}')
    for key, value in labels.items():
        if value not in allowed[key]:
            # The whole point: an id can never become a label by accident, because nothing
            # outside these closed sets is emittable.
            raise MetricError(f'{family.name}: {key}={value!r} is not one of {allowed[key]}')


def sample(name, value, **labels):
    """One sample, with its labels checked against the closed sets before it can be rendered."""
    family = BY_NAME.get(name)
    if family is None:
        raise MetricError(f'{name} is not a declared metric family')
    _check_label(family, labels)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MetricError(f'{name}: expected a number, got {value!r}')
    if value != value or value in (float('inf'), float('-inf')):
        raise MetricError(f'{name}: a non-finite value is not a measurement')
    rendered = ','.join(f'{key}="{labels[key]}"' for key in sorted(labels))
    return f'{name}{{{rendered}}} {value}' if rendered else f'{name} {value}'


def cardinality():
    """The most series this exposition can ever produce. A bound, not an estimate."""
    sizes = {'source': len(SOURCES), 'state': len(STATES),
             'classification': len(CLASSIFICATIONS), 'runtime': len(RUNTIMES),
             'route': len(ROUTES), 'wait': len(WAITS)}
    total = 0
    for family in FAMILIES:
        product = 1
        for label in family.labels:
            product *= sizes[label]
        total += product
    return total


# ---------------------------------------------------------------------- the sources


def dispatch_samples(store, *, now=None):
    """The outbox, as bounded counts. Raises nothing: an unreadable source reports itself down."""
    from eforecast.config import dispatch_outbox as outbox
    moment = time.time() if now is None else now
    lines = []
    try:
        counts, failures, oldest, attempts = _dispatch_counts(store, outbox)
    except Exception as error:                       # a source that cannot be read is not zero
        return [sample('eforecast_source_up', 0, source='dispatch')], str(error)
    lines.append(sample('eforecast_source_up', 1, source='dispatch'))
    for state in STATES:
        lines.append(sample('eforecast_dispatch_rows', counts.get(state, 0), state=state))
    for classification in CLASSIFICATIONS:
        lines.append(sample('eforecast_dispatch_failures', failures.get(classification, 0),
                            classification=classification))
    lines.append(sample('eforecast_dispatch_oldest_age_seconds',
                        max(0.0, moment - oldest) if oldest else 0.0))
    lines.append(sample('eforecast_dispatch_attempts_total', attempts))
    try:
        lines.append(sample('eforecast_dispatch_undeliverable',
                            len(outbox.undeliverable(store, limit=1024))))
    except Exception:
        pass                                         # the counts above are still worth having
    return lines, None


def _dispatch_counts(store, outbox):
    """One pass over the outbox, aggregated in the database rather than in Python."""
    import json
    counts, failures, oldest, attempts = {}, {}, None, 0
    with store._transaction() as cursor:
        store._execute(cursor, '''SELECT state, COUNT(*), MIN(created_at), SUM(attempts)
            FROM eforecast_dispatch_outbox WHERE namespace=? GROUP BY state''',
            (store.namespace,))
        for state, number, earliest, tried in cursor.fetchall():
            counts[state] = int(number)
            attempts += int(tried or 0)
            if state == 'pending' and earliest is not None:
                oldest = float(earliest)
        store._execute(cursor, '''SELECT failure FROM eforecast_dispatch_outbox
            WHERE namespace=? AND state=? AND failure IS NOT NULL''',
            (store.namespace, 'failed'))
        for (failure,) in cursor.fetchall():
            try:
                classification = json.loads(failure).get('classification')
            except (TypeError, ValueError):
                continue
            if classification in CLASSIFICATIONS:
                failures[classification] = failures.get(classification, 0) + 1
    return counts, failures, oldest, attempts


def broker_samples(depths):
    """``depths`` maps a route to ``(messages, unacknowledged)``, or is None if unreachable."""
    if depths is None:
        return [sample('eforecast_source_up', 0, source='broker')], 'broker unreachable'
    lines = [sample('eforecast_source_up', 1, source='broker')]
    for route in ROUTES:
        if route not in depths:
            return [sample('eforecast_source_up', 0, source='broker')], 'incomplete queue observation'
        messages, unacknowledged = depths[route]
        lines.append(sample('eforecast_queue_messages', messages, route=route))
        lines.append(sample('eforecast_queue_unacknowledged', unacknowledged, route=route))
    return lines, None


def run_samples(store, *, deadline=None, limit=10000):
    """Bounded durable observations, never counts obtained by mutating/reconciling runs.

    The verified total comes from `verified_completions`, a durable counter incremented in the
    transaction that accepts a completion -- never from a count over retained runs, which is
    not cumulative and whose rows may be deleted.
    """
    import json
    from .run_model import Run
    waiting = {key: 0 for key in WAITS}
    active = cleanup = count = 0
    try:
        with store._transaction(write=False) as cursor:
            store._execute(cursor, 'SELECT payload FROM eforecast_runs WHERE namespace=? LIMIT ?',
                           (store.namespace, limit + 1))
            while True:
                rows = cursor.fetchmany(128)
                if not rows:
                    break
                for (payload,) in rows:
                    count += 1
                    if count > limit or (deadline is not None and time.monotonic() >= deadline):
                        raise MetricError('run observation exceeded its bound; partial counts withheld')
                    run = Run.from_json(payload)
                    if run.deleted:
                        continue
                    active += run.state == 'running'
                    cleanup += run.cleanup not in ('not_required', 'complete')
                    # A run does not wait for a slot or for disk any more; one that has not
                    # started is one whose start has not been observed yet.
                    if run.state in ('unknown', 'queued'):
                        waiting['unknown'] += 1
        # The cumulative counter, read from its own rows rather than counted over retained
        # history: A16-06 declared this series and nothing produced it (R27-15).
        from .verified_completions import totals as verified_totals
        accepted = verified_totals(store)
        return ([sample('eforecast_source_up', 1, source='runs'),
                 sample('eforecast_runs_active', active), sample('eforecast_cleanup_pending', cleanup)] +
                [sample('eforecast_completions_verified_total', value, runtime=key)
                 for key, value in sorted(accepted.items())] +
                [sample('eforecast_runs_waiting', value, wait=key) for key, value in waiting.items()]), None
    except Exception as error:
        return [sample('eforecast_source_up', 0, source='runs')], str(error)


def cluster_samples(jobs, pending_seconds):
    """``jobs`` is None when the cluster API could not be reached."""
    if jobs is None:
        return [sample('eforecast_source_up', 0, source='cluster')], 'cluster API unreachable'
    return [sample('eforecast_source_up', 1, source='cluster'),
            sample('eforecast_cluster_jobs', jobs),
            sample('eforecast_cluster_pods_pending_seconds', float(pending_seconds or 0.0))], None


def storage_samples(path):
    """Workspace capacity, or nothing at all if the filesystem could not be stated.

    A ratio rather than a constant threshold: the lab's workspace and a second workstation's are
    different sizes, and an alert written against bytes would be wrong on one of them.
    """
    import os as _os
    try:
        stats = _os.statvfs(path)
    except OSError as error:
        return [sample('eforecast_source_up', 0, source='storage')], str(error)
    return [sample('eforecast_source_up', 1, source='storage'),
            sample('eforecast_storage_free_bytes', stats.f_bavail * stats.f_frsize),
            sample('eforecast_storage_total_bytes', stats.f_blocks * stats.f_frsize)], None


def render(groups):
    """The exposition text: declared types and help, then whatever the sources could supply."""
    emitted = {line.split('{')[0].split(' ')[0] for group in groups for line in group}
    out = []
    for family in FAMILIES:
        if family.name in emitted:
            out.append(f'# HELP {family.name} {family.help}')
            out.append(f'# TYPE {family.name} {family.kind}')
            out += [line for group in groups for line in group
                    if line.split('{')[0].split(' ')[0] == family.name]
    return '\n'.join(out) + '\n'
