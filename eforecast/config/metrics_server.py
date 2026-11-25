# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""A16-06: serve the durable-state exposition, with the same stdlib server the orchestrator uses.

Deliberately not a framework and not a new dependency. The orchestrator already exposes its own
metrics from a `ThreadingHTTPServer`; this is the same shape for the state that outlives any one
process -- the outbox, the runs, the broker depths, the workspace.

Every collection is wrapped: a source that raises reports itself **down** rather than taking the
endpoint down with it. A scrape must not be the thing that breaks because the database is busy,
or the whole dashboard goes blank for a reason that has nothing to do with the lab's work.
"""
from __future__ import annotations

import os
import time
import threading

from eforecast.config import fleet_metrics as metrics

PORT = int(os.environ.get('EFORECAST_FLEET_METRICS_PORT', '9109') or 0)
#: A scrape that takes longer than Prometheus will wait is a failed scrape with extra load.
BUDGET_SECONDS = 8.0
_COLLECTIONS = {}
_COLLECTION_LOCK = threading.Lock()


def _bounded(key, call, deadline):
    """One in-flight observation per source, shared by concurrent HTTP scrapes.

    Database/network libraries can block past our budget. Such a source is down
    for this scrape; it cannot create an unbounded number of stuck threads.
    """
    with _COLLECTION_LOCK:
        entry = _COLLECTIONS.get(key)
        if entry is None or entry['done'].is_set():
            entry = {'done': threading.Event()}
            _COLLECTIONS[key] = entry

            def observe():
                try:
                    entry['result'] = call()
                except Exception as error:
                    entry['error'] = error
                finally:
                    entry['done'].set()
            threading.Thread(target=observe, daemon=True, name='eforecast-metrics').start()
    if not entry['done'].wait(max(0, deadline - time.monotonic())):
        raise TimeoutError('source did not answer within the scrape budget')
    if 'error' in entry:
        raise entry['error']
    return entry['result']


def collect(*, workspace=None, store=None, broker_depths=None, cluster=None, deadline=None):
    """Every source, each one allowed to fail on its own.

    Returns ``(groups, notes)``: the sample groups to render, and a note per source that could
    not be read -- which the caller may log, but which never becomes a metric value.

    A source that was not given is down (0) like any unreadable one. What tells the two apart is
    ``eforecast_source_configured``: 1 for the run store and the workspace, which every
    installation is meant to have, and for a broker or a cluster only when one was given. A
    workstation without either is not in an outage, and its alerts must not say so.
    """
    limit = (time.monotonic() + BUDGET_SECONDS) if deadline is None else deadline
    groups, notes = [], {}

    def take(name, call):
        if time.monotonic() >= limit:
            groups.append([metrics.sample('eforecast_source_up', 0, source=name)])
            notes[name] = 'skipped: the scrape budget was already spent'
            return
        try:
            key = (name, id(store), str(workspace), id(broker_depths), id(cluster))
            lines, note = _bounded(key, call, limit)
        except Exception as error:                   # a source, not the endpoint, is what failed
            lines, note = [metrics.sample('eforecast_source_up', 0, source=name)], str(error)
        groups.append(lines)
        if note:
            notes[name] = note

    if store is not None:
        take('dispatch', lambda: metrics.dispatch_samples(store))
        take('runs', lambda: metrics.run_samples(store, deadline=limit))
    if workspace is not None:
        take('storage', lambda: metrics.storage_samples(workspace))
    if broker_depths is not None:
        take('broker', lambda: metrics.broker_samples(broker_depths()))
    if cluster is not None:
        take('cluster', lambda: metrics.cluster_samples(*cluster()))
    supplied = {name for group in groups for line in group for name in metrics.SOURCES
                if line.startswith(f'eforecast_source_up{{source="{name}"}}')}
    for name in metrics.SOURCES:
        if name not in supplied:
            groups.append([metrics.sample('eforecast_source_up', 0, source=name)])
            notes[name] = 'source is not configured'
    given = {'broker': broker_depths is not None, 'cluster': cluster is not None}
    groups.append([metrics.sample('eforecast_source_configured',
                                  int(name not in metrics.OPTIONAL_SOURCES or given[name]), source=name)
                   for name in metrics.SOURCES])
    return groups, notes


def render(**sources):
    groups, notes = collect(**sources)
    return metrics.render(groups), notes


def trials_text():
    """``/trials``: the trials training on this machine now and their error after every epoch,
    read from the trial processes (``trial_progress``) -- so for any training, whatever version of
    the code started it. Kept off ``/metrics``, whose series are bounded by construction: a trial's
    series exist while it runs. A failure here blanks this page only."""
    from eforecast.config import trial_progress
    try:
        return trial_progress.render(trial_progress.running_trials())
    except Exception as error:                       # noqa: BLE001 -- never the service's end
        return f'# the trials could not be read: {type(error).__name__}\n'


def processes_text():
    """``/processes``: every process of the pipeline with what it uses, the heaviest others, the
    machine and the cards (``process_watch``). Like ``/trials`` it is read from the processes
    themselves, and a failure here blanks this page only."""
    from eforecast.config import process_watch
    try:
        return process_watch.render(process_watch.snapshot(), process_watch.machine(), process_watch.cards())
    except Exception as error:                       # noqa: BLE001 -- never the service's end
        return f'# the processes could not be read: {type(error).__name__}\n'


def studies_text(watcher):
    """``/studies``: every study of the workspace's cases -- its trials by how they ended, how many
    are left, its best value -- and the trials that finished last (``study_progress``). The
    studies are read in the background; a scrape gets what was read last and never waits."""
    from eforecast.config import study_progress
    if watcher is None:
        return '# no workspace is configured, so there are no studies to read\n'
    if watcher.snapshot is None:
        return f'# the studies have not been read yet{": " + watcher.error if watcher.error else ""}\n'
    try:
        return study_progress.render(watcher.snapshot)
    except Exception as error:                       # noqa: BLE001 -- never the service's end
        return f'# the studies could not be shown: {type(error).__name__}\n'


def studies_json(watcher, path, query):
    """``/studies/index`` and ``/studies/export?name=...`` for the copier of the Optuna dashboard:
    which studies there are and the version of each, and one study with every trial. None for
    anything else, which the server answers with 404."""
    import json
    from urllib.parse import parse_qs
    if watcher is None:
        return None
    if path == '/studies/index':
        return json.dumps({'read': None if watcher.snapshot is None else watcher.snapshot['at'],
                           'studies': watcher.reader.index()})
    if path == '/studies/export':
        name = (parse_qs(query).get('name') or [''])[0]
        return watcher.reader.whole(name)
    return None


def serve(port=PORT, **sources):
    """Start the endpoint and return the server, or None when the port is 0 (off).

    ``/metrics`` (and ``/``) is the installation's state; ``/trials`` the trials training now;
    ``/processes`` the processes of the pipeline, the machine and the cards; ``/studies`` what
    every study has done and has left, with ``/studies/index`` and ``/studies/export`` for the
    copier of the Optuna dashboard."""
    if not port:
        return None
    import http.server
    import threading

    watcher = None
    if sources.get('workspace'):
        from eforecast.config import study_progress
        watcher = study_progress.Watcher(sources['workspace'])
        watcher.start()

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            path, _, query = self.path.partition('?')
            kind = 'text/plain; version=0.0.4'
            if path == '/trials':
                body = trials_text().encode()
            elif path == '/processes':
                body = processes_text().encode()
            elif path == '/studies':
                body = studies_text(watcher).encode()
            elif path in ('/studies/index', '/studies/export'):
                answer = studies_json(watcher, path, query)
                if answer is None:
                    self.send_response(404)
                    self.end_headers()
                    return
                body, kind = answer.encode(), 'application/json'
            elif path in ('/metrics', '/'):
                body = render(**sources)[0].encode()
            else:
                self.send_response(404)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(('0.0.0.0', port), Handler)
    server.studies = watcher                         # for whoever shuts the server down, and for tests
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def main(argv=None):
    """``--print`` renders once and exits; otherwise serve until interrupted."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--port', type=int, default=PORT, help='0 serves nothing')
    parser.add_argument('--print', action='store_true', dest='once',
                        help='render one exposition to stdout and exit')
    parser.add_argument('--workspace', default=os.environ.get('EFORECAST_WORKSPACE'))
    arguments = parser.parse_args(argv)
    # Reading the studies loads the numeric libraries, which would each start a pool of threads
    # as wide as the machine and keep them idle for as long as this service runs.
    for pool in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ.setdefault(pool, '1')

    store = None
    try:
        from eforecast.config.run_store import configured_store
        store = configured_store(arguments.workspace)
    except Exception:                                # no store is a source that is down, not a crash
        store = None

    from .metrics_sources import configured
    sources = {'workspace': arguments.workspace, 'store': store, **configured()}
    if arguments.once:
        text, notes = render(**sources)
        print(text, end='')
        for name, note in sorted(notes.items()):
            print(f'# note {name}: {note}')
        return 0
    server = serve(arguments.port, **sources)
    if server is None:
        print('metrics endpoint disabled (--port 0)')
        return 0
    print(f'fleet metrics on http://0.0.0.0:{arguments.port}/metrics')
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        server.shutdown()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
