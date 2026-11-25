# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""One step of a case, executed as a Job on the cluster this control plane is connected to.

    deploy/kubernetes/local_cluster.sh run <case> datasets|training|combine [--options JSON]

``local_cluster.sh`` starts this program inside the worker image, with the settings of the
trial cluster and nothing of the installation's own: its run database is the trial's.

What it does is what the Runs tab does, in the same order and through the same HTTP endpoints:
it previews the step, submits it, and then is the run coordinator for as long as the run takes
-- handing the run to the broker, creating the Job, watching the Pod, recording the outcome
and tidying up. Every change of state is printed. The case is decided to the ``broker`` venue
in the trial's own run database; the installation's route for the case is not touched.

A step that replaces what another configuration built says what it replaces and goes on, as a
run typed in a terminal does (``docs/cli.md``): there is nothing to answer and no flag to
answer with. ``--options '{"dry_run": true}'`` shows the plan and changes nothing.

Ctrl-C stops the run the way the page's Stop does and waits until the cluster has let go of it.

Exit status: 0 when the run ended well (also when it had nothing to do), 1 when it failed or
was stopped, 2 for a case the workspace does not have, 3 for a step that was refused, 4 when
this program gave up waiting (the run is left as it is; run the command again to go on
watching the cluster).
"""
import argparse
import json
import os
import sys
import time

GOOD = ('success', 'nothing_to_do')


def replaced(preview, case):
    """What the step replaces because the case no longer says what it was built from; '' if nothing."""
    found = (preview or {}).get('changes')
    if not found:
        return ''
    from eforecast.config.document_changes import describe
    from eforecast.config.stage_resume import describe_changes
    settings = [describe(change) for change in (preview.get('since') or {}).get('changes', ())]
    return describe_changes(found, case, settings)


def outside(document, shared_root):
    """The data, weather and model folders of a case that the Pod would not see.

    A Job mounts one folder of the host, under the path it has on the host. A case that keeps
    its series, its weather or its models anywhere else finds them missing inside the Pod, and
    says so only when a stage goes to read them.
    """
    from pathlib import Path
    project, folders = document.get('project') or {}, document.get('folders') or {}
    named = {'project.filename': project.get('filename'), 'folders.sys_folder': folders.get('sys_folder'),
             'folders.nwp_folder': folders.get('nwp_folder')}
    for index, model in enumerate(project.get('nwp_models') or ()):
        if isinstance(model, dict):
            named[f'project.nwp_models[{index}].path_nwp_source'] = model.get('path_nwp_source')
    shared = Path(shared_root)
    return {name: value for name, value in named.items()
            if isinstance(value, str) and value.startswith('/') and not Path(value).is_relative_to(shared)}


def line(state, cleanup, job, waiting):
    """One printed line for a state the run is in."""
    text = f"{time.strftime('%H:%M:%S')}  {state:<10} job {job or 'not created':<12} cleanup {cleanup}"
    return f'{text}   ({waiting})' if waiting else text


def summary(record):
    """The end of a run as the Runs page would show it."""
    outcome = record.get('outcome') or {}
    lines = [f"run {record['id']}: {outcome.get('kind') or record.get('state')}"
             + (f" -- {outcome['detail']}" if outcome.get('detail') else '')]
    for stage in record.get('stages') or ():
        result = stage.get('outcome') or {}
        lines.append(f"  {stage['stage']:<14} {result.get('kind') or stage['state']}"
                     + (f"   {result['detail']}" if result.get('detail') and result.get('kind') not in GOOD else ''))
    if record.get('log'):
        lines.append(f"  log: {record['log']}")
    lines.append(f"  cleanup on the cluster: {record.get('cleanup')}")
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('case', help='the name of the case, as the Cases page lists it')
    parser.add_argument('script', choices=('datasets', 'training', 'combine'))
    parser.add_argument('--options', default='{}',
                        help='the options of the step as JSON, the ones the Runs tab sends: '
                             '{"stages": ["all"]}, {"cpu": true}, {"dry_run": true} ...')
    parser.add_argument('--seconds', type=int, default=6 * 3600, help='how long to wait for the run (default 6 hours)')
    parser.add_argument('--note', default='a step run on the trial cluster', help='why, for the route history')
    args = parser.parse_args(argv)
    try:
        options = json.loads(args.options)
        if not isinstance(options, dict):
            raise ValueError('not an object')
    except ValueError as error:
        parser.error(f'--options is not a JSON object: {error}')

    os.environ['EFORECAST_RUN_COORDINATOR'] = 'off'       # this program is the coordinator, for this one run
    from eforecast.config import api, dispatch_route, kubernetes_runs as managed, runs
    from eforecast.config.run_coordinator import Coordinator
    from eforecast.config.run_store import configured_store
    from eforecast.config.service import Workspace
    from fastapi.testclient import TestClient

    target = managed.configured_target()
    if target is None:
        print('no cluster is configured for this control plane (EFORECAST_KUBE_RUNS_FILE)', file=sys.stderr)
        return 3
    workspace = Workspace(target['workspace'], admin=True)
    store = configured_store(workspace.root)
    manager = runs.RunManager(workspace, store=store)
    api._run_manager = manager
    client = TestClient(api.create_app())
    coordinator = None
    try:
        found = client.get(f'/api/cases/{args.case}')
        if found.status_code != 200:
            print(f'the workspace {workspace.root} has no case {args.case!r} ({found.status_code})', file=sys.stderr)
            return 2
        elsewhere = outside(found.json().get('document') or {}, target['shared_root'])
        if elsewhere:
            print(f"the Pods of this cluster see {target['shared_root']} and nothing else of this machine; "
                  f"the case {args.case} keeps these outside it:", file=sys.stderr)
            for name, value in elsewhere.items():
                print(f'  {name} = {value}', file=sys.stderr)
            print("create the cluster again with a --shared folder that holds them, or move them", file=sys.stderr)
            return 3
        request = {'script': args.script, 'options': options}
        preview = client.post(f'/api/cases/{args.case}/runs/preview', json=request)
        plan = preview.json() if preview.status_code == 200 else {}
        if preview.status_code != 200:
            print(f'no plan could be made before submitting ({preview.status_code}); the run itself will say why',
                  flush=True)
        text = replaced(plan, args.case)
        if text:
            # A run typed in a terminal is its own decision: it is told what that costs and goes on.
            print(text + ('\n  a dry run: nothing is replaced' if options.get('dry_run') else
                          '\n  this run replaces them now. Stop it (Ctrl-C) if that is not what you want; '
                          '--options \'{"dry_run": true}\' shows the same list and changes nothing'), flush=True)
        dispatch_route.decide(store, args.case, 'broker', by=os.environ.get('EFORECAST_USER') or 'local-cluster',
                              note=args.note)
        response = client.post(f'/api/cases/{args.case}/runs', json={
            **request, 'accept_changes': True, 'client_key': f'cluster-{args.case}-{args.script}-{int(time.time())}'})
        if response.status_code not in (200, 201, 202):
            print(f'the step was refused ({response.status_code}): {response.text[:1500]}', file=sys.stderr)
            return 3
        record = response.json()
        print(f"accepted as run {record['id']} for the cluster {target['server']}, image {target['image'][-19:]}",
              flush=True)
        coordinator = Coordinator(workspace, store=store)
        last, stopping, deadline = None, False, time.monotonic() + args.seconds
        while True:
            try:
                coordinator.tick()
                run = store.get(record['owner_id'], record['case_id'], record['id'])
                bound = managed.assignment(store, run.id)
                state = (run.state, run.cleanup, bound['state'] if bound else None, run.report_error)
                if state != last:
                    print(line(*state), flush=True)
                    last = state
                if bound and bound['state'] == 'complete':
                    break
                if run.state == 'terminal' and bound is None:
                    break                       # it ended before anything was asked of the cluster
                if time.monotonic() >= deadline:
                    print(f'gave up waiting after {args.seconds} s; the run is left as it is', flush=True)
                    return 4
                time.sleep(.5)
            except KeyboardInterrupt:
                if stopping:
                    print('interrupted again: the run is left as it is', flush=True)
                    return 4
                stopping, deadline = True, time.monotonic() + 300
                print('stopping the run, as the Stop button does; waiting for the cluster to let go of it '
                      '(Ctrl-C again to leave it)', flush=True)
                client.post(f"/api/cases/{args.case}/runs/{record['id']}/stop")
        final = manager.get(args.case, record['id'])
        print(summary(final), flush=True)
        kind = (final.get('outcome') or {}).get('kind')
        return 0 if kind in GOOD else 1
    finally:
        if coordinator is not None:
            coordinator.close()
        manager._preparer.shutdown(wait=True)


if __name__ == '__main__':
    sys.exit(main())
