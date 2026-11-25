# eForecast project author: George Sideratos and Claude
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""A16-07: what placing a model run in Kubernetes costs, measured against the local Docker path.

**Predeclared before measuring** (A12 practice), so the conclusion cannot be chosen afterwards:

* The work is identical: the same image, the same `pilot_target` payload, the same seed, the same
  epochs. Nothing about batch, precision or search budget changes between arms -- a faster
  Kubernetes produced by shrinking the work would be a fabrication.
* Each arm is measured **as it actually works**, end to end, from "submit" to "a verified result
  exists". For Kubernetes that includes scheduling, the image already being present in
  containerd, the bootstrap's A08 admission, and the worker. For local Docker it includes
  container creation and the worker. Neither arm is given a head start by excluding its own
  setup; what each includes is listed in the output.
* Arms alternate and are counterbalanced (L,K,K,L,...) so a machine that warms up or cools down
  over the run cannot favour whichever went first.
* The comparison reports the **median and the full range** of each arm. If the ranges overlap,
  the difference is reported as not established. A single pair is never a result.
* The measurement is about *overhead*, not about throughput. Throughput is decided by how many
  model runs fit on a card at once, which is a scheduling property and is reported separately
  by a separate placement drill -- not by this script.
"""
import json
import os
import statistics
import subprocess
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = Path(__file__).resolve().parent.parent
PILOT = Path(os.environ['EFORECAST_PILOT_DIR'])
IMAGE = os.environ.get('EFORECAST_WORKER_IMAGE', 'eforecast-worker:a16-11')
KUBE_CONTAINER = os.environ.get('EFORECAST_KUBE_CONTAINER', 'a16-05-k3s')
PAIRS = int(os.environ.get('EFORECAST_MEASURE_PAIRS', '3'))
PAYLOAD = {'workdir': '/eforecast/scratch/work', 'rows': 4096, 'epochs': 40, 'seed': 11}


def kubectl(*args, stdin=None, timeout=300):
    return subprocess.run(['docker', 'exec', '-i', KUBE_CONTAINER, 'kubectl', *args],
                          capture_output=True, text=True, input=stdin, timeout=timeout)


def local_arm():
    """The lab's existing path: one container per model run, created and run by Docker."""
    work = PILOT / ('local-' + uuid.uuid4().hex[:8])
    work.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    done = subprocess.run(
        ['docker', 'run', '--rm', '--network', 'none',
         '--cpus', '4', '--memory', '8192m',
         '-e', 'PYTHONPATH=/pilot', '-e', 'HOME=/tmp/pilot-home',
         '--mount', f'type=bind,source={PILOT}/pilot,target=/pilot,readonly',
         '--mount', f'type=bind,source={work},target=/eforecast/scratch',
         '--entrypoint', 'python', IMAGE, '-c',
         'import json,sys;sys.path.insert(0,"/pilot");import pilot_target;'
         f'print("PILOT RESULT "+json.dumps(pilot_target.run({PAYLOAD!r}, '
         'type("C",(),{"allocation_id":"local","fence":1,"device_uuid":None})()), sort_keys=True))'],
        capture_output=True, text=True, timeout=1800)
    elapsed = time.monotonic() - started
    return elapsed, _result_of(done.stdout + done.stderr), done.returncode


def _drill():
    """The A16-05 drill module, reused so both arms exercise the real paths, not re-implementations."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'smoke_kubernetes', str(ROOT / 'docker' / 'smoke_kubernetes.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def kubernetes_arm(drill, worker):
    """The pilot path: a Job, scheduled, admitted by A08 inside the Pod, then the worker."""
    # Each arm starts from the state its own path starts from: the drill reaps grants whose
    # Pods are gone, exactly as a coordinator would. Without this the measurement would
    # eventually be measuring A08 refusing for lack of capacity.
    drill.reset()
    intent = drill.intent_for('measure-' + uuid.uuid4().hex[:6])
    intent['payload'].update(PAYLOAD)
    started = time.monotonic()
    handle, submitted = drill.launch(worker, intent)
    observed = drill.settle(worker, handle, until=('finished', 'unknown'), limit=1800)
    elapsed = time.monotonic() - started
    result = _result_of(drill.pod_log(handle))
    if observed.state != 'finished' or observed.returncode != 0 or result is None:
        print(f'    [kubernetes] submit={submitted.state} final={observed.state} '
              f'exit={observed.returncode} detail={(observed.detail or "")[:90]}', flush=True)
        return elapsed, result, 1
    return elapsed, result, 0


def _result_of(text):
    for line in text.splitlines():
        if line.startswith('PILOT RESULT '):
            try:
                return json.loads(line[len('PILOT RESULT '):])
            except ValueError:
                return None
    return None


def summarise(name, samples):
    times = [t for t, _ in samples]
    return {'arm': name, 'runs': len(times), 'median_seconds': round(statistics.median(times), 2),
            'min_seconds': round(min(times), 2), 'max_seconds': round(max(times), 2),
            'compute_seconds': round(statistics.median(
                [r['seconds'] for _, r in samples if r and 'seconds' in r] or [0]), 2)}


def overlapping(a, b):
    return not (a['max_seconds'] < b['min_seconds'] or b['max_seconds'] < a['min_seconds'])


def main():
    print(__doc__.split('"""')[0].strip() or '', flush=True)
    print(f'image {IMAGE}; {PAIRS} counterbalanced pairs; payload {PAYLOAD}\n', flush=True)
    drill = _drill()
    drill.install_code()
    worker = drill.executor()
    local, kubernetes, failures = [], [], []

    for index in range(PAIRS):
        order = ['local', 'kubernetes'] if index % 2 == 0 else ['kubernetes', 'local']
        for arm in order:
            if arm == 'local':
                elapsed, result, code = local_arm()
                local.append((elapsed, result))
            else:
                elapsed, result, code = kubernetes_arm(drill, worker)
                kubernetes.append((elapsed, result))
            ok = code == 0 and result and result.get('beats_naive')
            if not ok:
                failures.append(f'{arm} run {index + 1}: exit {code}, result {result}')
            print(f'  pair {index + 1} {arm:<11} {elapsed:7.1f}s  '
                  f'compute {(result or {}).get("seconds", "?")}s  '
                  f'mae {(result or {}).get("best_mae_test", "?")}', flush=True)

    summary = {'local': summarise('local docker', local),
               'kubernetes': summarise('kubernetes job', kubernetes),
               'failures': failures}
    overlap = overlapping(summary['local'], summary['kubernetes'])
    median_gap = summary['kubernetes']['median_seconds'] - summary['local']['median_seconds']
    summary['overhead_seconds'] = round(median_gap, 2)
    summary['difference_established'] = (not overlap) and not failures
    # Quality must match: the same work, the same seed, the same answer. Compared to six
    # decimal places, not bit-for-bit -- CPU reductions differ in the last places between
    # containers with different thread counts, and claiming bit-identity would be false.
    exact = {r['best_mae_test'] for _, r in local + kubernetes if r}
    summary['quality_agrees_to_6dp'] = len({round(value, 6) for value in exact}) == 1
    summary['mae_values'] = sorted(exact)

    print(f"\nlocal docker    median {summary['local']['median_seconds']}s "
          f"(range {summary['local']['min_seconds']}-{summary['local']['max_seconds']})")
    print(f"kubernetes job  median {summary['kubernetes']['median_seconds']}s "
          f"(range {summary['kubernetes']['min_seconds']}-{summary['kubernetes']['max_seconds']})")
    print(f"placement overhead, median difference: {median_gap:+.1f}s")
    print('the ranges overlap, so the difference is NOT established'
          if overlap else 'the ranges do not overlap')
    print(f"quality agrees to 6 dp across every run: {summary['quality_agrees_to_6dp']} "
          f"({len(summary['mae_values'])} distinct raw values, "
          f"spread {max(summary['mae_values']) - min(summary['mae_values']):.2e})")
    if failures:
        print('failures:', *failures, sep='\n  ')
    out = os.environ.get('EFORECAST_MEASURE_RESULTS')
    if out:
        Path(out).write_text(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == '__main__':
    raise SystemExit(main())
