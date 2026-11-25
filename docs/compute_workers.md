# Work in a process of its own (compute workers)

Technical reference. Start with the [documentation index](README.md) or the
[administrator guide](administrator.md) for setup and recovery.

Some of a run's work is done in a process of its own: fitting the clusterer, a whole CPU
study, the image encoder, prediction and combination, a forecast, a site graph. Such a process
is called a **worker**. This page says what a worker is given, where it writes, how it ends,
and which operations use one.

**Nothing is reserved for a worker and nothing is asked before it starts.** There is no ledger
of runs, no allocation of CPU, memory, GPU memory or disk, no queue and no wait. A worker is
started with a number of threads and, if it asked for one, a card; from then on it is a
process like any other, and the operating system gives it processor time and memory. If the
machine does not have what the work needs, the work fails the way any process does, and the
failure says which way (see [How a worker ends](#how-a-worker-ends)).

What a stage gets from running work this way:

- the memory of the work goes back to the machine when the process exits, whatever the work
  did with it;
- the work can be stopped without stopping the stage, and it **stops when the stage goes
  away** (below);
- its output goes to a log of its own, which is named when the worker starts:
  `clustering runs in its own process; its output goes to …/worker.log`.

Deep (GPU) trials are not started through this interface. The training scheduler starts them
itself and places them on the cards by what it measures; see
[GPU training](gpu_training_optuna.md).

## The interface

The host side is `eforecast.config.compute_workers.ComputeWorkers`. A target needs only
`eforecast.compute_worker.WorkerContext`, which imports the standard library and nothing of
training, Optuna or Torch, so the online image can run targets too.

```python
from uuid import uuid4
from eforecast.config.compute_workers import ComputeWorkers, failure_reason

workers = ComputeWorkers()
request = workers.request(run_id, task_id, uuid4().hex, threads=4)        # device='GPU-…' for a card
handle = workers.prepare(request, 'my_package.operation:execute', {'input_path': path},
                         input_files={'static.pickle': data}, execution={'device': 'cpu', 'threads': 4})
result = workers.wait(handle, timeout=None)       # starts it, and returns when its whole tree has exited
if result['state'] != 'success':
    raise RuntimeError(failure_reason(result))
value, folder = result['value'], workers.root(handle)
workers.discard_result(handle)                    # removes the worker's own folder, never a case's
```

| Call | What it does |
| --- | --- |
| `request(run_id, task_id, attempt_id, threads=1, *, device=None)` | Names one start of one task. `device` is a `GPU-`/`MIG-` UUID or `None`; an ordinal is refused. |
| `prepare(request, 'module:function', payload, *, execution=None, input_files=None)` | Creates the worker's folder, writes its input, and returns the handle of a process that does not exist yet. |
| `submit(handle)` | Starts it. `wait` does this itself. Submitting twice starts nothing twice. |
| `collect(handle)` | Where it is: `absent` (not started), `running`, `cleaning` (it has ended and its tree is being cleared), then `success`, `failure`, `cancelled` or `unknown`. A result is returned only when the worker's whole process tree has exited. |
| `wait(handle, timeout=120)` | Submits and polls `collect`; checks for a cancelled run; on a timeout the worker is told to stop. |
| `cancel(handle, force=False)` | Stops the worker and everything it started. Before the start, the target is never imported. |
| `pid(handle)`, `root(handle)` | The worker's process while it runs; its own folder. |
| `discard_result(handle)` | Removes that folder once the tree has exited. The small record of the launch stays with the executor. |

A target is `function(payload, context)` and returns a JSON mapping. Pass paths and identities,
not loaded models or large arrays. `context` is a `WorkerContext`: `run_id`, `task_id`,
`attempt_id`, `launch_id` (what names this start and no other), `threads`, `device_uuid` and
`result_directory`.

## What a worker is started with

**Threads.** `request.threads` is how many the worker's libraries are told to use: the
thread-count variables of OpenMP, MKL, OpenBLAS, NumExpr and Torch are set before the target
is imported. They are a setting, not a limit: nothing stops a library that ignores them, and
nothing else on the machine is held back for them.

**A card.** A worker that is given a card sees that card and no other
(`CUDA_VISIBLE_DEVICES` is its UUID), addresses it as `cuda:0`, and checks its UUID before it
loads a model. Nothing is taken from the card: the driver hands out its memory as the worker
asks, and a worker that asks for more than is there fails with the driver's own error.
`eforecast.config.model_compute.device_for(preference)` turns a preference into a card:

| Preference | Card |
| --- | --- |
| `cpu`, `none`, nothing | none: the work runs on the CPU |
| `auto` | the visible card with the most free memory at that moment |
| `cuda`, `cuda:N` | the card with that order among the visible ones |
| `GPU-…`, `MIG-…` | that card |

A preference that no visible card answers runs on the CPU, with a warning that says so. Bundle
qualification is the exception: asked for a card that is not there, it fails rather than
qualify a model on another device. `cards_for(preference)` is every card a preference names,
the one with the most free memory first; `foundation_cards(static)` is what a case's foundation
families ask for between them.

**Readers on cards.** A worker on the CPUs can still have part of its work done on cards:
`eforecast.bounded_pools.CardReaders('module:function', cards, native_threads)` starts one
reader a card, which take work as they become free (`submit(*arguments)` returns a future). A
reader is a spawned process with its parent's environment, and its parent sees no card, so the
card is put in place before the work is imported -- which is why the work is given by name.
Nothing is retried there: a reader that dies fails every future still out, and whoever asked
decides what becomes of that work. The `predict` stage uses it for its foundation models.

**The execution policy.** `execution={'device': …, 'threads': …, 'n_jobs': …}` are the stored
preferences of the operation. They are normalized to what the worker was started with and
written into its input as `policy`: `requested`, `given` (`cpu_threads`, `device_uuid`),
`effective` and the `normalization` notes. The case that holds the preferences is never
edited. A selector or a CPU study asks for `parallel_layer: estimator_or_cv`: one native
thread, and an estimator built with the threads of the work.

## Where a worker writes

Every start has a folder of its own, `<executor directory>/compute/<launch id>/`, under
`~/.local/state/eforecast/executors` unless `EFORECAST_RUN_EXECUTOR_DIR` says otherwise:

| File | Content |
| --- | --- |
| `input.json` | context, entry point, payload, policy and the digests of `inputs/` |
| `inputs/` | the `input_files` given to `prepare`, checked against their digests before the target is imported |
| `worker.log` | everything the worker and its libraries print |
| `worker_result.json` | what the target returned |
| `artifacts/` | files the target leaves for the host; each is hashed, and `collect` checks them again |

A worker writes its results there, and the host moves them into the case once the worker has
exited and the inputs are found unchanged. Two kinds of work write into the case themselves:
a CPU study publishes each finished trial into its trial folder, and the clusterer's worker
reads the case's datasets and scalers.

## How a worker ends

| State | Meaning |
| --- | --- |
| `success` | the target returned; `value`, `files` and `context` are in the result |
| `failure` | the target raised, could not be started, or was killed |
| `cancelled` | it was stopped: by `cancel`, by a cancelled run, or because its stage went away |
| `unknown` | its supervisor ended without recording a result, or an artifact no longer matches |

`failure_reason(result)` is one line for a person: the state, what the worker said last, and
the path of its log. For example:

```text
failure: ValueError: attempt to get argmax of an empty sequence (worker log: …/worker.log)
failure: it was killed (SIGKILL); when nobody stopped it, that is the kernel ending the largest
         process because the machine ran out of memory (worker log: …/worker.log)
failure: the worker was not started: [Errno 2] No such file or directory: … (worker log: …)
```

**Running out of memory.** Since nothing is set aside, a machine can run out. Linux then ends
the largest process, which is usually the worker, and its exit status says `SIGKILL`. The
remedies are the settings that decide how much runs at once: fewer CPU studies side by side
(`cpu_workers`), fewer readers (`n_jobs`), a smaller batch, or closing what else is using the
machine. CPU training does one thing by itself: a study that was killed is started once more,
alone, when the other studies have finished ([CPU training](cpu_training_scopes.md)).

**A worker stops with its stage.** The worker's supervisor reads a pipe whose other end only
the process that started it holds. When that process ends, however it ends (finished, an
exception, Ctrl-C, a debugger's Stop, `kill -9`), the kernel closes the pipe and the supervisor
stops the worker and everything it started. Nobody is waiting for the worker then, and the case
it writes into belongs to the next run. So there is nothing to reclaim after a stopped run and
nothing to wait out before the next one. A whole run started by the web service is not watched
this way: it must outlive a restart of the service.

**Between two runs of a case** there is one guard, the case's lock, held by the run's own
process for as long as it lives. `python -m eforecast.config.direct_stage status <path_model>`
names the process that holds it, and `stop-mine <path_model>` stops it if it is yours.

## The operations that run in a worker

| Operation | Threads | Device | Settings |
| --- | --- | --- | --- |
| Fuzzy clusterer: fit, transfer, activation | `clustering.n_jobs`, else `n_cpus`, else 1. With more than one, the six candidate widths of the rule start are fitted side by side: 19 s instead of 60 on a five-month wind case, the same result. The preset is 6; a case made before November 2026 carries 1 | `clustering.params.device`, else `auto` | `EFORECAST_FUZZY_DEVICE` |
| CPU study (one method on one cluster) | its share of `n_cpus` | CPU; a foundation family's own preference | `n_cpus`, `cpu_workers`, `cpu_study_threads`, `cpu_study_device` |
| Foundation model (Chronos-2, TimesFM, TiRex) | the caller's `n_jobs` | `cpu_study_device`, else the family's `device`, else `auto` (a case made before 14 November 2026 says `cpu` for Chronos-2 and TimesFM) | the family's parameters |
| Image encoder (statistics of the image dataset) | `n_jobs`; 1 on a card | `auto` | `EFORECAST_IMAGE_DEVICE`, `EFORECAST_IMAGE_BATCH_SIZE` |
| Prediction, combination, online forecast | 2 | `cpu`; in the `predict` stage the foundation models whose family asks for a card are predicted in readers on the cards, one a card | `EFORECAST_INFERENCE_THREADS`, `EFORECAST_INFERENCE_DEVICE` |
| The `predict` stage of combine (every trained model) | `n_cpus`: its models side by side, 2 threads each. A foundation model that runs in process is one of them, and each reader holds its own copy of it (about 2 GB) | `cpu` | the same two; a thread count said outright means one model at a time |
| Site graph: train, predict | `--threads`, else the cases' `n_cpus` | `--gpu` | |
| Bundle capture, validation, replay | 2 | `cpu` | `EFORECAST_BUNDLE_DEVICE`, `EFORECAST_BUNDLE_TIMEOUT` |
| Weather archive conversion | 1 | CPU | |
| Network graph preview (web UI) | 1 | CPU | |

Dataset stages (extract, the datasets, split, preprocess, features) run in the process of the
run itself, with reader pools sized from the case's `n_jobs`
([how it works](how_it_works.md#9-the-dataset-stages-one-by-one)).

### The fuzzy clusterer

`TfRBFClusterer.fit`, the transfer of a fitted clusterer to another case and offline activation
run in a worker. The host hands it the configuration and the masks as named inputs; the worker
reads the case's datasets and global scalers, fits, and leaves a bundle under `artifacts/`. It
never fits a cluster-local scaler and never redraws the frozen partition. The host copies the
verified bundle to `cluster_organizer/RBF/.fuzzy-generations/<launch id>/` with a
`bundle_manifest.json`, and then replaces the pointer `fuzzy_bundle.json`. A reader that finds
a pointer uses that generation for its lifetime; one that finds none reads the older layout.
Rule folders are not replaced by this, and the clustering stage still writes its receipt last.

### Prediction, combination and forecasts

One worker loads the models and the data, predicts, and leaves the frames and an `evidence`
record of what it read. The host checks that the inputs have not changed since, and only then
copies the prediction caches into the case, records a routine forecast in the forecast store
and writes `latest.csv`, or writes the forecast CSV with its manifest. A worker whose inputs
changed while it ran publishes nothing: `inference inputs changed; private output was not
published: <the files>`. A file that a sync client marked as uploaded while the worker ran is
not a changed input: its change time moved with its attributes and nothing else did
([troubleshooting](troubleshooting.md#-inputs-changed--in-a-folder-that-is-synced)).

### The site graph

One worker trains the graph of several cases or predicts with it. Publication replaces the
group's bundle and every site's export together: the new folders are staged, moved into place,
and one `commit.json` is written last. Until it exists the exports are not finished work for
any reader; once it exists the publication stands, whatever fails after it. A publisher that
was killed in between is rolled back by the next command that takes the same cases, or by
`python run_site_graph.py --recover-publication <group folder>`.

## What the Runs page shows

A run writes a small status file as it goes, and the Runs page shows it under **Resources**:
the threads its running work is told to use, the memory its processes hold (PSS, measured),
and for CPU training how many studies run at once, each study's threads and device, and the
models finished per minute. For GPU training it shows, for each card, what is in use and what
the running trials are expected to take. These are observations of what is running. None of
them is a limit.

## Kubernetes

The Kubernetes venue and the run coordinator keep an allocator of their own
(`eforecast/training/command_line/allocator*.py`, `allocation_contract.py`,
`eforecast/training/compute_ready.py`): several nodes need something to arbitrate between
them. A run on this workstation never touches it. See
[managed runs on Kubernetes](../deploy/kubernetes/MANAGED_RUNS.md).
