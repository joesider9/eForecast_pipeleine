# GPU training, Optuna, Docker and CPU studies

For maintainers and researchers. Updated **13 November 2026**. Colleagues can use
[the UI walkthrough](user.md). Load the [service environment](README.md#before-copying-a-terminal-command)
and work from the repository root; `python` means the configured worker interpreter.
`CASE_CONFIG` below is a YAML file or parts-directory path, not a workspace case name.

## Runbook

### 1. Check inputs before training

```bash
export CASE_CONFIG=/absolute/path/to/your/case
python run_datasets.py --config "$CASE_CONFIG" --dry-run
python run_training.py --config "$CASE_CONFIG" --dry-run
```

Build required datasets with `run_datasets.py --config "$CASE_CONFIG" --stage all`, or submit
Datasets through the UI for a durable run record. Resolve missing input/runtime readiness
findings first. Frozen whole-day masks and global training scalers are shared across cluster
views; neither a GPU option nor a new model revision changes that protocol.

### 2. Train native deep models

```bash
python run_training.py --config "$CASE_CONFIG"
python run_training.py --config "$CASE_CONFIG" --gpus 1 --n-jobs 2
python run_training.py --config "$CASE_CONFIG" --global-method MLP
```

Each command is a separate requested run; choose the one you need. `--gpus` names the cards by
their order among the visible ones. Each trial is a process of its own that sees one card,
named by its UUID. The scheduler decides which trial goes on which card from what it measures
there, as described next. Nothing else is reserved for a trial: not CPUs, not host memory, not
disk. A smaller experiment does not wait behind a larger tag that cannot fit.

#### What the plan says is left to do

Every run prints its plan first, and `--dry-run` prints it and stops:

```
  training     resume   0 trial(s) still wanted, 0 unfinished to redo, 17 transfer trial(s) to train, 208 compatible trial(s) kept
      global/LSTM: 3/3 trials done, reuse 3, redo 0
      RBF_rule_0/TRANSFORMER: 8/8 trials done, reuse 8, redo 0, transfers 0/8 trained, 8 to train
      RBF_rule_5/LSTM: 3/3 trials done, reuse 5, redo 0, transfers 2/3 trained, 1 to train
```

`3/3 trials done` is the study's own budget (`n_trials`). **Transfers** are counted beside it:
a family with `transfer_learning_from_global` trains every finished trial of its global study
once more in each cluster (`test_tl<n>`), and no study wants those trials, so no budget counts
them. `transfers 2/3 trained, 1 to train` is what this cluster has of them; `up to N more after
the global study` appears while the global study itself still wants trials, each of which
becomes one more transfer per cluster when it finishes. A stage that has transfers to train is
not done, however complete its studies are. (Until 17 November 2026 they were not counted: a
plan could say `0 trial(s) still wanted` with dozens of them untrained, and a training stage
whose studies were complete was reused without training any. And until 18 November a run that
trained the global study itself did not train its transfers: they did not wait for that study,
found nothing to start from and were left for the next run of the case.)

#### How many trials share a card, and how big each one is

Without `--n-jobs`, both are **measured**, not configured.

- **Size.** A trial of a kind nobody has measured (a kind is a method and its experiment tag)
  starts alone on a card with everything that is free there. After its first epoch it lowers
  its own memory limit to what it used plus 15% and reports it, and the scheduler gives the rest
  of the card back. The log says `… measured after epoch 0: 287 MB on gpu 0 (admitted with
  10353 MB), 1 core(s)`. Later trials of that kind start with room over the measured size
  instead of with the whole card.
- **Trials per card.** Processes on one card take turns, so a card that is running kernels all
  the time gains nothing from another trial. Each finished trial says how busy its card was,
  on average over the trial's whole life, with so many of its kind on it; under
  `EFORECAST_GPU_BUSY_TARGET` percent (default 80), a card takes one more of that kind. The log
  says `MLP|mlp1: a card with 2.0 of them was busy 69% of the time over a whole trial, so a card
  takes 3`. A network that fills a card stays alone on it; small networks are grouped; a card
  holding several kinds takes as many trials as its least sociable kind allows.
- **Sharing a card.** The trials on one card run their kernels together, through NVIDIA MPS. A
  run starts one MPS daemon for each of its cards and stops it at its end; the log says `cards
  [0, 1] are shared through MPS`. Without it the processes on a card take turns, which costs
  small networks most of what a card could give them: eight small networks on one RTX 2080 Ti
  made 5000 training steps a second taking turns and 8374 through MPS, four made 4343 and 4973,
  and one made the same either way. A card also reads less busy with the same trials on it (78%
  where it read 100% with four), so the rule above finds room for another trial where it saw
  none. If a card's daemon goes away, the trials on it fail with `the remote procedural call
  between the MPS server and the MPS client failed`; the run says `gpu N is no longer shared
  through MPS`, starts those trials again and goes on with that card unshared. It is not used
  for trials in containers, for a card that cannot be named by UUID, with `--n-jobs 1`, or
  while somebody else's MPS daemon is running on the machine. `EFORECAST_GPU_MPS=0` turns it
  off.
- **Order.** The largest kind waiting is offered a freed card first. A trial that needs an empty
  card gets one cleared for it: one card stops taking new arrivals until that trial has started,
  and that trial is the one arrival it takes. A trial that waits for this reason says so: `each
  card already runs the 1 trial(s) of this kind it is known to take`. "No GPU with N MB free"
  means memory.
- **An empty card.** A card with none of this run's trials on it and at least 85% of its memory
  free is as empty as it will get: what a desktop session holds on it is nobody's to wait for.
  It takes any trial that fits a card, with what it has. (A kind measured at 10148 MB once
  waited beside a card with 9922 MB free, and the run stopped starting trials.) A card that
  somebody else has taken a real part of is waited for.
- **Threads.** A trial's libraries are told to use up to four threads, fewer when more trials
  are running. After its first epoch the trial reports the cores it really keeps busy, which
  is about one, and the next trial's share is counted from that.
- **Out of memory.** A trial that runs out of memory on a card it shared is retried alone on an
  empty card, with everything that card has: a card is given, not a number, because a card
  also carries things that are not ours (a desktop session takes 600 to 1100 MB on this
  workstation) and a retry held to a number above that waits for ever. A trial that already had
  a card to itself is retried once with its inputs kept on the host, so the whole card is the
  network's. If that runs out of memory too, the trial fails -- it does not fit this card at its
  batch size -- and the study moves on to its next trial; with `--halve-batch` it is tried once
  more at half the batch. Nothing guessed after an OOM is kept for later trials.
- **A trial expected to need more than a card.** It is not refused. What a trial is expected to
  need is an estimate from other trials of its experiment, whose layer sizes were other ones, so
  it is tried: the log says `… is expected to need 12180 MB and the largest card offers 10964 MB
  usable. That is an estimate from other trials of its experiment, so it is tried alone on an
  empty card`, a card is cleared for it, and it starts there with everything the card has.
  After its first epoch it holds itself to what it used, and its card takes other trials again
  (`… was expected to need 12180 MB and holds 7936 MB: its card is no longer kept for it alone`).
  If it runs out of memory it goes the way of any trial that does (above). Until 24 November 2026
  such a trial was failed at once (`cannot run here and will not be retried`), and because a
  search learns nothing from a failed trial it proposed the same region again: in Total_wind an
  experiment that had trained at a batch of 471 the day before was refused at 408 to 501,
  eighteen times in one study. A unit three of whose trials in a row ran out of memory with a
  card to themselves is closed for the run (`… cannot run on this machine and is skipped`), and
  tried again by the next run.
- **A card that is lost.** When the driver reports a card lost (it fell off the bus:
  `NVMLError_GpuIsLost`), the run takes the card out: `gpu 0 is taken out of this run: the
  driver reports it lost … Nothing more is started on it`. The trials on it are stopped and
  told FAIL as `card_lost`. From then on a trial that dies of a driver error, or before its
  first epoch, takes its own card out too, because a driver that has lost one card is not to be
  trusted with the others; with no card left the run ends as failed (exit 1) and says `every
  card of this run was lost`. A lost card comes back only when the machine is restarted
  ([troubleshooting](troubleshooting.md#the-screen-is-dead-and-the-machine-is-not-a-graphics-card-was-lost)).
- **The attempts of a study.** A study is asked for at most twice the trials wanted of it
  (`n_trials`), which is what ends a study whose every trial fails. Every trial is one of those
  attempts except a failed one that says nothing about the hyper-parameters drawn: `cancelled`
  or `abandoned` (the run was stopped, or ended, under it), a trial a run that died left
  RUNNING, and `card_lost`. A study that has used up its attempts with trials still lacking is
  shown with what it gave up (`bin/eforecast studies`, the *will not run* column of the
  dashboard). Failures that turn out to have been the machine's after the study was told can be
  taken out of the count afterwards, in the study's own record
  (`optuna_search.excuse(study, trial numbers, why)`): a finished trial cannot be written to, a
  study can.

What was learned is saved in `<path_model>/exe_files/memory_estimates.json` (`footprints`,
`capacity`) and used by the next run of the case; delete the file to learn again. The run's
status (`orchestrator_status.json`) shows each card's busy share, what its running trials are
expected to take of it, and the learned trials per card.

Measured on this workstation (two RTX 2080 Ti) with twelve short MLP trials: 299 s at one
trial per card, 122 s at three per card, and 108 s with measured admission once it had learned
that a card takes four of them. The first run, which starts every kind at one per card, took
152 s.

#### Mixed precision

`mixed_precision: true` in a method's parameters runs the forward pass of training in half
precision on the card; CNN, TRANSFORMER and TIMM show the switch, turned off. The weights stay
32-bit, so everything that reads a trained network reads it as before, and validation is done in
full precision. The numbers of such a fit are not those of a full-precision one, which is why a
case has to ask for it.

What it gives depends on the network and on what else is on the card, so measure it on your own
case before leaving it on. A small convolutional network alone on an idle card made 56 training
steps a second where it had made 36, and held 410 MB where it had held 996. A real `cnn1` trial
of a wind case, on a card it shared with two other trials, gained nothing in time (16.8 steps a
second against 18.9), held 13% less of the card (2568 MB against 2944) and ended with a slightly
higher error (0.0644 against 0.0620 on the test days, one run each).

`--n-jobs N` is an operator's own bound and keeps the declared rules it was written for: at most
`N` trials per card, each sized by an estimate (1.5 × the median peak of earlier trials, a table
before the first). The same rules apply when a card's busy share cannot be read, with the
largest `n_jobs` of the methods as `N`. `EFORECAST_GPU_BUSY_TARGET=0` selects them too.

**Host memory is not estimated and nothing is set aside for a trial.** A case that has its
[scaled weather inputs](how_it_works.md#9-the-dataset-stages-one-by-one) maps them, so a
network that reads the weather holds the rows it trains on and the trials share one copy in the
system's file cache. If the machine runs out all the same, Linux ends the largest process, and
the scheduler acts on what it then knows:

```text
global/CNN 4 was killed (SIGKILL) with 6 trial(s) running. When nobody stopped it, that is the
kernel ending the largest process because the machine ran out of memory: this run now runs at
most 5 trial(s) at a time, and the trial is tried again
```

The bound comes down by one each time it happens and lasts for the run. A trial that is killed
when it was the only one running is tried twice more and then failed, with that reason in
`orchestrator_summary.json`. A trial that waits for this says so: `the machine ran out of memory
with more trials running; this run runs at most 5 at a time now`.

Studies persist in their journal/SQL storage. Completion and compatible artifact evidence,
not merely the existence of a results CSV, determines reuse. Read the final outcome and unit
failures: exit 0 can include `succeeded_with_failures`; it does not mean every requested model
completed. `orchestrator_summary.json` records that distinction, and the last lines of the run
sum the failed trials up by cause -- method, graph and error, the commonest first -- and say
when no trial of a graph ended well
([troubleshooting](troubleshooting.md#some-trials-failed-and-the-run-still-ended-well)).
`orchestrator_failures.csv` has a row a failed trial: `unit, trial, state, error, path, method,
graph`.

### 3. Run deep trials in Docker

Use a new tag for a candidate build; do not overwrite a live worker tag during qualification:

```bash
export WORKER_IMAGE=eforecast-worker:qualification-YYYYMMDD
# Replace YYYYMMDD with your chosen candidate name first.
docker build -f docker/Dockerfile.worker -t "$WORKER_IMAGE" .
```

The build context is the repository root. Docker uses
`docker/Dockerfile.worker.dockerignore`, not the inference-oriented root `.dockerignore`.
The base provides Torch/CUDA; `docker/requirements-worker.txt` is not a standalone native
installation recipe. GPU execution also needs a qualified driver/container-toolkit pair.

Before using a candidate, run the **synthetic CPU smoke**, which changes only a fresh scratch
folder and launches a disposable container:

```bash
export WORKER_SMOKE_DIR="$(mktemp -d /tmp/eforecast-worker-smoke.XXXXXX)"
docker run --rm --network none --user "$(id -u):$(id -g)" -e HOME=/tmp/smoke-home \
  --mount "type=bind,source=$PWD/docker/smoke_worker.py,target=/smoke/smoke_worker.py,readonly" \
  --mount "type=bind,source=$PWD/tests/fixtures/synthetic_case.yaml,target=/smoke/case.yaml,readonly" \
  --mount "type=bind,source=$WORKER_SMOKE_DIR,target=/smoke-artifacts" \
  --entrypoint python "$WORKER_IMAGE" /smoke/smoke_worker.py \
  --fixture /smoke/case.yaml --workdir /smoke-artifacts/run1
```

Expect exit 0, completed synthetic dataset stages and one real CPU training trial with weights.
The smoke also checks excluded private/credential/artifact paths. Preserve failure logs; do not
turn a failed smoke into a successful qualification by skipping its check. This does not
qualify GRIB, every model family or GPU fitting. The scratch directory is retained for inspection.

After runtime/image qualification and release-coherence checks, select that explicit image:

```bash
python run_training.py --config "$CASE_CONFIG" --docker --image "$WORKER_IMAGE"
```

The orchestrator stays on the host and mounts needed case/data/model paths consistently.
Managed model-family Docker runtimes use their own pinned contracts. Neither this option nor a
built image enables Kubernetes automatically; see [managed Kubernetes](../deploy/kubernetes/MANAGED_RUNS.md).
Use [image operations](administrator.md) for witness/release
refresh.

### 4. Train CPU and foundation studies deliberately

```bash
python run_training.py --config "$CASE_CONFIG" --cpu
python run_training.py --config "$CASE_CONFIG" --cpu --cpu-scope global --cpu-method RF
```

Each study (one method on one cluster, or on the whole case) runs in a process of its own, and
the studies run side by side: as many as the run has, up to one a CPU, sharing the CPUs the
case uses (`n_cpus`). Trials inside a study stay serial. `--cpu-workers` and
`--cpu-study-threads` set the numbers by hand. A study that fails does not stop the others.
`global`, `clusters` and `both` have distinct identities. A foundation method can use a card
even though it enters through `--cpu`. See [CPU scopes and studies side by side](cpu_training_scopes.md).

### 5. Combine, assess, then approve

```bash
python run_combine.py --config "$CASE_CONFIG" --dry-run
python run_combine.py --config "$CASE_CONFIG"
```

Inspect coverage and holdout results, not only the best trial score. Internal test scoring
and the later selection/calibration/holdout slices keep their established roles. Routine
forecast publication requires a [captured, replayed and approved bundle](user.md).
Finishing training does not approve a model.

### 6. Refitting, revision and debugging

`--refit` replaces the selected study/trial history; dataset `--recreate` can invalidate its
models. Use the UI consequence preview and back up important results first. **New revision**
uses additive generation paths; it is different from refit. See
[generation storage](cpu_training_scopes.md#storage-and-selection).

**Continue** keeps what is there, with one exception. A study whose finished trials were
trained under another configuration of its family -- a parameter or the search space changed,
or the datasets were built again -- is trained from the beginning: its trials, its journal and
its results table go first. That is `--refit` for that study alone, and it is named before it
is done: a run from the command line prints the list and goes on, a run started from the browser
shows it in its preview and starts when the case's name has been typed. Raising `n_trials` is
not another configuration: the study continues and
asks only for the trials it lacks. A family nobody changed is never touched, whatever its
siblings were given, and neither is any family when only the combination's settings changed
(`combine_methods`, `keep_best_models`, the intervals): those are read after training. See
[how it works §11](how_it_works.md#11-what-makes-an-old-trial-reusable).

`--prepare-only` writes deep trial inputs and prints the appropriate debugger command. A
`trial_runner.py` started by hand trains on the card `--gpu N` names by its order (the first
visible card when the flag is left out) and on the CPU with `--cpu`. It is not a run the web UI
knows about, and the scheduler of a run going on at the same time does not know about it
either: it sees only the memory the trial takes on the card.

### 7. Stop a training, and continue it later

Stopping is safe at any moment: nothing trained is lost but the trials that were in flight, and
the same command continues the case later. How you stop depends on how the training was started.

| The training was started | Stop it with |
| --- | --- |
| from the browser | the run's page under *Runs*: **Stop** |
| in a terminal you still have | `Ctrl-C` in that terminal, once |
| in a terminal, by you, and you are elsewhere | `python -m eforecast.config.direct_stage stop-mine <the case's model folder>` |
| as a service, to survive a closed session (a `systemd` user unit) | `systemctl --user stop <unit>` |

```bash
MODEL=/media/smartrue/HHD1/George/models/MyOwn/DemoCases_ver1/wind/Total_wind/long_term/model_ver1
python -m eforecast.config.direct_stage status "$MODEL"      # who is training this case, if anybody
# stage training of Total_wind, started 08:36:03 by smartrue (pid 3738052)
python -m eforecast.config.direct_stage stop-mine "$MODEL"   # ask it to stop; only a run of your own
python -m eforecast.config.direct_stage status "$MODEL"
# nothing is writing this case
```

The model folder is the case's `path_model`: `<sys_folder>/<owner>/<group>_ver<N>/<type>/<project>/<horizon>/model_ver<M>`.
`stop-mine` never touches somebody else's run; it names the person, and you ask them.

A service is stopped through its unit, because the unit would otherwise be told that its
program died. The chain that trained Total_wind in November 2026 ran as the user unit
`eforecast-resume`, which also starts at every boot until the chain has ended:

```bash
systemctl --user stop eforecast-resume       # stop it now
systemctl --user disable eforecast-resume    # and keep it from starting again when the machine restarts
systemctl --user is-active eforecast-resume  # inactive
```

**What a stop does.** Every way above sends the training the same request (`SIGTERM`; `Ctrl-C` is
`SIGINT`). The scheduler stops asking for trials, ends the ones in flight (they get 15 seconds,
then they are killed), tells each of them to its study as `cancelled`, writes its summary and
exits with code 2; its log ends with `run outcome: cancelled (exit 2)`. A cancelled trial is not
one of its study's attempts ([above](#how-many-trials-share-a-card-and-how-big-each-one-is)), so
stopping costs the study nothing. It takes a few seconds.

**Check that it has stopped.** `bin/eforecast status` shows no trial training, and
`bin/eforecast processes` no training of the case; `nvidia-smi` shows the cards free of it.

**Do not** end the scheduler with `kill -9`, and do not press `Ctrl-C` a second time while it is
stopping: it then dies without settling. Its trials stay `RUNNING` in their studies until the
next start fails them, and their processes and the card-sharing daemon may be left running;
`bin/eforecast processes` shows what is left, and those can be ended by hand.

**Continue.** Start the same command again (or press **Continue**, or
`systemctl --user enable --now <unit>` for a service). The studies go on from their files: the
trials that finished are kept, and the ones that were cancelled are asked for again.

## Monitor and recover

| What you need | Where to look |
| --- | --- |
| Accepted settings, stages and cleanup | Runs → accepted inputs, stages, resource report and log |
| Trial traceback/outcome | Attempt's `train.log`, `error.txt`, `trial_result.json` and completion manifest |
| The whole picture: what trains, where each trial is, what is left, what runs | `bin/eforecast status` ([cli](cli.md#status--the-whole-picture)); Grafana, dashboard *eforecast training* |
| What is training now, its error at every epoch, the loop it is in and its speed | `bin/eforecast trials` ([cli](cli.md#trials--what-is-training-now)); Grafana, dashboards *eforecast training* and *eforecast trial* |
| What every study has done and has left, the trials that finished last | `bin/eforecast studies`; Grafana, dashboard *eforecast training*; Studies & results |
| Every trial of every study with its curve and its parameters | The Optuna dashboard (port 8080), which the service `optuna-mirror` fills from the studies of the cases (below) |
| Why trials failed, by reason and one by one | Grafana, dashboard *eforecast training*, for the run that is going on (the orchestrator's own numbers, below); the training log |
| Which processes run, and what each uses of the processors, the memory and the cards | `bin/eforecast processes`; Grafana, dashboard *eforecast processes* |
| CPU, memory, disk and restarts of each container | Grafana, dashboard *eforecast containers* (cAdvisor) |
| Why a trial waits | The log line `<trial> waits: …`, and each card's use in the resource report |
| Model-quality report | `Results/summary.csv`, coverage tables and unsupported-cluster report |
| Which hyper-parameter values did best | `Results/hyper_parameter_report.txt` and `Results/hyper_parameters.csv`, written by every evaluation ([combine and evaluation](combine_evaluation.md#what-the-hyper-parameters-say-written-with-every-evaluation)) |
| Installation/service issue | `bin/eforecast doctor`, then [lab operations](administrator.md) |

While it runs, the orchestrator serves its state for Prometheus on port 9108 (`--metrics-port`,
`EFORECAST_METRICS_PORT`; `0` serves nothing): the counts of the run, each study's
`eforecast_unit_best_error` and `eforecast_unit_trials_total{state=complete|pruned|failed}`, the
failures `eforecast_trials_failed_by_reason_total{reason}` and, for the newest twenty,
`eforecast_trial_failed_info{unit,trial,reason}`.

The running trials and their error after every epoch come from the metrics service instead. It
reads them from the trial processes -- each one's folder, with the `progress.json` it writes after
every epoch, and its card -- so it sees every training, whatever version of the code started it,
a chain pinned to an older one included: `bin/eforecast trials` in a terminal, `/trials` on port
9109 for Prometheus. The series are `eforecast_trial_epoch`, `eforecast_trial_error` and
`eforecast_trial_best_error` (`set="val"` or `"test"`), `eforecast_trial_running_seconds`,
`eforecast_trial_last_epoch_age_seconds`, `eforecast_trial_loops_per_second` (training batches
per second in the last training pass, from the trial's `train.log`) and
`eforecast_trial_epochs_per_second`, labelled by case, cluster, method, trial and card; a
trial's series end when the trial does.

**Inside an epoch** (since 23 November 2026) the same log says where a trial is. Its last
progress bar gives the loop over batches it is in: `eforecast_trial_loop_done` of
`eforecast_trial_loop_total`, `eforecast_trial_loop_rate` (it/s now),
`eforecast_trial_loop_left_seconds`, and `eforecast_trial_loop_training` -- 1 while that loop is
a training pass, 0 while it evaluates, told by its number of batches. `eforecast_trial_epochs_total`
is the most epochs it may run, `eforecast_trial_log_age_seconds` how long since it last wrote
anything (a trial that hangs stops writing), and `eforecast_trial_cpu_seconds_total`,
`eforecast_trial_memory_bytes` and `eforecast_trial_gpu_memory_bytes` what it and its data
loaders use. Under MPS the driver does not say how busy one process keeps a card, only how much
of its memory it holds; the load is per card.

**The processes** are at `/processes` (job `processes`): every process of the pipeline, told by
its command line -- a training, each trial and its data loaders, a combination or dataset run
and its workers, the pages, the services, the daemons that share a card -- with
`eforecast_process_cpu_seconds_total` (a rate over it is processors in use),
`eforecast_process_memory_bytes`, `eforecast_process_threads`,
`eforecast_process_gpu_memory_bytes` and its reads and writes, labelled by role, case, cluster,
method, trial, name and pid; beside them the heaviest other processes (role `other`), the
machine (`eforecast_machine_cpu_seconds_total{mode}`, `eforecast_machine_load`,
`eforecast_machine_memory_bytes`) and `eforecast_gpu_thermal_slowdown` /
`eforecast_gpu_thermal_slowdown_seconds_total`: whether, and for how long, a card slows itself
down for heat, which the cards' own exporter does not report.

**The studies** are at `/studies` (job `studies`, once a minute). The trainings keep each study
in a file beside its models (`study.journal`); the metrics service reads those files, for every
case of the workspace, again only where one changed: `eforecast_study_trials{state}` (complete,
pruned, failed, running, waiting), `eforecast_study_trials_wanted` (the method's `n_trials`),
`eforecast_study_trials_left` (what the study will still be given: its running trials and the
ones still to be asked for, as far as its attempts last), `eforecast_study_trials_given_up` (what
it lacks beyond that) and `eforecast_study_failures_not_counted` (failed trials that are not its
attempts),
`eforecast_study_best_value`, `eforecast_study_trial_seconds` (how long a finished trial took on
average) and when the last one finished; and the forty trials that finished last, one by one
(`eforecast_finished_trial_value`, `..._timestamp_seconds`, `..._seconds`). A study whose
`failed` count grows while nothing else does is drawing trials that cannot run -- too large for a
card, for instance.

**The Optuna dashboard** reads a database, not those files. The compose service `optuna-mirror`
copies the one into the other about once a minute: it asks the metrics service which studies
changed (`/studies/index`, `/studies/export`) and writes them into the dashboard's database, a
running trial brought up to date after every epoch and then ended. It writes only into studies it
made itself, opens no model folder, and runs in the dashboard's own image
(`docker/monitoring/optuna_mirror/mirror.py`). Trainings that are pointed at the database
themselves (`EFORECAST_OPTUNA_STORAGE`) need no copier.

The dashboard lists its studies in one row and has no groups, so the copy is grouped: **one study
there for a case and a cluster**, `Total_wind / RBF_rule_04`, with the trials of every method of
that cluster in the order they began. (The trainings' own studies stay one for a method,
`<project>__<cluster>__<method>__<id>`; copied one for one they were 87 on 25 November 2026, and
the charts of a method that asks for a single trial had nothing to compare.) In such a study:

- the parameter `(method)` says which method a trial is of. Its name sorts first, so *Slice*
  opens on it: every trial of the cluster by method, which is where to see which family does best
  there. A trial's own parameters carry the method's name -- `MLP.learning_rate`,
  `CNN.cnn1_nwp_conv_1_filters` -- and a chart of one of them shows that method's trials only;
- the attributes `method`, `folder` and `source` say where a trial comes from: `folder` is
  `test_12`, the folder the training keeps it in, `source` the training's study and the trial's
  number there. **The number a trial has in the dashboard is its place in the cluster's history,
  not the number of its folder**;
- *Hyperparameter Importance* ranks the parameters of every method together, by the dashboard's
  own reckoning; read it as a hint. The values compared are each study's objective, `mae_test`.

A group is written anew when a training begins one of its studies again or a method joins the
cluster; the trials of a study the metrics service does not offer at that moment (a case whose
model folder moved) are kept as they are.

**The dashboards**: *eforecast training* is the page to look at -- what trains now with a bar
for its epochs and one for its loop, the curves inside the epochs, the studies with what is done
and what is left, the trials that finished last (with their hyper-parameters, read from the
Optuna database), the machine and its processes; *eforecast trial* is one trial in depth;
*eforecast processes* is the machine, the cards, every process and the containers.

PostgreSQL/Optuna, RabbitMQ and monitoring compose setup belong in
[lab operations](administrator.md). Configure credentials first; do not copy
an example password into a production DSN. Use `EFORECAST_OPTUNA_STORAGE` or protected service
configuration instead of a credential-bearing command line. PostgreSQL Optuna storage is
separate from the run store.

A trial that runs out of memory on its card fails that attempt and is retried as described
above. A run that was stopped leaves nothing to reclaim: its trials stop with it, and the next
run of the case starts as soon as the case's lock is free
(`python -m eforecast.config.direct_stage status <path_model>` names who holds it). See
[run recovery](administrator.md).

More on the processes a run starts is in [compute workers](compute_workers.md),
[how it works](how_it_works.md) and [the administrator guide](administrator.md).
