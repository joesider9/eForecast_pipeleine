# Troubleshooting

Find the symptom, go to the section. Nothing in the first half of this page deletes a result,
cancels anybody else's work, or changes a model. Where a step does change something, it says so.

| symptom | section |
| --- | --- |
| a run failed and I do not know why | [1](#1-a-run-failed) |
| `Busy: <case> is being written by another run` | [2](#2-a-run-will-not-start) |
| `BlockedStage: ... requires ...` before anything ran | [2](#2-a-run-will-not-start) |
| a worker failed and I want its reason | [2](#a-worker-failed) |
| `it was killed (SIGKILL)`: the machine ran out of memory | [2](#the-machine-ran-out-of-memory) |
| `scratch budget is occupied`, `a conflicting stage is still active`, `admission timed out` | [2](#messages-of-earlier-versions) |
| work was accepted but nothing is running | [3](#3-work-was-accepted-but-nothing-runs) |
| "Cleanup still running", and I cannot start again | [3](#3-work-was-accepted-but-nothing-runs) |
| the forecast looks old, or wrong | [4](#4-the-forecast-looks-old-or-wrong) |
| `latest.csv` is stale although the forecast was published | [4](#4-the-forecast-looks-old-or-wrong) |
| a model will not serve | [5](#5-a-model-will-not-serve) |
| the weather data stops before my period ends | [6](#6-the-inputs-are-not-what-the-case-expects) |
| `no sample of <variable> could be built`, or readiness says the lags cannot be met | [6](#no-sample-of-a-weather-variable-could-be-built) |
| `gpu N is no longer shared through MPS` | [GPU training](gpu_training_optuna.md#how-many-trials-share-a-card-and-how-big-each-one-is) |
| a stage stopped on an empty or missing weather file | [6](#6-the-inputs-are-not-what-the-case-expects) |
| the installation says something must be fixed | [7](#7-the-installation-itself) |
| an alert in Grafana or Prometheus, or a target DOWN on its *Target health* page | [7](#an-alert-in-grafana-or-prometheus) |
| tests fail after a library upgrade | [8](#8-after-an-upgrade) |

---

## 1. A run failed

### Note the run id

It is on the run in the UI, and in the support report. Everything below is easier with it, and
the maintainer cannot help much without it.

### Read the reason, not just "failed"

A failed run says what happened, what follows from it, and one thing to do. Most failures are one
of these:

| what it says | what it means | what you do |
| --- | --- | --- |
| a date, path or column is wrong | the case points at something that is not there | correct it, then **Check readiness** before starting again |
| the weather archive stops before the period ends | the period reaches past the data | shorten the period, or ask for the archive to be extended |
| a trial waits for a card | the cards are full, with this run's own trials or with another program | nothing — it starts when a trial finishes. If nothing starts at all, go to §3 |
| it was killed (SIGKILL) | the machine ran out of memory and Linux ended the largest process | usually nothing: the run retries it with less running at once. See §2 |
| cleanup still running | owned workers are being confirmed stopped | wait. **This is not a failure**, and starting again now can be refused |
| a method is not installed | the case selects a model this installation cannot run | clear that method, or ask for it to be installed |

**If the reason names a field, that field is the thing to change.** You do not need to understand
the stack trace underneath it.

### Correct it, then Continue

* **you changed the case** → re-check readiness first. A verdict from before your edit does not
  describe the case any more, and the panel says so;
* **nothing was wrong with the case** — a machine restarted, a disk filled, a worker died →
  first let the recorded run and its cleanup settle; a service restart may recover it by itself.
  Then preview **Continue**. It creates a new run and reuses whatever still passes the
  compatibility and completeness checks. Read the preview: it says what it would rebuild;
* **clicking twice is safe.** The same request reaches the same run.

**Do not delete the case or its folder to "start clean".** That throws away the frozen data split
and the fitted scalers. Reproducing the exact model inputs may need the original data,
configuration and runtime, and every model trained against the old split becomes incomparable.

### Hand it over

Send the maintainer: the **run id**; the **support report** from the run's log drawer (*Copy
support report*, or *Download support report* for a file); and what you were trying to do and
what you had changed since it last worked.

The report carries the case revision, the failing stage, the doctor's verdict, what the
installation looked like **when that run was accepted**, the worker and image identity, the
dispatch reference, and a bounded piece of the log. Recognised credentials are removed and no
dataset or model payload is included — but review its log excerpt and local paths before sending
it outside the lab, because a custom log message can contain things the redactor cannot
recognise.

### Stop and ask straight away if

* the same run fails twice in the same place after you corrected what it named;
* a result looks **wrong rather than missing** — a forecast that is flat, or far from reality;
* the doctor says **must be fixed**;
* anything mentions a disk being full, or a model file that cannot be read;
* you are about to do something described as rebuilding, invalidating or deleting.

That last one matters most. Waiting to ask costs an hour. The alternative can cost a retraining
campaign.

---

## 2. A run will not start

Nothing is reserved for a run and nothing has to be free before it starts: no slots, no memory
allowance, no disk budget. Three things can refuse a run before it does any work, and each says
what to do.

### The case is being written by another run

```
Busy: Total_wind is being written by another run: stage nwp of Total_wind, started 14:37:00 by
smartrue (pid 2253163). One run writes a case at a time; wait for it to finish, or, if it is
yours, stop it: python -m eforecast.config.direct_stage stop-mine <the case's model folder>
```

One run writes a case at a time. The lock is the kernel's and is given back the moment that
process ends, however it ends, so there is never a stale one to clear and nothing to reclaim.

```bash
python -m eforecast.config.direct_stage status <the case's model folder>      # who holds it
python -m eforecast.config.direct_stage stop-mine <the case's model folder>   # stop it, if it is yours
```

`stop-mine` asks the process to stop the way Ctrl-C does; its workers stop with it. It acts only
on a process that runs as you: somebody else's run is named and left alone, and you ask them.
Every way of stopping a training, and what a stop does to its studies, is in
[GPU training](gpu_training_optuna.md#7-stop-a-training-and-continue-it-later).

From the browser the same situation reads `case already has a running or unresolved run (<run
id>)`. Open that run: stop it, or wait for it and for its cleanup.

### `The configuration of <case> is not the one this was built from`

```
The configuration of pv_park is not the one this was built from.
  changed since it was built:
    data.variables[Cloud].lags: [0, 1, 2, 3, 4] -> [0, 1, 2, 3, 4, 5]
  built again, replacing what is there:
    nwp         1 file(s) -- its settings changed
    ...
  trained from the beginning, their trials removed first:
    TRANSFORMER 8 studies, 101 trial folder(s)
```

The case was edited after something was built from it, and the run replaces that work:
datasets made from other variables are built again, a family whose parameters or search space
changed is trained from the beginning and its trials are removed first. A case builds one
model, the model of one configuration, so the old work cannot stay beside the new.

Where you see it decides what it means:

* **In the browser, before a run starts** (or before a save): nothing has been changed. Type
  the case's name where it is asked to accept the list, or cancel and put the setting back.
* **In the log of a run started from a terminal**: the run is telling you what it replaces, and
  it goes on. It ends with `this run replaces them now`. If you did not mean the change, stop
  the run (`Ctrl-C`, or `stop-mine` above) and put the setting back — a stage that has not
  started yet has replaced nothing. `--dry-run` shows the same list and changes nothing.
* **As the reason a run from the browser failed** (`The service started this run without the
  answer of whoever submitted it`): nothing was changed. Start it again from the Runs tab, which
  asks.

Putting the setting back makes what is built this case's again, exactly as it is, and the next
run replaces nothing. The `changed since it was built` lines say which setting; a model built
before its configuration was kept (`<model folder>/.stage-artifacts/configuration/`) shows the
message without them.
* **You want both.** *Save as new case* gives the changed configuration a model of its own. For
  one family, the **new revision** action trains beside the old trials and keeps them.

A run started from the browser without the answer ends at once as failed, with this message as
its reason and nothing launched. Start it again from its preview.

### `BlockedStage`: the plan refuses the run

These are refusals of the plan. Nothing has been computed or changed when you see one.

| the message contains | what it means | what to do |
| --- | --- | --- |
| `stage lstm does not apply to this case (a long_term case has no sequence dataset)` | a method of this run reads sequences, and a `long_term` case has none | switch that method off for the case (a foundation model, or a graph with an `lstm` branch) |
| `<method> asks for <selection>, and the case computes no selector` | the method searches a feature selection that `methods.feature_selection_methods` does not list | name the selection in the case and run the `features` stage, or set the method's `feature_selection_method` to none |
| `<stage> configuration changed; rebuild this stage` | a setting that stage depends on differs from the one it was built with, and this run needs what that stage makes | run the datasets step for that stage and accept the change it asks about (above), or put the setting back |
| `the trained models of <family> are of another configuration` | the family's parameters or search space changed after it was trained; what it left is not part of this configuration's model, so nothing is predicted or combined with it | train the family again and accept that its trials are removed (above), or put its parameters back. Switching the family off also lets the combination run without it |
| `<cluster> <family>: … finished trial(s) of its study were trained under another configuration` | a script started the training functions directly, without the step that asks | start the training with `run_training.py`, which shows what would be removed and asks; `--refit` trains the family from nothing without asking |

### A worker failed

Clustering, a CPU study, prediction and a few other pieces of work run in a process of their
own ([compute workers](compute_workers.md)). One that fails says why, and where its log is:

```
fuzzy fit failure: ValueError: attempt to get argmax of an empty sequence (worker log: /home/.../worker.log)
```

The text after `failure:` is the last exception in that log; the log has the rest. In CPU
training a study that fails does not stop the others: the run ends `partial`, the failed study
is named with its reason, and **Continue** trains it again once the cause is fixed.

### The machine ran out of memory

```
failure: it was killed (SIGKILL); when nobody stopped it, that is the kernel ending the largest
process because the machine ran out of memory (worker log: …/worker.log)
```

Since nothing is set aside, memory can run out, and Linux then ends the largest process. What
the pipeline does about it by itself:

| Where | What happens |
| --- | --- |
| CPU training | the study that was killed is started once more, alone, when the other studies have finished |
| GPU training | the trial is tried again, and from then on the run starts one trial fewer at a time than were running when it happened (`this run now runs at most N trial(s) at a time`) |
| a reader of a dataset stage (`extract`, `nwp`, images) | the stage goes on with half as many readers and does the work that was in flight again (`with 15 readers running, one was killed (SIGKILL) … done again with 7 reader(s)`); with one reader left it fails, saying so |
| clustering, prediction | the stage fails with the message above; nothing half-written is taken for a result |
| the stage's own process (`features` fits its selectors in threads of it; `nwp` holds the dataset it is building) | the shell prints `Killed`, a run started from the browser ends as failed, and nothing half-written is taken for a result |

If it keeps happening, start less at once: fewer CPU studies side by side (`cpu_workers`, or
`--cpu-workers`), fewer dataset readers and fewer feature selectors fitted at once (the case's
`n_jobs`: both are as many as it says, whatever memory the machine has), one candidate width of
the clusterer at a time (`clustering.n_jobs: 1`), fewer trials on a card (`--n-jobs`), a
smaller batch. Closing whatever else is using the machine helps as much. The kernel's own log
confirms that it was the kernel: `journalctl -k | grep -i "out of memory"`.

A GPU trial that runs out of memory **on its card** is a different message (`CUDA out of
memory`) and is retried alone on a card; see [GPU training](gpu_training_optuna.md).

### Some trials failed, and the run still ended well

GPU training ends `succeeded_with_failures` when trials failed and others did not. A failed
trial is replaced: the study asks for another until it has its number of trials, and stops
asking at twice that number. The last lines of the run say what failed and how often, the
commonest cause first:

```
22 failed trials, see …/logging/training/orchestrator_failures.csv
  22 of them: MLP with graph mlp1, 'int' object is not iterable. No trial of mlp1 ended well in this run: its dry build in the graph editor shows whether the graph can be built as it is
```

`No trial of … ended well` is the line to act on. Then it is not a draw of the search but the
graph itself, or what it is given, and the search goes on without that graph: the method is
trained on its other graphs only. Open the graph in **Experiments** and press its dry build; a
layer that cannot be built is named there with its error. **Check readiness** names what it can
see without building (below). A cause that some trials of the graph survived is a draw: a
combination of sizes that does not fit, a card that was full.

The table has one row a failed trial, with the method, the graph and the folder of the trial;
its `error.txt` there holds the whole traceback.

### A layer parameter holds a value the layer does not take

```
output[0] cross_attention: attention_query is 64, and the layer takes branch names there
```

Readiness and the dry build of a graph both say this (the check is `applicability.parameter_value`).
A layer parameter is a number, text, yes or no, or -- `attention_query` -- a list of branch
names, and the graph editor now keeps each to its kind. A case written before 14 November 2026
can carry another: until then **add a parameter** started every parameter that has no default
at the number 64, `attention_query` among them, and there was no way to enter a list. A trial
that drew such a graph ended at once in `'int' object is not iterable`.

What is done about it without you:

| The value | What the layer does |
| --- | --- |
| `attention_query` holds something that is no name (a number, yes or no) | it is read as not set: the flat branches are the queries, as when the parameter is absent |
| `attention_query` names a branch the graph does not have | the other names are used; when none is left, a learned query pools every branch |
| any parameter with an empty value | it is not set, and the method's value or the layer's default stands; an empty field never stands in front of them |
| a number where text is taken, text where a number is, and the like | the layer is handed the value as it stands, and one that computes with it fails as the network is built |

So nothing fails any more for the 64, and nothing has to be done about it: such a graph never
trained before, and now it trains as if the parameter were absent.

To clear the finding, open the layer in **Experiments** and remove the parameter with its ✕, or
pick the branches you mean -- **in a case that has no trained models yet**. A change to any graph
of a case changes what the case is for every method of it, also for the methods that do not use
that graph, and the next run trains all of them again; the plan shown before a run starts says
how many trials it would keep. In a case that is trained, leave the value where it is.

### `… was not predicted on a card`

```
RBF_rule_2 TiRex test_0 was not predicted on a card (OutOfMemoryError: CUDA out of memory); it is tried on one again after the others
RBF_rule_2 TiRex test_0 was not predicted on a card (OutOfMemoryError: CUDA out of memory); it is predicted on the CPUs
```

The `predict` stage of the combine step predicts its foundation models (Chronos-2, TimesFM,
TiRex) on the cards, and a card could not take this one: something else holds its memory, or its
driver failed. Nothing is lost. The trial is tried on a card once more when the cards have done
the rest, and after that it is predicted on the CPUs, so the stage still finishes -- only later,
and for a whole case much later: a card forecasts these models fifty times faster.

If every foundation trial of a run says this, look at what is on the cards (`nvidia-smi`): a
training run of another case, a notebook, a desktop application that keeps video memory. Run
the combine step when the cards are free, or keep these families on the CPUs on purpose with
`cpu_study_device: cpu` under `jobs`.

### `GPU ids [0] are unavailable; CUDA sees 0 device(s)`

The deep-training step has something to fit, and the machine it was started on does not have
the cards the run names: a cluster node without a GPU, a terminal where `CUDA_VISIBLE_DEVICES`
hides them, a driver that is not loaded (`nvidia-smi` says). Nothing was touched; the check
comes before any folder or study is. Start the step where the cards are, or name the ones that
are there with `--gpus`.

A training step that has nothing to fit does not ask. When the plan finds the stage whole —
`training  reuse  the budget is satisfied by 225 compatible trial(s) …; content and input
identities match the receipt` — the stage is reused and the run ends well on any machine. (Until
17 November 2026 the cards were asked for first, and the finished case of a card-less cluster
stopped with this message.) So when you see it on a case you believed finished, the plan above
it says what is left: trials still wanted, transfers to train, a family trained from the
beginning.

### The disk is full

A write fails with the system's `No space left on device`. A file that was being written has a
`.tmp` name and is nobody's dataset, the stage is not recorded as done, and the next run
rebuilds it. Free space and run again. The scaled weather inputs are the one thing that looks
first: `scaled inputs need 8.8 GB and <folder> has 3.1 GB free; they are not written and
readers use the datasets as before` is a notice, not a failure.

### A model folder that was copied or moved

```
…/model_ver1/clusters.pickle names 7 cluster folder(s) of another model tree (/…/rule_0 ...); this tree was copied or moved, and its own folders under …/model_ver1/cluster_organizer are used
```

The clusters of a case are kept with the full path of their folders. A model folder that was
copied to try something, or moved to another disk, therefore still names the folders of the
place it came from. Everything that reads the clusters -- prediction, selection, training,
feature selection -- now works in the folders of the tree it was given, and says so once. It
never reads or writes the tree the file was written in.

```
ForeignTree: …/clusters.pickle names the cluster RBF_rule_0 at /…/rule_0, outside this model tree, and the tree has no folder of its own for it
```

is the same situation without the `cluster_organizer` folder: there is nothing of the copy's
own to work in, so nothing is done. Copy that folder too, or run the clustering stage for the
case.

A copy is still not a case of its own for the stage scripts: its stage records name the
original's files, and `run_datasets.py`, `run_training.py` and `run_combine.py` refuse it
(`names a foreign output location`). To try something on a finished case, make a case with
another `sys_folder` and build it from the datasets.

### `… inputs changed …` in a folder that is synced

```
ValueError: inference inputs changed during execution
IncompleteStage: stage inputs changed during execution; nothing was recorded
ValueError: trial inputs changed during fitting; private result was not published
```

A run watches the files it reads and publishes nothing when one of them changed while it
worked. It watches without reading them: device, inode, size, modification time and *change
time*, the one clock that cannot be put back and so gives away a file that was rewritten under
its old size and time.

A sync client moves that clock too. Dropbox sets an attribute on every file it has uploaded --
a moment after the file was written, and again when the upload ends -- and that is a change of
the file's change time with no change of its content. Until 17 November 2026 the guards took it
for a change: a combine run started straight after a training run in a model folder inside
Dropbox ended with the first message above, as often as the client was still catching up, and
passed when it was repeated after `dropbox status` said `Up to date`.

Now a change time that moved *together with the file's own attributes* (its extended
attributes, permission bits or link count) is a change of metadata and the file is the same
input. Everything else is still a change and still ends the work: another inode, another size,
another modification time, a file that appeared or went, and a change time that moved with
nothing to explain it. The message names the files:

```
ValueError: inference inputs changed during execution: /…/DATA/dataset_row_data.pickle, /…/scalers.pickle and 3 more
```

So when one of these messages appears now, something did write to the named files while the
run worked -- usually a second run on the same case, or a script that refreshes the series.
Find it, and run the step again.

What this does not change:

- **The limit of a guard that reads nothing.** A file that is rewritten under its old size and
  modification time *and* has its attributes changed while a run works is not told apart from
  one that only had its attributes changed. The dataset stages do not rest on it; they compare
  the SHA-256 of what they read.
- **Between runs nothing is explained away.** A file whose change time moved since its stage
  recorded it is read again before it is used (`metadata changed; full content verification
  required` in a plan). In a synced folder that is every file a stage wrote, once. Keep a
  case's large inputs -- the weather folder above all -- outside synced folders if the reading
  costs time.
- **The weather conversion utility** (`convert_nwp_pickles`) replaces the files it watched and
  keeps the strict comparison. Run it when the sync client is idle.
- **Filesystems.** The attributes are read on Linux. On ext4 an attribute that is set to the
  value it already has does not move the change time; a filesystem on which it does would
  still show such a file as changed.
- **Images of older code.** The host and a worker compare what each of them saw. A worker
  image built before this change compares the old way, and every prediction it is asked for
  ends with `inference inputs changed before worker loading`. Rebuild the images with the code
  (`bash docker/refresh_images.sh <release>`); `python -m eforecast.image_release check` says
  when they are behind.

The run database and the workspace do not belong in a synced folder at all
([`EFORECAST_RUN_STORE_PATH`](cli.md#where-records-are-kept)).

### What a stage leaves in `.stage-artifacts`

`receipts/` (what lets the next run reuse a finished stage), `<stage>.running` (a stage that
was interrupted, and will be done again), `holder.json` while a run holds the case, and beside
a dataset `<name>.previous` (the output a rebuild set aside, restored if the rebuild failed).
An `attempts/` folder left by an earlier version can be deleted.

### Messages of earlier versions

These came from a ledger of reservations that no longer exists:

```
AllocatorBusy: a conflicting stage is still active
scratch budget is occupied on <volume>          filesystem headroom is insufficient on <volume>
CPU study admission timed out after 300s        dispatch policy differs for volume:…; use the existing pinned limits
ResourceUnavailable: data nwp failure; private attempt log retained
```

If one of them appears, the process that printed it is running the old code: most often the
web service was not restarted after the upgrade (`sudo systemctl restart eforecast-webui
eforecast-metrics`), or a terminal is using another checkout. Nothing has to be reclaimed or
released; `python -m eforecast.config.stage_dispatch …` is gone with the ledger.

---

## 3. Work was accepted but nothing runs

A run does not wait for room on the machine, so there are few reasons left, and the doctor
tells the first of them:

```bash
bin/eforecast doctor
```

**Nobody is advancing the work.**

```
[WARN] services.coordinator: nothing holds the service lock for this workspace
       so: a submitted run stays recorded but nothing advances it, so work appears to
           wait forever
       do: Ask the maintainer to start it   (maintainer)
```

The record is durable and nothing is lost; it resumes when the coordinator is started.

**Another run has the case.** A second run of a case is refused at once rather than queued
(§2). A run that is shown as accepted and not started is waiting for the service, not for the
case.

**Cleanup is still running.** This is not a failure. A run is finished when its children are
confirmed stopped, not when it stops. Starting the same work during cleanup can be legitimately
refused. Wait for it.

**GPU training started and no trial starts.** The log says why each trial waits (`<trial>
waits: …`): every card is full with this run's trials, or with another program's memory, or the
run holds itself to fewer trials after running out of memory. A trial no card of this machine
could hold even when empty is failed at once with what it needs. If nothing starts for
`EFORECAST_NO_PROGRESS_SECONDS` (900) with nothing running, the run stops and says so rather
than wait all night.

**CPU training shows studies as waiting.** `waiting — its turn has not come` means the studies
that run at once are running; the rest start as those finish. `cpu_workers` sets how many.

To see what the installation currently believes it is running:

```bash
python -m eforecast.config.run_processes
```

---

## 4. The forecast looks old, or wrong

```bash
bin/eforecast forecast-status <case>
```

You get **two separate answers**, and the difference between them is usually the whole story:
what is **published**, and whether the last **attempt** worked. A published forecast with a
failed attempt means the last accepted numbers remain — and they may be too old for the period
you need. A forecast marked `STALE` means nobody has produced a valid one for a while, and the
attempt line says why.

The same three fields are in `status.json` next to the forecast files: `state`,
`publication.reasons` and `warnings`.

### What the reasons mean

| it says | what happened | whose |
| --- | --- | --- |
| `empty` | the run produced no rows, usually because inputs were missing | maintainer — the weather or target data did not arrive |
| `partial` | fewer hours than the horizon needs. **Not published**, because half a day is not a forecast | maintainer |
| `nonfinite` | a `NaN` reached the output | maintainer |
| `partial` with timestamp or column detail | the dates, order or horizon columns differ from what was requested | maintainer |
| `units` | the numbers are implausible for this site's rated capacity | maintainer |
| `quantiles` | a band the case expects is missing | maintainer |
| `stale_fence` | the attempt has no current verified completion or qualified model hold | maintainer — inspect it; do not bypass the gate |
| `superseded` | a newer forecast already exists. Nothing is wrong | nobody |
| `backfill` | somebody deliberately filled a past day. Nothing is wrong | nobody |
| `undeclared_correction` | something tried to replace an existing forecast for the same time without saying so. **Refused on purpose** | maintainer |
| `failed` | the run crashed before producing anything | maintainer |
| `cancelled` | somebody stopped the run | nobody, unless it keeps happening |

A `rejected` run is not a broken installation. It is the protection working.

### The old forecast is still there

**A failed run never overwrites a good forecast.** `latest.csv` and the published files keep the
last valid one until a new one passes every check. Check the retained forecast's issue time and
horizon before using it.

### `latest.csv` is stale although the pointer moved  *(maintainer, changes state)*

That is the case where the database accepted the forecast but the output filesystem failed.
Restore access, then:

```bash
bin/eforecast forecast-status <case> --repair-export /path/to/predictions
```

It replaces `latest.csv` from the current durable pointer and checks the retained payload's
digest. It cannot promote an older attempt over a newer issue, and it never republishes a
forecast. If the retained bytes are missing or corrupt, that generation must be restored — not
substituted with a candidate CSV.

---

## 5. A model will not serve

Two likely causes:

**The image is stale.** The doctor's image-coherence finding names it. A source file an image
carries has changed, so running on that image would not be running the code the release was
qualified with. The maintainer runs `bin/eforecast images refresh`.

**The approval is a legacy one.** `bin/eforecast approved <case>` says what is in use. A legacy
approval is labelled as such and needs a fresh reference captured and approved before it serves.

To go back to the previous version *(maintainer, changes state)*:

```bash
bin/eforecast approved <case>      # which model is in use, and what preceded it
bin/eforecast rollback <case>      # back to the previous approved model
```

Rollback loads and replays the retained model before moving the pointer. It does not retrain.

---

## 6. The inputs are not what the case expects

**Run readiness before a long run, not after it.** The **Review & save** tab checks the coverage
of the target series, the declared variables against what the files actually hold, and timestamp
ambiguity, and names the field each finding belongs to.

**A missing or zero-byte weather file does not stop the run, and is not quietly ignored
either.** Why it cannot simply be skipped is worth understanding. Some weather fields are
*accumulated*: the value at hour 6 is the total since the forecast run began, so the pipeline
subtracts the previous step to get the hourly value. A missing step means the next step's
subtraction spans two hours instead of one — which roughly **doubles** that hour's value. For a
solar case that is the main input and also the clustering variable. A gap in an accumulated field
does not produce a gap in the output; it produces a plausible, wrong number.

So a run handles one in three steps:

1. **It fetches the file again.** A file with no bytes is almost always a download that stopped
   partway, and the archive it came from still has it. This happens before the stage records what
   it will read, so a refilled file counts as a real input rather than a gap. The route is the
   `rclone` remote `gdrive:ECMWF`, one named file at a time, falling back to the public shared
   folder and a whole day at a time. `EFORECAST_NWP_REFILL=off` keeps a run off the network;
   `EFORECAST_NWP_REMOTE` names a different remote.
2. **It names what is still empty, once and in full,** and continues.
3. **The extractor drops that step and the first readable step after it,** rather than difference
   across the hole. One missing hour therefore costs two hours of that day's accumulated fields —
   and nothing wrong is published. The dropped hours are usually covered anyway: with
   `nwp_selection_policy: as_of` the dataset tries the cycles of this day and the day before,
   newest first, and a sample with no weather at all is a case it already handles.

Deleting the empty files changes nothing: they are already skipped. Steps beyond the extractor's
`max_step_hours` (60 hours) are never read at all, so an empty file past that costs nothing — the
step is in the file name, which is the quickest way to tell whether a gap matters.

**A whole day with no weather in it does not stop the extraction either.** Sometimes a day never
arrived: its folder is absent, or holds nothing but empty files, or only steps past the horizon.
No pickle can be made from such a day however often the extraction is repeated, so the stage says
which days they are — `extract ecmwf: N days without a pickle` in the log, and a listing naming
each one — and publishes the days that do hold weather. The dataset then treats those days as
samples without weather, which is the case it already handles. Fetch the days from the provider if
you want them in the dataset; the extraction will pick them up on the next run and leave the days
already extracted alone.

**Find all of them in one go.** A run names every unusable file it found, not just the first, so
one run is a complete list. To look without starting a run:

```bash
find <your nwp root>/new/data -type f -size 0 | sort
```

Two things that look like the same problem and are not. A file sitting directly in `data/`
instead of in a `data/<YYYYMMDD>/<cycle>/` folder is never read, so it neither helps nor blocks —
the extraction only walks day folders. And a cycle folder holding files whose *names* carry a
different cycle is a misfiled copy. Only the first copy of each cycle-and-step is read now, and
the repeat is named in the log; before that, reading both let the second overwrite the first with
a value differenced against a step already past — a duplicate did not repeat a number, it
replaced a good one with a wrong one.

**`extract …: N of M day(s) were not downloaded from Google Drive`.** The case reads its weather
from the shared Google Drive folder (**Project → Nwp Models → source**), and some days did not
arrive. The step stops rather than carry on, on purpose: a day that never arrived would be
recorded as a day without weather, and with no raw file of it on this machine nothing would ever
ask for it again. Nothing is lost — the days that were converted are kept. The log names each
day and what Google answered for it:

* *Too many users have viewed or downloaded this file recently* — Google's own limit on a shared
  file. It lifts by itself, usually within a day.
* *HTTP 404* — the file or the day folder is no longer in the shared folder. Check the link.
* a timeout or a reset connection — the network. Each file is tried three times first.

Then start the datasets step again: it asks only for the days it lacks. When several days in a
row are refused it stops asking for the rest, so that a folder that has stopped answering costs
minutes and not the afternoon. `EFORECAST_NWP_REFILL=off` keeps the run off the network
altogether; the dataset is then built without the days that are not on the machine.

An *empty* file in the shared folder is not a failed download. The archive holds a few, and they
are treated as empty files always are — see the three steps above.

**A timezone that reads wrong.** *Local timezone* describes the source timestamps; *Site
timezone* describes the location. Readiness checks the Local one. `CET` is treated as UTC+1 in
winter and UTC+2 in summer; a source that stays at a fixed offset all year needs a fixed-offset
name such as `Etc/GMT-1`. Confirm what the source file actually does before relabelling, and note
that an ambiguous timestamp is **not** proof that existing data was shifted or dropped.

---

### No sample of a weather variable could be built

```
ValueError: no sample of Cloud (ecmwf, park) could be built from 775 day(s) of weather. A sample
needs every one of its 73 lag(s), from 0 to 72 hours counted from each predicted hour, and the
forecasts a row may use end about 72 hours after its issue day begins.
```

A sample needs weather for **every** lag of the variable, and a forecast is kept for 60 hours
after its analysis. Which hour a lag is counted from depends on the kind of case:

| case | a lag is counted from | lags that work |
| --- | --- | --- |
| `short_term` | the issue time of the sample | the hours of the horizon: `[0, 1, 2, 3, 4]` for four hours ahead |
| `long_term` | **each predicted hour**: a row is one predicted hour of one issue day | the hours around it: `[-1, 0, 1]` |

So a list of every hour of the horizon, right for a short-term case, leaves a long-term case
without a single sample: the first predicted hour of a day would already need the weather of
three days later. **Check readiness** says so before a run (`the lags of <variable> can be met by
a forecast`), at the field to change, and says how many predicted hours are left when the lags
are long but not impossible.

### `clustering sample limits are infeasible`

```
ValueError: clustering sample limits are infeasible for 1152 training rows: min_samples=8000,
max_samples=518; reduce min_samples or increase max_samples_ratio
```

A cluster must hold at least `min_samples` rows and at most `max_samples_ratio` of the training
rows, and the first cannot be more than the second. It happens to a case copied from one with a
finer series: 8000 rows is a ninth of two years of quarter-hours and nearly half of the same two
years by the hour. Lower `min_samples` (Methods → clustering). **Check readiness** says so
before a run and names the largest value that fits (`a cluster of min_samples rows fits in
max_samples_ratio of the training rows`). It counts the timestamps of the series before the
evaluation period, less the night hours when the case removes them -- a training run keeps the
hours from three before dawn to three after dusk at the site, about four fifths of a year in
Greece -- less the share the split holds out. Hours without weather are dropped later and are
not counted, so leave a margin below the value it names: for a park of 20,232 hourly rows it
named 3,713 where the run, with its days without weather, had room for 3,078.

### Readiness said every cluster has 0 validation days

Until 2026-11-14 it read the wrong field of the partition report, and a case showed `cluster
RBF_rule_0 has 0 validation and 54 test days` for each of its clusters, and `cannot start`, as
soon as its clustering stage had run. The clusters were fine. Check again after the upgrade.

### `scaled inputs of <case> are not used: …`

Not an error: the run goes on and reads the datasets as it did before these files existed, more
slowly and holding more memory. The line names the reason, and each has one answer.

| the line says | what happened | what to do |
| --- | --- | --- |
| `not written yet` | the case's datasets are older than this feature, or its folder was removed | nothing: the next `run_datasets.py`, `run_training.py` or `run_combine.py` writes them |
| `nwp changed` / `scalers changed since they were written` | the weather dataset was rebuilt, or the scalers were fitted again | the same; a `preprocess` run does it at once |
| `… is not the file its manifest describes` | a file in the model folder's `scaled_inputs` was replaced or cut short by something else | delete that `scaled_inputs` folder; the next run writes it whole |
| `written in format N, this code reads M` | the code changed how it writes them | nothing: they are written again |

`scaled inputs need 30.2 GB and … has 12.0 GB free; they are not written` means what it says:
they take as much disk as the weather dataset for each scaling method the case names. Free the
room, or leave them out — every reader works without them. `scaled inputs: no fitted scaler for
some <method> inputs` means the case names a scaling method the `preprocess` stage has not
fitted; run it.

To rule them out while looking for something else, `EFORECAST_SCALED_INPUTS=0` makes one run
read the datasets directly. The inputs are the same to the last bit either way, so the
prediction of a trained model that differs with it is a bug to report. (Two trainings on a GPU
differ a little from each other with or without it.)

## 7. The installation itself

```bash
bin/eforecast doctor                  # everything
bin/eforecast doctor --case <case>    # and one case's models too
bin/eforecast doctor --verbose        # including what passed
```

`0` ready, `1` needs attention, `2` must be fixed, `3` the doctor itself could not run.

It reads only: it installs nothing, restarts nothing, changes no model. `--snapshot` is the single
exception, and writes one file.

Every finding carries one remedy and says whether it is **yours** or the **maintainer's**. A
finding marked the maintainer's is not for you to fix.

Two things the doctor does *not* mean:

* *Installation ready* means the checks passed at the displayed time. It does not rule out a
  case-specific dependency, a later outage, missing data, or an untested model path;
* the health chip and **Check readiness** answer different questions — the installation, and this
  case's inputs. Include both when reporting a problem.

Use `bin/eforecast doctor`, not `python -m eforecast.config.doctor`: the wrapper reads the
installation's own settings, so the answer is about the lab rather than about your terminal.

### The screen is dead and the machine is not: a graphics card was lost

**What it looks like.** In the middle of a training the screen freezes or goes black, the
remote desktop stops answering, and nothing brings them back. The machine itself is running:
the web pages still answer, `ssh` still works.

**What happened.** A card stopped answering on its slot. The kernel log of that hour says so:

```bash
journalctl -k -b -1 --grep "NVRM: Xid"      # -b -1: the start of the machine before this one
# NVRM: Xid (PCI:0000:17:00): 79, GPU has fallen off the bus.
# NVRM: Xid (PCI:0000:17:00): 154, GPU recovery action changed from 0x0 (None) to 0x2 (Node Reboot Required)
```

The trials that were on the cards end with `CUDA error: unspecified launch failure` or
`cuDNN error: CUDNN_STATUS_EXECUTION_FAILED`; those are the consequence, not the cause. A
training takes the lost card out of its run and starts nothing more on it (`gpu 0 is taken out
of this run`), tells the trials it ended to their studies as `card_lost`, which does not count
against the study, and ends as failed once it has no card left; the next run continues those
studies. (A training from code older than 24 November 2026 went on starting a trial every few
seconds on the dead card, each one a failed attempt of its study: on 19 and 23 November 2026
that closed studies of Total_wind short. `optuna_search.excuse` takes such trials out of the
count afterwards.) The
desktop goes with the card because it is drawn through the same driver, and the driver waits
for the lost card for good. Only a restart of the machine ends that.

**What to do.**

1. Restart the machine: over `ssh`, `sudo nvidia-bug-report.sh` first (NVIDIA's record of the
   fault is lost at the restart). Do not rely on `sudo reboot`: an ordinary shutdown waits for
   the programs the lost card left stuck in the driver, and on 19 November 2026 one hung for 23
   hours after it had stopped every service. Where the installation has the watchdog of
   [deploy/README.md](../deploy/README.md#optional-restart-the-machine-when-a-graphics-card-is-lost),
   the restart has happened by itself a minute after the fault and the report is in
   `/var/lib/eforecast-gpu-watch/reports`; by hand,
   `sudo /usr/local/sbin/eforecast-gpu-lost-reboot --restart-now` does the same (it stops
   Docker, writes everything to disk and restarts through the kernel). A machine that froze so
   completely that neither runs is restarted by the board's own watchdog timer about five
   minutes later, where that is enabled
   ([deploy/README.md](../deploy/README.md#optional-the-boards-own-watchdog-timer)).
2. Start the run again. It continues from what it had finished.
3. Find out which card it was and whether it is fit to work — before it is given work again.
   The PCI address in the message is the card (`nvidia-smi --query-gpu=index,pci.bus_id,uuid
   --format=csv` names them). While a training runs, ask each card whether anything holds it
   back:

   ```bash
   nvidia-smi -q -i 0 | grep -A 12 "Clocks Event Reasons"
   #     HW Thermal Slowdown        : Active        <- the card is too hot somewhere
   #     HW Thermal Slowdown        : 74721748 us   <- for how long since the machine started
   ```

   `HW Thermal Slowdown` on a card whose temperature reads 60 °C means a part of it other than
   its core is too hot — its voltage regulators or its memory — or a sensor is faulty.
   `HW Power Brake Slowdown` means its power supply. A healthy card under full load shows
   neither; `SW Power Cap` now and then only says it is working at its limit. This is how the
   card of 15 November 2026 was found: three quarters of its working time in hardware thermal
   slowdown, at 62 °C and half the power of its twin.

**Training without that card, until somebody has looked at it.** From a terminal, name the
cards the training may use, and show every other step the good card alone — a clusterer or a
foundation model takes the card with the most free memory, which is the idle one:

```bash
python run_training.py --config <case> --gpus 1
CUDA_VISIBLE_DEVICES=GPU-<uuid of the good card> python run_combine.py --config <case>
```

For the runs people start from the browser, the same line in the installation's settings file
(`CUDA_VISIBLE_DEVICES=GPU-<uuid>` in `/etc/eforecast/webui.env`, then restart the two
services) hides the card from all of them. A case that asks for more cards than are shown
(`jobs.n_gpus`) then refuses its deep training with
[`GPU ids [1] are unavailable; CUDA sees 1 device(s)`](#gpu-ids-0-are-unavailable-cuda-sees-0-devices)
before it touches anything, which is the right answer until its `n_gpus` says what is there.

### An alert in Grafana or Prometheus

Prometheus (`http://<workstation>:9090`, page *Alerts*) evaluates the alerts of
`docker/monitoring/prometheus/rules/eforecast.yml`, and each carries its consequence and remedy.
What they mean, and what to do:

<a id="the-dispatch-outbox-cannot-be-read"></a>
**The dispatch outbox cannot be read** (`EforecastDispatchSourceUnreadable`). The run store is
unreachable or refusing, so nothing can be said about queued work, and no backlog alert can fire
while this is true. `bin/eforecast doctor` reports the store. Nothing is lost: dispatch records
are durable and are published under the same identity once the store answers.

<a id="eligible-work-is-not-being-published"></a>
**Eligible work is not being published** (`EforecastDispatchStuck`). Tasks are recorded but do not
reach a worker, so runs show a dispatch wait rather than failing. Check the broker first; if it
is up, the publisher is not draining.

<a id="deliveries-are-being-held-open"></a>
**Deliveries are being held open** (`EforecastDeliveriesHeldOpen`). A short dispatcher claims and
acknowledges in milliseconds, so this is zero in normal operation. A delivery open for fifteen
minutes means a consumer is working inside the acknowledgement: restart it. The claim it already
committed is durable and is not repeated.

<a id="messages-that-are-not-dispatches"></a>
**Messages that are not dispatches** (`EforecastUndeliverableMessages`). Something publishes to the
dispatch exchange what this version cannot read. Such messages are recorded and dropped, so
nothing is lost, and nothing is done by them either. Usually publisher and consumer are of
different versions.

<a id="the-broker-cannot-be-reached"></a>
**The broker cannot be reached** (`EforecastBrokerUnavailable`). New work stays recorded and
unpublished; nothing is lost or duplicated, and runs show a dispatch wait. Start the broker, and
the recorded work is published by itself. Only where the installation uses a broker (the metrics
service has `EFORECAST_BROKER_MANAGEMENT_URL`): on a workstation without one,
`eforecast_source_configured{source="broker"}` is `0` and this alert stays silent.

<a id="the-cluster-api-cannot-be-reached"></a>
**The cluster API cannot be reached** (`EforecastClusterApiUnavailable`). Running Jobs carry on, but
none can be launched, observed or cancelled; an attempt already handed off stays recoverable and
is never launched twice. Only where a cluster is configured, as for the broker.

<a id="a-monitoring-target-stopped-answering"></a>
**A monitoring target stopped answering** (`EforecastTelemetryMissing`). The panels it feeds are
blank, which is not calm. Prometheus's page *Status → Target health* names the target and why:

* `eforecast-app`, *context deadline exceeded*: a firewall on the host drops what the monitoring
  containers send to the host itself. Let them in on ports 9108 and 9109,
  [installation §2.7](installation.md#27-optional-the-monitoring-stack).
* `eforecast-app`, *connection refused*: the metrics service is not running,
  `sudo systemctl status eforecast-metrics`.
* `dcgm`: the GPU exporter's container; `docker ps` lists `eforecast-dcgm-exporter-1`.
* `cadvisor`: the container exporter; `docker ps` lists `eforecast-cadvisor-1`. While it is
  down, the container panels are blank and the container alerts below are silent.

The orchestrator (port 9108) answers only while a training runs, so its target is DOWN between
trainings. That is normal, and it is not part of this alert.

<a id="the-workspace-is-nearly-full"></a>
**The workspace is nearly full** (`EforecastStoragePressure`). Under 10 % free. Stages and trials
start failing on writes; they leave no partial result behind, so nothing is damaged, but nothing
finishes. Free space, or move retained results off the workspace.

<a id="the-workspace-cannot-be-stated"></a>
**The workspace cannot be stated** (`EforecastStorageUnreadable`). Usually it is not mounted.
Nothing can run, and the free-space panel is blank rather than zero. Check the mount and
`EFORECAST_WORKSPACE`.

<a id="a-pod-stays-pending"></a>
**A Pod stays pending** (`EforecastPodsPendingTooLong`). It holds nothing and has started nothing,
so it is safe, and not progressing. The executor's status says which: no node can take it, the
image cannot be pulled, or a volume cannot be mounted.

<a id="a-run-has-lost-worker-contact"></a>
**A run has lost worker contact** (`EforecastLostWorkerContact`). The attempt may be alive and
quiet, or gone; either way it is not launched twice. `bin/eforecast doctor` names the attempt and
whether its process is still alive.

<a id="cleanup-is-owed"></a>
**Cleanup is owed** (`EforecastCleanupOwed`). A run that ended or was stopped may still have
processes or containers of its own alive, holding a card, memory or disk that the next run
expects to find free. `bin/eforecast doctor` lists owed cleanup and the run that owes it.

<a id="a-container-is-not-running"></a>
**A container is not running** (`EforecastServiceContainerDown`): PostgreSQL, Grafana, the Optuna
dashboard or the GPU exporter. `docker ps -a` says how it ended, `docker logs <name>` why, and
`docker compose -f docker/compose.yaml --env-file docker/.env up -d <service>` starts it again.
Prometheus cannot report itself: when it is down, no alert fires at all.

<a id="a-container-keeps-restarting"></a>
**A container keeps restarting** (`EforecastContainerRestarting`): three starts or more in half an
hour. Docker starts it again each time it ends, so it looks up most of the time while what it
serves keeps failing. `docker logs <name>` shows why it ends.

<a id="a-container-ran-out-of-memory"></a>
**A container ran out of memory** (`EforecastContainerOutOfMemory`). The kernel ended a process in
it, and what that process was doing is lost: a worker's task fails, a service restarts. The
dashboard *eforecast containers* shows its memory before; give it more, or run less beside it.

---

## 8. After an upgrade

A library upgrade under the checkout can break the pipeline without anything in the repository
changing, and the symptom is a wall of failing tests rather than a clear message. When that
happens:

1. find out **what moved**: `pip list --outdated`, or compare against the release's lock;
2. find the **one call** that broke, not the 25 tests that report it. A removed keyword argument
   in a library function will fail every test that reaches it;
3. decide deliberately whether to follow the library or pin it. If the old call encoded a
   guarantee — "raise rather than round", "fail rather than guess" — then silently dropping the
   argument changes the numbers. Reproduce the guarantee explicitly, or pin the version, but do
   not just delete the argument to make the error go away.

The same rule applies to the images: after any change to a file an image carries, `bin/eforecast
images refresh` — otherwise routing is refused, and the reason is that a stale image returns a
wrong number rather than an error.

### One configuration for a model, from 17 November 2026

A case builds one model, the model of one configuration, and the pipeline now holds itself to
that ([how it works §6](how_it_works.md#one-model-one-configuration)). What an existing case
meets:

* **The frozen copies under `runs/inputs/` go.** There was one for every run ever started. The
  next run of a case removes the copies of its earlier runs — never one a live run still uses —
  and from then on a case holds only the copy of the run in progress or of the last one.
* **The model starts keeping its configuration at its next stage.** A model built earlier has
  none on record (`<model folder>/.stage-artifacts/configuration/`) until a stage of it finishes
  or is reused. Until then a message about a changed configuration lists what is built again
  without the `changed since it was built` lines. Nothing has to be rebuilt for this.
* **Trained work stays as it is.** A deep-model trial written earlier records the case's
  fingerprint in its folder, which a sibling's change moves; its study, which always recorded
  the family's, is believed instead. A tree or foundation trial was bound to the configuration
  of the whole case; it is still recognised by the content of the files it was fitted on. A
  plan that used to read `3/3 trials done, reuse 0, 3 in another revision` now reads `reuse 3`.
* **A setting of the combination no longer sends a family back.** `combine_methods` and the
  combiners' settings, `keep_best_models`, `interval_coverage` and `cluster_coverage_policy` were
  part of every family's fingerprint: changing one made all trained work look like another
  configuration's. They are the case's now, not a family's. Trials trained before carry the old
  fingerprint, which is recognised through the configurations the case was run under: the next
  run reads the frozen copies of the earlier runs before it removes them, and keeps what they
  say in `<model folder>/.stage-artifacts/identities.json`. If a run still says that every
  family is trained from the beginning after you changed only such a setting, the case had no
  copy of a run under the earlier value left: put the setting back, run any step once (it keeps
  the name), and change it again. Nothing is removed before you accept it.
* **Continue no longer trains beside a study of another configuration.** Where a family's
  parameters or search space were changed after it was trained, the next training run names
  that family, removes its trials and trains it from the beginning. It used to keep them: the
  deep families trained nothing new, the tree and foundation families added trials to the old
  study, and the combination step read both. If you relied on the old trials staying, use the
  **new revision** action, which keeps them beside the new ones.
* **A run from the command line after a configuration change replaces what was built**, and
  says what before its first stage: datasets built again, a changed family's trials removed and
  the family trained from the beginning. It asks nothing ([cli.md §1](cli.md)); use `--dry-run`
  first when in doubt. A run started from the browser asks, with the case's name to type.
* **A case that reads weather is no longer told in the browser that its datasets changed.** The
  service made its plan one step before a run does: before the area of each weather provider was
  worked out, which is part of what every dataset stage is built from. So for such a case the
  Runs tab showed every dataset stage as `configuration changed`, training and combine as
  `blocked`, and a run started there ended with `launch preparation failed: BlockedStage` —
  while the same step typed in a terminal ran. The plan of a preview, of a prepared run and of
  its launch are now made on the configuration as the run has it. Restart the web service to
  get this.
* **A run on the cluster venue no longer ends at its first stage** with
  `KeyError: 'getpwuid(): uid not found'`: a Pod runs under a uid its image has no name for,
  and the record of who holds a case asked for that name
  ([deploy/kubernetes](../deploy/kubernetes/README.md)).
* **Refresh the images** (`bin/eforecast images refresh`): a trial that runs from the worker
  image records its family's fingerprint only once the image carries this version. Until then
  such trials are read the earlier way, through their study, and nothing is lost.

### Networks trained with the weather in groups, before 15 November 2026

A family whose `merge` is `by_variable`, `by_area`, `by_area_variable` or `by_nwp_provider`
reads the weather as several branches, one per group. Until 15 November 2026 a network built
that way joined those branches and stopped there: any other branch of its graph — the calendar
in every shipped graph — was computed and never read. Such a network forecasts from the weather
alone. Nothing failed and nothing was logged; the trial only scored what a network without a
calendar scores.

* **Nothing breaks.** A network saved before the correction is built the way it was trained,
  and its forecasts are exactly what they were. Finished trials stay finished.
* **What is trained from now on is the whole graph.** In a case that is trained further, a study
  then holds trials of both kinds; they are compared by their scores as always.
* **To give a trained case the correction**, train the affected families again. A *new
  revision* keeps the present trials and searches beside them; a rebuild
  (`run_training.py --refit`) deletes them and searches again. It is worth it where the
  family's `merge` is, or may be chosen as, one of the grouped values. A trial's
  `parameters.pickle` holds the `merge` it was trained with.

How the branches are joined is in [deep_models.md](deep_models.md#grouped-branches).

---

### Recurrent layers of a case without observations, from 24 November 2026

`lstm` and `lstm_att` now take a case's past data for their past when the case has no
observations ([deep_models.md](deep_models.md#32-on-sequences--the-lstm-branch)). What an existing
case meets:

* **A case with observations** (a measured past of the target among its inputs): nothing. Its
  networks are built as before, saved ones and new ones alike.
* **A case without observations whose `lstm` or `lstm_att` networks were trained before**: the
  saved networks are built as they were trained and forecast what they forecast. A trial trained
  from now on reads the past data too, so a study continued across that date holds trials built in
  two ways. They are comparable as models and not as architectures; to have them all alike,
  train that family again.
* **A case whose sequences are observations only** (every lag in `global_past_lags`, an empty
  `global_future_lags`, no calendar): it could not build any sequence layer before
  (`KeyError: 'past_calendar'`); it can now, with `use_nwp` off.

## Where to go next

* what is actually happening → [how_it_works.md](how_it_works.md)
* the commands → [cli.md](cli.md)
* running the installation → [administrator.md](administrator.md)
