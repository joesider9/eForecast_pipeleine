# How it works, step by step

This is the guide to read if you want to know what actually happens when you start a run:
what runs, in what order, what gets written where, what keeps two runs from writing one case,
and how the training is driven.

It is written in plain words. Every term that is special to this pipeline is explained the
first time it appears, and nothing below needs you to read the code first.

**Contents**

1. [Nine words you need](#1-nine-words-you-need)
2. [Where things live on disk](#2-where-things-live-on-disk)
3. [The databases, and what is in them](#3-the-databases-and-what-is-in-them)
4. [What happens when a run starts](#4-what-happens-when-a-run-starts)
5. [Step one: freezing the configuration](#5-step-one-freezing-the-configuration)
6. [Step two: the plan](#6-step-two-the-plan) — and [one model, one configuration](#one-model-one-configuration)
7. [Step three: who may run, and with what](#7-step-three-who-may-run-and-with-what)
8. [Step four: doing the work safely](#8-step-four-doing-the-work-safely)
9. [The dataset stages, one by one](#9-the-dataset-stages-one-by-one)
10. [How training is orchestrated](#10-how-training-is-orchestrated)
11. [What makes an old trial reusable](#11-what-makes-an-old-trial-reusable)
12. [Combining, calibrating and evaluating](#12-combining-calibrating-and-evaluating)
13. [From a trained model to a published forecast](#13-from-a-trained-model-to-a-published-forecast)
14. [The daily forecast](#14-the-daily-forecast)
15. [What happens if you stop it in the middle](#15-what-happens-if-you-stop-it-in-the-middle)
16. [Everything that can refuse you](#16-everything-that-can-refuse-you)

---

## 1. Nine words you need

| word | what it means here |
| --- | --- |
| **case** | one plant and one forecasting problem: the park, the target file, the horizon, the weather variables, the model families to try. It is a YAML document, split into part files, that lives in the workspace. |
| **static_data** | the case after it has been read and expanded: every path filled in, every default applied. The case is what you edit; `static_data` is what the code actually runs on. One is turned into the other by the loader at the start of every run. |
| **stage** | one step of work with a name, like `extract`, `nwp`, `target`, `training`, `evaluate`. A stage knows which other stages must finish before it can start. |
| **artifact** | a file a stage produces and a later stage reads: a dataset pickle, a fitted scaler, a trial's weights, a results table. |
| **receipt** | a small JSON file a stage writes when it finishes, saying *what it produced and from which inputs*. The receipt — not the presence of the file — is what proves a stage is done. |
| **fingerprint** | a short hash computed from the settings that affect a result. If two fingerprints match, the two results were produced from the same inputs and can be compared or reused. If they differ, they cannot. The code calls these *identities*. |
| **worker** | a process of its own that a run starts for one piece of work: a CPU study, the clusterer, prediction. Its memory goes back to the machine when it exits, and it stops when the run that started it does. |
| **run record** | the durable entry for one run: its id, its state, its case, what it was asked to do, what happened. It survives a restart of everything. |
| **trial** | one attempt at one model with one set of hyper-parameters, chosen by Optuna. A *study* is the collection of trials for one (cluster, method) pair. |

Two sentences that explain most of the design:

> **Nothing is trusted because the file exists.** It is trusted because a receipt says who
> wrote it and from what.
>
> **Nothing writes into the final place directly.** It writes somewhere private, and the move
> into the final place is a single atomic rename.

Those two rules are why an interrupted run never leaves a half-written dataset that a later
run mistakes for a good one.

---

## 2. Where things live on disk

Three separate trees, and it matters that they are separate.

**The workspace** — the cases, the accounts, the run state. One directory, named by
`EFORECAST_WORKSPACE`:

```
<workspace>/
  users.json                     accounts
  .secret                        the key that signs session cookies
  users/<name>/cases/<case>/     one directory per case: main.yaml + part files
  tenants/<id>/cases/<case>/     cases shared with a group
  .run-state/
    health.json                  the cached health reading the UI and new runs read
    scratch/                     temporary working areas
  users/<name>/cases/<case>/runs/inputs/<uuid>/<case>/
                                 the frozen copy of the configuration of the run in progress, or of
                                 the last one: the next run of the case removes it (see §5)
```

**The model tree** — the data and the models. This is the big one, named by the case's own
`folders` section, and it usually lives on a separate disk:

```
<path_group>/<case>/<horizon_type>/model_ver<N>/
  DATA/
    dataset_nwps.pickle          the weather arrays
    dataset_row_data.pickle      the flat feature table
    dataset_lstm_data.pickle     the sequence inputs
    dataset_target_data.csv      the target series
    dataset_images.pickle        satellite frames, if the case uses them
  scaled_inputs/                 the weather scaled once, mapped by every reader (§9)
  global/<method>/trial<N>/      one trained candidate: weights, parameters, its own result
  <cluster>/<method>/trial<N>/   the same, per cluster
  combine_models/                the fitted combiners
  Results/                       selection.csv, evaluation.csv, summary.csv, …
  exe_files/memory_estimates.json  how much GPU memory each kind of trial needed
  .stage-artifacts/
    holder.json                  who holds the case, while a stage is writing it
    writer.json                  who wrote here last, and a counter that goes up with every writer
    <stage>.running              a stage that is working, or one that was interrupted
    receipts/<stage>.json        one receipt per finished stage
    configuration/               the configuration this model was built from: one copy (§5)
    identities.json              what earlier fingerprints called each family's inputs (§6)
  predictions/
    .inference-generations/<launch id>/forecast.csv
                                 the forecast of one cycle, kept as it was accepted
    latest.csv                   a copy of the forecast the published pointer names
    status.json                  what the last forecast cycle did
```

**The state directory** — the databases. Default `~/.local/state/eforecast/`:

```
runs.sqlite3                     the run store (see §3)
executors/<id>/                  one directory per supervised child process
executors/compute/<launch id>/   one worker's own folder: its input, its log, its result (§7)
```

Why separate? Because the model tree can be rebuilt and the workspace can be restored, but if
they shared a disk, filling the disk with datasets would also take out the record of what is
running. The administrator's backup destination must be on a third, physically different disk,
and the doctor refuses one that is not.

---

## 3. The databases, and what is in them

There is **one** database. By default it is a single SQLite file; on a bigger installation the
same tables live in PostgreSQL instead (`EFORECAST_RUN_STORE_DSN`). Nothing in the pipeline
cares which — the code asks the store, and the store knows.

Optuna keeps its own separate database for the trials themselves
(`EFORECAST_OPTUNA_STORAGE`, usually PostgreSQL) because many trial processes write to it at
once.

The tables, grouped by what they answer. You do not need to memorise these; this is the map for
when you want to look something up.

**"What runs exist, and what happened to them?"**

| table | holds |
| --- | --- |
| `eforecast_runs` | one row per run: id, case, state, outcome, cleanup state, what was requested |
| `eforecast_run_events` | the timeline of one run: every state change, with when and why |
| `eforecast_run_children` | the processes a run owns, so a dead run's children can be found |
| `eforecast_run_case_claims` | which run currently holds a case, so two runs cannot train the same case |
| `eforecast_run_identities` | the fingerprints of the configuration a run was accepted under |
| `eforecast_run_submissions` | the request as submitted, before it became a run |

A run's **state** moves through a fixed set of steps and never goes backwards:

```
pending ──► preparing ──► queued ──► running ──► terminal
   └──────────────┴────────────┴─────────┴──► unknown ──► running / terminal
```

`unknown` is not a failure. It means the record exists and nobody can currently prove what the
process is doing — for example after the machine rebooted. It is a state you can leave, in
either direction, once there is evidence.

Separately from the state, every run has a **cleanup** state: `not_required`, `pending`,
`running`, `unknown`, `complete`. A run is not finished when it stops; it is finished when its
children are confirmed stopped. This is why "Cleanup still running" is not an error, and why
starting the same work again during cleanup can be refused.

**"Where should this work run?"**

| table | holds |
| --- | --- |
| `eforecast_dispatch_routes` | the venue decided for each scope: `local` or `broker` |
| `eforecast_dispatch_route_history` | every decision ever made, with who, when and why |
| `eforecast_dispatch_outbox` | messages waiting to be handed to a broker |
| `eforecast_dispatch_undeliverable` | messages that could not be delivered, and the classification why |
| `eforecast_kubernetes_runs` | the Jobs a managed run owns |

**"What may be served to a user?"**

| table | holds |
| --- | --- |
| `eforecast_model_approvals` | the pointer: which captured bundle a case serves |
| `eforecast_model_approval_history` | every approval and rollback |
| `eforecast_model_holds` | a model pinned for the duration of one forecast cycle, so it cannot change under it |
| `eforecast_model_pins` | longer-lived pins |
| `eforecast_serving_bindings` | which predictor inside a bundle is the one served |
| `eforecast_verified_completions` | proof that a particular piece of work really finished, used as a gate before publishing |
| `eforecast_verified_completion_marks` | the marks that make the above checkable |
| `eforecast_forecast_issues` | one row per forecast attempt, published or rejected: the issue time, its state, the payload digest. Written once, never updated |
| `eforecast_forecast_valid` | the publication pointer: one row per case, naming the attempt that is the published forecast |

No table says how much of the machine a run may use: nothing is reserved (§7). A store made by
an earlier version may still hold tables named `eforecast_dispatch…`, `eforecast_cache_…`,
`eforecast_library_…` or `eforecast_profiles…`. Nothing reads or writes them any more.

The honest summary: the database holds **decisions, claims and pointers**. The actual numbers —
datasets, weights, forecasts — are files. The database never holds a copy of a result; it holds
the statement that a result exists, where, and whether it may be used.

---

## 4. What happens when a run starts

Whether you click **Run** in the browser or type `python run_datasets.py --config pv_park`, the
same sequence happens. The browser path adds a durable record and a supervisor in front of it;
the command line is the same engine with you as the supervisor.

```
   you ask for work
        │
        ▼
 1. freeze the configuration        → runs/inputs/<uuid>/   (§5)
        │
        ▼
 2. build the plan                  → which stages, in what order, reuse or rebuild   (§6)
        │
        ▼
   for each stage, in dependency order:
        │
        ├─ 3. take the case         → one writer per case, a lock the kernel holds   (§8)
        │        held by another run → refused at once, with who holds it
        │
        ├─ 4. note the writer       → writer.json, fence + 1; <stage>.running   (§8)
        │
        ├─ 5. do the work           → in the case's own folders, each file under a
        │                             temporary name until it is whole   (§8)
        │
        ├─ 6. write the receipt     → what was made, and from which inputs   (§8)
        │
        └─ 7. let the case go
        │
        ▼
   record the outcome, run cleanup, release the case claim
```

Every arrow in that diagram is a place a crash can land, and each one is designed so that the
next run can tell what happened. That is the whole reason it has this many parts.

Nothing in this sequence asks whether there is room to run. A run takes its case, and the
machine's processors, memory and disk are the operating system's to give (§7).

---

## 5. Step one: freezing the configuration

The first thing a run does is take a copy of its own configuration and never look at the
original again.

It creates a new directory — `runs/inputs/<random hex>/<case>/` — and writes two things into it:

* a copy of the raw configuration files, exactly as they were;
* `effective_config.json`: the fully resolved configuration, with every path filled in and
  every default applied, plus the options you asked for on the command line.

Then it writes a marker file, `.eforecast-inputs.json`, **last**. The marker is what makes the
snapshot valid. If the process dies halfway, the payload is there but the marker is not, and
the next reader correctly treats the snapshot as absent rather than as a half-configuration.

When the snapshot is complete, the run loads *that* and runs from it.

**Why.** A training campaign can last a day. If you edit the case while it runs — even a
harmless-looking change — the second half of the run would be using different settings from the
first half, and nothing in the result would say so. Freezing the inputs means the run has one
configuration from beginning to end, and the snapshot is the evidence of which one.

The random name in the path exists only so that two runs starting at the same second cannot
overwrite each other's snapshot. Nothing else in the pipeline looks a run up by it; the run
knows the path because it was just told it.

**One copy is kept, and it is kept with the model.** A snapshot is a run's working copy: once
the run has ended it is nobody's input. These directories used to stay, one for every run that
was ever started, and none of them said which one the model on disk was made from. Now:

* when a stage finishes — or is found finished and reused — the configuration the run executes
  from becomes the model's own, in `<model folder>/.stage-artifacts/configuration/`: the case's
  files as they were (`case/`), the configuration as the run resolved it
  (`effective_config.json`), and `record.json`, which says when and for which stages. One copy;
  the configuration of a later run takes its place. It goes with the model when the folder is
  copied, moved or backed up;
* it takes that place only when the model is **wholly** its own: every dataset stage was built
  from it and no family that is switched on still holds trials of another configuration. A
  change is worked off in steps — the datasets are built again, then each family is trained
  again — and all the way through, the copy that is kept is the last configuration the whole
  model was built from. That is why the run that trains a family again, after the datasets are
  already done, can still say which variable was changed;
* when a run starts, the snapshots of the runs of that case that have ended are removed. A
  process holds a lock on the snapshot it runs from for as long as it lives, so a run that is
  still going is never touched; the service removes the snapshots of its own runs when it knows
  they have ended and been collected;
* before they go they are read once more, for the one thing only they can still say: what an
  earlier fingerprint called each family's inputs (§6). The model keeps that, in
  `.stage-artifacts/identities.json`.

So a case holds the snapshot of the run in progress (or of the last one, until the next starts),
and its model holds the configuration it was built from. That copy decides nothing: receipts
and trial records decide what may be reused (§6, §11). It is read for the sentence they cannot
give — *which setting* changed since the model was built — and, before another takes its
place, for the earlier names of the families' inputs.

---

## 6. Step two: the plan

Next the run decides what actually needs doing. It does not simply run every stage.

First it builds the **graph**: the stages for this kind of run, each with the stages it depends
on. For a dataset run that is:

```
extract ──► nwp ──┐
                  ├──► lstm
target ───────────┤
row ──────────────┘
images  (skipped unless the case uses images)
```

and then the four model-input stages: `split`, `preprocess`, `clustering`, `features`.

Then, for each stage, it asks one question: **is what is already there usable?** The answer is
one of six words:

| decision | meaning |
| --- | --- |
| **reuse** | there is a receipt, its inputs match, nothing to do |
| **resume** | partly done, and the part that is done is still valid; carry on from there |
| **verify** | it looks done, but something needs checking before it is trusted |
| **rebuild** | it must be done again |
| **blocked** | it cannot be done, and this says what is missing |
| **skip** | the configuration disables it (for example `images` on a case with no satellite data) |

It decides by comparing **fingerprints**. The receipt of a finished stage records a fingerprint
of the settings that stage depends on. If the current settings hash to the same value, the
stage's output was built from the same inputs and is reused. If not, it is rebuilt.

One rule makes the plan safe: **if a stage will be rebuilt, everything downstream of it is
rebuilt too.** You saw it in this plan:

```
extract      rebuild  this stage has produced nothing yet
nwp          rebuild  extract will be rebuilt, so this stage's inputs change
images       skip     images are disabled
target       rebuild  this stage has produced nothing yet
row          rebuild  this stage has produced nothing yet
lstm         rebuild  nwp will be rebuilt, so this stage's inputs change
total: 0 reuse, 0 resume, 0 verify, 5 rebuild, 0 blocked, 1 skip
nothing was deleted: this is a plan.
```

You can always see the plan without doing anything:

```bash
python run_datasets.py --config <case> --dry-run
python plan_revision.py --config <case>
```

The three **actions** you can ask for change the question, not the mechanism:
`continue` (the default — reuse what is valid), `new`, and `rebuild` (treat everything as
needing to be done again).

### One model, one configuration

A case builds one model, and that model is the model of one configuration. Editing a case after
something was built from it therefore costs something, and how much depends on what was edited:

| you changed | what it means |
| --- | --- |
| a variable, its lags or its parameters, the target, a provider, the horizon, the resolution | the **datasets** were built from other settings: they are built again, and every model with them |
| the split, a scaling method, the clustering | the stage that reads it is built again, and what depends on it |
| a parameter or the search space of one family | **that family** is trained from the beginning; the others are kept (§11) |
| something every family learns from: the network definitions (`experiments`), the folds, the slice a trial is scored on (`selection_fraction`, `calibration_fraction`), `max_performance` | **every family** is trained from the beginning |
| how the trained models are combined afterwards: `combine_methods` and the combiners' own settings, `keep_best_models`, `interval_coverage`, `cluster_coverage_policy` | the **combination** is computed again from the trials that are there; nothing is trained |
| a budget (`n_trials`), a path, how many processes | nothing: the work that exists is still this configuration's |

The plan finds this by itself — a receipt records a fingerprint of the settings its stage was
built from, a study the fingerprint of its family — and it says *why* a stage is done again. One
reason is set apart from the others: **the configuration changed**. A stage that is missing,
was interrupted or whose file was damaged is simply done; a stage whose output is whole and was
built from settings the case no longer has is *named* first, because doing it again throws away
work that somebody may have wanted to keep — and putting the setting back keeps it.

Where that is confirmed depends on where the change is made.

**In the browser it is asked, twice**, each time with the list in front of you and the case's
name to type: when the change is **saved** (*Review & save* shows the same list without saving),
and when the **run** that replaces the work is started. The save replaces nothing — the next
run does — but it is the save that makes the next run do it.

**On the command line it is said, and done.** Whoever edited the case and typed the command is
told what that costs before the first stage starts, and the run goes on. There is no question
and no flag:

```
The configuration of pv_park is not the one this was built from.
  changed since it was built:
    data.variables[Cloud].lags: [0, 1, 2, 3, 4] -> [0, 1, 2, 3, 4, 5]
  built again, replacing what is there:
    nwp         1 file(s) -- its settings changed
    row         1 file(s) -- its settings changed
    ...
  this run replaces them now. Stop it within the next stage if that is not what you want;
  --dry-run shows the same list and changes nothing
```

`--dry-run` shows the list without doing anything. A run the service started on someone's
behalf never takes this shortcut: it carries the answer its submitter gave in the browser, and
one that arrives without it stops before it has touched a file.

What is neither asked about nor announced: an explicit `--recreate`, `--refit` or *rebuild*
(whoever asks for it has said so), a first build, an interrupted stage, and the combination
step, whose results are
derived from the datasets and the models in minutes and are computed again whenever those
change. `extract` keeps the daily weather files it finds and reads only the missing days, so
doing it again replaces its receipt and not the archive; a change of the weather *variables*
is another folder (§9) and is extracted from nothing.

Until the datasets are built again nothing is trained or combined on them (`training requires
row: row configuration changed`), and until a family is trained again the combination step does
not read what it left (`the trained models of TRANSFORMER are of another configuration`). A
model is never a mixture of two configurations under the name of one.

**What a family's fingerprint is made of.** Its own parameters without the budgets, everything
shared that a family learns from, and the content of the datasets, the partition and the
scalers — and *not* its siblings' parameters, which families are switched on, or the settings
that are only read after training (`lineage.NOT_LEARNED`: the combination's, and legacy job
counts nothing reads). Trying another combiner trains nothing again. Until 17 November 2026
those settings were part of every family's fingerprint, so the trials trained before carry a
name that moves with them. That name can only be worked out from a configuration that still has
the old values, so before a run removes the frozen copies of the runs before it, it reads them
and the model keeps what they called each family's inputs
(`<model folder>/.stage-artifacts/identities.json`). A trial is recognised by the new
fingerprint or by a name kept there for exactly these inputs; nothing else is read from it.

---

## 7. Step three: who may run, and with what

Two questions, and the pipeline answers only the first.

### Who may run: one run writes a case at a time

A run takes the **case's lock** before each stage: a file lock on the case's `.stage-artifacts`
folder (and on the shared weather archive for `extract`, the image folder for `images`). The
kernel holds it for as long as the process lives and lets it go when the process ends, however
it ends. A second run of the same case is refused after two seconds, with who holds it and the
one command that stops a run of your own:

```
Busy: Total_wind is being written by another run: stage nwp of Total_wind, started 14:02:11
by maria (pid 41873). One run writes a case at a time; wait for it to finish, or, if it is
yours, stop it: python -m eforecast.config.direct_stage stop-mine <the case's model folder>
```

`stop-mine` asks that process to stop the way Ctrl-C does, and only if it runs as you; somebody
else's run is never touched. `python -m eforecast.config.direct_stage status <model folder>`
only says who holds the case. A run started from the browser is also recorded as holding its
case in the run store, so a second submission for that case is turned away before it starts
anything.

Runs of *different* cases do not wait for each other. The installation is used by one person
at a time; nothing counts how many runs are going.

### With what: nothing is reserved

No run, stage or worker asks for processors, memory, GPU memory or disk before it starts, and
nothing is set aside for it. There is no ledger of reservations, no queue for room and no
deadline to wait out. A process is started and the operating system gives it what it gives any
process.

What the pipeline still decides is **how much it starts**, from the case's own settings and
from what it measures:

| Work | How much runs at once | Setting |
| --- | --- | --- |
| Dataset stages | reader processes inside the stage | the case's `n_jobs` (§9) |
| CPU studies | as many as the run has, up to one a CPU, sharing the CPUs the case uses | `n_cpus`, `cpu_workers`, `cpu_study_threads` (§10) |
| Deep trials | per card, by what each kind of trial was measured to take of it | measured; `--n-jobs` fixes it (§10) |
| Workers (clusterer, prediction, …) | one per piece of work, with the threads of its operation | [compute workers](compute_workers.md) |

**When the machine does not have what the work needs**, the work fails the way any process does
and says which way:

- *Memory.* Linux ends the largest process. A worker that ended so reports `it was killed
  (SIGKILL); when nobody stopped it, that is the kernel ending the largest process because the
  machine ran out of memory`, with the path of its log. CPU training starts a study that was
  killed once more, alone, after the others (§10). Otherwise start less at once: fewer studies
  (`cpu_workers`), fewer readers (`n_jobs`), a smaller batch.
- *A card's memory.* The driver refuses the allocation and the trial reports an out-of-memory
  error; the training scheduler retries it alone on a card (§10).
- *Disk.* The write fails with the system's "No space left on device". A file being written has
  a `.tmp` name and is nobody's dataset; the stage is not recorded as done and is rebuilt by the
  next run.

**A worker stops with its run.** A worker's supervisor reads a pipe whose other end only the
process that started it holds. When that process ends, the kernel closes the pipe and the
supervisor stops the worker and everything it started. So a run that was stopped or killed
leaves no work going in its case, and nothing to give back or reclaim before the next run.

---

## 8. Step four: doing the work safely

A stage does four things in order. [Stages write in place](#stages-write-in-place), at the end
of this page, has every rule; this is the outline.

**1. Take the case.** A file lock on `.stage-artifacts` — and on the shared weather archive for
`extract`, the image folder for `images` — which the kernel holds for as long as the process
lives and lets go when it dies.
`holder.json` beside it says who holds it, so that a second run can be told. The stage then adds
one to the `fence` counter in `writer.json`; the number only increases, so an older writer can
always be recognised as older.

**2. Leave a marker.** `<stage>.running` is written before the work and removed after the
receipt. A marker found by the next run means the stage did not finish, and its outputs are
built again rather than trusted.

**3. Work in place.** Each file is written in the folder it belongs to, as `<name>.tmp`, and
renamed with `os.replace` when it is whole: a reader sees the old file or the new one, never
half of one. A rebuild first sets the old outputs aside as `<name>.previous`, and they come back
if it fails.

**4. Write the receipt.** `.stage-artifacts/receipts/<stage>.json`, written **after** the files
are in place. The receipt records what was produced and the fingerprint of the inputs it was
produced from. Until the receipt exists, the stage is not done — which is why a crash between
step 3 and step 4 costs you one re-run of a stage and never produces a wrong result.

Then the case is let go, and the next stage starts.

---

## 9. The dataset stages, one by one

| stage | what it does | what it leaves |
| --- | --- | --- |
| `extract` | reads the raw weather files (GRIB) for every day in the period — from the local folder, or downloaded from the case's shared Google Drive folder — and turns them into per-run arrays | the group's extracted weather, one file a day (`<group>/nwp/…`) |
| `nwp` | assembles the extracted weather into the case's own variables and grid | `DATA/dataset_nwps.pickle` |
| `images` | the same for satellite frames — skipped unless the case uses them | `DATA/dataset_images.pickle` |
| `target` | reads the measured production series, aligns it to the case's resolution and horizon | `DATA/dataset_target_data.csv` |
| `row` | builds the flat feature table: lags, calendar features, the weather columns | `DATA/dataset_row_data.pickle` |
| `lstm` | builds the sequence-shaped inputs the recurrent and transformer models need — skipped for a `long_term` case | `DATA/dataset_lstm_data.pickle` |
| `split` | decides, once, which timestamps are training, selection, calibration and holdout | the frozen split |
| `preprocess` | fits the scalers on the training part only, and then scales the weather with them once | the fitted scalers, `scaled_inputs/` |
| `clustering` | assigns each timestamp to a cluster, if the case trains per cluster | the cluster assignment |
| `features` | runs the feature selection the case asks for | the selected feature sets |

Two of these deserve a warning.

**`extract` is the one that is expensive and fragile.** It reads real weather files, and some
weather fields are *accumulated*: the value at hour 6 is the total since the run started, so the
pipeline subtracts the previous step to get the hourly value. If one step is missing, the next
step's subtraction is wrong — and for a solar case, that roughly doubles the solar radiation for
that hour. A gap in an accumulated field does not produce a gap in the output, it produces a
plausible wrong number, which is the harder of the two to notice.

So a zero-byte or missing weather file is handled in three steps rather than skipped or refused.
The run first fetches it again from the archive it came from, because a file that holds no bytes
is usually a download that stopped partway. Whatever is still empty is named in full, once. And
the extractor then drops that step **and the first readable step after it**, rather than
difference across the hole: one missing hour costs two, and no wrong number is published. The
dropped hours are usually covered by another analysis cycle, because `as_of` selection tries the
cycles of this day and the day before, newest first.

**The raw files need not be on the machine.** A provider's entry in the case can name the shared
Google Drive folder its archive is kept in (`path_nwp_gdrive`; the *Google Drive* source of the
Project tab). `extract` then reads a day from the local folder when it is there, and otherwise
downloads it, reads it, and removes it again: what stays is the daily file, a few megabytes in
place of the provider's forty-five. Three rules keep that safe.

* *Only what was downloaded is removed.* A day found in the local folder is read from it and
  left alone. Downloads pass through a scratch folder beside the day folders
  (`.nwp-gdrive-<machine>-<process>-…`), never through a day folder of their own name, so what
  a killed run left can never be read as a day of the archive; the next run removes it.
* *A day arrives whole or not at all.* With the raw files gone, a day read with a file missing
  could never be repaired, so one file that cannot be had fails its day.
* *A day that did not arrive fails the stage.* It is not a day without weather — the folder has
  it — and recording it as one would mean never asking again. The days already converted are
  kept, and the next run asks only for the rest. A day the shared folder does not hold at all is
  a day without weather, as in a local archive.

Only the steps the extractor reads are downloaded, a few days at a time whatever `n_jobs` says.
The folder is listed once a run, and not at all when every day is already extracted, so a
finished stage works with no network. `EFORECAST_NWP_REFILL=off` keeps a run from downloading.

**The extracted weather belongs to the group, not to the case.** Every case of a group reads the
same daily files, so a second case over the same period extracts nothing. A file holds the
variables it was extracted with, and each set of variables has a folder of its own, named after
them: `nwp/Cloud+Flux+High_Cloud+Low_Cloud+Medium_Cloud`. Two parks that ask for the same
variables share one; a wind case beside them has another. They are kept apart rather than merged
into one file because a file's steps depend on its variables -- a step is dropped whole only
when an accumulated field is among them -- so one case joining a group would change what another
reads. Files that an earlier version wrote directly into `nwp` stay there and serve the cases
that need exactly their variables; `nwp/variables.json` says which those are. A dry run prints
the folder a case uses as `path_group_nwp`.

An *interval* field that is not differenced — a gust maximum — at the step after a gap is left
empty rather than wrong, and the sample that needs it then exists with no numbers in it. The
`nwp` stage leaves such samples out of that variable and says how many (`... samples hold
missing values ... and are left out`). The grid form of the weather goes to a convolutional
network as it is, and one such sample makes a whole batch NaN.

**`extract`, `nwp` and `features` use the CPUs of the machine; the others use one.** The stage
log says so (`stage nwp: 16 CPUs (15 readers beside this process)`). The number is the case's
`n_jobs` within the CPUs the machine has. Memory is not looked at: a reader holds about 0.8 GB
and thirty-two times the file it is decoding, and on a machine that cannot carry as many readers
as the case asks for [the kernel ends one](troubleshooting.md#the-machine-ran-out-of-memory).
The stage then goes on with half as many and does the work that was in flight again; a lower
`n_jobs` spares it the detour. `--serial` keeps everything in the process you started, on one
CPU, which is what a debugger wants.

The `features` stage spends its CPUs on whichever there is more of. A case has a selector for
every cluster, method and group of inputs, and when a dataset has several to fit they are fitted
side by side, each estimator with its part of the CPUs — one thread each once there are as many
selectors as CPUs (`row_all minmax minmax by_variable dense: 78 selectors, 16 at a time with 1
thread(s) each`). An estimator given sixteen threads for a few thousand rows spends them waiting
for each other. How many are fitted at once is the stage's CPUs and nothing else: memory is not
looked at. While it is fitted a selector holds about nine times its inputs as 8-byte numbers,
and a random-forest ranking (`FeatureImportance`) 20 KB more for every training row, because it
keeps its trees; where the machine does not carry as many as the case's `n_jobs`, that is the
setting to lower. A selector that is alone is fitted by itself with every CPU, as before. The
selectors come out the same and are saved in the same order either way.

A `features` stage that is stopped does not continue from where it was, although each selector
file is written whole and the fitting skips the selectors it finds. [An interrupted stage is
done again, not trusted](#stages-write-in-place): what the stopped run wrote is removed, what a
rebuild had set aside is put back, and the next run fits every selector again.

**The weather is scaled once, not by every reader.** A trial, a selector and a prediction all
want the same thing of `dataset_nwps.pickle`: its rows in single precision, scaled, padded and
with the variables side by side. Each used to make that for itself — load the dataset whole, take
its rows, copy, scale, join — and held the dataset about six times over while it did. So it is
made once, when the scalers are there:

```
<model folder>/scaled_inputs/
  manifest.json            what these files are made from, and what they hold
  dates.pickle             the row labels
  nwp_<method>_0.npy       rows x lags x lat x lon x variables, one per scaling method
  tables/                  compressed tables, each kept the first time it is computed
```

The log says `scaled inputs: 4.4 GB written for minmax, standard in 5.8s`. A reader maps the
array and says which rows it wants — its side of the evaluation date, the hours the target has,
the hours of its cluster — and each of those steps narrows a list of row numbers. The rows are
read once, when a network takes its training, validation and test sets, by any number of
processes from the one copy the system keeps in its file cache. A compressed input (`dense`,
`semi_full`, …) is computed from the rows it is asked for and kept in `tables/` under exactly
those rows, so the next reader of the same rows opens a small file.

On a five-month copy of a real case (a weather dataset of 2.9 GB), every one of 51 inputs came
out the same to the last bit both ways, and so did the 741 selectors of the `features` stage
and the predictions of trained networks. A trial loaded what it trains on in 0.5 to 1 s instead
of 6 s and held 1.7 to 2.2 GB at its peak instead of 8.5 GB; a compressed input took 0.02 s
the second time instead of 5 s. The clustering stage took 136 s instead of 266 s and the
`features` stage 329 s instead of 381 s.

These files are a copy and are trusted only as one. The manifest records the size and
modification time of the dataset and of `scalers.pickle`. A reader that finds either changed,
finds no manifest, or finds a file other than the one the manifest describes reads the dataset
as before and says so once (`scaled inputs of <case> are not used: …`). They are written by
`run_datasets.py` after `preprocess` — also when that stage is reused — and by
`run_training.py` and `run_combine.py` when they start, so a case whose datasets are older than
this gets them at its next run; a copy that already matches is left alone. They are not a
stage's product: no receipt names them, and deleting the folder costs the next run the time to
write it. They lie beside `DATA` and not in it, because `DATA` is what a model was made from —
a bundle captured for serving takes all of it — and this is a copy any run can make again.
They take as much disk as the weather dataset for each scaling method the case names, and are
not written when the disk does not have that much. `clean_model.py` parks them with the full
datasets. `EFORECAST_SCALED_INPUTS=0` turns them off, for writing and for reading.

What is not in them: values asked for unscaled, the inputs of an online forecast (its dataset
is another file), and anything a managed reader session is fed.

**A `long_term` case has no sequence dataset.** Its rows are indexed by *(prediction time, issue
time)*: every predicted hour appears once per issue day, up to three days back. A sequence is a
run of rows shifted along one clock, and that index has two, so the `lstm` stage is skipped for
such a case and nothing waits for it. A method that reads sequences — a foundation model
(Chronos2, TimesFM, TiRex) or a graph with an `lstm` branch — cannot run for it: Review & save
says so, and a training run that includes one is refused before it starts. Everything else places
a row by its predicted hour: the split, the scalers, the clustering and the feature selection.

**`target` is where the source's own clock meets the case's grid.** The measured series is put on
the case's resolution before anything reads it, and a slot with several readings in it becomes
their mean — quarter-hourly rows in an hourly case average into the hour, so the value means what
its stamp says and no reading is discarded. Picking one of them instead would file the measurement
taken at `HH:15` or `HH:30` under `HH:00`, against weather and calendar inputs aligned to `HH:00`.
A timestamp that repeats is a different thing: one instant described twice, where the later row is
a correction and wins. Slots with no row at all stay empty and are dropped later rather than
filled. Review & save says how many rows of a file fall between the slots and what the import does
with them; if you would rather keep the finer rows as rows of their own, the case's resolution is
what to change, not the file. The resolutions are `1min`, `5min`, `15min`, `H` and `D`.

**`row` reads the other series of the data file by their column.** A `timeseries` variable whose
source is `target` is the target itself at the variable's lags. One whose source is the name of
another column of the file is that column — a measured wind speed beside the production, say —
and a source ending in `.csv` is a column of another file.

A quantity that wraps around, such as a wind direction, is tagged **`cyclical`** (its
transformation). 359 degrees and 1 degree are neighbours; as numbers they are 358 apart, so a
scaler, a difference between two lags and a network would all read the passage through north as
the largest change there is. A tagged variable `DWIN` is therefore given to the run as two
inputs, `DWIN_sin` and `DWIN_cos`, each with the lags and settings of the variable, and those two
names are the ones the row data, the sequences and a clustering (`rbf_var_imp`) use; the angle
itself is not an input. The period is 360 unless the variable says otherwise
(`transformer_params: {cyclical: 24}` for an hour of the day). Only a `timeseries` variable can
carry the tag.

**`split` and `preprocess` are the ones you must not casually throw away.** The split is what
makes every later number comparable, and the scalers were fitted on the training part of that
split. Deleting them and rebuilding gives you a different split, and every model trained against
the old one becomes incomparable. This is the reason `clean_model.py` moves things to a `TEMP`
folder by default instead of deleting them.

---

## 10. How training is orchestrated

This is the most machinery in one place, so here is the shape before the detail.

```
run_training.py
   │
   ├─ builds the list of units:  one study per (cluster, method)
   │     global/MLP   global/LSTM   kmeans_0/MLP   kmeans_1/MLP   …
   │
   └─ the orchestrator loop, once per tick:
         ├─ reap:    collect finished trial processes, tell Optuna the result
         ├─ report:  forward each running trial's epoch score to the pruner
         └─ place:   start new trials where there is room on a GPU
```

### The units

A **study** is one Optuna study for one (cluster, method) pair. Optuna's job is to choose
hyper-parameters: it is asked for a trial, it proposes settings, the trial runs, and it is told
the score so its next proposal is better informed.

There is a second kind of unit: a **fixed unit**, which retrains a global method's winning
trials on one cluster (transfer learning). A fixed unit waits for the global study of its method
to finish, because it has nothing to copy until then. No study wants these trials, so no
budget counts them; the plan counts them on their own -- `transfers 3/15 trained, 12 to train`
for a cluster, `84 transfer trial(s) to train` for the stage -- and a training stage that has
some to train is not reused, however complete its studies are.

### One tick of the loop

**Reap.** Every trial runs as a separate child process (`trial_runner.py`, optionally inside the
worker Docker image). When one exits, the orchestrator reads its result, tells the study, and
decides whether to retry it.

**Report progress.** A running trial writes its score each epoch. The orchestrator forwards
those to Optuna's pruner, so a trial that is clearly going nowhere is stopped early instead of
running to its last epoch.

**Place.** This is the interesting part. The orchestrator packs trials onto the GPUs by what
it measures there; [GPU training](gpu_training_optuna.md#how-many-trials-share-a-card-and-how-big-each-one-is)
has the rules it uses by default. With `--n-jobs`, or where a card cannot be read, it packs **by
declared memory**:

* it keeps a table of how much GPU memory each (method, experiment shape) needed, computed as
  **1.5 × the median peak** reported by earlier trials of that shape. Before any trial of a
  shape has run, a conservative built-in table is used. The table is saved in
  `exe_files/memory_estimates.json`, so what the machine learned about itself survives a restart;
* it takes the study with the most remaining work, for each method in turn, and tries the GPU
  with the most free memory;
* **placing has two answers**: the trial starts, or it waits for a running trial to free memory.
  It is never refused for its size. A trial expected to need more than the largest card is
  given a card to itself and tried there, because what it is expected to need is an estimate
  from other trials of its shape; only running out of memory on that card, alone, fails it.
  (Until 24 November 2026 it was failed at once as impossible, on the estimate.)

One subtlety worth knowing because it affects fairness: the exact shape of a trial is only known
*after* Optuna has proposed it, so before asking a study the orchestrator checks whether the
**cheapest** shape of that method could start. Asking with the largest shape in mind let one
oversized experiment block every smaller one of the same method.

At most one proposed trial per GPU slot waits in the queue, so Optuna is never far ahead of what
is actually running.

### Memory, and what happens on an out-of-memory error

Each child is told its share of its card as `EFORECAST_MEM_FRACTION` and fails fast if it grows
past it — a trial cannot quietly take the card memory another trial was counted for.

If a trial still runs out of memory, the policy is three strikes:

1. retry with a raised estimate (and the raised estimate is remembered);
2. retry alone on the emptiest GPU — and with `--halve-batch`, at half the batch size, recorded
   on the trial so you can see it happened;
3. the third failure is final: the trial is told FAIL with reason `oom`.

Any other error fails the trial at once. A failed trial is a fact about that trial, not about the
study.

A study is asked for at most twice the trials wanted of it, so one whose every trial fails does
end. A trial the machine ended is not one of those attempts: one that was cancelled with its run,
one a dead run left behind, and one whose card the driver lost. A lost card is taken out of the
run at once and gets no further trial; with no card left the run ends as failed.

### Stopping, and how a run ends

If nothing starts for `EFORECAST_NO_PROGRESS_SECONDS` (default 900, `0` disables it), the run
stops and says why, rather than waiting all night on a machine that is busy with something else.

`SIGINT`/`SIGTERM` ask the run to stop politely: running trials are terminated, given
`CANCEL_GRACE_SECONDS` to exit, killed if they do not, and **every proposed trial is told FAIL** —
so no study is left with a trial marked RUNNING that no process will ever finish. (If the
orchestrator itself is killed outright, those RUNNING trials do survive in the study; the next
start fails them.) The ways to send that request, for a run started in the browser, in a
terminal or as a service, are in
[GPU training](gpu_training_optuna.md#7-stop-a-training-and-continue-it-later).

A run ends with an outcome, and the exit code follows it:

| outcome | exit | meaning |
| --- | --- | --- |
| `succeeded` | 0 | everything that was asked for completed |
| `succeeded_with_failures` | 0 | some trials failed; usable work was produced |
| `nothing_to_do` | 0 | there was nothing left to train |
| `failed` | 1 | there was work, and nothing completed |
| `cancelled` | 2 | somebody stopped it |

### The shallow CPU path

`run_training.py --cpu` trains the `methods_cpu` families instead, with a scope
(`--cpu-scope global|clusters|both`) saying whether to train the global models, the per-cluster
ones, or both. Each study, one method on one cluster, runs in a worker of its own, and the
studies run side by side: as many as the run has, up to one a CPU, sharing the CPUs the case
uses (`n_cpus`). The threads of a study that finishes go to the ones still running. A study
that fails is recorded and the others go on; one the kernel killed for memory is started once
more, alone, at the end. `--cpu-workers` and `--cpu-study-threads` set the numbers by hand.
[CPU scopes and studies side by side](cpu_training_scopes.md) has the details and a
measurement.

---

## 11. What makes an old trial reusable

When you start training again, each study asks: *are the trials already in here comparable with
what I am about to do?*

The answer is a fingerprint called the **comparability id**, computed from four things:

* the configuration — but with the search budget removed, because asking for 40 trials instead
  of 20 does not make the first 20 wrong;
* the dataset identities;
* the partition (the frozen split);
* the scalers.

Every trial records that fingerprint when it is created: its study does, and so does its
folder. On the next run, a trial whose fingerprint matches is kept and counts towards the
budget.

The configuration part is computed **per model family**. That is what makes incremental work
possible:

* change a parameter of one family, or what it searches over → **that family** is trained from
  the beginning, the others are reused;
* add or remove a family → the other families' trials stay valid; the combining step runs again
  because the set of candidates changed;
* change something shared that a family learns from — a variable, the split seed, the target,
  a scaling method, a network definition → **every** family's fingerprint changes, because the
  data or the search under all of them changed;
* change how the trained models are treated afterwards — the combine methods and their
  settings, `keep_best_models`, `interval_coverage`, `cluster_coverage_policy` → **no**
  family's fingerprint changes. Those settings are the case's and no family's: the combining
  step runs again on the trials that are there (§6 has the list and what it does for trials
  trained before the fingerprint left them out);
* change a budget → nothing: asking for 40 trials instead of 20 continues the study.

That is also the plain-language version of the restart rule: *how far back a change sends you
depends on how deep the thing you changed sits.*

**Trained from the beginning means exactly that.** A study that holds finished trials of
another configuration of its family is not continued beside them: its trial folders, its search
history and its results table are removed, and the search starts over with the whole budget —
what `--refit` does, for that study only. It used to be otherwise, in two ways that both left a
model made of two configurations: the deep families counted the old trials against the budget
and trained nothing, and the tree and foundation families added new trials to the old study;
either way the old trials stayed in the folder and were predicted, selected and combined.

Because trials are deleted, a run names each such study and what it removes before it opens
one (§6): in the browser that list has to be accepted with the case's name, on the command line
it is printed and the run goes on. What a run may start over is what it named — a study found
to be of another configuration without its family having been named stops the run instead. A
study that merely wants more trials is continued as before, and the **new** action still keeps
a separate history beside the old one for whoever wants both.

Three details make "that family, and only it" true:

* a tree or foundation study is also bound to the *data* of its own cluster — its masks, its
  selectors, the scalers. The configuration in that binding is the family's own too; it used to
  be the whole case's, so one network's learning rate sent every tree and every foundation
  model to be fitted again. Trials bound the old way are still recognised: their folders keep
  the content of every file they were fitted on;
* the folder of a trial records the fingerprint of its own family. Folders written before they
  did carry the case's, which any sibling's change moves, so for those the study — which always
  recorded the family's — is believed;
* a fingerprint recorded before 17 November 2026 also hashed the combination's settings. It is
  recognised as an earlier name of the same inputs: the one this configuration computes, or
  one the model kept from a configuration it was run under before (§6).

---

## 12. Combining, calibrating and evaluating

`run_combine.py` has five stages, and the order is the point.

| stage | what it does |
| --- | --- |
| `predict` | runs every surviving trial over the evaluation periods and caches its predictions |
| `select` | scores the trials on the **selection** slice and keeps the ones worth combining |
| `combine` | fits the combiners — the models that blend several predictors into one |
| `calibrate` | fits the prediction bands on the **calibration** slice |
| `evaluate` | measures everything on the **holdout** slice and writes the report |

**The three slices are not interchangeable, and this is the single most important honesty rule
in the pipeline:**

* **selection** chose the trials and fitted the combiners;
* **calibration** fitted the bands;
* **holdout** was read by the final report and by nothing else.

A number measured on a slice that was used to make choices is optimistic. Only `summary.csv`,
which is the holdout, is an honest estimate of how the model will behave on data it has never
influenced. This is also why the Results page warns you when you pick a predictor *because* it
scored best on the holdout: doing that turns the holdout into another selection slice, and the
honest number is gone.

What lands in `Results/`:

| file | what it is |
| --- | --- |
| `selection.csv` | which trials were kept, and their selection-slice scores |
| `evaluation.csv` | MAE / RMSE / bias per predictor, level, slice and step |
| `summary.csv` | the holdout numbers |
| `coverage.csv` | how much of each slice each predictor actually answered |
| `hyper_parameters.csv`, `hyper_parameter_report.txt` | every finished trial with its hyper-parameters and its errors, and which values did best — judged on the selection slice, checked on the holdout ([combine and evaluation](combine_evaluation.md#what-the-hyper-parameters-say-written-with-every-evaluation)) |
| `intervals_summary.csv`, `calibration.json` | the bands and their achieved coverage |

A case with several methods serves `models / kmeans` by default; a case that trained one method
serves `models / single`. The Results page names which one it is reporting rather than leaving
you to guess.

**The combine step predicts on the CPU, two threads a model, as many models at once as the
case's CPUs carry.** Every trained model is predicted once over the evaluation period, and that
was nearly all of a combine run: 718 of 759 seconds for forty-four models one after another, on
a machine with twenty CPUs. The `predict` stage therefore predicts its models side by side
(`inference predict starts: device cpu (asked cpu), 2 thread(s), its trials side by side on 16
CPUs`): five convolutional networks of a wind case took 27 seconds that way and 77 one after
another, and came out the same to the bit. Two threads a model is kept because a network's
numbers depend on the count -- one thread or sixteen moved them by 1e-7 -- and a prediction is
the same wherever it is made. `EFORECAST_INFERENCE_THREADS` says the count outright, and then
there is one worker with that many threads and one model at a time; `EFORECAST_INFERENCE_DEVICE`
(default `cpu`) puts that one worker on a card instead.

**A foundation model is predicted on a card when its family asks for one.** Chronos-2, TimesFM
and TiRex are the slow trials of a case: on a solar park with twenty weather columns as
covariates, Chronos-2 forecast three rows a second on a CPU thread and about 150 on an RTX
2080 Ti (TimesFM 400, TiRex 280), holding half a gigabyte of the card. So the worker of the
`predict` stage, which runs on the CPUs, starts one reader on each card for these trials and
predicts the others side by side on the CPUs meanwhile:

```
inference predict starts: device cpu (asked cpu), 2 thread(s), its trials side by side on 16 CPUs; Chronos2, TimesFM, TiRex on 2 card(s)
```

That park has 203 trials, 21 of them foundation models for seven clusters. Its `predict` stage
took 81 minutes with everything on the CPUs -- about seventeen hours of CPU time -- and under
eight with the 21 on two cards: 228 seconds for the seven Chronos-2 alone, 106 for TimesFM, 110
for TiRex.

A family asks for a card with `device: auto`, which is what a new case says: a card when one is
there, the CPUs when none is. `cpu` keeps a family on the CPUs; `cuda:1` or a GPU UUID names one
card. `cpu_study_device`, under `jobs`, says it for all three families at once -- for their
studies and for their predictions. **A case made before 14 November 2026 says `cpu` for
Chronos-2 and TimesFM**, because the preset did; change either setting to move it. Doing so
makes it no other case: nothing of a foundation model is learned on a device, so its studies,
its predictions and its finished stages stay as they are.

What to know about it:

- **The numbers are not the same digits.** A card and a CPU round differently. Against the
  predictions made on the CPUs, Chronos-2 and TimesFM differed by 4e-8 on average and 5e-6 at
  most, on values up to 2; TiRex, which has kernels of its own for a card, by 2e-5 on average
  and 1.4e-3 at most.
- **A card that cannot take a trial is not the end of it.** If the card's memory is taken by
  something else, or its driver fails, the trial is tried on a card once more after the others,
  and is then predicted on the CPUs; the stage says which (`RBF_rule_2 TiRex test_0 was not
  predicted on a card (OutOfMemoryError: CUDA out of memory); it is predicted on the CPUs`).
  That is the long way round: TiRex on the CPUs had not finished those seven clusters after
  three quarters of an hour.
- **One reader a card, and the family's own batch.** The cards are busy computing, not short of
  rows: Chronos-2 made 136 rows a second with a batch of 256, 120 with 1024 and 68 with 4096
  (and 6 GB). Two readers a card were 16 % faster for Chronos-2, 22 % for TimesFM and none for
  TiRex, for twice the memory, and are not started.
- A foundation model that stays on the CPUs is predicted side by side like the other trials;
  each reader then holds its own copy of the model, about 2 GB. Site graphs, and foundation
  models that a model image serves, are predicted one after another as before.

The long trials are given readers first, and predictions are taken as they finish rather than
in the order they were asked for. Both matter once trials differ this much: in order, the first
Chronos-2 held back every trial behind it, and with sixteen of them done the other readers had
nothing to start -- eight readers, two of them working.

---

## 13. From a trained model to a published forecast

A finished run produces **candidates**. Nothing a user sees has changed yet. Three deliberate
steps separate a candidate from a forecast.

**Capture a bundle.**

```bash
python -m eforecast.model_bundle --config <case> --intervals 0.8 --reference-mb 512
```

A bundle is not a copy of the model folder. It is the *declared* set of components needed to
serve: the weights, the scalers, the selections, the combiners, the calibration — plus bounded
reference inputs so the bundle can be replayed later and checked against the numbers it produced
when it was captured. Capture runs the model in a worker of its own as part of making the bundle,
so a bundle that cannot run is never produced.

**Approve it.**

```bash
bin/eforecast approve <case> /candidate/root --note "why"
```

Approval is three things in a fixed order: **copy** the bundle to an immutable snapshot,
**replay** it there, and only then **move the pointer**. The order is what matters. A mutable
training directory is never what serves a forecast, and a bundle whose weights no longer load
cannot become the approved one, because the replay happens before the pointer moves.

`bin/eforecast rollback <case>` moves the pointer back to the previous approved version, with the
same re-validation. It never retrains anything.

**Then, and only then, forecasts use it.** Finishing a run is not permission to publish from it.
Until a maintainer approves a version, the Results page says *experimental* — which is a
statement about permission, not about the numbers.

---

## 14. The daily forecast

```bash
python -m eforecast_online run --config <case> --at 2026-10-28T06:00 --days 1 --intervals 0.8
```

Normally a cron entry. One cycle does five things:

1. extracts the weather (and satellite frames, if used) for the days being forecast;
2. **pins** the approved bundle for the whole cycle and loads it — its scalers, selections and
   combiners, not whatever is currently in the training directory. The pin is a row in the
   database: the model cannot change under a cycle that is halfway through;
3. computes the forecast and writes it into an attempt-private folder, with a request to publish;
4. the host verifies the bytes, checks the forecast's **geometry** — the right number of hours,
   the right columns, the expected bands, numeric cells, no duplicate or misdated rows — checks
   that the inputs it was made from have not changed, and then, in one database transaction,
   records the issue and moves the published pointer;
5. `latest.csv` is projected from that pointer.

Step 4 is the one that keeps the published forecast trustworthy. Each of those geometry checks
exists because something that was not a forecast once looked like one: half a day of hours, a
missing band, a text cell where a number belonged.

If there is no publication authority — a standalone machine, a database that cannot be reached —
step 4 does not happen. The cycle writes a clearly labelled **preview** under
`predictions/previews/<attempt>/` and says so. It never overwrites the published forecast. The
same rule in one sentence: **a failed attempt never replaces a good forecast.**

```bash
bin/eforecast forecast-status <case>
```

gives you two separate answers — what is published, and whether the last attempt worked —
because collapsing them into one lets a failure five minutes ago make a two-day-old forecast
look fresh.

---

## 15. What happens if you stop it in the middle

This is worth its own section because it is the most common thing that will confuse you.

Stop a dataset stage with Ctrl-C, or kill the debugger, and:

* a file it had not finished has a `.tmp` name and is nobody's dataset. On the way out of a
  Ctrl-C, what the stage had finished under a real name is removed and the outputs a rebuild had
  set aside are put back (`extract` keeps the days it finished, and `preprocess` its scalers:
  each of those checks its own output). No stage is wrongly marked done, because the receipt is
  written last;
* after a kill nothing runs on the way out, so `<stage>.running` is still there. The next run of
  that stage says `stage nwp did not finish last time; its outputs are rebuilt` and does the work
  again. `extract`, `preprocess` and the training and combine stages check their own output, so
  they say `it checks what that run left and does what is missing` instead: the days already
  extracted and the trials already trained are kept;
* the case's lock went with the process — the kernel held it — so the next run is not refused by
  a run that no longer exists. `holder.json` and `writer.json` may still name it. **Harmless**:
  the next run writes itself in and takes the next writer number.

Stop a training or combination run and the same holds, with one thing more to know: its
workers — the CPU studies, the clusterer, the prediction worker — and its trials **stop with
it** (§7). Nothing of the run goes on fitting in the case, nothing is left to reclaim, and the
next run starts as soon as the lock is free. What a worker had not finished is in its own
folder under `~/.local/state/eforecast/executors/compute/` with its log, and is not part of the
case.

A study continues from its journal: the trials it finished are kept, the one that was being
fitted is marked failed, and **Continue** trains what is left.

If a second run is refused because the case is still being written, the refusal names the
process (§7). When that process is yours and should not be running:

```bash
python -m eforecast.config.direct_stage status <the case's model folder>      # who holds it
python -m eforecast.config.direct_stage stop-mine <the case's model folder>   # stop it, if it is yours
```

---

## 16. Everything that can refuse you

A refusal in this pipeline always comes with a reason and one thing to do. The reason *is* the
instruction. These are all of them, and the one-line answer to "why does this exist at all":

| refusal | it blocks | because |
| --- | --- | --- |
| **the case's lock** | a second run of a case that is being written | two writers in one tree is corruption, not contention |
| **a changed configuration, in the browser** | a save, and then a run, that makes datasets be built again or a family be trained from the beginning — until it was read and accepted by the case's name. A run typed in a terminal is not refused: it prints the same list and goes on (§6) | the work it replaces cannot be had back, and putting the setting back keeps it |
| **work of another configuration** | training on datasets built from other variables; combining with a family that was trained under other parameters | a model is the model of one configuration, never a mixture of two |
| **a card that cannot hold a trial** | a trial larger than any card here, even empty | waiting for memory that will never exist helps nobody |
| **release coherence** | running on an image whose source has moved | a stale image returns a wrong number rather than an error |
| **readiness** | starting with inputs that do not cover the period | the failure would arrive an hour in, having cost the hour |
| **bundle qualification** | approving a bundle that does not replay here | hashing files once certified models nobody had run |
| **approval** | serving a version nobody approved | finishing a run is not permission to publish from it |
| **forecast geometry** | a short horizon, a missing band, a text cell, misdated rows | "half a day" once looked like a forecast |
| **publication authority** | an issue recorded without verified bytes, or made from inputs that have since changed | a database pointer moved while the files stayed old |
| **route change** | changing venue while work is outstanding | two venues must never claim one attempt |

---

## Where to go next

* running a case, in the browser → [user.md](user.md)
* the commands → [cli.md](cli.md)
* something went wrong → [troubleshooting.md](troubleshooting.md)
* running the installation → [administrator.md](administrator.md)

## Stages write in place

The installation serves many people, one at a time, so a stage does not hide its work from a
second run. Every stage of `run_datasets.py`, `run_training.py` and `run_combine.py` takes the
case's lock and works in the case's own folders:

- **One writer per case.** A second run is refused at once and told who holds the case: the
  stage, the user, the pid and when it started.
- **A file with its real name is a finished file.** Each dataset is written as `<name>.tmp` and
  renamed when complete.
- **An interrupted stage is done again, not trusted.** A stage leaves a `<stage>.running` marker
  in `.stage-artifacts` while it works; if the marker is still there next time, a dataset or
  model-preparation stage rebuilds its outputs.
- **A rebuild that fails puts the previous output back.** The old files are set aside as
  `<name>.previous`, restored if the stage fails, and removed only when it succeeds. Whatever
  the failed stage wrote under a real name is removed.
- **A rebuild that is killed outright can be undone.** Nothing runs after a kill, so the old
  output stays aside and the stage reads as rejected. The case's *Recover* action (or
  `POST /api/cases/<case>/stages/recover`) lists it, and applied, puts the old output back
  under the case's lock; the receipt that was there describes it again. Running the stage
  again does the same and then rebuilds.
- **Output without a receipt beside a `.running` marker is not trusted.** Without the marker
  it is old work of unknown provenance and is used as such; with it, it is what an interrupted
  run left, and a stage that needs it is refused until the stage is run again or recovered.
- **Inputs that change while a stage works stop it from being recorded.** A file a sync client
  marked as uploaded meanwhile has not changed: its change time moved together with its
  attributes and nothing else did (`eforecast/common_utils/file_stamps.py`; the same rule
  guards a trial's inputs while it is fitted and a worker's while it predicts -- see
  [troubleshooting](troubleshooting.md#-inputs-changed--in-a-folder-that-is-synced)).
- **Whether the disk has room is the operating system's to say.** A stage that runs out fails
  with that error, and what it had written before is what the next run of the stage rebuilds.
- **A receipt per stage** is written, which is what lets the next run reuse a finished stage.
  Output that was already in place with no receipt is recorded as legacy: inspected, of unknown
  provenance.
- **A file is read once for as long as it stays the same file.** A receipt records, for every
  input and output, its SHA-256 and the identity it was taken from: path, inode, size,
  modification time and change time. Where a file still has that identity the recorded digest
  stands and the file is not read again; anything written, replaced or merely touched has
  another identity and is hashed. (Every check used to read every file, so a run with nothing
  to do spent minutes hashing a case's weather on one core.) A digest taken within three seconds
  of the file's last change is not kept, because a second write in the same clock tick would
  not show. `EFORECAST_REHASH=1` reads everything again at every check: use it for a disk you
  no longer trust, since a file that rots without being written keeps its identity.

There is no private attempt directory, no copy of the model tree and no promotion step.

**Dataset stages run in the process you started.** What they print is in your terminal and in
`run_datasets.log`; a debugger attaches directly.

**Training and combination launch workers** — separate processes, each started with some
threads and perhaps a card ([compute workers](compute_workers.md)). A worker builds its result
in its own folder under the executor's state directory (`~/.local/state/eforecast/executors`),
and the verified files are copied into the case's model tree; a CPU study publishes each
finished trial into its trial folder itself. The stage holds the case meanwhile and records the
receipt afterwards.

This is the same for a run started from the web UI: the service starts the run, and each stage
it launches writes in place and reports a receipt the service verifies.
