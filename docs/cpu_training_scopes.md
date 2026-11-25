# Shallow CPU scopes and trial generations

Technical reference. Start with the [documentation index](README.md) or the
[administrator guide](administrator.md) for setup, prerequisites and recovery. The measurements
below describe the case they were taken on; another family or dataset needs its own.

Use `methods.cpu_scope` to choose `global`, `clusters` or `both`. The default is
`clusters`, preserving old cases. Global CPU training uses the enabled shallow
families in `global_methods`; cluster training uses `methods_cpu`. Deep global
families are handled by the deep training command. A non-fuzzy case can train a
global shallow model without creating artificial clusters.

```yaml
methods:
  cpu_scope: both
  global_methods:
    RF: true
  methods_cpu:
    RF: true
    lasso: true
```

Keep the corresponding `method_params` families configured. Shallow global
families do not require a network experiment template. Runtime overrides are
available through the CLI and the script catalogue used by API run submission:

```bash
python run_training.py --config case.yaml --cpu --cpu-scope global
python run_training.py --config case.yaml --cpu --cpu-scope both --cpu-method RF
EFORECAST_STUDY_ACTION=new python run_training.py --config case.yaml --cpu
EFORECAST_STUDY_REVISION=weather-v2 python run_training.py --config case.yaml --cpu
```

`--cpu-method` explicitly selects that method within the requested scopes. It
does not change or save the case's method flags. `--cpu-scope` takes precedence
over the case value. Missing optional scope flags normalize to the old request
shape when comparing accepted snapshots; no stored case or snapshot is rewritten.

Each global/cluster study has its own unit identity, masks, search and estimator
fit. Existing deep transfer units remain distinct from ordinary cluster studies.
Each study runs in a process of its own, and several run side by side
([below](#studies-side-by-side)); inside a study the trials stay serial.

There is no fitted-model alias from a global unit to a cluster unit, even if
their masks happen to match. This preserves scope diversity and avoids claiming
equivalence from a small dataset or matching method name. The existing forecast
coverage fallback is unchanged. New CPU trials record a unit binding covering
ordered dataset identities, supplied masks/selectors, global scalers/transforms,
parameters and seeds. Budget checks exclude a recorded conflicting binding.
Historical trials without it retain the legacy compatibility reading and are
counted as unverified; missing content hashes never authorize cross-scope sharing.

The parameters in that binding are the family's own, with everything every family
learns from -- and without the settings of the combination step, which no family
reads. They used to be the whole case's, so a change to one network's learning rate,
or trying another combiner, made every tree and every foundation study "another
unit", to be fitted again for nothing. A trial bound the old way is still recognised
as its unit's: its manifest keeps the content of every file it was fitted on, and
when those and its own family's settings are unchanged, nothing it depends on is
different.

A study that does hold finished trials of another configuration of its family -- its
parameters or search space changed, or the data under it was built again -- is not
continued beside them. It is trained from the beginning: its trial folders, journal
and results table are removed first, by a run that named it before it started
([how it works §11](how_it_works.md#11-what-makes-an-old-trial-reusable)).

## Storage and selection

Cases without a generation marker retain their historical layout:

```text
<scope>/<method>/study.journal
<scope>/<method>/test_0/
<scope>/results_<cluster>_<method>.csv
```

A new/named revision uses additive locations:

```text
<scope>/<method>/.generations/<revision>/
    static_data.pickle          # deep/manual runner input
    study.journal
    trials/                    # deep compatibility records
    test_0/
    results_<cluster>_<method>.csv
    selected_trials.json
<scope>/<method>/trial_generation.json
```

Only the publication of a successful, validated trial selects a generated history. The small marker contains a relative generation
name, so moving the case does not detach it. Failed/refused work does not switch
the selection. Continue follows the current marker; New revision routes both the
study and artifacts to its named generation. Rebuild targets the selected
generation only and remains an explicit destructive operation. Dataset products,
global scalers and frozen/shared-cluster partitions remain shared read inputs.
Automatic New revision is stable for the current learned inputs; use a distinct
explicit revision label when separate histories of identical inputs are wanted.

A study's generation is pinned when its unit opens; later marker/environment
changes cannot redirect an attached unit's trials. Generated study names carry
the layout suffix `__g1`. Older revision-qualified histories used shared trial
folders; the new namespace leaves those histories untouched rather than attaching
to their trial numbers. Unqualified legacy histories retain their original names.

Discovery, selection, prediction caches, model readiness, resume previews, study
inspection and transfer sources resolve the same method directory. GRAPH keeps
its existing independent immutable group-bundle contract. Private inference and
combination views also protect caches inside generation directories. Old trial
bytes are retained; the current marker is a per-method selection, not an atomic
whole-case model promotion API.

A malformed marker fails visibly rather than falling back to an unrelated legacy
model. Keep previous generation directories and diagnose the marker from the
trial folders it names.

See [compute workers](compute_workers.md) for what a study's process is given and how it ends.

## How a study runs

`run_training.py --cpu` calls `eforecast.training.cpu_dispatch`; existing imports of
`train_cpu.train_study` and `train_on_cpus` still work. The dispatcher loads no datasets,
estimators or Optuna. It decides how many studies run at once, starts each in a
[worker](compute_workers.md) and collects what it returns. A study is handed its configuration
and paths, never masks, loaded datasets or estimators, and its generation is pinned before it
starts.

Nothing is reserved for a study and none waits to be admitted. The operating system schedules
the studies like any other processes.

- **Threads.** A shallow estimator (or its cross-validation) is built with the study's threads,
  with one native thread underneath and serial outer loops. A foundation family computes with
  its threads natively: its worker is started with every thread it may ever be given, and
  before each forecast it tells torch its share of them at that moment. (Until 2026-11-14 it
  kept the first number. On a case with sixteen studies side by side, eleven of them foundation
  models with sixteen threads each, none of the eleven had finished after ten minutes; with one
  thread each the first five finished within five.)
- **Device.** Shallow models use the CPU. A foundation family follows `cpu_study_device`, else
  its own `device`, else `auto`: a card when one is there. A new case says `auto` for all three;
  a case made before 14 November 2026 says `cpu` for Chronos-2 and TimesFM, because the preset
  did. A preference no visible card answers runs on the CPU and says so. One card runs one
  foundation study of a run at a time, and the shallow studies run on the CPUs meanwhile, with
  the CPUs that the studies queued for a card cannot use. The
  same preference places the family's trials in the `predict` stage of the combine step, and
  changing it leaves finished studies and predictions as they are
  ([how it works](how_it_works.md), "A foundation model is predicted on a card").
- **Publication.** A trial is fitted in a private folder, `test_<k>/.attempt-cpu-<id>/`, and is
  moved into `test_<k>` whole once it is complete and verified, under the folder's lock. Its
  Optuna record is written before that; success is reported after Optuna has confirmed it. A
  trial whose publication fails leaves no finished model, and a failed publication of a new
  generation puts the previous selection back.
- **Memory.** A study's memory goes back to the machine when its process exits.

### When a study fails

A study whose process fails **does not stop the others**. It is recorded as failed with its
reason and the path of its log, the rest are trained, and the run ends `partial`:

```text
CPU study rule_3:CatBoost:study did not finish: failure: ValueError: … (worker log: …/worker.log)
```

A study **the kernel killed** is the failure this way of running invites: with nothing set
aside, memory can run out, and Linux ends the largest process. Such a study is started once
more when every other study has finished, alone, with every CPU the case uses:

```text
… did not finish: failure: it was killed (SIGKILL); when nobody stopped it, that is the kernel
ending the largest process because the machine ran out of memory …; it is started again, alone,
when the other studies have finished
```

It continues from its journal, so the trials it had finished are not fitted again. If it is
killed alone too, it is recorded as failed; lower `cpu_workers` or the family's own size
(`n_estimators`, `iterations`, depth) and use **Continue**.

A foundation study **that fails on a card** is the other one. Nothing holds a card for a study,
so by the time it wants the card's memory something else may have it. Such a study is started
once more on the CPUs, where it is slower and not short of memory:

```text
… did not finish: failure: OutOfMemoryError: CUDA out of memory …; it ran on a card and is
started again on the CPUs
```

If it fails there too, it is recorded as failed like any other study.

### After an interruption

A study stops with the run that started it: when the run's process ends, however it ends, the
study's supervisor stops it. There is nothing to reclaim and nothing to wait out. Use
**Continue** (or run the command again): each study opens its journal, adopts a trial whose
model was published but whose Optuna record was lost, marks the trial that was being fitted as
failed, and trains what is left of its budget. A failed Optuna write does not make a finished
model disappear. **New revision** and **Rebuild/refit** keep their own meanings; refit is not
the way to recover from a crash. Continue removes nothing, with one exception it asks about
first: a study whose trials are of another configuration of its family (above).

## Studies side by side

```yaml
jobs:
  n_cpus: 16               # CPUs the CPU studies of a run use between them
  cpu_workers: null        # studies at once; null = as many as there are, up to one a CPU
  cpu_study_threads: null  # most threads a study uses; null = the CPUs divided among those running
  cpu_study_device: null   # foundation families: cpu, auto, cuda, cuda:N or a GPU UUID
```

| Setting | Meaning |
| --- | --- |
| `n_cpus` | How many CPUs the studies use between them, never more than this process may use (affinity, cgroup quota). Empty means all of them. |
| `cpu_workers` | How many studies run at once. Empty: as many as the run has studies, up to one a CPU, or up to as many as get the threads `cpu_study_threads` asks for. |
| `cpu_study_threads` | The most threads one study uses. Empty: the CPUs are divided among the studies running. |
| `cpu_study_device` | Device preference of foundation studies; empty inherits each family's own. |

`n_jobs` remains the setting of the dataset readers and is not used here.

**How the CPUs are divided.** With `n_cpus: 16` and twelve studies, twelve start at once: four
with two threads and eight with one. When a study finishes, its threads go to the ones still
running, fewest first, and each study builds the estimator of its next trial with what it has
by then; a trial keeps the threads it started with. So the CPUs the case uses stay in use until
the last study ends. With `cpu_study_threads: 4` the same case runs four studies at a time with
four threads each, and no study is ever given more than four. The first line of the log says
what was decided:

```text
CPU studies: 12 at a time, 1 thread(s) each or more (12 to train, 16 CPUs)
CPU studies: 4 at a time, 4 thread(s) each (12 to train, 16 CPUs)
```

Methods are interleaved over their global and cluster units, so one method's clusters do not
all go first.

Command-line and API overrides are kept with the accepted run and never save the case:

```bash
python run_training.py --config case.yaml --cpu --cpu-workers 4 --cpu-study-threads 4
python run_training.py --config case.yaml --cpu --cpu-study-device auto
```

These flags need `--cpu`. They change neither the identity of a learned model nor data
partitions, scalers, study generations or finished trial budgets: a run with other numbers
continues the same studies.

**Settings that no longer exist.** `cpu_study_ram_mb`, `cpu_study_vram_mb` and
`cpu_wait_timeout` reserved memory for a study and bounded its wait for admission. There is
nothing to reserve and nothing to wait for now. A case file or a stored run that still names
them loads as before and they are ignored; the command line no longer has
`--cpu-study-ram-mb`, `--cpu-study-vram-mb` or `--cpu-wait-timeout`.

## What the Runs page shows

The Runs panel offers the CPU scope and the four settings above. Blank controls inherit the
saved case; Preview shows what would be used; Start submits the same overrides. Neither saves
the case.

While CPU training runs, **Resources** shows how many studies run at once and with how many
threads, each study (running, or waiting for its turn) with its threads and device, the memory
the run's processes hold (PSS, measured), and the models finished per minute. The same snapshot
is written to `<path_model>/cpu_resource_status.json`. These are observations of what is
running, not limits, and a snapshot of a run that has ended is not evidence about a process.

The completion rate counts the new COMPLETE trials of the studies collected so far, over the
time since dispatch began. It excludes reused trials, pruned trials and failures, and it moves
when a study is collected, not after each trial.

## Measurements

Twelve CatBoost studies of two trials each, on a scratch copy of a wind case with 8,000 rows,
`n_cpus: 16`, on a 20-CPU workstation (13 November 2026). Every run produced the same 24 trial
values, bit for bit.

| Studies at once | Threads each | Wall time |
| --- | --- | --- |
| 1 (the former default) | 16 | 1740 s |
| 12 (automatic) | 1 or 2, then what finished studies gave back | 1423 s and 1454 s |
| 8 (`--cpu-workers 8`) | 2 or more | 1343 s |
| 4 (`--cpu-workers 4`) | 4 or more | 1377 s |

Side by side was 16 to 23% faster than one at a time. The three side-by-side settings are
within what the machine's other activity moved them (tests were running during the last two),
so they do not rank one another. A study's process held about 0.8 GB. Two trials a study is
the least favourable case for handing threads on, because a trial keeps the threads it started
with; studies with many trials gain more from it.

An earlier measurement on another case (four RF studies with four threads each against one
study with sixteen) found 3.3 to 3.4 times the throughput. RF and small tables scale worse
with threads than CatBoost does, so they gain more from running side by side. Other families
and full weather datasets need their own measurement.
