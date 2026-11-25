# Studying the extract path with PyCharm

For the owner, learning the pipeline by stepping through it rather than reading it. It follows
one stage — `extract` — from the Debug button to a GRIB message and back to the receipt, and then
says how the other stages, and training, differ.

A stage of `run_datasets.py` runs **in the process you start**. There is no worker to attach to
and no private copy to find: a breakpoint in `stage_extract` is simply reached. Everything here
was run as written against a three-day scratch copy of `Total_wind`.

Line numbers are as of commit `b16ff88` plus a few lines. Function names are stable; if a number
is off, search for the name.

Companions: [how it works](how_it_works.md) ("Stages write in place" is the model this guide
walks), [troubleshooting](troubleshooting.md), and [os_lab](os_lab/README.md) for the kernel
primitives underneath (locks, atomic rename).

---

## 0. A playground that cannot hurt anything

**Never debug a stage against the real case.** Only `--dry-run` is safe there: a real run writes
the datasets that trained models depend on. Make a scratch copy instead:

```bash
python docs/study/make_study_case.py \
    /media/smartrue/HHD1/George/eforecast_workspace/users/joesider/cases/Total_wind \
    ~/eforecast_study
```

It prints the path of `~/eforecast_study/case.yaml`. The copy keeps the case's configuration,
reads the same weather archive, and writes everything under `~/eforecast_study/models`. Its
series is the real one cut to **2026-03-29 .. 2026-03-31** (pass two more arguments for other
days). Those days are chosen on purpose: `20260330` holds three zero-byte provider files, so the
gap logic runs in a study that takes half a minute.

**Set `EFORECAST_NWP_REFILL=off` while studying.** Otherwise `extract` reaches the network to
fetch the empty files again before it records its inputs. Off makes every run offline and
identical.

**The GRIB archive is read-only input.** `extract` reads `/media/smartrue/HHD2/ECMWF/new/data`
and writes pickles into the scratch group folder.

To start over, delete `~/eforecast_study/models`. To redo one stage, add `--recreate`.

---

## 1. The PyCharm run configuration

*Run → Edit Configurations → + → Python.*

| Field | Value |
|---|---|
| Script path | `<repo>/run_datasets.py` |
| Parameters | `--config /home/smartrue/eforecast_study/case.yaml --stage extract --serial` |
| Working directory | the repository root |
| Python interpreter | `/home/smartrue/pipe_env/bin/python` |
| Environment variables | `EFORECAST_NWP_REFILL=off` |

Three things about it matter:

- **`--serial` is what makes every breakpoint reachable.** Without it the days are handed to a
  pool of *spawned reader processes* (`bounded_pools.map_bounded`), and a breakpoint inside the
  extractor is in a process PyCharm is not watching. With it, line 88 of `run_datasets.py` is a
  plain loop in your process.
- If you do want the pool: *Settings → Build, Execution, Deployment → Python Debugger → "Attach
  to subprocess automatically while debugging"*. Expect one debugger tab per reader.
- Change `--stage` to study another stage: `nwp`, `target`, `row`, `lstm`, or several at once.
  A stage that already finished is **reused and its code never runs**; add `--recreate` to make
  it run again.

Run it once without breakpoints first. The console should end like this:

```
INFO run_datasets: stage extract    start
INFO run_datasets: extract ecmwf: 4 days, variables ['Pressure', 'Uwind', ...] ...
WARNING ...ecmwf_extractor: ECMWF 2026-03-30: 3 of 93 provider file(s) hold no bytes, ...
INFO run_datasets: extract ecmwf: {'done': 4}
INFO run_datasets: stage extract    done in   24.5s
```

Run it a second time and the plan says `extract  reuse  content and input identities match the
receipt`: the receipt is doing its job.

---

## 2. The spine: from the Debug button to a GRIB message and back

Put a breakpoint on each stop the first time through; *Resume* (F9) carries you to the next.

| # | Stop | Where | What happens | Look at |
|---|---|---|---|---|
| 1 | `_main` | `run_datasets.py:247` | arguments parsed, the case loaded into `static_data` | `stages`, `static_data['path_group_nwp']`, `static_data['path_data']` |
| 2 | `resolve` | `run_datasets.py:257` → `stage_catalogue.py:188` | the stage plan: which stages exist, what each needs and produces | `plan['nodes']`, `plan['products']['extracted_nwp']` |
| 3 | `decide` | `run_datasets.py:262` → `stage_resume.py:180` | reuse or rebuild, per stage, from the receipts on disk | `resume` — this is the table printed as "resume plan" |
| 4 | `observe_stage` | `run_datasets.py:306` → `run_reporting.py:107` | the stage boundary opens | `key`, `node['enabled']` |
| 5 | `require_inputs` | `run_reporting.py:115` → `stage_catalogue.py:567` | a fresh check that what this stage needs is there | `current['blocked']` — empty, or the reason you would be refused |
| 6 | reuse check | `run_reporting.py:132` | if the plan said reuse and the receipt still verifies, the stage is skipped here | `evidence['status']`, `evidence['reason']` |
| 7 | `DirectStage.__enter__` | `direct_stage.py:147` | the case is taken | — |
| 8 | `_lock` | `direct_stage.py:152` | one writer per case; `holder.json` says who | `.stage-artifacts/holder.json` on disk |
| 9 | `_space` | `direct_stage.py:153` | the free-space floor | `floor_bytes()` |
| 10 | `input_records` | `direct_stage.py:172` → `stage_artifacts.py:405` | every raw provider file this stage will read, hashed; empty ones become recorded gaps | `self.inputs['external:nwp_raw']['gaps']` |
| 11 | marker | `direct_stage.py:186` | `<stage>.running` is written; it is removed only on success | `.stage-artifacts/extract.running` |
| 12 | `run_stage` | `run_datasets.py:216` | `DatasetCreator` reads the series (line 227), then the stage function is called (line 232) | `dataset.dates` |
| 13 | `stage_extract` | `run_datasets.py:71` | one `EcmwfExtractor` per provider; line 88 is the loop over days | `days`, `variables`, `extractor.path_nwp` |
| 14 | `nwps_extract` | `ecmwf_extractor.py:322` | a day that already has a pickle returns `'exists'` at once | `t`, `self.pickle_file(t)` |
| 15 | `extract_day` | `ecmwf_extractor.py:289` | find the day folder, drop duplicate and empty files | `path_date`, `files`, `usable`, `empty` |
| 16 | `read_gribs` | `ecmwf_extractor.py:187` | open each GRIB, de-accumulate, drop steps across a gap | `gaps`, and at line 233 `base_step`, `step` |
| 17 | the write | `ecmwf_extractor.py:340` | `joblib.dump` to `<file>.tmp`, then `os.replace` — a file with its real name is whole | the `.tmp` file appearing and vanishing |
| 18 | `finish` | `direct_stage.py:289` | back at the boundary: inputs unchanged? (`_guard`, line 290) | — |
| 19 | `output_records` | `direct_stage.py:292` → `stage_artifacts.py:654` | every declared pickle must exist and load; days the provider never supplied are excused (line 689) | `paths`, `without` |
| 20 | the receipt | `direct_stage.py:338` | `.stage-artifacts/receipts/extract.json` is written; the marker is removed | `receipt['outputs']`, `receipt['inputs']` |

After stop 20 the `with` block at `run_datasets.py:306` closes, the lock is released, and the
loop moves to the next stage.

Do not step into `pygrib.open` or `message.data()` — they are C extensions. Put the breakpoint
on the line after and inspect `data`, `message.step`, `message.analDate`.

**The other dataset stages share stops 1–12 and 18–20.** Only the middle differs:

| Stage | Stop 13 becomes | Then |
|---|---|---|
| `nwp` | `stage_nwp` → `dataset_creator.py:147` `create_nwp_dataset` | reads the daily pickles; `files_manager.py:248` `save_nwps` writes through `_whole` (line 27) |
| `target`, `row`, `lstm` | `stage_target` / `stage_row` / `stage_lstm` | the matching `save_*` in `files_manager.py`, also through `_whole` |
| `split` … `features` | `stage_split` … `stage_features` | the classes each one names in `run_datasets.py:118-155` |

---

## 3. What the stage leaves on disk, and how to read it

While a stage is stopped at a breakpoint, look at the case's `.stage-artifacts` folder
(`~/eforecast_study/models/MyOwn/DemoCases_ver1/wind/Total_wind/long_term/model_ver1/.stage-artifacts`):

| File | Meaning |
|---|---|
| `holder.json` | who holds the case right now: stage, user, pid, start time. Gone when the stage ends. |
| `<stage>.running` | the stage is working, or was interrupted. Removed only on success. |
| `receipts/<stage>.json` | the stage finished: what it read, what it wrote, with hashes. |
| `writer.json` | a counter of stage runs on this case. |

Beside a dataset you may see `<name>.tmp` (being written) and `<name>.previous` (the output a
rebuild set aside; it comes back if the rebuild fails).

Three experiments worth doing once, each from a breakpoint inside `stage_extract`:

- **Start the same configuration a second time** (Run, not Debug). It is refused at once:
  `Total_wind is being written by another run: stage extract of Total_wind, started … (pid …)`.
- **Press Stop** (the red square). `holder.json` goes, `extract.running` stays. Run again: the
  log says `stage extract did not finish last time`, and the days already extracted return
  `'exists'` at stop 14, so only the remainder is done.
- **Run `--stage target --recreate` and stop inside `stage_target`.** `dataset_target_data.csv`
  is gone and `dataset_target_data.csv.previous` is there. Press Stop, run again normally: the
  previous file is put back first.

---

## 4. The extractor alone: no case, no stage

The fastest loop for the extract logic itself. Save as `scratch/study_extract.py` and Debug it.

```python
import logging
from eforecast.nwp_extraction.ecmwf_extractor import EcmwfExtractor

logging.basicConfig(level=logging.DEBUG, format='%(levelname)s %(message)s')
extractor = EcmwfExtractor(['Flux', 'Temperature'], '/media/smartrue/HHD2/ECMWF',
                           path_group_nwp=None)     # None: return the dict, write nothing
nwps = extractor.extract_day('2026-03-30')          # a day with three zero-byte files
print(sorted(k for k in nwps if k not in {'format', 'lat', 'long', 'variables', 'max_step_hours'}))
```

Breakpoints that pay for themselves: `ecmwf_extractor.py:187` (`read_gribs` entry — inspect
`gaps`), line 233 (`if field in ACCUMULATED:` — watch `base_step`, `base`, `step` and which arm
is taken), and the loop that reports dropped steps after the file loop.

---

## 5. Debugging one test instead

Every behaviour in section 3 has a test. Open the file, click the green arrow beside the test,
choose *Debug*:

| To study | Test |
|---|---|
| in-place writes, reuse, the busy refusal, the free-space floor | `tests/test_direct_stage.py` |
| a run killed while writing a dataset | `test_a_run_that_dies_while_writing_leaves_no_dataset` (same file) |
| a failed rebuild restoring the old output | `test_a_rebuild_that_fails_puts_the_previous_dataset_back` (same file) |
| a day the provider never supplied | `tests/test_stage_artifacts.py`, `tests/test_nwp_gaps.py` |
| the whole dataset chain on a synthetic case | `tests/test_dataset_compute.py::test_real_datasets_partition_and_global_scalers_match_inline_reference` |

*Settings → Tools → Python Integrated Tools → Default test runner: pytest*, working directory
the repository root.

---

## 6. Exercises

Do them in order; each one's answer is something you watch happen.

### Tier 1 — the extract code (section 4, the extractor alone)

**1. De-accumulation, by hand.** Break at the `if field in ACCUMULATED:` branch in `read_gribs`
with `variables=['Flux']`. Step through steps 0, 1 and 2 of cycle `0000` and write down `base_step`,
`base[0,0]`, `raw[0,0]` and the stored value at each.
*Answer afterwards:* why the first readable step stores its total unchanged, and why that is correct
rather than a special case.

**2. The gap, and the hour after it.** Same breakpoint, continue to step 41. Watch
`any(base_step < gap < step for gap in gaps.get(run, ()))` evaluate true.
*Answer afterwards:* what number would have been published for 41h without that branch, and why
nothing downstream could have detected it.

**3. A gap that costs nothing.** In `tests/test_nwp_gaps.py`, debug
`test_a_missing_step_0_costs_nothing_because_its_total_is_zero`. Break in `read_gribs` and watch
the same expression evaluate **false**.
*Answer afterwards:* why a missing step 0 is free but a missing step 1 is not.

**4. The misfiled copy.** Point the script of section 4 at `2026-04-30` and break in `one_file_per_step`. Watch 53
files get dropped.
*Answer afterwards:* what `read_gribs` would have computed for a duplicated step if that filter
were removed — and why the answer is worse than a repeated value.

**5. The step roster.** Break after `files = one_file_per_step(self.grib_files(path_date))` and
build, in the debugger console, the set of steps actually present:
`sorted(sources.forecast_step(f)[1] for f in files if sources.forecast_step(f))`.
*Answer afterwards:* where the hourly roster becomes 3-hourly, and why `max_step_hours=60` means
six of the fourteen empty files in this archive cost nothing at all.

### Tier 2 — the stage boundary (sections 1–3)

**6. Who decides that nothing runs?** Run `extract` twice. On the second run break at
`run_reporting.py:132`.
*Answer afterwards:* which three things `inspect_stage` compared before it let the stage be
skipped, and what you would have to change on disk to make it run again without `--recreate`.

**7. The excused day.** Make a study case for `2026-03-26 .. 2026-03-28` (the provider's folder
for the 27th holds only steps past 60 h). Break at `stage_artifacts.py:689`.
*Answer afterwards:* why `without` has one entry, why the stage still succeeds, and what happens
to that entry on the day the provider's files appear.

**8. The lock is the kernel's.** Stop at a breakpoint inside `stage_extract`, find the pid in
`holder.json`, and `kill -9` it from a terminal. Start the stage again.
*Answer afterwards:* why it starts at once although `holder.json` was never cleaned up — and what
would be different if the lock were a file that merely exists.

**9. A file with its real name is whole.** Break on `files_manager.py:27` (`_whole`) during
`--stage nwp --recreate`, step over `write(temporary)` and list the `DATA` folder before the
`os.replace`.
*Answer afterwards:* what a run killed at exactly that line leaves behind, and why the next run
is not fooled by it.

**10. Inputs that move.** Stop inside `stage_target`, append a blank line to
`~/eforecast_study/series.csv`, and resume.
*Answer afterwards:* which line raised, why the freshly written dataset is gone, and why that is
better than a receipt.

---

## 7. Training and combination are different in one way

`run_training.py` and `run_combine.py` use the same stage boundary (stops 4–11 and 18–20), but
the work in the middle is done by **workers**: separate processes, each started with some
threads and perhaps a card. Your debugger is attached to the script, not to them.

- When a worker starts, the console names its log: `… runs in its own process; its output goes
  to …/worker.log`. Follow it with `tail -f`. That log, not the console, has the traceback when
  a worker fails.
- A worker's folder is under `~/.local/state/eforecast/executors/compute/`. `input.json` is what
  it was asked to do and `worker_result.json` what it answered.
- To step through worker code, turn on *"Attach to subprocess automatically while debugging"*
  (section 1) or debug the test that drives it: `tests/test_cpu_stage_publication.py` for CPU
  training, `tests/test_inference_compute.py` for prediction and combination.

---

## 8. Traps

**A finished stage does not run.** If your breakpoint in a stage function is never reached, read
the "resume plan" lines at the top of the console: the stage was `reuse`. Use `--recreate`.

**Without `--serial`, extractor breakpoints are in other processes.** See section 1.

**Stopping the debugger is an interruption.** The `.running` marker stays and the next run
rebuilds that stage's outputs. That is the design, not a fault.

**Paused at a breakpoint, you hold the case.** Any other run on the same case — including one
from the web UI — is refused until you resume or stop.

**`sys_folder` decides where everything is written.** Check `static_data['path_data']` at stop 1
before letting a new configuration run; it must be under your scratch folder.

**PyCharm's "Rerun" keeps the old parameters.** After editing the configuration, start it from
the toolbar, not from the console's rerun arrow.
