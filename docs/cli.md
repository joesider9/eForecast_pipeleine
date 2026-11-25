# Using it from the command line

Everything here runs from the repository root. Load the installation's own settings first, so
that commands answer about the lab rather than about your terminal:

```bash
set -a; . /etc/eforecast/webui.env; set +a
```

`bin/eforecast` takes seven of those settings from the file by itself, when your shell has not
set them, and says on stderr which: the interpreter, the workspace, the release, the backup
destination, the browse roots, the metrics port and the login switch. It does not take the
others — in particular not the ones that say where the run database is (§4). The spelled-out
`python -m eforecast.config.…` form takes none.

**`<case>`** means two different things, and the difference matters:

* after `--config` in a `run_*.py` script: a **path** — a YAML file, a parts directory, or its
  `main.yaml`;
* after a `bin/eforecast` verb: a **workspace case name**. A name this installation does not
  have is refused, not created.

---

## 1. The pipeline: four scripts, in order

```bash
python run_datasets.py --config <case> --stage all
python run_training.py --config <case>              # deep models, on the GPUs
python run_training.py --config <case> --cpu        # shallow models
python run_combine.py  --config <case>
```

Always look before you leap:

```bash
python run_datasets.py --config <case> --dry-run    # the plan, and what already exists
python run_combine.py  --config <case> --dry-run
python plan_revision.py --config <case>             # what a change would make you redo
python plan_revision.py --config <case> --action rebuild --json
```

### run_datasets.py

| flag | what it does |
| --- | --- |
| `--stage S [S …]` | which stages: `extract nwp images target row lstm split preprocess clustering features`, or the groups `datasets` and `all` |
| `--recreate` | delete the outputs of the selected stages first |
| `--recreate-extraction` | also redo the raw weather extraction |
| `--dry-run` | resolve the configuration, print paths and the plan, change nothing |
| `--serial` | no process pools, `n_jobs=1` — for debugging |
| `--log-level` | `DEBUG`, `INFO`, … |

### run_training.py

| flag | what it does |
| --- | --- |
| `--refit` | delete the studies and trial folders, and search again |
| `--gpus N [N …]` | which GPUs to use (default: all the case declares) |
| `--n-jobs N` | maximum concurrent trials per GPU |
| `--global-method M` | train only this global method |
| `--cluster-method M` | train only this per-cluster method |
| `--cpu` | train the shallow CPU families instead |
| `--cpu-scope {global,clusters,both}` | which scope the CPU families are trained for |
| `--cpu-method M` | train only this CPU family |
| `--cpu-workers N` | how many CPU studies run at once (left out: as many as there are, up to one a CPU) |
| `--cpu-study-threads N` | the most threads one study uses (left out: the CPUs are divided among the studies running) |
| `--cpu-study-device D` | device of the foundation studies of this run: `cpu`, `auto`, `cuda`, `cuda:N` or a GPU UUID (left out: the case's `cpu_study_device`, else each family's `device`, else `auto`) |
| `--docker` / `--image IMG` | run each trial from the worker image |
| `--storage URL` | the Optuna storage (default `EFORECAST_OPTUNA_STORAGE`) |
| `--halve-batch` | halve the batch size on the second out-of-memory retry |
| `--prepare-only` | ask one trial per study, write its parameters, start nothing |
| `--metrics-port N` | expose this run's progress for monitoring: each study's best error, the failures ([GPU training](gpu_training_optuna.md#monitor-and-recover)) |

Exit codes: `0` succeeded (including "some trials failed" and "nothing to do"), `1` there was
work and nothing completed, `2` cancelled.

### run_combine.py

| flag | what it does |
| --- | --- |
| `--stage S [S …]` | `predict select combine calibrate evaluate`, or `all` |
| `--refit` | recompute the cached predictions and refit the combiners |
| `--dry-run` | preview the dependencies and prerequisites |

### When the configuration has changed

A case builds one model, the model of one configuration
([how it works §6](how_it_works.md#one-model-one-configuration)). When the case was edited after
something was built from it, the next run finds work that is whole and was built from settings
the case no longer has: datasets made from other variables, a family trained under other
parameters. **A run from the command line replaces it, and says what it replaces first.** There
is no question and no flag: whoever edited the case and typed the command is told what that
costs, in the log, before the first stage starts:

```
The configuration of pv_park is not the one this was built from.
  changed since it was built:
    methods.method_params.TRANSFORMER.model_name.choices: ["iTransformer", "PatchTST"] -> ["iTransformer"]
  trained from the beginning, their trials removed first:
    TRANSFORMER 8 studies, 101 trial folder(s)
  this run replaces them now. Stop it within the next stage if that is not what you want;
  --dry-run shows the same list and changes nothing
```

* **`--dry-run` first** when you are not sure: it prints the same list and changes nothing;
* **putting the setting back** makes what is built this case's again, and the next run
  replaces nothing;
* what the run names is what it replaces. A study that turns out to hold trials of another
  configuration without its family having been named stops the run instead of being emptied;
* **a run started from the browser is asked first**: its preview shows this list, and it
  starts when the name of the case has been typed. So is the save that made the change
  ([user.md](user.md)). The confirmation lives there, where a change is made and a run is
  started by people who did not necessarily write the YAML.

`--recreate`, `--recreate-extraction` and `--refit` say themselves what they replace.
`run_combine.py` replaces only what is derived in minutes and computed again whenever the
datasets or the models change — but it does not *start* while a family that is switched on
still holds trials of another configuration:

```
predict requires models: the trained models of TRANSFORMER are of another configuration of it
-- TRANSFORMER (7 studies, 91 trial folder(s)) -- and are not part of this model; train it
again, or put the parameters back
```

The `changed since it was built` lines come from the configuration the model keeps of itself
(`<model folder>/.stage-artifacts/configuration/`); a model built before that was kept shows
the rest of the message without them.

### Flags every script shares

| flag | what it does |
| --- | --- |
| `--model-cache-mode {prefer-local,online,offline,prewarmed}` | where foundation-model weights come from |
| `--log-level L` | how much the script prints |

The flags of earlier versions that reserved or tuned resources are gone with what they set:
`--execution-profile`, `--profile-resources`, `--array-reader`, the `--cache-…` bounds,
`--cpu-study-ram-mb`, `--cpu-study-vram-mb` and `--cpu-wait-timeout`. A case file that still
names those settings loads as before and they are ignored.

### clean_model.py — reclaiming space

```bash
python clean_model.py --config <case> --dry-run        # the plan and the sizes, change nothing
python clean_model.py --config <case> --keep-days 30
python clean_model.py --config <case> --restore        # undo: move TEMP back
```

By default it **moves** things to `<path_model>/TEMP` rather than deleting them, which is what
makes `--restore` possible. `--delete` skips the safety net. `--skip-trials` and
`--skip-datasets` limit it to one or the other; `--drop-checkpoints` also parks the checkpoints
of trials it keeps; `--purge-temp` empties the parked folders for good; `--yes` skips the
confirmation.

Always `--dry-run` first.

---

## 2. `bin/eforecast` — the installation

The four scripts of §1 build and train a case. `bin/eforecast <verb>` is about everything
around them: is the installation healthy, which model does a case forecast with, was today's
forecast published, are the Docker images current, is there a backup, where do runs execute,
where are passwords kept.

Each verb either **looks** — reads, and changes nothing — or **changes** something. `<case>`
is the case's *name* in the workspace, not a path.

| verb | the question it answers, or what it does | does it change anything? |
| --- | --- | --- |
| `doctor` | is this installation working, and what do I do about what is not? | no (`--snapshot` saves the reading) |
| `forecast-status <case>` | is a forecast published, and did the last attempt work? | no (`--repair-export` rewrites `latest.csv`) |
| `approved <case>` | which model does this case forecast with? | no |
| `approve <case> <bundle>` | use this model for the forecasts of the case | **yes** — the approved version |
| `rollback <case>` | go back to the model approved before | **yes** — the approved version |
| `images` | do the Docker images still match the code? | no |
| `images refresh` | rebuild the Docker images from the code | **yes** — the images and the release record |
| `backup copy`, `restore` | keep a checked copy of a model on another disk; bring one back | **yes** — they write files |
| `backup list`, `verify` | which copies exist, and is one still whole? | no |
| `route`, `route <case>` | on which machine do runs execute? | no |
| `route <case> local`\|`broker` | decide it | **yes** — the route |
| `secret set` | store a password or key | **yes** — one file |
| `secret list`, `where` | which are stored, and where | no |
| `metrics` | the numbers a monitoring dashboard reads | no |
| `status` | the whole picture: the machine, the trials training, the studies and what is left, the processes | no |
| `trials` | which trials are training now, their error at every epoch, the loop each is in and how fast | no |
| `studies` | what every study of every case has done and has left; the trials that finished last | no |
| `processes` | every process of the pipeline and what it uses; the heaviest other processes | no |

**Which of these you need.** On one workstation, with cases run from the browser or a terminal,
`doctor` is the only one you use regularly. `approve`, `rollback`, `forecast-status` and
`backup` matter from the day a case produces a **daily forecast** somebody reads. `images`
matters only when something runs in Docker, `route` only with a cluster, `secret` when a data
source or a hosted model needs a key, `metrics` when a dashboard is set up.

`bin/eforecast help` prints every verb with its options and exit codes.

### `doctor` — is this installation working?

```bash
bin/eforecast doctor                     # check everything and say what to do
bin/eforecast doctor --case <case>       # also check one case and its retained models
bin/eforecast doctor --verbose           # list the checks that passed too
bin/eforecast doctor --json              # machine-readable
bin/eforecast doctor --snapshot          # also record the reading the UI and new runs read
bin/eforecast doctor --client http://host:8000    # only what a browser needs
```

One verdict — *ready*, *needs attention* or *must be fixed* — then one line per problem with
what follows from it and the one thing to do. It only reads; `--snapshot` additionally saves
the reading where the web page's health chip takes it from. The same checks run behind that
chip.

Exit: `0` ready, `1` needs attention, `2` must be fixed, `3` the doctor could not run.

### The three words the next verbs use

A finished training run changes nothing for the people who read forecasts. Three separate things
stand between it and a forecast, and each has a verb:

* A **bundle** is a trained model packed for forecasting: a folder with a list (`bundle.json`)
  of exactly the files a forecast needs — weights, scalers, selections, combiners — a checksum
  for each, and a small reference prediction (`fixture.json`) to test it against. You make one
  with `python -m eforecast.model_bundle --config <case>`; it prints `Candidate: <folder>`.
* **Approving** a bundle makes it the one the daily forecast of a case uses. The record of that
  — the *approved version* — is one row per case in the run database, with a revision number
  that goes up by one at every approval or rollback.
* **Publishing** is what the daily forecast does when it has passed its checks: the database
  then points at that forecast as *the* forecast of the case, and `latest.csv` is a copy of it.

A bundle is a folder of files. Approvals and publications are records in the run database
(§4, "Where records are kept"), not in the case's files.

### `approved`, `approve`, `rollback` — which model a case forecasts with

```bash
bin/eforecast approved pv_park                 # look: which bundle is in use
```

Read-only. With nothing approved it says so, and exits `1`:

```
no approved version: routine forecasts for this case have no model.
Approve a validated bundle with: bin/eforecast approve <case> <bundle>
```

With an approved version it prints the revision, a short id of the bundle, the folder the
approved copy lives in, who approved it and their note, and one `previous revision …` line for
each earlier version:

```
active   revision 3  9f2c41d07ab35e18
         /…/model_ver1/.approved-bundles/9f2c41d07ab3…
         approved by george -- qualified on the holdout
         runtimes: …
previous revision 2  51c0e6aa90d2b7f3  /…/.approved-bundles/51c0e6aa…
```

(`runtimes` lists what the bundle needs in order to run: the pipeline's own environment, or a
model image.)

```bash
bin/eforecast approve pv_park /path/to/candidate --note "qualified on the holdout"   # changes state
```

Makes that bundle the version in use. In this order:

1. **copies** the bundle's files to `<parent of the bundle>/.approved-bundles/<id>/`, so that
   what forecasts read can no longer be changed by training in the model folder;
2. **replays** the copy in a process of its own — loads the weights and recomputes the reference
   prediction — and compares the numbers with the ones stored at capture;
3. only then **records** it as the approved version, with revision = previous + 1.

Nothing is retrained and the candidate folder is not touched. `--expect-revision N` refuses if
somebody else approved in between. Exit `2` with `refused: …` when the bundle does not validate
here (a missing `bundle.json`, a file whose checksum no longer matches, a prediction that is not
reproduced); exit `3` with `nothing was changed: …` when the approved version moved meanwhile.

```bash
bin/eforecast rollback pv_park                 # back to the version before the current one   (changes state)
bin/eforecast rollback pv_park --to 2          # ... or to a named revision
```

Points the case back at an earlier approved bundle — the copy that approval kept — after
replaying it again. It is recorded as a **new** revision that names the old bundle, so the
history only grows, and a second plain `rollback` returns you to where you were. It never
retrains, deletes nothing, and does not read a backup. It refuses when there is no earlier
version, or when the kept copy is no longer what was approved.

`<case>` is a case name of this installation; an unknown name is refused, never created.

### `forecast-status` — is there a forecast, and did the last attempt work?

```bash
bin/eforecast forecast-status pv_park
```

Read-only. Two separate answers, because they are different questions:

```
forecast  issued 2026-11-15 05:10 UTC (3.2 h ago)
attempt   the last attempt published <issue id>
```

| line | what it says |
| --- | --- |
| `forecast` | the forecast that is **published** for the case and when it was issued; `-- STALE` when that is more than 6 hours ago; `none published yet` when there has never been one |
| `attempt` | what the **latest forecast run** did: `published …`, or `was not published (<reason>)`, or `no attempt has been recorded` |

A failed attempt never replaces a published forecast: the first line can be a good forecast of
this morning while the second says tonight's run was rejected, and why. The reasons are listed
in [troubleshooting.md §4](troubleshooting.md#4-the-forecast-looks-old-or-wrong).

Exit `0` a current forecast and no rejected attempt; `1` the forecast is stale or the last
attempt was rejected; `2` nothing is published. That makes it usable in a cron job or a monitor.

```bash
bin/eforecast forecast-status pv_park --repair-export /path/to/predictions    # rewrites one file
```

`latest.csv` in the predictions folder is only a **copy** of the published forecast, made for
whoever reads files. If writing that copy failed after the forecast itself was accepted (a full
disk, a network folder that was away), the database is right and the file is old. This form
writes `<folder>/latest.csv` again from the forecast the database points at, checks its
checksum, and prints `export    restored <issue> to <folder>/latest.csv`. It computes nothing
and changes nothing in the database.

### `backup` — a second copy of a model, on another disk

**The idea.** What cannot be recomputed is small: the bundle — weights, scalers, the files the
forecast reads. This verb copies exactly the files a bundle lists to another disk, as a dated
folder called a **generation**, and can prove later that the copy is still whole.

```bash
bin/eforecast backup copy /path/to/bundle /media/backup_disk/eforecast --case pv_park --label 2026-11-15
bin/eforecast backup list    /media/backup_disk/eforecast pv_park
bin/eforecast backup verify  /media/backup_disk/eforecast/pv_park/2026-11-15
bin/eforecast backup restore /media/backup_disk/eforecast/pv_park/2026-11-15 /somewhere/new
```

| form | what it does | changes |
| --- | --- | --- |
| `copy <bundle> <destination>` | copies every file the bundle lists into `<destination>/<case>/<label>/`, re-reads each copy to check it, and writes the generation's own list (`generation.json`) last | writes the new generation only |
| `list <destination> <case>` | the generations there are for a case, with the size of each; `INCOMPLETE` marks a copy that was interrupted | nothing |
| `verify <generation>` | re-reads every file of a generation and compares it with the checksum recorded at copy time — `ok`, or `fail` with the file and why | nothing |
| `restore <generation> <into>` | verifies the generation, then copies its files into `<into>` and checks them again; notes the restore in `<destination>/<case>/restores.json` | writes `<into>` |

* `--case` defaults to the case named in the bundle, `--label` to the current UTC time.
* **The destination must be another device.** A destination on the same disk as the bundle is
  refused: `… is on the same device … Another directory on the same disk does not survive the
  disk.` (Two partitions of one disk pass this check and are still one disk — choose a
  different drive.) `EFORECAST_BACKUP_DIR` (§4) is where the doctor *expects* backups to be; the
  commands themselves take the destination as an argument.
* **A restored bundle is not in use.** It is a folder. `bin/eforecast approve <case> <into>`
  replays it and makes it the approved version, as for any other bundle.
* **What is not in it:** the run database (approvals, published forecasts), the case's
  configuration, credentials, datasets and trial folders. The database has its own backup —
  [administrator.md §6](administrator.md#6-backup-and-restore--changes-state).

Any failure exits `2` with `nothing was changed: …`.

### `images` — do the Docker images still match the code?

**The idea.** Most work here runs *natively*: the pipeline's own Python environment on this
machine. Four Docker images exist for the work that does not, and each carries **a copy of the
code** from the day it was built:

| image | what runs in it | you need it when |
| --- | --- | --- |
| `worker` | a training trial | you train with `run_training.py --docker` |
| `online` | the daily forecast | a machine that only forecasts runs from the image |
| `foundation` | the pretrained time-series families | a case uses a family this machine cannot import (TimeGPT, Timestral, T0 …), or asks for the image through `methods.runtime_ids` |
| `lag_llama` | Lag-Llama | a case uses Lag-Llama |

Chronos2, TimesFM and TiRex are importable here and run natively unless a case asks otherwise.

A copy goes out of date the moment the code in the checkout changes, and an out-of-date image
does not fail: it computes with yesterday's code and returns **a different number**. So the
images are compared against a **release**: a named record of what each image should be, kept in
`docker/releases/<release>.json` — the image's id, the hash of its Dockerfile and of its
requirements lock, and a hash of every source file it carries. `EFORECAST_RELEASE` in the
settings file says which record this installation goes by.

A **witness** is the last piece: the image itself is started once and asked what it contains,
and its answer is written to `<workspace>/.run-state/image-runtime.json`. It is the image's own
statement, as opposed to what the build was supposed to produce.

```bash
bin/eforecast images                    # compare, change nothing
bin/eforecast images check 20261029     # ... against another release
```

Read-only (it starts a short-lived container per image to hash the files inside). It must be
run from the repository root. One line per image:

```
release 20261029-dst-cleanup
  online       stale  (eforecast-online:20261029-dst-cleanup)
      - source: 104 file(s) changed: configuration/experiment_template.py, …
      - implementation: … the image was built from different code
      -> docker build -f eforecast_online/Dockerfile -t eforecast-online:step02 .
  worker       stale  (eforecast-worker:20261029-dst-cleanup)
  foundation   coherent
  lag_llama    coherent
```

| status | meaning |
| --- | --- |
| `coherent` | the image is what the release says, and matches the code in the checkout |
| `stale` | something differs; each `-` line says what (`source`, `lock`, `recipe`, `base`, `image`, `implementation`, `witness`) and `->` is the command that repairs it |
| `missing` | the image is not on this machine |
| `unknown` | the release record does not describe it |

Exit `0` all coherent, `1` something is stale or missing, `3` no release of that name is
recorded. A release other than the configured one needs the word `check` in front of it.

**What a stale image costs you.** Nothing, for work that runs natively — which is every stage
of an ordinary case. The doctor and the health chip report it (`needs attention`). Work that
*would* go to a stale model image is refused instead of run: for a case that uses one of the
image-only families, readiness shows a blocking `runtime.release` finding, and the run says
`… is stale against release … A stale image produces a wrong number rather than an error, so it
is not used.`

```bash
bin/eforecast images refresh             # rebuild all four for the configured release   (changes state)
bin/eforecast images refresh 20261115    # ... as a new, dated release
```

`refresh` does the whole repair in one go, because the steps only make sense together:

1. **builds** the four images from the checkout as it is now (`docker build` — slow the first
   time, when the base images and the packages are downloaded);
2. **witnesses** each one — runs it once and writes its answer to the witness file;
3. **captures** the release — rewrites `docker/releases/<release>.json` to describe what was
   just built;
4. **checks** the result, and exits non-zero if anything is still not coherent.

The worker and online images are built under the release's own name
(`eforecast-worker:<release>`), so the tags a running deployment may still point at
(`eforecast-worker:latest`, `eforecast-online:step02`) are never moved, and the build a dated
tag replaces stays available as `<repo>:<release>-previous`. It needs `EFORECAST_WORKSPACE`
(the wrapper supplies it) so that the witness lands where the service reads it. The doctor and
the service go on judging by the release `EFORECAST_RELEASE` names: after a refresh under a new
name, put that name there and restart the service.

Run it after the code changed **and** you use one of the four cases in the table above.
Procedure and cautions: [administrator.md §3](administrator.md#3-keeping-the-images-honest--changes-state).

### `route` — on which machine does a run execute?

**The idea.** A run started from the browser is handed to a **venue**. There are two:
`local` — this machine, the tested path and the default — and `broker`, an optional route that
sends the run through RabbitMQ to a Kubernetes cluster. Which one applies is a recorded
**decision**, per installation or per case, not a setting in a file: it is stored in the run
database with who decided, when and why.

```bash
bin/eforecast route                     # the installation's venue
bin/eforecast route pv_park             # one case: its venue, and work still on its way
bin/eforecast route pv_park --history   # every decision for it, newest first
```

These three only read. On an installation where nobody has decided anything:

```
scope    installation
venue    local  (default: nobody has decided)
waiting  0 outstanding dispatch(es)
```

`scope` is what the answer is about (the installation, or one case — a case's own decision wins
over the installation's). `waiting` counts runs of that scope that are accepted but not yet
finished and cleaned up; `blocked` appears when the venue cannot be used and says why.

```bash
bin/eforecast route pv_park broker --note "trying the cluster for one case"    # changes state
bin/eforecast route pv_park local  --note "back to the workstation"
```

A decision adds one row to the history and applies to runs accepted **from then on**; a run
that is already accepted keeps the venue it was accepted under. Two refusals protect you:

* **exit 3 — work is outstanding.** `N dispatch(es) are outstanding …`: a run of that scope is
  still going, and changing the venue under it could let two machines claim the same run. Wait
  for it to finish (or stop it), then decide again.
* **exit 2 — `broker` is not connected.** Choosing `broker` is refused unless this installation
  has a cluster configured (`EFORECAST_KUBE_RUNS_FILE`). A broker address alone is not enough.

The scripts you start in a terminal (`run_datasets.py` …) do not consult the route: they run
where you start them. If you only ever use this workstation, the answer is `local` and you never
need this verb. Background: [administrator.md §9](administrator.md#9-where-work-runs-and-changing-it--changes-state).

**To see the cluster venue work without connecting anything**, one script makes a cluster of
one node on this machine, with a run database of its own, and runs a step of a case on it:

```bash
export EFORECAST_CLUSTER_HOME=/media/sider/data/eforecast-local-cluster
deploy/kubernetes/local_cluster.sh up --shared /media/sider/data       # once
deploy/kubernetes/local_cluster.sh image                                # after every change of the code
deploy/kubernetes/local_cluster.sh drill                                # the venue's own drill
deploy/kubernetes/local_cluster.sh run pv_park datasets --options '{"dry_run": true}'
deploy/kubernetes/local_cluster.sh stop                                 # start | status | down <name>
```

The installation's service, run database and routes are not touched by it. Each step, what it
prints and what such a cluster cannot do:
[deploy/kubernetes/README.md](../deploy/kubernetes/README.md#try-it-on-one-workstation-start-to-finish).

### `secret` — where passwords and keys are kept

**The idea.** A credential — the EUMETSAT key for satellite downloads, an OpenWeather key, the
broker address with its password, a token for a hosted model — must not sit in a case file, in
the code, or in your shell history. This verb keeps each one in a file of its own that only
your user can read, and the pipeline looks it up **by name** when it needs it.

```bash
bin/eforecast secret set eumetsat_consumer_key    # asks for the value twice, never shows it
bin/eforecast secret list                         # the names that are stored — never the values
bin/eforecast secret where eumetsat_consumer_key  # the path of that secret's file
```

* `set` reads the value from a prompt (or from standard input when piped) and never from the
  command line, so it does not end up in the shell history. It writes
  `~/.config/eforecast/secrets/<name>` with mode `0600` (or under `EFORECAST_SECRETS_DIR`),
  replacing a previous value. Names are lowercase letters, digits and underscores.
* `list` and `where` only read; `where` prints the path whether or not the file exists.

The names the pipeline asks for:

| name | used by | can also be given as the environment variable |
| --- | --- | --- |
| `eumetsat_consumer_key`, `eumetsat_consumer_secret` | satellite image downloads | `EFORECAST_EUMETSAT_CONSUMER_KEY`, `EFORECAST_EUMETSAT_CONSUMER_SECRET` |
| `openweather` | the OpenWeather weather source | `OPENWEATHER_API_KEY` |
| `broker_url` | the `broker` venue (`amqp://user:password@host:5672/`) | `EFORECAST_BROKER_URL` |
| `kube_token` | the Kubernetes venue | a token file, `EFORECAST_KUBE_TOKEN_FILE` |
| `hf_token`, `nixtla_api_key`, `tabpfn_token` | the hosted model families T0, TimeGPT, TabPFN | `HF_TOKEN`, `NIXTLA_API_KEY`, `TABPFN_TOKEN` |
| `session_secret` | signing the browser's login cookie | `EFORECAST_SECRET` |

An environment variable, when set, wins over the file. A secret that is missing stops the step
that needs it with the path to put it in. The PostgreSQL connection of the run store is **not**
a secret of this kind: it is the setting `EFORECAST_RUN_STORE_DSN` (§4).

### `metrics` — numbers for a monitoring dashboard

**The idea.** [Prometheus](https://prometheus.io) is a monitoring program that visits an address
every few seconds and stores the numbers it finds there; Grafana draws them. This verb is that
address for the *installation*: how many runs are waiting or active, whether any work is stuck
on its way to a venue, how much disk the workspace has left.

```bash
bin/eforecast metrics --print           # print the numbers once and exit — for a look
bin/eforecast metrics --port 9109       # serve them at http://<this machine>:9109/metrics
```

Read-only. You normally run neither by hand: the `eforecast-metrics` service runs the second
form permanently, on the port `EFORECAST_FLEET_METRICS_PORT` names (default `9109`; `0` serves
nothing). `--print` is how you see what monitoring sees:

```
eforecast_source_up{source="runs"} 1
eforecast_source_up{source="storage"} 1
eforecast_source_up{source="broker"} 0
eforecast_source_configured{source="broker"} 0
eforecast_runs_active 0
eforecast_cleanup_pending 0
eforecast_storage_free_bytes 3376276365312
```

| numbers | what they count |
| --- | --- |
| `eforecast_source_up{source=…}` | whether each of the five sources below could be read (`1`) or not (`0`) |
| `eforecast_source_configured{source=…}` | whether the installation is meant to have that source: `0` only for a broker or a cluster it does not use |
| `eforecast_runs_active`, `eforecast_runs_waiting{wait=…}`, `eforecast_cleanup_pending` | runs accepted from the browser that are executing, waiting, or finished but not yet tidied up |
| `eforecast_completions_verified_total{runtime=…}` | stages whose outputs were verified when they finished |
| `eforecast_dispatch_rows{state=…}`, `eforecast_dispatch_oldest_age_seconds`, `eforecast_dispatch_failures{…}`, `eforecast_dispatch_attempts_total`, `eforecast_dispatch_undeliverable` | work on its way to the `broker` venue: how much, how old, how often it failed |
| `eforecast_queue_messages{route=…}`, `eforecast_queue_unacknowledged{route=…}` | messages in the RabbitMQ queues |
| `eforecast_cluster_jobs`, `eforecast_cluster_pods_pending_seconds` | jobs on the Kubernetes cluster |
| `eforecast_storage_free_bytes`, `eforecast_storage_total_bytes` | the disk the workspace is on |

The first row matters most. The numbers come from five **sources** — `runs` and `dispatch` (the
run database), `storage` (the workspace's disk), `broker` and `cluster` (only when that venue
is configured). `eforecast_source_up{source="…"} 0` means *that source could not be read*, and
then none of its numbers are printed at all — so an empty panel on a dashboard is never
mistaken for "nothing is happening". On a workstation `broker` and `cluster` are `0` because
they are not configured; that is normal, and `eforecast_source_configured` says so with a `0` of
its own, so their alerts stay silent and the *Source availability* panels leave them out.
`--print` adds a `# note <source>: <why>` line for each source that is down.

Only runs **started from the browser** are counted: a script you start in a terminal is not a
run in the database, so `eforecast_runs_active` stays `0` while it works.

This is not the same as `run_training.py --metrics-port N`, which serves the progress of **one
training run** (trials running, completed, failed, GPU memory) while that run lasts.

### `status` — the whole picture

```bash
bin/eforecast status                     # once
bin/eforecast status --watch 10          # again every 10 seconds, until Ctrl-C
bin/eforecast status --case Total_wind   # one case; --all also lists its studies that are done
```

```
2026-11-24 08:20:04   processors 3.1 of 20 in use; memory 21 of 125 GB; card 0 0 %, 0.2 of 11 GB, 38 C; card 1 96 %, 6.0 of 11 GB, 44 C

TRIALS TRAINING NOW
case        cluster     method  trial    gpu  epoch   loop now           it/s  left    val     test    best val  best test  loops/s  epochs/s  running  last epoch
Total_wind  RBF rule_1  MLP     test_15  1    16/100  train 241/499 48%  1.12  3m 50s  0.0515  0.0448  0.0497    0.0437     1.14     0.00236   2h 03m   3m 40s ago

STUDIES
Total_wind: 49 studies, 14 done, 22 open; trials 320 complete, 87 pruned, 144 failed, 1 running, 116 left of 553; about 65.1 h of training left, one trial after another
  30 more trial(s) are lacking and will not be asked for: 17 studies have used up their attempts (twice the trials wanted). They are listed with what they gave up.
  study             complete  pruned  failed  running  left  given up  wanted    best  a trial  last finished
  RBF rule_0 CNN           3       3       0        0     9                15  0.0359      32m         2d ago
  RBF rule_1 MLP           9       5       1        1     1                15  0.0415      65m     7h 41m ago
  RBF rule_4 MLP          10       1      19        0     1         3      15  0.0414      94m    23h 06m ago
  ...
finished last
  when          case        study            trial    ended      value  took
  24 Nov 03:05  Total_wind  RBF rule_2 MLP   test_29  failed              0m
```

Everything the Grafana dashboard *eforecast training* shows, as text, from the same three readers:
the machine and each card in a line (a card that slows itself down for heat says so), then
`trials`, `studies` and `processes` below, which are also verbs of their own. Read-only; it
reads the processes, the trials' logs and the studies' files, and asks nothing of the training.

### `trials` — what is training now

```bash
bin/eforecast trials                 # the trials training on this machine now
bin/eforecast trials --watch 30      # again every 30 seconds, until Ctrl-C
```

```
case        cluster     method  trial    gpu  epoch   loop now           it/s  left    val     test    best val  best test  loops/s  epochs/s  running  last epoch
Total_wind  RBF rule_1  MLP     test_14  1    39/100  train 119/446 27%  1.04  5m 13s  0.0482  0.0427  0.0482    0.0427     1.04     0.00236   4h 44m   1m 55s ago
```

One line per trial: its case, cluster, method and folder, the card it was given, the last epoch it
finished of how many it may run, the loop it is in, the error of that epoch and the best so far
on the validation and the test set, how fast it trains, how long it has run, and how long ago it
finished an epoch -- one whose last epoch keeps growing old has stalled. It reads the trial
processes, the `progress.json` each writes after every epoch and the end of its `train.log`, so
it shows any training, whatever version of the code started it. Read-only.

* **loop now**, **it/s**, **left** -- the last progress bar of the trial's log: `train 119/446 27%`
  is batch 119 of the 446 of a training pass, at 1.04 batches per second, with 5 min 13 s to go.
  `eval` is a pass that does not train (validation, test); `loop` means no training pass has
  ended yet, so nothing says which kind this one is.
* **loops/s** -- training batches per second in the trial's last training pass: the
  `Run training step with … iter/sec` its `train.log` gets after every pass. Blank until the
  first pass of an attempt is over.
* **epochs/s** -- 1 over the length of its last epoch, validation and test included: 0.00604 is
  an epoch every 2 min 45 s. A trial started by code older than 21 November 2026 does not report
  that length; for it, the epochs over the time since it began, its setup included.

The metrics service serves the same at `/trials` (port 9109), which Prometheus reads as job
`trials` for the dashboard *eforecast training*; `--prometheus` prints that text here.

### `studies` — what is done and what is left

```bash
bin/eforecast studies                    # every case: its totals, and its studies that are not done
bin/eforecast studies --case pv_park --all
```

One line per case -- studies done and open; trials complete, pruned, failed, running, and left
of how many wanted; the training left if one trial ran after another -- then its open studies
(with `--all`, every study): the trials of each by how they ended, the best value, how long a
trial takes, when the last one finished; then the twelve trials that finished last. *left* is
what the study will still be given: its running trials and the ones still to be asked for. A
study is asked for at most twice the trials wanted of it (a trial stopped with its run, or whose
card was lost, is not counted), so one whose trials keep failing can end short. Then a line
says how many trials are lacking and will not be asked for, the studies concerned are listed
even when nothing of theirs is open, and a *given up* column says how many each one lacks:

```
Total_wind: 49 studies, 14 done, 22 open; trials 320 complete, 87 pruned, 144 failed, 1 running, 116 left of 553; about 65.1 h of training left, one trial after another
  30 more trial(s) are lacking and will not be asked for: 17 studies have used up their attempts (twice the trials wanted). They are listed with what they gave up.
  study             complete  pruned  failed  running  left  given up  wanted    best  a trial  last finished
  RBF rule_4 MLP          10       1      19        0     1         3      15  0.0414      94m    22h 43m ago
  RBF rule_6 CNN           3       2      21        0     5         5      15  0.0348      14m         2d ago
```

The scheduler counts the same way: both use `optuna_search.attempts`. The methods trained on the
processors have no such bound and are asked for what they lack at every run.

It reads the study files the trainings keep beside the models, for every case of the workspace
(`EFORECAST_WORKSPACE`). The metrics service serves the same at `/studies`; the Optuna dashboard
gets them, one for a case and a cluster, through the service `optuna-mirror`
([GPU training](gpu_training_optuna.md#monitor-and-recover)).

### `processes` — what runs, and what it uses

```bash
bin/eforecast processes                  # the pipeline's processes and the heaviest others
bin/eforecast processes --watch 5
```

One line per process, a script and the copies of itself on one line: what it is (a training, a
trial, a data loader, a combination, a worker, the pages, a service, the daemons that share a
card), its case, the processors it uses now, its memory, its memory on a card, its threads and
how long it has run. Role `other` is the heaviest of everything else on the machine. A process
is told by its command line, so nothing has to register itself. The metrics service serves the
same at `/processes`, with the machine's totals and each card's heat slowdown.

---

## 3. Modules you occasionally need directly

```bash
# the configuration document
python -m eforecast.config.serve --host 127.0.0.1               # the UI and API
python -m eforecast.config.auth add <name> [--admin]            # accounts
python -m eforecast.config.auth revoke <name>
python -m eforecast.config.schema_export                        # the JSON Schema

# who is running what
python -m eforecast.config.direct_stage status <path_model>     # who is writing this case
python -m eforecast.config.direct_stage stop-mine <path_model>  # stop it, if it is your own run
python -m eforecast.config.run_processes                        # what this installation is running

# models and forecasts
python -m eforecast.model_bundle --config <case> --intervals 0.8 --reference-mb 512
python -m eforecast.bundle_qualification /backup/<case>/<generation> --work /scratch/new-dir
python -m eforecast.config.model_approval active <case>
python -m eforecast_online run      --config <case> --at 2026-10-28T06:00 --days 1 --intervals 0.8
python -m eforecast_online forecast --config <case> --dates 2026-10-28 2026-10-29 [--out F]
python -m eforecast_online check    --config <case>             # what is missing before it runs

# venues: where a run executes
python -m eforecast.config.dispatch_route                         # the same as `bin/eforecast route`
python -m eforecast.venues.aws_pilot <envelope.json> [--check-account]
                                                                  # only checks the file that describes a
                                                                  # cloud trial; it starts nothing
```

The two optional venues — a Kubernetes cluster, and the one-machine cloud trial that tests it —
are explained in [deploy/README.md](../deploy/README.md#the-two-optional-directories-and-when-they-run):
what each is, when it is executed, and what happens step by step.

`<path_model>` is the case's model folder, the one that holds `DATA/`; a refusal prints it.

---

## 4. The settings that matter

A setting is an environment variable. Three things about them explain most surprises.

**Where they are kept.** On a configured machine they are lines of `/etc/eforecast/webui.env`.
`bash deploy/configure.sh --write` writes that file: it works out nine settings for the machine
— `EFORECAST_APP_DIR`, `_PYTHON`, `_WORKSPACE`, `_BROWSE_ROOTS`, `_AUTH`, `_HOST`, `_PORT`,
`_FLEET_METRICS_PORT`, `_RELEASE` — and keeps every other `EFORECAST_…` line that is already
there. Any other setting below is in the file only because somebody added it:
`sudoedit /etc/eforecast/webui.env`.

**Who sees them.** A program sees a setting only if it was started with it:

| started how | the settings it sees |
| --- | --- |
| the web service, the metrics service, and **every run started from the browser** | the whole file, read **once, when the service starts**. After editing it: `sudo systemctl restart eforecast-webui eforecast-metrics` |
| `bin/eforecast <verb>` | what your shell has set, plus seven it takes from the file when your shell has not: `EFORECAST_PYTHON`, `_WORKSPACE`, `_RELEASE`, `_BACKUP_DIR`, `_BROWSE_ROOTS`, `_FLEET_METRICS_PORT`, `_AUTH`. Its first line says which |
| anything else typed in a terminal — `python run_training.py …`, `python -m eforecast.…` | only what your shell has set. To give it the installation's settings: `set -a; . /etc/eforecast/webui.env; set +a` |

So a setting that changes how *runs* behave works for runs started from the browser once it is
in the file and the service was restarted, and for a run you start in a terminal only when your
shell has it (`export NAME=value`, or the line above).

**What "the default" means.** The value used when the setting is absent. Most installations
set only the first group.

### Where things are

| setting | what it is for | takes effect in · default |
| --- | --- | --- |
| `EFORECAST_PYTHON` | The Python environment that has the pipeline's packages. `bin/eforecast` runs its verbs with it, and the web service starts the script of a run with it. A `python run_….py` you type yourself uses whatever `python` your shell finds instead. Change it when the environment moves; `configure.sh` detects it. | `bin/eforecast`, runs started from the browser, the doctor · the interpreter that is running; e.g. `/home/you/pipe_env/bin/python` |
| `EFORECAST_WORKSPACE` | The folder where the web pages keep what people enter: the accounts, every user's cases, and `.run-state/` (the health reading, the image witnesses). **Not** the models — those are under each case's `sys_folder` — and not the run database. Point it elsewhere and every account, case and run record seems to vanish: they are still in the old folder. | the web service, most `bin/eforecast` verbs · none on a new machine: the service refuses to start without it; e.g. `/srv/eforecast` |
| `EFORECAST_RELEASE` | The name of the record the Docker images are compared with, `docker/releases/<name>.json` (§2, `images`). Changing it rebuilds nothing and deploys nothing: it only says what the images *should* be. Change it after `images refresh <new name>`. | the doctor and health chip, readiness, work routed to a model image, `bin/eforecast images` · the only record if there is exactly one, else none (the doctor says so); e.g. `20261029-dst-cleanup` |
| `EFORECAST_BACKUP_DIR` | Where backups of approved models are expected to be. Only the doctor reads it: unset, it reports `no backup destination is configured`; set, it checks that the folder is on another device than the workspace and tells you the last backup and the last tested restore of each case. The `backup` verbs do not read it — they are given the destination. | the doctor and health chip · none; e.g. `/media/backup_disk/eforecast_backups` |
| `EFORECAST_WEBUI_DIST` | The folder of built web pages the service shows. Building into a new, dated folder and pointing this at it lets you change the pages without replacing files people are using, and the previous folder is the way back ([installation §2.2](installation.md)). If the folder does not exist, only the API answers and the browser shows no page. | the web service, at start · `<repository>/webui/dist` |
| `EFORECAST_BROWSE_ROOTS` | The folders the file browser of the web pages may show, and a case may point into (its data file, its model folder). A path outside them is refused: `… is outside the allowed folders`. Add a folder when data moves to a new disk. Several are separated by `:`. | the web service · the workspace, `/media` and your home; e.g. `/media/data:/media/models` |
| `EFORECAST_AUTH` | Whether the web pages ask for a login. `on`: accounts, each with their own cases. `off`: no login, and whoever opens the page is the one administrator — for a single-user desktop. | the web service · `on` |
| `EFORECAST_HOST`, `EFORECAST_PORT` | The address and port the web service listens on. `127.0.0.1` answers only this machine, `0.0.0.0` the whole network. | the service unit; the doctor checks that it answers · written by `configure.sh` |
| `EFORECAST_FLEET_METRICS_PORT` | The port of the `eforecast-metrics` service (§2, `metrics`). `0` serves nothing. | the metrics service, `bin/eforecast metrics` · `9109` |
| `EFORECAST_ENV_FILE` | Tells `bin/eforecast` to take its seven settings from another file — a second installation on one machine — or from `/dev/null`, for none. | `bin/eforecast` only · `/etc/eforecast/webui.env` |
| `EFORECAST_SECRETS_DIR` | The folder of the credential files (§2, `secret`). | everything that needs a key · `~/.config/eforecast/secrets` |

### Where records are kept

What must survive a restart is kept in a database, the **run store**: the runs accepted from
the browser and what became of them, which model each case is approved to forecast with, which
forecast is published, the route decisions. Datasets and models are files and are not in it.

| setting | what it is for | takes effect in · default |
| --- | --- | --- |
| `EFORECAST_RUN_STORE` | Which kind of database: `sqlite` — one file, one machine — or `postgres`, a PostgreSQL server, needed only when several machines share the records. | everything that opens the run store (below) · `sqlite`, or `postgres` when a DSN is set |
| `EFORECAST_RUN_STORE_PATH` | The SQLite file. It must be on a local disk: a path inside the workspace, in a synchronised folder (Dropbox, OneDrive …) or on a network mount is refused, because such a file gets damaged. | as above · `~/.local/state/eforecast/runs.sqlite3` |
| `EFORECAST_RUN_STORE_DSN` | The address of the PostgreSQL database, `postgresql://user:password@host:5432/name`. It holds a password: keep it out of case files and scripts, and mind that the settings file itself can be read by every user of the machine. | as above · none |
| `EFORECAST_RUN_STORE_NAMESPACE` | A label written on every row, so that two installations — or a test — can share one database without seeing each other's rows. | as above · `default` |
| `EFORECAST_OPTUNA_STORAGE` | Where the search for hyper-parameters keeps its trials. By default each study has a file of its own beside the models, `<cluster>/<method>/study.journal`. A database address (`postgresql://…/optuna`) is needed when training runs on several machines, or for the optuna-dashboard. The **Studies** page reads the storage the *service* was started with, so studies recorded somewhere else do not show there. `run_training.py --storage` overrides it for one run. | GPU and CPU training, the Studies page · a file per study |
| `EFORECAST_RUN_EXECUTOR_DIR` | Clustering, each CPU study and each prediction run in a process of their own, and each such process gets a folder here with what it was given, its log (`worker.log`) and its result. A path printed as `worker log: …` in an error points into it. Local disk, as for the SQLite file. | the web service, every stage that starts a worker, the doctor · `~/.local/state/eforecast/executors` |

The run store is opened by the web service, by the forecast cycle, by the doctor, and by the
verbs `approved`, `approve`, `rollback`, `forecast-status`, `route` and `metrics`. **All of them
must be given the same four settings.** `bin/eforecast` does *not* take these from the file: if
this installation's file sets any of them, load the file into your shell before those verbs
(`set -a; . /etc/eforecast/webui.env; set +a`). Otherwise they open an empty database of their
own and answer, truthfully about that one, `no approved version` and `none published yet`.

### How runs behave

None of these needs setting for normal work; each exists for one situation.

| setting | what it is for | takes effect in · default |
| --- | --- | --- |
| `EFORECAST_MODEL_CACHE_MODE` | Whether a pretrained network that is not on this machine may be downloaded when a run first needs it. `prefer-local` and `online` download; `offline` and `prewarmed` never do, and the trial that needs it fails saying which model is missing. Every script also takes it as `--model-cache-mode`. | every stage that loads a pretrained network · `prefer-local` |
| `EFORECAST_MODEL_CACHE_DIR` | Where the record of checked pretrained networks is kept; the weights themselves are in the libraries' own caches (`~/.cache/huggingface`). | as above · `~/.cache/eforecast/model-library` |
| `EFORECAST_NO_PROGRESS_SECONDS` | A GPU training run that has nothing running and starts nothing for this many seconds stops with a report instead of waiting all night — something outside it holds the cards. `0` never stops it. | GPU training · `900` |
| `EFORECAST_GPU_BUSY_TARGET` | How busy, in percent, a card may be before it is given no more trials at once. A card takes one more trial of a kind as long as whole trials of that kind kept it under this; the log says what it measured (`… was busy 69% of the time over a whole trial, so a card takes 2`). `0` switches the measuring off: a fixed number per card, as with `--n-jobs`. | GPU training · `80` |
| `EFORECAST_GPU_MPS` | `0` makes the trials on one card take turns instead of computing together through NVIDIA MPS. Small networks then train slower ([GPU training](gpu_training_optuna.md)); it is there to rule MPS out when a card gives trouble. | GPU training · on |
| `EFORECAST_SCALED_INPUTS` | `0` stops a case's scaled weather from being written to, or read from, `<model>/scaled_inputs`; every reader then scales it again from the dataset. The numbers are the same either way — it is for ruling the copy out while looking for a fault, at the cost of time and memory. | datasets, training, combine · on |
| `EFORECAST_INFERENCE_THREADS`, `EFORECAST_INFERENCE_DEVICE` | Threads and device (`cpu`, `auto`, `cuda:N`, a GPU id) of the worker that computes predictions. Unset, the `predict` stage of combine uses the case's CPUs to predict its models side by side, two threads each, and its foundation models on the cards. Set either one and there is a single worker with that many threads on that device, predicting one model at a time. | combine's `predict` stage, the forecast scripts · `2`, `cpu` |
| `EFORECAST_FUZZY_DEVICE` | The device the clusterer is fitted on, when it should differ from what the case says. | the clustering stage · the case's `clustering.params.device`, else `auto` |
| `EFORECAST_REHASH` | `1` makes every check read every file again. Normally a file whose path, size and times have not changed keeps the checksum recorded for it and is not read. For a disk you no longer trust. | every stage, when it checks what earlier stages produced · off |
| `EFORECAST_TRIAL_CMD` | A command that starts a trial some other way than this machine's Python — a wrapper script, a container — with the places `{path_weights}`, `{gpu}` and `{refit}` filled in for each trial. `run_training.py --docker` is the ordinary way to get that. | GPU training · none |
| `EFORECAST_CASE` | The case the scripts use when `--config` is left out. | the `run_*.py` scripts · none |

`EFORECAST_MEM_FRACTION`, which earlier versions of this page listed, is not yours to set: a
training run works it out for each trial and overwrites whatever it finds.

There are no settings for how much of the machine a run may take: nothing is reserved, and the
variables that set such limits in earlier versions (`EFORECAST_SCRATCH_LIMIT_MB`,
`EFORECAST_SCRATCH_MIN_FREE_MB`, `EFORECAST_STAGE_SCRATCH_MB`, `EFORECAST_DISPATCH_…`,
`EFORECAST_HOST_HEADROOM_MB`, `EFORECAST_HOST_DEFAULT_FACTOR`, `EFORECAST_HOST_MAPPED_FACTOR`,
`EFORECAST_DATA_RAM_MB`) are no longer used. They can stay in `/etc/eforecast/webui.env` or be
removed. The Kubernetes venue keeps an allocator and settings of its own; see
[managed runs on Kubernetes](../deploy/kubernetes/MANAGED_RUNS.md).

---

## 5. Two things that will bite you

**A second run of a case is refused after two seconds.** One run writes a case at a time. The
refusal names who holds it — the stage, whose run it is, since when, its pid — and ends with
the one command that stops a run of your own:
`python -m eforecast.config.direct_stage stop-mine <path_model>`. Nothing has to be released
after a run that was stopped: the lock goes with its process. See
[troubleshooting.md](troubleshooting.md#2-a-run-will-not-start).

**`python -m eforecast.config.doctor` is not the same as `bin/eforecast doctor`.** The spelled-out
form does not read `/etc/eforecast/webui.env`, so it reports on your shell, and `--snapshot`
would write the reading where the UI never looks. Use the wrapper.
