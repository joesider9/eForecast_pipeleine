# deploy/kubernetes/ — executing a case's runs on a cluster (the `broker` venue)

Three readers, three places to start:

| you want to | read |
| --- | --- |
| **see a run execute on a cluster**, on this machine, without changing the installation | [Try it on one workstation, start to finish](#try-it-on-one-workstation-start-to-finish): six commands |
| **run your case on the cluster** the installation is connected to | [user.md, "Running a case on the cluster"](../../docs/user.md#running-a-case-on-the-cluster). You keep using the same web pages and never need `kubectl` |
| **connect the installation** to a cluster, so that colleagues can do that | [Connecting the installation, in order](#connecting-the-installation-in-order), then [MANAGED_RUNS.md](MANAGED_RUNS.md) |

## In short

**What it is.** A run started from the web pages normally executes on the machine that serves
the pages — the `local` venue. This directory is the other possibility, the `broker` venue: the
same run is executed inside a container on a Kubernetes cluster. The pages, the case and the
results are the same; only *where the script runs* changes. It exists so that runs can, one
day, use machines other than the workstation.

**When it is executed.** Never by itself. A run goes to the cluster only when **all** of these
are true:

1. it was started from the web pages (or their API). A script typed in a terminal always runs
   where you type it and never looks at the route;
2. somebody decided it for that case, or for the whole installation:
   `bin/eforecast route <case> broker --note "why"`;
3. the installation is connected to a cluster: the setting `EFORECAST_KUBE_RUNS_FILE` names a
   valid **target file** (which cluster, which image, how much of it), the run database is
   PostgreSQL, and the run coordinator has the cluster's address, a token and the broker's
   address.

In every other case — the default, and this lab today — a run executes locally and neither the
broker nor a cluster is contacted. There is no automatic switching in either direction: a run
keeps the venue it was accepted under, and going back is another explicit decision,
`bin/eforecast route <case> local --note "why"`.

**Where it stands.** It has been tried in isolation — one node, CPU runs, and two fits each on
a whole GPU, on disposable databases. The lab itself has never been switched to it. This
directory holds reusable definitions and one script, [local_cluster.sh](local_cluster.sh), that
builds such an isolated cluster on a workstation; it is not a record of an installed cluster.

## The pieces

| piece | what it is | where it lives |
| --- | --- | --- |
| the **route** | the recorded decision `local` or `broker`, per case or for the installation, with who, when and why | the run database; `bin/eforecast route` |
| the **outbox** | the list of accepted runs that still have to be handed to the cluster | the run database |
| the **broker** | RabbitMQ, a message queue. It carries a short note — "run X is waiting", with identities and limits — never data, commands or passwords | a container: `docker compose -f docker/compose.yaml … up -d rabbitmq` |
| the **run coordinator** | a program on the host, beside the web service. It hands runs over, creates the cluster's Jobs, watches them and tidies up | `python -m eforecast.config.run_coordinator`; the web service starts it by itself |
| the **target file** | JSON: the cluster's address and namespace, the image, CPU / memory / GPU of one Job, two time limits, the shared folder ([example](managed-run.example.json)) | wherever `EFORECAST_KUBE_RUNS_FILE` points, outside the repository |
| the **Job** and its **Pod** | Kubernetes's way of saying "run this container once" | on the cluster, namespace `eforecast` |
| the **worker image** | a Docker image carrying a copy of the pipeline's code | built from `docker/Dockerfile.worker`, named by its digest |
| the **shared folder** | one folder present under the *same path* on the host and inside the Pod. The workspace, the run's state, the data, the weather archive and the models must all lie under it | the target file's `shared_root`; a host folder on a one-node cluster |

## What happens to one run, step by step

1. **Run is pressed.** The API accepts the run and, in one database transaction, writes on it
   the venue (`broker`) together with a copy of the target file as it is at that moment, and
   adds a row to the outbox. From here on, editing the target file changes nothing for this
   run.
2. **The coordinator hands it over.** About once a second it publishes the notes of waiting
   outbox rows to RabbitMQ, and takes notes back off the queue. Publishing and taking are the
   same program; the queue between them is what lets either side be restarted without losing
   or doubling a run.
3. **A Job is created.** For a note it took, the coordinator checks again — the run was not
   cancelled, its inputs did not change, fewer than `max_jobs` Jobs are unfinished — records
   what it is about to do, and asks the cluster for one Job: one container from the worker
   image, with exactly the CPU, memory and GPU the target file states, the shared folder
   mounted, and a time limit (`active_deadline`). The Job's name is derived from the run, so
   asking twice cannot create two.
4. **The Pod runs the script.** Its command is
   `python -m eforecast.config.kubernetes_run_agent <run id>`. That program checks that the
   code in the image is, byte for byte, the code the run was accepted with; puts the case's
   files back exactly as they were when Run was pressed; and starts the same script a local
   run would — `run_datasets.py`, `run_training.py` or `run_combine.py` — with its whole
   sequence of stages. One Job is one run; the stages are not Jobs of their own.
5. **You watch it as usual.** Progress goes to the run database and the log to the shared
   folder, so the **Runs** page shows the run like a local one. What the cluster is waiting
   for — no node free, the image not pulled — appears as the run's wait reason.
6. **It ends.** The Pod exits; the coordinator reads how, records the outcome and deletes the
   Job. A Pod that ended without a recorded outcome counts as a failure: a Job that "succeeded"
   is only evidence that a process exited.
7. **Stop** records a cancellation. A run not yet handed over is cancelled at once; one whose
   Job has no Pod yet has the Job deleted; a running one is told to stop its processes, and the
   time limit ends it if it cannot.
8. **Nothing is retried by itself**, and a second Pod can never take over a run. If the API or
   the coordinator restarts in the middle, the Pod goes on, and the new coordinator finds the
   run again in the database.

Not available on this venue: training with `--docker`, and model families that run in a
container of their own — a Job cannot start Docker containers inside itself. Such a run is
refused when it is submitted.

## Try it on one workstation, start to finish

[local_cluster.sh](local_cluster.sh) makes a whole `broker` venue out of three containers — one
Kubernetes node (k3s), PostgreSQL and RabbitMQ — and runs the API and the coordinator against
them. It is the way to watch a run go through every step above, and to find out whether a case
*can* run in a Pod, before anybody decides anything for the installation.

**It changes nothing of the installation.** The web service, its run database and its routes are
neither read nor written; the cluster has a run database of its own. What it does share is the
files: a case run on it reads and writes that case's real datasets and models. So
[one at a time](#what-the-trial-cluster-cannot-do): never start a case here while the pages are
running it.

**It needs** Docker that you may use without `sudo`; about 20 GB in the cluster's folder (the
worker image is copied into the node); the interpreter the web service runs, because the API's
packages are borrowed from it (`fastapi` and `pika`, both in `requirements.txt`); and, the first
time, three images that `up` downloads if they are not there — k3s (about 80 MB), `postgres:16`
and `rabbitmq:4.1-management-alpine`.

### 1. Create the cluster — once

```bash
export EFORECAST_CLUSTER_HOME=/media/sider/data/eforecast-local-cluster
deploy/kubernetes/local_cluster.sh up --shared /media/sider/data \
    --workspace /media/sider/data/eforecast_workspace
```

`--shared` is the important choice. It is the **one folder of this machine a Pod will see**,
mounted under the same path, so that a path means the same file on the host and in the Pod.
Everything a run needs must lie under it: the workspace, the series, the weather archive, the
model folders. `--workspace` is the workspace whose cases you want to run (the installation's
own is taken when you leave it out); `EFORECAST_CLUSTER_HOME` is a new folder for the cluster's
own files, under `--shared`, and every later command finds the cluster through it.

With the three images on the machine it takes a quarter of a minute, and ends with

```
the cluster eforecast-local-cluster is up: 12 CPUs, 31g, API on 127.0.0.1:56444, no GPU
next:
  export EFORECAST_CLUSTER_HOME=/media/sider/data/eforecast-local-cluster
  deploy/kubernetes/local_cluster.sh image
```

The node is given half of the machine unless you say otherwise (`--cpus`, `--memory`), and one
Job most of the node (`--job-cpus`, `--job-memory-mb`, and `--job-hours`, 6 by default — the
time after which the cluster ends a Job that is still running). Two clusters on one machine
need different `--name`, `--port` and `--subnet`.

### 2. Give it the code — again after every change

```bash
deploy/kubernetes/local_cluster.sh image
```

builds the worker image from this checkout, checks that the code inside it is byte for byte
the code of the checkout, copies it into the node (the image is about 16 GB: a minute from one
fast disk to another, several otherwise) and writes the two target files — one for the drill,
one for the workspace. With nothing changed it finds the node up to date and takes two seconds.

**Run it again whenever a file under `eforecast/` changed.** A Pod compares the code it carries
with the code the run was accepted with and refuses to start otherwise. `status` tells you:

```
$ deploy/kubernetes/local_cluster.sh status
cluster    eforecast-local-cluster in /media/sider/data/eforecast-local-cluster
shared     /media/sider/data   (what a Pod sees of this machine)
workspace  /media/sider/data/eforecast_workspace
node       running   (container eforecast-local-cluster)
database   running   (container eforecast-local-cluster-pg)
broker     running   (container eforecast-local-cluster-rabbit)
kubernetes Ready, v1.34.1+k3s1; 12 CPUs, 31g, no GPU; API https://127.0.0.1:56444
jobs       0 in the namespace eforecast
token      valid until 2026-11-20 18:19   ('deploy/kubernetes/local_cluster.sh token' issues another)
image      eforecast-worker:eforecast-local-cluster carries the code of this checkout
```

The last line is the one to look at before a run. When it says `carries OTHER code than this
checkout`, run `image`.

### 3. The drill — does the venue work here?

```bash
deploy/kubernetes/local_cluster.sh drill
```

submits a synthetic case three times through the same HTTP endpoint the Runs tab posts to, and
drives the coordinator: a run whose note is delivered twice and whose coordinator restarts, a
run that is cancelled with a child process alive, and a real dataset stage. It prints every
state the runs pass through and ends, after about a minute, with

```json
{
  "passed": [
    {"scenario": "claim/restart/redelivery", "outcome": "success",   "cleanup": "complete"},
    {"scenario": "cancel-owned-descendants", "outcome": "cancelled", "cleanup": "complete"},
    {"scenario": "real-datasets-stage",      "outcome": "success",   "cleanup": "complete", "stages": 6}
  ]
}
```

Anything else is a finding about this machine, and the lines before it say which step.

### 4. A step of your own case

First the plan, which changes nothing:

```bash
deploy/kubernetes/local_cluster.sh run AG_XRIST_LKDM_1 datasets --options '{"dry_run": true}'
```

Then the step:

```bash
deploy/kubernetes/local_cluster.sh run AG_XRIST_LKDM_1 datasets --options '{"stages": ["all"]}'
```

```
accepted as run 20261117-161804-b7abf887… for the cluster https://127.0.0.1:56444, image 598772cdff4c81a3692
16:18:05  pending    job not created  cleanup not_required
16:18:05  pending    job planned      cleanup not_required   (Kubernetes: Pod is pending and has not started a container yet)
16:18:06  preparing  job running      cleanup not_required   (Kubernetes: Pod is pending and has not started a container yet)
16:18:07  queued     job running      cleanup not_required
16:18:08  running    job running      cleanup not_required
16:18:10  terminal   job running      cleanup complete
16:18:13  terminal   job complete     cleanup complete
run 20261117-161804-b7abf887…: success -- datasets
  extract        success
  nwp            success
  images         nothing_to_do
  target         success
  row            success
  lstm           success
  split          success
  preprocess     success
  clustering     success
  features       nothing_to_do
  log: /media/sider/data/eforecast_workspace/users/george/cases/AG_XRIST_LKDM_1/runs/20261117-161804-b7abf887….log
  cleanup on the cluster: complete
```

Each line is a state the run passed through: the run's own (`pending`, `preparing`, `queued`,
`running`, `terminal`), the Job's on the cluster, the cleanup, and in brackets what the cluster
is waiting for. The times are the container's clock (UTC). This case's datasets were built
already, so every stage found its work whole and reused it — eleven seconds. What the stages
*decided* is in the log the last lines name, exactly as for a run started on the page.

The case is named as the Cases page lists it. `--options` are the options the Runs tab sends:

| step | options |
| --- | --- |
| `datasets` | `{"stages": ["all"]}`, or some of them; `"recreate": true`; `"dry_run": true` |
| `training` | `{"cpu": true}` for the families that train on the CPU, with `"cpu_method"`, `"cpu_scope"`; `"dry_run": true`. Deep families need a card, which this cluster does not have |
| `combine` | `{"stages": ["all"]}`, or some of `predict`, `select`, `combine`, `calibrate`, `evaluate`; `"dry_run": true` |

The command waits for the run and is its coordinator meanwhile; its exit status is 0 for a run
that ended well. **Ctrl-C stops the run**, the way the page's Stop does, and waits until the
cluster has let go of it:

```
stopping the run, as the Stop button does; waiting for the cluster to let go of it (Ctrl-C again to leave it)
16:20:40  pending    job planned      cleanup pending   (Kubernetes: Pod is pending and has not started a container yet)
16:20:42  terminal   job complete     cleanup complete
run 20261117-162039-0d7b48f0…: cancelled -- assigned Pod 438c8b20-… terminated before managed cleanup
```

A step that would replace what another configuration built prints what it replaces and goes
on, as every run typed in a terminal does ([cli.md](../../docs/cli.md)); nothing is asked.

While it runs, in another terminal:

```bash
deploy/kubernetes/local_cluster.sh kubectl -n eforecast get jobs,pods
deploy/kubernetes/local_cluster.sh kubectl -n eforecast logs job/<name> --follow
```

### 5. Put it away

```bash
deploy/kubernetes/local_cluster.sh stop      # the containers stop; everything is kept
deploy/kubernetes/local_cluster.sh start     # ... and come back, with a fresh token
deploy/kubernetes/local_cluster.sh down eforecast-local-cluster    # gone: containers, network, databases, folder
```

`down` wants the cluster's name because it cannot be undone. A stopped cluster costs disk only.
After a reboot of the machine the containers are stopped; `start` is all it takes.

### What is where

Everything of the cluster is in its folder (`EFORECAST_CLUSTER_HOME`) and in three containers
named after it.

| in the folder | what it is |
| --- | --- |
| `cluster.env` | the choices made at `up`; every command reads it. Nothing secret |
| `private/` | readable by you alone: the passwords of the cluster's own database and broker, the cluster's certificate, the coordinator's token, the settings of the control plane (`control.env`) and the two target files (`target-drill.json`, `target-workspace.json`) |
| `integration/` | the drill's own little workspace and what its synthetic cases leave. The drill refuses a folder that is not named so: it is not to be pointed at a workspace of real cases |
| `state/` | what the Pods of your own cases write beside their work: the run's inputs as they were accepted, and its reports |
| `k3s/`, `kubelet/` | the node: Kubernetes's own state and the images copied into it |
| `pgdata/`, `rabbitdata/` | the cluster's run database and its broker's queues |

The target files are ordinary [target files](#the-settings-involved): to give a Job other
limits, change `JOB_CPUS`, `JOB_RAM_MB` or `JOB_SECONDS` in `cluster.env` and run `image`, which
writes them again.

### What the trial cluster cannot do

| | |
| --- | --- |
| **no GPU** | the node advertises no card, so nothing deep can be *fitted* on it: a training step with a trial left to train, a transfer to train or a `--refit` stops, before it has touched anything, with `GPU ids [0] are unavailable; CUDA sees 0 device(s)`. A case whose deep training is finished passes the step — the stage is found whole and reused, and a stage that fits nothing asks for no card. Datasets, CPU training and the combine stages run |
| **only what the worker image can import** | a case whose families are served by a container of their own on the host (TiRex, TimesFM) cannot train or combine them in a Pod. The image carries Chronos |
| **one at a time with the pages** | the installation's service and this cluster do not know of each other's runs. Do not start a case here while it runs there, or the other way round |
| **its runs are not on the Runs page** | they are in the cluster's own run database. What a run *made* — datasets, trials, results — is in the case's folders as always, and its log is where the last lines of `run` say |
| **the code must match** | after any change under `eforecast/`: `image`. A run accepted with other code than the image carries ends at once with `worker image Python source differs from the accepted source` |
| **the token expires** | after 72 hours (`--token-hours`). `start` and `token` issue a new one; a coordinator with an expired token cannot create Jobs |
| **paths outside `--shared`** | a case that keeps its series, weather or models outside the shared folder is refused by `run`, which names the paths |

### If a step does not work

| what you see | what it is |
| --- | --- |
| `the cluster … is not running` | `start` |
| `… has no fastapi` / `… has no pika` | the interpreter named at `up --python` (or `EFORECAST_PYTHON`) lacks a package the API needs; install it there, from `requirements.txt` |
| the run stays `pending`, `job planned`, with `Pod is pending` for minutes | the node has no room for the Job — `status` shows its CPUs and memory, the target file under `private/` what one Job asks — or the image is not in the node: `image` |
| `worker image Python source differs …` | the code changed after `image`: run `image` again |
| a stage fails with a missing file or `ModuleNotFoundError` | a path outside `--shared`, or a family the image cannot import: see the table above |
| `training: SystemExit: GPU ids [0] are unavailable; CUDA sees 0 device(s)` | the deep-training step has something to fit and this cluster has no card: see the table above. Train it on the workstation; `--options '{"cpu": true}'` is the CPU families' step |
| `403`, `Unauthorized` in the coordinator's lines | the token expired: `token` |
| it is all in a state you cannot read | `down <name>` and `up` again: nothing of the installation is lost with it |

## Connecting the installation, in order

This is the other thing: making the *installation's own* runs go to a cluster, so that a
colleague presses Run as always and the work executes there. Everything here changes state
somewhere; do it on a cluster and databases meant for it, and only after the
[trial](#try-it-on-one-workstation-start-to-finish) has shown that the cases in question can run
in a Pod. The details and the guarantees are in [MANAGED_RUNS.md](MANAGED_RUNS.md).
`local_cluster.sh` is a readable example of every step below, done for a cluster of its own.

1. **The run database must be PostgreSQL** (`EFORECAST_RUN_STORE_DSN`), and RabbitMQ needs a
   virtual host of its own for this database. The interpreter that runs the web service needs
   the driver and the broker client, both in `requirements.txt`: `psycopg2-binary` and `pika`
   (`bin/eforecast doctor` says when one is missing).
2. **Build the worker image** and note its registry digest (`…@sha256:…`). Rebuild it whenever
   the code changes: step 4 above refuses an image whose code differs from the run's.
3. **Create the cluster's side.** Check which cluster you are talking to first.

   ```bash
   kubectl config current-context
   kubectl apply -f deploy/kubernetes/00-namespace.yaml   # the namespace `eforecast`
   kubectl apply -f deploy/kubernetes/10-rbac.yaml        # two accounts: the executor may manage Jobs
                                                          # and Pods there; the worker may do nothing
   ```

   `20-storage.yaml` describes two shared volumes for a cluster of several nodes. It is a
   template with a placeholder and is **not** part of the tested one-node setup, which mounts a
   host folder; do not apply it unchanged.
4. **Give the Pods the database address.** A Kubernetes Secret named `managed-connections`,
   made from a protected file ([the command](MANAGED_RUNS.md#apply-the-settings-and-inspect-before-submitting)).
   The address never goes into the target file, the broker's notes, the image or a case.
5. **Start the broker** and store its address:

   ```bash
   docker compose -f docker/compose.yaml --env-file docker/.env up -d rabbitmq
   bin/eforecast secret set broker_url        # amqp://<user>:<password>@127.0.0.1:5672/
   ```

   Set `RABBITMQ_USER` and `RABBITMQ_PASSWORD` in `docker/.env` first.
6. **Give the coordinator the cluster.** A token of the executor account —
   `bin/eforecast secret set kube_token`, or a file named by `EFORECAST_KUBE_TOKEN_FILE` — and,
   in the settings file, `EFORECAST_KUBE_SERVER`, `EFORECAST_KUBE_NAMESPACE` and
   `EFORECAST_KUBE_CA`. The coordinator talks to the cluster with these three values and the
   token; it does not use a kubeconfig file.
7. **Write the target file** from [the example](managed-run.example.json), outside the
   workspace and the repository, and name it in `EFORECAST_KUBE_RUNS_FILE` for the web service.
   Restart the service. Its 30-minute `active_deadline` is far too short for a training run.
8. **Check, then decide.**

   ```bash
   bin/eforecast doctor
   bin/eforecast route <case>                              # what it would do, and what is outstanding
   bin/eforecast route <case> broker --note "first case"   # the decision
   ```

   `bin/eforecast` does not take the run-database and cluster settings from the settings file
   ([cli.md §4](../../docs/cli.md#4-the-settings-that-matter)): load the file into your shell
   first, or `route … broker` answers that the venue is not connected.

Then press Run for that case as always. One case, with the run's page open; a second case only
after the first has behaved for a while. What the colleague sees and does from here on is in
[user.md, "Running a case on the cluster"](../../docs/user.md#running-a-case-on-the-cluster).

## The settings involved

| setting | what it is for |
| --- | --- |
| `EFORECAST_KUBE_RUNS_FILE` | the path of the target file. Unset — the default — the `broker` venue does not exist for this installation |
| `EFORECAST_KUBE_SERVER`, `EFORECAST_KUBE_NAMESPACE`, `EFORECAST_KUBE_CA` | the cluster's address, the namespace and the certificate to trust; they must agree with the target file |
| `EFORECAST_KUBE_TOKEN_FILE`, or the secret `kube_token` | the executor account's token |
| `EFORECAST_BROKER_URL`, or the secret `broker_url` | RabbitMQ's address |
| `EFORECAST_RUN_STORE_DSN` | the PostgreSQL run database, required on this venue |
| `EFORECAST_RUN_COORDINATOR` | `auto` (default): the web service starts the coordinator; `external`: you run `python -m eforecast.config.run_coordinator` as a service of its own |

The target file's own fields: `image`, `server`, `namespace`, `shared_root`, `workspace`,
`state_root` (outside the workspace), `source_root` and `python` (where the code and the
interpreter are inside the image), `service_account`, `connection_secret`, `resources`
(`cpu_threads`, `ram_mb`, `gpu_slots` 0 or 1), `node_selector`, `pending_deadline` (seconds a
Job may wait for a node), `active_deadline` (seconds it may run), `max_jobs` (at most 32) and
`volume`.

## What has been tried, and what has not

| tried, in isolation | not tried — do not plan on it |
| --- | --- |
| one node; CPU runs: submit, submit again, restart of the coordinator, the same note delivered twice, cancel before and after start, a killed worker, a real dataset stage | the lab itself switched to this venue |
| two fits, each on a whole GPU, on a one-node cluster with two cards | more than one node; storage shared between nodes; two Jobs sharing one GPU |
| a reboot of the host between accepting a run and executing it (on the rented AWS machine, [../aws/](../aws/README.md)) | a GPU machine at AWS, AWS's managed Kubernetes |
| | restoring the run database and the cluster's state together after a loss |

The drill was repeated on 17 November 2026 on the lab workstation itself, still in isolation: a
one-node k3s in a container, throwaway PostgreSQL and RabbitMQ containers on a network of their
own, and the API and coordinator run from the worker image against those — the lab's own
service, run database and route were not touched. That setup is now
[local_cluster.sh](local_cluster.sh). Submission and redelivery, cancellation and a
real dataset stage passed, the last one only after a fix: a stage records who holds the case,
and a Pod runs under a uid its image has no name for, so every stage ended with
`KeyError: 'getpwuid(): uid not found'` before it had read anything
(`direct_stage.whoami`; the Job is now told whose run it is, `EFORECAST_USER`). The datasets
step of a real case was then run as a Job with the same script (`run <case> datasets`).

Two things a case needs before its whole suite can run here, both seen on a real case that day:
its foundation families must be importable in the worker image (the image carries Chronos but
not TiRex or TimesFM, and a Job cannot start the Docker runtimes that serve them on the host),
and deep training needs a card advertised to the cluster — for a case that has anything left
to train. A case whose training is finished ran its whole suite here as four Jobs that evening
(datasets, training, CPU training, combine), each stage found whole and reused: the training
step used to ask for its card before it looked at what was left to do, and no longer does.

The local checks are scripts under `docker/` (`smoke_managed_kubernetes.py`,
`smoke_managed_restart.py`), run against disposable databases and a disposable cluster
([MANAGED_RUNS.md](MANAGED_RUNS.md#repeat-the-isolated-drill)). Two of their options — the
fitting check `--compute cpu|gpu` and `smoke_managed_deadlines.py` — have a Job run a helper
script (`docker/managed_compute_probe.py`, `docker/managed_deadline_probe.py`), which the drill
copies into the shared folder so that no image has to carry it. On the cluster of one node
of [the section above](#try-it-on-one-workstation-start-to-finish), the fitting check is
`deploy/kubernetes/local_cluster.sh drill --compute cpu`.

## Preparing a cluster: what to record first

Before installing anything, write down the machine(s), the Kubernetes distribution and its
version, the GPU driver and device plugin, the worker image's digest, the shared paths, the
user and group ids, and the namespace — and confirm that the work will not replace the host's
Docker configuration or interrupt the lab.

| item | what must be true |
| --- | --- |
| Kubernetes | a known version, an explicit context, and a way to remove it again |
| GPUs | whole cards advertised as `nvidia.com/gpu`, without changing the host's default Docker runtime. Check with `kubectl get node NODE -o jsonpath='{.status.capacity.nvidia\.com/gpu}{"\n"}'`: the number is that node's cards |
| image | the full registry digest, present on the node, built from the code the installation runs |
| storage | every path a run needs reachable under the shared folder, with the same user and group ids on the host and in the Pod (`id -u`, `id -g`). A folder of one node is not shared storage: before using a volume across nodes, prove writing on one and reading on another |
| databases | PostgreSQL and a RabbitMQ virtual host of their own for a test — never copies of the lab's |
| the executor's rights | `kubectl auth can-i create jobs.batch --namespace eforecast` and `… patch pods …`, asked **as the executor account**, answer `yes`; asking as the cluster's administrator tests the wrong identity |

## Retiring a node without losing track of its work

1. Stop sending new work to it (`bin/eforecast route <case> local …`); keep a coordinator
   running for the runs already accepted.
2. Use the page's Stop for work that cannot finish, and wait until each run shows its outcome
   **and** its cleanup.
3. Look at Jobs and Pods by their recorded names. Do not delete every Job by a broad label, and
   do not strip a Pod's `eforecast.io/termination-evidence` mark to empty the queue: it is what
   keeps the record of how the Pod ended until the coordinator has stored it.
4. Only then use the distribution's own drain or uninstall procedure, and check the host's
   networking and Docker services afterwards.

If the coordinator cannot reach the cluster or the database, restore that and let it finish;
see [MANAGED_RUNS.md, "Diagnose and recover"](MANAGED_RUNS.md#diagnose-and-recover).
