# For the administrator

Running the installation: starting it, keeping its images honest, approving models, backing it
up, and deciding where work runs.

People who only run cases need none of this — they use the case editor, Run, Continue, the
Results tab and `forecast-status`, and when something fails they send a support report. Their
guide is [user.md](user.md); this page is what sits underneath it.

**Load the installation's own settings before copying any command here:**

```bash
set -a; . /etc/eforecast/webui.env; set +a
```

`bin/eforecast` does this for itself for seven settings — the interpreter, the workspace, the
release, the backup destination, the browse roots, the metrics port and the login switch — when
your shell has not already set them, and says on stderr which it took. It does not take the
settings that say where the run database is ([cli.md §4](cli.md#4-the-settings-that-matter)).
`python -m eforecast.config.…` spelled out takes **none** — which is why a cron entry should use
`bin/eforecast`.

Inspection commands are read-only. Start/stop, route decisions, image changes, approval, backup
and restore change state, and are marked below.

---

## 1. Accounts

```bash
"$EFORECAST_PYTHON" -m eforecast.config.auth add <name> [--admin]
"$EFORECAST_PYTHON" -m eforecast.config.auth revoke <name>     # end sessions, keep the account
```

Each person owns the cases created under their account (`<workspace>/users/<name>/cases`) and can
mark one as *shared*: others see it read-only and can copy it into their own account. Cases
belonging to a group live under `<workspace>/tenants/<id>/cases`. Case names are unique across
the whole workspace, so a case is addressed by name alone.

Accounts can also be managed from the **Users** page of the UI.

## 2. Starting and stopping

```bash
sudo systemctl start   eforecast-webui          # the UI and the API
sudo systemctl stop    eforecast-webui
sudo systemctl status  eforecast-webui eforecast-metrics
sudo systemctl restart eforecast-webui          # needed after editing the env file
```

Two units are installed by `deploy/configure.sh --write`: the web service, and the metrics
exposition that monitoring scrapes. Enable both.

An `EnvironmentFile` is read when the unit **starts**, not reloaded — so a change to
`/etc/eforecast/webui.env` needs a restart.

### After the code changes

The web service and the metrics unit keep running the code they were started with, and a run
started from the browser is launched from the checkout as it is on disk. A new version is
therefore three things done together, at a moment when no run is going:

1. bring the code in (a pull or a merge);
2. build the web pages from it — `cd webui && npm run build`, or into the directory
   `EFORECAST_WEBUI_DIST` names ([installation §2.2](installation.md)). The pages and the API
   must be of one version: pages of an older version ask the API for things it no longer has;
3. `sudo systemctl restart eforecast-webui eforecast-metrics`.

Then `bin/eforecast doctor`. Where training runs with `--docker`, or forecasts run from the
online image, refresh those images too (§3): they carry a copy of the code, and the release
check refuses one that is stale.

The version that reserves nothing (November 2026) needs nothing beyond those steps. Variables
it no longer reads may stay in `/etc/eforecast/webui.env` or be removed
([the list](cli.md#4-the-settings-that-matter)); tables of the old reservation ledger stay in
the run store and are not read; case files that still name removed settings load as before.

The run coordinator runs beside the API. `bin/eforecast doctor` reports whether it is there and
whether anything holds its lock. Stopping the API is not the same as stopping a run: the
supervisor owns execution, so use the UI's Stop action to cancel a run, and inspect its outcome
and cleanup afterwards. That is what the durable record is for.

## 3. Keeping the images honest  *(changes state)*

A stale image returns a **wrong number rather than an error**, so this gate is not optional:

```bash
bin/eforecast images                     # what differs from the release this lab runs
bin/eforecast images refresh             # rebuild all four, witness, re-capture, check
bin/eforecast images refresh 20261029    # ... as a new dated release
```

It is one command because its steps only make sense together. In order, it:

1. builds every runtime the release declares — the two **model** images at the tags their
   contract names, the **worker** and **online** images at the release's own tag;
2. records each one's witness **under its own entrypoint**, in the file the *service* reads.
   Which file that is depends on `EFORECAST_WORKSPACE`; the verb prints which it used. A witness
   in the wrong file reports a correctly built image as never witnessed;
3. captures the release with the dated worker and online references pinned, so the floating tags
   production may run from are never moved, and the build each dated tag replaces stays
   reachable as `<repo>:<release>-previous`;
4. checks the result and exits non-zero if anything is still stale, so a half-finished refresh
   cannot look finished.

Run it after **any** edit to a file an image carries: anything under `eforecast/`,
`eforecast_online/`, `configuration/`, the run scripts, or a requirements lock. The doctor and
the UI's health chip report the staleness until you do, and routing to a stale image is refused.

`EFORECAST_RELEASE` selects which release record the coherence check uses. Changing that setting
is **not** a deployment or a rollback: confirm the source checkout, the actual worker and online
image references, the runtime registry and the witnesses all match the release you selected.
`bin/eforecast images check <release>` inspects a candidate without changing anything.

Drain accepted work before changing bindings, and preserve its pinned images until cleanup
finishes.

## 4. Offline model preparation

An isolated runtime cannot reach a model hub from inside the lab. Pre-stage the checkpoint into
the mounted cache once, with the network allowed, and the launcher then resolves the tag to the
commit it cached:

```bash
python -c "from eforecast.runtime_launcher import resolve_checkpoint; \
           print(resolve_checkpoint('Chronos2', 'amazon/chronos-2'))"
python -m eforecast.image_release check --release "$EFORECAST_RELEASE"
```

`resolve_checkpoint` refuses a placeholder commit and reads the cached ref before it asks the
hub, so a lab with no network still resolves what it has already staged.

## 5. Approving a model, and rolling it back  *(changes state)*

```bash
bin/eforecast approved <case>                  # what routine forecasts use, and what preceded it
bin/eforecast approve  <case> /candidate/root --note "qualified on the holdout"
bin/eforecast rollback <case>                  # the previous approved version, no retraining
bin/eforecast forecast-status <case>           # what is published, and whether the last try worked
bin/eforecast forecast-status <case> --repair-export /path/to/predictions
```

Approval replays the bundle in a worker of its own **before** the pointer moves, so a bundle whose
weights no longer load cannot become the one people use. Rollback is a pointer move with the
same re-validation; it never retrains.

`<case>` is a workspace case name. A name this installation does not have is refused rather than
created — before that was true, a typo minted a new identity and reported a successful approval
no forecast would ever read. A machine with **no** workspace names its model directory instead
(the folder holding `scalers.pickle` and `Results`), and that resolves to the same case the
service serves from those models.

Capturing the bundle in the first place is in [how_it_works.md §13](how_it_works.md#13-from-a-trained-model-to-a-published-forecast).

## 6. Backup and restore  *(changes state)*

### The destination

It must be a **physically different device**. The doctor refuses one that shares a device with
the source, because another directory on the same disk is not a backup.

```bash
sudoedit /etc/eforecast/webui.env     # EFORECAST_BACKUP_DIR="/media/…/eforecast_backups"
sudo systemctl restart eforecast-webui
```

### The commands

```bash
bin/eforecast backup copy    /path/to/approved/bundle "$EFORECAST_BACKUP_DIR" --case <case> --label weekly
bin/eforecast backup list    "$EFORECAST_BACKUP_DIR" <case>
bin/eforecast backup verify  "$EFORECAST_BACKUP_DIR"/<case>/<generation>
bin/eforecast backup restore "$EFORECAST_BACKUP_DIR"/<case>/<generation> /restore/here
```

`verify` exits 2 when a generation is not whole, and says which file and why — an interrupted
copy is reported as an interrupted copy, never as a backup. A restored bundle is validated by
replay, with the original paths unavailable, before it is trusted.

**The first real approval should be followed immediately by one `copy` and one `verify`** — and
the doctor will say so until it is.

### What it costs

A bundle is small because it is the manifest's *declared components*, not a model folder, so a
copy and verify are seconds. The whole artifact population is a different scale entirely — on
the order of hours for a terabyte-class tree, dominated by sequential throughput rather than file
count. No command here copies the whole tree, and it should not: much of it is datasets and trial
weights that are not guaranteed byte-reproducible but *are* reproducible by retraining. What must
be backed up is what cannot be recomputed: the exact partitions, the fitted scalers, and the
bundle's declared serving dependencies.

Measure your own disks before promising a recovery time. An extrapolation from a small bundle is
not a measured recovery.

### The control state matters as much as the artifacts

The run store holds the approvals, the publication pointers, the holds, the outbox and the route
decisions. Back it up **with** the artifacts, not separately:

```bash
pg_dump "$EFORECAST_RUN_STORE_DSN" > /media/backup/run-store-$(date +%Y%m%d).sql
sqlite3 "$EFORECAST_RUN_STORE_PATH" ".backup '/media/backup/runs.sqlite3'"
```

Those cover the run database only. A complete recovery plan also needs the authority identity,
the artifact references, encrypted secret and configuration backups, and the pinned release and
checkpoint evidence; where the Kubernetes venue is used, its allocator's store as well. A model
backup does not back up these.

**After restoring control state**, keep API submission, the coordinator, broker dispatch and
worker publication isolated until you know that nothing started before the restore is still
running: work that outlived the restore would write into a case the restored state believes is
free. Do not start a restored copy beside the original authority. This combined restore is
**not yet qualified** here — see §10.

On a workstation nothing is reserved for a run, so there is no ledger to restore or to reset. A
run that is gone holds nothing: the lock of a case goes with the process that held it, and its
workers stop with it. A failed PID lookup from another namespace is still not evidence that a
process died.

## 7. Keeping the health reading fresh

The UI's health chip and every new run read **one cached reading**, so a page answers immediately
(about 1.6 s to take a reading, 0.2 ms to read the cached one) and submitting a run never waits
on Docker.

```bash
bin/eforecast doctor --snapshot      # the one thing the doctor ever writes
```

Anyone opening the UI refreshes it when it is older than two minutes, so normally nothing is
needed. Two cases where it is:

* **nobody has opened the UI for a while** — a run accepted then records a stale reading. It is
  labelled stale, so nothing false is claimed, but it explains less. A cron entry keeps it
  current: `*/10 * * * * bin/eforecast doctor --snapshot >/dev/null`;
* **the reading must land where the service reads it** — `--snapshot` keys off
  `EFORECAST_WORKSPACE`, exactly as the image witnesses do. `bin/eforecast` fills that in from
  the env file; `python -m eforecast.config.doctor` does not, and would write to `~/.cache`,
  where the UI never looks.

A colleague's view of that reading is deliberately narrower than yours: the verdict, the findings
addressed to them, and how many are not. This machine's paths, ports and image ids go to
administrators only.

## 8. Credentials

```bash
bin/eforecast secret set <name>      # read from a prompt, never echoed
bin/eforecast secret list            # names only, no values
bin/eforecast secret where <name>    # the file it is read from
```

Keep credentials in the secret source rather than in a shell profile or a committed file. If a
credential has ever been in a shell history, a log, a transcript or a commit, treat it as
exposed and rotate it; removing the file does not un-expose it.

## 9. Where work runs, and changing it  *(changes state)*

Two venues exist. `local` is this machine and is the tested path. `broker` is an opt-in
PostgreSQL + RabbitMQ + Kubernetes route with isolated qualification; enabling it for a case
still requires that case's own runtime, storage and recovery gates. What that route is, the
conditions under which a run takes it, and what then happens to the run step by step are in
[deploy/kubernetes/README.md](../deploy/kubernetes/README.md).

Before deciding anything, try it: `deploy/kubernetes/local_cluster.sh` makes a one-node cluster
with its own database and broker on this machine and runs a step of a real case on it, without
reading or changing this installation's service, run database or routes
([the walkthrough](../deploy/kubernetes/README.md#try-it-on-one-workstation-start-to-finish)).
It answers the question a cutover depends on — can this case run in a Pod at all — in an hour.

The venue is a **decision**, not an environment variable:

```bash
bin/eforecast route                                   # what this installation does today
bin/eforecast route <case>                            # one case, and what is outstanding
bin/eforecast route <case> broker --note "why"        # decide it
bin/eforecast route <case> local  --note "rolled back"
bin/eforecast route <case> --history                  # every decision, newest first
```

Four properties are what make a cutover reversible:

* **the default is `local`.** A broker URL alone changes nothing;
* **a run keeps the venue it was accepted under.** Deciding or rolling back mid-run does not
  retarget work in flight, and a finished run never looks as though it ran elsewhere;
* **a change is refused while work is outstanding** for that scope (exit 3), naming the dispatch
  to look at. Two venues must never claim one attempt, and a message must not be left in a queue
  nobody is listening on;
* **every decision is recorded** with who, when and why. A rollback is another decision, not an
  erasure.

The broker itself:

```bash
docker compose -f docker/compose.yaml --env-file docker/.env up -d rabbitmq
bin/eforecast secret set broker_url          # amqp://<user>:<password>@127.0.0.1:5672/
```

Set `RABBITMQ_USER` and `RABBITMQ_PASSWORD` in `docker/.env` first. The compose service
deliberately does not use `guest`, which RabbitMQ refuses through a published port -- but its
*fallback* is a literal in `docker/compose.yaml`, which is pushed, so a published port with
that password is no better than a refused one. Until 2026-11-04 `docker/.env.example` did not
even list these five keys, so a machine configured from the template ran the broker on that
fallback. If this installation has ever started the broker without them set, change them and
restart it.

The drill before any cutover is `docker/smoke_dispatch.py`, run against a **disposable**
PostgreSQL and RabbitMQ on private ports and the release's own worker image. It kills the
publisher and consumer from inside at each boundary, so it is deterministic about *where* it
fails. Never run it against production: it creates and deletes dispatch rows in the namespace it
is given.

When you do cut over: one case, in a window, with the dashboards open, and a week before a
second follows. The one thing not to do is cut over every case at once.

## 10. Topology, and what is not qualified

One control service: one API/coordinator pair, one run store. **Several workers do not make this
highly available** — the API and the store are each a single point of failure, and the planned
disaster-recovery path is the combined restore in §6. A run in flight survives a control-service
restart, because the record is durable and the supervisor owns the process; it does not survive
losing the store without the backup. Automatic failover is not supported.

Not qualified, stated plainly so nobody plans around it:

* **the production broker/Kubernetes cutover.** The opt-in CPU path passed an isolated drill; the
  lab deployment has not switched;
* **more than one node, and sharing one GPU card.** A single node has been measured; a
  disposable single-node cluster has run two GPU fits on distinct physical cards with
  CPU parity and an exact checkpoint reload. Sharing a card, and a second node, are unmeasured;
* **GPU placement and failed-node behaviour** on the managed route;
* **deployed collectors and dashboards** against production bindings;
* **a backup schedule, a retention policy and a named owner**, and a measured recovery time for
  the real artifact population;
* **a combined restore of the control store and the Kubernetes venue's allocator**, with whatever survived the restore accounted for;
* **the cloud.** A bounded single-instance CPU pilot ran once and was terminated and verified
  removed the same day. Managed cloud Kubernetes and cloud GPUs remain unqualified, and that
  pilot used synthetic cases, so weather ingestion and a real case end to end are untested there.
  It is started only by hand, never by the installation: [deploy/aws/README.md](../deploy/aws/README.md).

Nothing above claims a venue is ready because its parts passed separately.

## 11. Where to look when something is wrong

| symptom | where to look |
| --- | --- |
| anything at all, first | the **health chip** in the UI, or `bin/eforecast doctor` |
| a colleague's run failed | their **support report** (log drawer → Copy/Download). It carries the case revision, the run, the failing stage, the image, the dispatch reference, the doctor's verdict, and what the installation looked like **when that run was accepted** |
| the forecast looks old | `bin/eforecast forecast-status <case>` |
| `latest.csv` is stale but the pointer moved | `forecast-status <case> --repair-export <folder>` |
| work is recorded but nothing runs | `bin/eforecast doctor` → the route and dispatch findings. A backlog nobody drains says so |
| a model will not serve | the image-coherence finding, and `bin/eforecast approved <case>` |
| a worker died | its run says why and names the worker's log (`failure: …`). Nothing is left to release — see [troubleshooting.md](troubleshooting.md#a-worker-failed) |

The full symptom list is [troubleshooting.md](troubleshooting.md).
