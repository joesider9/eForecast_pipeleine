# Installing on a new machine

Two different things are called "installing", and almost everybody wants the first one.

* **I want to use the lab from my own computer** → [§1](#1-using-the-lab-from-your-own-computer).
  Nothing to install.
* **I want this machine to run cases by itself** → [§2](#2-a-machine-that-runs-cases-by-itself).

Read one section, not both.

---

## 1. Using the lab from your own computer

No Python, no CUDA, no GPU, no copy of this repository.

1. Ask the administrator for the lab address and an account. It looks like
   `http://<workstation>:8000`.
2. Open it in a browser and sign in.

That is the whole setup. Cases you create are yours; runs you start execute on the lab machine,
using its GPUs and its data, not your computer's.

If the page does not load, one command says whether the problem is your machine or the lab. Run
it from anywhere the repository is available, or ask the administrator to run it:

```bash
bin/eforecast doctor --client http://<workstation>:8000
```

It checks only what the browser route depends on, and says so. A clean result means the lab is
reachable from here. It deliberately says **nothing** about whether your computer could run a
case by itself, because for this role it does not need to.

---

## 2. A machine that runs cases by itself

This needs a supported Python, a writable workspace, and access to the data the cases declare.

### 2.1 Get the code and an environment

```bash
git clone <repository-url> ProjectPipelineTest
cd ProjectPipelineTest
python3 -m venv ~/eforecast_env
~/eforecast_env/bin/python -m pip install -r requirements.txt
~/eforecast_env/bin/python -m pip check
mkdir -p ~/eforecast_workspace
```

Three things to know about that `requirements.txt`:

* **every line states a version**, and that is deliberate. It reproduces the interpreter this
  installation actually runs. Before 2026-11-04 almost every line was a bare name, and one pip
  run silently downgraded scikit-learn below the range in which the retained models had ever
  been read, and moved pandas to a release that had removed a keyword the dataset reader used.
  Prefer changing one line and running the suite over re-freezing the file;
* two lines are ranges rather than pins — `scikit-learn` because its window was *measured*
  against real retained estimators, and `pandas` because the host and the training environment
  run different 3.x releases and the reader is verified on both. The header explains both;
* the *worker* requirements alone are not enough for a native install — their Torch comes from
  the Docker base image, and they do not install the UI server. The foundation model packages
  are in none of these files by design: they live in `docker/requirements-foundation.txt`,
  because the host carries every runtime adapter and no model package.

Never upgrade a shared environment that something is running in, and resolve anything
`pip check` reports before going on.

A successful `pip install` is not qualification that this machine can run a case. That is what
§2.6 is for.

### 2.2 Build the web pages

```bash
export EFORECAST_WEBUI_DIST="$HOME/.local/share/eforecast/ui-$(date +%Y%m%dT%H%M%S)"
(cd webui && npm ci && npm run build -- --outDir "$EFORECAST_WEBUI_DIST" --emptyOutDir)
```

Build into a **new** directory each time and keep the previous one: that is your rollback. Never
rebuild into the directory currently being served to people.

### 2.3 Let the machine describe itself

Nothing in `deploy/` is edited by hand. The script detects the user, the checkout, the
interpreter, the workspace and the disks to wait for, and writes them to
`/etc/eforecast/webui.env` and to a systemd drop-in:

```bash
bash deploy/configure.sh             # show what it found; writes nothing
bash deploy/configure.sh --write     # install the units and the environment file
```

**Read the printout before `--write`.** Anything it got wrong can be given to it instead:

```bash
EFORECAST_WORKSPACE=/srv/eforecast EFORECAST_PORT=8080 bash deploy/configure.sh --write
```

The interpreter it picks is one that can import `fastapi` and `uvicorn`, preferably also `torch`.
That test does not qualify GRIB reading, foundation models, or every dependency a particular
case needs. Pin it explicitly if you care which one it is:
`EFORECAST_PYTHON=/absolute/path/to/python`.

Two things the script cannot know, which you add by hand once:

```bash
sudoedit /etc/eforecast/webui.env
#   EFORECAST_WEBUI_DIST="/absolute/path/to/the/completed/ui-release"   from §2.2
#   EFORECAST_BACKUP_DIR="/path/on/a/physically/different/disk"         see administrator.md
```

Re-running `--write` later is safe: it keeps the workspace the installation already has, the
release it already chose, and every `EFORECAST_*` setting in the file that it does not generate
itself. The preview lists those under *kept from the env file* — check that line.

The file is a plain `KEY=value` list, so a terminal can use the same settings:

```bash
set -a; . /etc/eforecast/webui.env; set +a
```

### 2.4 Create the first account

The first one is the administrator:

```bash
"$EFORECAST_PYTHON" -m eforecast.config.auth add george --admin
```

Later accounts can be added from the **Users** page of the UI, or with the same command without
`--admin`.

### 2.5 Start it

```bash
sudo systemctl enable --now eforecast-webui eforecast-metrics
systemctl status eforecast-webui eforecast-metrics
journalctl -u eforecast-webui -n 50 --no-pager
```

Both units matter. The second one exposes the installation's state for monitoring, and until it
runs, every dashboard panel is blank — and a blank panel reads as a quiet system.

If this host uses a firewall, open only your actual LAN subnet and the configured port. Replace
the example values:

```bash
sudo ufw allow from 192.168.1.0/24 to any port 8000 proto tcp
```

The UI is then at `http://<workstation>:8000`.

Without systemd, the same server is:

```bash
set -a; . /etc/eforecast/webui.env; set +a
"$EFORECAST_PYTHON" -m eforecast.config.serve --host 0.0.0.0 --port 8000
# or, for a single local user with no login at all:
"$EFORECAST_PYTHON" -m eforecast.config.serve --host 127.0.0.1 --no-auth
```

### 2.6 The doctor is the acceptance test, not the install script

```bash
set -a; . /etc/eforecast/webui.env; set +a
cd "$EFORECAST_APP_DIR"
bin/eforecast doctor
```

| exit | meaning |
| --- | --- |
| 0 | ready |
| 1 | working, with things that need attention |
| 2 | something must be fixed first |
| 3 | the doctor itself could not run |

Every finding carries one remedy and says whether it is yours or the administrator's. **What
must be zero is the *must be fixed* count.** A fresh machine normally reports *attention*
findings for things it genuinely lacks — prewarmed model weights, the model images, a captured
release — and those are lab infrastructure, not a broken install.

Then confirm the machine can actually train, on a synthetic case with no private data, before
pointing it at a real one: see the synthetic worker smoke commands in
[gpu_training_optuna.md](gpu_training_optuna.md#runbook). That smoke tests the *image*, not the
native environment and not a private case. After it passes: copy a working case, set its real
data paths, run **Check readiness**, and train the families it needs.

Do not enable a production route because a synthetic smoke passed.

### 2.7 Optional: the monitoring stack

PostgreSQL for Optuna, the Optuna dashboard, Grafana, Prometheus, DCGM and cAdvisor are one
compose file:

```bash
docker compose -f docker/compose.yaml --env-file docker/.env up -d
```

**Set credentials in `docker/.env` first, and set all of them.** Copy `docker/.env.example`
and fill it in: every variable compose reads has a fallback compiled into `docker/compose.yaml`,
which is a tracked and pushed file, so anybody who can read the repository knows those
fallbacks. A missing line does not disable a service, it runs on the public default -- and each
of these services publishes a port. `docker/.env` is gitignored and never committed, so it does
not arrive with a clone: a new machine creates its own.

**Grafana reads its password from there once**, at its first start, when it creates its
database: user `admin`, password `GRAFANA_PASSWORD`. A later change to `docker/.env` does not
reach it; this sets the current value without anybody having to see it:

```bash
grep '^GRAFANA_PASSWORD=' docker/.env | cut -d= -f2- \
  | docker exec -i eforecast-grafana-1 grafana cli admin reset-admin-password --password-from-stdin
```

**PostgreSQL does the same with `OPTUNA_DB_PASSWORD`**: it takes it once, when it first creates
its data. A database that was started before `docker/.env` was filled in keeps the fallback of
`compose.yaml`; the containers started back then go on working with it, and the first one started
later -- the copier of the Optuna dashboard, or the dashboard itself once it is recreated -- is
refused with *password authentication failed*. Give the database the value of `docker/.env`
(nobody sees it; it answers `ALTER ROLE`), then recreate what reads it, one right after the other:

```bash
PW="$(grep '^OPTUNA_DB_PASSWORD=' docker/.env | cut -d= -f2-)" \
  docker exec -i -e PW eforecast-postgres-1 sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB"' <<'SQL'
\set pw `printf %s "$PW"`
ALTER USER CURRENT_USER PASSWORD :'pw';
SQL
docker compose -f docker/compose.yaml --env-file docker/.env up -d --no-deps optuna-dashboard grafana optuna-mirror
```

**A firewall on this host** (`ufw`) drops what the monitoring containers send to the host
itself, and Prometheus reads two endpoints there: the orchestrator's (port 9108, while a training
runs) and the installation's numbers (9109, the `eforecast-metrics` service). Let the monitoring
network in on both:

```bash
docker network inspect eforecast_default -f '{{(index .IPAM.Config 0).Subnet}}'   # e.g. 172.18.0.0/16
sudo ufw allow from 172.18.0.0/16 to any port 9108,9109 proto tcp
```

Prometheus's page *Status → Target health* (`http://localhost:9090/targets`) then shows
`eforecast-app` UP within 15 seconds; *context deadline exceeded* there means the rule is
missing. Its page *Alerts* lists the alerts of `docker/monitoring/prometheus/rules/`, and
[troubleshooting §7](troubleshooting.md#an-alert-in-grafana-or-prometheus) says what each means.

**cAdvisor** reports every container's CPU, memory, disk, network and restarts (dashboard
*eforecast containers*). A stack started before it joined, on 20 November 2026, adds it and has
Prometheus read the new job:

```bash
docker compose -f docker/compose.yaml --env-file docker/.env up -d --no-deps cadvisor
docker compose -f docker/compose.yaml --env-file docker/.env restart prometheus
```

**The Optuna dashboard shows the cases' studies through `optuna-mirror`.** The trainings keep
each study in a file beside its models and the dashboard reads the database, so a service copies
the one into the other, about once a minute, from the metrics service on port 9109 (the firewall
rule above lets it in). A stack started before it joined, on 23 November 2026, picks up the new
code of the metrics service, has Prometheus read its two new jobs, and starts the copier:

```bash
sudo systemctl restart eforecast-metrics
docker compose -f docker/compose.yaml --env-file docker/.env restart prometheus
docker compose -f docker/compose.yaml --env-file docker/.env up -d --no-deps optuna-mirror
docker logs -f eforecast-optuna-mirror-1      # one line per cluster as it is copied; Ctrl-C to leave
```

Prometheus then shows the targets `processes` and `studies` UP, the dashboard *eforecast training*
fills, and `http://localhost:8080` lists one study for every case and cluster, with the trials of
every method of that cluster in it ([what such a study shows](gpu_training_optuna.md#monitor-and-recover)).
The copier writes only into studies it made itself; it leaves alone a study of the same name that
something else created. A copier that is changed takes effect when its container starts again
(`docker restart eforecast-optuna-mirror-1`); the copies an earlier version made, one for every
method, are folded into the groups and removed on its first round.

The Studies page links to the dashboards through
`EFORECAST_OPTUNA_DASHBOARD_URL` and `EFORECAST_GRAFANA_URL` (defaults: localhost ports 8080 and
3000 — on another machine, set them to this workstation's address).

Note that `docker compose up -d` with no service named also starts the **message broker**.
Nothing dispatches to it until a venue is decided, so this is harmless, but set its credentials
before starting it. Prometheus reads the broker's own numbers only once the target of job
`rabbitmq` in `docker/monitoring/prometheus/prometheus.yml` is filled in; it is left empty, so
that a machine without a running broker does not show a target that is DOWN for good.

---

## 3. When the machine's details change

Re-run `bash deploy/configure.sh --write` after anything it detected moves: a new checkout, a
new environment, a workspace on another disk. It is the only thing that is per-machine, which is
why `deploy/eforecast-webui.service` can be used straight from git.

Then restart the service — an `EnvironmentFile` is read when the unit starts, not reloaded:

```bash
sudo systemctl restart eforecast-webui
```

Drain accepted work before changing the interpreter, the release, the venue or the store
bindings, and restart the coordinator as well as the API, so that new work uses the environment
you intend.

---

## 4. Retiring a machine

Retiring a browser client does not remove its account. Retiring a machine that *runs* cases means
preserving its cases, its durable state and its artifacts first — nothing is replicated just
because another machine has the repository.

1. **End the sessions.** A cookie records who it belongs to, not where it was issued, so this is
   per account, across every machine that account is signed in on. The password is unchanged, so
   the person signs back in wherever they still want to be:

   ```bash
   "$EFORECAST_PYTHON" -m eforecast.config.auth revoke <name>
   ```

2. **Inventory and retire its credentials.** `bin/eforecast secret list` names what is in the
   configured secret source. Also check protected connection files, cluster tokens, cloud
   profiles and deployment env files — do not assume everything is in one directory.

3. **If it ran work**, let its runs finish or stop them before removing it.
   `python -m eforecast.config.run_processes` shows what it is running.

4. **Remove the services**, if it ran them:

   ```bash
   sudo systemctl disable --now eforecast-webui eforecast-metrics
   ```

   Do **not** remove its run store, its execution receipts or its data as part of uninstalling a
   service.

Leaving the account itself alone is deliberate. Revoking sessions is reversible by signing in
again; removing a user is not, and it takes their case ownership with it.
