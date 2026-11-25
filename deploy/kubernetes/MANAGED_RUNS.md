# Managed runs through RabbitMQ and Kubernetes

For maintainers. **Read [README.md](README.md) first**: it says in plain words what this venue
is, when a run goes to the cluster, and what happens to it step by step. This page is the
detailed contract behind it: the exact configuration, what is guaranteed, the drill, and
recovery. Load the
[installation environment](../../docs/README.md#before-copying-a-terminal-command) before the
commands; they configure execution and change routes, and are not colleague setup steps.

New managed Jobs keep the evidence of how their Pod ended across an outage of the coordinator.
Evidence of older Jobs that was already lost stays unknown; a failed Job alone never proves
that it was cleaned up safely.

This opt-in route runs **one accepted entry point in one Job**. The Job hosts a coordinator of
its own. That coordinator prepares the accepted snapshot and runs the existing sequence of
stages, their compute workers, result verification and the cleanup of descendants. RabbitMQ
transports a small reference to that run; it does not carry data, commands or credentials.
Stages inside the Job are not dispatched again as independent Jobs.

The local route remains the default. This integration does not perform a lab-wide cutover,
provide a Kubernetes implementation of the separate Docker foundation-family adapters, or
qualify a multi-node shared filesystem. An explicitly selected Docker model runtime is refused
at submission on this route. A case's runtime dependencies still need image qualification.

## Configure one qualified target

1. Use PostgreSQL for both durable run state and A08. Keep separate run/controller/compute
   namespaces as usual. Use a RabbitMQ vhost dedicated to this durable store; consumers for
   different stores must not consume each other's queue. Install `pika` in the host coordinator.
2. Build and qualify the worker image, then use its **registry manifest digest**. The accepted
   request also records a content digest of package Python files and the three managed entry
   points. The Pod verifies these before preparing inputs. Rebuild when code changes; a mutable
   tag or a stale package is not silently accepted.
3. Apply `00-namespace.yaml` and `10-rbac.yaml`. The executor service account can manage Jobs
   and read/patch Pods only in this namespace. **Reapply the updated Role before using evidence
   retention**: patch permission is required to remove our finalizer with UID/revision checks.
   The worker service account has no API rights and its
   token is not mounted. Populate the `managed-connections` Secret from a protected env file:
   `EFORECAST_RUN_STORE_DSN`, `EFORECAST_ALLOCATOR_DSN`, and `EFORECAST_ALLOCATOR_NAMESPACE`.
   Do not put their values in the target file, broker payload, image or case YAML.
4. Adapt [managed-run.example.json](managed-run.example.json). Replace the placeholders. Mount
   the workspace and execution state at **the same absolute paths** on host and Pod. Only the
   target's `shared_root` is mounted in the Pod, so the case's input files, the weather archive,
   the model folders and the caches must all lie under it. The qualified hostPath form is tied
   to one explicit hostname; state is outside the mutable workspace. A PVC requires a separately
   tested shared-storage deployment. Set UID/GID and filesystem permissions consistently: the
   Job uses the submitting coordinator's UID/GID.
5. Set `EFORECAST_KUBE_RUNS_FILE` on the API and independent coordinator. Also configure
   `EFORECAST_KUBE_SERVER`, `EFORECAST_KUBE_NAMESPACE`, `EFORECAST_KUBE_CA`,
   `EFORECAST_KUBE_TOKEN_FILE`, and `EFORECAST_BROKER_URL` on the coordinator. The endpoint and
   namespace must match the target. Keep API tokens and connection material in protected files
   or the deployment's credential source. The Pod receives only its explicit environment and
   the connection Secret, not the host's AWS login or Kubernetes token.
6. After the isolated qualification, use `bin/eforecast route CASE broker --note "qualification
   reference"` for one drained case. No UI change is needed to submit: the normal Run request
   uses the case's durable route. The target, image and resource bounds are pinned at acceptance.
   Changing the target file never redirects accepted work; retain a coordinator for the accepted
   target until it drains.

### Apply the settings and inspect before submitting

Keep the target JSON outside the mutable workspace and the repository. For example, set
`EFORECAST_KUBE_RUNS_FILE=/etc/eforecast/managed-run.json` consistently on API and coordinator.
Do not copy the example unchanged: it contains placeholders and a 30-minute active deadline,
which may be too short for a real training run. CPU/RAM/Job bounds and deadlines are accepted
inputs; changing the file affects future submissions, not already accepted ones.

Create the connection Secret from a protected env file containing only the three variables
listed in step 3. The file uses `NAME=value` lines, without shell quotes. Do not print it:

```bash
# State-changing, using the intended cluster context and an authorized provisioning identity.
export MANAGED_CONNECTIONS_FILE=/protected/path/managed-connections.env
chmod 600 "$MANAGED_CONNECTIONS_FILE"
kubectl -n eforecast create secret generic managed-connections \
  --from-env-file="$MANAGED_CONNECTIONS_FILE" --dry-run=client -o yaml | kubectl apply -f -
```

This sends connection material to the cluster. Keep its source file protected; never retain
rendered Secret YAML in logs/evidence. The ordinary executor cannot read Secrets through the
API. For rotation, drain/reconcile accepted work and follow the credential owner's procedure;
replacing a Secret is not proof that every running consumer has reloaded it.

Start the API and the coordinator with their matching settings. `python -m
eforecast.config.serve` starts a coordinator by itself (`--coordinator auto`, the default);
with `--coordinator external` you run `python -m eforecast.config.run_coordinator` under your
own supervision instead. Inspect `bin/eforecast route CASE`
and `bin/eforecast doctor`. A missing target, unreachable broker/store or unavailable image
must be resolved before the case switch. Then make the explicit route decision in step 6 and
submit a synthetic qualification case through the normal API/UI. Expect one durable run ID,
one assigned Job/Pod identity, verified stage outcomes and separately completed cleanup.

## Guarantees and limits

- Submission and root outbox insertion commit together. Claim, cancellation/generation/input
  re-check and Job intent insertion also commit together. A database-locked queue cap bounds
  launch intents across dispatchers. Redelivery reuses the same dispatch and Job identity.
- The Pod UID binds once before preparation. A replacement cannot adopt the attempt. The
  ordinary coordinator skips broker runs, and direct host preparation cannot launch them.
- Job success is only process evidence. Managed outcomes and stage artifacts still pass the
  existing A09 checks. The manifest is content-bound to its durable intent.
- Pending placement has a deadline and visible wait reasons. Cancellation before attachment
  closes launch authority. Attached cancellation lets the Pod coordinator stop and reap its
  descendants. If it cannot do so, the Job's active deadline bounds execution; its exit must
  then be observed before cleanup is accepted. A signal or deletion request is not completion.
- Confirmed termination of the **exact assigned Pod** fences its A08 controller and permits
  cleanup only for matching managed allocations/native namespaces. Missing Jobs, unreachable
  nodes, foreign Pod identities, external containers and corrupt journals retain ownership as
  unknown. They do not authorize a replacement or infer success.
- New managed Pods carry `eforecast.io/termination-evidence` from creation. Reconciliation
  persists the exact container exit receipt and durable run cleanup before conditionally
  removing only this finalizer. A coordinator outage therefore retains evidence through the
  active deadline. Finalizers retain API records, not running containers. API conflicts and
  missing patch permission are visible retryable waits; the bounded assignment slot stays held.
  Do not manually strip the key to free capacity. Restore the coordinator/access/storage and
  let it reconcile. Existing accepted intents are not rewritten; an already-lost legacy Pod
  cannot acquire a receipt retroactively.
- CPU/RAM limits and zero/one whole GPU are explicit: they are the Job's requests and limits,
  and inside the Job nothing further is reserved for a fit.
  This does not claim fractional GPU sharing between Kubernetes Jobs. GPU runtime/device-plugin
  setup and each image's model compatibility require their own live qualification.
- A broker outage retains pending work and reports the wait. A coordinator restart rediscovers
  persisted intents. Changing execution venue is refused until the scope's work and cleanup
  drain. Rollback is the existing explicit `route CASE local` decision, not an automatic fallback.

## Repeat the isolated drill

`local_cluster.sh` does all of this on one workstation — `up`, `image`, then `drill`
([README.md, "Try it on one workstation"](README.md#try-it-on-one-workstation-start-to-finish)).
By hand, it is:

Provision disposable PostgreSQL, RabbitMQ and a local Kubernetes cluster with the same mount
layout; side-load or pull the exact worker image digest. Configure the variables above for those
isolated services, never the live lab. Then run:

```bash
python docker/smoke_managed_kubernetes.py \
  --scratch /tmp/eforecast-kube-integration-YYYYMMDD/shared \
  --datasets --cluster-container eforecast-kube-integration
```

The script uses the FastAPI submission endpoint used by React, tests idempotent HTTP retry,
claim-before-create recovery, repeated real broker delivery, cancellation with a child process,
and optionally an actual agent SIGKILL and the synthetic real dataset stage. Its JSON receipt
records run identities and outcomes. Keep the deployment alive until all its owned Jobs and
reservations are inspected, then delete only these disposable resources. This drill does not
qualify GPU/AWS execution or restoration of the control and allocator databases together.
For the additional fitting check, pass `--compute cpu` or `--compute gpu --compute-count 2`;
the latter requires two qualified cards and a two-Job target. (That check and the deadline
drill at the end of this page have their Job run a helper, `docker/managed_compute_probe.py`
and `docker/managed_deadline_probe.py`. The drill copies the helper into the shared folder,
beside the script the Job runs, so no image has to carry it. Both follow the worker interface
of 13 November 2026 — a worker is a process of its Pod and holds nothing in a ledger — and
`tests/test_managed_probes.py` runs both without a cluster.) The separately invoked
`docker/smoke_managed_restart.py accept|resume` helper tests an actual host reboot between
acceptance and execution. It does not simulate disk loss or authority restoration.


## Diagnose and recover

Inspect the run in the UI first. Its run ID/dispatch reference identifies the assigned Job;
use the recorded names below, not a broad deletion selector:

```bash
kubectl -n eforecast get job JOB_NAME -o wide
kubectl -n eforecast get pod POD_NAME -o wide
kubectl -n eforecast logs POD_NAME -c worker --tail=100
kubectl -n eforecast get events --sort-by=.metadata.creationTimestamp
```

These are inspection commands. Keep logs inside the lab or redact before sharing.

| Symptom | Meaning | Action |
| --- | --- | --- |
| Accepted, no Job yet | Broker/store/queue capacity or target readiness may be waiting | Read dispatch reason; restore the same coordinator/bindings. Retry the same client key if the HTTP response was lost. |
| Pending Pod | Placement, image pull, mounts or permissions failed | Read Events and the run wait reason; fix the named prerequisite. The pending deadline bounds waiting. |
| Terminal container, cleanup held | The coordinator still needs the exact exit receipt and allocation settlement | Restore coordinator, PostgreSQL and receipt storage; let reconciliation collect them. |
| Pod is Terminating with `eforecast.io/termination-evidence` | Its API record is retained until durable cleanup; the key does not keep the container running | Check executor Pod `patch` permission, UID/revision conflicts and storage failures. Do not strip the key. |
| Missing/foreign Pod or unreachable node | Worker death is unproved | Retain ownership. Restore access/evidence; absence or elapsed time does not authorize a new writer. |
| Cancelled before attachment | Launch authority is revoked; no worker may attach | Let the same coordinator finish owned Job deletion and unattached finalizer cleanup. Do not resubmit through another venue. |

For new Jobs, reconciliation persists the exact worker exit, records outcome/cleanup, then
conditionally removes our finalizer and finishes assignment cleanup. Legacy Pods deleted before
this contract cannot gain an exit receipt retrospectively. Those cases remain unknown.

The ordinary smoke above does not exercise the whole retention failure matrix. In the same
**disposable** environment, also run `docker/smoke_managed_deadlines.py --help`, then follow
the exact deadline/cancellation replay commands.
That drill deliberately stops an agent/coordinator and changes only its isolated target's
deadline/placement before restoring it. Finish by inspecting owned Jobs/Pods, the controller's
active claims, pending settlements/completions and the evidence files (a worker has no
allocation of its own to inspect: what it was doing is in the marker it wrote). Remove only
the disposable resources recorded in that drill's inventory.
