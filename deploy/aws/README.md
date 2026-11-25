# deploy/aws/ — one rented machine, to test the cluster route away from the lab

For the maintainer. Nothing here is needed to run cases.

## In short

**What it is.** A set of tools to rent **one** machine from Amazon (AWS EC2) for a few hours,
build a miniature cluster on it, run the cluster route's tests there, and remove everything
again. It answers one question: *does the route that runs cases on a cluster
([../kubernetes/](../kubernetes/README.md)) also work on a machine that is not the lab's?* It is
a test of portability. It is not a place where your cases run, and it is not a way to add
computing power to the lab.

**When it is executed.** Only when a person types the launch command of this page. Nothing else
can start it:

* the pipeline scripts, the web pages and their services never call it;
* `bin/eforecast route` knows two places where a run may execute, `local` and `broker`. AWS is
  not one of them: no run is ever sent "to AWS" by the installation;
* there is no timer, cron entry or service for it.

**What it costs.** Real money, on the AWS account of whoever launches it. The launcher refuses
a trial declared above USD 50 or 5 hours, and switches the machine off before the deadline in
two independent ways — but the "USD 50" is a number you wrote, not a limit AWS enforces.

**What has run so far.** One trial, once, with synthetic data (below). No real case has been
on AWS, and no GPU machine has started there.

## The pieces, and where each one runs

| file | runs on | what it does |
| --- | --- | --- |
| [`pilot.example.json`](pilot.example.json) | — | the template of the **envelope**: a file in which you declare the account, the region, the deadline, the budget and the one machine you want |
| `eforecast/venues/aws_pilot.py` | your workstation | checks an envelope. Contacts AWS only with `--check-account`, and then only to ask "which account is this profile?" |
| `cpu_pilot.py` | your workstation | `launch`, `inspect`, `destroy`: the only program here that creates or deletes anything at AWS |
| `bootstrap_cpu.sh` | the rented machine | starts the miniature cluster there: PostgreSQL, RabbitMQ and a one-node Kubernetes (k3s), each as a Docker container |
| `run_harness.sh` | the rented machine | runs one test script inside the pipeline's worker image, against that miniature cluster |

## How a trial is executed, step by step

Steps 1–5 and 10 run on your workstation, from the repository root; 7–8 on the rented machine;
6 and 9 are copies you make by hand between the two.

**0. What you need first.** `boto3` in the Python you use; an AWS profile on your machine that
can sign in to the account (credentials stay in that profile — they never go into the
envelope); a new SSH key pair made for this trial; your own public IPv4 address; the id of a
subnet and of an Ubuntu image (AMI) in the region you choose.

**1. Write the envelope.** Copy the template to a folder **outside the repository** and fill
it ([the fields](#the-envelope-field-by-field)).

**2. Check it — offline.**

```bash
python -m eforecast.venues.aws_pilot /protected/pilot/envelope.json
```

Prints `"status": "envelope_valid"` with `"provisioned": false`, or `"status": "blocked"` with
the reason. Nothing is contacted, nothing is created.

**3. Check the account — one question to AWS.**

```bash
python -m eforecast.venues.aws_pilot /protected/pilot/envelope.json --check-account
```

Signs in with the envelope's profile and compares the account it reaches with the one written
in the envelope: `AWS profile resolves to a different account; nothing was changed` if they
differ. Still nothing is created.

**4. Launch — this is the step that creates things and starts the bill.**

```bash
PYTHONPATH=. python deploy/aws/cpu_pilot.py launch \
    --envelope   /protected/pilot/envelope.json \
    --inventory  /protected/pilot/inventory.json \
    --public-key /protected/pilot/id_ed25519.pub \
    --operator-cidr <your public IPv4>/32 \
    --subnet <subnet id>  --ami <Ubuntu image id>
```

It first refuses whatever is out of bounds — more than one machine, a machine type it does not
know (`m6i.xlarge`, `m7i-flex.large` or the GPU type `g5.xlarge`), more than USD 50 or 5 hours,
a deadline less than 30 minutes or more than 5 hours away, an inventory file that already
exists, an account other than the envelope's, and for a GPU an account that is not on AWS's
paid plan. Then it creates, in this order:

1. a **firewall rule** (security group) that lets in SSH from your address only;
2. the **SSH key** entry for your public key;
3. a **permission** (IAM role) that can do exactly one thing: terminate this trial's machine;
4. the **machine** itself, with an 80 GB encrypted disk, tagged with the trial's name;
5. a **timer at AWS** (EventBridge Scheduler) that terminates the machine 10 minutes before
   the deadline — if this timer cannot be set, the machine is terminated at once.

The machine also carries a timer of its own that powers it off 15 minutes before the deadline,
and for this machine power-off means termination. Every created item is written into the
**inventory** file as it appears; that file is what `destroy` later works from, so keep it.
The command ends by printing the machine's id, the deadline and the inventory's path.

**5. Look.** `cpu_pilot.py inspect --envelope … --inventory …` prints the inventory and the
machine's state and public address.

**6. Put the code and the worker image on the machine — by hand.** There is no script for
this. In the trial that ran, `scp` copied an archive of the source, an archive of the worker
image and the test helpers to `/home/ubuntu/pilot/`; `run_harness.sh` expects the source at
`/opt/eforecast` and the image loaded into Docker.

**7. On the machine: build the miniature cluster.**

```bash
sudo bash /home/ubuntu/pilot/bootstrap_cpu.sh
```

**8. On the machine: run the tests.** Each test is one script, run inside the worker image:

```bash
bash /home/ubuntu/pilot/run_harness.sh docker/smoke_managed_kubernetes.py --datasets …
```

They are the same scripts that test the cluster route locally
([../kubernetes/MANAGED_RUNS.md](../kubernetes/MANAGED_RUNS.md#repeat-the-isolated-drill)):
submit a run and submit it again, kill a worker, cancel, and — the reason for renting a real
machine — reboot the host between accepting a run and executing it
(`docker/smoke_managed_restart.py accept`, reboot, `… resume`). The data are synthetic.

**9. Bring the results back — by hand.** `docker/collect_managed_qualification.py` writes one
JSON file on the machine; copy it out before the machine is gone. Records that carry the
account id or machine ids stay off the repository (`deploy/aws/evidence/` is ignored by git).

**10. Remove everything.**

```bash
PYTHONPATH=. python deploy/aws/cpu_pilot.py destroy \
    --envelope /protected/pilot/envelope.json --inventory /protected/pilot/inventory.json
```

Terminates the machine and waits for it, then deletes the timer, the permission, the key entry
and the firewall rule — each only after checking that it carries this trial's tag — and
refuses to report success while a disk of the trial still exists. It prints
`{"status": "removed", …, "remaining_volumes": 0}`. A passed deadline does not stop `inspect`
or `destroy`. Afterwards, look in the AWS console once more: `destroy` does not list the
deleted items again.

## What protects you, and what does not

| it does | it does not |
| --- | --- |
| start at most one machine per launch, of three allowed types | check whether a trial of yours is already running |
| end the machine before the deadline twice over: its own timer, and AWS's | cap the bill: no budget or billing call is made |
| give the machine no AWS permissions at all, and admit SSH from one address | undo a launch that failed half-way: the firewall rule, key entry and permission are created before the machine and stay until you run `destroy` |
| delete only what carries the trial's tag and is in the inventory | touch the shared network (the subnet and its VPC), which it never created |

## The envelope, field by field

| field | meaning | enforced? |
| --- | --- | --- |
| `version` | must be `1` | yes |
| `account_id` | the 12-digit AWS account | yes: `--check-account` and `launch` compare it with the account the profile reaches |
| `region`, `profile` | where, and with which AWS profile of your machine | used for every call |
| `deadline_utc` | the fixed end of the trial; a retry does not move it | yes: both shutdown timers are set from it |
| `max_duration_seconds` | the longest the trial may last | yes: `launch` requires it, at most 18000. **It is not in the template: add it** |
| `spending_limit_usd` | the budget you declare | only that the number is at most 50 |
| `cpu`, `gpu` | each `{"instance_type": …, "nodes": …}` | yes: exactly one may be used. **The template has `nodes: 1` in both; switch the other off with `{"instance_type": null, "nodes": 0}`**, or `launch` refuses |
| `namespace`, `lab_namespace` | the cluster namespace of the trial and of the lab; they must differ | checked |
| `authority` | must be `new-isolated`: the trial has databases of its own, never a copy of the lab's | checked |
| `architecture` | `amd64` or `arm64` | checked |
| `images` | the worker image, pinned by digest | format checked; not used by the launcher |
| `max_transfer_bytes`, `scratch_bytes`, `owner`, `teardown_owner` | declarations for the record | format checked; nothing reads them |

## What has run, and what has not

* **28 October 2026 — the one completed trial.** One `m7i-flex.large` in `eu-north-1`, synthetic
  data, its own PostgreSQL and RabbitMQ. The run lifecycle, a dataset stage, a model fit and a
  real reboot of the host passed. AWS's timer terminated the machine, and the machine, its
  disk, the firewall rule, the key entry, the permission and the timer were then confirmed
  gone; the shared network was kept. The compute came to about USD 0.11 at list price (the
  actual bill, disk and address charges were not measured). The trial found a defect in how a
  finished job's evidence was kept past a deadline; it was repaired and tested locally and
  **not** run again on AWS.
* **9 November 2026 — a GPU attempt that started nothing.** AWS had raised the account's GPU
  quota to 4 vCPUs (room for one `g5.xlarge`), but refused the launch: the account is on the
  Free plan, which excludes GPU machines. The firewall rule, key entry and permission the
  launcher had already created were removed with `destroy`. Since then the launcher asks for
  the account's plan first and stops there. A paid plan changes how the account is billed and
  is the account owner's decision.
* **Not done, so not known:** a GPU machine at AWS; AWS's managed Kubernetes (EKS); more than
  one machine; storage shared between machines; a real case with real weather data.

## Before another trial

Start from a new envelope with a new deadline, a new SSH key and a new inventory file — never
the old ones. Recheck the quota, the image id, the price and your address on the day. Decide
beforehand what the trial is to prove, stop launching work early enough for it to finish, and
do step 10 yourself rather than waiting for the timers. The lab keeps working locally
throughout: a trial uses nothing of the lab's but a copy of the code.
