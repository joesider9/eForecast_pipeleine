# deploy/

What turns this checkout into a running installation on one particular machine.

The code is the same on every machine. What differs — which user owns it, which Python has the
packages, where the workspace and the data disks are, which port to listen on — is worked out
by one script and written into files **outside** git. Nothing in this directory is edited by
hand.

## The pieces

| file | what it is |
| --- | --- |
| `configure.sh` | looks at this machine, shows what it found, and with `--write` saves it |
| `eforecast-webui.service` | the description of the **web service** for systemd, the part of Linux that keeps background programs running: it serves the web pages and their API, and is started again after a crash or a reboot |
| `eforecast-metrics.service` | the same for the **metrics service**, which publishes the numbers a monitoring dashboard reads ([cli.md, `metrics`](../docs/cli.md#metrics--numbers-for-a-monitoring-dashboard)) |
| `gpu_lost_reboot.sh`, `eforecast-gpu-watch.service` | optional: restart this machine when the NVIDIA driver reports that a graphics card is lost, so that the screen and the cards come back without a person ([below](#optional-restart-the-machine-when-a-graphics-card-is-lost)) |
| `eforecast-hardware-watchdog.service`, `eforecast-watchdog.conf` | optional: let the board's own watchdog timer restart a machine that has frozen outright ([below](#optional-the-boards-own-watchdog-timer)) |
| `kubernetes/` | what executing runs on a cluster instead of this machine needs — the `broker` venue. Optional, and off by default ([below](#the-two-optional-directories-and-when-they-run)) |
| `aws/` | tools to rent one machine from Amazon for a few hours and test that cluster route there. Started only by hand ([below](#the-two-optional-directories-and-when-they-run)) |

## What `configure.sh` does

```bash
bash deploy/configure.sh             # show what it found; writes nothing
bash deploy/configure.sh --write     # save it and install the two services (asks for sudo)
```

It works out:

* **the user** who owns the checkout. The services run as that user, never as root, so that a
  run started from the browser has the same GPUs, data folders and Python as your terminal;
* **the Python environment** — one that has both the web packages and torch;
* **the workspace**, the folder where accounts and cases are kept, and the folders the file
  browser of the pages may show;
* **the disks** the installation needs, so that the service waits for them after a reboot
  instead of starting with an empty folder where the data should be;
* **the address and port** of the pages, the port of the metrics, and which image release the
  installation goes by.

Anything it would detect can be given instead:

```bash
EFORECAST_WORKSPACE=/srv/eforecast EFORECAST_PORT=8080 bash deploy/configure.sh --write
```

`--write` produces these, and copies the two `.service` files beside them:

| file | what it holds | who reads it |
| --- | --- | --- |
| `/etc/eforecast/webui.env` | the settings, one `NAME=value` a line | both services when they start; `bin/eforecast`; your shell, if you load it with `set -a; . /etc/eforecast/webui.env; set +a` |
| `/etc/systemd/system/eforecast-webui.service.d/10-host.conf` | the four things systemd must know before the program exists, which it cannot take from the settings file: the user, the folder to start in, the disks to wait for, and the exact command | systemd |
| `/etc/systemd/system/eforecast-metrics.service.d/10-host.conf` | the same for the metrics service | systemd |

**Read the printout before `--write`.** Running it again is safe: it keeps the workspace the
installation already has, the release it already chose, and every `EFORECAST_…` line of the
settings file that it does not work out itself — the printout lists those under *kept from the
env file*, so you can see that a backup destination or a database address you added is still
there.

## After it

```bash
sudo systemctl enable --now eforecast-webui eforecast-metrics    # start now, and at every boot
sudo systemctl status eforecast-webui eforecast-metrics          # are they running?
journalctl -u eforecast-webui -f                                 # the service's own log
```

The settings file is read when a service **starts**, not while it runs. After changing it
(`sudoedit /etc/eforecast/webui.env`):

```bash
sudo systemctl restart eforecast-webui eforecast-metrics
```

The same goes for new code: the services keep running the version they were started with
([administrator.md §2](../docs/administrator.md#2-starting-and-stopping)).

## Optional: restart the machine when a graphics card is lost

A graphics card can stop answering while it works — the NVIDIA driver calls it *Xid 79, GPU has
fallen off the bus*. The machine keeps running, but the driver wants a reboot before it uses any
card again, and where the desktop is drawn through the same driver the screen and the remote
desktop stay dead until somebody restarts the machine. On 15 November 2026 that took two and a
half days, because nobody was in the room.

`eforecast-gpu-watch.service` runs `gpu_lost_reboot.sh` as root. It reads the kernel log and acts
on two messages only, *Xid 79* and *Node Reboot Required*: it keeps NVIDIA's own report of the
fault under `/var/lib/eforecast-gpu-watch/reports`, waits a minute, and restarts the machine. The
web service and the login screen come back by themselves; a run that was going is started again
by whoever started it, and continues from what it had finished.

It does not ask for an ordinary shutdown. A lost card leaves the display server and every program
that was drawing stuck inside the driver, where nothing can end them, and a shutdown waits for
them: on 19 November 2026 the first version of this script asked for a reboot, the shutdown
stopped every service — the script and its forced restart among them — and then hung for 23
hours. So the script stops Docker itself (90 seconds at most; the databases end cleanly, and
containers the daemon stopped come back at boot), writes everything to disk, and tells the
kernel to restart at once through its SysRq switch: sync, file systems read-only, restart. No
service and no driver is asked.

```bash
sudo install -m 755 deploy/gpu_lost_reboot.sh /usr/local/sbin/eforecast-gpu-lost-reboot
sudo install -m 644 deploy/eforecast-gpu-watch.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now eforecast-gpu-watch
journalctl -u eforecast-gpu-watch          # what it saw and did
```

* The script is copied, not run from the checkout: a root service must not execute a file an
  ordinary user can rewrite. After changing it here, install it again.
* It restarts the machine at most three times in 24 hours and then waits for a person
  (`EFORECAST_GPU_WATCH_MAX_REBOOTS` in the unit): a card that is lost again and again needs
  one.
* `sudo /usr/local/sbin/eforecast-gpu-lost-reboot --check` says whether everything a real fault
  would need is there, and changes nothing.
* To try the decision without restarting anything, as root:
  `EFORECAST_GPU_WATCH_DRY_RUN=1 /usr/local/sbin/eforecast-gpu-lost-reboot --from-stdin`, and type
  `NVRM: Xid (PCI:0000:17:00): 79, GPU has fallen off the bus.`
* **Try the restart itself once, with somebody in the room:**
  `sudo /usr/local/sbin/eforecast-gpu-lost-reboot --restart-now` restarts the machine the way a
  lost card would, at once. It interrupts whatever is running. A restart that has never been seen
  to finish is not one to rely on from another city. (Done on 20 November 2026: 51 seconds from
  the request to the new boot.)
* A machine that freezes outright, so that not even this script runs, is the job of the board's
  own watchdog timer: [next section](#optional-the-boards-own-watchdog-timer).
* It does not repair anything. What to look at on the card, and how to keep a card out of the
  training until it has been looked at:
  [troubleshooting §7](../docs/troubleshooting.md#the-screen-is-dead-and-the-machine-is-not-a-graphics-card-was-lost).

## Optional: the board's own watchdog timer

The script above needs a running machine. A machine whose kernel or service manager has frozen
needs something that does not: the watchdog timer on the board. Once started, it counts down,
and unless it is told in time that the machine is alive, it restarts the machine in hardware.
systemd tells it, every half of `RuntimeWatchdogSec`; when systemd stops, so does the telling.

That is what was missing on 19 November 2026. In the shutdown after the lost card, systemd itself
stopped at 10:01:07: it logged nothing more, and not even its own 30-minute limit on a restart
(`reboot.target`, which would have forced one) ever fired. The machine stayed so for 23 hours.
With the timer, the board would have restarted it about five minutes after 10:01.

Intel boards have one (driver `iTCO_wdt`), but Ubuntu puts the driver on the kernel's blacklist,
and systemd's module loader honours that list, so a line in `/etc/modules-load.d` would load
nothing. `eforecast-hardware-watchdog.service` loads the driver by name at boot, which the list
does not stop, and `eforecast-watchdog.conf` tells systemd to keep the timer alive:

```bash
sudo install -m 644 deploy/eforecast-hardware-watchdog.service /etc/systemd/system/
sudo mkdir -p /etc/systemd/system.conf.d
sudo install -m 644 deploy/eforecast-watchdog.conf /etc/systemd/system.conf.d/
sudo systemctl daemon-reload                                  # systemd now wants a timer ...
sudo systemctl enable --now eforecast-hardware-watchdog       # ... and the driver brings one
cat /sys/class/watchdog/watchdog0/{identity,state,timeout}    # iTCO_wdt, active, 300
```

* The order is the order of a boot: systemd wants the timer before the driver has announced it,
  and takes it up when it appears. On 20 November 2026 that was the same second: systemd's own
  log (`journalctl -b _PID=1 | grep -i watchdog`) says *Using hardware watchdog 'iTCO_wdt'* and
  *Watchdog running with a timeout of 5min*. If `state` still says `inactive` after three
  minutes, `sudo systemctl daemon-reexec` makes systemd take it up at once -- and the next boot
  would leave it unused, so the unit needs another way of loading the driver.
* If `enable --now` fails and `journalctl -k | grep -i tco` says *unable to reset NO_REBOOT flag,
  device disabled by hardware/BIOS*, the board's firmware does not let the operating system use
  the timer. Nothing else is changed; `sudo systemctl disable eforecast-hardware-watchdog`.
* `sudo /usr/local/sbin/eforecast-gpu-lost-reboot --check` says on its last lines whether the
  timer is active, and after how long it would restart the machine.
* **Try it once, with somebody in the room**, the only way to know that this board really
  restarts. Make the disks safe, then freeze the kernel on purpose (this kernel does not restart
  itself after a crash, `kernel.panic=0`):

  ```bash
  sudo systemctl stop docker.socket docker.service
  sudo sh -c 'echo s > /proc/sysrq-trigger; sleep 5; echo u > /proc/sysrq-trigger; sleep 5; echo c > /proc/sysrq-trigger'
  ```

  The screen stops, and about five minutes later the board restarts the machine. If it is still
  frozen after fifteen minutes, the timer does not work on this board: press reset, and take it
  out again as below.
* Taking it out: remove `/etc/systemd/system.conf.d/eforecast-watchdog.conf`, `sudo systemctl
  daemon-reload` (systemd stops the timer), `sudo systemctl disable eforecast-hardware-watchdog`.
* What it costs: a machine on which systemd is silent for five minutes is restarted. On a working
  machine that does not happen, however busy it is; a training interrupted that way resumes as it
  does after any restart.

## The two optional directories, and when they run

Neither is part of a workstation installation, and neither does anything unless a person
decides it.

| | `kubernetes/` | `aws/` |
| --- | --- | --- |
| **what it is** | a second place where a run can execute: a container on a Kubernetes cluster, instead of a process on this machine | a test: one rented machine with a miniature cluster on it, to see whether the cluster route works away from the lab |
| **when it is executed** | for a run started from the web pages, of a case somebody routed there with `bin/eforecast route <case> broker`, on an installation that has a cluster configured. Otherwise never | when somebody types `python deploy/aws/cpu_pilot.py launch …`. Never otherwise: no run is ever sent to AWS |
| **how** | the API records the run, the run coordinator hands it over through a message queue and creates one Job; the Job runs the same `run_*.py` script on a folder shared with the host | check an envelope file, launch one machine with two timers that end it, copy the code there by hand, run the cluster tests, destroy everything |
| **today** | tried in isolation; the lab runs `local`. `kubernetes/local_cluster.sh` builds the isolated trial on a workstation | ran once, on 28 October 2026, with synthetic data |
| **read** | [kubernetes/README.md](kubernetes/README.md) | [aws/README.md](aws/README.md) |

`bin/eforecast route` shows which venue applies: `venue    local  (default: nobody has decided)`
on an installation that has never used the cluster.

## Where to read on

| for | read |
| --- | --- |
| what every setting in the file is for, and which programs it reaches | [docs/cli.md §4](../docs/cli.md#4-the-settings-that-matter) |
| the whole procedure, from an empty machine to a passing health check | [docs/installation.md](../docs/installation.md) |
| running it afterwards: accounts, new versions, images, backups | [docs/administrator.md](../docs/administrator.md) |
