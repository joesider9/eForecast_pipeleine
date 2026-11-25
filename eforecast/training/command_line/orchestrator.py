# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""The training orchestrator: one queue of trials for every study, packed on the GPUs by memory.

Units of work
    * study units: one Optuna study per (cluster, method) of every project in the run
    * fixed units: transfer-learning retrains of the global trials on a cluster; they wait for the
      global study of their method to finish

Every tick the orchestrator reaps finished trials (tell / retry / record), forwards the per-epoch
progress of the running ones to the pruner, reads the cards, and admits new trials.

Admission is measured when the cards can be read and nobody fixed the trials per card:

    * a trial of a kind nobody has measured (method and experiment tag) starts alone on a card
      with everything that is free there; after its first epoch it lowers its own limit to what
      it used and reports it, and the rest of the card is given back (``observe_running``);
    * how many trials of a kind a card takes is learned from how busy the card was over whole
      trials of that kind (``learn_from``): a network that fills a card stays alone, small ones
      are grouped, and a mixed card takes what its least sociable kind allows (``pick_gpu``);
    * the largest kind waiting is offered a freed card first, and a trial that needs an empty
      card has one cleared for it;
    * a trial that runs out of memory on a shared card is retried alone, sized by what it held
      and asked for, and at last with its inputs kept on the host.

With ``--n-jobs``, or cards whose busy share cannot be read, the declared rules apply instead: at
most that many trials per card, the study with the most remaining work of each method in turn,
on the GPU with the most free memory, when the memory estimate of that kind of trial fits.

A trial starts, or waits for a running trial to free what it needs. It is never refused for its
size. An estimate above the largest card is a prediction made from other trials of the same
experiment, whose layer sizes were other ones, so the trial is tried alone on an empty card and
what it does there decides: out of memory, it is retried with its inputs kept on the host, then
with half its batch under ``--halve-batch``, and only then told FAIL (``sized``, ``try_alone``).
Until 2026-11-23 such a trial was failed at once as "impossible"; a search that kept coming back
to one region was refused every time, for an experiment that had trained at that batch size the
day before. A unit three of whose trials in a row ran out of memory with a card to themselves is
closed for the run (``OOM_STREAK_CLOSES``).  Before a study is asked, the question is whether the
*cheapest* tag of the method could start, because the tag is only known after ``ask()``; asking
with the largest tag let one oversized experiment block every smaller one of the same method.  At
most one asked trial per GPU slot waits in the queue, and a run that starts nothing for
``EFORECAST_NO_PROGRESS_SECONDS`` (default 900, 0 disables) stops with a diagnosis instead of
waiting for a machine that is busy with something else.

A card the driver reports lost (``NVML_ERROR_GPU_IS_LOST``: it fell off the bus or stopped
answering) is taken out of the run at once (``lose_card``): nothing more is started on it, its
trials are stopped, and they are told FAIL as ``card_lost``, which is not an attempt of their
study. From then on a trial that dies of a driver error, or before its first epoch, takes its
own card out with it, because a driver that has lost one card is not to be trusted with the
others; with no card left the run ends as failed. On 19 and 23 November 2026 the scheduler
logged the loss and went on starting a trial every few seconds on the dead card -- 85 of them,
which used up the attempts of the studies they belonged to.

A study is asked for at most ``ASK_BUDGET_FACTOR`` times the trials wanted of it. What counts
toward that is defined once, in ``optuna_search.attempts``: every trial but the failed ones that
say nothing about the hyper-parameters drawn -- cancelled, abandoned, interrupted, card lost.

A run ends with an outcome, and the exit code of ``run_training.py`` follows it: ``succeeded``,
``succeeded_with_failures`` and ``nothing_to_do`` exit 0, ``failed`` (nothing completed while there
was work, because every trial failed or every unit was skipped) exits 1, and ``cancelled`` exits 2.
SIGINT and SIGTERM ask the run to stop: the trials in flight are terminated, given
``CANCEL_GRACE_SECONDS`` to go, killed if they do not, and every asked trial -- running or parked --
is told FAIL so that no study is left with a RUNNING trial that no process will ever finish.

Only the cards are placed by this scheduler. Threads and host memory are the operating system's:
a trial is told how many threads to use, and nothing is set aside for it on the host and nothing
is recorded outside this process -- one run trains a case at a time.

Memory, declared rules: the estimate of a (method, experiment tag) is 1.5 x the median peak
reported by earlier trials (``trial_result.json``), a conservative table before the first one,
persisted in ``<path_model>/exe_files/memory_estimates.json``.  The child receives its admitted
share as ``EFORECAST_MEM_FRACTION`` and fails fast if it grows beyond it.  After an OOM the
estimate is raised by half, never past the card: first retry later, second retry alone on the
emptiest GPU (and, with ``halve_batch``, half the batch size, recorded as the user attribute
``batch_size_effective``), third failure is told FAIL with ``fail_reason`` oom.  Other errors are
told FAIL at once.

Trial processes are ``trial_runner.py`` children (``EFORECAST_TRIAL_CMD`` template to run them from
the Docker image, e.g. ``docker/run_trial.sh {path_weights} {gpu} {refit}``).  A killed orchestrator leaves
RUNNING trials in the studies; the manager fails them on the next start and completed trials are
skipped, so a run is resumable.
"""
import collections
import contextlib
from eforecast.config import run_reporting
import json
import logging
import math
import os
import shlex
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import time
import uuid

from eforecast.common_utils import trial_layout

import joblib
import optuna

path_pycharm = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, os.pardir))
from optuna.trial import TrialState

from eforecast.common_utils import completion, lineage
from eforecast.common_utils.devops_utils import get_results
from eforecast.training.command_line import docker_task, gpu_identity, gpu_sharing, leases
from eforecast.training.command_line import resource_reporting as resources
from eforecast.training.command_line.trial_prep import (KILLED, forward_progress, prepare_study, prepare_trial,
                                                        read_result, record_trial)
from eforecast.training.command_line import optuna_search, revisions
from eforecast.training.command_line.optuna_search import tell_once, trial_counts

log = logging.getLogger('orchestrator')

TICK_SECONDS = 3
# How long one card's resource measurement stays usable. It costs a PSS read per worker process,
# and admission asks for it once per candidate unit per card.
GPU_RESOURCE_TTL = float(os.environ.get('EFORECAST_GPU_RESOURCE_TTL', 1.0))
STATUS_EVERY = 60
CONTEXT_MB = 500            # the CUDA context of a process, not counted in max_memory_reserved
HEADROOM_MB = 300           # never fill a card completely
# A card that carries none of this run's trials and has this share of its usable memory free is
# an empty card: what else is on it (a desktop session takes 600 to 1100 MB here) is not ours
# to wait for.
EMPTY_CARD_SHARE = 0.85
SAFETY = 1.5                # estimate = SAFETY x median observed peak (+ context)
MIN_FREE_FRACTION = 0.05
OOM_RAISE = 1.5
MAX_OOM_RETRIES = 2
ASK_BUDGET_FACTOR = optuna_search.ASK_BUDGET_FACTOR     # never ask more than this x n_trials of a study
# A unit this many of whose trials in a row ran out of memory with a card to themselves -- and
# with half their batch, where that is allowed -- is closed for the run: what it draws does not
# fit this machine, and that is known from trying, not from an estimate.
OOM_STREAK_CLOSES = 3
# What a trial says when the card or the driver under it has gone. Not "out of memory", which a
# trial reports as a state of its own.
CARD_ERRORS = ('CUDA error', 'CUDA driver error', 'CUDA unknown error', 'cuDNN error', 'GPU is lost',
               'No CUDA GPUs are available', 'NVML')
NVML_GPU_IS_LOST = 15       # NVML_ERROR_GPU_IS_LOST: "the GPU has fallen off the bus or has otherwise become inaccessible"
PARKED_PER_SLOT = 1         # asked trials waiting for a GPU, at most this x the GPU slots
# A run that cannot start anything for this long is stuck on something outside itself (another run
# holding the cards, NVML lying, the host full of someone else's work): report and stop instead of
# waiting out the night.  0 disables it, for a deployment that deliberately waits for a worker.
NO_PROGRESS_SECONDS = int(os.environ.get('EFORECAST_NO_PROGRESS_SECONDS', 900) or 0)
# Card size when NVML cannot be asked.  It is a declaration, not a measurement, and the orchestrator
# says so wherever it uses it, because a wrong one decides which trials are impossible.
GPU_TOTAL_MB = int(os.environ.get('EFORECAST_GPU_TOTAL_MB', 11000))
CANCEL_GRACE_SECONDS = 15   # after SIGTERM, before SIGKILL, for a trial asked to stop
EXIT_CODES = {'succeeded': 0, 'succeeded_with_failures': 0, 'nothing_to_do': 0, 'failed': 1, 'cancelled': 2}
DEFAULT_MB = {'MLP': 1500, 'LSTM': 2500, 'TRANSFORMER': 4000, 'CNN': 5000, 'TIMM': 6000, 'CrossViVit': 7000,
              'Distributed': 2500}
DEFAULT_MB_OTHER = 3000
DEFAULT_MAX_PER_GPU = 4
# Admission by measurement. Processes on one card take turns, so once a card is running kernels
# all the time another trial only slows the ones already there. Measured on the RTX 2080 Ti of
# this workstation: one small MLP keeps a card busy 37% of the time, two keep it busy 100% and
# together do twice the work, and a third adds nothing; two convolutional networks on one card
# do 14% less work than one alone. But a trial is not on the card all its life -- a short one
# spends half of it starting up and evaluating -- so what counts is how busy a card is on
# average with so many trials of a kind on it. That number is learned per kind of network
# (method and experiment): a card takes another trial of a kind while its average busy share
# with that many is under this many percent. The trials per card is a result, not a setting.
# 0 keeps the declared rules: a fixed number of trials per card, each sized by an estimate.
BUSY_TARGET = float(os.environ.get('EFORECAST_GPU_BUSY_TARGET', 80) or 0)
BUSY_WINDOW_SECONDS = 30    # the span of the busy share shown in the status and in a refusal
MIN_LIFE_TICKS = 5          # a trial shorter than this many readings says nothing about its kind
# A trial of a kind that has been measured starts with this much room over what its kind used,
# instead of with the whole card: several of them can then start on one card at once.
KNOWN_ROOM = 1.5
KNOWN_MARGIN_MB = 512
MAX_TRIAL_THREADS = 4       # a trial whose work is on the card does nothing with more
RESERVED_CORES = 2          # left for the scheduler and the system when cores bound the trials per card
SETTLE_GROWTH = 0.25        # room kept for a trial that reported a footprint but set itself no limit


FAILURE_CAUSES_SHOWN = 5


def failure_causes(failures, ended_well=None, shown=FAILURE_CAUSES_SHOWN):
    """The failed trials of a run by cause, the commonest first: one line a (method, graph, error).

    Twenty-two rows of a table that all say ``'int' object is not iterable`` are one cause, and
    it is the graph they share that says which: every one of them had drawn ``mlp1``. When no
    trial of a graph ended well in this run, the line says so, because then it is the graph, or
    what it is given, and not a draw of the search.
    """
    counts = collections.Counter(
        (str(failure.get('method') or ''), str(failure.get('graph') or ''), str(failure.get('state') or ''),
         ' '.join(str(failure.get('error') or '').split())) for failure in failures)
    lines = []
    for (method, graph, state, error), count in counts.most_common(shown):
        whose = ' with graph '.join(part for part in (method, graph) if part) or state
        line = f'  {count} of them: ' + ', '.join(part for part in (whose, error or (state if whose != state else '')) if part)
        if method and graph and ended_well is not None and not ended_well.get((method, graph)):
            line += f'. No trial of {graph} ended well in this run'
            if state == 'error':
                line += ': its dry build in the graph editor shows whether the graph can be built as it is'
        lines.append(line)
    if len(counts) > shown:
        lines.append(f'  and {len(counts) - shown} other cause(s) in the table')
    return lines

def card_error(error):
    """Whether a trial's error is the card's or the driver's, as far as its text says."""
    text = str(error or '')
    return any(mark in text for mark in CARD_ERRORS)


def _text(value):
    """NVML answers with bytes on some versions and str on others."""
    return value.decode() if isinstance(value, bytes) else value


# ---------------------------------------------------------------------------------- GPU memory
class GpuPool:
    """Memory per GPU from NVML, and an explicit ``None`` when NVML cannot say.

    Returning zero for "I could not measure it" made an unreachable card look empty, which is the
    worst possible reading: the orchestrator would pack trials onto a card already full of someone
    else's work. Unknown is now unknown, and :meth:`Orchestrator.free_mb` falls back to counting
    only what this run reserved -- all it honestly knows.
    """

    def __init__(self, gpus, total_mb=None):
        self.gpus = list(gpus)
        self.nvml = None
        self.handles = {}
        self.query_failures = 0
        self.unavailable_gpus = set()
        self.lost_gpus = set()           # cards of this run the driver has lost (``check``)
        self.others = {}                 # the machine's other cards, by name: their loss is the driver's trouble too
        self.lost_others = set()
        self.declared_total = True       # until NVML says otherwise, the card size is a guess
        self.names = {}
        # A card is named by its UUID, which is the same string in every process that can see it.
        # The ordinal is not: inside a container with one device mapped it is 0, and 0 is also the
        # first card of the host, so two runs could reserve "gpu 0" on different hardware -- or,
        # worse, on the same card without seeing each other.
        self.keys = {gpu: f'{socket.gethostname()}#{gpu}' for gpu in self.gpus}
        budget = GPU_TOTAL_MB if total_mb is None else total_mb
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nvml = pynvml
            for gpu in self.gpus:
                # CUDA ordinals need not be NVML indices (visibility, ordering, containers).
                value = gpu_identity.cuda_uuid(gpu)
                try:
                    self.handles[gpu] = pynvml.nvmlDeviceGetHandleByUUID(f'GPU-{value}'.encode())
                except pynvml.NVMLError:
                    self.handles[gpu] = pynvml.nvmlDeviceGetHandleByUUID(f'MIG-{value}'.encode())
                self.keys[gpu] = _text(pynvml.nvmlDeviceGetUUID(self.handles[gpu])) or self.keys[gpu]
                self.names[gpu] = _text(pynvml.nvmlDeviceGetName(self.handles[gpu]))
            self.total = {gpu: pynvml.nvmlDeviceGetMemoryInfo(self.handles[gpu]).total >> 20 for gpu in self.gpus}
            self.declared_total = False
            self.others = self.other_cards()
        except Exception as error:  # noqa: BLE001
            # Do not retain partially resolved handles/UUIDs after a failed mapping.
            self.nvml, self.handles, self.names = None, {}, {}
            self.keys = {gpu: f'{socket.gethostname()}#{gpu}' for gpu in self.gpus}
            self.total = {gpu: budget for gpu in self.gpus}
            log.error('NVML is not available (%r). The memory in use on a card cannot be read, so this run '
                      'counts only its own reservations: a trial of another process or another run is '
                      'invisible to it and the cards can be overcommitted. Card size is taken as %d MB '
                      '(EFORECAST_GPU_TOTAL_MB), which also decides which trials are reported impossible. '
                      'Install pynvml, or make sure nothing else uses these cards.', error, budget)

    def other_cards(self):
        """``{name: handle}`` of the machine's cards this run does not use; one already lost is noted."""
        others, mine = {}, set(self.keys.values())
        try:
            count = self.nvml.nvmlDeviceGetCount()
        except Exception:               # noqa: BLE001 -- the run's own cards are known; this is extra
            return others
        for index in range(count):
            try:
                handle = self.nvml.nvmlDeviceGetHandleByIndex(index)
                name = _text(self.nvml.nvmlDeviceGetUUID(handle)) or f'card {index}'
            except Exception as error:  # noqa: BLE001
                if self.is_lost(error):
                    self.lost_others.add(f'card {index}')
                continue
            if name not in mine:
                others[name] = handle
        return others

    @staticmethod
    def is_lost(error):
        """Whether NVML's answer is that the card is gone -- not merely that it could not say."""
        return getattr(error, 'value', None) == NVML_GPU_IS_LOST or type(error).__name__ == 'NVMLError_GpuIsLost'

    def check(self):
        """Ask the driver about every card of the machine; returns the cards of this run it has lost.

        ``NVML_ERROR_GPU_IS_LOST`` is the driver's own word for a card that fell off the bus or
        stopped answering. Nothing can be started on such a card until the machine is restarted,
        and a trial started on it dies in seconds. A card the driver has lost stays lost here: it
        does not come back by itself.
        """
        if self.nvml is None:
            return set()
        # a pool made without ``__init__``, as tests make them, has had no card to lose yet
        lost_gpus = self.__dict__.setdefault('lost_gpus', set())
        lost_others = self.__dict__.setdefault('lost_others', set())
        for gpu, handle in list(self.handles.items()):
            if gpu in lost_gpus:
                continue
            try:
                self.nvml.nvmlDeviceGetMemoryInfo(handle)
            except Exception as error:  # noqa: BLE001
                if self.is_lost(error):
                    lost_gpus.add(gpu)
        for name, handle in list(self.__dict__.get('others', {}).items()):
            if name in lost_others:
                continue
            try:
                self.nvml.nvmlDeviceGetMemoryInfo(handle)
            except Exception as error:  # noqa: BLE001
                if self.is_lost(error):
                    lost_others.add(name)
                    log.error('The driver has lost a card of this machine that this run does not use (%s). Trials on '
                              'the cards of this run may die with it: one that does takes its card out of the run.',
                              name)
        return set(lost_gpus)

    def key(self, gpu):
        """The identity of a card across processes and containers: its UUID when NVML can say."""
        return self.keys.get(gpu, f'{socket.gethostname()}#{gpu}')

    def selection(self, gpu):
        """The same physical card for a fresh local process or Docker's device selector."""
        if gpu in self.handles:
            return self.key(gpu)
        # Advisory compatibility retains the parent's mask, never treats a remapped 0 as host 0.
        visible = os.environ.get('CUDA_VISIBLE_DEVICES')
        return visible.split(',')[gpu].strip() if visible is not None else str(gpu)

    def inventory(self):
        """``{ordinal: {key, name, total_mb}}``: which local ordinal is which physical card."""
        return {str(gpu): {'key': self.key(gpu), 'host': socket.gethostname(), 'name': self.names.get(gpu),
                           'total_mb': self.total.get(gpu), 'identified': gpu in self.handles}
                for gpu in self.gpus}

    @property
    def telemetry(self):
        """True while the memory in use on a card can actually be read."""
        return self.nvml is not None and not getattr(self, 'unavailable_gpus', set()) and self.query_failures == 0

    def used(self, gpu):
        """Memory in use on ``gpu``, or ``None`` when that cannot be measured."""
        if self.nvml is None:
            return None
        try:
            used = self.nvml.nvmlDeviceGetMemoryInfo(self.handles[gpu]).used >> 20
            self.unavailable_gpus.discard(gpu)
            if not self.unavailable_gpus:
                self.query_failures = 0
            return used
        except Exception as error:  # noqa: BLE001
            if not hasattr(self, 'unavailable_gpus'):
                self.unavailable_gpus = set()
            self.unavailable_gpus.add(gpu)
            if self.is_lost(error) and hasattr(self, 'lost_gpus'):
                self.lost_gpus.add(gpu)
            self.query_failures += 1
            if self.query_failures == 1 or self.query_failures % 200 == 0:
                log.error('NVML cannot read gpu %d (%r; %d failed queries). Its memory is reported unknown and '
                          'only this run\'s own reservations are counted there.', gpu, error, self.query_failures)
            return None

    def used_text(self, gpu):
        used = self.used(gpu)
        return 'unknown' if used is None else f'{used:.0f} MB'

    def utilization(self, gpu):
        """Percent of the last sample period in which the card ran somebody's kernels, or ``None``."""
        if self.nvml is None:
            return None
        try:
            return float(self.nvml.nvmlDeviceGetUtilizationRates(self.handles[gpu]).gpu)
        except Exception:               # noqa: BLE001 -- unknown is an answer; the caller admits less
            return None

    def process_memory(self, gpu):
        """Current per-host-PID NVML occupancy, or None when attribution is unavailable."""
        if self.nvml is None:
            return None
        try:
            usage = {}
            for query in (self.nvml.nvmlDeviceGetComputeRunningProcesses,
                          self.nvml.nvmlDeviceGetGraphicsRunningProcesses):
                for process in query(self.handles[gpu]):
                    amount = process.usedGpuMemory
                    if amount is None or amount < 0 or amount >= 2 ** 63:
                        return None
                    usage[process.pid] = max(usage.get(process.pid, 0), amount / 2 ** 20)
            return usage
        except Exception:
            return None


class MemoryModel:
    """Memory estimate per (method, experiment tag), learned from the peaks the trials report."""

    def __init__(self, path_file):
        self.path_file = path_file
        self.peaks = collections.defaultdict(list)
        self.raised = {}
        self.footprints = collections.defaultdict(list)     # key -> [[batch size, MB on the card], ...]
        self.capacity = {}                                  # key -> trials of this kind one card keeps busy
        self.limit_mb = None                                # the most a card of this machine can give
        if os.path.exists(path_file):
            try:
                with open(path_file, encoding='utf-8') as stream:
                    saved = json.load(stream)
                self.peaks.update({k: list(v) for k, v in saved.get('peaks', {}).items()})
                self.raised.update(saved.get('raised', {}))
                self.footprints.update({k: [list(pair) for pair in v] for k, v in saved.get('footprints', {}).items()})
                self.capacity.update({k: int(v) for k, v in saved.get('capacity', {}).items() if int(v) >= 1})
            except (OSError, ValueError):
                pass

    def bound(self, limit_mb):
        """No guess made after an OOM may exceed the card it was made for.

        A raise is a guess (half as much again), it is kept in the model's file, and ``estimate``
        never returns less. Two of them took a 5,000 MB network past an 11 GB card, after which
        every trial of that experiment was "impossible" on this machine, in that run and in every
        later one, without any of them having been tried alone on an empty card.
        """
        self.limit_mb = float(limit_mb)
        self.raised = {key: int(min(value, self.limit_mb)) for key, value in self.raised.items()}

    @staticmethod
    def predict(samples, batch=None):
        """The footprint to expect at ``batch`` from measured ``[batch size, MB]`` pairs.

        A footprint is a part that does not move (the process on the card, the weights, the
        optimiser) and a part that grows with the batch. With two batch sizes measured that is a
        line; with one, only the part above the process is scaled, and never downwards.
        """
        largest = max(mb for _, mb in samples)
        sized = [(float(b), float(mb)) for b, mb in samples if b and b > 0]
        if not batch or not sized:
            return largest
        if len({b for b, _ in sized}) >= 2:
            count = len(sized)
            mean_b, mean_mb = sum(b for b, _ in sized) / count, sum(mb for _, mb in sized) / count
            slope = max(0.0, sum((b - mean_b) * (mb - mean_mb) for b, mb in sized) /
                        sum((b - mean_b) ** 2 for b, _ in sized))
            intercept = mean_mb - slope * mean_b
            above = max(mb - (intercept + slope * b) for b, mb in sized)    # every measurement stays under the line
            return max(min(mb for _, mb in sized), intercept + slope * batch + max(0.0, above))
        measured_at, footprint = max(sized, key=lambda pair: pair[1])
        fixed = min(footprint, CONTEXT_MB)
        return fixed + (footprint - fixed) * max(1.0, batch / measured_at)

    @staticmethod
    def key(method, tag):
        return f'{method}|{tag}'

    def estimate(self, method, tag, batch=None):
        key = self.key(method, tag)
        measured = getattr(self, 'footprints', {}).get(key)
        peaks = [p for p in self.peaks.get(key, []) if p and p > 0]
        if measured:
            # what trials of this experiment really held on the card after their first epoch
            base = self.predict(measured[-20:], batch)
        elif peaks:
            base = SAFETY * statistics.median(peaks[-20:]) + CONTEXT_MB
        else:
            base = DEFAULT_MB.get(method, DEFAULT_MB_OTHER)
        return int(max(base, self.raised.get(key, 0)))

    def known_tags(self, method):
        prefix = f'{method}|'
        keys = list(self.peaks) + list(self.raised) + list(getattr(self, 'footprints', {}))
        return sorted({k.split('|', 1)[1] for k in keys if k.startswith(prefix)})

    def observe_footprint(self, method, tag, footprint_mb, batch=None):
        """A trial reported its size after its first epoch: that is knowledge, and replaces a guess."""
        if not footprint_mb or footprint_mb <= 0:
            return
        key = self.key(method, tag)
        self.footprints[key].append([int(batch or 0), float(footprint_mb)])
        del self.footprints[key][:-40]
        self.raised.pop(key, None)          # a guess made after an OOM does not outlive a measurement
        self.save()

    def estimate_method(self, method):
        """The largest estimate of any known tag of ``method`` (the default table before the first trial)."""
        tags = self.known_tags(method)
        if not tags:
            return DEFAULT_MB.get(method, DEFAULT_MB_OTHER)
        return max(self.estimate(method, t) for t in tags)

    def min_estimate_method(self, method):
        """The smallest estimate of any known tag: what the cheapest trial of ``method`` may need.

        This is the honest question to ask before sampling a trial, because the tag is only known
        after ``study.ask()``.  Asking with the largest estimate instead let one oversized tag
        block every smaller tag of the same method for the whole run."""
        tags = self.known_tags(method)
        if not tags:
            return DEFAULT_MB.get(method, DEFAULT_MB_OTHER)
        return min(self.estimate(method, t) for t in tags)

    def observe(self, method, tag, peak_mb):
        if peak_mb and peak_mb > 0:
            self.peaks[self.key(method, tag)].append(float(peak_mb))
            self.save()

    def learn_capacity(self, key, trials):
        """How many trials of this kind one card was seen to keep busy; saved for the next run."""
        trials = max(1, int(trials))
        if getattr(self, 'capacity', {}).get(key) != trials:
            self.capacity[key] = trials
            self.save()
            return True
        return False

    def raise_after_oom(self, method, tag, current):
        key = self.key(method, tag)
        raised = max(self.raised.get(key, 0), current * OOM_RAISE)
        limit = getattr(self, 'limit_mb', None)
        self.raised[key] = int(min(raised, limit) if limit else raised)
        self.save()
        return self.raised[key]

    def save(self):
        try:
            os.makedirs(os.path.dirname(self.path_file), exist_ok=True)
            with open(self.path_file, 'w', encoding='utf-8') as stream:
                json.dump({'peaks': dict(self.peaks), 'raised': self.raised,
                           'footprints': dict(getattr(self, 'footprints', {})),
                           'capacity': dict(getattr(self, 'capacity', {}))}, stream, indent=1)
        except OSError as error:
            log.debug('cannot save memory estimates: %r', error)


# ---------------------------------------------------------------------------------- units of work
class StudyUnit:
    def __init__(self, static_data, cluster_name, cluster_dir, method, storage_url, refit, run_id=None):
        self.static_data = trial_layout.pin(static_data, cluster_dir, method, cluster_name)
        self.cluster_name = cluster_name
        self.cluster_dir = cluster_dir
        self.method = method
        self.refit = refit
        # the run id travels into the settlement: a trial another worker is still holding is left
        # alone, and only this run's own or expired work is recovered
        self.study, self.objective, self.path_trials = prepare_study(self.static_data, cluster_name, cluster_dir, method,
                                                                     storage_url, refit, run_id=run_id)
        self.n_trials = int(static_data[method]['n_trials'])
        self.running = 0
        self.finished = False
        self.closed = False              # no experiment of it can run on this machine

    @property
    def name(self):
        return self.study.study_name

    def remaining(self):
        """Trials that may be asked of the study now: what it lacks, while its attempts last."""
        return optuna_search.attempts(self.study, self.n_trials)['ask']

    def is_done(self):
        if self.running:
            return False
        return self.closed or self.remaining() == 0

    def ready(self, orchestrator):
        return True


class FixedUnit:
    """Transfer-learning tasks of one (cluster, method): built when the global study of the method is done."""

    def __init__(self, static_data, cluster_name, cluster_dir, method, refit, depends_on):
        self.static_data = trial_layout.pin(static_data, cluster_dir, method, cluster_name)
        self.static_data = trial_layout.pin(self.static_data, os.path.join(static_data['path_model'], 'global'), method, 'global')
        self.cluster_name = cluster_name
        self.cluster_dir = cluster_dir
        self.method = method
        self.refit = refit
        self.depends_on = depends_on     # name of the global study unit
        self.tasks = None                # discovered lazily
        self.running = 0
        self.closed = False
        self.name = optuna_search.study_name(self.static_data, cluster_name, method) + '__tl'

    def discover(self):
        # What a transfer is trained from, and which of them this cluster still lacks, is said
        # in one place: the plan a person reads before the run counts the same trials.
        sources, passed = revisions.transfer_sources(self.static_data, self.method)
        for exp_name, reason in passed:
            # A source is a trained model this unit will start from: it has to be finished work,
            # not merely a folder holding the parameters that were sampled for it.
            log.info('%s: %s is not a transfer source (%s)', self.name, exp_name, reason)
        _, wanted = revisions.transfer_tasks(self.static_data, self.cluster_dir, self.method, sources, refit=self.refit)
        self.tasks = [{'trial_name': task['trial_name'],
                       'parameters': joblib.load(os.path.join(task['source'], 'parameters.pickle')),
                       'tl_path_weights': task['source']} for task in wanted]

    def ready(self, orchestrator):
        dep = orchestrator.units.get(self.depends_on)
        # A resource-closed global unit is terminal too. Discover existing sources
        # as usual; an empty source directory settles this transfer unit immediately.
        ok = dep is None or ((dep.finished or dep.closed) and dep.running == 0)
        if ok and self.tasks is None:
            self.discover()
        return ok

    def remaining(self):
        return 0 if self.tasks is None else len(self.tasks)

    def is_done(self):
        if self.running:
            return False
        return self.closed or (self.tasks is not None and not self.tasks)


class RunningTrial:
    def __init__(self, unit, trial, path_weights, gpu, estimate_mb, attempt, params=None, structure=None, tag=None,
                 fixed_name=None):
        self.counted = False               # counted in unit.running (started, or parked after asking)
        self.unit = unit
        self.trial = trial
        self.path_weights = path_weights
        self.gpu = gpu
        self.estimate_mb = estimate_mb
        self.attempt = attempt
        self.params = params
        self.structure = structure
        self.tag = tag
        self.fixed_name = fixed_name
        self.container = None              # the container running it, once it has been launched
        self.exclusive = False             # holds its GPU alone, for as long as it runs
        self.process = None
        self.logf = None
        self.started = time.time()
        self.last_epoch = -1
        self.measured = True               # False from launch until it reports its size, when sizes are measured
        self.batch = None                  # its batch size, which is what its footprint grows with
        self.host_only = False             # keep the inputs on the host: the last attempt after an OOM

    @property
    def label(self):
        return f'{self.unit.name} {self.fixed_name or self.trial.number}'


# ``blocking``: the wait will not resolve by itself -- nothing this run can finish will free what the
# trial asks for -- so the reason is worth a warning and a remedy rather than an information line.
Admission = collections.namedtuple('Admission', 'verdict gpu reason blocking', defaults=(False,))


class Stalled(RuntimeError):
    """The run started nothing for the deadline with nothing running: reported, not a bug."""


#: The cards of a run were all lost: said by the run when it ends, in its log and its summary.
NO_CARD_LEFT = ('every card of this run was lost, so nothing more can be trained in it. A card the driver has lost '
                'comes back only when the machine is restarted; the trials that were on the cards are not counted '
                'as attempts of their studies, and the next run continues them')


class _NoTrial:
    """Stands in for an Optuna trial in fixed tasks: no pruning, no report."""

    def __init__(self, number):
        self.number = number

    def report(self, *args, **kwargs):
        pass

    def should_prune(self):
        return False

    def set_user_attr(self, *args, **kwargs):
        pass


# ---------------------------------------------------------------------------------- metrics
#: How a failed trial ended, as the metrics name it. Any other state is ``other``, so that a state a
#: later version adds cannot widen the label set unnoticed.
FAILURE_REASONS = ('error', 'oom', 'preparation', 'unschedulable', 'card_lost', 'other')
#: Failed trials listed one by one -- the newest -- for the table of the training dashboard.
RECENT_FAILURES = 20


def _label(value):
    """A label value as the exposition format wants it: backslash, quote and newline escaped."""
    return str(value).replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')


def _number(value):
    value = float(value)
    if math.isnan(value):
        return 'NaN'
    if math.isinf(value):
        return '+Inf' if value > 0 else '-Inf'
    return repr(value)


def best_error(unit):
    """The lowest objective value -- an error, the studies minimise -- of a study's complete trials,
    or None before its first. Read where ``remaining()`` reads, at each scrape."""
    study = getattr(unit, 'study', None)
    if study is None:
        return None
    try:
        values = [trial.value for trial in study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
                  if trial.value is not None]
    except Exception:          # noqa: BLE001 -- a scrape must not fail because a study could not be read
        return None
    return min(values) if values else None


def failure_reason(record):
    state = record.get('state') or 'other'
    return state if state in FAILURE_REASONS else 'other'


def unit_lines(units, outcomes, failures):
    """Each study: its best error, and the trials of this run that ended, by how they ended."""
    failed = collections.Counter(record.get('unit') for record in failures)
    best, ended = [], []
    for name, unit in units.items():
        labels = f'unit="{_label(name)}",method="{_label(unit.method)}"'
        value = best_error(unit)
        if value is not None:
            best.append(f'eforecast_unit_best_error{{{labels}}} {_number(value)}')
        for state, count in (('complete', outcomes[(name, 'complete')]), ('pruned', outcomes[(name, 'pruned')]),
                             ('failed', failed[name])):
            ended.append(f'eforecast_unit_trials_total{{{labels},state="{state}"}} {count}')
    return best + ended


def failure_lines(failures):
    """The failed trials of this run by reason, and the newest of them one by one."""
    reasons = collections.Counter(failure_reason(record) for record in failures)
    lines = [f'eforecast_trials_failed_by_reason_total{{reason="{reason}"}} {reasons[reason]}'
             for reason in FAILURE_REASONS]
    recent = dict.fromkeys((str(record.get('unit', '')), str(record.get('trial', '')), failure_reason(record))
                           for record in failures[-RECENT_FAILURES:])
    lines += [f'eforecast_trial_failed_info{{unit="{_label(unit)}",trial="{_label(trial)}",reason="{reason}"}} 1'
              for unit, trial, reason in recent]
    return lines


class Metrics:
    """Prometheus exposition of the orchestrator state on ``EFORECAST_METRICS_PORT`` (default 9108, 0 = off).

    Besides the run's counts, each study's best error and the trials of the run that ended
    (``unit_lines``), and the failures by reason (``failure_lines``): what the training dashboard
    draws, since the studies are kept in a file each and the Optuna dashboard reads only a
    database. The running trials and their error after every epoch are not here: the metrics
    service reads them from the trial processes (``eforecast/config/trial_progress.py``), which
    covers every training, whatever version of the code started it."""

    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.server = None
        port = int(os.environ.get('EFORECAST_METRICS_PORT', '9108') or 0)
        if not port:
            return
        try:
            import http.server
            import threading
            owner = self

            class Handler(http.server.BaseHTTPRequestHandler):
                def do_GET(self):
                    body = owner.render().encode()
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/plain; version=0.0.4')
                    self.send_header('Content-Length', str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)

                def log_message(self, *args):
                    pass

            self.server = http.server.ThreadingHTTPServer(('0.0.0.0', port), Handler)
            threading.Thread(target=self.server.serve_forever, daemon=True).start()
            log.info('metrics on http://localhost:%d/metrics', port)
        except OSError as error:
            log.warning('metrics endpoint not started on port %d: %r', port, error)

    def render(self):
        o = self.orchestrator
        lines = ['# TYPE eforecast_trials_running gauge', '# TYPE eforecast_trials_completed_total counter',
                 '# TYPE eforecast_trials_failed_total counter', '# TYPE eforecast_oom_retries_total counter',
                 '# TYPE eforecast_trials_waiting gauge', '# TYPE eforecast_units_open gauge',
                 '# TYPE eforecast_unit_remaining gauge', '# TYPE eforecast_gpu_memory_used_mb gauge']
        for gpu in o.gpus:
            lines.append(f'eforecast_trials_running{{gpu="{gpu}"}} {len(o.running_on(gpu))}')
            lines.append(f'eforecast_gpu_memory_used_mb{{gpu="{gpu}"}} {o.pool.used(gpu)}')
        lines.append(f'eforecast_trials_completed_total {o.completed}')
        lines.append(f'eforecast_trials_failed_total {len(o.failures)}')
        lines.append(f'eforecast_oom_retries_total {sum(o.oom_retries.values())}')
        lines.append(f'eforecast_trials_waiting {len(o.requeue)}')
        lines.append(f'eforecast_units_open {sum(1 for u in o.units.values() if not u.is_done())}')
        for name, unit in o.units.items():
            lines.append(f'eforecast_unit_remaining{{unit="{name}",method="{unit.method}"}} {unit.remaining()}')
        failures = list(o.failures)              # copies: the run goes on changing them meanwhile
        lines += ['# TYPE eforecast_unit_best_error gauge', '# TYPE eforecast_unit_trials_total counter']
        lines += unit_lines(dict(o.units), getattr(o, 'outcomes', collections.Counter()), failures)
        lines += ['# TYPE eforecast_trials_failed_by_reason_total counter', '# TYPE eforecast_trial_failed_info gauge']
        lines += failure_lines(failures)
        return '\n'.join(lines) + '\n'

    def close(self):
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()


# ---------------------------------------------------------------------------------- orchestrator
class Orchestrator:
    def __init__(self, gpus, static_data, refit=False, storage_url=None, max_per_gpu=None, halve_batch=False,
                 exclusive_methods=(), declared_per_gpu=None, begin_again=False):
        self.gpus = list(gpus)
        self.refit = refit
        # The answer to what the run asked before it started: a study that holds finished trials
        # of another configuration of its family may be emptied and trained from the beginning.
        self.begin_again = begin_again
        self.storage_url = storage_url or os.environ.get('EFORECAST_OPTUNA_STORAGE') or None
        self.max_per_gpu = max_per_gpu or declared_per_gpu or DEFAULT_MAX_PER_GPU
        self.halve_batch = halve_batch
        self.exclusive_methods = set(exclusive_methods)
        self.units = {}                       # name -> unit, in insertion order
        self.results_targets = []             # (clusters dict, method, refit) for get_results when a study ends
        self.running = []                     # RunningTrial
        self.failures = []
        self.ended_well = collections.Counter()   # (method, graph) -> trials of this run that completed or were pruned
        self.outcomes = collections.Counter()     # (unit name, 'complete' | 'pruned') -> trials of this run, for the metrics
        self.completed = 0
        self.pool = GpuPool(self.gpus)
        path_exe = os.path.join(static_data['path_model'], 'exe_files')
        self.memory = MemoryModel(os.path.join(path_exe, 'memory_estimates.json'))
        self.status_file = os.path.join(path_exe, 'orchestrator_status.json')
        self.path_logs = os.path.join(static_data['path_logs'], 'training')
        os.makedirs(self.path_logs, exist_ok=True)
        self.runner = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trial_runner.py')
        self.oom_retries = collections.Counter()
        self.pending_reserve = {}             # run -> launch time; young trials count their estimate
        self.metrics = Metrics(self)
        self.requeue = []                     # asked trials waiting for a GPU; each knows if it is exclusive
        self.run_id = f'{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:6]}'
        # trials in containers: the orchestrator builds the argv itself, so nothing is re-parsed
        self.docker_image = os.environ.get('EFORECAST_WORKER_IMAGE') or None
        self.docker_folders = json.loads(os.environ.get('EFORECAST_WORKER_FOLDERS') or '[]')
        self.docker_image_id = None
        self._gpu_resource_cache = {}      # gpu -> (expiry, reservation signature, measurement)
        # the orchestrator can serve several projects, so it keeps the two fields a container
        # needs rather than a whole static_data: the first project names the run
        self.case_id = str(static_data.get('_id', ''))
        self.n_cpus = static_data.get('n_cpus')
        # Sizes and the trials per card are measured when the cards can be measured and nobody
        # fixed the number: a bound given by the operator keeps the declared rules it was given for.
        self.measuring = self.measures(self.pool, self.gpus, max_per_gpu)
        if self.measuring:
            # what bounds the trials on a card is then the card itself; this is only the ceiling
            # the cores put on it, because every trial keeps one busy
            self.max_per_gpu = max(1, (self.cpu_budget() - RESERVED_CORES) // len(self.gpus))
        # what each card was doing at every tick: (time, busy percent, kinds on it, trials on it)
        self.card_log = {gpu: collections.deque(maxlen=60) for gpu in self.gpus}
        self.capacity_changed = {}                          # kind -> when its trials per card last changed
        self.draining = {}                                  # card -> the trial that has to be alone on it
        self.memory.bound(self.gpu_usable_mb())
        self.stop_requested = False
        self.stop_signal = None
        self.outcome = 'nothing_to_do'        # until run() says otherwise
        # A parked trial is sampled and holds a RUNNING slot in its study. With measured admission
        # the slots are a ceiling nobody expects to reach, so the queue is sized by the cards.
        self.max_parked = max(1, len(self.gpus) * (2 if self.measuring else self.max_per_gpu) * PARKED_PER_SLOT)
        self.unschedulable_trials = []        # trials this machine can never admit, with what they need
        self.unschedulable_units = []         # units closed for this run, with why
        self.cards_lost = {}                  # card -> when it was taken out of this run (``lose_card``)
        self.cards_gone = None                # why the run ended, when it ended for having no card left
        self.no_progress_seconds = NO_PROGRESS_SECONDS
        self.last_progress = time.time()      # a trial started or finished; the stall clock
        self.sharing = None                   # the MPS daemons of this run, once it has started them

    @staticmethod
    def measures(pool, gpus, max_per_gpu=None):
        """Whether this run sizes its trials and fills its cards by measurement.

        It does when every card can say how busy it is and nobody fixed the trials per card. A
        bound given by the operator (``--n-jobs``) keeps the declared rules it was written for,
        and so does a machine whose cards cannot be read.
        """
        reading = getattr(pool, 'utilization', None)
        return bool(BUSY_TARGET > 0 and not max_per_gpu and gpus and reading is not None
                    and all(reading(gpu) is not None for gpu in gpus))

    def reserved(self, run, field):
        """What this scheduler counts a running trial for: ``gpu_mb`` on its card, or ``cpu_threads``."""
        if field == 'cpu_threads':
            threads = getattr(run, 'cpu_threads', None)
            return self.worker_threads() if threads is None else threads
        return run.estimate_mb

    # ------------------------------------------------------------------ building the queue

    def budget_of(self, static_data, cluster_name, cluster_dir, method, comparability):
        """What this study has already done, or ``None`` when it cannot be read."""
        try:
            study = optuna_search.load_study(static_data, cluster_name, cluster_dir, method, self.storage_url)
            return optuna_search.budget_state(static_data, study, method, comparability)
        except Exception as error:                   # noqa: BLE001 -- an unreadable study is work to do
            log.warning('%s %s: the study could not be read for its budget (%r); it is queued',
                        cluster_name, method, error)
            return None


    def add_studies(self, static_data, method, clusters):
        """One study unit per cluster of ``clusters`` (name -> dir) for ``method``.

        A unit is skipped when its budget is already satisfied by compatible completed work --
        not because a results table happens to exist, which used to make an increased budget, an
        empty table or a table of failures all look like finished work.
        """
        added = []
        # This method's own comparability, not the case's: trials record the id of the family
        # they belong to (`trial_prep`), so asking with the case-wide id would find nothing
        # compatible and queue a full budget for a family nothing had changed.
        comparability = lineage.comparability_id(static_data, method=method)
        for cluster_name, cluster_dir in clusters.items():
            refit = self.refit
            if not self.refit:
                state = self.budget_of(static_data, cluster_name, cluster_dir, method, comparability)
                if optuna_search.begins_again(state):
                    # Its finished trials are of another configuration of this family -- its
                    # parameters or its search space changed, or its data was built again. They
                    # are not continued and nothing new is trained beside them: with the answer
                    # the run was given, the unit is emptied and its search starts over; without
                    # it the run stops here, before any study of it has been touched.
                    if not optuna_search.may_begin_again(getattr(self, 'begin_again', False), method):
                        raise optuna_search.refuse_other_configuration(cluster_name, method, state)
                    log.warning('%s %s: %d finished trial(s) are of another configuration of %s; they are removed '
                                'and the study starts from the beginning', cluster_name, method,
                                state['incompatible'], method)
                    refit = True
                elif state is not None and state['missing'] == 0:
                    log.info('skipped, %s', optuna_search.describe_budget(f'{cluster_name} {method}', state))
                    continue
            unit = StudyUnit(static_data, cluster_name, cluster_dir, method, self.storage_url, refit,
                             run_id=self.run_id)
            if unit.remaining() == 0:
                unit.finished = True
            self.units[unit.name] = unit
            added.append(unit)
        self.results_targets.append((dict(clusters), method, static_data))
        return added

    def add_transfer(self, static_data, method, clusters):
        """Transfer-learning units of ``method`` for every non-global cluster, after its global study."""
        if not static_data[method].get('transfer_learning_from_global', False):
            return []
        global_name = f"{static_data['_id']}__global__{method}"
        for unit in self.units.values():
            # A study keeps a pinned copy of the settings it was given, so it is known by the model
            # it belongs to and never by being the same object. Looked for as the same object it
            # was not found: the transfers waited for nothing, looked for their sources before the
            # first global trial of the run was trained, found none and were settled empty.
            if isinstance(unit, StudyUnit) and unit.cluster_name == 'global' and unit.method == method \
                    and unit.static_data.get('_id') == static_data.get('_id') \
                    and unit.static_data.get('path_model') == static_data.get('path_model'):
                global_name = unit.name
        added = []
        for cluster_name, cluster_dir in clusters.items():
            if cluster_name == 'global':
                continue
            unit = FixedUnit(static_data, cluster_name, cluster_dir, method, self.refit, global_name)
            self.units[unit.name] = unit
            added.append(unit)
        return added

    # ------------------------------------------------------------------ the loop

    def run(self):
        self.report_children()
        if self.docker_image:
            self.docker_image_id = docker_task.image_identity(self.docker_image)
            log.info('trials run in %s (%s), %.2f cpus each, mounts %s', self.docker_image,
                     self.docker_image_id or 'unknown image id', self.docker_cpus(), self.docker_folders)
            docker_task.reconcile(self.run_id)
        log.info('orchestrator run %s: %d units on gpus %s (%s MB%s), max %d trials per gpu',
                 self.run_id, len(self.units), self.gpus,
                 {g: self.pool.total[g] for g in self.gpus},
                 ', declared, NVML unavailable' if self.pool.declared_total else '', self.max_per_gpu)
        last_status = 0
        failure = None
        try:
            self.start_sharing()
            with self.stop_on_signal():
                try:
                    while not self.stop_requested:
                        self.drop_lost_cards()
                        self.reap()
                        if getattr(self, 'cards_lost', None) and not self.gpus:
                            # nothing can be started any more: the trials of the lost cards are
                            # given their moment to end, so that each is told why, and the run ends
                            if self.cards_are_gone():
                                break
                            time.sleep(TICK_SECONDS)
                            continue
                        self.heartbeat()
                        self.forward_all()
                        self.sample_busy()
                        self.observe_running()
                        self.admit_requeued()
                        self.admit()
                        if all(unit.is_done() for unit in self.units.values()) and not self.running:
                            break
                        if self.stalled():
                            raise Stalled(f'the orchestrator started nothing for {self.stall_seconds():.0f} s '
                                          f'with nothing running: {self.stall_diagnosis()}')
                        if time.time() - last_status > STATUS_EVERY:
                            self.report_status()
                            last_status = time.time()
                        time.sleep(TICK_SECONDS)
                except KeyboardInterrupt:          # a second one, while the first is being handled
                    self.stop_requested = True
        except BaseException as error:             # noqa: BLE001 -- re-raised below, after cleanup
            failure = error
        self.finalize(failure)
        if failure is not None:
            raise failure
        return self.outcome

    def finalize(self, failure=None):
        """Settle, release and report, whatever ended the loop.

        The cleanup used to sit after the loop, so it ran only when the loop ended the way it was
        meant to. An ordinary error -- an unreadable progress file, a storage hiccup in
        ``forward_all`` -- skipped **all** of it: asked trials stayed RUNNING in their studies, the
        metrics server stayed up and no summary was
        written. Every step here is guarded on its own, because a failure in one must not hide the
        exception that brought us here, nor stop the rest of the cleanup.
        """
        if isinstance(failure, Stalled):
            self.outcome = 'failed'
        elif failure is not None:
            self.outcome = 'failed'
            log.error('the orchestrator run ended on an unexpected error: %r', failure, exc_info=failure)
        for step in (self.cancel if self.stop_requested else self.settle_abandoned,
                     self.stop_containers, self.finish_results_safely, self.report_status):
            try:
                step()
            except Exception as error:             # noqa: BLE001 -- cleanup never replaces the cause
                log.error('cleanup step %s failed: %r', getattr(step, '__name__', step), error)
        if failure is None:
            self.outcome = self.compute_outcome()
        for step in (self.write_summary, self.metrics.close, self.stop_sharing):
            try:
                step()
            except Exception as error:
                log.error('cleanup step %s failed: %r', getattr(step, '__name__', step), error)
        log.info('orchestrator %s (exit %d): %d trials completed, %d failures%s%s', self.outcome, self.exit_code,
                 self.completed, len(self.failures),
                 f', {len(self.unschedulable_trials)} trials this machine cannot run'
                 if self.unschedulable_trials else '',
                 f', {len(self.unschedulable_units)} units skipped: '
                 f"{', '.join(u['unit'] for u in self.unschedulable_units)}"
                 if self.unschedulable_units else '')

    def start_sharing(self):
        """Let the trials of one card run their kernels together (NVIDIA MPS), where that can be had.

        Only for cards that take more than one trial, that have a name of their own, and for
        trials that are processes of this machine: see ``gpu_sharing``. A run goes on without it
        wherever it cannot be started.
        """
        if self.docker_image or self.max_per_gpu <= 1 or not gpu_sharing.wanted():
            return
        cards = {gpu: self.pool.key(gpu) for gpu in self.gpus
                 if gpu in getattr(self.pool, 'handles', {}) and str(self.pool.key(gpu)).startswith('GPU-')}
        self.sharing = gpu_sharing.Sharing(cards)
        self.sharing.start()

    def stop_sharing(self):
        if getattr(self, 'sharing', None) is not None:
            self.sharing.stop()

    def sharing_lost(self, run):
        """A trial failed because its card's MPS daemon went: the card goes on without it.

        Every trial on the card fails the same way at the same moment, so each is given one more
        start, which finds the card no longer shared. Returns False when the trial has had it.
        """
        self.sharing.drop(run.gpu, f'{run.label} lost its connection to the MPS server')
        if getattr(run, 'sharing_retried', False):
            return False
        run.sharing_retried = True
        run.attempt += 1
        run.unit.running += 1
        run.counted = True
        self.requeue.append(run)
        return True

    # ------------------------------------------------------------------ a card that is lost

    def drop_lost_cards(self):
        """Take the cards the driver has lost out of this run (asked at every tick)."""
        check = getattr(self.pool, 'check', None)
        if check is None or getattr(self, 'cards_lost', None) is None:
            return
        for gpu in sorted(check()):
            if gpu in self.gpus:
                self.lose_card(gpu, 'the driver reports it lost: it fell off the bus or stopped answering')

    def lose_card(self, gpu, why):
        """``gpu`` takes no more trials in this run, and the ones on it are stopped.

        A lost card does not come back before the machine is restarted, and a trial started on
        it dies in seconds. Until 2026-11-23 the scheduler counted such a card by its own
        reservations and went on filling it: 39 trials in five minutes on 19 November, 45 in
        two and a half on the 23rd, each a failed attempt of its study.
        """
        if gpu not in self.gpus:
            return
        self.gpus.remove(gpu)
        self.cards_lost[gpu] = time.time()
        on_it = self.running_on(gpu)
        log.error('gpu %s is taken out of this run: %s. Nothing more is started on it, and the %d trial(s) on it '
                  'are stopped; they are not counted as attempts of their studies. %d card(s) left%s', gpu, why,
                  len(on_it), len(self.gpus), '' if self.gpus else ': the run ends when those trials have ended')
        for run in on_it:
            with contextlib.suppress(ProcessLookupError, OSError):
                run.process.terminate()
        if getattr(self, 'docker_image', None) and on_it:
            # ending the client leaves its container running, attached to nothing
            with contextlib.suppress(Exception):
                docker_task.stop([self.container_of(run) for run in on_it])
        sharing = getattr(self, 'sharing', None)
        if sharing is not None:
            with contextlib.suppress(Exception):   # the daemon of a card that is gone may be gone with it
                sharing.drop(gpu, 'the card was lost')
        getattr(self, 'draining', {}).pop(gpu, None)
        self.card_changed(gpu)

    def cards_are_gone(self):
        """Whether a run without a card left can end: its trials have ended, or have had their time to."""
        if self.running and time.time() - max(self.cards_lost.values()) < CANCEL_GRACE_SECONDS:
            return False
        self.cards_gone = NO_CARD_LEFT
        log.error('This run ends: %s.', NO_CARD_LEFT)
        return True

    def trained_an_epoch(self, run):
        """Whether this start of the trial got as far as reporting an epoch."""
        return (self.read_progress(run) or {}).get('epoch') is not None

    def its_card_failed(self, run, result):
        """Whether a trial that ended in error was ended by its card or the driver, not by itself.

        It was when the driver has lost the card -- asked now, since the trial can die before
        the next reading. And when a card of this machine is already lost, a trial that dies of
        a driver error or before its first epoch is taken for the same event: on 23 November
        2026 the trial on the other card died of "CUDA driver error" eighty seconds after card
        0 fell off the bus, and every trial started there after it died within four seconds.
        Its card is then taken out too. An error of the trial's own -- after it has trained,
        and not the driver's -- stays its own.
        """
        lost, gpu = getattr(self, 'cards_lost', None), run.gpu
        if lost is None or gpu is None:
            return False
        if gpu not in lost:
            self.drop_lost_cards()
        if gpu in lost:
            return True
        if not lost and not getattr(self.pool, 'lost_others', None):
            return False
        if card_error(result.get('error')) or not self.trained_an_epoch(run):
            self.lose_card(gpu, f'{run.label} ended in error on it after a card of this machine was lost, and a '
                                f'driver that has lost a card is not trusted with the others')
            return True
        return False

    def lost_with_its_card(self, run, result):
        """Settle a trial its card ended: told FAIL as ``card_lost``, which is not an attempt of its study."""
        error = ' '.join(str(result.get('error') or '').split())[:300]
        log.error('%s ends: its card was lost (%s). It is not counted as an attempt of its study', run.label,
                  error[:160] or 'no error text')
        if run.fixed_name is None:
            try:
                run.trial.set_user_attr('fail_reason', 'card_lost')
                tell_once(run.unit.study, run.trial, state=TrialState.FAIL)
            except Exception as told:          # noqa: BLE001 -- the next run settles what this one could not
                log.error('cannot settle %s in its study (%r)', run.label, told)
        self.failures.append({'unit': run.unit.name, 'trial': run.fixed_name or run.trial.number, 'state': 'card_lost',
                              'error': error, 'path': run.path_weights, 'method': run.unit.method,
                              'graph': getattr(run, 'tag', None)})

    def settle_abandoned(self):
        """A closed unit, or a run that ended on an error, can leave asked trials behind."""
        self.stop_workers()
        for run in list(self.running):
            self.settle(run, 'abandoned')
        self.running = []
        self.settle_asked_trials('abandoned')

    def stop_containers(self):
        """Nothing of this run may still be in a container once the run is over."""
        if self.docker_image:
            docker_task.stop([run.container for run in self.running if getattr(run, 'container', None)])
            docker_task.reconcile(self.run_id)

    def finish_results_safely(self):
        if not self.stop_requested:
            self.finish_results()

    # ------------------------------------------------------------------ stopping

    @contextlib.contextmanager
    def stop_on_signal(self):
        """SIGINT and SIGTERM ask the loop to stop; the handlers are put back afterwards.

        The web UI stops a run by sending SIGTERM to the whole process group, so the trials get it
        too -- but the orchestrator must still settle their studies and write its summary, which an
        unhandled SIGTERM would skip entirely.
        """
        def request(signum, _frame):
            if self.stop_requested:                # a second signal: let the default behaviour win
                signal.signal(signum, signal.SIG_DFL)
                return
            self.stop_requested = True
            self.stop_signal = signal.Signals(signum).name
            log.warning('%s received: stopping, %d trials running and %d parked',
                        self.stop_signal, len(self.running), len(self.requeue))

        previous = {}
        for signum in (signal.SIGINT, signal.SIGTERM):
            try:
                previous[signum] = signal.signal(signum, request)
            except ValueError:                     # not the main thread: the caller handles stopping
                log.debug('cannot install a handler for %s outside the main thread', signum)
        try:
            yield
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)

    def cancel(self):
        """Stop the trials in flight and settle every trial that was asked.

        A terminated trial writes no result, so without this its study keeps a RUNNING trial that
        no process will ever finish, and the next run counts it as work in progress for ever.
        """
        self.stop_workers()
        log.warning('cancelled: %d running trials stopped, %d parked trials dropped',
                    len(self.running), len(self.requeue))
        for run in list(self.running):
            self.settle(run, 'cancelled')
        self.running = []
        self.settle_asked_trials('cancelled')

    def stop_workers(self):
        """Stop before releasing: finalization after an error has the same duty as cancellation."""
        for run in self.running:
            try:
                run.process.terminate()
            except ProcessLookupError:
                pass  # the child exited between the previous tick and shutdown
        deadline = time.time() + CANCEL_GRACE_SECONDS
        for run in self.running:
            remaining = max(0.0, deadline - time.time())
            try:
                run.process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                log.warning('%s did not stop within %ds: killing it', run.label, CANCEL_GRACE_SECONDS)
                run.process.kill()
                try:
                    run.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    log.error('%s survived SIGKILL; its process is left behind', run.label)
            if run.logf is not None and not run.logf.closed:
                run.logf.close()
        if self.docker_image and self.running:
            # killing the client leaves the container running, attached to nothing and holding a GPU
            docker_task.stop([self.container_of(run) for run in self.running])

    def _settlement_record(self, run, state, error=''):
        record = getattr(run, '_settlement_record', None)
        if record is None:
            record = {'unit': run.unit.name, 'trial': run.fixed_name or run.trial.number,
                      'path': run.path_weights}
            run._settlement_record = record
            self.failures.append(record)
        record.update(state=state, error=error)

    def settle(self, run, reason):
        """Tell the study that a trial which was asked will not finish, and why."""
        unit = run.unit
        if run.counted:
            unit.running -= 1
            run.counted = False
        self.pending_reserve.pop(run, None)
        if run.fixed_name is None:
            try:
                run.trial.set_user_attr('fail_reason', reason)
                tell_once(unit.study, run.trial, state=TrialState.FAIL)
            except Exception as error:
                log.error('cannot settle %s in its study (%r)', run.label, error)
        self._settlement_record(run, reason)

    def settle_asked_trials(self, reason):
        for run in self.requeue:
            self.settle(run, reason)
        self.requeue = []

    def compute_outcome(self):
        """What this run did, in one word.  ``nothing_to_do`` and the successes exit 0."""
        if self.stop_requested:
            return 'cancelled'
        if getattr(self, 'cards_gone', None):
            # whatever it completed before: its work is not done and cannot be continued here
            return 'failed'
        if self.completed:
            return 'succeeded_with_failures' if (self.failures or self.unschedulable_units) else 'succeeded'
        # nothing completed: a run with nothing to do is not a failed run, one that lost every
        # trial or skipped every unit is -- and used to exit 0 all the same
        if self.failures or self.unschedulable_units or self.unschedulable_trials:
            return 'failed'
        return 'nothing_to_do'

    @property
    def exit_code(self):
        return EXIT_CODES.get(self.outcome, 1)

    # ------------------------------------------------------------------ admission

    def free_mb(self, gpu):
        """Measured occupancy plus reserved growth; unknown attribution reserves the full claim."""
        return self.gpu_resources(gpu)['free_mb']

    def cpu_budget(self):
        """The cores the trials of this run may use: the case's ``n_cpus``, never more than the machine."""
        return int(min(getattr(self, 'n_cpus', None) or resources.cpu_slots(), resources.cpu_slots()))

    def worker_threads(self):
        """Threads for the next trial.

        Declared rules: the budget split over every slot there could ever be, so a trial alone on
        the machine still got one sixth of it. Measured: the budget split over the trials that
        are running, bounded by what a trial can use and by what is not already taken. A trial is
        charged for what it then really uses (``observe_running``), which is about one core.
        """
        budget = self.cpu_budget()
        if not getattr(self, 'measuring', False):
            return max(1, int(budget / max(1, len(self.gpus) * self.max_per_gpu)))
        running = getattr(self, 'running', [])
        taken = sum(self.reserved(run, 'cpu_threads') for run in running)
        return int(max(1, min(MAX_TRIAL_THREADS, budget // (len(running) + 1), budget - taken)))

    @staticmethod
    def worker_pids(run):
        if getattr(run, 'container', None):
            # docker run's client is not the container's process tree on the host.
            if time.time() - getattr(run, '_pid_checked', 0) > 2:
                run._host_pid = docker_task.host_pid(run.container)
                run._pid_checked = time.time()
            return [run._host_pid] if getattr(run, '_host_pid', None) else []
        process = getattr(run, 'process', None)
        return [process.pid] if process is not None else []

    def gpu_resources(self, gpu):
        """Cached for a fraction of a tick: ``pick_gpu`` asks once per candidate, per card.

        The measurement reads the memory of every process of every worker. ``admit`` calls
        ``pick_gpu`` twice per candidate unit and ``pick_gpu`` calls this per card, so a run with
        fourteen open units on two cards with four trials each spent about 2.7 s of a 3 s tick
        measuring rather than scheduling. The cache is keyed on what the answer depends on --
        the trials of this run on the card and what each is counted for -- so a trial that
        starts or is reaped invalidates it at once.
        """
        signature = tuple(sorted((id(run), self.reserved(run, 'gpu_mb')) for run in self.running_on(gpu)))
        cached = self._gpu_resource_cache.get(gpu)
        if cached is not None and cached[0] > time.time() and cached[1] == signature:
            return cached[2]
        value = self.measure_gpu_resources(gpu)
        self._gpu_resource_cache[gpu] = (time.time() + GPU_RESOURCE_TTL, signature, value)
        return value

    def measure_gpu_resources(self, gpu):
        """What is free on a card: its size, less what is used, less what our trials may still grow by.

        ``used`` is what the device reports, whoever holds it -- a desktop session, another
        program. A trial of this run is counted for the larger of what it holds now and what
        it was started for, so that one which has not filled its share yet is not started over.
        """
        used = self.pool.used(gpu)
        query = getattr(self.pool, 'process_memory', None)
        by_pid = query(gpu) if query else None
        reservations = []
        for run in self.running_on(gpu):
            pids = resources.process_tree(self.worker_pids(run))['pids']
            reservations.append((self.reserved(run, 'gpu_mb'), sum((by_pid or {}).get(pid, 0) for pid in pids)))
        charged = resources.memory_charge(used, reservations)
        planned = sum(value for value, _ in reservations)
        # Without NVML nothing is known of what else is on the card.
        free = self.pool.total[gpu] - (planned if charged is None else charged) - HEADROOM_MB
        return {'measured_used_mb': used, 'reserved_mb': planned, 'charged_mb': charged,
                'free_mb': free, 'attribution': 'per_pid' if by_pid is not None else 'unknown_conservative',
                'telemetry': 'measured' if used is not None else 'unavailable'}

    def running_on(self, gpu):
        return [run for run in self.running if run.gpu == gpu]

    def gpu_usable_mb(self):
        """The most a single trial can ever reserve on this machine: the biggest card, less headroom."""
        return max((self.pool.total[gpu] - HEADROOM_MB for gpu in self.gpus), default=0)

    def admission(self, estimate_mb, exclusive=False, kind=None):
        """``Admission(verdict, gpu, reason)`` with verdict ``ok`` / ``wait`` / ``impossible``.

        ``wait`` means a running trial will eventually free what this one needs on a card.
        ``impossible`` means the run has no card at all. It is never the answer to a size: a
        trial expected to need more than the largest card is tried alone on an empty one
        (``sized``). Only the cards are asked: host memory and processor time are the operating
        system's to give. The one thing kept of the host is what it has shown: after the kernel
        killed a trial, fewer run at once (``killed_for_memory``)."""
        if not self.gpus:
            return Admission('impossible', None, 'no GPU was selected for this run')
        limit = getattr(self, 'host_limit', None)
        if limit is not None and len(self.running) >= limit:
            return Admission('wait', None, f'the machine ran out of memory with more trials running; '
                                           f'this run runs at most {limit} at a time now')
        estimate_mb, exclusive = self.sized(estimate_mb, exclusive)
        gpu = self.pick_gpu(estimate_mb, exclusive=exclusive, **({'kind': kind} if kind else {}))
        if gpu is None:
            busiest = ', '.join(f'gpu{g}:{self.free_mb(g):.0f}MB free,{len(self.running_on(g))} running'
                                + (f',{self.busy(g):.0f}% busy' if getattr(self, 'measuring', False)
                                   and self.busy(g) is not None else '')
                                for g in self.gpus)
            counted = getattr(self, 'full_by_count', {})
            if counted and all(g in counted or self.free_mb(g) < estimate_mb for g in self.gpus):
                # The memory is there. What holds the trial is how many of its kind a card takes,
                # and the message said "no GPU with 3476 MB free" beside cards with 5 GB free.
                limit = min(counted.values())
                return Admission('wait', None, f'each card already runs the {limit} trial(s) of this kind it is '
                                 f'known to take; one more is tried once a card stays under {BUSY_TARGET:.0f}% busy '
                                 f'over a whole trial with that many ({busiest})')
            return Admission('wait', None, f'no GPU with {estimate_mb:.0f} MB free ({busiest})')
        return Admission('ok', gpu, '')

    def sized(self, estimate_mb, exclusive=False):
        """``(what to ask a card for, whether the trial has to be alone on it)``.

        An estimate above the largest card is not a verdict. It is a prediction: a line through
        what other trials of the same experiment held at their batch sizes, or half as much
        again as an earlier peak, or a table. What a trial really needs also depends on the
        layer sizes it drew, which no estimate looks at. On 22 November 2026 the experiment
        ``mlp2`` trained at a batch of 471 in seven studies; from the next morning its estimate
        at 408 to 501 was 11.6 to 12.5 GB on an 11 GB card, and ten draws in a row of the study
        that had done best there were failed without being tried. So such a trial is asked for
        a whole card, alone, and what it does on it decides (``try_alone``, ``retry_after_oom``).
        """
        usable = self.gpu_usable_mb()
        if self.gpus and estimate_mb > usable:
            return usable, True
        return estimate_mb, exclusive

    def try_alone(self, run, expected_mb):
        """Mark a trial expected to need more than a card: it starts alone, with all an empty card has."""
        usable = self.gpu_usable_mb()
        log.warning('%s is expected to need %d MB and the largest card offers %d MB usable. That is an estimate '
                    'from other trials of its experiment, so it is tried alone on an empty card%s',
                    run.label, expected_mb, usable,
                    '; if it runs out of memory there it is tried again, at last with half its batch'
                    if getattr(self, 'halve_batch', False) else '; if it runs out of memory there it is tried once more')
        run.expected_mb = int(expected_mb)     # kept: until it has measured itself it shares its card with nobody
        run.estimate_mb = int(usable)
        if getattr(run, 'prior_mb', None):
            run.prior_mb = int(usable)
        run.exclusive = True
        with contextlib.suppress(Exception):    # a note on the trial; a study that cannot take it changes nothing
            run.trial.set_user_attr('memory_expected_mb', int(expected_mb))

    def kind_of(self, run):
        """The kind of network a trial is: its method and experiment. Sizes and capacities are learned per kind."""
        return self.memory.key(run.unit.method, getattr(run, 'tag', None))

    def needs_a_card(self, run):
        """Whether this trial can only start on an empty card: a retry after an OOM, or a kind not yet shared."""
        if run.exclusive:
            return True
        return bool(getattr(self, 'measuring', False)) and self.trials_per_card(self.kind_of(run)) <= 1

    def trials_per_card(self, kind):
        """How many trials of ``kind`` one card takes: what was learned, and one until something was."""
        return int(getattr(self.memory, 'capacity', {}).get(kind, 1)) if kind else 1

    def busy(self, gpu, seconds=BUSY_WINDOW_SECONDS):
        """The card's average busy share over the last ``seconds``, or ``None`` when it was not read that long."""
        log_ = getattr(self, 'card_log', {}).get(gpu) or ()
        cutoff = time.time() - seconds
        recent = [entry[1] for entry in log_ if entry[0] >= cutoff]
        if not recent or not log_ or log_[0][0] > cutoff + TICK_SECONDS:
            return None
        return sum(recent) / len(recent)

    def sample_busy(self):
        """Read every card, and note for each trial what its card was doing while it lived."""
        if not getattr(self, 'measuring', False):
            return
        now = time.time()
        for gpu in self.gpus:
            value = self.pool.utilization(gpu)
            on_gpu = self.running_on(gpu)
            kinds = {self.kind_of(run) for run in on_gpu}
            if value is None:
                continue
            self.card_log[gpu].append((now, value, tuple(sorted(kinds)), len(on_gpu)))
            alone_in_kind = len(kinds) == 1 and not any(run.exclusive for run in on_gpu)
            for run in on_gpu:
                seen = run.__dict__.setdefault('card_seen', {'busy': 0.0, 'trials': 0, 'ticks': 0, 'mixed': False})
                seen['busy'] += value
                seen['trials'] += len(on_gpu)
                seen['ticks'] += 1
                seen['mixed'] = seen['mixed'] or not alone_in_kind
                # A trial that runs for an hour should not keep its kind waiting an hour to be
                # understood. Once it has trained ten times as long as it took to start, its
                # start no longer weighs on the average, and it can say what it has to say.
                started_after = seen.get('first_report')
                if started_after and not seen.get('taught') and seen['ticks'] >= max(MIN_LIFE_TICKS, 10 * started_after):
                    seen['taught'] = True
                    self.learn_from(run)

    def learn_from(self, run):
        """A finished trial says how busy its card was, on average, with so many of its kind on it.

        The average is over the trial's whole life -- starting up, training, evaluating -- with
        the card holding only its kind and about as many of them as a card is believed to take.
        Under the target, a card takes one more of that kind. A convolutional network that
        spends three minutes loading and forty filling the card reads 93% and stays alone; a
        small MLP that spends half its short life off the card reads 45% with two of them, and
        gets a third. The number only grows: a card that reads full is at its capacity, and a
        card that is full is what was wanted.
        """
        seen = getattr(run, 'card_seen', None)
        if not getattr(self, 'measuring', False) or not seen or seen['mixed'] or seen['ticks'] < MIN_LIFE_TICKS:
            return
        seen['taught'] = True
        kind = self.kind_of(run)
        capacity = self.trials_per_card(kind)
        if run.started < getattr(self, 'capacity_changed', {}).get(kind, 0.0):
            return                          # it lived partly under the previous number
        trials, busy = seen['trials'] / seen['ticks'], seen['busy'] / seen['ticks']
        if trials < capacity - 0.5 or busy >= BUSY_TARGET or capacity >= self.max_per_gpu:
            return
        if self.memory.learn_capacity(kind, capacity + 1):
            self.capacity_changed[kind] = time.time()
            log.info('%s: a card with %.1f of them was busy %.0f%% of the time over a whole trial, so a card '
                     'takes %d', kind, trials, busy, capacity + 1)

    def card_changed(self, gpu):
        """A trial on ``gpu`` started or left: what was reserved there is read again."""
        if getattr(self, 'measuring', False):
            self._gpu_resource_cache.pop(gpu, None)

    def pick_gpu(self, estimate_mb, exclusive=False, kind=None):
        best, best_rank = None, None
        measuring = getattr(self, 'measuring', False)
        draining = getattr(self, 'draining', {})
        self.full_by_count = {}                 # cards passed over for the number of trials on them
        for gpu in self.gpus:
            on_gpu = self.running_on(gpu)
            if exclusive and on_gpu:
                continue
            if len(on_gpu) >= self.max_per_gpu:
                continue
            # ``run.exclusive`` and not the method: an OOM retry is given a card to itself, and used
            # to lose it the moment the next trial looked for somewhere to go, so it OOMed again on
            # its last attempt. Exclusivity is a property of the running trial, not of its method.
            if any(run.exclusive for run in on_gpu):
                continue
            if not exclusive and gpu in draining and draining[gpu] is not getattr(self, 'placing', None):
                # Kept clear of new arrivals for a trial that has to be alone -- and that trial is
                # the one arrival it is kept for. Without the exception a card emptied for a waiting
                # trial was refused to that very trial, and stood idle until another card emptied
                # too: trials of a kind that fills a card started in pairs, and the card that
                # finished first waited for the other (54 s of every 133 s on the scratch case).
                continue
            if measuring and on_gpu:
                # A card takes as many trials as the least sociable kind on it allows: a kind
                # nobody has measured is alone, a convolutional network that fills a card is
                # alone, and small networks of a kind found to share one are grouped.
                kinds = {self.kind_of(run) for run in on_gpu} | {kind}
                takes = min(self.trials_per_card(k) for k in kinds)
                if len(on_gpu) >= takes:
                    self.full_by_count[gpu] = takes
                    continue
            free = self.free_mb(gpu)
            if free < estimate_mb and not (measuring and exclusive and not on_gpu) \
                    and not self.empty_enough(gpu, on_gpu, free, estimate_mb):
                # A trial that has to be alone is given a card, not a number. Its number is a guess
                # made after an OOM, half as much again, and a card carries things that are not
                # ours -- a desktop session takes 600 to 1100 MB here. A retry "needing at least
                # 10948 MB" beside 10353 MB free on an empty card waited for memory that nothing
                # would ever free; with four of them parked, nothing else was started either.
                continue
            # the card with the fewest trials first, then the one with the most memory free
            rank = (len(on_gpu) if measuring else 0, -free)
            if best is None or rank < best_rank:
                best, best_rank = gpu, rank
        return best

    def empty_enough(self, gpu, on_gpu, free, estimate_mb=0):
        """Whether ``gpu`` is as empty as waiting will ever make it, for a trial that fits the card.

        The estimate of a trial that fits a card can still be more than an empty card has free:
        a kind was measured at 10148 MB, a card with nothing of ours on it had 9922 MB free beside
        a desktop session, and every further trial of that kind waited for 226 MB that nothing
        this run could finish would free -- one card idle, and then the whole run. A card none
        of our trials is on, with nearly all of its memory free, takes such a trial with what
        there is; if that is not enough the trial says so (out of memory) and is retried. A
        card somebody else has taken a real part of is not empty, and is waited for.
        """
        usable = self.pool.total[gpu] - HEADROOM_MB
        return not on_gpu and estimate_mb <= usable and free >= EMPTY_CARD_SHARE * usable

    def candidates(self):
        """Ready units with remaining work: the biggest backlog of every method, methods in rotation."""
        by_method = collections.defaultdict(list)
        for unit in self.units.values():
            if unit.is_done() or not unit.ready(self):
                continue
            remaining = unit.remaining()
            if remaining > 0:
                by_method[unit.method].append((remaining, unit))
        if getattr(self, 'measuring', False):
            # Largest first: a card that has just emptied goes to the biggest network waiting,
            # which could not start beside anything, and the small ones take what is left of the
            # cards after it. The other order strands the big ones behind a stream of small ones.
            ready = [(self.unit_estimates(unit)[1], remaining, unit)
                     for queue in by_method.values() for remaining, unit in queue]
            return [unit for _, _, unit in sorted(ready, key=lambda item: (-item[0], -item[1]))]
        ordered = []
        queues = {m: sorted(v, key=lambda x: -x[0]) for m, v in by_method.items()}
        while any(queues.values()):
            for method in list(queues):
                if queues[method]:
                    ordered.append(queues[method].pop(0)[1])
        return ordered

    def admit(self):
        for unit in self.candidates():
            if len(self.running) >= len(self.gpus) * self.max_per_gpu:
                return
            if len(self.requeue) >= self.max_parked:
                # Every parked trial has been sampled, has its folder and its parameters.pickle, and
                # holds a RUNNING slot in its study.  Without this bound a tick on a full machine
                # parked one trial of every unit, every three seconds.
                return
            if isinstance(unit, StudyUnit):
                self.launch_study_trial(unit)
            else:
                self.launch_fixed(unit)

    def launch_study_trial(self, unit):
        # The question before sampling is whether the cheapest experiment of this unit could start,
        # because the tag -- and with it the real estimate -- is only known after ``ask()``.
        if not self.gpus:
            # nothing can be started, so nothing is asked: a trial asked for here could only be failed
            self.close_unit(unit, 'no GPU was selected for this run')
            return
        # A cheapest experiment estimated above every card is tried alone, like any trial of
        # that size (``sized``): the unit is closed by what its trials do, not by the estimate.
        cheapest, alone = self.sized(self.unit_estimates(unit)[0])
        if self.pick_gpu(cheapest, exclusive=alone, **self.unit_kind(unit)) is None:
            # Cards that keep taking small trials are never empty. A unit whose every kind needs
            # an empty card is allowed one trial in the queue, which is what gets a card cleared.
            wants_alone = alone or (getattr(self, 'measuring', False) and
                                    self.trials_per_card(self.unit_kind(unit).get('kind')) <= 1)
            if not wants_alone or unit.running or any(self.needs_a_card(run) for run in self.requeue):
                return
        trial = unit.study.ask()
        # from here the trial is RUNNING in the study: take the lease before anything can fail, so
        # another worker never sees it as unowned work to recover
        leases.claim(trial, self.run_id)
        try:
            path_weights, params, structure, tag, experiment_params = prepare_trial(
                unit.static_data, unit.study, trial, unit.objective, unit.cluster_dir, unit.method, unit.refit, 0)
        except Exception as error:  # noqa: BLE001
            # The trial is already asked, so an escaping exception ended the whole run and left it
            # RUNNING in the study for the next run to settle.  One trial's preparation is one
            # trial's failure; a deterministic fault will fail the next ones the same way and the
            # study's ask budget bounds that.
            self.preparation_failed(unit, trial, error)
            return
        batch = self.batch_of(experiment_params)
        estimate = self.estimate_of(unit.method, tag, batch)
        run = RunningTrial(unit, trial, path_weights, gpu=None, estimate_mb=estimate, attempt=0, params=params,
                           structure=structure, tag=tag)
        run.batch = batch
        run.exclusive = unit.method in self.exclusive_methods
        admission = self.place(run)
        if admission.verdict == 'impossible':
            self.unschedulable(run, admission.reason)
            return
        if admission.verdict == 'wait':
            unit.running += 1          # counted as running while it waits, so the unit is not over-asked
            run.counted = True
            self.requeue.append(run)
            (log.warning if admission.blocking else log.info)('%s waits: %s', run.label, admission.reason)
            return

    def known_argument(self, run):
        """The kind of this trial, for an admission that fills cards by kind."""
        return {'kind': self.kind_of(run)} if getattr(self, 'measuring', False) else {}

    @staticmethod
    def batch_of(parameters):
        try:
            return int(parameters.get('batch_size'))
        except (AttributeError, TypeError, ValueError):
            return None

    def estimate_of(self, method, tag, batch=None):
        """What a trial is expected to hold on a card: a measurement of its kind, else the table."""
        if getattr(self, 'measuring', False):
            return self.memory.estimate(method, tag, batch)
        return self.memory.estimate(method, tag)

    def claim_card(self, run):
        """A trial starts with everything that is free on its card, and gives most of it back.

        Its estimate decided that the card is worth trying; it does not decide how much the
        trial may use. Capping a trial at an estimate fails the ones the estimate is wrong about,
        and the estimate is keyed by experiment while the batch size and the layer sizes are
        sampled. So the trial is admitted with what is there, measures itself in its first
        epoch, lowers its own limit to that and reports it (``observe_running``). Nothing else
        is admitted to the card in between, so nothing is promised twice.
        """
        if not getattr(self, 'measuring', False):
            return
        run.prior_mb = getattr(run, 'prior_mb', None) or run.estimate_mb
        free = self.free_mb(run.gpu)
        # whether it has the card to itself: what a retry after an OOM can still be given depends on it
        run.alone = not [other for other in self.running_on(run.gpu) if other is not run]
        if run.alone:
            # the card is what there is: a retry that has to be alone, or a trial sized above what
            # an empty card has free beside what is not ours (``empty_enough``)
            run.prior_mb = min(run.prior_mb, max(free, 0))
        if self.kind_known(run):
            # its kind has been measured: room over that, so that others can start beside it
            free = min(free, max(run.prior_mb * KNOWN_ROOM, run.prior_mb + KNOWN_MARGIN_MB))
        run.estimate_mb = int(max(run.prior_mb, free))
        run.measured = False

    def kind_known(self, run):
        """Whether trials of this experiment have reported their size before, and this is not a retry."""
        measured = getattr(self.memory, 'footprints', {}).get(self.kind_of(run))
        return bool(measured) and not getattr(run, 'attempt', 0) and not run.exclusive

    def unit_kind(self, unit):
        """The most sociable kind this unit can sample, for the question asked before sampling.

        The experiment is only known after the study is asked, so the question beforehand is
        whether its most sociable kind could start; a trial that turns out to be another kind
        waits in the queue for a card that takes it.
        """
        if not getattr(self, 'measuring', False):
            return {}
        from eforecast.config.validation import parameter_values
        try:
            tags = [tag for tag in parameter_values(unit.static_data[unit.method].get('experiment_tag'))
                    if isinstance(tag, str) and tag]
        except Exception:               # noqa: BLE001 -- an unreadable declaration is one unknown kind
            tags = []
        kinds = [self.memory.key(unit.method, tag) for tag in tags] or [self.memory.key(unit.method, None)]
        return {'kind': max(kinds, key=self.trials_per_card)}

    def place(self, run):
        """Start the trial on a card if one takes it now; otherwise say whether waiting can help."""
        run_reporting.check_cancelled()
        self.report_trial(run)
        self.placing = run              # so a card being cleared for this trial is offered to it
        try:
            expected = getattr(run, 'prior_mb', None) or run.estimate_mb
            if self.gpus and expected > self.gpu_usable_mb():
                self.try_alone(run, expected)
            admission = self.admission(getattr(run, 'prior_mb', None) or run.estimate_mb,
                                       exclusive=run.exclusive, **self.known_argument(run))
            if admission.verdict != 'ok':
                return admission
            run.gpu = admission.gpu
            self.claim_card(run)
            try:
                self.start(run)
            except Exception as error:
                self.failed_launch(run, error)
                return Admission('failed', None, f'worker launch failed: {error}', blocking=True)
            return admission
        finally:
            self.placing = None

    def failed_launch(self, run, error):
        """A trial whose process could not be started: stop what there is of it and fail the trial."""
        log.error('%s: worker launch failed (%r)', run.label, error)
        try:
            if run.process is not None and run.process.poll() is None:
                run.process.kill()
                run.process.wait(timeout=5)
            if self.docker_image and run.container:
                docker_task.stop([run.container])
        except Exception as cleanup_error:  # keep the original launch error
            log.error('%s: launch cleanup incomplete (%r)', run.label, cleanup_error)
        if run.logf is not None and not run.logf.closed:
            with contextlib.suppress(OSError):
                run.logf.close()
        if run in self.running:
            self.running.remove(run)
            self.card_changed(run.gpu)
        self.settle(run, 'launch')

    @staticmethod
    def trial_name(run):
        return str(run.fixed_name if run.fixed_name is not None else f'test_{run.trial.number}')

    def unit_estimates(self, unit):
        """``(cheapest, dearest)`` estimate over the experiment tags this unit can sample.

        An unseen tag is worth the default of its method, so a method with one huge observed tag and
        others never tried is not held back by it. The cheapest is what is asked of the cards
        before a trial is sampled; estimated above every card, it means a card alone (``sized``)."""
        from eforecast.config.validation import parameter_values
        try:
            declared = [tag for tag in parameter_values(unit.static_data[unit.method].get('experiment_tag'))
                        if isinstance(tag, str) and tag]
        except Exception:  # noqa: BLE001 -- an unreadable declaration must not block the unit
            declared = []
        # Profiles of retired tags or other units cannot make this unit sample
        # them. Only its declared candidates decide its cheapest possible trial.
        tags = sorted(set(declared))
        if not tags:
            default = DEFAULT_MB.get(unit.method, DEFAULT_MB_OTHER)
            return default, default
        estimates = [self.memory.estimate(unit.method, tag) for tag in tags]
        return min(estimates), max(estimates)

    def close_unit(self, unit, reason):
        """This unit gets no more trials in this run: close it and let the rest of the run proceed.

        For a unit that has shown it does not fit this machine -- its trials ran out of memory
        with a card to themselves, ``OOM_STREAK_CLOSES`` in a row -- and for a run without a
        card. Never for an estimate: until 2026-11-23 a unit whose cheapest experiment was
        estimated above every card was closed before anything of it had been tried. The unit is
        recorded in the summary; the next run tries it again."""
        log.error('%s cannot run on this machine and is skipped: %s', unit.name, reason)
        unit.closed = True
        cheapest, dearest = self.unit_estimates(unit)
        self.unschedulable_units.append({'unit': unit.name, 'method': unit.method, 'reason': reason,
                                         'cheapest_experiment_mb': int(cheapest),
                                         'dearest_experiment_mb': int(dearest)})

    def preparation_failed(self, unit, trial, error):
        """Settle a trial whose preparation raised: it is asked, so something must tell the study."""
        text = f'{type(error).__name__}: {error}'
        log.error('%s trial %s: preparation failed, telling FAIL (%s)', unit.name, trial.number, text,
                  exc_info=True)
        trial.set_user_attr('fail_reason', 'preparation')
        tell_once(unit.study, trial, state=TrialState.FAIL)
        self.failures.append({'unit': unit.name, 'trial': trial.number, 'state': 'preparation',
                              'error': text[:300], 'path': ''})
        if unit.remaining() == 0 and unit.running == 0 and not unit.finished:
            unit.finished = True
            self.study_finished(unit)

    def unschedulable(self, run, reason):
        """A trial that cannot be started in this run at all -- it has no card: fail it, do not queue it."""
        unit = run.unit
        self.report_trial(run, outcome={'kind': 'failure', 'detail': reason})
        log.error('%s cannot run here and will not be retried: %s', run.label, reason)
        if run.counted:
            unit.running -= 1
            run.counted = False
        if run.fixed_name is None:
            run.trial.set_user_attr('fail_reason', 'unschedulable')
            run.trial.set_user_attr('required_mb', int(run.estimate_mb))
            tell_once(unit.study, run.trial, state=TrialState.FAIL)
        self.failures.append({'unit': unit.name, 'trial': run.fixed_name or run.trial.number,
                              'state': 'unschedulable', 'error': reason[:300], 'path': run.path_weights})
        self.unschedulable_trials.append({'unit': unit.name, 'trial': run.fixed_name or run.trial.number,
                                          'required_gpu_mb': int(run.estimate_mb), 'reason': reason})
        # only a study can finish here: a fixed unit has no study and no results table, exactly as in
        # ``handle_result``, which returns before this block for a fixed task
        if run.fixed_name is None and unit.remaining() == 0 and unit.running == 0 and not unit.finished:
            unit.finished = True
            self.study_finished(unit)

    def launch_fixed(self, unit):
        task = unit.tasks[0]
        params = task['parameters']
        tag = params.get('experiment_tag', '')
        estimate = self.estimate_of(unit.method, tag, self.batch_of(params))
        exclusive = unit.method in self.exclusive_methods
        kind = {'kind': self.memory.key(unit.method, tag)} if getattr(self, 'measuring', False) else {}
        admission = self.admission(estimate, exclusive=exclusive, **kind)   # is it worth preparing at all
        if admission.verdict == 'impossible':
            # a fixed task is not sampled, so it is dropped from the queue instead of told FAIL;
            # leaving it in place would keep the unit open for the rest of the run
            unit.tasks.pop(0)
            run = RunningTrial(unit, _NoTrial(task['trial_name']), str(trial_layout.trial_directory(unit.static_data, unit.cluster_dir, unit.method, task['trial_name'])), None, estimate, attempt=0, tag=tag,
                               fixed_name=task['trial_name'])
            self.unschedulable(run, admission.reason)
            return
        if admission.verdict == 'wait':
            return
        unit.tasks.pop(0)
        path_weights = str(trial_layout.trial_directory(unit.static_data, unit.cluster_dir, unit.method, task['trial_name']))
        if os.path.exists(path_weights) and unit.refit:
            shutil.rmtree(path_weights)
        os.makedirs(path_weights, exist_ok=True)
        parameters = dict(params)
        parameters.update({'trial_number': task['trial_name'], 'name': unit.cluster_name,
                           'cluster_dir': unit.cluster_dir, 'tl_path_weights': task['tl_path_weights'],
                           'cluster': {'cluster_name': unit.cluster_name, 'cluster_path': unit.cluster_dir}})
        joblib.dump(parameters, os.path.join(path_weights, 'parameters.pickle'))
        run = RunningTrial(unit, _NoTrial(task['trial_name']), path_weights, None, estimate, attempt=0, tag=tag,
                           fixed_name=task['trial_name'])
        run.batch = self.batch_of(params)
        run.exclusive = exclusive
        admission = self.place(run)
        if admission.verdict == 'impossible':
            self.unschedulable(run, admission.reason)
            return
        if admission.verdict == 'wait':
            # Keep this exact attempt: its folder and its parameters are written.
            unit.running += 1
            run.counted = True
            self.requeue.append(run)
            (log.warning if admission.blocking else log.info)('%s waits: %s', run.label, admission.reason)

    def trial_command(self, run):
        """The argv of one trial: a container, a template, or this interpreter."""
        if self.docker_image:
            # the name is kept on the run: launch and cleanup must not each derive it again
            run.container = self.container_of(run)
            # not self.runner: that is this repository's path on the host, and the repository is
            # not mounted into the container -- only the model tree is
            return docker_task.trial_argv(
                self.worker_path(run), self.gpu_selection(run.gpu), refit=run.unit.refit, image=self.docker_image,
                folders=self.docker_folders, run_id=self.run_id,
                case=self.case_id, trial=str(run.fixed_name or run.trial.number),
                attempt=run.attempt, task=run.unit.name, name=run.container,
                image_id=getattr(self, 'docker_image_id', None))
        template = os.environ.get('EFORECAST_TRIAL_CMD')
        refit = '--refit' if run.unit.refit else ''
        if template:
            # the escape hatch, kept: a template is a string, so it is split on whitespace and a
            # path with a space in it is still two arguments. --docker does not go through here.
            return shlex.split(template.format(path_weights=self.worker_path(run), gpu=self.gpu_selection(run.gpu), refit=refit))
        command = [sys.executable, self.runner, '--path-weights', self.worker_path(run), '--gpu', '0']
        if refit:
            command.append(refit)
        return command

    @staticmethod
    def worker_path(run):
        """Where a trial works: its own folder, which is also where its result stays."""
        return run.path_weights

    def docker_cpus(self):
        """CPU share of one trial: the case's budget over the trials that may run at once."""
        return float(self.worker_threads())

    def container_of(self, run):
        """The container of this attempt: the one launched, or the name it would be given."""
        if getattr(run, 'container', None):
            return run.container
        return docker_task.container_name(self.run_id, str(run.fixed_name or run.trial.number),
                                          run.attempt, run.unit.name)

    def gpu_selection(self, gpu):
        pool = getattr(self, 'pool', None)
        return pool.selection(gpu) if hasattr(pool, 'selection') else str(gpu)

    def start(self, run):
        run_reporting.check_cancelled()
        self.report_trial(run)
        fraction = min(0.95, max(MIN_FREE_FRACTION, run.estimate_mb / self.pool.total[run.gpu]))
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=self.gpu_selection(run.gpu), EFORECAST_MEM_FRACTION=f'{fraction:.3f}',
                   EFORECAST_RUN_ID=self.run_id, EFORECAST_TASK_ID=run.unit.name, EFORECAST_ATTEMPT=str(run.attempt))
        if run_reporting.context() is not None:
            root = run_reporting.node_id('orchestrator', self.run_id)
            env.update(EFORECAST_MANAGED_STUDY_NODE=run_reporting.node_id('study', f'{self.run_id}/{run.unit.name}', root),
                       EFORECAST_MANAGED_TRIAL_KEY=f'{self.run_id}/{run.unit.name}/{self.trial_name(run)}/{run.attempt}')
        run.cpu_threads = self.worker_threads()
        env.update(resources.thread_environment(run.cpu_threads))
        from eforecast import execution_policy
        # The card is named by its UUID, and the trial sees that card only. A card that cannot be
        # named that way (no NVML) is still the card this trial was placed on: it is given by the
        # entry of the visible-device mask that selects it, and the trial trains on it without
        # being able to check which card that is. Left out, the trial would train on the CPU.
        key = self.pool.key(run.gpu)
        card = key if str(key).startswith(('GPU-', 'MIG-')) else self.gpu_selection(run.gpu)
        run.execution_policy = execution_policy.resolve(run.cpu_threads, card, {'device': run.gpu})
        env.update(execution_policy.environment(run.execution_policy))
        env.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')
        if getattr(self, 'measuring', False):
            env['EFORECAST_GPU_SETTLE'] = '1'          # lower your own limit to what the first epoch used
        if getattr(run, 'host_only', False):
            env['EFORECAST_BATCHES_DEVICE_SHARE'] = '0'   # the whole card for the network, the inputs from the host
        sharing = getattr(self, 'sharing', None)
        shared = sharing.environment(run.gpu) if sharing is not None else {}
        env.update(shared)
        run.shared = bool(shared)                         # this start goes through the card's MPS daemon
        command = self.trial_command(run)
        self.report_trial(run)  # includes the intended labelled container name before any launch
        run.logf = open(os.path.join(run.path_weights, 'train.log'), 'a', encoding='utf-8')
        run.logf.write(f"# {time.strftime('%Y-%m-%d %H:%M:%S')} gpu {run.gpu} attempt {run.attempt} "
                       f"estimate {run.estimate_mb} MB fraction {fraction:.2f}: {' '.join(command)}\n")
        run.logf.flush()
        for stale in ('progress.json', 'prune.flag'):
            if os.path.exists(os.path.join(self.worker_path(run), stale)):
                os.remove(os.path.join(self.worker_path(run), stale))
        if self.docker_image:
            run._container_create_requested = True
            self.report_trial(run)  # an absent name after this point is an uncertain Docker request
        run.process = subprocess.Popen(command, stdout=run.logf, stderr=subprocess.STDOUT, env=env, cwd=path_pycharm)
        self.report_trial(run, attached=True)
        run.started = time.time()
        self.last_progress = run.started
        if not run.counted:
            run.unit.running += 1
        run.counted = True
        run.trial.set_user_attr('gpu', run.gpu)
        run.trial.set_user_attr('run_id', self.run_id)       # which run asked for this trial
        run.trial.set_user_attr('memory_estimate_mb', run.estimate_mb)
        self.running.append(run)
        self.pending_reserve[run] = time.time()
        log.info('%s starts on gpu %d (attempt %d, %d MB reserved, %d running there)%s', run.label, run.gpu,
                 run.attempt, run.estimate_mb, len(self.running_on(run.gpu)),
                 f': {run.params}' if run.params else '')

    # ------------------------------------------------------------------ progress and completion

    def heartbeat(self):
        """Refresh the lease of every trial this run holds: the ones running and the ones parked.

        A parked trial is asked, so it is RUNNING in its study and looks exactly like abandoned
        work to another machine; it needs the heartbeat as much as a trial on a GPU does.
        """
        for run in list(self.running) + list(self.requeue):
            if run.fixed_name is None:
                try:
                    leases.beat(run.trial, self.run_id, attempt=run.attempt)
                except Exception as error:  # noqa: BLE001 -- a storage hiccup must not end the run
                    log.debug('cannot refresh the lease of %s (%r)', run.label, error)

    def read_progress(self, run):
        try:
            with open(os.path.join(self.worker_path(run), 'progress.json'), encoding='utf-8') as stream:
                progress = json.load(stream)
            return progress if isinstance(progress, dict) else None
        except (OSError, ValueError):
            return None

    def footprint_mb(self, run, progress):
        """What this trial holds on its card now that it has trained an epoch, or ``None``.

        The trial's own limit plus what the process costs outside it (the CUDA context and the
        libraries, which the device reports and the allocator does not). A trial that reported no
        limit -- a worker image older than this scheduler -- is taken at what the device says it
        holds, with more room, because nothing in that process enforces it.
        """
        query = getattr(self.pool, 'process_memory', None)
        by_pid = query(run.gpu) if query else None
        held = None
        if by_pid is not None:
            pids = resources.process_tree(self.worker_pids(run))['pids']
            held = sum(by_pid.get(pid, 0) for pid in pids) or None
        if progress.get('gpu_cap_mb'):
            pool_mb = float(progress.get('gpu_reserved_mb') or 0)
            outside = CONTEXT_MB if held is None else min(1500.0, max(100.0, held - pool_mb))
            return float(progress['gpu_cap_mb']) + outside
        if held is not None:
            return held * (1 + SETTLE_GROWTH) + 64
        if progress.get('gpu_peak_mb'):
            return float(progress['gpu_peak_mb']) * (1 + SETTLE_GROWTH) + CONTEXT_MB
        return None

    def resize_run(self, run, **changes):
        """Change what a running trial is counted for: ``gpu_mb`` and ``cpu_threads``."""
        if 'gpu_mb' in changes:
            run.estimate_mb = changes['gpu_mb']
        if 'cpu_threads' in changes:
            run.cpu_threads = changes['cpu_threads']
        self._gpu_resource_cache.pop(run.gpu, None)
        return True

    def observe_running(self):
        """Replace the reservation of each trial that has trained an epoch by what it measured.

        Until then a trial holds everything that was free on its card when it started, and the
        card takes nobody else. Afterwards it holds its footprint and is charged the cores it
        keeps busy, and the card is judged again by how busy it is.
        """
        if not getattr(self, 'measuring', False):
            return
        changed = False
        for run in self.running:
            if getattr(run, 'measured', True):
                continue
            progress = self.read_progress(run)
            if not progress:
                continue
            footprint = self.footprint_mb(run, progress)
            if footprint is None:
                continue
            admitted = run.estimate_mb
            changes = {'gpu_mb': int(min(admitted, footprint))}
            elapsed = float(progress.get('elapsed') or 0)
            if progress.get('cpu_seconds') and elapsed > 0:
                # averaged over the whole start: loading is one thread, and so is the training loop
                cores = float(progress['cpu_seconds']) / elapsed
                granted = self.reserved(run, 'cpu_threads')
                changes['cpu_threads'] = int(max(1, min(granted, math.ceil(cores - 0.2))))
            if not self.resize_run(run, **changes):
                continue
            run.measured = True
            changed = True
            if getattr(run, 'expected_mb', None) and not getattr(run, 'attempt', 0) and run.exclusive \
                    and progress.get('gpu_cap_mb') and run.unit.method not in self.exclusive_methods:
                # It was alone for what it was expected to need. Now it has said what it does
                # need and holds itself to that (its own limit), so its card can take others again.
                run.exclusive = False
                log.info('%s was expected to need %d MB and holds %d MB: its card is no longer kept for it alone',
                         run.label, run.expected_mb, changes['gpu_mb'])
            self.card_changed(run.gpu)
            if getattr(run, 'card_seen', None):
                run.card_seen['first_report'] = max(1, run.card_seen['ticks'])   # how long it took to start
            if progress.get('gpu_cap_mb'):      # a size the trial enforces on itself is worth learning
                self.memory.observe_footprint(run.unit.method, run.tag, changes['gpu_mb'], getattr(run, 'batch', None))
            log.info('%s measured after epoch %s: %d MB on gpu %d (admitted with %d MB), %s core(s)',
                     run.label, progress.get('epoch'), changes['gpu_mb'], run.gpu, admitted,
                     changes.get('cpu_threads', '?'))

    def forward_all(self):
        for run in self.running:
            if run.fixed_name is None:
                run.last_epoch = forward_progress(run.trial, os.path.join(self.worker_path(run), 'progress.json'),
                                                  run.last_epoch, self.worker_path(run))

    def reap(self):
        for run in list(self.running):
            code = run.process.poll()
            if code is None:
                continue
            self.report_trial(run, outcome={'kind': 'success' if code == 0 else 'failure',
                                           'returncode': code, 'detail': 'observed trial process exit'})
            self.running.remove(run)
            self.pending_reserve.pop(run, None)
            self.card_changed(run.gpu)
            run.unit.running -= 1
            run.counted = False
            run.logf.close()
            if run.fixed_name is None:
                run.last_epoch = forward_progress(run.trial, os.path.join(self.worker_path(run), 'progress.json'),
                                                  run.last_epoch, self.worker_path(run))
            result = read_result(self.worker_path(run), code)
            self.last_progress = time.time()
            if result.get('state') in ('complete', 'pruned') and not (getattr(run, 'card_seen', None) or {}).get('taught'):
                self.learn_from(run)
            if result.get('state') == 'error' and self.its_card_failed(run, result):
                self.lost_with_its_card(run, result)
                continue
            if code in KILLED and result.get('state') == 'error' and self.killed_for_memory(run):
                continue
            if (result.get('state') == 'error' and getattr(run, 'shared', False)
                    and gpu_sharing.lost(result.get('error')) and self.sharing_lost(run)):
                continue
            self.handle_result(run, result)

    def killed_for_memory(self, run):
        """A trial the kernel killed: start fewer at once from now on, and give the trial another go.

        Nothing is set aside on the host for a trial, so that the machine cannot hold what was
        running is found out by its happening: Linux ends the largest process. From then on this
        run starts one trial fewer at a time than were running when it happened -- a bound it has
        seen, not one it estimated -- and the trial waits its turn again. Returns False when the
        trial has had its attempts; it is then failed, with the reason.
        """
        were = len(self.running) + 1
        self.host_limit = max(1, min(getattr(self, 'host_limit', None) or were, were) - 1)
        again = run.attempt < MAX_OOM_RETRIES
        log.warning('%s was killed (SIGKILL) with %d trial(s) running. When nobody stopped it, that is the kernel '
                    'ending the largest process because the machine ran out of memory: this run now runs at most '
                    '%d trial(s) at a time%s', run.label, were, self.host_limit,
                    ', and the trial is tried again' if again else '')
        if not again:
            return False
        run.attempt += 1
        self.oom_retries[run.unit.name] += 1
        run.unit.running += 1
        run.counted = True
        self.requeue.append(run)
        return True

    def count_outcome(self, unit, state):
        """One trial of ``unit`` that ended ``complete`` or ``pruned`` in this run, for the metrics."""
        if not hasattr(self, 'outcomes'):         # an orchestrator made without __init__, as tests make them
            self.outcomes = collections.Counter()
        self.outcomes[(unit.name, state)] += 1

    def handle_result(self, run, result):
        """Tell the study how a trial ended, or give it another attempt after an out-of-memory error."""
        unit, state = run.unit, result.get('state')
        duration = result.get('duration') or (time.time() - run.started)
        if state in ('complete', 'pruned'):
            self.memory.observe(unit.method, run.tag, result.get('peak_memory_mb'))
            self.ended_well[(unit.method, run.tag)] += 1
            unit.oom_streak = 0
        if state == 'oom' and run.attempt < MAX_OOM_RETRIES and self.worth_retrying(run):
            self.retry_after_oom(run, result)
            return
        if state == 'oom':
            # it has had a card to itself, and half its batch where that is allowed: this is known
            unit.oom_streak = getattr(unit, 'oom_streak', 0) + 1
        if run.fixed_name is not None:
            if state != 'complete':
                self.failures.append({'unit': unit.name, 'trial': run.fixed_name, 'state': state,
                                      'error': (result.get('error') or '')[:300], 'path': run.path_weights,
                                      'method': unit.method, 'graph': run.tag})
            else:
                self.completed += 1
                self.count_outcome(unit, 'complete')
            log.info('%s ends: %s value=%s in %.0fs', run.label, state, result.get('value'), duration)
            return
        if state == 'complete':
            if tell_once(unit.study, run.trial, result['value']):
                record_trial(unit.path_trials, run.trial.number, run.params, run.structure, result)
                self.completed += 1
                self.count_outcome(unit, 'complete')
        elif state == 'pruned':
            if tell_once(unit.study, run.trial, state=TrialState.PRUNED):
                self.completed += 1
                self.count_outcome(unit, 'pruned')
        else:
            run.trial.set_user_attr('fail_reason', state)
            tell_once(unit.study, run.trial, state=TrialState.FAIL)
            self.failures.append({'unit': unit.name, 'trial': run.trial.number, 'state': state,
                                  'error': (result.get('error') or '')[:300], 'path': run.path_weights,
                                  'method': unit.method, 'graph': run.tag})
        log.info('%s ends: %s value=%s in %.0fs%s', run.label, state, result.get('value'), duration,
                 f" ({(result.get('error') or '')[:160]})" if state in ('error', 'oom') else '')
        if getattr(unit, 'oom_streak', 0) >= OOM_STREAK_CLOSES and not getattr(unit, 'closed', False):
            self.close_unit(unit, f'{unit.oom_streak} of its trials in a row ran out of memory with a card to '
                                  f'themselves{" and half their batch" if getattr(self, "halve_batch", False) else ""}')
        if unit.remaining() == 0 and unit.running == 0 and not unit.finished:
            unit.finished = True
            self.study_finished(unit)

    def worth_retrying(self, run):
        """Whether another attempt can be given anything the failed one did not have.

        Alone on a card, with its inputs kept on the host, is the most a card can do for a trial.
        One that ran out of memory like that is too large for this machine at its batch size, and
        a third attempt under the same conditions only spends the time again -- unless the batch
        is to be halved for it.
        """
        if not getattr(self, 'measuring', False) or self.halve_batch:
            return True
        return not (getattr(run, 'alone', False) and getattr(run, 'host_only', False))

    def retry_after_oom(self, run, result):
        unit = run.unit
        usable = self.gpu_usable_mb()
        failed_with = run.estimate_mb
        run.attempt += 1
        self.oom_retries[unit.name] += 1
        if getattr(self, 'measuring', False):
            # The trial had everything that was free on a card it shared, so the only thing left
            # to give it is a card to itself -- and, on the last attempt, the card without its
            # inputs resident on it. What it held when it failed plus what it then asked for is
            # a lower bound on its size, and is what decides whether a card can hold it at all.
            evidence = float(result.get('peak_memory_mb') or 0) + float(result.get('oom_requested_mb') or 0)
            prior = getattr(run, 'prior_mb', None) or failed_with
            raised = int(min(usable, max(prior * OOM_RAISE, evidence + CONTEXT_MB if evidence else 0)))
            run.prior_mb = run.estimate_mb = raised
            run.measured = True             # not running: it is measured again when it starts
            exclusive = True
            # A trial that already had a card to itself gains nothing from being alone again:
            # the next thing to give it is the card without its inputs on it.
            run.host_only = run.attempt >= MAX_OOM_RETRIES or bool(getattr(run, 'alone', False))
        else:
            # Never past the card: a raise is a guess, and a guess larger than every card turned
            # the last attempt -- alone, on an empty card -- into "impossible" without trying it.
            raised = int(min(usable, self.memory.raise_after_oom(unit.method, run.tag, run.estimate_mb)))
            run.estimate_mb = raised
            exclusive = run.attempt >= MAX_OOM_RETRIES
        last = run.attempt >= MAX_OOM_RETRIES
        if last and self.halve_batch:
            params_file = os.path.join(run.path_weights, 'parameters.pickle')
            parameters = joblib.load(params_file)
            new_batch = max(8, int(parameters.get('batch_size', 256)) // 2)
            parameters['batch_size'] = new_batch
            joblib.dump(parameters, params_file)
            run.trial.set_user_attr('batch_size_effective', new_batch)
        needing = ('with everything an empty card has' if exclusive and getattr(self, 'measuring', False)
                   else f'needing at least {raised} MB')
        log.warning('%s: CUDA OOM on gpu %s with %d MB (attempt %d); retry %s, %s%s',
                    run.label, run.gpu, failed_with, run.attempt - 1,
                    'alone on a GPU' if exclusive else 'later', needing,
                    ', its inputs kept on the host' if getattr(run, 'host_only', False) else '')
        run.unit.running += 1
        run.counted = True
        # a trial expected to need more than a card stays alone, whatever the retry would give it
        run.exclusive = exclusive or unit.method in self.exclusive_methods or bool(getattr(run, 'expected_mb', None))
        self.requeue.append(run)

    def study_finished(self, unit):
        counts = trial_counts(unit.study)
        log.info('study %s finished: complete %d, pruned %d, failed %d', unit.name, counts['complete'],
                 counts['pruned'], counts['fail'])
        try:
            get_results({unit.cluster_name: unit.cluster_dir}, unit.method, unit.refit, static_data=unit.static_data)
        except Exception as error:  # noqa: BLE001
            log.error('get_results failed for %s: %r', unit.name, error)

    def finish_results(self):
        """Results tables for clusters whose trials came only from fixed tasks (transfer learning)."""
        for entry in self.results_targets:
            clusters, method = entry[:2]
            static = entry[2] if len(entry) > 2 else {}
            try:
                get_results(clusters, method, False, static_data=static)
            except Exception as error:  # noqa: BLE001
                log.error('get_results failed for %s: %r', method, error)

    # ------------------------------------------------------------------ retries of waiting trials

    def admit_requeued(self):
        # A trial that has to be alone needs an empty card, and a card that keeps taking small
        # trials is never empty. So each one waiting is given a card that takes no new arrivals
        # until it has started: the card with the fewest trials left to finish.
        measuring = getattr(self, 'measuring', False)
        alone = [run for run in self.requeue if self.needs_a_card(run)]
        emptiest = sorted(self.gpus, key=lambda gpu: len(self.running_on(gpu)))
        # with learned capacities most kinds share a card, so one card at a time is cleared
        self.draining = dict(zip(emptiest[:1] if measuring else emptiest, alone))
        if measuring:
            # those first, then the largest: what is waiting takes a freed card in that order
            self.requeue.sort(key=lambda run: (not self.needs_a_card(run),
                                               -(getattr(run, 'prior_mb', None) or run.estimate_mb)))
        kept = []
        for run in self.requeue:
            admission = self.place(run)
            if admission.verdict == 'impossible':
                # this is where a trial used to wait for the rest of the run: an OOM retry whose
                # raised estimate had grown past the card, or a tag that never fitted
                self.unschedulable(run, admission.reason)
                continue
            if admission.verdict == 'wait':
                kept.append(run)
                continue
        self.requeue = kept
        self.draining = {gpu: run for gpu, run in self.draining.items() if run in kept}

    # ------------------------------------------------------------------ no progress

    def stall_seconds(self):
        return time.time() - self.last_progress

    def stalled(self):
        """Nothing running and nothing started for the deadline: the run cannot get itself out.

        With nothing running the cards and the RAM are as free as this run can make them, so a trial
        that still does not fit is waiting on something outside the run.  An impossible trial is
        failed rather than parked, so it is never the reason."""
        if not self.no_progress_seconds or self.running:
            return False
        return self.stall_seconds() > self.no_progress_seconds

    def stall_diagnosis(self):
        """What the open units are waiting for, in one line."""
        parts = []
        for name, unit in self.units.items():
            if unit.is_done():
                continue
            if not unit.ready(self):
                parts.append(f'{name} waits for {getattr(unit, "depends_on", "a dependency")}')
                continue
            cheapest, dearest = self.unit_estimates(unit)
            parts.append(f'{name} needs {cheapest:.0f} MB (its experiments: {cheapest:.0f}-{dearest:.0f} MB)')
        gpus = ', '.join(f'gpu{g}: {self.free_mb(g):.0f} MB free of {self.pool.total[g]} MB, '
                         f'{self.pool.used_text(g)} used by someone' for g in self.gpus)
        if not self.pool.telemetry:
            parts.append('NVML cannot be read, so what else is on the cards is unknown and the card size '
                         f'is the declared {GPU_TOTAL_MB} MB (EFORECAST_GPU_TOTAL_MB)')
        return f'{len(self.requeue)} parked; {gpus}; ' + '; '.join(parts[:7])

    # ------------------------------------------------------------------ reporting

    def report_status(self):
        status = {'run_id': self.run_id, 'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'completed': self.completed,
                  'failures': len(self.failures), 'gpus': {}, 'units': {}}
        for gpu in self.gpus:
            status['gpus'][gpu] = {'used_mb': self.pool.used(gpu), 'total_mb': self.pool.total[gpu],
                                   'running': [run.label for run in self.running_on(gpu)]}
            if getattr(self, 'measuring', False):
                status['gpus'][gpu].update(
                    busy_percent=self.busy(gpu), reserved_mb=sum(self.reserved(r, 'gpu_mb') for r in self.running_on(gpu)),
                    finding_their_size=[r.label for r in self.running_on(gpu) if not getattr(r, 'measured', True)])
        for name, unit in self.units.items():
            status['units'][name] = {'remaining': unit.remaining(), 'running': unit.running, 'done': unit.is_done()}
        status['parked'] = len(self.requeue)
        status['admission'] = ('measured: a card takes one more trial of a kind while whole trials of that kind '
                               f'kept it busy less than {BUSY_TARGET:g}% of the time' if getattr(self, 'measuring', False)
                               else f'declared: at most {self.max_per_gpu} trials per card, sized by estimate')
        if getattr(self, 'measuring', False):
            status['trials_per_card'] = dict(getattr(self.memory, 'capacity', {}))
        status['unschedulable'] = {'trials': self.unschedulable_trials, 'units': self.unschedulable_units}
        status['cards_lost'] = sorted(getattr(self, 'cards_lost', {}))
        status['idle_seconds'] = round(self.stall_seconds()) if not self.running else 0
        status['gpu_telemetry'] = self.pool.telemetry
        status['stop_requested'] = self.stop_requested
        status['resources'] = self.resource_status()
        resources.write_status(status['resources'])
        try:
            with open(self.status_file, 'w', encoding='utf-8') as stream:
                json.dump(status, stream, indent=1)
        except OSError:
            pass
        waiting = sum(1 for u in self.units.values() if not u.is_done())
        log.info('status: %d running (%s), %d units open, %d trials completed, %d failures, parked %d%s',
                 len(self.running), ', '.join(f"gpu{g}:{len(self.running_on(g))}/{self.pool.used_text(g)}"
                                              for g in self.gpus), waiting, self.completed, len(self.failures),
                 len(self.requeue),
                 f', nothing started for {self.stall_seconds():.0f}s' if not self.running else '')
        if not self.running and self.stall_seconds() > STATUS_EVERY:
            log.warning('nothing is running: %s', self.stall_diagnosis())

    def resource_status(self):
        """What this run is doing on the cards and the cores, for the run's resource file."""
        return {'run_id': self.run_id, 'phase': self.outcome if not self.running else 'running',
                'cpu': resources.cpu_inventory(),
                'running': {'trials': len(self.running),
                            'cpu_threads': sum(self.reserved(r, 'cpu_threads') for r in self.running),
                            'worker_threads': self.worker_threads(),
                            # set once the kernel has killed a trial of this run for memory
                            'at_most_trials': getattr(self, 'host_limit', None)},
                'observed': resources.process_tree([pid for r in self.running for pid in self.worker_pids(r)]),
                'devices': self.pool.inventory(),
                'execution_policies': {r.label: r.execution_policy for r in self.running
                                       if hasattr(r, 'execution_policy')},
                'gpus': {str(g): self.gpu_resources(g) for g in self.gpus}}

    def report_children(self, finished=False):
        if run_reporting.context() is None:
            return None
        kind = {'succeeded': 'success', 'succeeded_with_failures': 'partial', 'failed': 'failure',
                'cancelled': 'cancelled', 'nothing_to_do': 'nothing_to_do'}.get(self.outcome, 'failure')
        parent = run_reporting.child('orchestrator', self.run_id,
            identity={'orchestrator_run_id': self.run_id}, execution=run_reporting.process(),
            outcome={'kind': kind} if finished else None)
        for unit in self.units.values():
            run_reporting.child('study', f'{self.run_id}/{unit.name}', parent=parent,
                identity={'study_name': unit.name, 'cluster': unit.cluster_name, 'method': unit.method,
                          'cluster_dir': unit.cluster_dir}, outcome={'kind': kind} if finished else None)
        return parent

    def report_trial(self, run, *, attached=False, outcome=None):
        if run_reporting.context() is None:
            return
        root = run_reporting.node_id('orchestrator', self.run_id)
        parent = run_reporting.child('study', f'{self.run_id}/{run.unit.name}', parent=root,
            identity={'study_name': run.unit.name, 'cluster': run.unit.cluster_name,
                      'method': run.unit.method, 'cluster_dir': run.unit.cluster_dir})
        key = f'{self.run_id}/{run.unit.name}/{self.trial_name(run)}/{run.attempt}'
        execution = None
        if attached and run.process is not None:
            from eforecast.config.executors import process_evidence
            process = process_evidence(run.process.pid)
            process.pop('pgid', None)
            execution = {'process': process}
        trial = run_reporting.child('trial', key, parent=parent,
            identity={'trial': self.trial_name(run), 'attempt': run.attempt, 'path_weights': run.path_weights},
            execution=execution, outcome=outcome)
        if getattr(run, 'container', None):
            exact_id = getattr(run, 'container_id', None)
            identity = {'name': run.container, 'orchestrator_run_id': self.run_id,
                        'task': run.unit.name, 'attempt': run.attempt, 'provenance': 'labelled_name'}
            # A foreground docker run exiting zero confirms completion. A failed or lost
            # client without an exact ID does not prove that its container stopped.
            terminal = outcome if exact_id or (outcome and outcome.get('returncode') == 0) else None
            execution = {'container_id': exact_id} if exact_id else {}
            if getattr(run, '_container_create_requested', False):
                execution['create_requested'] = True
            run_reporting.child('container', run.container, parent=trial, identity=identity,
                execution=execution or None, outcome=terminal)

    def write_summary(self):
        self.report_children(finished=True)
        summary = {'run_id': self.run_id, 'outcome': self.outcome, 'exit_code': self.exit_code,
                   'stopped_by': self.stop_signal, 'image': self.docker_image,
                   'image_id': self.docker_image_id,
                   'gpu_telemetry': self.pool.telemetry, 'gpu_total_declared': self.pool.declared_total,
                   'completed': self.completed, 'failures': self.failures, 'oom_retries': dict(self.oom_retries),
                   'unschedulable': {'trials': self.unschedulable_trials, 'units': self.unschedulable_units},
                   'cards_lost': {str(gpu): time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(when))
                                  for gpu, when in getattr(self, 'cards_lost', {}).items()},
                   'units': {name: trial_counts(unit.study) if isinstance(unit, StudyUnit) else
                             {'fixed_tasks_left': unit.remaining()} for name, unit in self.units.items()}}
        summary['resources'] = self.resource_status()
        with open(os.path.join(self.path_logs, 'orchestrator_summary.json'), 'w', encoding='utf-8') as stream:
            json.dump(summary, stream, indent=1, default=str)
        if self.failures:
            import pandas as pd
            pd.DataFrame(self.failures).to_csv(os.path.join(self.path_logs, 'orchestrator_failures.csv'), index=False)
            log.warning('%d failed trials, see %s', len(self.failures),
                        os.path.join(self.path_logs, 'orchestrator_failures.csv'))
            for line in failure_causes(self.failures, self.ended_well):
                log.warning('%s', line)
