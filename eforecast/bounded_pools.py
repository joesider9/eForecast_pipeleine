# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""One bounded reader layer within the threads a piece of work was started with; safe in the online image.

The coordinating process keeps one CPU. Children get one native thread each and
cannot create another pool. Calls made without an execution policy run inline.

``CardReaders`` is the same idea for the cards: one reader a card, for a process that itself
runs on the CPUs.
"""
from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager, suppress
from contextvars import ContextVar
import logging
import multiprocessing
import os
from pathlib import Path

from eforecast import execution_policy

log = logging.getLogger(__name__)
cancel_check = ContextVar('eforecast_pool_cancel_check', default=None)
_nested = ContextVar('eforecast_nested_pool', default=False)
_function = None
_warned = set()


def capacity():
    """Affinity and cgroup limits, without importing the training control plane."""
    count = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)
    # Check unified ancestors too: a nested group can inherit a tighter quota.
    mount = Path('/sys/fs/cgroup')
    unified = [mount / 'cpu.max']
    legacy = [(Path('/sys/fs/cgroup/cpu/cpu.cfs_quota_us'), Path('/sys/fs/cgroup/cpu/cpu.cfs_period_us'))]
    try:
        memberships = Path('/proc/self/cgroup').read_text().splitlines()
        for line in memberships:
            _, controllers, relative = line.split(':', 2)
            if 'cpu' not in controllers.split(','):
                continue
            for name in ('cpu', 'cpu,cpuacct', 'cpuacct,cpu'):
                base = mount / name
                group = base / relative.lstrip('/')
                legacy.extend((p / 'cpu.cfs_quota_us', p / 'cpu.cfs_period_us')
                              for p in [group, *group.parents] if p == base or base in p.parents)
        relative = next(line.split('::', 1)[1].strip() for line in memberships if '::' in line)
        group = mount / relative.lstrip('/')
        unified += [p / 'cpu.max' for p in [group, *group.parents] if p == mount or mount in p.parents]
    except (OSError, StopIteration, ValueError):
        pass
    # Both unified and legacy Linux cgroups. Fractions round down, at least one.
    for quota, period in (*[(p, None) for p in unified], *legacy):
        try:
            values = Path(quota).read_text().split()
            q, p = (float(values[0]), float(Path(period).read_text())) if period else map(float, values)
            if q > 0 and p > 0:
                count = min(count, max(1, int(q / p)))
        except (OSError, ValueError):
            pass
    return max(1, count)


def workers(requested):
    policy = execution_policy.current()
    if not policy and requested not in (None, 1) and str(requested) not in _warned:
        log.warning('reader count %r runs inline on one CPU: this process has no execution policy '
                    '(a stage or a worker installs one)', requested)
        _warned.add(str(requested))
    if not policy or _nested.get() or os.environ.get('EFORECAST_POOL_CHILD') == '1':
        return 0
    grant = min(capacity(), policy['given']['cpu_threads'])
    wanted = int(requested or grant)
    wanted = grant if wanted < 1 else wanted
    # Native-thread parallelism and reader parallelism are mutually exclusive.
    if policy['effective']['threads'] != 1:
        return 0
    return max(0, min(wanted, grant - 1))


def wide_workers(native_threads):
    """How many readers of ``native_threads`` threads each this process may start: its CPUs over that.

    Only when this process itself works with that many threads: such readers exist to do, several
    at once, exactly what this process would do one after another. Fewer than two is no pool.
    """
    policy = execution_policy.current()
    if not policy or _nested.get() or os.environ.get('EFORECAST_POOL_CHILD') == '1':
        return 0
    if int(policy['effective']['threads']) != int(native_threads):
        return 0
    count = min(capacity(), int(policy['given']['cpu_threads'])) // int(native_threads)
    return count if count >= 2 else 0


def _initialize(function, native_threads=1):
    global _function
    _function = function
    os.environ['EFORECAST_POOL_CHILD'] = '1'
    os.environ.update(dict.fromkeys(execution_policy.THREAD_ENV, str(native_threads)))
    # The spawn environment is already bounded before unpickling the callable.
    import sys
    if 'torch' in sys.modules:
        sys.modules['torch'].set_num_threads(native_threads)
    if 'cv2' in sys.modules:
        sys.modules['cv2'].setNumThreads(native_threads)


def _invoke(arguments):
    return _function(*arguments)


def _initialize_on_card(target, native_threads, cards):
    """What a reader of ``CardReaders`` does first: take a card and install it, then import the work.

    The reader is a spawned process with the environment of the process that started it, and
    that is a worker on the CPUs: it sees no card. A library that has once looked for the cards
    cannot be shown another set, so the card is put in place before anything that could look is
    imported -- which is why the work is given by name and not as the function itself.
    Unpickling a function imports its module, and that happens before any initializer runs.
    """
    global _function
    import importlib
    card = cards.get(timeout=60)
    os.environ['EFORECAST_POOL_CHILD'] = '1'
    policy = execution_policy.resolve(native_threads, card, {'device': 'auto', 'threads': native_threads})
    execution_policy.apply(policy)
    execution_policy.configure_torch(policy)
    module, name = target.split(':', 1)
    _function = getattr(importlib.import_module(module), name)


class CardReaders:
    """Readers started on cards, one a card, which take work as they become free.

    For work that is loaded and run whole in one process and that a card does far faster than
    the CPUs -- a foundation model of the predict stage -- in a process that itself runs on the
    CPUs. ``target`` is ``'module:function'``; each reader sees one of ``cards`` (GPU UUIDs) and
    nothing else, and works with ``native_threads`` threads beside it.

    ``submit`` returns a future. Nothing is retried here: a reader that dies breaks the pool,
    every future still out then fails, and whoever asked decides what becomes of that work.
    Nothing is reserved on a card either. The driver gives a reader memory as it asks, and a
    reader that asks for more than is there fails with the driver's own error.
    """
    def __init__(self, target, cards, native_threads=1):
        cards = list(cards)
        if not cards:
            raise ValueError('card readers need at least one card')
        context = multiprocessing.get_context('spawn')
        self.count = len(cards)
        self._cards = context.Queue()
        for card in cards:
            self._cards.put(card)
        self._executor = ProcessPoolExecutor(self.count, mp_context=context, initializer=_initialize_on_card,
                                             initargs=(str(target), max(1, int(native_threads)), self._cards))

    def submit(self, *arguments):
        return self._executor.submit(_invoke, arguments)

    def close(self, *, stop=False):
        """Wait for the readers to end. ``stop`` ends them where they are: nobody wants their work."""
        if stop:
            for process in list((getattr(self._executor, '_processes', None) or {}).values()):
                with suppress(Exception):
                    process.terminate()
        self._executor.shutdown(wait=True, cancel_futures=True)
        with suppress(Exception):
            self._cards.close()


@contextmanager
def _scope():
    token = _nested.set(True)
    try:
        yield
    finally:
        _nested.reset(token)


def _ended(executor, patience=2.0):
    """How the readers of a broken pool ended, as far as the pool still knows: their exit codes.

    A pool that breaks ends its other readers itself, and collects all of them in a thread of
    its own. Asked at once, a reader may not have been collected yet and has no exit code; on a
    busy machine -- which is where the kernel kills readers -- that was often the very one that
    broke the pool, and the report then said only that a reader "ended". So the codes are waited
    for, briefly.
    """
    import time
    try:
        processes = list((getattr(executor, '_processes', None) or {}).values())
        deadline = time.monotonic() + patience
        while time.monotonic() < deadline and any(process.exitcode is None for process in processes):
            time.sleep(.01)
        return sorted({process.exitcode for process in processes if process.exitcode not in (None, 0)})
    except Exception:                                           # noqa: BLE001 -- only the report is worse without it
        return []


def _how(codes):
    """How the reader that broke a pool ended, and what that usually means.

    The readers a broken pool stops itself end with SIGTERM; the one that broke it ended otherwise.
    Only a kill is read as the kernel's: a reader that ended on its own -- it could not start, or
    a library aborted under it -- is said to have done that, with its exit code.
    """
    if -9 in codes or 137 in codes:
        return ('was killed (SIGKILL). When nobody stopped it, that is the kernel ending the largest process '
                'because the machine ran out of memory (journalctl -k | grep -i "out of memory")')
    own = [code for code in codes if code != -15]
    return ((f'ended with exit code {own[0]}' if own else 'ended') + ' without handing back a result or an error: '
            'the reader itself failed, and what it printed is above')


def _as_finished(pool, invoke, arguments, count, check, kills):
    """``(position, result)`` of each piece of work as it finishes; see ``map_bounded``.

    2P pieces are with the readers at any time, counted by what has not finished, so a reader
    is never without work while there is some. A reader the kernel kills is treated as in the
    ordered way: the pool is made again with half as many and what it took with it is done
    again. The failure of a piece itself is raised at once with its ``position``, and the readers
    are not waited for then: they may be an hour into something nobody will ask for.
    """
    from concurrent.futures import FIRST_COMPLETED, wait
    from concurrent.futures.process import BrokenProcessPool
    executor = pool(count)
    flying = {}                                                 # future -> (position, arguments)
    numbered = enumerate(arguments)

    def fill():
        while len(flying) < 2 * count:
            item = next(numbered, None)
            if item is None:
                return
            check()
            flying[executor.submit(invoke, item[1])] = item
    try:
        fill()
        while flying:
            check()
            done, _ = wait(flying, timeout=.2, return_when=FIRST_COMPLETED)
            if any(isinstance(future.exception(), BrokenProcessPool) for future in done):
                how = _how(_ended(executor))
                executor.shutdown(wait=True, cancel_futures=True)
                if count == 1:
                    raise BrokenProcessPool(f'the reader of this work {how}. It was the only reader, so fewer '
                                            'could not be tried') from None
                log.warning('with %d readers running, one %s. The work that was in flight is done again with %d '
                            'reader(s)', count, how, count // 2)
                count //= 2
                executor = pool(count)
                # What had arrived is kept, a failure of the work itself too; the rest is asked again.
                flying = {(was if was.done() and not was.cancelled() and not isinstance(was.exception(), BrokenProcessPool)
                           else executor.submit(invoke, item[1])): item for was, item in flying.items()}
                continue
            for future in done:
                position, _ = flying.pop(future)
                try:
                    value = future.result()
                except Exception as error:
                    with suppress(Exception):
                        error.position = position
                    raise
                yield position, value
            fill()
    finally:
        for future in flying:
            future.cancel()
        if flying and kills:
            for process in list((getattr(executor, '_processes', None) or {}).values()):
                process.terminate()
        executor.shutdown(wait=True, cancel_futures=True)


def map_bounded(function, *iterables, n_jobs=1, threads=False, native_threads=1, in_order=True):
    """Ordered results, at most 2P in flight, no eager consumption of all inputs.

    ``in_order=False`` hands back ``(position, result)`` as each piece finishes instead. That is
    for work whose pieces differ widely in length. In order, a piece that takes an hour holds
    back everything behind it: once the 2P pieces after it are done, the readers sit idle until
    it ends. The predict stage met that with foundation models among its trials: eight readers,
    two of them working.

    Process targets must be importable/picklable (spawn, never inherited ML state).
    Thread mode is for I/O only; it uses the same subdivision and nested guard.
    Cancellation/errors propagate; shutdown waits for the children.

    A reader has one native thread. ``native_threads`` gives each of them more, and there are
    then that many times fewer of them (``wide_workers``): for work whose numbers depend on the
    thread count, which must be done with the count it is always done with.

    Nothing holds the number of readers to the memory of the machine, so the kernel may end one.
    The pool is then made again with half as many, and the work that was in flight is done again
    in its order: a reader's work is to read and hand back, or to write a file whole, and doing
    it twice changes nothing. With one reader left the failure is raised, saying what it was.
    """
    from concurrent.futures.process import BrokenProcessPool
    native_threads = max(1, int(native_threads))
    count = workers(n_jobs) if native_threads == 1 else wide_workers(native_threads)
    check = cancel_check.get() or (lambda: None)
    arguments = iter(zip(*iterables))
    if count == 0:
        with _scope():
            for position, args in enumerate(arguments):
                check()
                yield function(*args) if in_order else (position, function(*args))
        return
    log.info('bounded pool: requested=%s readers=%d native_threads=%d prefetch=%d', n_jobs, count, native_threads,
             2 * count)
    if threads:
        def invoke(args):
            with _scope():
                return function(*args)
    else:
        invoke = _invoke

    def pool(size):
        if threads:
            return ThreadPoolExecutor(size)
        return ProcessPoolExecutor(size, mp_context=multiprocessing.get_context('spawn'),
                                   initializer=_initialize, initargs=(function, native_threads))

    if not in_order:
        yield from _as_finished(pool, invoke, arguments, count, check, kills=not threads)
        return
    executor = pool(count)
    pending = deque()                                           # (arguments, future), in the order results are due
    try:
        for _ in range(2 * count):
            args = next(arguments, None)
            if args is None:
                break
            check()
            pending.append((args, executor.submit(invoke, args)))
        while pending:
            check()
            args, future = pending[0]
            from concurrent.futures import TimeoutError
            try:
                while True:
                    try:
                        value = future.result(timeout=.2)
                        break
                    except TimeoutError:
                        if future.done():
                            value = future.result()  # propagate the target's own TimeoutError
                            break
                        check()
            except BrokenProcessPool:
                how = _how(_ended(executor))
                executor.shutdown(wait=True, cancel_futures=True)
                if count == 1:
                    raise BrokenProcessPool(f'the reader of this work {how}. It was the only reader, so fewer '
                                            'could not be tried') from None
                log.warning('with %d readers running, one %s. The work that was in flight is done again with %d '
                            'reader(s)', count, how, count // 2)
                count //= 2
                executor = pool(count)
                # A result that had arrived is kept, and so is a failure of the work itself: it is
                # raised when its turn comes. Everything the broken pool took with it is asked again.
                pending = deque((sent, was if was.done() and not was.cancelled()
                                 and not isinstance(was.exception(), BrokenProcessPool)
                                 else executor.submit(invoke, sent)) for sent, was in pending)
                continue
            pending.popleft()
            yield value
            check()
            args = next(arguments, None)
            if args is not None:
                pending.append((args, executor.submit(invoke, args)))
    finally:
        for _, future in pending:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
