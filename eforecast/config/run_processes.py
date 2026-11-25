# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Linux descendant evidence. Numeric PIDs are never signal authority."""
import os
import ctypes
import errno
from pathlib import Path
import signal
import platform

from eforecast.config.executors import observe_process, process_evidence


def descendants(parent):
    """Ancestry census in this procfs namespace; callers retain prior birth identities."""
    # Kernel child lists avoid scanning every process on the machine for every active run.
    # Include every thread: joblib / data loaders may spawn from a non-main thread.
    owned, frontier = set(), {parent}
    while frontier:
        children = set()
        for pid in frontier:
            try:
                for task in Path(f'/proc/{pid}/task').iterdir():
                    try:
                        children.update(int(value) for value in (task / 'children').read_text().split())
                    except (FileNotFoundError, ProcessLookupError):
                        pass
            except (FileNotFoundError, ProcessLookupError):
                pass
        children -= owned | {parent}
        owned.update(children)
        frontier = children
    evidence = []
    for pid in sorted(owned):
        try:
            evidence.append(process_evidence(pid))
        except (FileNotFoundError, ProcessLookupError):
            pass
    return evidence


def _pidfd_syscall(libc, number, *arguments):
    # Linux UAPI: x86_64 syscall_64.tbl and arm64's asm-generic/unistd.h.
    # Older Conda Python/glibc images expose neither wrapper although their kernel supports
    # pidfds. Keep the descriptor-based identity boundary; never fall back to os.kill(pid).
    if (platform.system() != 'Linux' or platform.machine().lower() not in {'x86_64', 'aarch64'}
            or ctypes.sizeof(ctypes.c_void_p) != 8 or not hasattr(libc, 'syscall')):
        raise OSError(errno.ENOSYS, 'pidfd syscall ABI is unsupported on this runtime')
    libc.syscall.restype = ctypes.c_long
    result = libc.syscall(ctypes.c_long(number), *arguments)
    if result < 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))
    return result


def pidfd_open(pid):
    if hasattr(os, 'pidfd_open'):
        return os.pidfd_open(pid)
    libc = ctypes.CDLL(None, use_errno=True)
    if not hasattr(libc, 'pidfd_open'):
        return _pidfd_syscall(libc, 434, ctypes.c_int(pid), ctypes.c_uint(0))
    descriptor = libc.pidfd_open(pid, 0)
    if descriptor < 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))
    return descriptor


def pidfd_signal(descriptor, signum):
    if hasattr(signal, 'pidfd_send_signal'):
        return signal.pidfd_send_signal(descriptor, signum)
    libc = ctypes.CDLL(None, use_errno=True)
    if not hasattr(libc, 'pidfd_send_signal'):
        return _pidfd_syscall(libc, 424, ctypes.c_int(descriptor), ctypes.c_int(signum),
                              ctypes.c_void_p(), ctypes.c_uint(0))
    if libc.pidfd_send_signal(descriptor, signum, None, 0) < 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))


def signal_owned(evidence, *, force=False, signum=None):
    """Open a pidfd before the birth/namespace check; a reused PID receives no signal."""
    try:
        descriptor = pidfd_open(evidence['pid'])
    except ProcessLookupError:
        return observe_process(evidence)
    try:
        state = observe_process(evidence)
        if state == 'alive':
            pidfd_signal(descriptor, signum or (signal.SIGKILL if force else signal.SIGTERM))
        return state
    except ProcessLookupError:
        return observe_process(evidence)
    finally:
        os.close(descriptor)


def key(process):
    return f"{process['host']}:{process['boot_id']}:{process['pid_namespace']}:{process['pid']}:{process['start_ticks']}"
