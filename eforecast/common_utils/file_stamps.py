# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Is this still the file that was looked at? What a stat says, compared with care.

A guard that must not hash gigabytes compares what a stat says before and after the work:
device, inode, size and modification time -- and change time, which is what gives away a file
whose content was rewritten in place and whose modification time was put back.

Change time also moves when nothing of the content does: when an extended attribute is set,
when the permission bits change, when a link is added. A sync client does that to every file
it has uploaded (Dropbox writes ``user.com.dropbox.attrs`` a moment after a file is written,
and again when the upload ends); a backup that works with hard links does, an indexer may.
Compared as plain numbers that is "the input changed", and it ended combine runs, trials and
stages whose inputs nobody had touched -- reliably so in a model folder that is synced, right
after a training run had written new files into it.

So a stamp carries one thing more than the stat: a digest of the inode's own attributes. Two
stamps are the same file when they are equal, or when the change time moved *and those
attributes moved with it*: the move is then explained by a change of metadata, which is not a
change of the input. A change time that moved while nothing else did is still a change. That
is the file rewritten in place with its old size and its old modification time, and it is
what the change time is watched for.

The limit, said once: a file that was rewritten in that way *and* had its metadata changed
between the two looks cannot be told from one that only had its metadata changed. Nothing a
stat holds can tell, and nothing here reads a file's content. A guard for which that is not
good enough -- one that is about to replace the very file it watched -- keeps the plain
comparison of the five numbers (``numbers``), and fails when a sync client is at work.

Extended attributes are read where the platform lets Python read them, which is Linux.
"""
from __future__ import annotations

import hashlib
import os
import stat as stat_module

#: A stamp is ``[device, inode, size, mtime_ns, ctime_ns, attributes]``. The first four name
#: the file and its content as far as a stat can, the fifth is the change time, the sixth the
#: digest of the inode's attributes. Stamps taken before there was a sixth have five fields.
IDENTITY = 4
CHANGED_AT = 4
ATTRIBUTES = 5
#: How often a stamp is taken again when the file's metadata moved while it was being read.
ATTEMPTS = 3


def attributes(path, info=None, *, follow=True):
    """A digest of what an inode carries beside its content: permission bits, links, extended attributes.

    Whatever moves the change time without touching the content shows here -- as far as this
    process can see it. Owner and group are left out on purpose: a container may show the same
    file under another owner than the host does, and a stamp taken on one side is compared
    with a stamp taken on the other. An attribute that cannot be read counts as unreadable,
    not as absent, and a filesystem that keeps none gives the stat's own fields alone.

    ``path`` may be an open descriptor.
    """
    descriptor = isinstance(path, int)
    if info is None:
        info = os.stat(path) if follow or descriptor else os.lstat(path)
    digest = hashlib.sha256(f'{stat_module.S_IMODE(info.st_mode)} {info.st_nlink}'.encode())
    listing = getattr(os, 'listxattr', None)
    if listing is not None:
        how = {} if follow or descriptor else {'follow_symlinks': False}
        try:
            names = sorted(listing(path, **how))
        except OSError:
            names = None
        if names is None:
            digest.update(b'\0<no extended attributes can be read>')
        for name in names or ():
            try:
                value = os.getxattr(path, name, **how)
            except OSError:
                value = b'<unreadable>'
            digest.update(b'\0' + os.fsencode(name) + b'\0' + hashlib.sha256(value).digest())
    return digest.hexdigest()[:16]


def _read(path, follow):
    return os.stat(path) if follow or isinstance(path, int) else os.lstat(path)


def _fields(info):
    return [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns]


def stamp(path, *, follow=True):
    """``[device, inode, size, mtime_ns, ctime_ns, attributes]`` of a file that exists.

    The numbers and the attributes are of one moment: when the file's metadata moved between
    the stat and the reading of its attributes, both are read again. A stamp that paired the
    old change time with the new attributes would make the next comparison say "changed".
    """
    info = _read(path, follow)
    for _ in range(ATTEMPTS):
        marks = attributes(path, info, follow=follow)
        again = _read(path, follow)
        if _fields(again) == _fields(info) and (again.st_mode, again.st_nlink) == (info.st_mode, info.st_nlink):
            break
        info = again
    return [*_fields(info), marks]


def observe(path, *, follow=True):
    """``stamp``, or None for a file that is not there."""
    try:
        return stamp(path, follow=follow)
    except FileNotFoundError:
        return None


def _is_stamp(value):
    return isinstance(value, (list, tuple)) and len(value) > CHANGED_AT and \
        all(type(item) is int for item in value[:CHANGED_AT + 1])


def numbers(value):
    """The five numbers of a stamp, as stamps were before they carried attributes.

    For what must not depend on the attributes: an identity that is hashed and kept, a
    comparison that has to stay strict. Anything that is no stamp is returned as it is.
    """
    return list(value[:CHANGED_AT + 1]) if _is_stamp(value) else value


def bare(observations):
    """``numbers`` of every stamp in a ``{path: stamp}`` mapping."""
    if not isinstance(observations, dict):
        return observations
    return {path: numbers(value) for path, value in observations.items()}


def same(before, after):
    """Whether two stamps of one path are, as far as a stat can tell, the same file content.

    Equal stamps are, and stamps whose five numbers are equal. So are two that differ in the
    change time alone while the attributes of the inode differ as well: metadata moved, the
    content did not. Everything else is not -- another file, another size, another
    modification time, or a change time that moved with nothing to explain it. A stamp taken
    without the attributes (five fields, as they used to be) can explain nothing.

    Values that are no stamps -- None for a missing file, a note that nothing could be
    observed -- are the same when they are equal.
    """
    if before == after:
        return True
    if not _is_stamp(before) or not _is_stamp(after):
        return False
    if list(before[:IDENTITY]) != list(after[:IDENTITY]):
        return False
    if before[CHANGED_AT] == after[CHANGED_AT]:
        return True
    if len(before) <= ATTRIBUTES or len(after) <= ATTRIBUTES:
        return False
    return before[ATTRIBUTES] != after[ATTRIBUTES]


def differing(before, after):
    """The paths whose stamps are not ``same`` in two ``{path: stamp}`` mappings, sorted.

    A path only one of them has is one of them: a file that appeared or went is a change.
    """
    if not isinstance(before, dict) or not isinstance(after, dict):
        return [] if before == after else ['<observations>']
    return sorted(str(path) for path in set(before) | set(after)
                  if path not in before or path not in after or not same(before[path], after[path]))


def moved(before, after):
    """Whether anything in two ``{path: stamp}`` mappings is not the same file any more."""
    return bool(differing(before, after))


def named(paths, limit=3):
    """The first few of ``paths``, for a message that says what moved."""
    paths = list(paths)
    return ', '.join(paths[:limit]) + (f' and {len(paths) - limit} more' if len(paths) > limit else '')
