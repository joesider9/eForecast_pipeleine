# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Synchronous checkpoint staging; no ML imports, queues or extra weight copies.

Stage every member before replacing any named payload. Each rename is atomic;
the attempt's own folder and its manifest are what publish the complete bundle.
This is not a multi-file transaction for arbitrary external in-place writers.
"""
import os
from pathlib import Path
import stat
import uuid

_UNSET=object()


def _temporary(directory,destination,suffix):
    path=directory/f'.save-{uuid.uuid4().hex}{suffix}'
    fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_RDWR,0o666)
    try:
        if destination.exists():os.fchmod(fd,stat.S_IMODE(destination.stat().st_mode))
        return fd,path
    except BaseException:
        os.close(fd);path.unlink(missing_ok=True)
        raise


def publish(directory, *, weights=_UNSET, metadata=None, weight_name='net_weights.pt', metadata_name='net_weights.pickle'):
    """Prepare/fsync all members, then rename. Serialization failure preserves old files.

    Torch uses its native path writer, avoiding a Python bytes copy of tensor
    storage. Metadata uses joblib's existing streaming serializer.
    """
    directory=Path(directory);staged=[]
    try:
        if weights is not _UNSET:
            import torch
            destination=directory/weight_name
            fd,path=_temporary(directory,destination,'.pt');os.close(fd)
            staged.append((path,destination))
            torch.save(weights,str(path))
            with path.open('rb') as stream:os.fsync(stream.fileno())
        if metadata is not None:
            import joblib
            destination=directory/metadata_name
            fd,path=_temporary(directory,destination,'.pickle')
            staged.append((path,destination))
            with os.fdopen(fd,'wb') as stream:
                joblib.dump(metadata,stream)
                stream.flush();os.fsync(stream.fileno())
        for temporary,destination in staged:
            os.replace(temporary,destination)
        if staged:
            fd=os.open(directory,os.O_RDONLY|os.O_DIRECTORY)
            try:os.fsync(fd)
            finally:os.close(fd)
    finally:
        for path,_ in staged:
            try:path.unlink(missing_ok=True)
            except OSError:pass  # preserve the write/publication failure


def snapshot(model):
    """One independent CPU copy, whether the live state is on CPU or CUDA."""
    return {key:value.detach().to(device='cpu',copy=True) for key,value in model.state_dict().items()}
