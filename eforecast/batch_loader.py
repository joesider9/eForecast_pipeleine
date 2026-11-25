# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""CPU-only bounded mapped reads for one confirmed parent's loader subprocess.

The parent owns admission, reader pins, scratch and CUDA. This module imports no
Torch, host store or model code; numeric payloads never travel through pickle.
"""
import json
import os
from pathlib import Path
import sys


def stamp(path):
    s=Path(path).stat()
    return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]


def main(config):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('loader must start with CUDA hidden before imports')
    import numpy as np
    settings=json.loads(Path(config).read_text())
    root=Path(config).parent
    for line in sys.stdin.buffer:
        if len(line)>settings['command_bytes']:raise ValueError('loader command exceeds its bound')
        job=json.loads(line);sources=json.loads((root/job['source']).read_text())
        target=root/f"slot-{job['slot']}.bin";temporary=target.with_suffix('.partial')
        used=0
        try:
            with temporary.open('wb') as output:
                for spec,rows in zip(sources,job['positions'],strict=True):
                    path=Path(spec['path'])
                    if path.is_symlink() or stamp(path)!=spec['stamp']:
                        raise ValueError('mapped source changed before loader read')
                    if len(rows)>settings['batch_rows']:raise MemoryError('loader row allowance exceeded')
                    values=np.memmap(path,mode='r',dtype=spec['dtype'],shape=tuple(spec['shape']),offset=spec['offset'])
                    try:
                        if any(i<0 or i>=len(values) for i in rows):raise IndexError('mapped source row out of bounds')
                        block=np.asarray(values[rows],dtype=spec['cast']).astype('float32',order='C',copy=False)
                        used+=block.nbytes
                        if used>settings['batch_bytes']:raise MemoryError('loader payload exceeds its allowance')
                        block.tofile(output)
                        if stamp(path)!=spec['stamp']:raise ValueError('mapped source changed during loader read')
                    finally:
                        values._mmap.close()
            os.replace(temporary,target)
            print(json.dumps({'seq':job['seq'],'bytes':used,'pid':os.getpid(),'cuda_visible':os.environ['CUDA_VISIBLE_DEVICES'],
                              'torch_imported':'torch' in sys.modules}),flush=True)
        finally:
            temporary.unlink(missing_ok=True)


if __name__=='__main__':main(sys.argv[1])
