# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Cheap runtime/source identities; optional GPU metadata is probed only on request."""
import hashlib
import importlib.metadata
import os
from pathlib import Path
import platform
import subprocess
import site
import sysconfig

from eforecast.profile_policy import identity

ROOT = Path(__file__).resolve().parents[1]


def distribution_versions():
    """Inventory installed/startup roots, independent of import-time vendor paths.

    Setuptools can append its _vendor directory after a training import. Those
    private distributions must not replace installed versions or create false
    hardware drift between a coordinating process and a fresh model worker.
    Explicit PYTHONPATH and editable-install metadata remain part of the key.
    """
    roots=[ROOT,*[Path(p) for p in os.environ.get('PYTHONPATH','').split(os.pathsep) if p],
           Path(sysconfig.get_path('purelib')),Path(sysconfig.get_path('platlib')),
           *map(Path,site.getsitepackages())]
    if site.ENABLE_USER_SITE:roots.append(Path(site.getusersitepackages()))
    versions={}
    for root in dict.fromkeys(str(p.resolve()) for p in roots):
        for dist in importlib.metadata.distributions(path=[root]):
            name=dist.metadata['Name']
            if name:versions.setdefault(name.lower(),dist.version)
    return versions


def source(entrypoint):
    module = entrypoint.split(':', 1)[0]
    if (not module.startswith(('eforecast.', 'eforecast_online.', 'tests.'))
            or any(not p.isidentifier() for p in module.split('.'))):
        raise ValueError('profile target must be an explicit repository module')
    path = ROOT.joinpath(*module.split('.')).with_suffix('.py')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hardware(entrypoint, device_uuid=None):
    versions = distribution_versions()
    models = [line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
              if line.startswith('model name')]
    gpu = {'gpu_arch': None, 'gpu_mb': 0, 'compute_capability': None, 'driver': None}
    if device_uuid:
        # Identified through nvidia-smi, which this function already used as its fallback, rather
        # than through NVML. A neutral worker must not need the training side's pynvml: the
        # inference image is built without it, and tests/test_online.py reads the import graph
        # rather than trusting that a conditional import never fires. Measured on this host's
        # 2080 Ti -- name, memory.total MiB, driver_version and compute_cap are each identical to
        # what NVML reported, so the recorded hardware witness is unchanged.
        answer = subprocess.check_output(
            ['nvidia-smi', '--id=' + device_uuid,
             '--query-gpu=name,memory.total,driver_version,compute_cap',
             '--format=csv,noheader,nounits'], text=True, timeout=10).strip().splitlines()
        if not answer:
            raise RuntimeError(f'nvidia-smi identified no device for {device_uuid}')
        # Split from the right: a device name may contain a comma, the three numbers may not.
        name, memory, driver, capability = (part.strip() for part in answer[0].rsplit(',', 3))
        gpu.update(gpu_arch=name, gpu_mb=int(memory), compute_capability=capability, driver=driver)
    paths = [p for folder in ('eforecast', 'eforecast_online') for p in (ROOT / folder).rglob('*.py')]
    paths += list(ROOT.glob('run*.py'))
    code = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    code['target'] = source(entrypoint)
    return {'cpu': models[0] if models else platform.machine(), 'logical_cpus': os.cpu_count() or 1,
            'python': platform.python_version(), 'backend': 'torch:' + str(versions.get('torch')),
            'libraries': identity(versions), **gpu, 'image': os.environ.get('EFORECAST_IMAGE_ID'),
            'code': identity(code)}


def check(definition, entrypoint, device_uuid, *, image_witness=None):
    """Validate a pinned profile against the runtime that will actually execute it.

    Without a witness this is the ordinary self-check: the process asks whether it is
    itself the qualified environment. With one, the caller is a host about to launch an
    immutable image, so the image's own recorded Python, libraries, code and identity
    replace this host's, while the CPU and the physically allocated GPU still come from
    here. The worker inside the image performs the ordinary self-check before loading,
    so a witness can never be the only thing that activates a profile.
    """
    if image_witness is None:
        expected_implementation = source(entrypoint)
        expected_hardware = hardware(entrypoint, device_uuid)
    else:
        from eforecast.image_runtime import expected as merge
        if image_witness.get('entrypoint') != entrypoint:
            raise ValueError('image runtime witness describes another target')
        expected_implementation = image_witness['implementation']
        expected_hardware = merge(image_witness, hardware(entrypoint, device_uuid))
    if definition['key']['workload']['implementation'] != expected_implementation:
        raise ValueError('profile target source changed after acceptance')
    if definition['key']['hardware'] != expected_hardware:
        raise ValueError('profile hardware/software/code changed after acceptance')
