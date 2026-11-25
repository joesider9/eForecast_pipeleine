# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""What the Runs page shows of a run's use of the machine. No probes or ML imports.

Nothing here is a limit or a reservation: the controls say how many CPUs and studies a run
uses, and the view says what is running and what the machine was seen to hold for it.
"""
from copy import deepcopy
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, create_model
from .schema import JobsConfig

CONTROL_KEYS = ('n_cpus', 'cpu_workers', 'cpu_study_threads', 'cpu_study_device')
# The case schema owns field types, constraints, descriptions and defaults. Run
# overrides are optional, including counts with a non-null case default.
fields = {}
for key in CONTROL_KEYS:
    original = JobsConfig.model_fields[key]
    field = deepcopy(original)
    field.default = None
    fields[key] = (original.annotation | None, field)
ResourceControls = create_model('ResourceControls', __config__=ConfigDict(extra='forbid'), **fields)


class StudyResources(BaseModel):
    id: str
    method: str
    cluster: str
    state: str
    threads: float | None = None
    requested_device: str | None = None
    resolved_device: str | None = None
    normalization: str | None = None
    #: Why a study that is not running yet is not: its turn has not come.
    reason: str | None = None


class StudyDispatch(BaseModel):
    """How the CPU studies of a run are run side by side."""
    requested_workers: float | None = None
    effective_workers: float | None = None
    requested_threads: float | None = None
    threads_per_study: float | None = None
    requested_total_threads: float | None = None
    total_threads: float | None = None
    trials_per_study: float | None = None
    studies: float | None = None


class ResourceProgress(BaseModel):
    verified_complete: int
    pruned: int
    failed: int
    elapsed_seconds: float
    verified_models_per_minute: float


class ResourceView(BaseModel):
    version: Literal[2] = 2
    #: Who wrote the snapshot: the CPU study dispatcher, the training scheduler of the GPU
    #: trials, one study's own worker, or something this version does not name.
    source: Literal['cpu_studies', 'gpu_trials', 'worker', 'legacy'] = 'legacy'
    updated: float | None = None
    phase: str | None = None
    error: str | None = None
    #: Threads the running studies or trials are told to use, and what their processes hold.
    running_threads: float | None = None
    #: GPU trials: how many run at once since the kernel killed one of them for memory.
    trial_limit: float | None = None
    pss_mb: float | None = None
    pss_confidence: str | None = None
    dispatch: StudyDispatch | None = None
    studies: list[StudyResources] = Field(default_factory=list)
    progress: ResourceProgress | None = None
    warnings: list[str] = Field(default_factory=list)


class ResourcePreview(BaseModel):
    controls: ResourceControls
    note: str = ('Requested settings. How many studies run at once and with how many threads is decided '
                 'when the run starts, from the CPUs the case uses and the studies it has.')


class ResourceContract(BaseModel):
    controls: ResourceControls
    snapshot: ResourceView
    preview: ResourcePreview
