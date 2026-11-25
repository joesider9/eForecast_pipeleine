# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

from datetime import datetime
import math
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, RootModel, field_validator, model_validator
from eforecast.config.capabilities import (BACKENDS, CLUSTER_METHODS, COMBINE_METHODS, CPU_METHODS, GLOBAL_METHODS,
                                            NWP_EXTRACTORS)
from eforecast.config.validation import validate_experiment_references, validate_family_parameters


class HorizonType(str, Enum):
    """What the case forecasts, and therefore how its rows and targets are shaped.

    ``day_ahead``   one value per hour of the next day, issued once a day
    ``intraday``    the remaining hours of the current day, issued through the day
    ``short_term``  ``horizon`` steps ahead of every issue time, one row per step (multi-output)
    ``long_term``   up to three days ahead, indexed by (prediction time, issue time); this is
                    also the former ``two-day-ahead`` case, whose window it contains

    The names were ``day-ahead``, ``intra-ahead``, ``multi-output`` and ``multi-ahead``.  Both
    spellings are accepted so cases written before the rename keep loading; the new one is what is
    written back and what the pipeline compares against.
    """

    DAY_AHEAD = "day_ahead"
    INTRADAY = "intraday"
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"

    @classmethod
    def _missing_(cls, value: Any) -> "HorizonType | None":
        return cls._member_map_.get(LEGACY_HORIZON_TYPES.get(value, ""))  # type: ignore[return-value]


# the former spelling -> the member name it became
LEGACY_HORIZON_TYPES: dict[str, str] = {
    "day-ahead": "DAY_AHEAD",
    "intra-ahead": "INTRADAY",
    "multi-output": "SHORT_TERM",
    "multi-ahead": "LONG_TERM",
}


class ProblemType(str, Enum):
    PV = "pv"
    WIND = "wind"
    LOAD = "load"
    FA = "FA"


# lag unit per time step: lags are written in hours, so a 15-minute series has 4 lags per hour
RESOLUTION_STEP: dict[str, float] = {"1min": 1 / 60, "5min": 1 / 12, "15min": 0.25, "H": 1, "D": 1}


def lag_step(ts_resolution: str | None) -> float:
    return RESOLUTION_STEP.get(ts_resolution or "H", 1)


class NwpModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    resolution: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    levels: int | None = Field(default=None, ge=1)
    path_nwp_source: str | None = None
    path_nwp_gdrive: str | None = Field(default=None, description='Link of the shared Google Drive folder that holds the YYYYMMDD day folders of this provider. A day that is not in path_nwp_source is downloaded from it when extract runs, read into its daily pickle, and the downloaded files are removed; path_nwp_source is where they are kept meanwhile. Empty (null): only the local folder is read.')

    @field_validator('path_nwp_gdrive')
    @classmethod
    def check_gdrive_folder(cls, value: str | None) -> str | None:
        if value is not None:
            from eforecast.nwp_extraction.gdrive import folder_id
            folder_id(value)            # raises, and says what a link looks like
        return value

    @model_validator(mode="after")
    def check_gdrive_provider(self) -> "NwpModelConfig":
        if self.path_nwp_gdrive is not None and self.model not in NWP_EXTRACTORS:
            raise ValueError(f'path_nwp_gdrive: only {", ".join(NWP_EXTRACTORS)} is extracted here, so there is '
                             f'nothing to download for {self.model}; clear it')
        return self

    @model_validator(mode="after")
    def set_model_defaults(self) -> "NwpModelConfig":
        defaults = {
            "skiron": (0.05, 4, "SKIRON"),
            "ecmwf": (0.1, 8, "ECMWF"),
            "gfs": (0.25, 2, "GFS"),
        }
        if self.model in defaults:
            resolution, levels, subdir = defaults[self.model]
            if self.resolution is None:
                self.resolution = resolution
            if self.levels is None:
                self.levels = levels
            if self.path_nwp_source and self.model == "ecmwf" and "ECMWF" in self.path_nwp_source:
                return self
            if self.path_nwp_source and not self.path_nwp_source.endswith(subdir):
                self.path_nwp_source = str(Path(self.path_nwp_source) / subdir)
        elif self.resolution is None or self.levels is None:
            raise ValueError('custom NWP providers need explicit resolution and levels for pre-extracted data')
        return self


class FoldersConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pycharm_path: str
    sys_folder: str
    nwp_folder: str
    sat_folder: str | None = None
    path_image: str | None = None
    check_exists: bool = False

    @model_validator(mode="after")
    def check_paths(self) -> "FoldersConfig":
        if self.check_exists:
            for name in ("sys_folder", "nwp_folder", "pycharm_path"):
                path = getattr(self, name)
                if not Path(path).exists():
                    raise FileNotFoundError(f"{path} does not exist")
        return self


class JobsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_cache_mode: Literal['prefer-local', 'online', 'offline', 'prewarmed'] | None = Field(default=None, description='Optional pretrained-model access policy. prefer-local and online download a model that is not on the machine; offline and prewarmed never do. Null inherits EFORECAST_MODEL_CACHE_MODE.')

    n_cpus: int = Field(default=16, ge=1, description='CPUs the CPU studies of a run use between them, never more than this process may use.')
    cpu_workers: int | None = Field(default=None, ge=1, description='CPU studies that run side by side. Empty (null): as many as there are studies, up to one a CPU. Trials inside a study stay serial.')
    cpu_study_threads: int | None = Field(default=None, ge=1, description='Most threads a study uses. Empty (null): the CPUs are divided among the studies running, and a study is given the threads of the ones that finish.')
    cpu_study_device: str | None = Field(default=None, description='Device preference of foundation studies: cpu, auto, cuda, cuda:N or a GPU UUID. Empty (null) inherits each family. A preference no visible card answers runs on the CPU.')
    n_jobs: int = Field(default=16, ge=1, description='Workers in dataset/preprocessing process pools.')
    n_jobs_rbfnn: int = Field(default=3, ge=1, description='Legacy setting; current GPU concurrency uses method_params.<family>.n_jobs.')
    n_jobs_lstm: int = Field(default=2, ge=1, description='Legacy setting; current GPU concurrency uses method_params.LSTM.n_jobs.')
    n_jobs_cnn_3d: int = Field(default=2, ge=1, description='Legacy setting; current GPU concurrency uses method_params.CNN.n_jobs.')
    n_jobs_cnn: int = Field(default=2, ge=1, description='Legacy setting; current GPU concurrency uses method_params.CNN.n_jobs.')
    n_jobs_mlp: int = Field(default=3, ge=1, description='Legacy setting; current GPU concurrency uses method_params.MLP.n_jobs.')
    intra_op: int = Field(default=2, ge=1, description='Legacy framework setting, stored for compatibility; not applied by the current Torch trainer.')
    n_gpus: int = Field(default=2, ge=0, description='Default GPU count for deep training. Zero permits CPU-only configuration; --gpus can override it.')

    #: Settings of what this pipeline no longer has -- throughput profiles, the managed dataset
    #: cache with its mapped readers, the allowances and the deadline of a reservation. A case
    #: file written when they existed still loads; the settings are dropped.
    RETIRED: ClassVar[frozenset] = frozenset({
        'execution_profile', 'profile_resources', 'array_reader', 'cache_max_bytes', 'cache_max_entries',
        'cache_batch_rows', 'cache_buffer_mb', 'cpu_study_ram_mb', 'cpu_study_vram_mb', 'cpu_wait_timeout'})

    @model_validator(mode='before')
    @classmethod
    def without_retired_settings(cls, value):
        if isinstance(value, dict) and cls.RETIRED & set(value):
            return {key: item for key, item in value.items() if key not in cls.RETIRED}
        return value

    @field_validator('*', mode='before')
    @classmethod
    def integer_not_boolean(cls, value):
        if isinstance(value, bool):
            raise ValueError('resource counts must be integers, not booleans')
        return value


class ProjectConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    group: str
    owner: str
    version_model: int = Field(ge=0)
    version_group: int = Field(ge=0)
    horizon_type: HorizonType
    horizon: int
    coordinates: list[float] | dict[str, Any]
    area_group: list[list[float]]
    use_images: list[str] | bool | None = None
    image_coord: dict[str, Any] | None = None
    image_size: dict[str, Any] | None = None
    site_indices: dict[str, Any] | None = None
    rated_power: float | int | None = Field(default=None, gt=0, allow_inf_nan=False)
    type: ProblemType
    filename: str
    evaluation_start: str | None = None
    local_timezone: str = "UTC"
    site_timezone: str = "UTC"
    country: str | None = None
    ts_resolution: str
    is_global: bool = False
    is_fuzzy: bool = False
    is_probabilistic: bool = False
    docker: bool = False
    nwp_models: list[NwpModelConfig] = Field(default_factory=list)

    @field_validator("coordinates")
    @classmethod
    def valid_coordinates(cls, value: list[float] | dict[str, Any]) -> list[float] | dict[str, Any]:
        def check_seq(seq: Any, where: str) -> int:
            if not isinstance(seq, list) or len(seq) not in (2, 4):
                raise ValueError(
                    f"{where} must be [lat, long] or [lat_min, long_min, lat_max, long_max]")
            if not all(isinstance(item, (int, float)) and not isinstance(item, bool)
                       for item in seq):
                raise ValueError(f"{where} must contain numbers only")
            return len(seq)

        if isinstance(value, dict):
            if not value:
                raise ValueError("coordinates dict must not be empty")
            lengths = {check_seq(coords, f"coordinates[{city!r}]")
                       for city, coords in value.items()}
            if len(lengths) > 1:
                raise ValueError(
                    "coordinates dict values must all be points or all be areas")
        else:
            check_seq(value, "coordinates")
        return value

    @field_validator("ts_resolution")
    @classmethod
    def valid_resolution(cls, value: str) -> str:
        from eforecast.common_utils.resolutions import SPELLINGS
        value = SPELLINGS.get(value, value)
        if value not in RESOLUTION_STEP:
            raise ValueError(f"ts_resolution must be one of {', '.join(RESOLUTION_STEP)}")
        return value

    @field_validator("evaluation_start", mode="before")
    @classmethod
    def valid_evaluation_start(cls, value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value.strftime("%Y-%m-%d %H:%M")
        text = str(value).strip()
        if not text:
            return None
        for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(text, fmt).strftime("%Y-%m-%d %H:%M")
            except ValueError:
                continue
        raise ValueError('evaluation_start must match "%Y-%m-%d %H:%M" or be null')

    @model_validator(mode="after")
    def valid_horizon(self) -> "ProjectConfig":
        if self.horizon_type == HorizonType.SHORT_TERM:
            if self.horizon <= 0:
                raise ValueError(
                    "horizon must be greater than zero when horizon_type is short_term")
        else:
            self.horizon = 0
        return self


class VariableConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    type: str = "nwp"
    source: str = "grib"
    lags: None | int | list[int | float] = None
    timezone: str = "UTC"
    transformer: None | str | list[str] = None
    transformer_params: dict[str, Any] | None = None
    bands: list[str] | None = None
    nwp_provider: None | str = None
    use_diff_between_lags: bool = False

    @model_validator(mode='after')
    def valid_cyclical(self) -> 'VariableConfig':
        """``cyclical`` turns one series into its sine and cosine, which only a series has."""
        if not is_cyclical(self.transformer):
            return self
        if self.type != 'timeseries':
            raise ValueError(f"{self.name}: the cyclical transformation applies to a timeseries variable, "
                             f"not to one of type {self.type}")
        period = (self.transformer_params or {}).get('cyclical')
        if period is not None and (isinstance(period, bool) or not isinstance(period, (int, float))
                                   or not math.isfinite(period) or period <= 0):
            raise ValueError(f"{self.name}: transformer_params.cyclical is the period, a positive number "
                             f"(360 for degrees, 24 for hours); it is {period!r}")
        return self


def is_cyclical(transformer: None | str | list[str]) -> bool:
    return 'cyclical' in (transformer if isinstance(transformer, list) else [transformer])


class TargetVariableConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    source: str
    lags: None | int | list[int | float] = None
    columns: list[str] = Field(min_length=1)
    transformer: None | str | list[str] = None
    transformer_params: dict[str, Any] | None = None

    @model_validator(mode='after')
    def valid_outputs(self) -> 'TargetVariableConfig':
        if any(not column.strip() for column in self.columns) or len(set(self.columns)) != len(self.columns):
            raise ValueError('target columns must be non-empty and unique')
        if isinstance(self.lags, int):
            # ``lags: 0`` is the legacy spelling of "no explicit lags", the same convention
            # ``loader._variable_lags`` uses for every other variable; the outputs are then
            # derived from the horizon.  A positive integer is a count of steps.
            if self.lags < 0:
                raise ValueError('target lag count must not be negative')
            self.lags = list(range(self.lags)) or None
        if self.lags is not None:
            if not self.lags or any(not math.isfinite(lag) or lag < 0 or int(lag) != lag for lag in self.lags):
                raise ValueError('target lags must be nonnegative integer steps')
            if len(set(self.lags)) != len(self.lags) or len(self.lags) != len(self.columns):
                raise ValueError('target lags must be unique and match the number of target columns')
            self.lags = [int(lag) for lag in self.lags]
        return self


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    nwp_data_merge: list[Any] = Field(default_factory=lambda: [None])
    compress_data: list[Any] = Field(default_factory=lambda: [None])
    img_data: list[Any] = Field(default_factory=lambda: [None])
    scale_row_method: list[Any] = Field(default_factory=lambda: [None])
    scale_img_method: list[Any] = Field(default_factory=lambda: [None])
    scale_nwp_method: list[Any] = Field(default_factory=lambda: [None])
    scale_target_method: str = "maxabs"
    use_data_before_and_after_target: bool = True
    # which NWP analysis cycle may answer a sample: the newest one that existed at its issue time
    # (``as_of``), or the merged day as before, where a later cycle overwrote an earlier one
    nwp_selection_policy: Literal['as_of', 'legacy_merge'] = Field(
        default='as_of',
        description='as_of uses the newest analysis cycle available at the sample issue time; '
                    'legacy_merge reproduces datasets built before this choice existed.')
    nwp_availability_hours: float = Field(
        default=0.0, ge=0, le=48, allow_inf_nan=False,
        description='How long after its analysis time a cycle can be read. 0 keeps only the '
                    'necessary constraint; declare the provider delay to be exact.')

    @field_validator("nwp_data_merge", "compress_data", "img_data", "scale_row_method",
                     "scale_img_method", "scale_nwp_method", mode="before")
    @classmethod
    def option_list(cls, value: Any) -> list[Any]:
        """Legacy option lists: ``null`` or an empty list means "unused" (``[None]``), a scalar is one option."""
        if value is None:
            return [None]
        if isinstance(value, (list, tuple, set)):
            return list(value) if len(value) > 0 else [None]
        return [value]
    remove_night_hours: bool = True
    use_polars: bool = False
    global_past_lags: list[int | float] | None = None
    global_future_lags: list[int | float] | None = None
    time_merge_variables: dict[str, Any] = Field(default_factory=dict)
    variables: list[VariableConfig] = Field(default_factory=list)
    target_variable: TargetVariableConfig


class MethodsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    env_name: str = Field(default='', title='Legacy environment name',
        description='Preserved for compatibility. This does not select the interpreter; see the runtime readiness report.',
        json_schema_extra={'readOnly': True, 'deprecated': True})
    path_env: str = Field(default='', title='Legacy interpreter path',
        description='Preserved for compatibility. Runs use the configured runtime, not this stored path.',
        json_schema_extra={'readOnly': True, 'deprecated': True})
    runtime_ids: dict[str, str] = Field(default_factory=dict,
        description='Advanced: approved runtime ids by method, or default. Empty preserves current execution. '
                    'Use native for the existing interpreter; other ids require a maintainer registry and executor support.')
    backend: str

    @field_validator('runtime_ids')
    @classmethod
    def validate_runtime_ids(cls, value):
        from eforecast.runtime_registry import selections
        result = selections({'runtime_ids': value})
        unknown = set(result) - {'default', *GLOBAL_METHODS, 'clustering', 'preprocessing', 'feature_selection'}
        if unknown:
            raise ValueError(f'unknown runtime method(s): {", ".join(sorted(unknown))}')
        return result
    cpu_scope: Literal['global', 'clusters', 'both'] = Field(default='clusters',
        description='Shallow CPU training scope. Legacy cases keep cluster-only training; global uses enabled shallow global_methods.')
    methods_cpu: dict[str, bool] = Field(default_factory=dict)
    cluster_methods: dict[str, bool] = Field(default_factory=dict)
    global_methods: dict[str, bool] = Field(default_factory=dict)
    transfer_learning: bool = False
    transfer_learning_from: dict[str, Any] = Field(default_factory=dict)
    transfer_learning_merge_data: bool = True
    feature_selection_methods: list[Any] = Field(default_factory=list)
    combine_methods: list[str] = Field(default_factory=list)
    max_performance: float | int = Field(default=2, gt=0, allow_inf_nan=False)
    val_test_ratio: float = Field(default=0.125, gt=0, lt=0.5, allow_inf_nan=False)
    keep_best_models: int = Field(default=4, ge=1)
    selection_fraction: float = Field(default=0.5, gt=0, lt=1, allow_inf_nan=False)
    # the share of the selection part kept back for the conformal bands, so they are not fitted on
    # the dates that chose the predictor; 0 restores the single-slice behaviour, with a warning
    calibration_fraction: float = Field(
        default=0.3, ge=0, lt=1, allow_inf_nan=False,
        description='Share of the selection slice reserved for fitting prediction intervals; '
                    '0 fits them on the selection slice, which under-covers.')
    # what happens to a timestamp no selected cluster covers: complete it from the global model
    # as a separately named predictor, or leave the gap and report it
    cluster_coverage_policy: Literal['fallback_global', 'abstain'] = Field(
        default='fallback_global',
        description='Timestamps no selected cluster covers: complete them from the global model as a '
                    'separate predictor, or abstain and report the gap.')
    # empty means no prediction intervals: the calibrate stage then has nothing to fit.
    # The default is said in the JSON Schema as well. A default that comes from a factory is
    # not, and a form that is told nothing writes an empty list for an array it shows: opening
    # the methods of a case that left this key out, and saving, switched its intervals off.
    interval_coverage: list[Annotated[float, Field(gt=0, lt=1, allow_inf_nan=False)]] = Field(
        default_factory=lambda: [0.8, 0.9], json_schema_extra={'default': [0.8, 0.9]})
    CVs: int = Field(default=5, ge=1)
    split_seed: int = Field(default=1, ge=0, le=2 ** 31 - 1, description='Seed for date splits and default training/search randomness.')
    split_policy: Literal['frozen', 'legacy_resplit'] = Field(
        default='frozen',
        description='How the train/validation/test partition reaches the learned components. '
                    '"frozen": the global split is drawn once and every cluster takes the intersection of it '
                    'with its membership, and feature selection is fitted on train and assessed on validation, '
                    'so nothing selects a model on data the clusterer or the selector has seen. '
                    '"legacy_resplit": the behaviour of models trained before 2026-10-13, where the cluster '
                    'stage redrew the whole partition after the clusterer was fitted and feature selection was '
                    'assessed on the internal test mask. Use it only to reproduce such a model.')
    kernels: dict[str, Any] = Field(default_factory=dict)
    base_data_types: dict[str, Any] = Field(default_factory=dict)
    method_params: dict[str, dict[str, Any]] = Field(default_factory=dict)

    @field_validator('backend')
    @classmethod
    def valid_backend(cls, value):
        if value not in BACKENDS:
            raise ValueError(f'backend must be one of {", ".join(BACKENDS)}')
        return value

    @field_validator('methods_cpu', 'cluster_methods', 'global_methods')
    @classmethod
    def valid_methods(cls, value, info):
        supported = {'methods_cpu': CPU_METHODS, 'cluster_methods': CLUSTER_METHODS, 'global_methods': GLOBAL_METHODS}[info.field_name]
        unknown = [name for name, enabled in value.items() if enabled and name not in supported]
        if unknown:
            raise ValueError(f'{info.field_name} enables unsupported methods: {", ".join(unknown)}')
        return value

    @field_validator('combine_methods')
    @classmethod
    def valid_combiners(cls, value):
        unknown = [name for name in value if name not in COMBINE_METHODS]
        if unknown:
            raise ValueError(f'unsupported combine_methods: {", ".join(unknown)}; implemented: {", ".join(COMBINE_METHODS)}')
        return value

    @field_validator('interval_coverage')
    @classmethod
    def unique_coverages(cls, value):
        if len(set(value)) != len(value):
            raise ValueError('interval_coverage must contain unique probabilities')
        return value

    @field_validator('method_params')
    @classmethod
    def valid_parameters(cls, value):
        validate_family_parameters(value)
        return value

    @model_validator(mode='after')
    def configured_enabled_methods(self) -> 'MethodsConfig':
        missing = {name for flags in (self.methods_cpu, self.cluster_methods, self.global_methods)
                   for name, enabled in flags.items() if enabled and name != 'GRAPH' and name not in self.method_params}
        if missing:
            raise ValueError(f'enabled methods need method_params families: {", ".join(sorted(missing))}')
        return self


def _sorted_choices(values: Any) -> list[Any]:
    """A set has no order; give it a stable one, numeric where that makes sense."""
    items = list(values)
    if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in items):
        return sorted(items)
    return sorted(items, key=repr)


class LayerParam(BaseModel):
    """One parameter of one layer: a fixed value, a set of choices, or a numeric range.

    ``choice`` and ``range`` are search dimensions -- Optuna picks a value per trial
    (:func:`eforecast.training.command_line.search_space.SearchSpace.define_structure_space`);
    ``fixed`` is used as it stands.  Shorthands accepted on input and normalised here:

        size: 64                    ->  {tag: fixed,  value: 64}
        filters: {choices: [16,32]} ->  {tag: choice, value: [16, 32]}
        pool: {range: [1, 3]}       ->  {tag: range,  value: [1, 3]}
    """

    model_config = ConfigDict(extra="forbid")

    tag: Literal["fixed", "choice", "range"] = "fixed"
    value: Any = None

    @model_validator(mode="before")
    @classmethod
    def normalize(cls, value: Any) -> Any:
        if isinstance(value, dict):
            if "tag" in value or set(value) <= {"value"}:
                return value
            if set(value) == {"choices"}:
                return {"tag": "choice", "value": list(value["choices"])}
            if set(value) == {"range"}:
                return {"tag": "range", "value": list(value["range"])}
            if set(value) == {"values"}:                       # an older editor form
                return {"tag": "choice", "value": list(value["values"])}
        if isinstance(value, set):
            return {"tag": "choice", "value": _sorted_choices(value)}
        return {"tag": "fixed", "value": value}

    @model_validator(mode="after")
    def validate_value(self) -> "LayerParam":
        if self.tag == "choice":
            if not isinstance(self.value, list) or not self.value:
                raise ValueError("a choice parameter needs a non-empty list of values")
        elif self.tag == "range":
            if not isinstance(self.value, list) or len(self.value) != 2:
                raise ValueError("a range parameter needs exactly [low, high]")
            if not all(isinstance(bound, (int, float)) and not isinstance(bound, bool) for bound in self.value):
                raise ValueError("the bounds of a range must be numbers")
            if self.value[0] > self.value[1]:
                raise ValueError("the low bound of a range must not exceed the high bound")
        return self

    def resolved(self) -> Any:
        """The value to use when nothing searches this parameter: the first choice, the low bound."""
        if self.tag == "choice":
            return self.value[0]
        if self.tag == "range":
            return self.value[0]
        return self.value

    def legacy(self) -> Any:
        """``set`` for choices and ``list`` for a range -- what the search space has always read."""
        if self.tag == "choice":
            try:
                return set(self.value)
            except TypeError:
                return list(self.value)
        if self.tag == "range":
            return list(self.value)
        return self.value


def _legacy_size(size: Any) -> Any:
    """What a bare ``size`` has always meant, as a tagged parameter.

    The search space read the Python type: a ``set`` was a set of choices, a ``list`` was a range
    from its first entry to its last, a one-entry list was that entry, and an empty list was the
    placeholder of a layer that takes no size.  Keeping that mapping here is what makes the tag the
    editor shows agree with what the search actually does.
    """
    if isinstance(size, dict):
        return size                       # {choices: ...}, {range: ...} or an already tagged entry
    if isinstance(size, set):
        return {"tag": "choice", "value": _sorted_choices(size)}
    if isinstance(size, (list, tuple)):
        values = list(size)
        if len(values) == 1:
            return {"tag": "fixed", "value": values[0]}
        if len(values) > 1:
            numeric = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values)
            if numeric:
                return {"tag": "range", "value": [values[0], values[-1]]}
            return {"tag": "choice", "value": values}
        return {"tag": "fixed", "value": []}
    return {"tag": "fixed", "value": size}


class NetworkLayerConfig(BaseModel):
    """A layer of an experiment: its type and every parameter underneath ``params``.

    There is no ``size`` field of its own any more -- the width, the kernel or the attention name is
    ``params.size`` like every other parameter, so all of them can be fixed or searched the same way.
    The older forms (``[type, size]``, a top-level ``size`` / ``choices`` / ``range``, bare extra
    keys) are still read and folded into ``params``.
    """

    model_config = ConfigDict(extra="forbid")

    type: str
    name: str | None = None
    params: dict[str, LayerParam] = Field(default_factory=dict)
    enabled: bool = True

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_layer(cls, value: Any) -> Any:
        if isinstance(value, (list, tuple)):
            if len(value) != 2:
                raise ValueError("legacy layer definitions must contain [type, size]")
            return {"type": value[0], "params": {"size": _legacy_size(value[1])}}
        if not isinstance(value, dict):
            return value
        data = dict(value)
        if "layer" in data:
            layer = data.pop("layer")
            if "type" in data and data["type"] != layer:
                raise ValueError("layer and type cannot define different values")
            data["type"] = layer
        params = dict(data.pop("params", None) or {})
        # every key that is not a field of this model is a layer parameter
        params.update({key: data.pop(key) for key in list(data) if key not in {"type", "name", "enabled"}
                       and key not in {"size", "choices", "range"}})
        # the former dedicated size fields, in order of precedence
        choices, range_ = data.pop("choices", None), data.pop("range", None)
        size = data.pop("size", None)
        if choices is not None and range_ is not None:
            raise ValueError("choices and range are mutually exclusive")
        if size is not None and (choices is not None or range_ is not None):
            raise ValueError("size cannot be combined with choices or range")
        if choices is not None:
            params["size"] = {"tag": "choice", "value": list(choices)}
        elif range_ is not None:
            params["size"] = {"tag": "range", "value": list(range_)}
        elif size is not None:
            params["size"] = _legacy_size(size)
        data["params"] = params
        return data

    def param(self, name: str) -> Any:
        """The resolved value of one parameter, or ``None`` when it is not configured."""
        entry = self.params.get(name)
        return None if entry is None else entry.resolved()

    def legacy_size(self) -> Any:
        """What the builder receives positionally: the value, a ``set`` of choices or a ``[lo, hi]`` list."""
        entry = self.params.get("size")
        return None if entry is None else entry.legacy()


class NetworkExperimentConfig(RootModel[dict[str, list[NetworkLayerConfig]]]):
    @model_validator(mode="after")
    def validate_output_branch(self) -> "NetworkExperimentConfig":
        if "output" not in self.root:
            raise ValueError("network experiments must define an output branch")
        return self


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project: ProjectConfig
    folders: FoldersConfig
    jobs: JobsConfig = Field(default_factory=JobsConfig)
    data: DataConfig
    methods: MethodsConfig
    experiments: dict[str, NetworkExperimentConfig] | None = None
    experiments_module: str | None = None

    @model_validator(mode="after")
    def check_clusterer_of_a_fuzzy_project(self) -> "ExperimentConfig":
        """A fuzzy project is trained per cluster, so it needs the ``clustering`` family.

        ``run_training`` asks ``ClusterOrganizer`` for the clusters before any method runs, and that
        reads ``methods.method_params.clustering``.  Catching it here turns a KeyError deep inside
        the clusterer into something the editor can show.
        """
        if self.project.is_fuzzy and "clustering" not in self.methods.method_params:
            raise ValueError(
                "is_fuzzy is on, so the case is trained per cluster, but methods.method_params has "
                "no 'clustering' family and the clusterer has nothing to read. Press Sync on the "
                "Methods tab to add it from the presets, or turn is_fuzzy off."
            )
        if self.data.target_variable.lags is None:
            expected = int(self.project.horizon / lag_step(self.project.ts_resolution)) \
                if self.project.horizon_type == HorizonType.SHORT_TERM else 1
            if len(self.data.target_variable.columns) != expected:
                raise ValueError(f'data.target_variable.columns needs {expected} output(s) for this horizon; '
                                 'provide matching explicit target lags for a custom output selection')
        if not self.experiments_module:
            validate_experiment_references(self, self.experiments or {})
        return self
