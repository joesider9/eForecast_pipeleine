# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Implemented pipeline capabilities, shared by validation, CLIs and UI catalogues."""

FOUNDATION_METHODS = ('Chronos2', 'TimesFM', 'TiRex')
CPU_METHODS = ('RF', 'CatBoost', 'lightgbm', 'lasso', 'RBFols', 'GA_RBFols') + FOUNDATION_METHODS
CLUSTER_METHODS = ('CNN', 'TIMM', 'LSTM', 'TRANSFORMER', 'MLP', 'CrossViVit', 'Distributed')
GLOBAL_METHODS = ('CrossViVit', 'CNN', 'TIMM', 'TRANSFORMER', 'Distributed', 'LSTM', 'MLP', 'GRAPH') + CPU_METHODS
CLUSTERING_METHODS = ('RBF',)
COMBINE_METHODS = ('mean', 'bcp', 'elastic_net', 'kmeans')
BACKENDS = ('TORCH', 'TF_2', 'TF_1')  # legacy configurations still load; current deep CLI requires TORCH
NWP_EXTRACTORS = ('ecmwf',)

NO_SEQUENCES = 'a long_term case has no sequence dataset'


def has_sequences(static_data):
    """Whether the case is given a sequence ('lstm') dataset.

    The rows of a ``long_term`` case are indexed by (prediction time, issue time): every predicted
    hour appears once per issue day. A sequence is a run of rows shifted along one clock, and
    there is no single clock in that index to shift along, so such a case never had one -- the
    per-project scripts this runner replaced built it ``if horizon_type == 'short_term'`` only.
    """
    return static_data.get('horizon_type') != 'long_term'


def foundation_device(static_data, method):
    """Where a foundation family of this case computes: ``cpu``, ``auto``, ``cuda``, ``cuda:N`` or a GPU UUID.

    The run's or the case's ``cpu_study_device`` if there is one, else the family's own
    ``device``, else ``auto``: a card when one is there. Nothing a foundation model computes is
    learned, so the device is not part of what a trial is -- a model that was studied on the
    CPU is predicted on a card when the case says so by then, and the other way round. Its
    numbers then differ in the last digits a float holds, as they do between two CPUs.
    """
    family = static_data.get(method)
    own = family.get('device') if isinstance(family, dict) else None
    return static_data.get('cpu_study_device') or own or 'auto'
