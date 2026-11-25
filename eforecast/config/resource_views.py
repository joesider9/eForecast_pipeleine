# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Read-only normalization of a run's resource snapshot for the typed UI.

Keep the raw fields beside this view. Missing measurements stay unknown; corrupt
optional sections degrade independently and never rewrite/reject a stored case.
A snapshot written by a version that reserved resources is read for what it still says:
what was running and what was measured.
"""
import math
from .resource_contract import ResourceView, StudyResources, StudyDispatch, ResourceProgress


def mapping(value):
    return value if isinstance(value, dict) else {}


def number(value):
    try:
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None
    except OverflowError:
        return None


def text(value):
    return value if isinstance(value, str) else None


def project(snapshot):
    raw = mapping(snapshot)
    running, reserved, observed = (mapping(raw.get(k)) for k in ('running', 'reservations', 'observed'))
    threads = number(running.get('cpu_threads'))
    view = ResourceView(updated=number(raw.get('updated')), phase=text(raw.get('phase')),
        error=text(raw.get('error')),
        running_threads=threads if threads is not None else number(reserved.get('cpu_threads')),
        trial_limit=number(running.get('at_most_trials')),
        pss_mb=number(observed.get('pss_mb')), pss_confidence=text(observed.get('confidence')))
    if number(running.get('trials')) is not None:
        view.source = 'gpu_trials'
    elif isinstance(raw.get('dispatch'), dict):
        view.source = 'cpu_studies'
        dispatch = dict(raw['dispatch'])
        dispatch.setdefault('total_threads', dispatch.get('effective_total_threads'))
        view.dispatch = StudyDispatch(**{key: number(dispatch.get(key)) for key in StudyDispatch.model_fields})
    elif isinstance(raw.get('policy'), dict):
        view.source = 'worker'
    if 'studies' in raw and not isinstance(raw['studies'], list):
        view.warnings.append('Study telemetry is unreadable.')
    for index, value in enumerate(raw.get('studies', []) if isinstance(raw.get('studies'), list) else []):
        if not isinstance(value, dict):
            view.warnings.append('One study entry is unreadable.')
            continue
        estimate = mapping(value.get('estimate'))
        state = text(value.get('state')) or 'unknown'
        view.studies.append(StudyResources(id=text(value.get('id')) or str(index),
            method=text(value.get('method')) or 'unknown', cluster=text(value.get('cluster')) or 'unknown',
            state=state, reason=text(value.get('reason')) if state == 'waiting' else None,
            threads=number(estimate.get('cpu_threads')), requested_device=text(estimate.get('requested_device')),
            resolved_device=text(estimate.get('device')), normalization=text(estimate.get('normalization'))))
    progress = mapping(raw.get('progress'))
    if raw.get('progress') is not None:
        values = {key: number(progress.get(key)) for key in ResourceProgress.model_fields}
        valid = all(value is not None and value >= 0 for value in values.values())
        if valid and all(values[key].is_integer() for key in ('verified_complete', 'pruned', 'failed')):
            view.progress = ResourceProgress(**values)
        else:
            view.warnings.append('Completion-rate telemetry is unavailable.')
    return view.model_dump(mode='json')
