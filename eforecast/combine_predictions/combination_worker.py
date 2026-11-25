# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Combination target: shapes and caches are validated in the worker that made them."""
from pathlib import Path
import joblib

from eforecast.compute_worker import current_worker
from eforecast.config import stage_artifacts
from .inference_worker import compute


def validation(values):
    """The worker-side contract itself, separate from the worker so it can be checked directly."""
    def validate(view, result):
        records = stage_artifacts.output_records(view.static, view.rebase(values['plan']), values['operation'])
        if values['operation'] == 'predict':
            required = {str(Path(r['path']).parent) for r in records['predictions']
                        if Path(r['path']).name == 'predictions_eval.pickle'}
            if set(result) != required:
                raise ValueError('prediction did not return every required model output')
            for folder, expected in result.items():
                if Path(folder).parent.name != 'GRAPH' and not joblib.load(Path(folder) / 'predictions_eval.pickle').equals(expected):
                    raise ValueError('prediction values were not fully published')
        return {str(view.base / Path(r['path']).relative_to(view.model)): r['sha256']
                for group in records.values() for r in group}
    return validate


def execute(payload, context):
    if current_worker() != context:
        raise RuntimeError('combination requires confirmed attachment')
    values = joblib.load(Path(context.result_directory) / 'inputs/inference.pickle')
    return compute(values, context, validate=validation(values))
