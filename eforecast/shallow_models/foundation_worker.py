# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""One foundation operation: the model is loaded, used and dropped inside this worker."""
from pathlib import Path

import joblib

from eforecast.compute_worker import current_worker


def execute(payload, context):
    if current_worker() != context:
        raise RuntimeError('a foundation model is loaded in its own worker process')
    from .foundation_model import FoundationModel, clear_pipelines
    root = Path(context.result_directory)
    values = joblib.load(root / 'inputs/model.pickle')
    from eforecast.config.stage_workspace import Workspace
    view = Workspace(values['static'], {'nodes': [{'key': 'features'}], 'products': {}, 'options': {}}, 'features')
    view.root, view.view = root / 'view', root / 'view/model'
    view.maps = [(view.base, view.view)]
    view._copy(view.base, view.view)
    view.static = view.rebase(view.static)
    output = root / 'artifacts'
    output.mkdir()
    model = FoundationModel(view.static, str(output), params=view.rebase(values['params']), refit=True,
                            online=values['online'], train=values['train'], n_jobs=context.threads)
    try:
        if payload['operation'] == 'fit':
            score = model.fit(values['masks'])
            # A too-small slice still has a resumable, untrained marker.
            model.params = view.rebase(model.params, canonical=True)
            model.save()
            return {'score': float(score), 'artifacts': ['artifacts/net_weights.pickle'],
                    'batch_size': values['params']['batch_size']}
        median = model.predict(values['dates'])
        joblib.dump({'median': median, 'quantiles': model.quantile_predictions}, output / 'predictions.pickle')
        return {'artifacts': ['artifacts/predictions.pickle'], 'batch_size': values['params']['batch_size']}
    finally:
        # Removing references helps ordinary teardown; the supervisor does not
        # release until the process tree has exited, even on an error/traceback.
        clear_pipelines()
