# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""CatBoost also needs bounds on implicit post-fit importance and score calls."""
from functools import wraps
from inspect import signature

from catboost import CatBoostRegressor, Pool

from .execution import jobs

FIT_PARAMETERS = signature(CatBoostRegressor.fit)


def _fit_with_bounded_pools(self, *args, **kwargs):
    values = FIT_PARAMETERS.bind(self, *args, **kwargs)
    data = values.arguments.get('X')
    count = jobs(self.get_param('thread_count'))
    column_description = values.arguments.get('column_description')
    if data is not None and not isinstance(data, Pool):
        # CatBoost's internal _build_train_pool does not forward the model's
        # thread_count. Build the same Pool explicitly, including weights and
        # categorical metadata, before entering fit's native work.
        fields = {'cat_features': 'cat_features', 'text_features': 'text_features',
                  'embedding_features': 'embedding_features', 'graph': 'graph',
                  'sample_weight': 'weight', 'baseline': 'baseline',
                  'column_description': 'column_description'}
        pooled = {output: values.arguments.get(name) for name, output in fields.items()}
        for name in ('cat_features', 'text_features', 'embedding_features'):
            if pooled[name] is None:
                pooled[name] = self.get_param(name)
        data = Pool(data, label=values.arguments.get('y'), thread_count=count, **pooled)
        values.arguments['X'] = data
        for name in ('y', *fields):
            values.arguments[name] = None  # these now belong to the Pool

    def evaluation(value):
        if isinstance(value, tuple) and len(value) == 2:
            return Pool(value[0], label=value[1], thread_count=count,
                        cat_features=data.get_cat_feature_indices(),
                        text_features=data.get_text_feature_indices(),
                        embedding_features=data.get_embedding_feature_indices())
        if isinstance(value, list):
            return [evaluation(item) for item in value]
        if isinstance(value, str):
            return Pool(value, column_description=column_description, thread_count=count)
        return value  # existing Pool or None retains its data/label identity

    if isinstance(data, Pool) and values.arguments.get('eval_set') is not None:
        values.arguments['eval_set'] = evaluation(values.arguments['eval_set'])
    return CatBoostRegressor.fit(*values.args, **values.kwargs)


def _bounded(method):
    parameters = signature(method)

    @wraps(method)
    def call(self, *args, **kwargs):
        values = parameters.bind(self, *args, **kwargs)
        values.arguments['thread_count'] = jobs(self.get_param('thread_count'))
        if 'task_type' in parameters.parameters:
            values.arguments['task_type'] = 'CPU'
        return method(*values.args, **values.kwargs)
    return call


class SelectorCatBoostRegressor(CatBoostRegressor):
    fit = wraps(CatBoostRegressor.fit)(_fit_with_bounded_pools)
    # CatBoost.fit calls get_feature_importance without forwarding thread_count.
    get_feature_importance = _bounded(CatBoostRegressor.get_feature_importance)
    # Both public predict and score enter _predict; score hardcodes -1, so just
    # bounding predict would leave permutation scoring outside the CPU budget.
    _predict = _bounded(CatBoostRegressor._predict)
