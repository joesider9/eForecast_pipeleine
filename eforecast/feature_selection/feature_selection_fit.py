# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Fit feature selectors per cluster, method and dataset (training side).

For every dataset (tag + data options) the data is loaded once and reused for
all clusters and methods.  A selector ranks the features with one estimator
(lasso coefficients, random-forest importances, Boruta history, CatBoost SHAP
values), builds candidate feature sets from the top fractions of that ranking,
scores every distinct candidate with a LightGBM model on the test dates and
keeps the best.  ``Leshy`` / ``BoostAGroota`` (arfs) return their own support.

LSTM datasets are selected in two passes: the (samples, lags, variables)
tensor is compressed along the lag axis (one linear component per lag) to
select lags, then along the variable axis on the kept lags to select
variables.

Products: ``<cluster>/feature_selectors.pickle`` = ``{name: {'indices',
'names'} | {'lags', 'variables'}}`` and the backup ``feature_selectors.pickle``
in the model folder.  Names come from
:func:`eforecast.feature_selection.feature_selection_transform.selector_name`.

The selectors of one dataset are fitted side by side when the stage was given
more than one CPU (:func:`eforecast.feature_selection.execution.side_by_side`),
each estimator with its part of them, and are saved in the order they would
have been fitted in one after another.
"""
from contextlib import closing
import itertools
import logging
import os
import threading
import time

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from boruta import BorutaPy
from catboost import Pool
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LassoCV, MultiTaskLasso, MultiTaskLassoCV
from sklearn.model_selection import train_test_split

import eforecast.feature_selection.arfs.feature_selection.allrelevant as arfsgroot
from eforecast.common_utils import trial_layout
from eforecast.common_utils.dataset_utils import load_data_shallow_models, sync_data_with_dates
from eforecast.datasets.data_feeder import DataFeeder
from eforecast.datasets.files_manager import FilesManager
from eforecast.feature_selection.feature_selection_transform import selector_name
from eforecast.feature_selection.estimators import SelectorCatBoostRegressor as CatBoostRegressor
from eforecast.feature_selection.execution import (jobs, limits, check_cancelled, share, shares,
                                                    side_by_side)

log = logging.getLogger(__name__)
# Selectors fitted side by side cut their rows from the same frames. pandas builds an index's
# lookup table on its first use and does not promise that two threads may ask for it at once,
# so the cut is taken one selector at a time. It is quick; the fit that follows is not.
_shared = threading.Lock()

CategoricalFeatures = ['dayweek', 'hour', 'month', 'sp_index']
CANDIDATE_FRACTIONS = (0.1, 0.15, 0.25, 0.35, 0.5, 0.75, 1.0)  # of the features, best first by importance
MIN_FEATURES = 4
SCORED_OUTPUTS = 3  # short_term targets: LightGBM scores this many evenly spaced horizons
ARFS_METHODS = {'Leshy', 'BoostAGroota'}
ARFS_TREES = 100  # CatBoost trees per fit inside the all-relevant methods
ARFS_MAX_SAMPLES = 20000  # rows used by the all-relevant methods (random subset above this size)
BOOSTAGROOTA_ITERS = 8  # shadow reshuffles averaged per BoostAGroota round (package default range 5-10)


class FeatureSelector:
    _granted = 1        # the CPUs of a selector nobody constructed (a test, a debugger): one

    @property
    def n_jobs(self):
        """The threads an estimator gets here: the stage's CPUs, or the part of them that a
        selector fitted beside others has (:func:`eforecast.feature_selection.execution.share`)."""
        return share(self._granted)

    @n_jobs.setter
    def n_jobs(self, granted):
        self._granted = granted

    def __init__(self, static_data, recreate=False, online=False, train=False):
        self.feature_selectors = dict()
        self.online = online
        self.train = train
        self.static_data = static_data
        self.is_Fuzzy = self.static_data['is_Fuzzy']
        self.clusters = dict()
        if self.is_Fuzzy:
            clusters_file = os.path.join(static_data['path_model'], 'clusters.pickle')
            if not os.path.exists(clusters_file):
                raise ImportError(f'Cannot find {clusters_file}: run the clustering stage first')
            self.clusters = trial_layout.cluster_folders(static_data)
        if any(list(static_data['global_methods'].values())):
            cluster_path = os.path.join(static_data['path_model'], 'global')
            os.makedirs(cluster_path, exist_ok=True)
            self.clusters.update({'global': cluster_path})
        self.recreate = recreate
        self.nwp_data_merge = self.static_data['nwp_data_merge']
        self.nwp_data_compress = self.static_data['compress_data']
        self.dataset_tags = set([tag for exp in self.static_data['experiments'].values() for tag in exp.keys()
                                 if tag not in {'output', 'hidden_layer'}])
        from eforecast.config.capabilities import has_sequences
        if not has_sequences(self.static_data):
            self.dataset_tags.discard('lstm')       # a long_term case has no sequence dataset to select from
        self.feature_selection_methods = [m for m in self.static_data['feature_selection_methods'] if m is not None]
        self.calendar_variables = [var_data['name'] for var_data in self.static_data['variables']
                                   if var_data['type'] == 'calendar']
        self.rated = static_data['rated']
        self.n_jobs = jobs(static_data['n_jobs'])
        log.info('feature selection requested jobs=%s; granted %s: selectors side by side when a dataset has '
                 'several, else the estimator/CV; BLAS=1', static_data['n_jobs'], self.n_jobs)
        self.files_manager = FilesManager(static_data, is_online=online)
        self.data_feeder = DataFeeder(self.static_data, online=self.online, train=self.train)

    # ------------------------------------------------------------------ estimators

    def estimator(self, method, alpha=None, multi_output=False, size=50000, m_estimator=1):
        loss_function = 'MultiRMSE' if self.static_data['horizon_type'] == 'short_term' else 'RMSE'
        model = CatBoostRegressor(loss_function=loss_function, allow_writing_files=False, silent=True,
                                  thread_count=self.n_jobs)
        family = method.split('_')[0]
        if family == 'Leshy':
            return arfsgroot.Leshy(model, n_estimators=ARFS_TREES, verbose=0, max_iter=20, random_state=42,
                                   importance=method.split('_')[1])
        if family == 'BoostAGroota':
            model.set_params(iterations=ARFS_TREES)  # the CatBoost default of 1000 trees made every round ten times slower
            return arfsgroot.BoostAGroota(estimator=model, cutoff=1, iters=BOOSTAGROOTA_ITERS, max_rounds=10,
                                          delta=0.1, importance=method.split('_')[1])
        if method == 'lasso':
            if alpha is None:
                if multi_output:
                    return MultiTaskLassoCV(max_iter=150000, n_jobs=self.n_jobs)
                return LassoCV(max_iter=150000, positive=True, n_jobs=self.n_jobs, precompute=False)
            if multi_output:
                return MultiTaskLasso(max_iter=150000, alpha=alpha)
            return Lasso(max_iter=150000, positive=True, alpha=alpha, precompute=False)
        if method == 'FeatureImportance':
            return RandomForestRegressor(n_estimators=int(m_estimator * 200), max_samples=size, n_jobs=self.n_jobs)
        if method == 'boruta':
            subsample = 1 / m_estimator if m_estimator > 1 else 1.
            n_estimators = int(m_estimator * 200) if m_estimator > 1 else 500
            if multi_output:
                return {f'boruta_{col}': BorutaPy(lgb.LGBMRegressor(n_estimators=n_estimators, n_jobs=self.n_jobs,
                                                                     subsample=subsample, verbose=-1),
                                                  n_estimators=200, verbose=0)
                        for col in self.static_data['target_variable']['lags']}
            return BorutaPy(lgb.LGBMRegressor(n_estimators=n_estimators, n_jobs=self.n_jobs, subsample=subsample,
                                              verbose=-1), n_estimators=n_estimators, max_iter=32, verbose=0)
        if method == 'ShapValues':
            return CatBoostRegressor(iterations=200, loss_function=loss_function, allow_writing_files=False,
                                     thread_count=self.n_jobs)
        raise ValueError(f'Unknown feature selection method {method}')

    @staticmethod
    def importance_vector(method, estimator, train_pool=None):
        if method == 'lasso':
            return np.abs(estimator.coef_)
        if method == 'FeatureImportance':
            return estimator.feature_importances_
        if method == 'boruta':
            if isinstance(estimator, dict):
                return sum(est.importance_history_.sum(axis=0) for est in estimator.values())
            return estimator.importance_history_.sum(axis=0)
        if method == 'ShapValues':
            fi = estimator.get_feature_importance(data=train_pool, type=method, prettified=True)
            if isinstance(fi, np.ndarray):
                return np.mean(np.mean(np.abs(fi), axis=1), axis=0)[:-1]
            return fi.abs().mean(axis=0).values[:-1]
        raise ValueError(f'Unknown feature selection method {method}')

    def fit_catboost(self, selector, x_train, y_train, cols, cat_feats):
        try:
            selector.fit(x_train, y_train[cols], cat_features=cat_feats, verbose=False)
        except Exception as error:
            if self.static_data['horizon_type'] == 'short_term':
                log.warning('CatBoost fit failed (%r); refitting with a jittered target', error)
                selector = self.estimator('ShapValues', multi_output=True)
                jitter = pd.DataFrame(np.random.uniform(0, 0.0001, list(y_train[cols].shape)),
                                      index=y_train.index, columns=cols)
                selector.fit(x_train, y_train[cols] + jitter, cat_features=cat_feats, verbose=False)
            else:
                log.warning('CatBoost fit failed (%r); all features kept', error)
                selector = None
        return selector

    def fit_boruta(self, selector, x_train, y_train):
        if isinstance(selector, dict):
            for col in self.static_data['target_variable']['lags']:
                selector[f'boruta_{col}'].fit(x_train.values, y_train.iloc[:, col].values)
        else:
            selector.fit(x_train.values, y_train.values.ravel())
        return selector

    # ------------------------------------------------------------------ candidate sets and scoring

    @staticmethod
    def candidate_sets(importance, fractions=CANDIDATE_FRACTIONS, min_features=MIN_FEATURES):
        """Distinct top-k feature index sets (fewest features first) from the importance ranking."""
        n = importance.shape[0]
        order = np.argsort(-importance, kind='stable')
        sets, seen = [], set()
        for fraction in sorted(fractions):
            k = min(n, max(min_features, int(np.ceil(fraction * n))))
            idx = tuple(sorted(order[:k].tolist()))
            if idx not in seen:
                seen.add(idx)
                sets.append(np.array(idx))
        return sets

    def scored_columns(self, y_train):
        cols = list(y_train.columns)
        if len(cols) <= SCORED_OUTPUTS:
            return cols
        picks = np.linspace(0, len(cols) - 1, SCORED_OUTPUTS).round().astype(int)
        return [cols[i] for i in picks]

    def score_features(self, x_train, y_train, x_test, y_test):
        """Mean test MAE of LightGBM models trained on ``x_train`` (one per scored target column)."""
        x_fit, x_val, y_fit, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=0)
        errors = []
        for col in self.scored_columns(y_train):
            model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, subsample=0.8,
                                      subsample_freq=1, colsample_bytree=0.9, n_jobs=self.n_jobs, verbose=-1,
                                      random_state=0)
            model.fit(x_fit, y_fit[col].values, eval_set=[(x_val, y_val[col].values)],
                      eval_metric='l1', callbacks=[lgb.early_stopping(20, verbose=False)])
            pred = model.predict(x_test)
            truth = y_test[col].values
            err = np.abs(pred - truth) if self.rated is not None else np.abs(pred - truth) / np.where(truth == 0, np.nan, truth)
            errors.append(float(np.nanmean(err)))
        return float(np.mean(errors))

    # ------------------------------------------------------------------ one selector

    def fit_method(self, method, x_train, y_train, x_test, y_test, cat_feats=None, label=''):
        multi_output = y_train.shape[1] > 1
        cols = y_train.columns if multi_output else y_train.columns[0]
        n_samples, n_features = x_train.shape
        big = n_samples >= 100000
        selector = self.estimator(method, multi_output=multi_output, size=n_samples - 1 if not big else 100000,
                                  m_estimator=1 if not big else n_samples / 100000)
        if cat_feats is not None:
            cat_feats = list(set(v for v in x_train.columns for c in CategoricalFeatures if c in v))
            x_train[cat_feats] = x_train[cat_feats].astype('int')
        if method == 'ShapValues':
            selector = self.fit_catboost(selector, x_train, y_train, cols, cat_feats)
        elif method == 'boruta':
            selector = self.fit_boruta(selector, x_train, y_train[cols])
        else:
            selector.fit(x_train, y_train[cols])
        if selector is None:
            indices = np.arange(n_features)
            return {'indices': indices, 'names': x_train.columns[indices]}
        importance = self.importance_vector(method, selector, Pool(x_train, y_train[cols], cat_features=cat_feats,
                                                                  thread_count=self.n_jobs))
        importance = np.asarray(importance, dtype=float)
        if importance.ndim > 1:
            importance = importance.sum(axis=0)
        spread = importance.max() - importance.min()
        importance = (importance - importance.min()) / spread if spread > 0 else np.zeros_like(importance)
        candidates = self.candidate_sets(importance)
        if len(candidates) == 1:
            chosen, scores = candidates[0], [None]
        else:
            scores = []
            for idx in candidates:
                # Restore BLAS/OpenMP defaults after the ranking estimator,
                # before another estimator or CV call uses the same budget.
                with limits(self.n_jobs):
                    scores.append(self.score_features(x_train.iloc[:, idx], y_train, x_test.iloc[:, idx], y_test))
            chosen = candidates[int(np.argmin(scores))]
        log.info('%s: %s ranked %d features; candidates %s -> kept %d',
                 label or method, method, n_features,
                 ', '.join(f'{len(idx)}:{score:.4f}' if score is not None else f'{len(idx)}:-'
                           for idx, score in zip(candidates, scores)), len(chosen))
        return {'indices': chosen, 'names': x_train.columns[chosen]}

    def fit_method_new(self, method, x_train, y_train, x_test, y_test, cat_feats=None, label=''):
        """arfs all-relevant methods: they decide the support themselves.

        Relevance does not need every row: above ``ARFS_MAX_SAMPLES`` a fixed random subset is used.
        """
        multi_output = y_train.shape[1] > 1
        n_rows = x_train.shape[0]
        if n_rows > ARFS_MAX_SAMPLES:
            rows = np.sort(np.random.RandomState(42).choice(n_rows, ARFS_MAX_SAMPLES, replace=False))
            x_fit, y_fit = x_train.iloc[rows], y_train.iloc[rows]
        else:
            x_fit, y_fit = x_train, y_train
        selector = self.estimator(method, multi_output=multi_output)
        selector.fit(x_fit, y_fit)
        indices = np.arange(x_train.shape[1])[selector.support_]
        if len(indices) < 2:
            indices = np.arange(x_train.shape[1])
        log.info('%s: %s kept %d of %d features (fitted on %d of %d rows)', label or method, method, len(indices),
                 x_train.shape[1], x_fit.shape[0], n_rows)
        return {'indices': indices, 'names': x_train.columns[indices]}

    def fit_frame(self, method, x_train, y_train, x_test, y_test, cat_feats=None, label=''):
        with limits(self.n_jobs):
            if method.split('_')[0] in ARFS_METHODS:
                return self.fit_method_new(method, x_train, y_train, x_test, y_test, label=label)
            return self.fit_method(method, x_train, y_train, x_test, y_test, cat_feats=cat_feats, label=label)

    # ------------------------------------------------------------------ LSTM: lags then variables

    @staticmethod
    def compress_lstm(x, x_test, axis=2):
        """One linear component per slice along ``axis`` (first principal direction, uncentred),
        the deterministic equivalent of the former one-unit linear autoencoder."""
        parts, parts_test = [], []
        for i in range(x.shape[axis]):
            X = np.take(x, i, axis=axis).astype(float)
            X_test = np.take(x_test, i, axis=axis).astype(float)
            X1 = np.concatenate([X, X_test])
            _, vectors = np.linalg.eigh(X1.T @ X1)
            v = vectors[:, -1]
            v = v if v.sum() >= 0 else -v
            parts.append(X @ v)
            parts_test.append(X_test @ v)
        return np.stack(parts, axis=1), np.stack(parts_test, axis=1)

    def fit_method_on_compressed(self, method, x_train, y_train, x_test, y_test, columns=None, label=''):
        cols = columns if columns is not None else [f'col_{i}' for i in range(x_train.shape[1])]
        x_train = pd.DataFrame(x_train, columns=cols)
        x_test = pd.DataFrame(x_test, columns=cols)
        return self.fit_frame(method, x_train, y_train, x_test, y_test, label=label)

    def fit_method_lstm(self, method, x_train, y_train, x_test, y_test, metadata, label=''):
        x_train_, x_test_ = self.compress_lstm(x_train, x_test, axis=1)
        lags = self.fit_method_on_compressed(method, x_train_, y_train, x_test_, y_test, label=f'{label} lags')
        ind_lags = lags['indices']
        x_train_, x_test_ = self.compress_lstm(x_train[:, ind_lags, :], x_test[:, ind_lags, :], axis=2)
        variables = self.fit_method_on_compressed(method, x_train_, y_train, x_test_, y_test,
                                                  columns=metadata['variables'], label=f'{label} variables')
        return {'lags': ind_lags, 'variables': [metadata['variables'][i] for i in variables['indices']]}

    @staticmethod
    def concat_lstm(x, metadata, dates):
        dates = metadata['dates'].intersection(dates)
        x_new, var_names = [], []
        for var_name in sorted(x['past'].keys()):
            past = np.expand_dims(x['past'][var_name].loc[dates].values, axis=-1)
            if var_name in x['future'].keys():
                future = np.expand_dims(x['future'][var_name].loc[dates].values, axis=-1)
            else:
                future = np.zeros([past.shape[0], len(metadata['future_lags']), 1])
            x_new.append(np.concatenate([past, future], axis=1))
            var_names.append(var_name)
        return np.concatenate(x_new, axis=-1), var_names

    def selection_dates(self, cv_mask):
        """``(dates to fit on, dates to assess on)`` for a selector.

        Under ``split_policy='frozen'`` a selector is fitted on the training dates and assessed on
        the validation dates: feature selection is part of the model, so the internal test mask must
        not take part in it.  ``'legacy_resplit'`` keeps what models trained before 2026-10-13 used,
        train + validation to fit and the test mask to assess.
        """
        if self.static_data.get('split_policy', 'frozen') == 'frozen':
            return cv_mask[0], cv_mask[1]
        return cv_mask[0].union(cv_mask[1]), cv_mask[2]

    def _fit(self, x, y, cv_mask, fs_selector_name, method, metadata=None):
        start = time.time()
        if 'lstm' in fs_selector_name:
            with _shared:
                train_dates, test_dates = self.selection_dates(cv_mask)
                x_train, _ = self.concat_lstm(x, metadata, train_dates)
                y_train = sync_data_with_dates(y, train_dates)
                x_test, var_names = self.concat_lstm(x, metadata, test_dates)
                y_test = sync_data_with_dates(y, test_dates)
            metadata['variables'] = var_names
            result = self.fit_method_lstm(method, x_train, y_train, x_test, y_test, metadata, label=fs_selector_name)
            log.info('%s: kept %d of %d lags and %d of %d variables in %.0fs', fs_selector_name,
                     len(result['lags']), x_train.shape[1], len(result['variables']), len(var_names),
                     time.time() - start)
            return result
        if not isinstance(x, pd.DataFrame):
            raise ValueError(f'{fs_selector_name}: expected a DataFrame, got {type(x)}')
        with _shared:
            train_dates, test_dates = self.selection_dates(cv_mask)
            x_train, y_train = sync_data_with_dates(x, train_dates), sync_data_with_dates(y, train_dates)
            x_test, y_test = sync_data_with_dates(x, test_dates), sync_data_with_dates(y, test_dates)
        result = self.fit_frame(method, x_train, y_train, x_test, y_test, label=fs_selector_name)
        log.info('%s: kept %d of %d features in %.0fs (train %d, test %d samples)', fs_selector_name,
                 len(result['names']), x_train.shape[1], time.time() - start, len(x_train), len(x_test))
        return result

    # ------------------------------------------------------------------ orchestration

    def fit_each(self, function, tasks, what):
        """Yield ``(task, function(task))`` for every task, in their order.

        Side by side when the stage has more than one CPU and there is more than one selector;
        one after another, each estimator with every CPU, otherwise.
        """
        at_once, each = shares(len(tasks), self.n_jobs)
        if at_once == 1:
            for task in tasks:
                check_cancelled()
                yield task, function(task)
            return
        log.info('%s: %d selectors, %d at a time with %d thread(s) each', what, len(tasks), at_once, each)
        # closed here, not when the collector gets to it: its workers are stopped and waited for
        with closing(side_by_side(function, tasks, at_once, each)) as results:
            for task in tasks:
                yield task, next(results)

    def data_type_combos(self, data_tag):
        if 'all' in data_tag or 'nwp' in data_tag or 'stats' in data_tag:
            keys = ('scale_row_method', 'scale_nwp_method', 'scale_img_method', 'merge', 'compress', 'image_data_type')
            options = (self.static_data['scale_row_method'], self.static_data['scale_nwp_method'],
                       self.static_data['scale_img_method'], self.static_data['nwp_data_merge'],
                       self.static_data['compress_data'], self.static_data['img_data'])
            return [dict(zip(keys, values)) for values in itertools.product(*options)]
        return [{'scale_row_method': m} for m in self.static_data['scale_row_method']]

    def cv_mask(self, cluster_name, cluster_path):
        if cluster_name == 'global':
            mask = self.files_manager.check_if_exists_cv_data()
        else:
            mask = joblib.load(os.path.join(cluster_path, 'cv_mask.pickle'))
        if mask is None:
            raise ImportError(f'Cannot find the cv masks of {cluster_name}: run the clustering stage first')
        return mask

    def fit(self, recreate_lstm=False):
        if not self.feature_selection_methods:
            log.info('feature selection: no method configured (feature_selection_methods), nothing to fit')
            return
        selectors = dict()
        from eforecast.feature_selection.partition import check_selector_partition, selector_partition_identity
        global_mask = self.files_manager.check_if_exists_cv_data()
        if not self.recreate:
            from eforecast.datasets.data_preprocessing.data_split import check_partition_fingerprint
            if global_mask is not None:
                # a selector was fitted from a cluster's masks, so it belongs to that partition
                check_partition_fingerprint(self.static_data, global_mask, 'the feature selectors')
        masks = {name: self.cv_mask(name, path) for name, path in self.clusters.items()}
        identities = {name: selector_partition_identity(self.static_data, global_mask, masks[name])
                      for name in self.clusters}
        provenance, dirty = {}, set()
        for cluster_name, cluster_path in self.clusters.items():
            filename = os.path.join(cluster_path, 'feature_selectors.pickle')
            stored = None
            if os.path.exists(filename) and not self.recreate:
                stored = check_selector_partition(cluster_path, identities[cluster_name])
                existing = joblib.load(filename)
                provenance[cluster_name] = stored.get('provenance', 'legacy_adopted') if stored else 'legacy_adopted'
            else:
                existing = dict()
                provenance[cluster_name] = 'fitted'
            if recreate_lstm:
                filtered = {k: v for k, v in existing.items() if 'lstm' not in k}
                if len(filtered) != len(existing):
                    dirty.add(cluster_name)
                existing = filtered
            selectors[cluster_name] = existing
        fitted = 0

        def pending(name_of):
            return [(cluster, method) for cluster in self.clusters for method in self.feature_selection_methods
                    if name_of(cluster, method) not in selectors[cluster]]

        # LSTM datasets
        if 'lstm' in self.dataset_tags:
            for scale_row_method in self.static_data['scale_row_method']:
                data_type = {'scale_row_method': scale_row_method}
                todo = pending(lambda c, m: selector_name(c, m, 'lstm', data_type))
                if not todo:
                    continue
                x, y, metadata = load_data_shallow_models(self.data_feeder, {'lstm': data_type}, 'lstm', True,
                                                          get_lstm_vars=True)

                def sequences(task, x=x, y=y, metadata=metadata, data_type=data_type):
                    cluster, method = task
                    return self._fit(x['lstm'], y, masks[cluster], selector_name(cluster, method, 'lstm', data_type),
                                     method, metadata=dict(metadata['lstm']))

                with closing(self.fit_each(sequences, todo, f'lstm {scale_row_method}')) as results:
                    for (cluster, method), selector in results:
                        selectors[cluster][selector_name(cluster, method, 'lstm', data_type)] = selector
                        self.save(self.clusters[cluster], selectors[cluster], identity=identities[cluster],
                                  provenance=provenance[cluster])
                        dirty.discard(cluster)
                        fitted += 1

        # row datasets
        for data_tag in sorted(self.dataset_tags):
            if 'row' not in data_tag:
                continue
            for data_type in self.data_type_combos(data_tag):
                base = lambda c, m: selector_name(c, m, data_tag, data_type)
                # grouped datasets carry the group in the name: load when no selector of the base name exists
                todo = [(c, m) for c, m in pending(base)
                        if not any(k.startswith(base(c, m) + '_') for k in selectors[c])]
                if not todo:
                    continue
                x, y, metadata = load_data_shallow_models(self.data_feeder, {data_tag: data_type}, data_tag, True)
                data = x[data_tag]
                groups = list(data.keys()) if isinstance(data, dict) else [None]
                tasks = [(cluster, method, group, name) for cluster, method in todo for group in groups
                         for name in [selector_name(cluster, method, data_tag, data_type, group)]
                         if name not in selectors[cluster]]
                # a cluster and method are saved whole: when the last of their groups is in
                closes = {(cluster, method): name for cluster, method, _, name in tasks}

                def rows(task, data=data, y=y, metadata=metadata[data_tag]):
                    cluster, method, group, name = task
                    return self._fit(data[group] if group is not None else data, y, masks[cluster], name, method,
                                     metadata=metadata)

                described = ' '.join(str(value) for value in data_type.values() if value is not None)
                with closing(self.fit_each(rows, tasks, f'{data_tag} {described}')) as results:
                    for (cluster, method, _, name), selector in results:
                        selectors[cluster][name] = selector
                        fitted += 1
                        if closes[cluster, method] == name:
                            self.save(self.clusters[cluster], selectors[cluster], identity=identities[cluster],
                                      provenance=provenance[cluster])
                            dirty.discard(cluster)
        for cluster_name, cluster_path in self.clusters.items():
            # --recreate rewrites every cluster, so a cluster with nothing to fit cannot keep a payload
            # of the previous partition and send the next run into a refusal it has already answered
            if (self.recreate or cluster_name in dirty
                    or not os.path.exists(os.path.join(cluster_path, 'feature_selectors.pickle'))):
                self.save(cluster_path, selectors[cluster_name], identity=identities[cluster_name],
                          provenance=provenance[cluster_name])
        log.info('feature selection: %d selectors fitted, %s per cluster', fitted,
                 {name: len(sel) for name, sel in selectors.items()})

    # ------------------------------------------------------------------ files

    @staticmethod
    def save(cluster_path, feature_selectors, *, identity=None, provenance='fitted'):
        from eforecast.feature_selection.partition import save_selectors
        save_selectors(cluster_path, feature_selectors, identity=identity, provenance=provenance)

    def create_backup(self):
        feature_selectors = dict()
        for cluster_name, cluster_path in self.clusters.items():
            filename = os.path.join(cluster_path, 'feature_selectors.pickle')
            if os.path.exists(filename):
                feature_selectors[cluster_name] = joblib.load(filename)
        joblib.dump(feature_selectors, os.path.join(self.static_data['path_model'], 'feature_selectors.pickle'))

    def rename_features(self, selectors, old_name, new_name):
        for k, v in selectors.items():
            if 'variables' in v.keys():
                selectors[k]['variables'] = [n.replace(old_name, new_name) for n in v['variables'] if old_name in n]
            elif 'names' in v.keys():
                selectors[k]['names'] = np.array([n.replace(old_name, new_name) for n in list(v['names'])
                                                  if old_name in n])
            else:
                selectors[k] = self.rename_features(v, old_name, new_name)
        return selectors

    def transfer_learning(self, static_data_base):
        # A payload written here was fitted on another project, so it carries no identity of this one:
        # the record of whatever it replaces is dropped and the selectors are adopted on trust, which
        # is what a selector without a record means everywhere else.
        from eforecast.feature_selection.partition import forget_selector_partition
        file_backup = os.path.join(static_data_base['path_model'], 'feature_selectors.pickle')
        if os.path.exists(file_backup):
            feature_selectors = joblib.load(file_backup)
            feature_selectors = self.rename_features(feature_selectors, static_data_base['project_name'],
                                                     self.static_data['project_name'])
            joblib.dump(feature_selectors, os.path.join(self.static_data['path_model'], 'feature_selectors.pickle'))
            for cluster_name, cluster_path in self.clusters.items():
                joblib.dump(feature_selectors[cluster_name], os.path.join(cluster_path, 'feature_selectors.pickle'))
                forget_selector_partition(cluster_path)
        else:
            clusters = trial_layout.cluster_folders(static_data_base)
            clusters.update({'global': os.path.join(static_data_base['path_model'], 'global')})
            for cluster_name, cluster_path in clusters.items():
                if cluster_name not in self.clusters:
                    continue
                fs = joblib.load(os.path.join(cluster_path, 'feature_selectors.pickle'))
                fs = self.rename_features(fs, static_data_base['project_name'], self.static_data['project_name'])
                target = self.clusters[cluster_name]
                joblib.dump(fs, os.path.join(target, 'feature_selectors.pickle'))
                forget_selector_partition(target)
