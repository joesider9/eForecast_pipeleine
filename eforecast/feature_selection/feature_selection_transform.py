# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Apply fitted feature selectors (inference side) and name them.

The names are shared with the fit side and with the data feeder, so a
selector is always looked up under the name it was saved with:

    feature_selector_<cluster>_<method>_<dataset name>[_<group>]

where the dataset name lists the non-null data options in the fixed order
``scale_row_method, scale_nwp_method, scale_img_method, merge, compress,
image_data_type`` followed by the dataset tag (``minmax_all_dense_row_all``,
``minmax_lstm``).
"""
import logging
import os

import joblib
import numpy as np

from eforecast.common_utils import trial_layout
from eforecast.datasets.files_manager import FilesManager

log = logging.getLogger(__name__)

DATA_TYPE_KEYS = ('scale_row_method', 'scale_nwp_method', 'scale_img_method', 'merge', 'compress', 'image_data_type')
CALENDAR_VARS = {'month', 'dayweek', 'hour', 'minute'}


def dataset_name(data_tag, data_type):
    values = [str(data_type.get(key)) for key in DATA_TYPE_KEYS if data_type.get(key) is not None]
    return '_'.join(values + [data_tag])


def selector_name(cluster_name, method, data_tag, data_type, group=None):
    name = f'feature_selector_{cluster_name}_{method}_{dataset_name(data_tag, data_type)}'
    return f'{name}_{group}' if group is not None else name


def calendar_names_to_sin_cos(names):
    """Selected calendar columns (``hour``, ``hour_lag_-1`` ...) become their ``_sin`` / ``_cos`` pair."""
    names = list(names)
    for var_name in CALENDAR_VARS:
        matching = [col for col in names if var_name == col or f'{var_name}_lag' in col]
        if matching:
            pairs = []
            for col in matching:
                head = col.split('_')[0]
                pairs += [col.replace(head, f'{head}_sin'), col.replace(head, f'{head}_cos')]
            names = [col for col in names if col not in matching] + pairs
    return np.array(names)


class FeatureSelector:
    def __init__(self, static_data, recreate=False, online=False, train=False):
        self.feature_selectors = dict()
        self.online = online
        self.train = train
        self.static_data = static_data
        self.is_Fuzzy = self.static_data['is_Fuzzy']
        self.clusters = dict()
        clusters_file = os.path.join(static_data['path_model'], 'clusters.pickle')
        if os.path.exists(clusters_file):
            try:
                self.clusters = trial_layout.cluster_folders(static_data)
            except trial_layout.ForeignTree:
                raise
            except Exception as error:
                log.warning('cannot read %s: %r', clusters_file, error)
        self.clusters.update({'global': os.path.join(static_data['path_model'], 'global')})
        self.calendar_variables = [var_data['name'] for var_data in self.static_data['variables']
                                   if var_data['type'] == 'calendar']
        self.files_manager = FilesManager(static_data, is_online=online)
        self._warned = set()

    def _load(self, cluster_path):
        filename = os.path.join(cluster_path, 'feature_selectors.pickle')
        if os.path.exists(filename):
            if self.train:
                from eforecast.feature_selection.partition import (check_selector_partition,
                                                                    selector_partition_identity)
                global_path = os.path.join(self.static_data['path_model'], 'global')
                global_masks = self.files_manager.check_if_exists_cv_data()
                if global_masks is None and os.path.exists(os.path.join(global_path, 'cv_mask.pickle')):
                    global_masks = joblib.load(os.path.join(global_path, 'cv_mask.pickle'))
                cluster_file = os.path.join(cluster_path, 'cv_mask.pickle')
                cluster_masks = global_masks if os.path.abspath(cluster_path) == os.path.abspath(global_path) \
                    else (joblib.load(cluster_file) if os.path.exists(cluster_file) else None)
                if cluster_masks is None:
                    raise ImportError(f'Cannot validate the selector partition of {cluster_path}: '
                                      'run the clustering stage first')
                identity = selector_partition_identity(self.static_data, global_masks, cluster_masks)
                check_selector_partition(cluster_path, identity)
            self.feature_selectors.update(joblib.load(filename))

    def _missing(self, selector_tag):
        if selector_tag not in self._warned:
            self._warned.add(selector_tag)
            log.warning('feature selector %s not found: data used unselected', selector_tag)

    def _select_frame(self, selector_tag, data, transform_calendar):
        if selector_tag not in self.feature_selectors:
            self._missing(selector_tag)
            return data
        names = self.feature_selectors[selector_tag]['names']
        if transform_calendar:
            names = calendar_names_to_sin_cos(names)
        return data[names] if len(names) > 0 else data

    def transform(self, selector_tag, cluster_path, data, transform_calendar=False):
        self._load(cluster_path)
        if 'lstm' in selector_tag:
            return self.transform_lstm(selector_tag, data)
        if not isinstance(data, dict):
            return self._select_frame(selector_tag, data, transform_calendar)
        return {group: self._select_frame(f'{selector_tag}_{group}', frame, transform_calendar)
                for group, frame in data.items()}

    def transform_lstm(self, selector_tag, data):
        """Keep the selected lags (positions in past + future) and the selected variables."""
        if selector_tag not in self.feature_selectors:
            self._missing(selector_tag)
            return data
        feature_selector = self.feature_selectors[selector_tag]
        lag_ids = set(int(lag) for lag in feature_selector['lags'])
        var_names = list(feature_selector['variables'])
        past_shape = next(iter(data['past'].values())).shape[1] if len(data['past']) > 0 else 0
        future_shape = next(iter(data['future'].values())).shape[1] if len(data['future']) > 0 else 0
        if len(set(data['future'].keys()).intersection(var_names)) == 0:
            var_names += list(data['future'].keys())  # no future variable selected: keep them all
        x_new = dict()
        for time_type, value_dicts in data.items():
            x_new[time_type] = dict()
            for var_name, value_df in value_dicts.items():
                if var_name not in var_names:
                    continue
                if time_type == 'past':
                    keep = [lag for lag in range(past_shape) if lag in lag_ids]
                else:
                    keep = [lag for lag in range(future_shape) if (lag + past_shape) in lag_ids]
                x_new[time_type][var_name] = value_df[value_df.columns[keep]]
        return x_new
