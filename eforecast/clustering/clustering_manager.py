# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Clustering stage: fuzzy RBF clusterers, cluster dates and per-cluster CV masks.

Products under ``path_model``:

* ``cluster_organizer/RBF/fuzzy_model.pickle`` and ``RBF_fuzzy_net_<i>/`` - the
  trained fuzzy networks (one per entry of ``rbf_var_imp``)
* ``DATA/cv_mask.pickle`` and ``global/cv_mask.pickle`` - the global split,
  adapted to the clusters
* ``cluster_organizer/RBF/rule_<i>/cv_mask.pickle`` - the split of each cluster
* ``clusters.pickle`` - ``{'<method>_<rule>': folder}``

The stage is idempotent: when the clusterers are trained and every mask exists
it only reports; ``refit`` retrains, recomputes the activations and rewrites
all masks together (the rule folders, and the models trained inside them, are
removed first).
"""
import logging
import os
import shutil

import joblib
import numpy as np
import pandas as pd

from eforecast.common_utils.clustering_membership import ensure_rule_coverage, rule_membership
from eforecast.clustering.tf_rbf_clusterer import TfRBFClusterer

from eforecast.datasets.data_preprocessing.data_split import (Splitter, check_partition_fingerprint,
                                                               save_partition_fingerprint)
from eforecast.datasets.data_preprocessing import partition_report
from eforecast.datasets.files_manager import FilesManager
from eforecast.feature_selection.partition import preserve_global_selector_partition

log = logging.getLogger(__name__)


class ClusterOrganizer:

    def __init__(self, static_data, is_online=False, train=False, refit=False):
        self.is_online = is_online
        self.train = train
        self.refit = refit
        self.sampled_data = None
        self.static_data = static_data
        self.path_model = self.static_data['path_model']
        self.is_Fuzzy = self.static_data['is_Fuzzy']
        clustering = self.static_data['clustering'] if self.is_Fuzzy else {
            'thres_act': None, 'data_type': {}, 'methods': [], 'clusters_for_method': None}
        self.thres_act = clustering['thres_act']
        self.scale_row_method = clustering['data_type'].get('scale_row_method')
        self.scale_nwp_method = clustering['data_type'].get('scale_nwp_method')
        self.merge = clustering['data_type'].get('merge')
        self.compress = clustering['data_type'].get('compress')
        self.methods = clustering['methods']
        self.make_clusters_for_method = clustering['clusters_for_method']
        self.file_manager = FilesManager(static_data, is_online=is_online, train=train)
        self.path_global = os.path.join(self.path_model, 'global')
        self.clusters_file = os.path.join(self.path_model, 'clusters.pickle')

    # ------------------------------------------------------------------ helpers

    def make_clusterer(self, method, refit=False):
        if method == 'RBF':
            return TfRBFClusterer(self.static_data, train=self.train, online=self.is_online, refit=refit)
        raise NotImplementedError(f'Clustering method {method} not implemented')

    def makes_clusters(self, method):
        return method == self.make_clusters_for_method or self.make_clusters_for_method == 'both'

    @staticmethod
    def create_cluster_folders(path):
        if os.path.exists(path):
            models = sorted(name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name)))
            log.warning('removing cluster folder %s%s', path,
                        f' and the trained models inside it: {models}' if models else '')
            shutil.rmtree(path)
        os.makedirs(path)

    def is_complete(self):
        """True when the clusterers are trained and every mask of every cluster exists."""
        if not self.is_Fuzzy:
            return os.path.exists(os.path.join(self.path_global, 'cv_mask.pickle'))
        if not os.path.exists(self.clusters_file):
            return False
        for method in self.methods:
            clusterer = self.make_clusterer(method)
            if not clusterer.is_trained:
                return False
            if self.makes_clusters(method):
                for cluster in clusterer.rule_names:
                    if not os.path.exists(os.path.join(clusterer.path_fuzzy, cluster, 'cv_mask.pickle')):
                        return False
        return True

    # ------------------------------------------------------------------ transfer learning

    def get_fuzzy_models(self):
        for method in self.methods:
            return self.make_clusterer(method).transfer_learning_source()

    def copy_fuzzy_models(self, paths, fuzzy_file):
        for method in self.methods:
            return self.make_clusterer(method).transfer_learning_target(paths, fuzzy_file)

    # ------------------------------------------------------------------ fit

    def fit(self):
        if not self.is_Fuzzy:
            return
        cv_mask = self.file_manager.check_if_exists_cv_data()
        if cv_mask is None:
            raise ImportError(f'Cannot find the global split {self.file_manager.file_cv_data()}: it is drawn by '
                              'the split stage, which runs before this one (--stage split clustering)')
        for method in self.methods:
            clusterer = self.make_clusterer(method, refit=self.refit)
            if clusterer.is_trained and not self.refit:
                log.info('clusterer %s is trained (%d rules), skipping', method, len(clusterer.rule_names))
                continue
            log.info('training clusterer %s', method)
            clusterer.fit(cv_mask)

    # ------------------------------------------------------------------ activations -> cluster dates

    def cluster_input_dates(self, clusterer):
        if self.thres_act is None:
            raise ValueError('rule membership requires a fuzzy clustering configuration')
        clustered_dates = dict()
        activations = clusterer.compute_activations()
        values, assigned = ensure_rule_coverage(activations.values, self.thres_act)
        activations = pd.DataFrame(values, index=activations.index, columns=activations.columns)
        active = rule_membership(values, self.thres_act)
        log.info('rule coverage: %d samples assigned to their strongest rule; %d remain uncovered',
                 assigned, int((~active.any(axis=1)).sum()))
        for cluster in clusterer.rule_names:
            indices = np.flatnonzero(active[:, activations.columns.get_loc(cluster)])
            if isinstance(activations.index, pd.MultiIndex):
                index = activations.index[indices]
                clustered_dates[cluster] = index.get_level_values(0)
            else:
                clustered_dates[cluster] = activations.index[indices]
        return clustered_dates, activations

    def log_activation_summary(self, activations, clustered_dates):
        """What the clustering looks like before the split: coverage, size and overlap of the rules."""
        if self.thres_act is None:
            raise ValueError('activation diagnostics require a fuzzy clustering configuration')
        values = activations.values.astype(float)
        active = rule_membership(values, self.thres_act)
        n = values.shape[0]
        log.info('clustering summary: %d samples, %d rules, threshold %g', n, values.shape[1], self.thres_act)
        for i, rule in enumerate(activations.columns):
            n_active = int(active[:, i].sum())
            mean_act = float(values[active[:, i], i].mean()) if n_active else float('nan')
            log.info('  %-10s %6d samples (%5.1f%%), mean activation %.3f', rule, n_active, 100 * n_active / n,
                     mean_act)
        per_sample = active.sum(axis=1)
        log.info('  samples with no active rule after assignment: %d; rules per sample: mean %.2f, '
                 'max %d', int((per_sample == 0).sum()), float(per_sample.mean()), int(per_sample.max()))
        if active.shape[1] > 1:
            inter = active.T.astype(int) @ active.astype(int)
            sizes = np.diag(inter)
            union = sizes[:, None] + sizes[None, :] - inter
            jaccard = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
            np.fill_diagonal(jaccard, 0)
            i, j = np.unravel_index(jaccard.argmax(), jaccard.shape)
            log.info('  largest overlap: %s and %s share %d samples (Jaccard %.2f)', activations.columns[i],
                     activations.columns[j], int(inter[i, j]), float(jaccard[i, j]))

    def predict(self, method):
        if method not in self.methods:
            raise ValueError(f'{method} is not in clustering methods')
        clusterer = self.make_clusterer(method)
        predictions, activations = clusterer.compute_activations(with_predictions=True)
        return predictions, activations

    def compute_activations(self, method):
        if method not in self.methods:
            raise ValueError(f'{method} is not in clustering methods')
        clusterer = self.make_clusterer(method)
        clustered_dates, activations = self.cluster_input_dates(clusterer)
        return activations, clustered_dates

    # ------------------------------------------------------------------ clusters and masks

    def log_existing_clusters(self):
        clusters = joblib.load(self.clusters_file) if os.path.exists(self.clusters_file) else {}
        for name, path in clusters.items():
            mask_file = os.path.join(path, 'cv_mask.pickle')
            if os.path.exists(mask_file):
                masks = joblib.load(mask_file)
                log.info('  %-16s train %6d, val %5d, test %5d  %s', name, *[m.shape[0] for m in masks], path)

    def create_clusters_and_cvs(self):
        cv_mask = self.file_manager.check_if_exists_cv_data()
        if not self.refit and self.is_complete():
            log.info('clusters and masks exist, skipping (use --recreate to redo the clustering): %s',
                     self.clusters_file)
            if cv_mask is not None:
                # the masks on disk say nothing about the partition or the policy they came from
                check_partition_fingerprint(self.static_data, cv_mask, 'the cluster masks')
            self.log_existing_clusters()
            if cv_mask is not None:
                saved_clusters = (joblib.load(self.clusters_file) if self.is_Fuzzy else {})
                cached = {'global': list(cv_mask)}
                cached.update({name: joblib.load(os.path.join(path, 'cv_mask.pickle'))
                               for name, path in saved_clusters.items()})
                partition_report.write_cluster_section(
                    self.static_data, cv_mask, cached,
                    min_samples=self.static_data.get('clustering', {}).get('min_samples', 0))
            return
        if cv_mask is None:
            raise ImportError(f'Cannot find the global split {self.file_manager.file_cv_data()}: it is drawn by '
                              'the split stage, which runs before this one (--stage split clustering)')
        preserve_global_selector_partition(self.static_data, self.path_global)
        os.makedirs(self.path_global, exist_ok=True)
        joblib.dump(cv_mask, os.path.join(self.path_global, 'cv_mask.pickle'))
        if not self.is_Fuzzy:
            fingerprint = save_partition_fingerprint(self.static_data, cv_mask)
            partition_report.write_cluster_section(self.static_data, cv_mask, {'global': list(cv_mask)})
            log.info('global masks written against partition %s (policy %s)',
                     fingerprint['masks_sha256'], fingerprint['split_policy'])
            return
        clusters = dict()
        report_cvs, report_memberships = {'global': list(cv_mask)}, {}
        for method in self.methods:
            clusterer = self.make_clusterer(method)
            if not self.makes_clusters(method):
                continue
            clustered_dates, activations = self.cluster_input_dates(clusterer)
            self.log_activation_summary(activations, clustered_dates)
            splitter = Splitter(self.static_data, is_online=self.is_online, train=self.train)
            cluster_cvs = splitter.split_cluster_data(clustered_dates, cv_mask=cv_mask)
            if splitter.replaces_global_split:
                # the legacy policy redraws the partition after the clusterer was fitted with the
                # first one; under 'frozen' there is nothing to rewrite, which is the whole point
                cv_mask = cluster_cvs['global']
                self.file_manager.remove_cv_data_files()
                self.file_manager.save_cv_data(cv_mask)
                joblib.dump(cv_mask, os.path.join(self.path_global, 'cv_mask.pickle'))
                log.warning('split_policy=legacy_resplit: the global split was redrawn after the clusterer was '
                            'fitted, so some of its training dates are now validation or test dates')
            report_cvs['global'] = list(cv_mask)
            report_cvs.update({f'{method}_{name}': masks for name, masks in cluster_cvs.items() if name != 'global'})
            report_memberships.update({f'{method}_{name}': dates for name, dates in clustered_dates.items()})
            for cluster in clusterer.rule_names:
                path = os.path.join(clusterer.path_fuzzy, cluster)
                self.create_cluster_folders(path)
                joblib.dump(list(cluster_cvs[cluster]), os.path.join(path, 'cv_mask.pickle'))
                clusters[f'{method}_{cluster}'] = path
        joblib.dump(clusters, self.clusters_file)
        partition_report.write_cluster_section(
            self.static_data, cv_mask, report_cvs, memberships=report_memberships,
            min_samples=self.static_data.get('clustering', {}).get('min_samples', 0))
        fingerprint = save_partition_fingerprint(self.static_data, self.file_manager.check_if_exists_cv_data())
        log.info('%d clusters written to %s, against partition %s (policy %s)', len(clusters), self.clusters_file,
                 fingerprint['masks_sha256'], fingerprint['split_policy'])

    def update_cluster_folders(self):
        if self.is_Fuzzy:
            clusters = dict()
            for method in self.methods:
                clusterer = self.make_clusterer(method)
                if self.makes_clusters(method):
                    for cluster in clusterer.rule_names:
                        path = os.path.join(clusterer.path_fuzzy, cluster)
                        if not os.path.exists(path):
                            raise ImportError(f'Cannot find {cluster} path: {path}')
                        clusters[f'{method}_{cluster}'] = path
            joblib.dump(clusters, self.clusters_file)
