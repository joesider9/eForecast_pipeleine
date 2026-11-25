# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

import copy
import gc
import os
import time
import traceback

import joblib
import numpy as np
import pandas as pd
import torch
from eforecast.execution_policy import torch_device

from tqdm import tqdm

from eforecast.common_utils.dataset_utils import get_slice_for_nets
from eforecast.common_utils.resolutions import pandas_freq
from eforecast.datasets.data_feeder import DataFeeder
from eforecast.datasets.files_manager import FilesManager
from eforecast.datasets.data_preprocessing.data_split import masks_for_training
from eforecast.datasets.data_preprocessing import partition_report

from eforecast.common_utils.dataset_utils import load_data_deep_models
from eforecast.common_utils.dataset_utils import create_farm_shards_centers
from eforecast.common_utils.train_utils import fix_convolutional_names
from eforecast.common_utils.train_utils import initialize_train_constants
from eforecast.common_utils.train_utils import initialize_fuzzy_train_constants
from eforecast.common_utils.train_utils import check_if_is_better
from eforecast.common_utils.train_utils import check_if_extend_training
from eforecast.common_utils.train_utils import store_results
from eforecast.common_utils.train_utils import check_early_stop
from eforecast.deep_models.pytorch_2x.builders import build_graph, adapt_legacy_heads, WHOLE_BRANCHES_JOIN
from eforecast.deep_models.pytorch_2x.layers import PAST_DATA_IS_PAST
from eforecast.deep_models.pytorch_2x.global_builders import get_rbf
from eforecast.deep_models.pytorch_2x.global_builders import check_fuzzy_performance
from eforecast.deep_models.pytorch_2x.trainer import compute_tensors, clip_config
from eforecast.deep_models.pytorch_2x.horizon import HorizonWeights
from eforecast.deep_models.pytorch_2x.optimizers import ReduceLROnPlateau
from eforecast.deep_models.pytorch_2x.optimizers import optimize
from eforecast.deep_models.pytorch_2x.trainer import gradient_scaler
from eforecast.deep_models.pytorch_2x.trainer import train_schedule_fuzzy
from eforecast.deep_models.pytorch_2x.trainer import train_step
from eforecast.deep_models.pytorch_2x.trainer import validation_step
from eforecast.deep_models.pytorch_2x.trainer import feed_data_eval
from eforecast.deep_models.pytorch_2x.trainer import feed_dataset
from eforecast.deep_models.pytorch_2x.trainer import feed_image_dataset
from eforecast.deep_models.pytorch_2x.trainer import feed_image_dataset_real_time
from eforecast.deep_models.pytorch_2x.trainer import feed_farm_shard_dataset
# ImageDataset / ImageDataloader are imported where image prediction uses them: they bring
# torchvision's optical-flow models, 1.4 s that every trial process paid at start-up.
from eforecast.common_utils.dataset_utils import get_data_dict_from_dates



pd.set_option('display.expand_frame_repr', False)


def saved_state(file):
    """What a network saved beside its weights; old ones need the tolerant reader."""
    try:
        return joblib.load(file)
    except AttributeError:
        # models saved by a trial runner before 2026-09-09 carry its monitor, pickled from __main__:
        # unknown classes become inert stubs
        return load_tolerant(file)


def reads_past_data(source=None):
    """Whether a network that is about to be trained may read its past data as the past (``layers.past_of``).

    It may (``layers.PAST_DATA_IS_PAST``), unless it starts from the weights of a network that
    did not: weights fit the network they were trained in, so the one that takes them over is
    built the same way.
    """
    if not source:
        return True
    file = os.path.join(source, 'net_weights.pickle')
    if not os.path.exists(file):
        return True
    params = saved_state(file).get('params')
    return bool(params.get(PAST_DATA_IS_PAST, False)) if isinstance(params, dict) else True


def joins_whole_branches(source=None):
    """Whether a network that is about to be trained joins its whole branches at the final merge.

    It does (``builders.WHOLE_BRANCHES_JOIN``), unless it starts from the weights of a network
    that did not -- ``source`` is the folder of that network: weights fit the network they were
    trained in, so the one that takes them over is built the same way.
    """
    if not source:
        return True
    file = os.path.join(source, 'net_weights.pickle')
    if not os.path.exists(file):
        return True
    params = saved_state(file).get('params')
    return bool(params.get(WHOLE_BRANCHES_JOIN, False)) if isinstance(params, dict) else True


def evaluation_batch(n_samples, n_cvs, training_batch):
    """How many samples a validation or a test pass of the training loop reads at once.

    A small set is read in two batches for each of its cross-validation parts and a large one
    in the batches the network trains with -- but never in batches larger than those. A small
    set used to be read ``n / (2 * n_cvs)`` at a time whatever the network: a TIMM network that
    trained with 64 samples on an 11 GB card ran out of memory in its first validation pass,
    several hundred at a time, in every cluster of a case while its global study -- a set large
    enough to be read in training batches -- fitted. Halving the training batch after the
    failure could not help a batch that did not depend on it; now it halves this one too.
    """
    if n_samples >= 5000:
        return int(training_batch)
    return max(1, min(int(training_batch), int(n_samples / (2 * n_cvs))))


def load_tolerant(file):
    """joblib.load that replaces classes it cannot import by inert stubs (old runtime objects in saved models)."""
    import inspect
    from joblib.numpy_pickle import NumpyUnpickler

    class _Tolerant(NumpyUnpickler):
        def find_class(self, module, name):
            try:
                return super().find_class(module, name)
            except (AttributeError, ImportError, ModuleNotFoundError):
                return type(name, (), {'__setstate__': lambda self, state: None})

    with open(file, 'rb') as handle:
        kwargs = {'ensure_native_byte_order': True} if 'ensure_native_byte_order' in inspect.signature(NumpyUnpickler.__init__).parameters else {}
        return _Tolerant(file, handle, **kwargs).load()


class DeepNetwork:
    def __init__(self, static_data, path_weights, params=None, is_global=False, is_fuzzy=False, is_for_cluster=False,
                 probabilistic=False, train=False, online=False, refit=False):
        self.use_data = None
        self.results = None
        self.best_sse_val = np.inf
        self.best_sse_test = np.inf
        self.best_min_act = np.inf
        self.best_max_act = np.inf
        self.best_mean_act = np.inf
        self.best_sum_act = np.inf
        self.best_mae_val = np.inf
        self.best_mae_test = np.inf
        self.best_weights = None
        self.n_batch = None
        self.n_out = None
        self.is_trained = False
        self.refit = refit
        self.probabilistic = probabilistic
        self.is_global = is_global
        self.is_fuzzy = is_fuzzy
        self.is_for_cluster = is_for_cluster
        self.static_data = static_data
        self.rated = static_data['rated']
        if params is not None:
            self.params = params
            self.params['rated'] = static_data['rated']
            if self.params.get('head') == 'quantile':
                self.probabilistic = True
                self.params.setdefault('quantiles', [0.1, 0.5, 0.9])
            self.method = self.params['method']
            self.name = self.params['name']
            self.model_layers = self.params['experiment']
            self.conv_dim = self.params.get('conv_dim', 2)
            self.cluster = self.params.get('cluster', None)
            self.data_types = self.params['data_types']
            self.epochs = self.params['max_iterations']
            self.learning_rate = self.params['learning_rate']
            self.batch_size = self.params['batch_size']
        self.path_weights = path_weights
        if not self.refit:
            try:
                self.load()
            except ImportError:
                pass                      # not trained yet: params must come from the caller
            except Exception as error:    # noqa: BLE001
                print(f'cannot load the model in {path_weights}: {error!r}')
        self.refit = refit
        self.is_online = online
        self.train = train
        self.data_feeder = DataFeeder(static_data, online=self.is_online, train=self.train)
        self.file_manager = FilesManager(static_data, is_online=self.is_online, train=self.train)
        if not hasattr(self, 'params'):
            raise RuntimeError('The network has no params. You should train the net providing params')
        if self.params.get('head') == 'quantile':          # a loaded model: the head decides
            self.probabilistic = True
            self.params.setdefault('quantiles', [0.1, 0.5, 0.9])
        self.quantile_predictions = None                   # {quantile: frame} after a predict() of a quantile head

    def _frame_predictions(self, y_pred, dates, cols):
        """Predictions as a frame; a quantile head gives the median and keeps every quantile in
        ``self.quantile_predictions`` ({quantile: frame}, same index and columns)."""
        y_pred = np.asarray(y_pred)
        if y_pred.ndim == 3:
            quantiles = sorted(float(q) for q in self.params.get('quantiles') or [0.1, 0.5, 0.9])
            median = int(np.argmin([abs(q - 0.5) for q in quantiles]))
            self.quantile_predictions = {q: pd.DataFrame(y_pred[:, i], index=dates, columns=cols).sort_index()
                                         for i, q in enumerate(quantiles)}
            y_pred = y_pred[:, median]
        return pd.DataFrame(y_pred, index=dates, columns=cols).sort_index()

    def store_results_or_exit(self, best_weights=None, results=None, best_iter=None, error=None, store=False,
                              fuzzy=False):
        if best_weights is None:
            best_weights = {}
        if best_iter is None and error is not None:
            tb = traceback.format_exception(error)
            print("".join(tb))
            with open(os.path.join(self.path_weights, 'error.txt'), mode='w') as fp:
                fp.write(" ".join(tb))
            self.best_mae_test, self.best_mae_val, self.best_sse_test, self.best_sse_val = np.inf, np.inf, np.inf, np.inf
            self.results, self.is_trained, self.best_weights = pd.DataFrame(), True, {}
            self.save()
        else:
            if len(best_weights) == 0:
                raise ValueError('Model weights cannot be empty')
            self.best_mae_test = results['mae_test_out'].iloc[best_iter]
            self.best_mae_val = results['mae_val_out'].iloc[best_iter]
            self.best_sse_test = results['sse_test_out'].iloc[best_iter]
            self.best_sse_val = results['sse_val_out'].iloc[best_iter]
            self.results, self.best_weights, self.is_trained = results, best_weights, True
            if store:
                self.results.to_csv(os.path.join(self.path_weights, 'results.csv'))
                print(f"Total accuracy of validation: {self.best_mae_val} and of testing {self.best_mae_test}")
            if fuzzy:
                self.best_sum_act = results['sum_activations'].iloc[best_iter]
                self.best_min_act = results['min_activations'].iloc[best_iter]
                self.best_max_act = results['max_activations'].iloc[best_iter]
                self.best_mean_act = results['mean_activations'].iloc[best_iter]
                print(f'SUM OF ACTIVATIONS IS {self.best_sum_act}')
                print(f'MIN OF ACTIVATIONS IS {self.best_min_act}')
                print(f'MAX OF ACTIVATIONS IS {self.best_max_act}')
                print(f'MEAN OF ACTIVATIONS IS {self.best_mean_act}')

            self.save()

    def fit(self, cv_masks, gpu_id=0):
        device = torch_device(gpu_id)
        self.params['device'] = device
        if self.is_trained and not self.refit and not bool(np.isinf(self.best_mae_test)):
            return self.best_mae_test
        # what is trained from here on joins a branch that came whole (the calendar) with the tags
        # that came in groups (the weather by variable); it is saved with the network
        self.params[WHOLE_BRANCHES_JOIN] = joins_whole_branches(self.params.get('tl_path_weights'))
        # and a recurrent layer of it takes the past data for its past when the case has no observations
        self.params[PAST_DATA_IS_PAST] = reads_past_data(self.params.get('tl_path_weights'))
        quantiles = self.params['quantiles'] if self.probabilistic else None

        self.params['experiment'] = fix_convolutional_names(self.static_data['horizon_type'], self.params['experiment'],
                                                            self.conv_dim)
        self.params['kernels'] = self.static_data['kernels']
        ## TODO Fix Image outputs in load_data_deep_models for Image2image case
        for tag in self.data_types.keys():
            if tag == 'images' or 'stats' in tag:
                self.data_types[tag].update({'image_data_type': self.params['image_data_type'],
                                             'scale_img_method': self.params['scale_img_method']})
        if not self.static_data['transfer_learning_merge_data']:
            # the effective masks are settled before the data is loaded, because the fuzzy rule
            # initialiser inside the load must see the same training dates the model will train on
            min_samples = self.static_data.get('clustering', {}).get('min_samples', 0)
            supplied_masks = cv_masks
            global_masks = (FilesManager(self.static_data, is_online=False, train=True).check_if_exists_cv_data()
                            if min_samples else partition_report.load_global_masks(self.static_data))
            if min_samples:
                cv_masks = masks_for_training(
                    cv_masks, min_samples, global_masks,
                    cluster_name=(self.cluster or {}).get('cluster_name'))
            partition_report.write_training_scope(
                self.static_data, self.path_weights, 'pytorch', (self.cluster or {}).get('cluster_name'),
                supplied_masks, cv_masks, global_masks,
                reason='deep min_samples fallback applied when usable global masks exist')
            X, y, metadata, model_layers, self.params = load_data_deep_models(self.data_feeder, self.data_types,
                                                                               self.model_layers, self.params, self.cluster,
                                                                               self.train, self.is_fuzzy, self.refit,
                                                                               train_dates=cv_masks[0])
            self.save()
            self.params['n_out'] = y.shape[1]
            X_train, y_train, mask = get_slice_for_nets(X, metadata, dates=cv_masks[0], y=y)
            n_cvs = self.static_data['CVs']
            self.params['n_cvs'] = n_cvs

            X_val, y_val, mask_val = get_slice_for_nets(X, metadata, dates=cv_masks[1], y=y)
            X_test, y_test, mask_test = get_slice_for_nets(X, metadata, dates=cv_masks[2], y=y)
            del X                      # the full datasets (the NWP tensor is 10+ GB): only the slices stay
            gc.collect()

            self.n_out = y_train.shape[1]
            N = cv_masks[0].intersection(mask).shape[0]
            N_val = mask_val.shape[0]
            N_test = mask_test.shape[0]

            self.n_batch = int(self.batch_size)
            n_batch_val = evaluation_batch(N_val, n_cvs, self.n_batch)
            n_batch_test = evaluation_batch(N_test, n_cvs, self.n_batch)
            if 'images' not in self.model_layers:
                train_dataset = feed_dataset(X_train, y_train, device, batch_size=self.n_batch, shuffle=True)
                val_dataset = feed_dataset(X_val, y_val, device, batch_size=n_batch_val, shuffle=False)
                test_dataset = feed_dataset(X_test, y_test, device, batch_size=n_batch_test, shuffle=False)
            else:
                train_dataset = feed_image_dataset_real_time(self.static_data, X_train, y_train, mask, self.params,
                                                             device,
                                                             batch_size=self.batch_size, shuffle=True,
                                                             train=True, use_target=False)
                val_dataset = feed_image_dataset_real_time(self.static_data, X_val, y_val, mask_val, self.params,
                                                           device,
                                                           batch_size=self.batch_size, shuffle=False,
                                                           train=True, use_target=False)
                test_dataset = feed_image_dataset_real_time(self.static_data, X_test, y_test, mask_test, self.params,
                                                            device,
                                                            batch_size=self.batch_size, shuffle=False,
                                                            train=True, use_target=False)
        else:
            partition_report.write_training_scope(
                self.static_data, self.path_weights, 'pytorch', (self.cluster or {}).get('cluster_name'),
                cv_masks, None, reason='merged farm shards do not expose timestamp masks here')
            if self.is_fuzzy:
                path_batches = os.path.join(self.static_data['path_data'], 'farm_batches')
                batch_size = 20000
            elif 'path_farm_batches' in self.static_data:
                rule_suffix = '_'.join(self.cluster['cluster_name'].split('_')[1:])
                path_batches = os.path.join(self.static_data['path_farm_batches'], rule_suffix)
                batch_size = self.batch_size
            else:
                path_batches = os.path.join(self.static_data['path_data'], 'farm_batches', '_'.join(self.cluster['cluster_name'].split('_')[1:]))
                batch_size = self.batch_size
            train_dataset = feed_farm_shard_dataset(path_batches, 'train', device, batch_size=batch_size)
            val_dataset = feed_farm_shard_dataset(path_batches, 'val', device, batch_size=batch_size)
            test_dataset = feed_farm_shard_dataset(path_batches, 'test', device, batch_size=batch_size)
            if self.is_fuzzy:
                temp_dataset = feed_farm_shard_dataset(path_batches, 'train', device, batch_size=50000)
            else:
                temp_dataset = None
            n_cvs = self.static_data['CVs']
            self.params['n_cvs'] = n_cvs

        print('Create graph....')
        r = int(self.params.get('seed', self.static_data.get('split_seed', 1)))
        print(f'Training seed {r}')
        torch.manual_seed(r)
        torch.cuda.manual_seed_all(r)
        np.random.seed(r)
        if 'n_farms' in self.static_data:
            self.params['n_farms'] = self.static_data['n_farms']
        if 'meta_dim' in self.static_data:
            self.params['meta_dim'] = self.static_data['meta_dim']
        # try:
        from eforecast.common_utils.devops_utils import get_cuda_available_memory

        x_sample, y_sample = next(iter(train_dataset))
        if self.static_data['transfer_learning_merge_data']:
            # farm shards: the fuzzy checks below work on one shard batch per set
            X_train, y_train = x_sample, y_sample
            X_val, y_val = next(iter(val_dataset))
            X_test, y_test = next(iter(test_dataset))
        # otherwise X_train / X_val / X_test are the complete slices from get_slice_for_nets,
        # so the fuzzy checks (activation counts, linear proxy) see the whole sets
        N = y_train.shape[0]
        self.params['n_out'] = y_train.shape[1]
        if (self.static_data['transfer_learning_merge_data'] and self.is_fuzzy and
                not ('centroids' in self.params.keys() or 'var_init' in self.params.keys())):
            x_sample, y_sample = next(iter(temp_dataset))

            self.params, model_layers = create_farm_shards_centers(x_sample, y_sample, self.params,
                                                                             self.model_layers, self.refit)
            self.save()
        else:
            if self.is_fuzzy:
                new_model_layers = dict()
                for data_tag in x_sample.keys():
                    if data_tag not in {'clustering', 'farm_id'} and not data_tag.startswith('_'):
                        new_model_layers[data_tag] = self.model_layers[data_tag.split('/')[0]]

                for layer in self.model_layers.keys():
                    if 'hidden' in layer or 'output' in layer or layer == 'images':
                        new_model_layers[layer] = self.model_layers[layer]
                self.model_layers = copy.deepcopy(new_model_layers)
                model_layers = self.model_layers
            elif self.static_data['transfer_learning_merge_data']:
                model_layers = self.model_layers
            # Otherwise the layers load_data_deep_models returned stand. They are keyed by the
            # inputs it produced -- one branch per variable for a by_variable merge -- which is
            # what predict builds from too. Falling back to the stored layers here stopped every
            # global network with such a merge on KeyError 'nwp/<variable>'.
        net_model = self._build_graph(x_sample, model_layers, self.params, is_fuzzy=self.is_fuzzy,
                                probabilistic=self.probabilistic, quantiles=quantiles, device=device)
        if 'tl_path_weights' in self.params:
            if self.params['tl_path_weights'] is not None:
                self.best_weights = torch.load(os.path.join(self.params['tl_path_weights'], 'net_weights.pt'),
                                               weights_only=False, map_location=device)
                net_model.load_state_dict(adapt_legacy_heads(self.best_weights, net_model), strict=False)
                self.transfer_learning_trained = True
            else:
                self.transfer_learning_trained = False
        else:
            self.transfer_learning_trained = False
        net_model.to(device)
        # except Exception as e:
        #     self.store_results_or_exit(error=e)
        #     raise RuntimeError(e)
        optimizers, schedulers, loss, Accuracy, Sse = optimize(net_model, device,
                                                               optimizer=self.params.get('optimizer', 'adam'),
                                                               scheduler=self.params.get('scheduler', 'CosineAnnealing'),
                                                               rated=self.rated,
                                                               learning_rate=self.learning_rate,
                                                               is_fuzzy=self.is_fuzzy,
                                                               probabilistic=self.probabilistic,
                                                               quantiles=quantiles)





        (mae_old, sse_old, mae_max, sse_max, mae_min, sse_min, results_columns, results, best_mae_val, best_mae_test,
         train_flag, best_weights, warm, wait, best_iteration, best_tot_iteration, loops,
         n_iter, patience, exam_period) = initialize_train_constants(self.params, self.epochs, len_performers=2 * n_cvs)
        epochs = self.epochs
        if self.is_fuzzy:
            (mae_old_lin, sse_old_lin, mae_max_lin, sse_max_lin,
             mae_min_lin, sse_min_lin, results) = initialize_fuzzy_train_constants(results_columns,
                                                                                   epochs, len_performers=2)

        print(f"Start training of {self.name} using {self.method} with {self.n_batch} batches and {self.epochs} epochs")

        if self.is_global or self.is_fuzzy:
            init_clusters = get_rbf(net_model)
            best_clusters = get_rbf(net_model)
        clip_cfg = clip_config(self.params)   # None unless clip_loss_mode is set; the graph built its Clip layer
        # mixed_precision: the forward pass of training in half precision on the card (trainer.train_)
        amp = gradient_scaler(self.params, device, is_fuzzy=self.is_fuzzy)
        if amp is not None:
            print('Mixed precision: the forward pass of training runs in half precision on the card')
        if clip_cfg:
            print(f'CLIP term on the branches of the final merge: {clip_cfg}')
        # horizon-dependent loss weights (persistence once, or the model's own error per epoch)
        horizon = HorizonWeights(self.params.get('horizon_weights'), self.params['n_out'],
                                 floor=float(self.params.get('horizon_weight_floor', 0.05)))
        if horizon.active:
            # get_slice_for_nets returns the targets as an array and their dates separately
            targets = None
            if not self.static_data['transfer_learning_merge_data'] and isinstance(mask, pd.DatetimeIndex) \
                    and len(mask) == len(y_train):
                targets = pd.DataFrame(np.asarray(y_train), index=mask)
            weights = horizon.initial(targets)
            if weights is not None and hasattr(loss, 'set_weights'):
                loss.set_weights(weights)
            print(horizon.describe())
        patience = int(self.params.get('patience') or max(8, epochs // 5))
        checkpoints = []          # the last accepted best epochs (weights on the cpu) besides the final best
        best_weights = None
        for stale in os.listdir(self.path_weights):
            if stale.startswith('checkpoint'):
                os.remove(os.path.join(self.path_weights, stale))
        start_time = time.time()
        for epoch in tqdm(range(epochs)):
            print(f'Training model: epoch {epoch}')
            start = time.time()
            if epoch == 0 and (self.transfer_learning_trained or self.static_data['transfer_learning_merge_data']):
                # Skip the first epoch if it is transfer learning, pretrained weights have been applied
                pass
            else:
                try:
                    if not self.is_fuzzy:
                        step_rmse = train_step(net_model, loss, optimizers['bulk'], train_dataset, device, clip_cfg,
                                               amp)
                    else:
                        step_rmse = train_schedule_fuzzy(net_model, loss, optimizers, train_dataset, device, warm,
                                                         clip_cfg)
                except Exception as e:
                    self.store_results_or_exit(error=e)
                    raise RuntimeError(e)
                if horizon.mode == 'model' and hasattr(loss, 'set_weights'):
                    loss.set_weights(horizon.update(step_rmse))
                    if epoch % 10 == 0:
                        print(horizon.describe())
            end = time.time()
            sec_per_iter = (end - start)
            warm = 0
            if self.is_fuzzy:
                (net_model, best_clusters, sum_act,
                 min_act, max_act, mean_act,
                 warm, mae_old_lin, mae_max_lin,
                 mae_min_lin, sse_old_lin,
                 sse_max_lin, sse_min_lin,
                 mae_val_lin, mae_test_lin) = check_fuzzy_performance(net_model, N, X_train, y_train, X_val, y_val,
                                                                     X_test, y_test, self.params, init_clusters,
                                                                     best_clusters, device, mae_old_lin,
                                                                     mae_max_lin, mae_min_lin, sse_old_lin,
                                                                     sse_max_lin, sse_min_lin,
                                                                     self.static_data['clustering']['explode_clusters'])

            mae_val, sse_val = validation_step(net_model, Accuracy, Sse, val_dataset, device, n_cvs=n_cvs)
            mae_test, sse_test = validation_step(net_model, Accuracy, Sse, test_dataset, device, n_cvs=n_cvs)
            if (np.isnan(np.mean(mae_val)) or np.isnan(np.mean(mae_test))) and best_weights is not None:
                print('NaN in the validation metrics: back to the best weights')
                net_model.load_state_dict(adapt_legacy_heads(best_weights, net_model))
                mae_val, sse_val = validation_step(net_model, Accuracy, Sse, val_dataset, device, n_cvs=n_cvs)
                mae_test, sse_test = validation_step(net_model, Accuracy, Sse, test_dataset, device, n_cvs=n_cvs)
            if np.isnan(np.mean(mae_val)) or np.isnan(np.mean(mae_test)):
                raise ValueError('NaN in mae_val or mae_test')
            for name_scheduler, scheduler in schedulers.items():
                # plateau schedulers take the metric; the cosine schedule takes no argument (a metric passed
                # positionally was read as the epoch, so the schedule never advanced)
                if isinstance(scheduler, (ReduceLROnPlateau, torch.optim.lr_scheduler.ReduceLROnPlateau)):
                    scheduler.step(np.mean(mae_val) + np.mean(mae_test))
                else:
                    scheduler.step()
            mae_old, mae_max, mae_min, sse_old, sse_max, sse_min, flag_best = check_if_is_better(mae_old, mae_max,
                                                                                                 mae_min, sse_old,
                                                                                                 sse_max, sse_min,
                                                                                                 mae_val, mae_test,
                                                                                                 sse_val, sse_test)
            mae_val, mae_test, sse_val, sse_test = (np.mean(mae_val), np.mean(mae_test), np.mean(sse_val),
                                                    np.mean(sse_test))
            if flag_best:
                # a real snapshot: state_dict() returns the live tensors, which later epochs overwrite
                if best_weights is not None:
                    checkpoints.append({'epoch': best_iteration, 'mae_val': best_mae_val, 'mae_test': best_mae_test,
                                        'weights': best_weights})
                    checkpoints = checkpoints[-3:]
                from eforecast.common_utils.checkpoint_io import snapshot
                best_weights = snapshot(net_model)
                best_mae_val, best_mae_test = mae_val, mae_test      # the metrics of the saved epoch
                best_tot_iteration = n_iter
                best_iteration = epoch
                wait = 0
            else:
                wait += 1
            if not self.is_fuzzy:
                results, best_mae_val, best_mae_test = store_results(results, results_columns, best_tot_iteration,
                                                                     n_iter, best_mae_val, best_mae_test, mae_val,
                                                                     mae_test, sse_val, sse_test)
            else:
                results, best_mae_val, best_mae_test = store_results(results, results_columns, best_tot_iteration,
                                                                     n_iter, best_mae_val, best_mae_test, mae_val,
                                                                     mae_test, sse_val, sse_test, fuzzy=True,
                                                                     sum_act=sum_act, min_act=min_act, max_act=max_act,
                                                                     mean_act=mean_act, mae_val_lin=mae_val_lin,
                                                                     mae_test_lin=mae_test_lin)
            lr_now = optimizers['bulk'].param_groups[0]['lr'] if 'bulk' in optimizers else float('nan')
            print(f'epoch {epoch}: mae_val {mae_val:.4f} mae_test {mae_test:.4f} | best epoch {best_iteration} '
                  f'({best_mae_val:.4f}, {best_mae_test:.4f}) | wait {wait}/{patience} | lr {lr_now:.2e}')
            n_iter += 1
            monitor = getattr(self, 'monitor', None)  # set by trial_runner: progress file + prune flag of the study
            if monitor is not None and monitor.report(epoch, mae_val, mae_test, best_mae_val, best_mae_test):
                print(f'Trial pruned by the study at epoch {epoch}; keeping the best weights so far')
                self.pruned = True
                break
            if (best_mae_test > self.static_data['max_performance']) and epoch > 100:
                self.store_results_or_exit(best_iter=best_tot_iteration, results=results,
                                           best_weights=best_weights)
                return
            end_time = time.time()

            if wait > patience:
                print(f'early stop at epoch {epoch}: no accepted improvement for {patience} epochs '
                      f'(best epoch {best_iteration})')
                break

        # the previous accepted bests stay available for the evaluation on the unseen period
        for k, checkpoint in enumerate(checkpoints):
            from eforecast.common_utils.checkpoint_io import publish
            publish(self.path_weights,weights=checkpoint['weights'],
                    metadata={'mae_val':float(checkpoint['mae_val']),'mae_test':float(checkpoint['mae_test']),
                              'epoch':int(checkpoint['epoch'])},
                    weight_name=f'checkpoint_{k}.pt',metadata_name=f'checkpoint_mae_{k}.pt')
        self.horizon_weights = None if horizon.current is None else [float(w) for w in horizon.current]
        if horizon.active:
            print(horizon.describe())
        self.store_results_or_exit(best_iter=best_tot_iteration, results=results, best_weights=best_weights, store=True,
                                   fuzzy=self.is_fuzzy and not self.static_data['transfer_learning_merge_data'])
        gc.collect()

    @torch.no_grad()
    def predict_image(self, cluster_dates=None, cuda='auto'):
        device = torch_device(cuda)
        quantiles = self.params['quantiles'] if self.probabilistic else None

        if not hasattr(self, 'best_weights'):
            raise NotImplementedError(f'The {self.method} network is not train. '
                                      f'The location path is {self.path_weights}')

        cluster = self.params.get('cluster', None)

        X, metadata, model_layers, self.params = load_data_deep_models(self.data_feeder, self.data_types,
                                                                               self.model_layers, self.params, cluster,
                                                                               False, self.is_fuzzy, False)

        if self.train:
            cv_masks = joblib.load(os.path.join(self.cluster['cluster_path'], 'cv_mask.pickle'))
            X_val, _, mask_val = get_slice_for_nets(X, metadata, dates=cv_masks[1])
            X_test, _, mask_test = get_slice_for_nets(X, metadata, dates=cv_masks[2])
            val_dataset = feed_image_dataset_real_time(self.static_data, X_val, None, mask_val, self.params,
                                                         device,
                                                         batch_size=self.batch_size, shuffle=False,
                                                         train=True, use_target=False)
            test_dataset = feed_image_dataset_real_time(self.static_data, X_test, None, mask_test, self.params,
                                                         device,
                                                         batch_size=self.batch_size, shuffle=False,
                                                         train=True, use_target=False)

            cluster_dates = mask_val.union(mask_test)

            dataset = [val_dataset, test_dataset]
        elif not self.train and not self.is_online:
            X_eval, _, mask = get_slice_for_nets(X, metadata, dates=cluster_dates)
            cluster_dates = mask
            if self.params['create_image_dataset_real_time']:
                eval_dataset = feed_image_dataset_real_time(self.static_data, X_eval, None, mask,
                                                            self.params, device,
                                                        batch_size=self.batch_size, shuffle=False,
                                                        train=False, use_target=False)
            else:
                from eforecast.common_utils.dataset_utils import create_image_batches
                prepared = create_image_batches(self.static_data, self.params, self.batch_size, 0, False,
                                                tags=None, dates=mask)
                path_dataset = prepared['path_image_batches']
                eval_dataset = feed_image_dataset(X_eval, None, mask, 'image', path_dataset, device,
                                                  train=False)
            dataset = [eval_dataset]
        else:
            inp_x, _, cluster_dates = get_slice_for_nets(X, metadata, dates=cluster_dates)
            if self.params['create_image_dataset_real_time']:
                dataset = feed_image_dataset_real_time(self.static_data, inp_x, None, cluster_dates,
                                                            self.params, device,
                                                            batch_size=1, shuffle=False,
                                                            train=False, use_target=False)
            else:
                from eforecast.datasets.image_data.image_dataset import ImageDataloader, ImageDataset
                image_dataset = ImageDataset(self.static_data, cluster_dates, self.params, use_target=False)
                eval_image_dataset = ImageDataloader(image_dataset, int(self.batch_size), 1)
                x_batch = eval_image_dataset.get_batch()
                dates_sat = pd.DatetimeIndex([d[0] for d in x_batch['dates']])
                dates = dates_sat.intersection(cluster_dates)
                indices = dates_sat.get_indexer(dates)
                inp_x = get_data_dict_from_dates(inp_x, dates, dates_dict=cluster_dates)
                inp_x = feed_data_eval(inp_x)
                inp_x['images'] = x_batch['images'][indices].squeeze(0)
                dataset = inp_x

        cols = [f'{self.method}_{col}' for col in self.static_data['target_variable']['columns']]
        print('Create graph....')
        if np.isinf(self.best_mae_test):
            return pd.DataFrame(-999, index=cluster_dates, columns=cols)

        torch.cuda.empty_cache()
        gc.collect()
        with torch.inference_mode():
            y_pred = []
            dates_eval = pd.DatetimeIndex([])
            if not isinstance(dataset, list):
                flag = 0
                if self.params['create_image_dataset_real_time']:
                    inp_x, dates = next(iter(dataset))
                    dates = pd.DatetimeIndex([pd.to_datetime(timestamp_tensor.numpy(), unit='s').round('15min')
                                                  for timestamp_tensor in dates])
                else:
                    inp_x, dates = dataset, cluster_dates
                net_model = self._build_graph(inp_x, model_layers, self.params,
                                        probabilistic=self.probabilistic, quantiles=quantiles, device=device)
                net_model.load_state_dict(adapt_legacy_heads(self.best_weights, net_model))
                net_model.to(device)
                net_model.eval()
                y_temp = net_model(dataset)
                y_pred = np.clip(y_temp.cpu().detach().numpy(), 0, None)
                dates_eval = dates
                del y_temp
                # del net_model
                torch.cuda.empty_cache()
            else:
                flag = 0
                for dset in dataset:
                    for inp_x, dates in tqdm(dset):
                        if inp_x is None:
                            continue
                        if self.params['create_image_dataset_real_time']:
                            dates = pd.DatetimeIndex([pd.to_datetime(timestamp_tensor.numpy(), unit='s').
                                                     round(pandas_freq(self.static_data['ts_resolution']))
                                                      for timestamp_tensor in dates])
                        net_model = self._build_graph(inp_x, model_layers, self.params,
                                                probabilistic=self.probabilistic, quantiles=quantiles, device=device)
                        net_model.load_state_dict(adapt_legacy_heads(self.best_weights, net_model))
                        net_model.to(device)
                        net_model.eval()
                        y_temp = net_model(inp_x)
                        y_temp = np.clip(y_temp.cpu().detach().numpy(), 0, None)
                        y_pred.append(y_temp)
                        dates_eval = dates_eval.append(dates)
                        del y_temp
                        del net_model
                        torch.cuda.empty_cache()
                y_pred = np.concatenate(y_pred, axis=0)


        y_pred = self._frame_predictions(y_pred, dates_eval, cols)
        try:
            del net_model
        except:
            pass
        torch.cuda.empty_cache()
        gc.collect()
        return y_pred


    def _create_and_load_model(self, inp_x, model_layers, device, quantiles):
        """Helper method to create, load weights, and prepare model for inference."""
        net_model = self._build_graph(inp_x, model_layers, self.params,
                                probabilistic=self.probabilistic, quantiles=quantiles, device=device)
        net_model.load_state_dict(adapt_legacy_heads(self.best_weights, net_model))
        net_model.to(device)
        net_model.eval()
        return net_model

    @torch.no_grad()
    def predict_func(self, inp_x, cluster_dates, model_layers, device, quantiles):
        net_model = self._build_graph(inp_x, model_layers, self.params, is_fuzzy=self.is_fuzzy,
                                probabilistic=self.probabilistic, quantiles=quantiles, is_train=False, device=device)

        net_model.load_state_dict(adapt_legacy_heads(self.best_weights, net_model))
        net_model.to(device)
        net_model.eval()
        y_pred = []
        if len(cluster_dates) > self.batch_size:
            inds = np.arange(len(cluster_dates))
            splits = len(cluster_dates) // self.batch_size
            inds = np.array_split(inds, splits + 1)
            for ind in tqdm(inds):
                x = feed_data_eval(inp_x, indices=ind, device=device)
                y_p = net_model(x)
                y_p = y_p.cpu().detach().numpy()
                y_pred.append(y_p)
        else:
            x = feed_data_eval(inp_x, device=device)
            y_p = net_model(x)
            y_p = y_p.cpu().detach().numpy()
            y_pred.append(y_p)

        return np.concatenate(y_pred, axis=0), net_model

    @torch.no_grad()
    def predict(self, cluster_dates=None, with_activations=False, cuda=0):
        if 'images' in self.model_layers:
            return self.predict_image(cluster_dates, cuda=cuda)
        device = torch_device(cuda)
        quantiles = self.params['quantiles'] if self.probabilistic else None

        if not hasattr(self, 'best_weights'):
            raise NotImplementedError(f'The {self.method} network is not train. '
                                      f'The location path is {self.path_weights}')

        cluster = self.params.get('cluster', None)
        if not self.static_data['transfer_learning_merge_data']:
            whole = cluster_dates is None
            if whole and getattr(self, '_held_inputs', None) is not None:
                # the same inputs as the call before: only the weights differ (see hold_inputs)
                inp_x, cluster_dates, model_layers = self._held_inputs
            else:
                X, metadata, model_layers, self.params = load_data_deep_models(self.data_feeder, self.data_types,
                                                                                       self.model_layers, self.params, cluster,
                                                                                       False, self.is_fuzzy, False)
                inp_x, _, cluster_dates = get_slice_for_nets(X, metadata, dates=cluster_dates)
                del X
                if whole and getattr(self, 'hold_inputs', False):
                    self._held_inputs = (inp_x, cluster_dates, model_layers)
        else:
            if self.is_fuzzy and self.train:
                path_batches = os.path.join(self.static_data['path_data'], 'farm_batches')
                datasets = [feed_farm_shard_dataset(path_batches, 'train', device, batch_size=20000),
                                 feed_farm_shard_dataset(path_batches, 'val', device, batch_size=20000),
                                 feed_farm_shard_dataset(path_batches, 'test', device, batch_size=20000)]
                inp_x, _ = next(iter(datasets[0]))
                if not all([k in self.model_layers for k in list(inp_x.keys()) if k not in {'clustering'}]):
                    new_model_layers = dict()
                    for data_tag in inp_x.keys():
                        if data_tag not in {'clustering'}:
                            if data_tag in self.model_layers.keys():
                                new_model_layers[data_tag] = self.model_layers[data_tag.split('/')[0]]
                    if not all([k in new_model_layers for k in list(inp_x.keys()) if k not in {'clustering'}]):
                        new_model_layers = dict()
                        for data_tag, data in inp_x.items():
                            if data_tag not in {'clustering'}:
                                data_tag1 = data_tag.split('/')[0]
                                if data_tag1 not in self.model_layers.keys():
                                    raise ValueError(f'Cannot find {data_tag} tag in network architecture')
                                else:
                                    new_model_layers[data_tag] = self.model_layers[data_tag1]
                    new_model_layers['output'] = self.model_layers['output']
                    self.model_layers = new_model_layers
                net_model = self._build_graph(inp_x, self.model_layers, self.params, is_fuzzy=self.is_fuzzy,
                                        probabilistic=self.probabilistic, quantiles=quantiles, is_train=False,
                                        device=device)

                net_model.load_state_dict(adapt_legacy_heads(self.best_weights, net_model))
                net_model.to(device)
                net_model.eval()
                y_pred = []

                activations = []
                for dataset in datasets:
                    for inp_x, _ in tqdm(dataset):
                        y_p = net_model(inp_x)
                        y_p = y_p.cpu().detach().numpy()
                        y_pred.append(y_p)
                        if with_activations:
                            y_p = compute_tensors(net_model, inp_x)
                            y_p = y_p.cpu().detach().numpy()
                            activations.append(y_p)
                y_pred = np.concatenate(y_pred, axis=0)
                if with_activations:
                    activations = np.concatenate(activations, axis=0)
                del inp_x
                del net_model
                gc.collect()
                torch.cuda.empty_cache()
                cols = [f'{self.method}_{col}' for col in self.static_data['target_variable']['columns']]
                y_pred = pd.DataFrame(y_pred, index=cluster_dates, columns=cols)
                if with_activations:
                    activations = pd.DataFrame(activations, index=cluster_dates,
                                               columns=sorted(self.params['rules']))
                    return y_pred, activations
                else:
                    return y_pred
        cols = [f'{self.method}_{col}' for col in self.static_data['target_variable']['columns']]
        print('Create graph....')
        if np.isinf(self.best_mae_test):
            return pd.DataFrame(-999, index=cluster_dates, columns=cols)

        y_pred, net_model = self.predict_func(inp_x, cluster_dates, model_layers, device, quantiles)

        if with_activations:
            activations = []
            if len(cluster_dates) > 2000:
                inds = np.arange(len(cluster_dates))
                splits = len(cluster_dates) // 2000
                inds = np.array_split(inds, splits)
                for ind in tqdm(inds):
                    x = feed_data_eval(inp_x, ind, device=device)
                    y_p = compute_tensors(net_model, x)
                    y_p = y_p.cpu().detach().numpy()
                    activations.append(y_p)
            else:
                x = feed_data_eval(inp_x, device=device)
                y_p = compute_tensors(net_model, x)
                y_p = y_p.cpu().detach().numpy()
                activations.append(y_p)

            activations = np.concatenate(activations, axis=0)
        del inp_x
        del net_model
        gc.collect()
        torch.cuda.empty_cache()
        y_pred = self._frame_predictions(y_pred, cluster_dates, cols)
        if with_activations:
            activations = pd.DataFrame(activations, index=cluster_dates, columns=sorted(self.params['rules']))
            return y_pred, activations
        return y_pred

    def _build_graph(self, *args, **kwargs):
        from eforecast import model_library
        with model_library.use_inputs(getattr(self, 'model_library', ())) as used:
            graph = build_graph(*args, **kwargs)
        self.model_library = used
        return graph

    def load(self):
        file = os.path.join(self.path_weights, 'net_weights.pickle')
        if not os.path.exists(file):
            raise ImportError('Cannot load weights for deep model ' + self.path_weights)
        tmp_dict = saved_state(file)
        for key in self.UNSAVED:
            tmp_dict.pop(key, None)
        self.__dict__.update(tmp_dict)
        if isinstance(getattr(self, 'params', None), dict):
            # a network saved before whole branches were joined is built as it was trained
            self.params.setdefault(WHOLE_BRANCHES_JOIN, False)
            # and one saved before a recurrent layer could take the past data for its past, likewise
            self.params.setdefault(PAST_DATA_IS_PAST, False)
        self.best_weights = torch.load(os.path.join(self.path_weights, 'net_weights.pt'), weights_only=False,
                                       map_location=torch.device('cpu'))

    # runtime objects: rebuilt by __init__ or owned by the caller, never part of the saved model
    UNSAVED = {'static_data', 'path_weights', 'refit', 'best_weights', 'monitor', 'data_feeder', 'file_manager', 'quantile_predictions',
               'hold_inputs', '_held_inputs'}

    #: Set by a caller that predicts the same dates again with other weights -- a trial scoring
    #: its final weights and then each checkpoint. The scaled inputs of the first call are kept
    #: for the next ones, which otherwise each read, scale and merge every dataset again: five
    #: full loads of the weather for one trial. The caller clears it (``release_inputs``).
    hold_inputs = False

    def release_inputs(self):
        self.hold_inputs = False
        self._held_inputs = None

    def save(self):
        tmp_dict = {k: v for k, v in self.__dict__.items() if k not in self.UNSAVED}
        from eforecast.common_utils.checkpoint_io import publish
        publish(self.path_weights,weights=self.best_weights,metadata=tmp_dict)
