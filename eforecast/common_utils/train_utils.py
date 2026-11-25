# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

import logging
import os
import sys
import copy
import psutil
import joblib
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans

try:
    import tensorflow as tf
except:
    import torch

import multiprocessing as mp

from time import sleep
from contextlib import contextmanager
from sklearn.linear_model import LinearRegression
from eforecast.common_utils.clustering_membership import (ensure_rule_coverage, rule_membership,
                                                        validate_rule_threshold)

log = logging.getLogger(__name__)


class GpuQueue:

    def __init__(self, N_GPUS, all_gpus=False):
        self.queue = mp.Manager().Queue()
        if all_gpus:
            all_idxs = list(range(N_GPUS))
        else:
            all_idxs = [N_GPUS]
        for idx in all_idxs:
            self.queue.put(idx)

    @contextmanager
    def one_gpu_per_process(self):
        current_idx = self.queue.get()
        yield current_idx
        self.queue.put(current_idx)


def find_min_max_var(inputs, n_rules, n_var, centroids, var, thres_act):
    s = np.shape(inputs)
    phi = []
    for n in range(n_rules):
        d1 = inputs.values - np.tile(centroids[n], [s[0], 1])
        d = np.sqrt(np.sum(np.power(d1 / np.tile(var.values[n], [s[0], 1]), 2), axis=1))
        phi.append(np.expand_dims(np.exp(-1 * np.square(d)), axis=1))
    act_all_eval = np.concatenate(phi, axis=1)
    act_all_eval[act_all_eval >= thres_act] = 1
    act_all_eval[act_all_eval < thres_act] = 0
    return act_all_eval.sum(axis=0)


def check_VAR_if_all_nans(inputs, n_rules, n_var, centroids, var, thres_act):
    s = np.shape(inputs)
    dist = []
    phi = []
    for n in range(n_rules):
        d1 = inputs.values - np.tile(centroids[n], [s[0], 1])
        dist.append(np.expand_dims(np.sqrt(np.sum(np.square(d1), axis=1)), axis=1))
        d = np.sqrt(np.sum(np.power(d1 / np.tile(var.values[n], [s[0], 1]), 2), axis=1))
        phi.append(np.expand_dims(np.exp(-1 * np.square(d)), axis=1))
    dist = np.concatenate(dist, axis=1)
    activations = np.concatenate(phi, axis=1)
    indices = np.flatnonzero(~rule_membership(activations, thres_act).any(axis=1))
    len_nan = 0
    while indices.shape[0] > 0:
        d = dist[indices[0]]
        clust = np.argmin(d)
        var.iloc[clust] += thres_act
        dist = []
        phi = []
        for n in range(n_rules):
            d1 = inputs.values - np.tile(centroids[n], [s[0], 1])
            dist.append(np.expand_dims(np.sqrt(np.sum(np.square(d1), axis=1)), axis=1))
            d = np.sqrt(np.sum(np.power(d1 / np.tile(var.values[n], [s[0], 1]), 2), axis=1))
            phi.append(np.expand_dims(np.exp(-1 * np.square(d)), axis=1))
        dist = np.concatenate(dist, axis=1)
        activations = np.concatenate(phi, axis=1)
        indices = np.flatnonzero(~rule_membership(activations, thres_act).any(axis=1))
    return var

def RBF_centroids(X_tr, X_val, y_tr, y_val, params, abstract=False):
    c_best = None
    inertia = np.inf
    from eforecast.clustering.rbf_ols_network import RBFols
    max_samples_rbf = 10000
    ind_x = np.random.choice(np.arange(X_tr.shape[0]), max_samples_rbf) if X_tr.shape[
                                                                               0] > max_samples_rbf else np.arange(
        X_tr.shape[0])
    ind_v = np.random.choice(np.arange(X_val.shape[0]), max_samples_rbf) if X_val.shape[
                                                                                0] > max_samples_rbf else np.arange(
        X_val.shape[0])
    widths = np.linspace(0.8, 1.4, 6)
    if abstract:
        c_best = RBFols(params['rated'], n_clusters=params['n_rules'], thres_act=params['thres_act'],
                        width=widths[-1] * np.abs(X_tr).mean())
        c_best.model['centroids'] = np.expand_dims(np.linspace(2.5 / params['n_rules'], 1 - (2.5 / params['n_rules']), params['n_rules']), axis=-1) * np.ones((params['n_rules'], X_tr.shape[1]))
        c_best.model['Radius'] =  (params['n_rules'] / 2.5) * np.ones((params['n_rules'], X_tr.shape[1]))
        return c_best
    x_fit, y_fit, x_check, y_check = X_tr[ind_x], y_tr[ind_x], X_val[ind_v], y_val[ind_v]
    scale = np.abs(X_tr).mean()

    def candidate(w):
        c = RBFols(params['rated'], n_clusters=params['n_rules'], thres_act=params['thres_act'], width=w * scale)
        c.fit(x_fit, y_fit, x_check, y_check)
        return c
    jobs = centroid_jobs(len(widths))
    if jobs > 1:
        # Six independent fits of one width each: the same numbers in any order, so they are
        # fitted side by side and compared in the order they would have run in.
        from concurrent.futures import ThreadPoolExecutor
        from threadpoolctl import threadpool_limits
        log.info('fuzzy rule initialiser: %d candidate widths, %d at a time', len(widths), jobs)
        with threadpool_limits(limits=1), ThreadPoolExecutor(jobs) as pool:
            candidates = list(pool.map(candidate, widths))
    else:
        candidates = map(candidate, widths)
    for c in candidates:
        if c.err < inertia:
            c_best = copy.deepcopy(c)
            inertia = c.err
    return c_best


def centroid_jobs(candidates):
    """How many candidate widths are fitted at once: as many as the clusterer has CPUs.

    A clusterer that was given one CPU (``clustering.n_jobs: 1``, the preset) fits them one after
    another, as before. Each fit works on rows x rows matrices of at most 10,000 rows -- about
    ``rows * rows * 8 * (columns + 4)`` bytes at its peak, a few GB -- and memory is not looked
    at: the operating system carries six of them or ends the stage.
    """
    from eforecast import execution_policy
    policy = execution_policy.current()
    granted = int(policy['effective']['threads']) if policy else 1
    return max(1, min(candidates, granted))


def adjust_rule_widths(X_train, n_rules, centroids, var_init, thres_act, min_samples, max_samples, rounds=5):
    """Widths so that every sample activates a rule and every rule has between ``min_samples``
    and ``max_samples`` active samples.  Coverage has priority: the last step always restores it,
    so a rule may end above ``max_samples`` (logged)."""
    if not 1 <= min_samples <= max_samples <= X_train.shape[0]:
        raise ValueError(f'cannot initialise rule widths: require 1 <= min_samples ({min_samples}) '
                         f'<= max_samples ({max_samples}) <= training rows ({X_train.shape[0]}); '
                         'reduce min_samples or increase max_samples_ratio')
    if n_rules < 1:
        raise ValueError('rule widths require at least one rule')
    validate_rule_threshold(thres_act)
    n_var = centroids.shape[1]

    def counts(var):
        return find_min_max_var(X_train, n_rules, n_var, centroids, var, thres_act)

    def uncovered(var):
        s = np.shape(X_train)
        phi = np.stack([np.exp(-np.square(np.sqrt(np.sum(np.power(
            (X_train.values - np.tile(centroids[n], [s[0], 1])) / np.tile(var.values[n], [s[0], 1]), 2), axis=1))))
            for n in range(n_rules)], axis=1)
        return int((~rule_membership(phi, thres_act).any(axis=1)).sum())

    for round_ in range(rounds):
        var_init = check_VAR_if_all_nans(X_train, n_rules, n_var, centroids, var_init, thres_act)
        n_samples = counts(var_init)
        while (n_samples < min_samples).any():
            var_init.iloc[np.where(n_samples < min_samples)[0]] += 0.001
            n_samples = counts(var_init)
        while (n_samples > max_samples).any():
            var_init.iloc[np.where(n_samples > max_samples)[0]] -= 0.001
            n_samples = counts(var_init)
        if uncovered(var_init) == 0:
            break
    var_init = check_VAR_if_all_nans(X_train, n_rules, n_var, centroids, var_init, thres_act)
    n_samples = counts(var_init)
    log.info('rule widths initialised: active samples per rule %s (min %d, max %d of %d), uncovered %d',
             n_samples.astype(int).tolist(), min_samples, max_samples, X_train.shape[0], uncovered(var_init))
    if (n_samples > max_samples).any():
        log.warning('rules %s exceed max_samples %d after restoring coverage',
                    np.where(n_samples > max_samples)[0].tolist(), max_samples)
    return var_init


def _pool_by_predicted_hour(X_train, y):
    """The 75/25 pool of a ``long_term`` case, whose rows are (prediction time, issue time).

    Every row takes the target of its predicted hour: that is how its training samples are made.
    Looking the rows up in the target by label, as a single-clock case does, finds none of them,
    and the initialiser then failed on an empty pool ("attempt to get argmax of an empty
    sequence"). The cut falls between hours, so the issue days of one hour stay on one side.
    """
    times = pd.DatetimeIndex(X_train.index.get_level_values(0))
    known = times.isin(y.index)
    X_train, times = X_train[known], times[known]
    hours = times.unique().sort_values()
    first = times.isin(hours[:int(0.75 * len(hours))])
    return (X_train[first].values, y.loc[times[first]].values,
            X_train[~first].values, y.loc[times[~first]].values, X_train)


def create_centroids(X_train, y, params, abstract=False):
    if X_train is None:
        raise ValueError('X_train is not provided')
    if isinstance(X_train, pd.DataFrame) and isinstance(X_train.index, pd.MultiIndex):
        X_tr, y_tr, X_val, y_val, X_train = _pool_by_predicted_hour(X_train, y)
    elif isinstance(X_train, pd.DataFrame):
        if X_train.shape[0] > 100000:
            X_train = X_train.resample('15min').mean().dropna(axis='index', how='any')
            y = y.resample('15min').mean().dropna(axis='index', how='any')
        dates = y.index.intersection(X_train.index)
        split = int(0.75 * len(dates))
        X_tr = X_train.loc[dates[:split]].values
        y_tr = y.loc[dates[:split]].values
        X_val = X_train.loc[dates[split:]].values
        y_val = y.loc[dates[split:]].values
    else:
        split = int(0.75 * len(X_train))
        X_tr = X_train[:split]
        y_tr = y[:split]
        X_val = X_train[split:]
        y_val = y[split:]
        X_train = pd.DataFrame(X_train)

    if not abstract and (X_tr.shape[0] == 0 or X_val.shape[0] == 0):
        raise ValueError(f'the fuzzy rule initialiser has no rows to fit on: {X_tr.shape[0]} for fitting and '
                         f'{X_val.shape[0]} for checking, of {X_train.shape[0]} clustering rows with a target. '
                         'The clustering inputs and the target share no dates on the training part of the split')
    min_samples = params['min_samples']
    max_samples = int(params['max_samples_ratio'] * X_train.shape[0])
    if not 1 <= min_samples <= max_samples <= X_train.shape[0]:
        raise ValueError(f'clustering sample limits are infeasible for {X_train.shape[0]} training rows: '
                         f'min_samples={min_samples}, max_samples={max_samples}; '
                         'reduce min_samples or increase max_samples_ratio')
    c_best = RBF_centroids(X_tr, X_val, y_tr, y_val, params, abstract=abstract)

    if c_best is not None:
        centroids = c_best.model['centroids'].astype(np.float32)
        widths = 1 /c_best.model['Radius'].astype(np.float32)

        var_init = pd.DataFrame(widths, index=['c' + str(i) for i in range(centroids.shape[0])],
                                columns=['v' + str(i) for i in range(centroids.shape[1])])
        var_init = adjust_rule_widths(X_train, params['n_rules'], centroids, var_init, params['thres_act'],
                                      min_samples, max_samples)
        params['centroids'] = centroids
        params['var_init'] = var_init
        params['rules'] = [f'rule_{i}' for i in range(params['n_rules'])]

    return params


def fix_convolutional_names(horizon_type, experiment, conv_dim):
    for group, branch in experiment.items():
        for i, layer in enumerate(branch):
            if 'conv' in layer[0]:
                if conv_dim is not None:
                    if horizon_type == 'short_term' and group not in {'images'}:
                        experiment[group][i] = (f'time_distr_conv_{conv_dim}d', layer[1])
                    else:
                        experiment[group][i] = (f'conv_{conv_dim}d', layer[1])
                else:
                    raise ValueError('Cannot find conv_dim parameter')
    return experiment


def feed_data(batch, x, y, data, target, lr_pl, lr):
    feed_dict = dict()
    if isinstance(x, dict):
        for key in x.keys():
            if isinstance(x[key], dict):
                for key_1 in x[key].keys():
                    feed_dict.update({x[key][key_1]: data[key][key_1][batch]})
            else:
                feed_dict.update({x[key]: data[key][batch]})
    else:
        feed_dict.update({x: data[batch]})
    feed_dict.update({y: target[batch]})
    feed_dict.update({lr_pl: lr})
    return feed_dict


def feed_data_eval(x, data):
    feed_dict = dict()
    if isinstance(x, dict):
        for key in x.keys():
            if isinstance(x[key], dict):
                for key_1 in x[key].keys():
                    feed_dict.update({x[key][key_1]: data[key][key_1]})
            else:
                feed_dict.update({x[key]: data[key]})
    else:
        feed_dict.update({x: data})
    return feed_dict


def distance(obj_new, obj_old, obj_max, obj_min, weights=None):
    if np.any(np.isinf(obj_old)):
        obj_old = obj_new.copy()
        obj_max = obj_new.copy()
        return True, obj_old, obj_max, obj_min
    if np.any(np.isinf(obj_min)) and not np.all(obj_max == obj_new):
        obj_min = obj_new.copy()
    d = 0
    for i in range(obj_new.shape[0]):
        if obj_max[i] < obj_new[i]:
            obj_max[i] = obj_new[i]
        if obj_min[i] > obj_new[i]:
            obj_min[i] = obj_new[i]
        if weights is None:
            if obj_max[i] - obj_min[i] < 1e-6:
                d += (obj_new[i] - obj_old[i])
            else:
                d += (obj_new[i] - obj_old[i]) / (obj_max[i] - obj_min[i])
        else:
            if obj_max[i] - obj_min[i] < 1e-6:
                d += weights[i] * ((obj_new[i] - obj_old[i]))
            else:
                d += weights[i] * ((obj_new[i] - obj_old[i]) / (obj_max[i] - obj_min[i]))

    if weights is not None:
        d = d / np.sum(weights)
    if d < 0:
        obj_old = obj_new.copy()
        return True, obj_old, obj_max, obj_min
    else:
        return False, obj_old, obj_max, obj_min


def split_validation_set(x):
    if isinstance(x, dict):
        for values in x.values():
            x1 = copy.deepcopy(values)
            break
    else:
        x1 = copy.deepcopy(x)
    if x1.shape[0] > 1000:
        partitions = 250
        ind_list = []
        ind = np.arange(x1.shape[0])
        for i in range(0, x1.shape[0], partitions):
            if (i + partitions + 1) > x1.shape[0]:
                ind_list.append(ind[i:])
            else:
                ind_list.append(ind[i:i + partitions])
    else:
        ind_list = [np.arange(x1.shape[0])]
    return ind_list


def calculate_cpus(n_cpus):
    warm = psutil.cpu_percent()
    average_load = np.mean(psutil.cpu_percent(interval=5, percpu=True)[:n_cpus])

    return n_cpus - int(n_cpus * average_load / 100)


def get_tf_config(n_jobs):
    n_cpus = calculate_cpus(n_jobs)
    if sys.platform != 'linux' and n_cpus > int(n_jobs / 3):
        config = tf.compat.v1.ConfigProto(allow_soft_placement=True,
                                          intra_op_parallelism_threads=2,
                                          inter_op_parallelism_threads=2)
        config.gpu_options.allow_growth = True
    else:
        config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
    return config


def lr_schedule(epoch, lr=1e-04):
    if epoch < 50:
        # WarmUp
        return np.linspace(lr / 10, lr, 50)[epoch]
    else:
        lr_step = 0.5 * lr * (1 + np.cos(np.pi * (epoch - 50) / float(20)))
        return np.maximum(lr / 10, lr_step)


def check_if_all_nans(activations, thres_act, return_len_nan=False):
    """Compatibility wrapper for the shared inclusive rule-membership policy."""
    values, len_nan = ensure_rule_coverage(activations.values, thres_act)
    activations = pd.DataFrame(values, index=activations.index, columns=activations.columns)
    if return_len_nan:
        return activations, len_nan
    else:
        return activations


def _proxy_matrix(X):
    """The clustering features of ``X`` as a 2-D float array (at most the first 6 columns)."""
    x = X
    if isinstance(x, dict):
        x = x['clustering'] if 'clustering' in x else x[list(x.keys())[-1]]
        if isinstance(x, dict):
            x = x[list(x.keys())[-1]]
    if isinstance(x, pd.DataFrame):
        x = x.values
    elif isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x, dtype=float)
    x = x.reshape(x.shape[0], -1)
    return x[:, :6] if x.shape[1] > 5 else x


def _proxy_target(y):
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()
    return np.asarray(y, dtype=float).reshape(y.shape[0], -1)[:, 0]


def _fix_uncovered(act, thres_act):
    return ensure_rule_coverage(act, thres_act)


def linear_output(X_train, X_val, X_test, y_train, y_val, y_test, act_train, act_val, act_test, params):
    """Quality proxy of the rules: one linear model per rule on its active training samples,
    predictions averaged over the active rules; MAE / SSE on validation and test plus the number
    of samples there without any active rule."""
    thres = params['thres_act']
    n_rules = len(params['rules'])
    act_train, _ = _fix_uncovered(act_train, thres)
    act_val, len_nan_val = _fix_uncovered(act_val, thres)
    act_test, len_nan_test = _fix_uncovered(act_test, thres)
    x, x_val, x_test = _proxy_matrix(X_train), _proxy_matrix(X_val), _proxy_matrix(X_test)
    y, y_val, y_test = _proxy_target(y_train), _proxy_target(y_val), _proxy_target(y_test)
    lin_models = dict()
    train_active = rule_membership(act_train, thres)
    for rule in range(n_rules):
        active = train_active[:, rule]
        if active.any():
            lin_models[rule] = LinearRegression().fit(x[active], y[active])

    def predict(x_eval, act_eval):
        preds = np.full((x_eval.shape[0], n_rules), np.nan)
        memberships = rule_membership(act_eval, thres)
        for rule, model in lin_models.items():
            active = memberships[:, rule]
            if active.any():
                preds[active, rule] = model.predict(x_eval[active])
        with np.errstate(all='ignore'):
            return np.nanmean(preds, axis=1)

    def errors(pred, y_eval):
        rated = y_eval if params['rated'] is None else 1
        return (pred - y_eval) / rated

    err_val = errors(predict(x_val, act_val), y_val)
    err_test = errors(predict(x_test, act_test), y_test)
    mae_val = np.nanmean(np.abs(err_val)) + len_nan_val
    mae_test = np.nanmean(np.abs(err_test)) + len_nan_test
    sse_val = np.nansum(np.square(err_val)) + len_nan_val
    sse_test = np.nansum(np.square(err_test)) + len_nan_test
    return mae_val, mae_test, sse_val, sse_test


def find_free_cpus(path_group):
    free_cpus = 0
    warm = psutil.cpu_percent(percpu=True)

    while free_cpus < 2:
        sleep(5)
        load = psutil.cpu_percent(interval=None, percpu=True)
        n_cpus = len(load)
        available_load = n_cpus - int(n_cpus * np.mean(load) / 100) - 1
        print(f'find total {n_cpus} cpus,  mean load {np.mean(load)}, non_cpus {1}')
        free_cpus = n_cpus - 1
        print(
            f'Find load {int(n_cpus * np.mean(load) / 100)}, available {available_load} cpus,'
            f' {free_cpus} cpus free')
        if free_cpus > available_load:
            free_cpus = available_load
    print(f'Find {free_cpus} cpus free')
    return free_cpus


def send_predictions(message, subject='Error for check'):
    """Mail a message; the account comes from the environment (nothing is stored in the code):
    EFORECAST_MAIL_USER, EFORECAST_MAIL_PASSWORD, EFORECAST_MAIL_TO, EFORECAST_MAIL_HOST (smtp.mail.yahoo.com)."""
    import os
    import yagmail
    user, password, to = (os.environ.get('EFORECAST_MAIL_USER'), os.environ.get('EFORECAST_MAIL_PASSWORD'),
                          os.environ.get('EFORECAST_MAIL_TO'))
    if not (user and password and to):
        print('send_predictions: mail account not configured (EFORECAST_MAIL_*); message not sent')
        return
    yag_smtp_connection = yagmail.SMTP(user=user, password=password,
                                       host=os.environ.get('EFORECAST_MAIL_HOST', 'smtp.mail.yahoo.com'))
    yag_smtp_connection.send(to=to, subject=subject, contents=message)


def get_padding(images, max_w, max_h):
    imsize = images.shape[-3:-1]
    h_padding = (max_w - imsize[0]) / 2
    v_padding = (max_h - imsize[1]) / 2
    l_pad = h_padding if h_padding % 1 == 0 else h_padding + 0.5
    t_pad = v_padding if v_padding % 1 == 0 else v_padding + 0.5
    r_pad = h_padding if h_padding % 1 == 0 else h_padding - 0.5
    b_pad = v_padding if v_padding % 1 == 0 else v_padding - 0.5
    zero_pad = tuple([(0, 0) for _ in imsize])
    padding = zero_pad + ((int(l_pad), int(r_pad)), (int(t_pad), int(b_pad)))

    return padding


def pad_images(images, max_w=32, max_h=32, constant=0.0):
    panding = get_padding(images, max_w, max_h)
    return np.pad(images, panding, mode='constant', constant_values=constant)

def initialize_train_constants(params, epochs, len_performers=2):
    mae_old, sse_old = np.inf * np.ones(len_performers), np.inf * np.ones(len_performers)
    mae_max, sse_max = np.inf * np.ones(len_performers), np.inf * np.ones(len_performers)
    mae_min, sse_min = np.inf * np.ones(len_performers), np.inf * np.ones(len_performers)

    best_mae_val, best_mae_test = np.inf, np.inf
    train_flag, best_weights = True, None
    warm = params['warming_iterations']
    wait, best_iteration, best_tot_iteration, loops, n_iter = 0, 0, 0, 0, 0
    patience, exam_period = 8, 30

    results_columns = ['Iteration', 'best_iteration', 'best_mae_val', 'best_mae_test', 'mae_val_out',
                       'mae_test_out', 'sse_val_out', 'sse_test_out']

    results = pd.DataFrame(columns=results_columns)
    return (mae_old, sse_old, mae_max, sse_max, mae_min, sse_min, results_columns, results, best_mae_val, best_mae_test,
            train_flag, best_weights, warm, wait, best_iteration, best_tot_iteration, loops,
            n_iter, patience, exam_period)

def initialize_fuzzy_train_constants(results_columns, epochs, len_performers=2):
    results_columns += ['sum_activations', 'min_activations', 'max_activations',
                        'mean_activations', 'mae_lin_val', 'mae_lin_test']
    if epochs < 2:
        raise ValueError('epochs should be greater than 400 when it is fuzzy')
    mae_old_lin, sse_old_lin = np.inf * np.ones(len_performers), np.inf * np.ones(len_performers)
    mae_max_lin, sse_max_lin = np.inf * np.ones(len_performers), np.inf * np.ones(len_performers)
    mae_min_lin, sse_min_lin = np.inf * np.ones(len_performers), np.inf * np.ones(len_performers)
    results = pd.DataFrame(columns=results_columns)

    return mae_old_lin, sse_old_lin, mae_max_lin, sse_max_lin, mae_min_lin, sse_min_lin, results

def check_if_is_better(mae_old, mae_max, mae_min, sse_old, sse_max, sse_min,
                       mae_val, mae_test, sse_val, sse_test):
    mae = np.hstack([mae_val, mae_test])
    flag_mae, mae_old, mae_max, mae_min = distance(mae, mae_old, mae_max, mae_min)
    sse = np.hstack([sse_val, sse_test]) / 10
    flag_sse, sse_old, sse_max, sse_min = distance(sse, sse_old, sse_max, sse_min)
    flag_best = flag_mae or flag_sse
    if flag_best:
        mae_old, sse_old = mae, sse
    return mae_old, mae_max, mae_min, sse_old, sse_max, sse_min, flag_best

def store_results(results, results_columns, best_tot_iteration, n_iter, best_mae_val, best_mae_test, mae_val,
                  mae_test, sse_val, sse_test, fuzzy=False, sum_act=None, min_act=None, max_act=None, mean_act=None,
                                                       mae_val_lin=None, mae_test_lin=None):
    # best_mae_val / best_mae_test are the metrics of the epoch whose weights were kept (set by the caller
    # when the selection rule accepts an epoch), so the table and the saved weights agree
    evaluation = np.array([n_iter, best_tot_iteration, best_mae_val, best_mae_test,
                           mae_val, mae_test, sse_val, sse_test])

    print_columns = ['best_mae_val', 'best_mae_test', 'mae_val_out', 'mae_test_out']
    if fuzzy:
        evaluation = np.concatenate([evaluation, np.array([sum_act,
                                                           min_act,
                                                           max_act, mean_act,
                                                           mae_val_lin, mae_test_lin])])
        print_columns += ['mae_lin_val', 'mae_lin_test']
    res = pd.DataFrame(evaluation.reshape(-1, 1).T, index=[n_iter], columns=results_columns)
    results = pd.concat([results, res])
    print(res[print_columns])
    return results, best_mae_val, best_mae_test


def check_if_extend_training(epochs, best_iteration, exam_period, patience, loops):
    train_flag = True
    if (epochs - best_iteration) <= exam_period:
        if loops >= 0:
            train_flag = False
        else:
            epochs, exam_period = int(patience), int(patience / 5)
            patience = int(epochs / 1.5)
            best_iteration = 0
            loops += 1
    else:
        train_flag = False
    return epochs, best_iteration, exam_period, patience, loops, train_flag

def check_early_stop(wait, patience, epoch, epochs, sec_per_iter, sec_all):
    train_flag = True
    # if sec_per_iter > 30600:
    #     train_flag = False
    # if sec_all > 144000:
    #     train_flag = False
    return train_flag
