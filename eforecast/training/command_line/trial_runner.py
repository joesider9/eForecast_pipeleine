# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Train and evaluate ONE trial in this process (the child of a GPU worker).

    python trial_runner.py --path-weights <cluster_dir>/<method>/test_<k> [--gpu 0] [--refit]

Reads ``parameters.pickle`` from the trial folder and ``static_data.pickle`` from the
cluster folder, trains, evaluates, and writes ``trial_result.json``.  While training it
writes ``progress.json`` after every epoch and stops early (keeping the best weights so
far) when the worker creates ``prune.flag``.  Exit code 0 = complete or pruned,
3 = CUDA out of memory, 1 = any other error (traceback in ``error.txt``).

It is also the way to debug a single trial: run the command above under a debugger.
"""
import argparse
import resource
import json
import os
import shutil
import sys
import time
import traceback

path_pycharm = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, os.pardir))
sys.path.append(path_pycharm)

from eforecast.common_utils import artifact, completion, lineage
from eforecast import execution_policy
from eforecast.training.command_line import resource_reporting as resources

EXIT_OK, EXIT_ERROR, EXIT_OOM = 0, 1, 3

# Keep importing the runner/help lightweight. Existing Python/debug callers can still
# access these public attributes; the CLI resolves placement before importing them.
_TRAINING_IMPORTS = {'joblib': ('joblib', None), 'np': ('numpy', None), 'pd': ('pandas', None),
    'torch': ('torch', None), 'DeepNetwork': ('eforecast.deep_models.pytorch_2x.network', 'DeepNetwork'),
    'DataFeeder': ('eforecast.datasets.data_feeder', 'DataFeeder'),
    'get_time_level': ('eforecast.common_utils.dataset_utils', 'get_time_level'),
    'intersection_index': ('eforecast.common_utils.dataset_utils', 'intersection_index'),
    'is_cuda_oom': ('eforecast.common_utils.devops_utils', 'is_cuda_oom'),
    'compute_metrics': ('eforecast.common_utils.eval_utils', 'compute_metrics')}


def __getattr__(name):
    if name not in _TRAINING_IMPORTS:
        raise AttributeError(name)
    import importlib
    module, attribute = _TRAINING_IMPORTS[name]
    value = importlib.import_module(module)
    globals()[name] = getattr(value, attribute) if attribute else value
    return globals()[name]


def _load_training():
    for name in _TRAINING_IMPORTS:
        if name not in globals():
            __getattr__(name)


# How much a trial may still grow after the epoch that measured it: the allocator's pool
# fragments a little over a long run, and a trial that dies of it late has lost all its epochs.
MEMORY_GROWTH = float(os.environ.get('EFORECAST_GPU_GROWTH', 0.15))
MEMORY_MARGIN_MB = 64


def settled_cap_mb(peak_mb, admitted_mb):
    """The limit a trial keeps once it knows its size: what it used, room to grow, never more than it was given."""
    return min(float(admitted_mb), float(peak_mb) * (1 + MEMORY_GROWTH) + MEMORY_MARGIN_MB)


def oom_evidence(text):
    """What a CUDA out-of-memory error says about the size that was needed.

    The scheduler used to learn nothing from a failed trial but the fact of it, and raised its
    guess by half. The process knows more: what it held when it failed, and how much the
    allocation that failed asked for. Together they are a lower bound on what it needs.
    """
    import re
    evidence = {}
    match = re.search(r'Tried to allocate ([0-9.]+) (MiB|GiB|KiB)', text or '')
    if match:
        scale = {'KiB': 1 / 1024, 'MiB': 1.0, 'GiB': 1024.0}[match.group(2)]
        evidence['oom_requested_mb'] = float(match.group(1)) * scale
    try:
        import torch
        if torch.cuda.is_available():
            evidence['peak_memory_mb'] = torch.cuda.max_memory_reserved() / 2 ** 20
    except Exception:           # noqa: BLE001 -- evidence is help for the retry, never a second failure
        pass
    return evidence


class TrialMonitor:
    """Progress file for the worker and prune flag from the worker.

    The file also carries what this process is really using. A scheduler that admits a trial
    on a guess has to keep the guess for the whole trial; one that is told the size after the
    first epoch can give the rest of the card to somebody else.
    """

    def __init__(self, path_weights, settle_after=0):
        self.progress_file = os.path.join(path_weights, 'progress.json')
        self.prune_flag = os.path.join(path_weights, 'prune.flag')
        self.start = time.time()
        self.last_report = self.start       # an epoch lasts from the report before it to its own
        # The first epoch whose training step really ran: a transfer-learning trial skips the
        # step of epoch 0, and a size measured there is the size of a validation pass.
        self.settle_after = int(settle_after)
        self.cap_mb = None
        self.cpu_start = sum(os.times()[:4])     # CPU time is reported over the same span as ``elapsed``

    def resources(self, epoch):
        """GPU memory held and peaked, the limit this trial settled on, and its CPU time since it began."""
        fields = {'cpu_seconds': float(sum(os.times()[:4]) - self.cpu_start)}
        try:
            import torch
            if not torch.cuda.is_available():
                return fields
            fields['gpu_peak_mb'] = torch.cuda.max_memory_reserved() / 2 ** 20
            fields['gpu_reserved_mb'] = torch.cuda.memory_reserved() / 2 ** 20
            if (self.cap_mb is None and epoch >= self.settle_after
                    and os.environ.get('EFORECAST_GPU_SETTLE') == '1' and os.environ.get('EFORECAST_MEM_FRACTION')):
                total = torch.cuda.get_device_properties(0).total_memory / 2 ** 20
                admitted = float(os.environ['EFORECAST_MEM_FRACTION']) * total
                self.cap_mb = settled_cap_mb(fields['gpu_peak_mb'], admitted)
                # From here this process cannot take more than it measured plus its growth room,
                # so what the scheduler now gives to a neighbour is really free.
                torch.cuda.set_per_process_memory_fraction(min(1.0, max(1e-4, self.cap_mb / total)), 0)
                print(f'memory settled at {self.cap_mb:.0f} MB after epoch {epoch}: peak '
                      f"{fields['gpu_peak_mb']:.0f} MB of the {admitted:.0f} MB it was admitted with", flush=True)
            if self.cap_mb is not None:
                fields['gpu_cap_mb'] = self.cap_mb
        except Exception as error:      # noqa: BLE001 -- a report must never end the training it reports on
            fields['resources_error'] = str(error)[:200]
        return fields

    def report(self, epoch, mae_val, mae_test, best_mae_val, best_mae_test):
        now = time.time()
        # epoch_seconds: what `bin/eforecast trials` shows as epochs per second (the first one
        # includes the setup before it)
        payload = {'epoch': int(epoch), 'mae_val': float(mae_val), 'mae_test': float(mae_test),
                   'best_mae_val': float(best_mae_val), 'best_mae_test': float(best_mae_test),
                   'elapsed': now - self.start, 'time': now, 'epoch_seconds': now - self.last_report}
        self.last_report = now
        payload.update(self.resources(epoch))
        tmp = self.progress_file + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as stream:
            json.dump(payload, stream)
        os.replace(tmp, self.progress_file)
        return os.path.exists(self.prune_flag)


class Trainer:
    def __init__(self, static_data, params, path_weights, refit, control=None):
        _load_training()
        self.params = params
        self.static_data = static_data
        self.path_weights = path_weights      # where this attempt writes: nothing of another attempt is here
        self.control = control or path_weights  # the trial folder the worker watches: progress and prune flag
        self.refit = refit
        self.cluster_dir = self.params['cluster_dir']
        self.model = None

    def train(self, gpu_id):
        if not self.static_data['transfer_learning_merge_data']:
            cv_masks = joblib.load(os.path.join(self.cluster_dir, 'cv_mask.pickle'))
        else:
            cv_masks = None
        if self.refit:
            for name in os.listdir(self.path_weights):
                if 'checkpoint' in name or 'net_weights' in name:
                    os.remove(os.path.join(self.path_weights, name))
        self.model = DeepNetwork(self.static_data, self.path_weights, self.params, train=True, refit=self.refit)
        skips_first_step = bool(self.params.get('tl_path_weights') or self.static_data.get('transfer_learning_merge_data'))
        self.model.monitor = TrialMonitor(self.control, settle_after=1 if skips_first_step else 0)
        self.model.fit(cv_masks, gpu_id=gpu_id)
        return bool(getattr(self.model, 'pruned', False))

    def evaluate(self, gpu_id):
        res = []
        self.model = DeepNetwork(self.static_data, self.path_weights, train=False)
        # The final weights and every checkpoint are predicted on the same inputs: prepare them once.
        self.model.hold_inputs = True
        try:
            return self._evaluate(gpu_id, res)
        finally:
            self.model.release_inputs()

    def _evaluate(self, gpu_id, res):
        pred = self.model.predict(cuda=gpu_id)
        y = DataFeeder(self.static_data, train=False).feed_target(inverse=False)
        common, dates, _, ind1, ind2 = intersection_index(pred.index, y.index)
        # only the selection slice of the evaluation period may influence the model (the holdout is the report)
        from eforecast.combine_predictions.ensemble import split_evaluation
        selection, _, _ = split_evaluation(y.index, self.static_data.get('selection_fraction', 0.5),
                                           calibration_fraction=self.static_data.get('calibration_fraction', 0.0))
        dates = dates[get_time_level(dates).isin(selection)]
        tdates = get_time_level(dates)
        res_eval = compute_metrics(pred.loc[dates], y.loc[tdates], 1, 'main')['mae'].to_frame()
        res_train = pd.DataFrame({'mae_val': self.model.best_mae_val, 'mae_test': self.model.best_mae_test}, index=['main'])
        res.append(pd.concat([res_eval, 100 * res_train], axis=1))
        for file_weights in os.listdir(self.path_weights):
            if 'checkpoint' in file_weights and 'mae' not in file_weights:
                self.model.best_weights = torch.load(os.path.join(self.path_weights, file_weights), weights_only=False,
                                                     map_location=torch.device('cpu'))
                pred = self.model.predict(cuda=gpu_id)
                res_eval = compute_metrics(pred.loc[dates], y.loc[tdates], 1, file_weights)['mae'].to_frame()
                info = joblib.load(os.path.join(self.path_weights, file_weights.replace('checkpoint', 'checkpoint_mae')))
                res_train = pd.DataFrame({'mae_val': 100 * info['mae_val'], 'mae_test': 100 * info['mae_test'],
                                          'epoch': info.get('epoch', np.nan)}, index=[file_weights])
                res.append(pd.concat([res_eval, res_train], axis=1))
        res = pd.concat(res)
        res['average'] = res[['mae', 'mae_val', 'mae_test']].mean(axis=1)
        best_weights_file = res['average'].idxmin()
        print(f'evaluation of the final weights and {len(res) - 1} checkpoints on the unseen period '
              f'(mae, and the training metrics x100):\n{res.round(3).to_string()}\n-> keeping {best_weights_file}')
        for file_weights in os.listdir(self.path_weights):
            if 'checkpoint' in file_weights and file_weights != best_weights_file:
                os.remove(os.path.join(self.path_weights, file_weights))
        if best_weights_file != 'main':
            self.model.best_weights = torch.load(os.path.join(self.path_weights, best_weights_file), weights_only=False,
                                                 map_location=torch.device('cpu'))
            os.remove(os.path.join(self.path_weights, best_weights_file))
            self.model.save()
        return {'exp': self.params['experiment'], 'mae': res.loc[best_weights_file]}


TRIAL_FILES = artifact.KINDS['gpu_trial']['required']
ATTEMPT_PREFIX = '.attempt-'


def attempt_directory(path_weights, attempt):
    """Where this attempt works. Its outputs reach the trial folder only once they are finished."""
    work = os.path.join(path_weights, f'{ATTEMPT_PREFIX}{attempt}')
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work, exist_ok=True)
    return work


def promote(work, path_weights):
    """Move a finished attempt's files into the trial folder, each one atomically.

    The result and the manifest are not here: they are written afterwards, in that order, so that
    an interruption anywhere in this loop leaves a folder that nothing claims to be finished.
    """
    moved = []
    for name in sorted(os.listdir(work)):
        source = os.path.join(work, name)
        if os.path.isfile(source):
            os.replace(source, os.path.join(path_weights, name))
            moved.append(name)
    for stale in sorted(os.listdir(path_weights)):      # this attempt won; no earlier one is kept
        if stale.startswith(ATTEMPT_PREFIX):
            shutil.rmtree(os.path.join(path_weights, stale), ignore_errors=True)
    return moved


def write_result(path_weights, depends_on=None, **fields):
    from eforecast import model_library
    if model_library.records():
        fields.setdefault('model_library', model_library.records())
    if execution_policy.current() is not None:
        fields.setdefault('execution_policy', execution_policy.current())
    fields.setdefault('peak_host_mb', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)   # Linux: KB
    identity = artifact.worker_identity(path_weights)
    for name, value in identity.items():
        fields.setdefault(name, value)
    with open(os.path.join(path_weights, 'trial_result.json'), 'w', encoding='utf-8') as stream:
        json.dump(fields, stream, indent=1, default=str)
    # the manifest goes last: it is the claim that the files above are finished work, and a reader
    # admits a trial folder on it rather than on the presence of weights the trainer writes early
    published = artifact.publish(path_weights, kind='gpu_trial', state=str(fields.get('state')),
                     value=fields.get('value'), duration=fields.get('duration'), depends_on=depends_on,
                     **identity)

    if published and fields.get('state') in artifact.USABLE_STATES:
        from eforecast.common_utils.trial_layout import activate
        activate(path_weights)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--path-weights', required=True, help='the trial folder <cluster_dir>/<method>/test_<k> holding parameters.pickle (written by the orchestrator or --prepare-only)')
    parser.add_argument('--gpu', type=int, default=0, help='CUDA device index as seen by this process')
    parser.add_argument('--refit', action='store_true', help='remove the checkpoints and weights of the folder and train again')
    parser.add_argument('--cpu', action='store_true', help='train on the CPU (debugging); otherwise a missing GPU is an error')
    args = parser.parse_args(argv)
    inherited = execution_policy.current()
    if inherited:
        # Started by the orchestrator: it chose the card and the threads.
        threads, device_uuid = inherited['given']['cpu_threads'], inherited['given'].get('device_uuid')
        requested = dict(inherited['requested'])
    else:
        # Started by hand, to debug one trial: --gpu names the card by its order, as CUDA shows
        # it, and the card is found by its UUID so that this process sees that card only.
        threads = int(os.environ.get('EFORECAST_TRIAL_THREADS') or 1)
        requested, device_uuid = {'device': args.gpu}, None
        if not args.cpu:
            from eforecast.config.model_compute import device_for
            device_uuid = device_for(f'cuda:{args.gpu}')
    if args.cpu:
        requested['device'] = 'cpu'
    policy = execution_policy.resolve(threads, device_uuid, requested)
    with execution_policy.activated(policy):
        print('execution policy: ' + json.dumps(policy, sort_keys=True), flush=True)
        execution_policy.configure_torch(policy)
        _load_training()
        return _run(args, policy)


def _run(args, policy):
    from eforecast.config.run_reporting import worker_started
    worker_started()
    path_weights = os.path.abspath(args.path_weights)
    identity = artifact.worker_identity(path_weights)
    # Finished work is reused because something says it is finished, not because weights are lying
    # about. Without this the network would load them and return its old answer in silence.
    if not args.refit:
        finished, reason = completion.is_finished(path_weights)
        if finished:
            print(f'{path_weights} is already finished work ({reason}); not training it again', flush=True)
            return EXIT_OK
    # Clear the previous attempt's completion claim before even loading inputs.
    for stale in (artifact.MANIFEST_FILE, 'progress.json', 'prune.flag', 'trial_result.json', 'error.txt'):
        if os.path.exists(os.path.join(path_weights, stale)):
            os.remove(os.path.join(path_weights, stale))
    if policy['effective']['device'] == 'cpu':
        args.gpu = 'cpu'
    else:
        execution_policy.torch_device(args.gpu)  # verify the physical UUID before model/data loading
        args.gpu = 0
    path_cluster = os.path.dirname(os.path.dirname(path_weights))
    local_static = os.path.join(os.path.dirname(path_weights), 'static_data.pickle')
    static_data = joblib.load(local_static if os.path.isfile(local_static) else os.path.join(path_cluster, 'static_data.pickle'))
    params = joblib.load(os.path.join(path_weights, 'parameters.pickle'))
    scope = os.path.dirname(os.path.dirname(path_weights))
    from eforecast.common_utils.image_assets import uses_pixels
    raw_pixels = isinstance(params.get('experiment'), dict) and uses_pixels(static_data, params['experiment'])
    # The id this trial may be compared by is its family's own: the folder then says by itself
    # whether a later change of the case was a change of this family (``revisions``).
    inputs = lineage.dependencies(static_data, scope=scope, raw_pixels=raw_pixels,
                                  method=params.get('method') if isinstance(params, dict) else None)
    input_observations = lineage.input_observations(static_data, scope, raw_pixels=raw_pixels)
    threads = policy['effective']['threads']
    resource_preferences = {'configured': {key: static_data.get(key) for key in ('n_cpus', 'n_jobs')},
                            'worker_effective': {'n_cpus': threads, 'n_jobs': 1},
                            'reason': 'per-worker launch budget; worker data/estimator jobs stay serial'}
    print('worker resource preferences: ' + json.dumps(resource_preferences, sort_keys=True, default=str), flush=True)
    static_data.update(n_cpus=threads, n_jobs=1)
    work = attempt_directory(path_weights, identity['attempt'] if identity['attempt'] is not None else 0)
    print(f"attempt {identity['attempt_id']} works in {work}", flush=True)
    start = time.time()
    fraction = os.environ.get('EFORECAST_MEM_FRACTION')
    if fraction and torch.cuda.is_available():
        # the share of the card the orchestrator started this trial with: growing beyond it
        # raises a CUDA OOM in this process only, instead of in a trial that shares the card
        torch.cuda.set_per_process_memory_fraction(min(1.0, max(0.05, float(fraction))), 0)
    torch.cuda.empty_cache()
    state, error_text = 'complete', None
    limits = resources.library_threads(threads)
    limits.__enter__()
    image_scope = None
    from eforecast import model_library
    model_library.reset()
    try:
        from eforecast.datasets.image_data.batches import owned_batches
        image_scope = owned_batches(work)
        image_scope.__enter__()
        trainer = Trainer(static_data, params, work, args.refit, control=path_weights)
        pruned = trainer.train(args.gpu)
        state = 'pruned' if pruned else 'complete'
        model = trainer.model
        peak = torch.cuda.max_memory_reserved() / 2 ** 20 if torch.cuda.is_available() else 0.0
        if state == 'complete' and not os.path.exists(os.path.join(work, 'results_eval.pickle')):
            evaluator = Trainer(static_data, params, work, False, control=path_weights)
            res = evaluator.evaluate(args.gpu)
            joblib.dump(res, os.path.join(work, 'results_eval.pickle'))
            model = evaluator.model
        value = float(model.best_mae_test)
        moved = lineage.inputs_changed(input_observations,
                                       lineage.input_observations(static_data, scope, raw_pixels=raw_pixels))
        if moved:
            raise ValueError('trial inputs changed during fitting; private result was not published: '
                             + ', '.join(moved[:3]))
        promote(work, path_weights)
        write_result(path_weights, depends_on=inputs,
                     resource_preferences=resource_preferences,
                     state=state, value=value, mae_test=float(model.best_mae_test),
                     mae_val=float(model.best_mae_val), sse_test=float(model.best_sse_test),
                     sse_val=float(model.best_sse_val), duration=time.time() - start, peak_memory_mb=peak)
        return EXIT_OK
    except Exception as error:
        error_text = ''.join(traceback.format_exception(error))
        print(error_text, file=sys.stderr)
        with open(os.path.join(path_weights, 'error.txt'), 'w', encoding='utf-8') as stream:
            stream.write(error_text)
        oom = is_cuda_oom(str(error))
        # The attempt's own files stay where they are: a failure publishes its record, not a payload.
        write_result(path_weights, depends_on=inputs, state='oom' if oom else 'error',
                     resource_preferences=resource_preferences, **(oom_evidence(str(error)) if oom else {}),
                     **({'prerequisite': error.to_dict()} if isinstance(error, model_library.ModelPrerequisite) else {}),
                     value=None, duration=time.time() - start, error=str(error)[:2000])
        return EXIT_OOM if oom else EXIT_ERROR
    finally:
        if image_scope is not None:
            image_scope.__exit__(None, None, None)
        limits.__exit__(None, None, None)


if __name__ == '__main__':
    sys.exit(main())
