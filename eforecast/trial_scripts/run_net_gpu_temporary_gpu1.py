# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

import copy
import os
import sys
import pandas as pd
import numpy as np
from collections import OrderedDict

path_pycharm = (os.path.normpath(os.path.dirname(__file__))).split(os.sep)
path_pycharm = os.path.join(*path_pycharm[:-3])
if sys.platform == 'linux':
    path_pycharm = '/' + path_pycharm
print(path_pycharm)
sys.path.append(path_pycharm)

import joblib



import torch
from eforecast.deep_models.pytorch_2x.network import DeepNetwork


# path_weights = sys.argv[1]
# gpu_id = int(sys.argv[2])
# refit = bool(sys.argv[3])

path_weights = ('/media/smartrue/HHD1/George/models/ADMIE/RejectedPower_ver0/wind/ΑΓΓΕΛΟΚΑ_ΑΓΓΕΛΟΚΑ/intra-output/'
                'model_ver1/cluster_organizer/RBF/rule_0/MLP/exp')
gpu_id = 1



class Trainer:
    def __init__(self, static_data, params, path_weights, refit):
        self.params = params
        self.static_data = static_data
        self.method = self.params['method']
        self.path_weights = path_weights
        self.refit = refit
        self.cluster_name = self.params['name']
        self.cluster_dir = self.params['cluster_dir']
        self.use_image = self.static_data['use_image']


    def train(self, gpu_id):
        cv_masks = joblib.load(os.path.join(self.cluster_dir, 'cv_mask.pickle'))
        if self.refit:
            for f in os.listdir(self.path_weights):
                if 'checkpoint' in f or 'net_weights' in f:
                    os.remove(os.path.join(self.path_weights, f))
        self.model = DeepNetwork(self.static_data, self.path_weights, self.params, train=True, refit=self.refit)
        self.model.fit(cv_masks, gpu_id=gpu_id)

    def evaluate(self, id):
        from eforecast.datasets.data_feeder import DataFeeder
        from eforecast.common_utils.eval_utils import compute_metrics
        res = []
        self.model = DeepNetwork(self.static_data, self.path_weights, train=False)
        pred = self.model.predict(cuda=None)
        data_feeder = DataFeeder(self.static_data, train=False)
        y = data_feeder.feed_target(inverse=False)
        dates = pred.index.intersection(y.index)
        res_eval = compute_metrics(pred.loc[dates], y.loc[dates], 1,
                                   'main')['mae'].to_frame()
        res_train =pd.DataFrame({'mae_val': self.model.best_mae_val, 'mae_test': self.model.best_mae_test}, index=['main'])
        res_eval = pd.concat([res_eval, 100*res_train], axis=1)
        res.append(res_eval)
        print(res_eval)
        for file_weights in os.listdir(self.path_weights):
            if 'checkpoint' in file_weights and not 'mae' in file_weights:
                best_weights = torch.load(os.path.join(self.path_weights, file_weights), weights_only=False,
                           map_location=torch.device('cpu'))
                self.model.best_weights = best_weights
                pred = self.model.predict(cuda=None)
                res_eval = compute_metrics(pred.loc[dates], y.loc[dates], 1,
                                           file_weights)['mae'].to_frame()
                res_train = pd.DataFrame(joblib.load(os.path.join(self.path_weights,
                                                                  file_weights.replace('checkpoint',
                                                                                       'checkpoint_mae'))), index=[file_weights])
                res.append(pd.concat([res_eval, 100*res_train], axis=1))
                print(res_eval)
        res = pd.concat(res)
        res['average'] = res.mean(axis=1)
        best_weights_file = res['average'].idxmin()
        for file_weights in os.listdir(self.path_weights):
            if 'checkpoint' in file_weights and file_weights != best_weights_file:
                os.remove(os.path.join(self.path_weights, file_weights))
        if best_weights_file == 'main':
            pass
        else:
            self.model.best_weights = torch.load(os.path.join(self.path_weights, best_weights_file),
                                                 weights_only=False,
                                                map_location=torch.device('cpu'))
            os.remove(os.path.join(self.path_weights, best_weights_file))
            self.model.save()
        return res.loc[best_weights_file]

## TODO: Run experiments with different Transformers
## https://github.com/thuml/Time-Series-Library
## https://github.com/thuml/TimeXer/tree/main
## https://github.com/lucidrains/iTransformer/tree/main
## https://github.com/lucidrains/rotary-embedding-torch/tree/main
## https://github.com/lucidrains/deep-cross-attention/blob/main/deep_cross_attention/dca_gpt.py
## https://github.com/lucidrains/bidirectional-cross-attention/tree/main
## https://github.com/lucidrains/strassen-attention/tree/main/strassen_attention
## https://github.com/lucidrains/axial-positional-embedding
if __name__ == '__main__':
    import torch
    torch.cuda.empty_cache()
    path_cluster = os.path.dirname(os.path.dirname(path_weights))
    static_data = joblib.load(os.path.join(path_cluster, 'static_data.pickle'))
    # for k, v in static_data.items():
    #     if isinstance(v, str):
    #         if '/home/smartrue/Dropbox' in v:
    #             static_data[k] = v.replace('/home/smartrue/Dropbox', '/media/sider/data/Dropbox')
    #         elif '/media/smartrue/HHD1/George/models/' in v:
    #             static_data[k] = v.replace('/media/smartrue/HHD1/George/models/', '/media/sider/data/Dropbox/data_transfer/models/')
    for k, v in static_data.items():
        if isinstance(v, str):
            if '/media/sider/data/Dropbox' in v:
                static_data[k] = v.replace('/media/sider/data/Dropbox', '/home/smartrue/Dropbox')
            elif '/media/sider/data/models/' in v:
                static_data[k] = v.replace('/media/sider/data/models/', '/media/smartrue/HHD1/George/models/')
    params = joblib.load(os.path.join(path_weights, 'parameters.pickle'))
    # trainer = Trainer(static_data, params, path_weights, True)
    # trainer.train(gpu_id)
    # res = trainer.evaluate()
    # joblib.dump(res, os.path.join(path_weights, f'results_eval.pickle'))


    id = 2
    params['data_types'] = {

                        'lstm': {
                            'scale_nwp_method': 'minmax',
                            'scale_row_method': 'minmax',
                            'compress': 'full',
                            'merge': 'all',
                        },

                        }

    experiments = [
        {'lstm': [('bi_transformer', 48),
                  ('Flatten', []),
                  ('dense', 128),
                  ],
         'output': [
             ('dense', {32})
         ]},

        { 'lstm': [('bi_lstm', 48),
                 ('Flatten', []),
                 ('dense', 128),
                 ],
        'output': [
            ('dense', 32)
        ]},
    ]
    params['inputs'] = [i for i in params['experiment'].keys() if 'output' not in i]
    params['max_iterations'] = 80
    params['batch_size'] = 256
    params['cluster_dir'] = path_cluster
    params['cluster']['cluster_path'] = path_cluster
    params['feature_selection_method'] = None
    for id, exp in enumerate(experiments):
        params['experiment'] = copy.deepcopy(exp)
        layer_name = params['experiment']['lstm'][0][0]
        for optimizer in {'adamw', 'prodigy'}:
            if layer_name == 'bi_transformer':
                for norm in {True, False}:
                    for name_model in [ 'TimesNet', 'DLinear', 'iTransformer', 'TimeXer', 'Nonstationary_Transformer']:
                        params['use_norm'] = norm
                        params['optimizer'] = optimizer
                        params['model_name'] = name_model
                        tag = f'{optimizer}_{name_model}' if norm else f'{optimizer}_{name_model}_norm'
                        if os.path.exists(os.path.join(path_weights, f'results_{tag}.csv')):
                            continue
                        trainer = Trainer(static_data, params, path_weights, True)
                        trainer.train(gpu_id)
                        res = trainer.evaluate(id)

                        res.to_csv(os.path.join(path_weights, f'results_{tag}.csv'))
            elif layer_name == 'bi_lstm':
                for kd in range(1, 11):
                    params['optimizer'] = optimizer
                    params['keep_dim'] = kd
                    tag = f'{optimizer}_{layer_name}_keep_dim_{kd}'
                    if os.path.exists(os.path.join(path_weights, f'results_{tag}.csv')):
                        continue
                    trainer = Trainer(static_data, params, path_weights, True)
                    trainer.train(gpu_id)
                    res = trainer.evaluate(id)

                    res.to_csv(os.path.join(path_weights, f'results_{tag}.csv'))

