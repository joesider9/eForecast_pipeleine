import os
import importlib
def get_ts_model():
    model_map = {}
    models_dir = os.path.dirname(os.path.abspath(__file__))

    # Iterate through all files in 'models' directory
    if os.path.exists(models_dir):
        for filename in os.listdir(models_dir):
            # Ignore __init__.py and non-.py files
            if filename.endswith('.py') and filename not in {'__init__.py', 'get_model.py'}:
                # Remove .py extension to get module name
                module_name = filename[:-3]

                # Build full import path
                full_path = f"eforecast.deep_models.pytorch_2x.time_series_library.models.{module_name}"

                # loading dict: {'Transformer': 'models.Transformer'}
                model_map[module_name] = full_path

    return model_map


def init_ts_model(name, seq_len, pred_len, enc_in, dec_in, size, params):
    ts_models = get_ts_model()
    module_path = ts_models[name]

    print(f"🚀 Lazy Loading: {name} ...")
    module = importlib.import_module(module_path)
    model_class = module.Model
    configuration = config_ts_model()
    configuration.seq_len = seq_len
    configuration.pred_len = pred_len
    configuration.enc_in = enc_in
    configuration.dec_in = dec_in
    configuration.c_out = dec_in
    configuration.d_model = size
    return model_class(configuration, use_norm=params['use_norm'])


class config_ts_model:
    def __init__(self):


        # model define
        self.expand=1
        self.d_conv=4
        self.tv_dt=0
        self.tv_B=0
        self.tv_C=0
        self.use_D=0
        self.top_k=3
        self.num_kernels=3

        self.n_heads=5
        self.e_layers=2
        self.d_layers=1
        self.d_ff=128
        self.moving_avg=3
        self.factor=2
        self.distil=True
        self.dropout=0.06
        self.embed='timeF'
        #help='time features encoding, options:[timeF, fixed, learned]')
        self.activation='gelu'
        self.channel_independence=1
        self.decomp_method='moving_avg'
        self.use_norm=0
        self.down_sampling_layers=0
        self.down_sampling_window=1
        self.down_sampling_method=None

        # de-stationary projector params
        self.p_hidden_dims=[128, 128]
        self.p_hidden_layers=2

        # metrics (dtw)
        self.use_dtw=False

        # Augmentation
        self.augmentation_ratio=0
        self.seed=2
        self.jitter=False
        self.scaling=False
        self.permutation=False
        self.randompermutation=False
        self.magwarp=False
        self.timewarp=False
        self.windowslice=False
        self.windowwarp=False
        self.rotation=False
        self.spawner=False
        self.dtwwarp=False
        self.shapedtwwarp=False
        self.wdba=False
        self.discdtw=False
        self.discsdtw=False
        self.extra_tag=""
        self.freq='h'
        # TimeXer
        self.patch_len=3

        # GCN
        self.node_dim=10
        self.gcn_depth=2
        self.gcn_dropout=0.3
        self.propalpha=0.3
        self.conv_channel=32
        self.skip_channel=32

        self.individual=False

        # TimeFilter
        self.alpha=0.1
        self.top_p=0.5
        self.pos=0