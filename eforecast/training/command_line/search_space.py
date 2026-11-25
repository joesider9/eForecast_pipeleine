# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""The search space of a (cluster, method): the method's parameters (sets = categorical choices,
lists = ranges, scalars = fixed) and the layer sizes of every experiment tag.  ``optuna_search``
turns it into Optuna distributions; ``init_experiment_params`` builds the parameter file of a trial.
"""
import copy
from functools import lru_cache

from sklearn.model_selection import ParameterGrid


@lru_cache(maxsize=None)
def declared_param_types(layer_type):
    """``{parameter: type}`` of the layer catalogue, or ``{}`` for a type it does not know."""
    try:
        from eforecast.config.layer_catalogue import layer_params
    except Exception:
        return {}
    return {entry['name']: entry['type'] for entry in layer_params(layer_type)}


class SearchSpace:

    def __init__(self, static_data, cluster_name, cluster_dir, method):
        self.space_structure = None
        self.static_data = static_data
        self.method = method
        self.cluster_dir = cluster_dir
        self.cluster_name = cluster_name
        self.warming = self.static_data[method]['warming_iterations']
        self.use_image = self.static_data['use_image']
        self.nwp_data_merge = self.static_data['nwp_data_merge']
        self.nwp_data_compress = self.static_data['compress_data']
        self.scale_nwp_methods = self.static_data['scale_nwp_method']
        self.scale_row_methods = self.static_data['scale_row_method']
        self.feature_selection_methods = self.static_data['feature_selection_methods']
        self.define_space()
        self.define_structure_space()

    def get_param(self, name, type, dtype, space=None, range=None, values=None):
        return {name: {'type': type,
                       'dtype': dtype,
                       'space': space,
                       'range': range,
                       'values': values}}

    # ------------------------------------------------------------------ the structure space
    #
    # Every layer parameter can be fixed or searched.  The names of the Optuna dimensions are
    #     <tag>_<branch>_<layer type>_<index>              the size (kept bare, so studies survive)
    #     <tag>_<branch>_<layer type>_<index>_<parameter>  every other parameter
    # The size comes from the ``(type, size)`` tuples of ``static_data['experiments']``; the rest
    # from ``static_data['experiment_layer_params']``, where the loader leaves each one tagged
    # ``{'tag': fixed|choice|range, 'value': ...}``.

    @staticmethod
    def _structure_param_name(experiment_tag, branch, layer_type, index, param=None):
        base = f'{experiment_tag}_{branch}_{layer_type}_{index}'
        return base if param is None else f'{base}_{param}'

    def _add_structure_param(self, experiment_tag, name, value, kind=None, dtype=None):
        """Record one layer parameter: a dimension of the space, or a fixed value.

        ``kind`` is the configured tag.  Without one -- the ``size`` taken from the legacy
        ``(type, size)`` tuples -- it is inferred the way it always was: a set is a set of choices,
        a list is a range, anything else is fixed.  Inferring is not safe for the other parameters,
        where a list can be the value itself (``attention_query: [row_calendar]``).
        """
        if kind is None:
            kind = 'choice' if isinstance(value, set) else 'range' if isinstance(value, list) else 'fixed'
        fixed = self.fix_params_structure[experiment_tag]
        if kind == 'choice':
            values = list(value) if isinstance(value, (set, list, tuple)) else [value]
            if len(values) > 1:
                self.space_structure[experiment_tag].update(
                    self.get_param(name, 'cat', dtype or 'float', values=values))
            else:
                fixed[name] = values[0] if values else value
        elif kind == 'range':
            bounds = list(value) if isinstance(value, (list, tuple)) else [value]
            if len(bounds) > 1:
                self.space_structure[experiment_tag].update(
                    self.get_param(name, 'int' if dtype == 'int' else 'real', dtype or 'float', range=bounds))
            elif len(bounds) == 1:
                fixed[name] = bounds[0]
            else:
                fixed[name] = value
        else:
            fixed[name] = value

    @staticmethod
    def _tagged_value(param):
        """``(kind, value)`` of a ``{'tag': ..., 'value': ...}`` entry, or ``('fixed', param)``."""
        if not isinstance(param, dict) or 'tag' not in param:
            return 'fixed', param                         # already a plain value (a resolved build)
        return param.get('tag') or 'fixed', param.get('value')

    def _layer_params(self, experiment_tag, branch, index):
        """The configured parameters of one layer, or ``{}``."""
        layers = ((self.static_data.get('experiment_layer_params') or {})
                  .get(experiment_tag, {}) or {}).get(branch) or []
        if index >= len(layers) or not isinstance(layers[index], dict):
            return {}
        return layers[index].get('params') or {}

    @staticmethod
    def _as_number(value):
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return value
        if isinstance(value, bool):
            return value
        if isinstance(value, int):
            return float(value)
        return value

    def select_structure(self, trial_structure, experiment_tag, exp):
        """The ``(layer type, size)`` tuples of this trial: the size of every layer resolved."""
        exp_sel = dict()
        for key, layers in exp.items():
            exp_sel[key] = []
            for i, layer in enumerate(layers):
                layer_type = layer[0]
                param = self._structure_param_name(experiment_tag, key, layer_type, i)
                size = trial_structure[param] if param in trial_structure.keys() \
                    else self.fix_params_structure[experiment_tag][param]
                exp_sel[key].append((layer_type, self._as_number(size)))
        return exp_sel

    def select_layer_params(self, trial_structure, experiment_tag):
        """``experiment_layer_params`` of this trial: every searched parameter resolved to one value.

        The builder merges these into the parameters of the layer, so they must be plain values --
        a set or a ``[lo, hi]`` list reaching a layer would be used as if it were the value itself.
        """
        configured = (self.static_data.get('experiment_layer_params') or {}).get(experiment_tag)
        if configured is None:
            return None
        selected = dict()
        for branch, layers in configured.items():
            branch_layers = []
            for i, layer in enumerate(layers):
                if not isinstance(layer, dict):
                    branch_layers.append(layer)
                    continue
                layer_type = layer.get('type')
                params = dict()
                for name, param in (layer.get('params') or {}).items():
                    key = self._structure_param_name(experiment_tag, branch, layer_type, i, name)
                    if key in trial_structure:
                        params[name] = trial_structure[key]
                    elif key in self.fix_params_structure.get(experiment_tag, {}):
                        params[name] = self.fix_params_structure[experiment_tag][key]
                    else:                                 # not searched and not fixed: resolve it here
                        kind, value = self._tagged_value(param)
                        params[name] = value[0] if kind in {'choice', 'range'} \
                            and isinstance(value, (list, tuple)) and value else value
                    params[name] = self._cast_param(layer_type, name, params[name])
                branch_layers.append({**layer, 'params': params})
            selected[branch] = branch_layers
        return selected

    @staticmethod
    def _cast_param(layer_type, name, value):
        """Optuna returns floats; a parameter the catalogue calls an int must arrive as one."""
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return value
        declared = declared_param_types(layer_type)
        if declared.get(name) == 'int':
            return int(round(value))
        if declared.get(name) == 'bool':
            return bool(value)
        return value

    def define_structure_space(self):
        experiment_tags = list(self.static_data[self.method]['experiment_tag'])
        self.space_structure = dict()
        self.fix_params_structure = dict()
        self.param_layer_names = dict()
        for experiment_tag in experiment_tags:
            exp = self.static_data['experiments'][experiment_tag]
            self.space_structure[experiment_tag] = dict()
            self.fix_params_structure[experiment_tag] = dict()
            self.param_layer_names[experiment_tag] = []
            for key, layers in exp.items():
                for i, layer in enumerate(layers):
                    layer_type = layer[0]
                    declared = declared_param_types(layer_type)
                    self._add_structure_param(experiment_tag,
                                              self._structure_param_name(experiment_tag, key, layer_type, i),
                                              layer[1], dtype=declared.get('size'))
                    for name, param in self._layer_params(experiment_tag, key, i).items():
                        kind, value = self._tagged_value(param)
                        self._add_structure_param(
                            experiment_tag,
                            self._structure_param_name(experiment_tag, key, layer_type, i, name),
                            value, kind=kind, dtype=declared.get(name))
        self.param_structure_names = dict()
        for experiment_tag, experiment in self.space_structure.items():
            self.param_structure_names[experiment_tag] = []
            for param_name, param_attr in experiment.items():
                self.param_structure_names[experiment_tag].append(param_name)

    def define_space(self):
        self.space = dict()
        self.fix_params = dict()
        params_method = copy.deepcopy(self.static_data[self.method])
        for param, value in params_method.items():
            if isinstance(value, set):
                if len(value) > 1:
                    if isinstance(list(value)[0], str):
                        self.space.update(self.get_param(param, 'cat', 'string', values=list(value)))
                    elif isinstance(list(value)[0], bool):
                        self.space.update(self.get_param(param, 'bool', 'bool'))
                    elif isinstance(list(value)[0], int):
                        self.space.update(self.get_param(param, 'cat', 'int', values=list(value)))
                    else:
                        self.space.update(self.get_param(param, 'cat', 'float', values=list(value)))
                else:
                    self.fix_params[param] = list(value)[0]
            elif isinstance(value, list):
                if len(value) > 1:
                    if isinstance(value[0], bool):
                        self.space.update(self.get_param(param, 'bool', 'bool'))
                    elif isinstance(value[0], int):
                        self.space.update(self.get_param(param, 'int', 'int', range=value))
                    else:
                        self.space.update(self.get_param(param, 'real', 'float', range=value))
                else:
                    self.fix_params[param] = value[0]
            else:
                self.fix_params[param] = value
        self.param_names = []
        for param_name, param_attr in self.space.items():
            self.param_names.append(param_name)

    def init_experiment_params(self, trial_number, trial, experiment_tag, structure=None):
        image_in_data = False
        merge = trial['merge'] if 'merge' in trial.keys() else self.fix_params['merge']
        compress = trial['compress'] if 'compress' in trial.keys() else self.fix_params['compress']
        scale_row_method = trial['scale_row_method'] if 'scale_row_method' in trial else self.fix_params['scale_row_method']
        scale_nwp_method = trial['scale_nwp_method'] if 'scale_nwp_method' in trial else self.fix_params['scale_nwp_method']
        feature_selection_method = trial['feature_selection_method'] if 'feature_selection_method' in trial.keys() \
            else self.fix_params['feature_selection_method']
        data_types = dict()
        for data_tag in self.static_data['experiments'][experiment_tag].keys():
            if 'output' in data_tag or 'hidden' in data_tag:
                continue
            if 'row' in data_tag:
                if 'all' in data_tag or 'nwp' in data_tag:
                    data_types[data_tag] = {'scale_row_method': scale_row_method,
                                            'scale_nwp_method': scale_nwp_method,
                                            'merge': merge,
                                            'compress': compress,
                                            'feature_selection_method': feature_selection_method
                                            }
                else:
                    data_types[data_tag] = {'scale_row_method': scale_row_method,
                                            'feature_selection_method': feature_selection_method}
            elif 'lstm' in data_tag:
                data_types[data_tag] = {'scale_row_method': scale_row_method,
                                        'feature_selection_method': feature_selection_method
                                        }
            elif 'nwp' in data_tag:
                data_types[data_tag] = {'scale_nwp_method': scale_nwp_method,
                                        'merge': merge}
            elif 'image' in data_tag:
                image_in_data = True
                data_types[data_tag] = {}
        experiment_params = {'trial_number': trial_number,
                             'method': self.method,
                             'name': self.cluster_name,
                             'cluster_dir': self.cluster_dir,
                             'cluster': {'cluster_name':self.cluster_name,
                                         'cluster_path':self.cluster_dir},
                             'data_types': data_types,
                             'image_in_data': image_in_data,
                             'experiment_tag': experiment_tag}
        # the configured parameters are tagged (fixed / choice / range); the builder needs the one
        # value this trial uses, so they are resolved here rather than copied as they stand
        experiment_layer_params = self.select_layer_params(structure or {}, experiment_tag)
        if experiment_layer_params is not None:
            experiment_params['experiment_layer_params'] = copy.deepcopy(experiment_layer_params)

        for param, value in trial.items():
            if param not in experiment_params.keys():
                experiment_params[param] = value
        experiment_params.update(self.fix_params)

        return experiment_params

    def grid_(self):
        """Every combination of the categorical/range values (fix_grid_search)."""
        space_temp = dict()
        for key, value in self.space.items():
            if value['values'] is not None:
                space_temp[key] = value['values']
            elif value['range'] is not None:
                space_temp[key] = value['range']
            elif value['type'] == 'bool':
                space_temp[key] = [True, False]
            else:
                print('space param is not defined correctly')
                space_temp[key] = None
        grid = ParameterGrid(space_temp)
        return list(grid)


ObjectiveAbstractClass = SearchSpace  # former name
