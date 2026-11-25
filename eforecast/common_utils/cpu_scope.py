# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Metadata-only shallow scope selection; no estimator imports."""
from pathlib import Path

from eforecast.config.capabilities import CPU_METHODS


def methods(static, scope=None, method=None):
    scope = scope or static.get('cpu_scope', 'clusters')
    if scope not in ('global', 'clusters', 'both'):
        raise ValueError(f'unknown CPU scope {scope!r}')
    glob = [m for m, on in (static.get('global_methods') or {}).items() if on and m in CPU_METHODS]
    clusters = [m for m, on in (static.get('methods_cpu') or {}).items() if on]
    if method is not None:
        if method not in CPU_METHODS:
            raise ValueError(f'unsupported CPU method {method!r}')
        glob = clusters = [method]
    return (glob if scope in ('global', 'both') else [], clusters if scope in ('clusters', 'both') else [])


def units(static, scope=None, method=None, clusters=None):
    glob, local = methods(static, scope, method)
    out = [{'kind': 'study', 'scope': 'global', 'cluster': 'global',
            'directory': str(Path(static['path_model']) / 'global'), 'method': m} for m in glob]
    if local and (static.get('is_Fuzzy') or clusters is not None):
        if clusters is None:
            from eforecast.common_utils import trial_layout
            clusters = trial_layout.cluster_folders(static)
        out += [{'kind': 'study', 'scope': 'global' if name == 'global' else 'cluster', 'cluster': name, 'directory': str(path), 'method': m}
                for m in local for name, path in clusters.items()]
    for unit in out:
        unit['id'] = f"{unit['scope']}:{unit['cluster']}:{unit['method']}:study"
    return list({unit['id']: unit for unit in out}.values())


#: The files a unit's trials are fitted on. Their content is what a trial is bound to.
BOUND = ('datasets', 'scalers', 'transformers', 'memberships', 'selectors')


def _semantic(value):
    """Published content hashes are preferred; historical size/mtime identities stay as they are."""
    if isinstance(value, dict):
        if value.get('sha256'):
            return {'sha256': value['sha256']}
        return {k: _semantic(v) for k, v in value.items()}
    return value


def bound_inputs(static, directory):
    """The data of one unit, by content: its datasets, scalers, transforms, masks and selectors."""
    from . import lineage
    inputs = lineage.dependencies(static, scope=str(directory))
    return {key: _semantic(inputs[key]) for key in BOUND}


def training_identity(static, directory, method, *, not_learned=False):
    """Bind a unit to its ordered data bytes, supplied masks, transforms and search.

    Published content hashes are preferred. Historical size/mtime identities stay
    explicitly unverified; this is never permission to alias a global fitted model.

    The search is the family's own: its parameters and everything every family learns from,
    and not the parameters of its siblings nor the settings of the combination steps
    (``lineage.config_identity`` with ``method``). It used to be the whole configuration, so
    changing one network's learning rate made every tree and every foundation model of the case
    another unit, to be fitted again for nothing. A family this configuration does not switch
    on anywhere keeps the whole configuration: a name that cannot be placed is compared more
    strictly, never less.

    ``not_learned`` true is the binding as it was while the settings no family learns from
    still counted (``lineage.NOT_LEARNED``): what a unit fitted in between recorded.
    """
    from . import lineage
    binding = bound_inputs(static, directory)
    family = method if method in lineage.family_keys(static) else None
    binding.update(config=lineage.config_identity(static, budgets=False, method=family, not_learned=not_learned),
                   method=method)
    return lineage.dependency_id(binding)


def earlier_identity(static, directory, method):
    """``training_identity`` as it was while the whole configuration was part of it."""
    from . import lineage
    binding = bound_inputs(static, directory)
    binding.update(config=lineage.config_identity(static, budgets=False), method=method)
    return lineage.dependency_id(binding)


class Binding:
    """Which of a unit's trials were fitted on what the unit has now.

    ``identity`` is what a new trial records. ``admits`` answers for a trial that recorded
    something else, which is every trial fitted before the identity became the family's own:
    its binding holds the configuration of the whole case as it was then, and a sibling's
    parameter -- or a setting of the combination steps -- has usually changed since. Such a
    trial is not another unit's. Whether its *own* family changed is said by the comparability
    id it carries; whether its *data* changed is said by its manifest, which keeps the content
    of every file it was fitted on. When both still hold, nothing the trial depends on is
    different, and it is admitted.
    """

    def __init__(self, static, directory, method):
        from . import trial_layout
        self.identity = training_identity(static, directory, method)
        self.accepted = {self.identity, training_identity(static, directory, method, not_learned=True),
                         earlier_identity(static, directory, method)}
        self._bound = bound_inputs(static, directory)
        self._trials = trial_layout.method_directory(static, directory, method)

    def admits(self, number, recorded, *, same_family=False):
        """Whether the trial ``number`` that recorded the binding ``recorded`` belongs to this unit."""
        if recorded is None or recorded in self.accepted:
            return True
        if not same_family:
            return False
        from . import artifact
        depends = (artifact.read(str(self._trials / f'test_{number}')) or {}).get('depends_on') or {}
        if depends.get('cpu_unit_identity') != recorded:
            return False                        # not the trial this record is about, or no record of its data
        # A trial fitted without a selector keeps none in its manifest, whatever the unit has.
        compared = [key for key in BOUND if key != 'selectors' or depends.get('selectors')]
        return all(_semantic(depends.get(key)) == self._bound[key] for key in compared)
