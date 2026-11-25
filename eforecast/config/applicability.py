# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""A13-08: selections that are legal, stored, and do nothing.

A method can carry a parameter that reads like a choice of backend while the graph it actually
runs never looks at it. Nothing fails; the search simply explores a dimension that changes
nothing, and the selection report appears to compare two things while comparing one.

The real case: ``pv_park1``'s TRANSFORMER method offers
``model_name: {'choices': ['TimesNet', 'TimeXer']}``, and its graphs ``transformer1/2/3`` are
built from ``cnn_att``, ``conv``, ``cross_attention``, ``dense`` and ``Flatten`` -- none of which
takes ``model_name``. Only ``transformer``, ``bi_transformer``, ``vit_net``, ``timm_net`` and
``time_distr_vit_net`` do.

Which kinds consume which parameter is read from the existing layer catalogue rather than
written down here, so the rule cannot drift from the layers. The findings use the same shape as
the A13-04 readiness report and are folded into it, so one panel and one command show them.

Nothing here edits a case. An inactive value is reported, never removed, and an architecture is
never silently replaced: that is the colleague's decision to make with the facts in front of them.
"""
from __future__ import annotations

from typing import Any

from eforecast.config.readiness import ATTENTION, BLOCKING, OK, _at, _finding

#: Where a method's graph choice is stored.
TAG_PARAMETER = 'experiment_tag'

#: Parameters that select *what a graph executes*, and are consumed nowhere else.
#:
#: Deliberately explicit and deliberately short. The first version of this check asked a more
#: general question -- "does any layer declare this parameter, and does this graph contain one?"
#: -- and on the owner's real cases that reported ``use_nwp`` and ``batch_norm`` as inert for
#: four methods. Both are ambient knobs: they sit in ``graph_service.DEFAULT_LAYER_PARAMS`` and
#: in the family presets, and are read outside the layer graph. Reporting them trains a reader
#: to ignore the panel, which is worse than having no panel.
#:
#: To add a parameter here, confirm that **nothing outside the layer classes reads it**. For
#: ``model_name`` that holds: only ``transformer``, ``bi_transformer``, ``vit_net``, ``timm_net``
#: and ``time_distr_vit_net`` consume it, and for the non-graph families it is a checkpoint
#: identity rather than a graph selector -- which is why this check only looks at graph families.
GRAPH_SELECTING = frozenset({'model_name'})


def _values(value) -> list[Any]:
    """The values a stored parameter can take: a plain one, or a search space's choices."""
    if isinstance(value, dict) and 'choices' in value:
        choices = value['choices']
        return list(choices) if isinstance(choices, (list, tuple)) else [choices]
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def consumers(parameter: str) -> set[str]:
    """The layer kinds whose catalogue entry declares ``parameter``."""
    from eforecast.config.layer_catalogue import LAYERS, param_names
    return {kind for kind in LAYERS if parameter in set(param_names(kind))}


def graph_kinds(document, tag: str) -> set[str] | None:
    """The layer kinds an experiment uses, or ``None`` when the experiment cannot be read."""
    experiment = (_at(document, 'experiments') or {}).get(tag)
    if not isinstance(experiment, dict):
        return None
    kinds = set()
    for layers in experiment.values():
        if not isinstance(layers, list):
            continue
        for layer in layers:
            if isinstance(layer, dict) and layer.get('type'):
                kinds.add(str(layer['type']))
    return kinds


def graph_families() -> set[str]:
    """The families that run an experiment graph, as their own presets declare.

    A shallow or foundation method has no graph, so asking which graph it runs is meaningless --
    and the first version of this check asked anyway, reporting "no experiment is named" for
    CatBoost, Chronos2, TimesFM and TiRex on the owner's cases.
    """
    from eforecast.config.family_presets import PRESETS
    return {name for name, preset in PRESETS.items()
            if isinstance(preset, dict) and TAG_PARAMETER in preset}


def enabled_methods(document) -> list[str]:
    """Methods this case would actually run. A method that is off keeps its settings in peace."""
    names = []
    for group in ('methods_cpu', 'cluster_methods', 'global_methods'):
        for name, on in (_at(document, f'methods.{group}') or {}).items():
            if on and name not in names:
                names.append(name)
    return names


def tags_of(document, method: str) -> list[str]:
    return [str(tag) for tag in _values(_at(document, f'methods.method_params.{method}.{TAG_PARAMETER}'))
            if tag not in (None, '')]


# ---------------------------------------------------------------------------- the checks

def _inert_findings(document, method, tags) -> list[dict[str, Any]]:
    """Parameters some layer kind takes, but no layer of *these* graphs does."""
    from eforecast.config.layer_catalogue import LAYERS
    params = _at(document, f'methods.method_params.{method}') or {}
    findings = []
    for name, stored in sorted(params.items()):
        if name not in GRAPH_SELECTING:
            continue
        takers = consumers(name)
        if not takers:
            continue                                            # the catalogue no longer lists it
        values = [value for value in _values(stored) if value not in (None, '')]
        if not values:
            continue                                            # nothing is being asked for
        ignored = []
        for tag in tags:
            kinds = graph_kinds(document, tag)
            if kinds is None or not kinds:
                continue                                        # the unestablished check reports it
            if not (kinds & takers):
                ignored.append(tag)
        if not ignored or len(ignored) < len([t for t in tags if graph_kinds(document, t)]):
            continue                                            # at least one graph does consume it
        shown = ', '.join(str(value) for value in values[:4])
        findings.append(_finding(
            'applicability.inert_parameter', f'methods.method_params.{method}.{name}', ATTENTION,
            f'whether the graphs of {method} use its {name}',
            f'{name} is set to {shown}, and none of {", ".join(ignored)} contains a layer that '
            f'reads it (only {", ".join(sorted(takers))} do)',
            f'the value is stored and legal but changes nothing: {method} runs the same graph '
            f'whichever is chosen'
            + (f', so a search over {len(values)} values explores a dimension that does not exist'
               if len(values) > 1 else ''),
            f'Either choose a graph built from {" or ".join(sorted(takers))}, or clear '
            f'{name} so the case does not appear to offer a choice it cannot make. '
            'Nothing is removed for you.'))
    return findings


def _modality_findings(document, method, tags) -> list[dict[str, Any]]:
    """A graph that reads a kind of input the case does not configure."""
    from eforecast.config.graph_service import data_shapes
    try:
        shapes = data_shapes(document)
    except Exception as error:                                  # noqa: BLE001
        return [_finding('applicability.modalities', f'methods.method_params.{method}', ATTENTION,
                         f'the inputs the graphs of {method} need',
                         f'the input shapes could not be resolved ({type(error).__name__})',
                         'whether a branch has data cannot be established here',
                         'Ask the maintainer to review this case\'s inputs.')]
    available = set((shapes.get('shapes') or {}))
    notes = {note.split(':', 1)[0]: note for note in (shapes.get('notes') or []) if ':' in note}
    findings = []
    for tag in tags:
        experiment = (_at(document, 'experiments') or {}).get(tag)
        if not isinstance(experiment, dict):
            continue
        for branch in sorted(experiment):
            if branch == 'output' or branch in available:
                continue
            root = branch.split('/')[-1]
            if root in available:
                continue
            findings.append(_finding(
                'applicability.missing_modality', f'experiments.{tag}', BLOCKING,
                f'whether {tag} has the inputs its branches need',
                f'branch {branch!r} has no input in this case (available: '
                f'{", ".join(sorted(available)) or "none"})',
                f'{method} would build a graph over data that is not there, so the run fails when '
                'that branch is first used',
                f'Configure the {root} input for this case, or choose a graph whose branches match '
                'the inputs it has.'))
    for name, note in sorted(notes.items()):
        if 'placeholder' not in note:
            continue
        if any(name in (graph_kinds(document, tag) or set()) for tag in tags):
            continue
        used = [tag for tag in tags if isinstance((_at(document, 'experiments') or {}).get(tag), dict)
                and any(branch.split('/')[-1] == name
                        for branch in (_at(document, 'experiments') or {})[tag])]
        if used:
            findings.append(_finding(
                'applicability.placeholder_input', f'experiments.{", ".join(used)}', ATTENTION,
                f'whether the {name} input of {method} is real',
                note,
                f'the graph builds and the shapes check out, but {name} carries no measurement, so '
                'what it learns from that branch is nothing',
                f'Configure a {name} variable for this case, or use a graph without a {name} branch.'))
    return findings


def _unknown_parameter_findings(document, method, tags) -> list[dict[str, Any]]:
    """Layer parameters the catalogue does not know, surfaced in the same shape as the rest."""
    from eforecast.config.graph_service import unknown_parameters
    findings = []
    for tag in tags:
        experiment = (_at(document, 'experiments') or {}).get(tag)
        if not isinstance(experiment, dict):
            continue
        try:
            warnings = unknown_parameters(experiment)
        except Exception:                                       # noqa: BLE001
            continue
        for warning in warnings:
            findings.append(_finding(
                'applicability.unknown_parameter', f'experiments.{tag}', ATTENTION,
                f'the layer parameters of {tag} against the catalogue', warning,
                'the builder passes anything it is given, so a misspelt name is accepted and then '
                'does nothing',
                'Correct the name, or tell the maintainer if the catalogue is behind the code.'))
    return findings


def _value_findings(document, method, tags) -> list[dict[str, Any]]:
    """Layer parameters that hold a value of another kind than the layer takes, or name a branch the graph lacks.

    The owner's three cases: the ``cross_attention`` of ``mlp1`` carried ``attention_query: 64``,
    the value the graph editor started that parameter at. It names no branch.
    """
    from eforecast.config.graph_service import mistyped_parameters, normalize_experiments
    findings = []
    for tag in tags:
        experiment = (_at(document, 'experiments') or {}).get(tag)
        if not isinstance(experiment, dict):
            continue
        try:
            entries = mistyped_parameters(normalize_experiments({tag: experiment})[tag])
        except Exception:                                       # noqa: BLE001 - an unreadable graph is reported elsewhere
            continue
        for entry in entries:
            if entry.get('strangers'):
                evidence = (f'{entry["where"]}: {entry["name"]} names {", ".join(map(str, entry["strangers"]))}, '
                            'and the graph has no such branch')
            else:
                evidence = (f'{entry["where"]}: {entry["name"]} is {entry["value"]!r}, and the layer takes '
                            f'{entry["takes"]} there')
            findings.append(_finding(
                'applicability.parameter_value', f'experiments.{tag}', ATTENTION,
                f'the values of the layer parameters of {tag}', evidence, entry['effect'], entry['remedy']))
    return findings


def _unestablished_findings(document, method, tags) -> list[dict[str, Any]]:
    """When applicability cannot be decided, say so rather than imply it is fine."""
    from eforecast.config.layer_catalogue import LAYERS
    findings = []
    if not tags:
        return [_finding(
            'applicability.unestablished', f'methods.method_params.{method}.{TAG_PARAMETER}',
            ATTENTION, f'which graph {method} would run',
            'no experiment is named for this method',
            'what this method would actually execute cannot be established, so none of its '
            'parameters can be checked for effect',
            f'Choose an experiment for {method}, or ask the maintainer to review it.')]
    for tag in tags:
        kinds = graph_kinds(document, tag)
        if kinds is None:
            findings.append(_finding(
                'applicability.unestablished', f'experiments.{tag}', ATTENTION,
                f'the graph {tag} named by {method}',
                f'{tag!r} is not an experiment of this case',
                'the method names a graph that does not exist, so what it would run is unknown',
                f'Name an experiment this case defines, or add {tag}.'))
            continue
        strangers = sorted(kinds - set(LAYERS))
        if strangers:
            findings.append(_finding(
                'applicability.unestablished', f'experiments.{tag}', ATTENTION,
                f'the layer kinds of {tag} against the catalogue',
                f'{", ".join(strangers)} are not in the catalogue',
                'whether the parameters of this method reach those layers cannot be established, '
                'so an ignored setting here would not be reported',
                'Ask the maintainer to review these layers; the catalogue may be behind the code.'))
    return findings


def _selector_findings(document, method) -> list[dict[str, Any]]:
    """A feature selection a family searches and the case never computes.

    The features stage builds the selectors ``methods.feature_selection_methods`` lists, and
    nothing else. A family's own ``feature_selection_method`` is a copy of that list taken when
    the family was added, and it does not follow the list when the list is edited. So a case
    can ask for a selector nobody builds: training is then refused before it starts, or -- when
    some other selector exists -- a trial fails the first time it draws the missing one.
    """
    asked = [value for value in _values(_at(document, f'methods.method_params.{method}.feature_selection_method'))
             if value not in (None, '', False)]
    computed = [value for value in (_at(document, 'methods.feature_selection_methods') or []) if value is not None]
    missing = [str(value) for value in asked if value not in computed]
    if not missing:
        return []
    return [_finding(
        'applicability.selector_not_computed', f'methods.method_params.{method}.feature_selection_method', BLOCKING,
        f'whether the feature selection {method} asks for is computed',
        f'{method} asks for {", ".join(missing)}; the case computes '
        + (", ".join(map(str, computed)) if computed else 'none (methods.feature_selection_methods names no method)'),
        'training is refused before it starts, because it requires a selector that is never built'
        if not computed else f'a trial of {method} that draws {" or ".join(missing)} fails when it loads its inputs',
        f'Add {", ".join(missing)} to the feature selection methods of the case and run the features stage, '
        f'or set the feature selection of {method} to none.')]


def _row_form_findings(document, method, tags) -> list[dict[str, Any]]:
    """A graph with a row input that holds the weather, in a family that keeps the weather as a grid.

    ``row_all`` and ``row_nwp`` join the weather to the other row inputs as columns, which is what
    a ``compress`` option produces. A family with ``compress`` none is handed the grid instead,
    and a trial that draws such a graph fails as it loads its inputs. It is legal and stored:
    the owner's ``Total_wind`` gave its TIMM family the graphs ``exp_fuzzy1`` and ``afno1``.
    """
    if not (_at(document, 'project.nwp_models') or []):
        return []
    if [value for value in _values(_at(document, f'methods.method_params.{method}.compress')) if value not in (None, '')]:
        return []
    needing = []
    for tag in tags:
        experiment = (_at(document, 'experiments') or {}).get(tag)
        if not isinstance(experiment, dict):
            continue
        for branch in experiment:
            parts = branch.split('/')[0].split('_')
            if parts[0] == 'row' and {'all', 'nwp'} & set(parts[1:]) and tag not in needing:
                needing.append(tag)
    if not needing:
        return []
    others = [tag for tag in tags if tag not in needing]
    return [_finding(
        'applicability.row_input_without_compress', f'methods.method_params.{method}.compress', ATTENTION,
        f'whether the graphs of {method} get the weather in the form their inputs need',
        f'{", ".join(needing)} read{"s" if len(needing) == 1 else ""} the weather as row columns, and {method} '
        'has compress set to none, which hands it the grid',
        f'every trial of {method} that draws {" or ".join(needing)} fails when it loads its inputs'
        + (f'; only {", ".join(others)} can train' if others else f', so {method} trains nothing'),
        f'Give {method} a compress option, or keep only graphs without a row_all or row_nwp input for it.')]


def _sequence_findings(document) -> list[dict[str, Any]]:
    """A method that reads sequences, in a case that has none.

    The rows of a ``long_term`` case are indexed by (prediction time, issue time), and it is
    given no sequence dataset. A foundation model's context and an ``lstm`` branch are read
    from that dataset.
    """
    from eforecast.config.capabilities import FOUNDATION_METHODS, has_sequences
    if has_sequences({'horizon_type': _at(document, 'project.horizon_type')}):
        return []
    needing = [name for name in enabled_methods(document) if name in FOUNDATION_METHODS]
    for method in enabled_methods(document):
        for tag in tags_of(document, method):
            experiment = (_at(document, 'experiments') or {}).get(tag)
            if isinstance(experiment, dict) and any('lstm' in branch for branch in experiment) and method not in needing:
                needing.append(method)
    return [_finding(
        'applicability.no_sequences', 'project.horizon_type', ATTENTION,
        f'whether {method} has the sequence inputs it reads',
        f'{method} is switched on, and a long_term case has no sequence dataset',
        f'a run that includes {method} is refused before it starts; the other methods are not affected '
        'as long as it is left out of the run',
        f'Switch {method} off for this case, or use a short_term case for it.') for method in needing]


def findings(document) -> list[dict[str, Any]]:
    """Every applicability finding for the methods this case would run."""
    out: list[dict[str, Any]] = []
    families = graph_families()
    methods = [name for name in enabled_methods(document) if name in families]
    for method in methods:
        if not isinstance(_at(document, f'methods.method_params.{method}'), dict):
            continue
        tags = tags_of(document, method)
        out.extend(_unestablished_findings(document, method, tags))
        out.extend(_inert_findings(document, method, tags))
        out.extend(_modality_findings(document, method, tags))
        out.extend(_unknown_parameter_findings(document, method, tags))
        out.extend(finding for finding in _value_findings(document, method, tags)
                   if finding not in out)                       # a graph two methods share is reported once
        out.extend(_selector_findings(document, method))
        out.extend(_row_form_findings(document, method, tags))
    out.extend(_sequence_findings(document))
    if methods and not out:
        out.append(_finding('applicability', 'methods.method_params', OK,
                            'whether every selected parameter reaches the graph that runs',
                            f'{len(methods)} graph method(s) checked, nothing ignored',
                            'what the case offers is what it runs', 'Nothing to do.'))
    return out
