# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Keep the data options of the method families in step with the data section.

In the legacy ``config_methods.py`` the per-family keys ``merge``, ``compress``,
``scale_row_method``, ``scale_nwp_method``, ``scale_img_method``,
``image_data_type`` and ``feature_selection_method`` were expressions over the
data lists (``set(NWP_DATA_MERGE)``, ``NWP_DATA_MERGE[0]``, ...).  In the YAML
they are literal values, so :func:`sync_families_with_data` refreshes them
from the current ``data`` section and ``methods.feature_selection_methods``,
keeping the *form* each family uses:

* a list or ``{choices: [...]}`` -> the full current option list
* a scalar equal to one of the options -> the first current option
* ``null`` -> left as ``null``
* a scalar that is not one of the options -> left alone and reported

``data_type`` mappings (``base_data_types``, ``clustering.data_type``,
``<family>.data_type``) follow the same rule key by key; the forced legacy
derivation (first option everywhere) is the editor's "Derive from the data
section" button.
"""
from typing import Any

FAMILY_KEYS: dict[str, tuple[str, str]] = {
    # family key: (section, key holding the option list)
    "merge": ("data", "nwp_data_merge"),
    "compress": ("data", "compress_data"),
    "scale_row_method": ("data", "scale_row_method"),
    "scale_nwp_method": ("data", "scale_nwp_method"),
    "scale_img_method": ("data", "scale_img_method"),
    "image_data_type": ("data", "img_data"),
    "feature_selection_method": ("methods", "feature_selection_methods"),
}
DATA_TYPE_KEYS: dict[str, str] = {
    "scale_row_method": "scale_row_method",
    "scale_nwp_method": "scale_nwp_method",
    "scale_img_method": "scale_img_method",
    "merge": "nwp_data_merge",
    "compress": "compress_data",
    "image_data_type": "img_data",
}


def _options(document: dict[str, Any], section: str, key: str) -> list[Any]:
    value = (document.get(section) or {}).get(key)
    if isinstance(value, (list, tuple)):
        return list(value) if len(value) > 0 else [None]
    return [None] if value is None else [value]


def current_options(document: dict[str, Any]) -> dict[str, list[Any]]:
    """``{family key: option list}`` from the data section and the feature selection methods."""
    return {family_key: _options(document, section, key) for family_key, (section, key) in FAMILY_KEYS.items()}


def data_type_from_options(options: dict[str, list[Any]]) -> dict[str, Any]:
    return {key: options[key][0] for key in DATA_TYPE_KEYS}


def sync_value(value: Any, options: list[Any]) -> tuple[Any, str]:
    """``(new value, outcome)`` with outcome in {'kept', 'changed', 'skipped'}."""
    if isinstance(value, dict) and set(value) == {"choices"}:
        new = {"choices": list(options)}
        return new, ("kept" if list(value["choices"]) == list(options) else "changed")
    if isinstance(value, (list, tuple)):
        return list(options), ("kept" if list(value) == list(options) else "changed")
    if value is None:
        return None, "kept"
    if value == options[0]:
        return value, "kept"
    if value in options:
        return options[0], "changed"
    return value, "skipped"


def sync_families_with_data(document: dict[str, Any]) -> tuple[dict[str, Any], list[dict], list[dict]]:
    """Refresh the data options of every family; returns ``(document, changes, skipped)``.

    ``document`` is modified in place.  Each change / skip is a dict with
    ``where`` (family or block), ``key``, ``old`` and ``new``.
    """
    changes: list[dict] = []
    skipped: list[dict] = []
    methods = document.get("methods")
    if not isinstance(methods, dict):
        return document, changes, skipped
    options = current_options(document)

    def sync_mapping(mapping: dict[str, Any], where: str) -> None:
        for key, option_list in options.items():
            if key not in mapping:
                continue
            new, outcome = sync_value(mapping[key], option_list)
            if outcome == "changed":
                changes.append({"where": where, "key": key, "old": mapping[key], "new": new})
                mapping[key] = new
            elif outcome == "skipped":
                skipped.append({"where": where, "key": key, "old": mapping[key], "new": option_list})

    def sync_data_type(mapping: dict[str, Any], where: str) -> None:
        # same form-preserving rule as the families: null stays null, a scalar among the
        # options follows the first option, anything else is left and reported.  The forced
        # legacy derivation (first option everywhere) stays behind the editor's "Derive" button.
        for key in DATA_TYPE_KEYS:
            if key not in mapping:
                continue
            new, outcome = sync_value(mapping[key], options[key])
            if outcome == "changed":
                changes.append({"where": where, "key": key, "old": mapping[key], "new": new})
                mapping[key] = new
            elif outcome == "skipped":
                skipped.append({"where": where, "key": key, "old": mapping[key], "new": options[key]})

    if isinstance(methods.get("base_data_types"), dict):
        sync_data_type(methods["base_data_types"], "base_data_types")
    for family, params in (methods.get("method_params") or {}).items():
        if not isinstance(params, dict):
            continue
        sync_mapping(params, family)
        # a case written before the families were flattened may still carry nested blocks
        if isinstance(params.get("data_type"), dict):
            sync_data_type(params["data_type"], f"{family}.data_type")
        if isinstance(params.get("params"), dict):
            sync_mapping(params["params"], f"{family}.params")
    return document, changes, skipped


def ensure_clustering_family(document: dict[str, Any]) -> tuple[dict[str, Any], list[dict]]:
    """Add the ``clustering`` family when the project is fuzzy and does not have one.

    ``is_fuzzy`` means the case is trained per cluster: ``run_training`` asks ``ClusterOrganizer``
    for the clusters before anything else, and that reads ``methods.method_params.clustering``.
    Without the family the run fails deep inside the clusterer, so the family is part of what being
    fuzzy means rather than something to remember separately.

    Modifies the document in place; returns it with the list of changes.
    """
    changes: list[dict] = []
    project = document.get("project") or {}
    methods = document.get("methods")
    if not isinstance(methods, dict) or not project.get("is_fuzzy"):
        return document, changes
    families = methods.get("method_params")
    if not isinstance(families, dict):
        families = {}
        methods["method_params"] = families
    if "clustering" in families:
        return document, changes
    from eforecast.config.family_presets import family_preset       # imports this module

    families["clustering"] = family_preset("clustering", document)
    changes.append({"where": "method_params", "key": "clustering",
                    "old": None, "new": "added from the presets (the project is fuzzy)"})
    return document, changes


def describe_sync(changes: list[dict], skipped: list[dict]) -> str:
    """One-paragraph, human readable summary of a sync."""
    if not changes and not skipped:
        return "Data options of the method families already match the data section."
    parts = []
    if changes:
        items = ", ".join(f"`{item['where']}.{item['key']}`" for item in changes)
        parts.append(f"Updated {len(changes)} data option(s) in the method families from the data section: {items}.")
    if skipped:
        items = ", ".join(f"`{item['where']}.{item['key']}` = `{item['old']}`" for item in skipped)
        parts.append(f"Left {len(skipped)} hand-set value(s) that are not among the current options: {items}.")
    return " ".join(parts)


def apply_base_data_types(document: dict[str, Any]) -> tuple[dict[str, Any], list[dict]]:
    """Push ``methods.base_data_types`` to every method family as fixed values.

    Only the keys a family already has are written (``merge``, ``compress``, ``scale_*``,
    ``image_data_type``), and they are written where the editor can reach them: on the family
    itself.  Nothing nested is created -- a dictionary inside a family cannot be edited in the
    parameter table, which is why ``clustering`` and ``Distributed`` are flat like the rest.  The
    shape the clusterer needs at runtime (``clustering.data_type`` and ``clustering.params``) is
    derived by :func:`eforecast.config.loader._clustering_runtime`, not stored.

    A case written before the flattening may still hold those blocks; they are kept in step too.
    The combiners read ``base_data_types`` directly.  Modifies the document in place; returns it
    with the list of changes."""
    changes: list[dict] = []
    methods = document.get("methods")
    if not isinstance(methods, dict) or not isinstance(methods.get("base_data_types"), dict):
        return document, changes
    base = {key: methods["base_data_types"].get(key) for key in DATA_TYPE_KEYS}

    def put(mapping: dict[str, Any], key: str, value: Any, where: str) -> None:
        if mapping.get(key, "<missing>") != value:
            changes.append({"where": where, "key": key, "old": mapping.get(key), "new": value})
            mapping[key] = value

    for family, params in (methods.get("method_params") or {}).items():
        if not isinstance(params, dict):
            continue
        targets = [(params, family)]
        for nested in ("data_type", "params"):                  # only when a legacy case has them
            if isinstance(params.get(nested), dict):
                targets.append((params[nested], f"{family}.{nested}"))
        for mapping, where in targets:
            for key, value in base.items():
                if key in mapping:
                    put(mapping, key, value, where)
    return document, changes
