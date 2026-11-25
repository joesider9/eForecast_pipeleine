# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Turn a project configuration into the legacy ``static_data`` dictionary.

:func:`initializer` accepts three shapes of configuration:

* a ``static_data`` dict built by a legacy ``configuration/config.py``
* a path to a YAML case: a single ``<case>.yaml``, a parts directory
  ``<case>/`` or its ``main.yaml`` (see ``eforecast.config.config_files``)
* an ``eforecast.config.ExperimentConfig`` instance

The work is split in three functions so each can be run and inspected alone:

* :func:`load_static_data` - configuration -> flat dict, no side effects
* :func:`create_paths` - adds the folder tree ``path_owner`` ... ``path_data``
  and ``_id``; creates the folders unless ``dry_run`` is set
* :func:`create_nwp_areas` - adds ``nwp['area']`` to every NWP model and
  replaces ``coord`` by the site centre used to build the areas
"""
import logging
import os
from pathlib import Path

import numpy as np

from eforecast.common_utils.nwp_utils import create_area

log = logging.getLogger(__name__)

PATH_KEYS = ('path_owner', 'path_group', 'path_group_type', 'path_group_nwp',
             'path_project', 'path_model', 'path_logs', 'path_data')


def load_static_data(config):
    """Return the flat ``static_data`` dict for ``config`` without touching the disk."""
    if isinstance(config, dict):
        return config
    if isinstance(config, (str, os.PathLike)):
        from eforecast.config.run_inputs import load as load_run_inputs
        pinned = load_run_inputs(config)
        if pinned is not None:
            return pinned
        from eforecast.config.config_files import config_paths, load_document
        from eforecast.config.loader import to_static_data
        from eforecast.config.schema import ExperimentConfig

        path = Path(config)
        raw = load_document(path)
        if not raw:
            raise FileNotFoundError(f'No configuration found at {path}')
        cfg = ExperimentConfig.model_validate(raw)
        # ``experiments_module`` is resolved relative to the YAML location, so
        # anchor a parts directory at its ``main.yaml`` even if it does not exist.
        anchor = path if path.is_file() else config_paths(path)[0]
        object.__setattr__(cfg, '_yaml_path', str(anchor.resolve()))
        return to_static_data(cfg)
    from eforecast.config.loader import to_static_data
    from eforecast.config.schema import ExperimentConfig

    if isinstance(config, ExperimentConfig):
        return to_static_data(config)
    raise TypeError(f'config should be a dict, a YAML path or an ExperimentConfig, got {type(config)}')


def _project_path(path_group_type, project_name, horizon_type):
    """``<group type>/<project>/<horizon type>``, falling back to the folder of the former name.

    The horizon type is a folder in the model store, so renaming ``multi-output`` to ``short_term``
    (and the other three) would otherwise orphan every model trained before the rename.  When the
    new folder does not exist and the old one does, the old one is used and a warning says how to
    migrate.  A case that has never been trained simply gets the new name.
    """
    from eforecast.config.schema import LEGACY_HORIZON_TYPES, HorizonType

    wanted = os.path.join(path_group_type, project_name, horizon_type)
    if os.path.isdir(wanted):
        return wanted
    try:
        member = HorizonType(horizon_type).name
    except ValueError:
        return wanted                                   # not one of ours: nothing to fall back to
    for old in [name for name, became in LEGACY_HORIZON_TYPES.items() if became == member]:
        candidate = os.path.join(path_group_type, project_name, old)
        if os.path.isdir(candidate):
            log.warning('%s: using the model folder of the former horizon type %r; rename it with '
                        '`python -m eforecast.config.migrate_horizon_type %s` to use %r',
                        project_name, old, path_group_type, horizon_type)
            return candidate
    return wanted


def create_paths(static_data, dry_run=False):
    """Add the project folder tree and ``_id`` to ``static_data``.

    Folders are created unless ``dry_run`` is True.
    """
    static_data['path_owner'] = os.path.join(static_data['sys_folder'], static_data['project_owner'])
    static_data['path_group'] = os.path.join(static_data['path_owner'],
                                             f"{static_data['projects_group']}_ver{static_data['version_group']}")
    static_data['path_group_type'] = os.path.join(static_data['path_group'], static_data['type'])
    # One folder for each set of weather variables the group's cases ask for; see archive.py.
    from eforecast.nwp_extraction import archive
    weather = [item['name'] for item in static_data.get('variables') or []
               if isinstance(item, dict) and item.get('type') == 'nwp' and item.get('name')]
    static_data['path_group_nwp'] = archive.folder(os.path.join(static_data['path_group'], 'nwp'), weather,
                                                   record=not dry_run)
    static_data['path_project'] = _project_path(static_data['path_group_type'],
                                                static_data['project_name'],
                                                static_data['horizon_type'])
    static_data['path_model'] = os.path.join(static_data['path_project'],
                                             f"model_ver{static_data['version_model']}")
    static_data['path_logs'] = os.path.join(static_data['path_project'], 'logging')
    static_data['path_data'] = os.path.join(static_data['path_model'], 'DATA')
    static_data['_id'] = static_data['project_name']
    if not dry_run:
        for key in PATH_KEYS:
            os.makedirs(static_data[key], exist_ok=True)
    return static_data


def _inside_group(area, area_group):
    return not ((area[0][0] < area_group[0][0]) or (area[0][1] < area_group[0][1]) or
                (area[1][0] > area_group[1][0]) or (area[1][1] > area_group[1][1]))


def _shift_area_into_group(area, area_group, resolution, levels):
    """Move a point-site area that sticks out of ``area_group`` back inside it.

    The area keeps (about) its size ``(2 * levels + 0.5) * resolution``; the
    side that crosses the group border is pinned on the border and the
    opposite side is pushed inwards by the same amount.  ``area`` is modified
    in place and returned.
    """
    dist_ = (2 * levels + 0.5) * resolution
    area_diff = np.round([(np.ceil(area[0][0] * 10) / 10 - area_group[0][0]),
                          (np.ceil(area[0][1] * 10) / 10 - area_group[0][1]),
                          (area_group[1][0] - np.ceil(area[1][0] * 10) / 10),
                          (area_group[1][1] - np.ceil(area[1][1] * 10) / 10)], 1)
    if area_diff[0] < 0:
        area[0][0] = area_group[0][0]
        area[1][0] += np.abs(area_diff[0])
        if np.abs(area[0][0] - area[1][0]) > dist_:
            area[1][0] -= np.abs(area[0][0] - area[1][0]) - dist_
    if area_diff[1] < 0:
        area[0][1] = area_group[0][1]
        area[1][1] += np.abs(area_diff[1])
        if np.abs(area[0][1] - area[1][1]) > dist_:
            area[1][1] -= np.abs(area[0][1] - area[1][1]) - dist_
    if area_diff[2] < 0:
        area[0][0] -= np.abs(area_diff[2])
        area[1][0] = area_group[1][0]
        if np.abs(area[0][0] - area[1][0]) > dist_:
            area[0][0] += np.abs(area[0][0] - area[1][0]) - dist_
    if area_diff[3] < 0:
        area[0][1] -= np.abs(area_diff[3])
        area[1][1] = area_group[1][1]
        if np.abs(area[0][1] - area[1][1]) > dist_:
            area[0][1] += np.abs(area[0][1] - area[1][1]) - dist_
    return area


def create_nwp_areas(static_data):
    """Add ``nwp['area']`` for every NWP model and set ``coord`` to the site centre.

    Safe to run on its own result. The areas are made from the coordinates the case declares,
    and ``coord`` is then replaced by their centre. Several entry points initialise twice -- a
    dry run for the plan, then the real one -- and the second pass used to take that centre
    for a point site: a regional case (``[lat1, long1, lat2, long2]``, or named areas) was given
    the small box of a single park, its stages all read as "configuration changed", and
    training was refused. For the same reason a second provider of such a case got the box
    instead of the region; every provider is now given the declared coordinates.
    """
    if static_data['NWP'] is None:
        return static_data
    area_group = static_data['area_group']
    declared, centre = static_data['coord'], None
    for nwp in static_data['NWP']:
        if nwp is None:
            continue
        if nwp.get('area') is not None:
            continue                # made on an earlier pass, from the coordinates the case declared
        area, centre = create_area(declared, nwp['resolution'], nwp['levels'])
        if isinstance(area, dict):
            for key, value in area.items():
                if not _inside_group(value, area_group):
                    raise ValueError(f'Area {key}  is smaller than static_data area group')
        elif not _inside_group(area, area_group):
            log.warning('NWP %s area %s crosses area_group %s, shifting it inside',
                        nwp['model'], np.asarray(area).tolist(), area_group)
            area = _shift_area_into_group(area, area_group, nwp['resolution'], nwp['levels'])
        nwp['area'] = area
    if centre is not None:
        static_data['coord'] = centre
    return static_data


def initializer(config, dry_run=False):
    """Configure ``static_data`` paths and NWP areas.

    ``config`` is a legacy dict, a YAML case path or an ``ExperimentConfig``.
    ``online`` and ``read_data`` are kept for the callers that pass them; they
    do not change the result.  ``dry_run`` resolves everything without
    creating folders.
    """
    static_data = load_static_data(config)
    create_paths(static_data, dry_run=dry_run)
    create_nwp_areas(static_data)
    log.info('Static data of project %s created (path_data=%s)', static_data['_id'], static_data['path_data'])
    return static_data
