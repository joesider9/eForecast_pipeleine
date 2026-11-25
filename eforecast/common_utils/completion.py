# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""One answer to "is this artifact finished work?", for every reader that has to decide.

Discovery, the trial runner, recovery, transfer and budget accounting used to each have their own
rule -- a weights file here, a results CSV there -- so the same folder could be finished work for
one of them and not for another. They ask this module instead.
"""
import json
import logging
import os

from eforecast.common_utils import artifact

log = logging.getLogger(__name__)


def discovery_policy():
    """``manifest`` (the default) or ``legacy_weights`` to restore acceptance by weights alone."""
    return os.environ.get('EFORECAST_TRIAL_DISCOVERY', 'manifest')


def graph_export(path_weights):
    """The GRAPH exporter's own evidence, for folders it wrote before manifests."""
    import joblib
    try:
        marker = joblib.load(os.path.join(path_weights, 'net_weights.pickle'))
    except Exception as error:                                  # noqa: BLE001 -- any unreadable marker
        return False, f'legacy GRAPH marker cannot be read ({error})'
    if not isinstance(marker, dict) or marker.get('method') != 'GRAPH' or marker.get('trained') is not True:
        return False, 'legacy GRAPH marker does not record a trained GRAPH model'
    if not os.path.isfile(os.path.join(path_weights, 'predictions_eval.pickle')):
        return False, 'legacy GRAPH export is missing predictions_eval.pickle'
    return True, 'legacy GRAPH export: trained marker and predictions, without manifest verification'


def is_finished(path_weights, method=None, *, allow_staged=False):
    """``(usable, reason)``: may this trial folder be trained on, predicted with and selected from?

    A folder with a manifest is judged on it. Without one, the available legacy completion evidence
    is used -- a finished result file, or the GRAPH exporter's trained marker and prediction cache.
    That evidence cannot prove payload integrity, so it is reported as legacy and warned about.
    """
    if discovery_policy() == 'legacy_weights':
        payload = artifact.read(path_weights)
        if payload is not None and 'graph_publication' in payload:
            verdict, reason = artifact.inspect(path_weights, allow_staged=allow_staged)
            return verdict == 'verified', reason
        return True, 'discovery policy is legacy_weights'
    verdict, reason = artifact.inspect(path_weights, allow_staged=allow_staged)
    if verdict == 'verified':
        return True, reason
    if verdict == 'rejected':
        return False, reason
    result_file = os.path.join(path_weights, 'trial_result.json')
    if not os.path.isfile(result_file):
        if (method or os.path.basename(os.path.dirname(path_weights))) == 'GRAPH':
            return graph_export(path_weights)
        return False, 'no manifest and no trial_result.json: nothing says this attempt finished'
    try:
        with open(result_file, encoding='utf-8') as stream:
            result = json.load(stream)
        if not isinstance(result, dict):
            return False, 'trial_result.json must be a mapping'
    except (OSError, ValueError) as error:
        return False, f'trial_result.json cannot be read ({error})'
    state = result.get('state')
    if state not in artifact.USABLE_STATES:
        return False, f'trial_result.json says the attempt ended as {state!r}'
    return True, f'legacy trial, {state} by its result file'
