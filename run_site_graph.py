# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Train the cross-site graph model on several cases (plants as nodes) and, optionally, hand its
predictions to every site's ensemble as the global method ``GRAPH``.

    python run_site_graph.py --configs configuration/cases/a.yaml configuration/cases/b.yaml ... --epochs 50
    python run_site_graph.py --configs ... --data-tag row_obs_calendar --k 5 --gpu 0 --export
    python run_site_graph.py --configs ... --nodes both --nwp-pool 2 --nwp-lag-step 4     # nodes carry their NWP grids too

Every site is read through its own datasets and scalers (``site_frames``), the sites are aligned on
their common dates and shared feature names (``assemble``), and one ``SiteGraphNet`` is trained on
the training periods (split by the first site's CV masks: mask 0 trains, mask 1 validates, mask 2 is
reported).  The model, its history and the attention between the sites go to
``<path_group>/site_graph/<name>/``.  With ``--export`` the predictions of the evaluation period are
written into ``<path_model>/global/GRAPH/test_0/`` of every site (``predictions_eval.pickle`` and a
marker ``net_weights.pickle``), where ``run_combine.py`` finds them once ``global_methods.GRAPH`` is
on for the case: the graph model becomes a candidate of the selection, the combiners and the
calibration like any trial, online too: the ensemble rebuilds the online datasets of the other sites
and runs the saved graph (``GraphTrial``).  ``--predict START END`` does the same from the command line:

    python run_site_graph.py --predict 2025-03-01 2025-03-02 --model <path_group>/site_graph/graph1 [--csv out]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
log = logging.getLogger('run_site_graph')


def publish_bundle(model_path):
    """Publish the site-graph bundle an export points at, and return how to recognise it again."""
    from eforecast.common_utils import artifact

    from eforecast.common_utils.graph_bundle import bundle_root
    model_path = bundle_root(model_path)
    existing = artifact.read(str(model_path))
    if existing is not None and existing.get('graph_publication') is not None:
        valid, reason = artifact.verify_digests(str(model_path))
        if not valid:
            raise ValueError(f'cannot reuse the committed graph bundle: {reason}')
        return {'path': str(model_path.resolve()),
                'files': {name: record['sha256'] for name, record in existing['files'].items()}}
    (model_path / artifact.MANIFEST_FILE).unlink(missing_ok=True)
    manifest = artifact.publish(str(model_path), kind='graph_bundle', state='complete',
                                **artifact.worker_identity(model_path, task='site_graph', attempt=0))
    if manifest is None:
        raise RuntimeError(f'the site graph bundle in {model_path} could not be published')
    return {'path': str(model_path.resolve()),
            'files': {name: record['sha256'] for name, record in manifest['files'].items()}}


def export_trial(folder, frame, model_path, site, sites, bundle=None, *, depends_on=None, identity=None):
    """Publish one site's GRAPH export after both of its local payloads are written.

    The export is a pointer to the model bundle, so the export records which bundle it was made
    from: a reader that follows the pointer can tell whether that is still the bundle on disk.
    """
    import joblib
    from eforecast.common_utils import artifact

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / artifact.MANIFEST_FILE).unlink(missing_ok=True)
    bundle = bundle or publish_bundle(model_path)
    frame = frame.copy()
    frame.columns = [f'GRAPH_{c}' for c in frame.columns]
    joblib.dump(frame, folder / 'predictions_eval.pickle')
    joblib.dump({'method': 'GRAPH', 'site_graph': bundle['path'],
                 'site': site, 'sites': sites, 'trained': True}, folder / 'net_weights.pickle')
    fields = dict(identity or artifact.worker_identity(folder, task='site_graph', attempt=0))
    if depends_on is not None:
        fields['depends_on'] = depends_on
    return artifact.publish(str(folder), kind='graph_export', state='complete', producer='GRAPH', bundle=bundle, **fields)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--configs', nargs='*', default=[], help='two or more case YAMLs (the sites); not needed with --predict')
    parser.add_argument('--name', default='graph1', help='name of the model folder under <path_group>/site_graph/')
    parser.add_argument('--data-tag', default='row_all', help='row features per site: row_all, row_obs_calendar, row_obs_nwp, ...')
    parser.add_argument('--nodes', default='row', choices=['row', 'nwp', 'both'],
                        help="what a node carries: its row features (default), its NWP grid through a shared convolutional encoder, or both")
    parser.add_argument('--nwp-pool', type=int, default=2, help='average pooling of the grid (17x17 -> 8x8 with 2)')
    parser.add_argument('--nwp-lag-step', type=int, default=1, help='keep every n-th NWP lag (4 for a 15-minute case: the NWP is hourly)')
    parser.add_argument('--merge', default='all', help='NWP merge of the row features (all | by_area | by_variable | by_area_variable)'); parser.add_argument('--compress', default='dense', help='NWP compression of the row features (dense | semi_full | full | load)')
    parser.add_argument('--scale-row', default='minmax', help='scaler of the row features (minmax | standard | maxabs)'); parser.add_argument('--scale-nwp', default='minmax', help='scaler of the NWP statistics')
    parser.add_argument('--hidden', type=int, default=64, help='features per node (rounded to a multiple of --heads)'); parser.add_argument('--depth', type=int, default=2, help='graph-attention layers')
    parser.add_argument('--heads', type=int, default=4, help='attention heads'); parser.add_argument('--dropout', type=float, default=0.1, help='dropout of the encoder, the attention and the feed-forward blocks')
    parser.add_argument('--k', type=int, default=None, help='nearest neighbours a site attends to (default all)')
    parser.add_argument('--length-scale', type=float, default=None, help='km of the distance prior (default median distance)')
    parser.add_argument('--epochs', type=int, default=100, help='maximum epochs (early stopping on the validation MAE)'); parser.add_argument('--batch-size', type=int, default=256, help='timestamps per batch (every site of a timestamp is in it)')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='AdamW learning rate'); parser.add_argument('--patience', type=int, default=10, help='epochs without a better validation MAE before stopping')
    parser.add_argument('--gpu', default='cpu', help='GPU id or cpu')
    parser.add_argument('--export', action='store_true', help='write the evaluation predictions into every site as global/GRAPH/test_0')
    parser.add_argument('--export-root', default=None, help='write the exports under this folder instead of the sites (testing)')
    parser.add_argument('--out', default=None, help='folder of the model (default <path_group of the first site>/site_graph/<name>)')
    parser.add_argument('--predict', nargs=2, default=None, metavar=('START', 'END'),
                        help='online mode: forecast every site of the model in --out (or --configs\' default folder) for these days')
    parser.add_argument('--model', default=None, help='with --predict: the folder of the trained site graph (default --out)')
    parser.add_argument('--csv', default=None, help='with --predict: write the forecasts (one file per site, <csv>_<site>.csv)')
    parser.add_argument('--threads', type=int, default=None, help='CPU threads of the graph worker (default: the n_cpus of the cases)')
    parser.add_argument('--recover-publication', default=None, metavar='MODEL', help='repair an interrupted publication without training')
    parser.add_argument('--log-level', default='INFO')
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level, format='%(asctime)s %(levelname)s %(name)s: %(message)s', datefmt='%H:%M:%S')
    from eforecast.config.site_graph_compute import cli
    return cli(args)


if __name__ == '__main__':
    sys.exit(main())
