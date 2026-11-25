# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""CLI definitions shared by entry points and API submission without ML imports.

Keep parsing here so a first Run click need not import dataset/training implementations.
Entry points supply their full help description; parsing never loads a case or starts work.
"""
import argparse
import os

from .capabilities import CPU_METHODS, CLUSTER_METHODS, GLOBAL_METHODS
from .pipeline_stages import (ALL_DATASET_STAGES, DATASET_STAGE_GROUPS, DATASET_DEFAULT_STAGES,
                              COMBINE_STAGES, COMBINE_DEFAULT_STAGES, positive_int, nonnegative_int)


def cache_arguments(parser):
    parser.add_argument('--model-cache-mode', choices=('prefer-local', 'online', 'offline', 'prewarmed'), default=None,
                        help='pretrained models: prefer-local and online download one that is not on this machine, '
                             'offline and prewarmed never do (default: the case, then EFORECAST_MODEL_CACHE_MODE)')


def parse_datasets_args(argv=None, *, description=None):
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    cache_arguments(parser)
    parser.add_argument('--config', default=os.environ.get('EFORECAST_CASE', 'configuration/cases/default.yaml'),
                        help='YAML case (file, parts directory or main.yaml) or legacy project package name')
    parser.add_argument('--stage', nargs='+', default=list(DATASET_DEFAULT_STAGES),
                        choices=list(ALL_DATASET_STAGES) + list(DATASET_STAGE_GROUPS), metavar='STAGE',
                        help=f"one or more of {', '.join(list(ALL_DATASET_STAGES) + list(DATASET_STAGE_GROUPS))} (default: datasets)")
    parser.add_argument('--recreate', action='store_true',
                        help='delete the outputs of the selected dataset stages before running them')
    parser.add_argument('--recreate-extraction', action='store_true',
                        help='extract stage: rewrite the daily NWP pickles of this project\'s days even if they '
                             'exist (the pickles are shared by every project of the group)')
    parser.add_argument('--serial', action='store_true', help='no process pools; n_jobs=1 (for debugging)')
    parser.add_argument('--dry-run', action='store_true',
                        help='resolve the configuration, print paths, stage plan and dataset status, then exit')
    parser.add_argument('--log-level', default='INFO')
    return parser.parse_args(argv)


def parse_training_args(argv=None, *, description=None, default_config=None):
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    cache_arguments(parser)
    parser.add_argument('--config', default=default_config if default_config is not None else os.environ.get('EFORECAST_CASE', 'configuration/cases/default.yaml'),
                        help='YAML case (file, parts directory or main.yaml); default: $EFORECAST_CASE or configuration/cases/default.yaml')
    parser.add_argument('--refit', action='store_true', default=False,help='delete the studies and trial folders and search again')
    parser.add_argument('--gpus', nargs='+', type=nonnegative_int, default=None, help='GPU ids (default: all n_gpus of the case)')
    parser.add_argument('--n-jobs', type=positive_int, default=None, help='max concurrent trials per GPU (default: n_jobs of the methods)')
    parser.add_argument('--global-method', default=None, choices=[m for m in GLOBAL_METHODS if m != 'GRAPH' and m not in CPU_METHODS], help='train only this global method')
    parser.add_argument('--cluster-method', default=None, choices=CLUSTER_METHODS, help='train only this cluster method')
    parser.add_argument('--docker', action='store_true', help='run every trial from the eforecast-worker Docker image')
    parser.add_argument('--image', default='eforecast-worker:latest', help='image for --docker')
    parser.add_argument('--storage', default=None, help='Optuna storage URL (default: EFORECAST_OPTUNA_STORAGE or journal files)')
    parser.add_argument('--halve-batch', action='store_true', help='halve the batch size on the second OOM retry')
    parser.add_argument('--metrics-port', type=int, default=None, help='Prometheus endpoint port (0 = off, default 9108)')
    parser.add_argument('--prepare-only', action='store_true',
                        help='ask one trial per study, write its parameters.pickle and print the trial_runner command; '
                             'the next run adopts the trials you finished by hand')
    parser.add_argument('--cpu', action='store_true', help='train the shallow CPU models (methods_cpu) of every cluster instead of the deep models')
    parser.add_argument('--cpu-scope', choices=('global', 'clusters', 'both'), default=None,
                        help='with --cpu: override methods.cpu_scope (legacy default: clusters)')
    for name, help_text in (
            ('cpu-workers', 'CPU studies run side by side (default: jobs.cpu_workers; when the case names none, as many as it has studies, up to one a CPU)'),
            ('cpu-study-threads', 'the most threads a study uses (default: jobs.n_cpus divided among the studies running)')):
        parser.add_argument('--' + name, type=positive_int, default=None, help='with --cpu: ' + help_text)
    parser.add_argument('--cpu-study-device', default=None, help='with --cpu: foundation device preference; inherits the family by default')
    parser.add_argument('--cpu-method' , default=None, choices=CPU_METHODS, help='with --cpu: only this method')
    parser.add_argument('--log-level', default='INFO')
    parser.add_argument('--dry-run', action='store_true', help='preview stage dependencies and prerequisites without training or writing')
    return parser.parse_args(argv)


def parse_combine_args(argv=None, *, description=None):
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    cache_arguments(parser)
    parser.add_argument('--config', default=os.environ.get('EFORECAST_CASE', 'configuration/cases/default.yaml'),
                        help='YAML case (file, parts directory or main.yaml)')
    parser.add_argument('--stage', nargs='+', default=list(COMBINE_DEFAULT_STAGES), choices=list(COMBINE_STAGES) + ['all'], metavar='STAGE',
                        help=f"one or more of {', '.join(COMBINE_STAGES)}, or all (default)")
    parser.add_argument('--refit', action='store_true', help='recompute the cached predictions and refit the combiners')
    parser.add_argument('--log-level', default='INFO')
    parser.add_argument('--dry-run', action='store_true', help='preview stage dependencies and prerequisites without predicting or writing')
    return parser.parse_args(argv)


PARSERS = {'datasets': parse_datasets_args, 'training': parse_training_args, 'combine': parse_combine_args}
