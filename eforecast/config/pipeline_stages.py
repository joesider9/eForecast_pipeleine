# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Stage names and defaults without importing dataset or training dependencies."""

DATASET_STAGES = ('extract', 'nwp', 'images', 'target', 'row', 'lstm')
MODEL_STAGES = ('split', 'preprocess', 'clustering', 'features')
ALL_DATASET_STAGES = DATASET_STAGES + MODEL_STAGES
DATASET_STAGE_GROUPS = {'datasets': DATASET_STAGES, 'all': ALL_DATASET_STAGES}
DATASET_DEFAULT_STAGES = ('datasets',)
COMBINE_STAGES = ('predict', 'select', 'combine', 'calibrate', 'evaluate')
COMBINE_DEFAULT_STAGES = ('all',)


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise ValueError('must be a positive integer')
    return value


def nonnegative_int(value):
    value = int(value)
    if value < 0:
        raise ValueError('must be a nonnegative integer')
    return value
