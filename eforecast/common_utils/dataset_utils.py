# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

import logging
import os
import joblib
import shutil
import copy
import time

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from einops import rearrange


log = logging.getLogger(__name__)
from eforecast.datasets.tensor_views import TensorView
from eforecast.datasets.lazy_rows import Rows, take as take_rows
from eforecast.common_utils.resolutions import pandas_freq

# A12-03: torch and torchvision enter this module only through the image dataset, which a CPU
# RF/lasso/CatBoost study never touches; importing them here cost every such study about 2.2 s.
# They now load on first image use. These module attributes stay the substitution seam that
# existing callers and tests already rely on, so a replacement set before first use still wins.
ImageDataset = None
ImageDataloader = None


def _image_classes():
    global ImageDataset, ImageDataloader
    if ImageDataset is None or ImageDataloader is None:
        from eforecast.datasets.image_data.image_dataset import ImageDataset as dataset
        from eforecast.datasets.image_data.image_dataset import ImageDataloader as loader
        if ImageDataset is None:
            ImageDataset = dataset
        if ImageDataloader is None:
            ImageDataloader = loader
    return ImageDataset, ImageDataloader



def concat_pandas_df(df1, df2):
    dates = df1.index.intersection(df2.index)
    if len(dates) == 0:
        raise ValueError('Cannot sync datasets. there is no common dates')
    print(f'Merge pandas datasets with {dates.shape[0]} common dates ')
    return pd.concat([df1.loc[dates], df2.loc[dates]], axis=1)


def get_dates_from_dict(dict_df):
    dates = None
    for key, df in dict_df.items():
        if dates is None:
            dates = df.index
        else:
            dates = dates.intersection(df.index)
    return dates


def get_time_level(dates):
    if isinstance(dates, pd.MultiIndex):
        return pd.DatetimeIndex(dates.get_level_values(0))  # or by name 'valid_time'
    if isinstance(dates, pd.DatetimeIndex):
        return dates
    if isinstance(dates, pd.Index):  # your current "YYYY-mm-dd_HHTn"
        return pd.DatetimeIndex([pd.to_datetime(d.split('T')[0], format='%Y-%m-%d_%H') for d in dates])
    raise ValueError("Unsupported dates index type")


def intersection_index(idx1, idx2):
    if isinstance(idx1, pd.MultiIndex) and isinstance(idx2, pd.MultiIndex):
        common = idx1.intersection(idx2)
        t1 = idx1
        t2 = idx2
    elif isinstance(idx1, pd.DatetimeIndex) and isinstance(idx2, pd.DatetimeIndex):
        common = idx1.intersection(idx2)
        t1 = idx1
        t2 = idx2
    else:
        t1 = get_time_level(idx1)
        t2 = get_time_level(idx2)
        common = t1.intersection(t2)

    ind1 = np.where(t1.isin(common))[0]
    ind2 = np.where(t2.isin(common))[0]
    return common, idx1[ind1], idx2[ind2], ind1, ind2


def concat_df_dict(dict_df, df):
    dates = get_dates_from_dict(dict_df)
    dates = df.index.intersection(dates)
    dict_df_new = dict()
    for key, df_temp in dict_df.items():
        dict_df_new[key] = pd.concat([df.loc[dates], df_temp.loc[dates]], axis=1)
    print(f'Merge dictionary with pandas datasets ')
    return dict_df_new, dates

def concat_dict_dict(dict_df1, dict_df2):
    dates1 = get_dates_from_dict(dict_df1)
    dates2 = get_dates_from_dict(dict_df2)
    dates, dates1_new, dates2_new, ind1, ind2 = intersection_index(dates1, dates2)
    if len(dates) == 0:
        raise ValueError('Cannot sync datasets. there is no common dates')
    dict_new = dict(zip(dates1_new, dates2_new))
    keys = set(list(dict_df1.keys()) + list(dict_df2.keys()))
    for key in keys:
        if key in dict_df1.keys() and key in dict_df2.keys():
            dict_new[key] = pd.concat([dict_df1[key].loc[dates], dict_df2[key].loc[dates]], axis=1)
        elif key in dict_df1.keys():
            dict_new[key] = dict_df1[key].loc[dates]
        else:
            dict_new[key] = dict_df2[key].loc[dates]
    print(f'Merge dictionary with dictionary found {dates.shape[0]} common dates ')
    return dict_new, dates


def concatenate_numpy(data1, dates1, data2, dates2):
    dates, dates1_new, dates2_new, ind1, ind2 = intersection_index(dates1, dates2)
    data1 = data1[ind1]
    data2 = data2[ind2]
    return np.concatenate([data1, data2], axis=1)


def sync_data_tensor_with_tensors(data_tensor1=None, dates_tensor1=None, data_tensor2=None, dates_tensor2=None):
    dates, dates_tensor1, dates_tensor2, ind1, ind2 = intersection_index(dates_tensor1, dates_tensor2)
    if isinstance(data_tensor1, dict):
        for key, data in data_tensor1.items():
            if not isinstance(data, np.ndarray):
                raise ValueError('data_tensor should be np.ndarray')
            data_tensor1[key] = data[ind1]
    else:
        data_tensor1 = data_tensor1[ind1]
    if isinstance(data_tensor2, dict):
        for key, data in data_tensor2.items():
            if not isinstance(data, np.ndarray):
                raise ValueError('data_tensor should be np.ndarray')
            data_tensor2[key] = data[ind2]
    else:
        data_tensor2 = data_tensor2[ind2]
    return data_tensor1, data_tensor2, dates


def sync_data_row_with_tensors(data_tensor=None, dates_tensor=None, data_row=None, dates_row=None):
    dates, dates_tensor, dates_row, ind1, ind2 = intersection_index(dates_tensor, dates_row)
    if not isinstance(data_row, pd.DataFrame):
        raise ValueError('data_row should be dataframe')
    if isinstance(data_tensor, dict):
        for key, data in data_tensor.items():
            if not isinstance(data, np.ndarray):
                raise ValueError('data_tensor should be np.ndarray')
            data_tensor[key] = data[ind1]
    else:
        data_tensor = data_tensor[ind1]
    data_row = data_row.loc[get_time_level(dates_tensor)]
    return data_tensor, data_row, dates


def np_ffill(arr, axis):
    idx_shape = tuple([slice(None)] + [np.newaxis] * (len(arr.shape) - axis - 1))
    idx = np.where(~np.isnan(arr), np.arange(arr.shape[axis])[idx_shape], 0)
    np.maximum.accumulate(idx, axis=axis, out=idx)
    slc = [np.arange(k)[tuple([slice(None) if dim==i else np.newaxis
        for dim in range(len(arr.shape))])]
        for i, k in enumerate(arr.shape)]
    slc[axis] = idx
    return arr[tuple(slc)]

def upsample_dict(data_dict, dates_dict, dates_new, periods=4):
    if isinstance(data_dict, dict):
        for key, data in data_dict.items():
            data_dict[key] = upsample_dict(data, dates_dict, dates_new, periods)
        return data_dict
    elif isinstance(data_dict, np.ndarray):
        # every new row takes the last source row at or before it (a row-level forward fill), as one
        # indexed copy in the source dtype: the former float64 NaN tensor, its element-wise forward
        # fill and the axis copy that followed took three times 27 GB for a four-year 15-minute case
        _, _, _, ind1, ind2 = intersection_index(dates_new, dates_dict)
        pos = np.full(dates_new.shape[0], -1, dtype=np.int64)
        pos[ind1] = ind2
        pos = np.maximum.accumulate(pos)
        missing = pos < 0
        tensor_new = data_dict[np.where(missing, 0, pos)]
        if missing.any():
            tensor_new = tensor_new.astype(np.float32, copy=False)
            tensor_new[missing] = np.nan
        return tensor_new
    else:
        return upsample_dataset(data_dict, dates_new, periods)

def rearrange_dict(data_dict):
    if isinstance(data_dict, dict):
        for key, data in data_dict.items():
            data_dict[key] = rearrange_dict(data)
        return data_dict
    elif isinstance(data_dict, Rows):
        return data_dict.over(lambda array: np.moveaxis(array, 4, 2))      # the same view of rows still on disk
    else:
        return np.moveaxis(data_dict, 4, 2)          # 'b t w h c -> b t c w h' as a view: no copy of the tensor

#: Steps in an hour for the resolutions hourly weather rows are spread over.
STEPS_PER_HOUR = {'15min': 4, '5min': 12}


def upsample_tensor(data_tensor, dates_tensor, freq='15min'):
    """Hourly rows repeated over the steps of their hour: four at 15 minutes, twelve at 5."""
    periods = STEPS_PER_HOUR[freq]
    if isinstance(dates_tensor, pd.DatetimeIndex):
        dates_new = pd.DatetimeIndex([])
        for date in dates_tensor:
            dates_new = dates_new.append(pd.date_range(date, periods=periods, freq=freq))
    elif isinstance(dates_tensor, pd.MultiIndex):
        dates_new = pd.MultiIndex.from_arrays([[], []], names=["predtime", "currtime"])
        for date in dates_tensor:
            temp = pd.MultiIndex.from_tuples((d, date[1]) for d in pd.date_range(date[0], periods=periods, freq=freq))
            dates_new = dates_new.append(temp)

    data_tensor = upsample_dict(data_tensor, dates_tensor, dates_new, periods)
    return data_tensor, dates_new

def downsample_dict(data_dict, dates_dict, resolution='D'):
    if isinstance(data_dict, dict):
        for key, data in data_dict.items():
            data_dict[key], dates_new = downsample_dict(data, dates_dict, resolution=resolution)
        return data_dict, dates_new
    elif isinstance(data_dict, np.ndarray):
        shape = data_dict.shape
        tensor_new = data_dict.reshape([shape[0], np.prod(shape[1:])])
        df = pd.DataFrame(tensor_new, index=dates_dict)
        if isinstance(dates_dict, pd.DatetimeIndex):
            df = df.resample(resolution).mean()
        elif isinstance(dates_dict, pd.MultiIndex):
            dates = df.index
            df = df.groupby(level="currtime").resample(resolution, level="predtime").mean()
            df = df.sort_index()
            if isinstance(df.index, pd.MultiIndex) and df.index.nlevels > len(dates.names):
                df.index = df.index.droplevel(0)
        return df.values.reshape(-1, *shape[1:]), df.index
    else:
        data_df = downsample_dataset(data_dict, resolution=resolution)
        return data_df, data_df.index


def downsample_tensor(data_tensor, dates_tensor, resolution='D'):
    data_tensor, dates_new = downsample_dict(data_tensor, dates_tensor, resolution=resolution)
    return data_tensor, dates_new

def upsample_dataset(df, dates, periods=4):
    """The hourly rows of ``df`` on the finer grid ``dates``, each step taking the nearest hour.

    With ``periods`` steps to the hour the first half of an hour (its own stamp included) keeps
    that hour's row and the rest takes the next one: at 15 minutes two steps forward and one
    back, at 5 minutes six forward and five back.
    """
    forward = periods // 2
    back = periods - 1 - forward
    df_temp = pd.DataFrame(index=dates, columns=df.columns, dtype=float)
    df_temp.loc[df.index] = df.astype(float)
    if isinstance(df_temp.index, pd.MultiIndex):
        df_temp = df_temp.sort_index().groupby(level="currtime").apply(lambda x: x.bfill(limit=back).ffill(limit=forward))
        if isinstance(df_temp.index, pd.MultiIndex) and df_temp.index.nlevels > len(dates.names):
            df_temp.index = df_temp.index.droplevel(0)
    else:
        df_temp = df_temp.bfill(axis=0, limit=back).ffill(axis=0, limit=forward)

    return df_temp.dropna(axis=0)

def downsample_dataset(df, resolution='D'):
    return df.resample(resolution).mean()

def sync_target_with_tensors(target=None, data_tensor=None, dates_tensor=None, data_row=None):
    dates, dates_tensor, index, ind1, ind2 = intersection_index(dates_tensor, target.index)
    if isinstance(data_tensor, dict):
        for key, data in data_tensor.items():
            if isinstance(data, np.ndarray):
                data_tensor[key] = data[ind1]
            else:
                data_tensor[key] = data.iloc[ind1]
    else:
        data_tensor = data_tensor[dates_tensor]
    target = target.loc[get_time_level(dates_tensor)]
    if data_row is not None:
        data_row = data_row.loc[get_time_level(dates_tensor)]
        return [data_tensor, data_row], target
    else:
        return data_tensor, target

def sync_datasets(df1, df2):
    """Two frames on their common dates, each in its own rows.

    For the two clocks of a ``long_term`` case this returns the inputs at their three rows an
    hour and the target at its one: what the split wants, which draws its periods from the
    target's own dates. A caller that needs them row for row wants ``sync_rows``.
    """
    dates, index1, index2, ind1, ind2 = intersection_index(df1.index, df2.index)
    if len(dates) == 0:
        raise ValueError('Cannot sync datasets. there is no common dates')
    print(f'Sync pandas datasets with {dates.shape[0]} common dates ')
    return df1.loc[index1], df2.loc[index2]


def sync_rows(df1, df2):
    """Two frames on their common dates, row for row.

    When one of them is on the two clocks of a ``long_term`` case -- (prediction time, issue
    time), several rows to the hour -- the other is repeated onto those rows and takes their
    index, so that row *i* of one is row *i* of the other. ``sync_datasets`` leaves the inputs
    at their three rows an hour and the target at its one, and a shallow model that sliced both
    by position paired most rows with the target of another hour: 431 of 500 on a real case.
    The branches are the ones the owner's long_term project carries in its ``sync_datasets``
    (``AdmieFC_train_TransferLearning``). On one clock the two functions return the same.
    """
    dates, index1, index2, ind1, ind2 = intersection_index(df1.index, df2.index)
    if len(dates) == 0:
        raise ValueError('Cannot sync datasets. there is no common dates')
    print(f'Sync pandas datasets with {dates.shape[0]} common dates ')
    one, two = isinstance(index1, pd.MultiIndex), isinstance(index2, pd.MultiIndex)
    if one and not two:
        repeated = df2.loc[index1.get_level_values(0)]
        repeated.index = index1
        return df1.loc[index1], repeated
    if two and not one:
        repeated = df1.loc[index2.get_level_values(0)]
        repeated.index = index2
        return repeated, df2.loc[index2]
    return df1.loc[index1], df2.loc[index2]


def rows_at(dates, wanted):
    """Positions in ``dates`` of the rows ``wanted`` names.

    A mask of the split or of a cluster names hours. On the two clocks of a ``long_term`` case
    an hour is several rows, one per issue day, and all of them are meant; they come in the
    order ``dates`` has them. On one clock it is the lookup it always was, in the mask's order.
    """
    if isinstance(dates, pd.MultiIndex) and not isinstance(wanted, pd.MultiIndex):
        return np.flatnonzero(get_time_level(dates).isin(wanted))
    if not dates.is_unique:
        # The same rows by another road: a target that ``sync_dict_df`` repeated onto two-clock
        # rows keeps a plain time index in which each hour appears once per issue day. A label
        # lookup refuses such an index ("Reindexing only valid with uniquely valued Index").
        return np.flatnonzero(dates.isin(wanted))
    return dates.get_indexer(wanted.intersection(dates))

def find_common_dates(dict_df, dates_dict=None):
    dates = []
    if isinstance(dict_df, dict):
        for k, v in dict_df.items():
            results = find_common_dates(v, dates_dict=dates_dict)
            for res in results:
                dates.append(res)
    elif isinstance(dict_df, pd.DataFrame):
        dates.append(dict_df.index)
    elif isinstance(dict_df, (np.ndarray, TensorView, Rows)):
        if dates_dict is None:
            raise ValueError('Data is not a dataframe. The corresponding dates are needed')
        dates.append(dates_dict)
    return dates

def get_data_dict_from_dates(dict_df, dates, dates_dict=None):
    if isinstance(dict_df, dict):
        for k, v in dict_df.items():
            dict_df[k] = get_data_dict_from_dates(v, dates, dates_dict=dates_dict)
    elif isinstance(dict_df, pd.DataFrame):
        dates_, dates_dict, dates, ind1, ind2 = intersection_index(dict_df.index, dates)
        return dict_df.loc[dates_dict]
    elif isinstance(dict_df, (np.ndarray, TensorView, Rows)):
        if dates_dict is None:
            raise ValueError('Data is not a dataframe. The corresponding dates are needed')
        dates, dates_tensor, index, ind1, ind2 = intersection_index(dates_dict, dates)
        return take_rows(dict_df, ind1)             # rows on disk are narrowed, not read
    return dict_df


def sync_dict_df(dict_df, df2, dates_dict=None):
    dates_list = find_common_dates(dict_df, dates_dict=dates_dict)
    dates = pd.DatetimeIndex([])
    for date in dates_list:
        if dates.shape[0] == 0:
            dates = date
        else:
            dates = dates.intersection(date)
    dates_, dates_dict, index, ind1, ind2 = intersection_index(dates, df2.index)
    df2 = df2.loc[get_time_level(dates_dict)]
    if len(dates) == 0:
        raise ValueError('Cannot sync datasets. there is no common dates')
    dict_df = get_data_dict_from_dates(dict_df, df2.index, dates_dict=dates)
    print(f'Sync pandas datasets with {dates_.shape[0]} common dates ')
    return dict_df, df2


def fix_timeseries_dates(df, freq='H'):
    """The frame on the case's own grid: one row per slot, in order, every slot in between present.

    Two different things put more than one row in a slot, and they do not mean the same.

    A **repeated timestamp** describes one instant twice, the later row being a correction of the
    earlier, so the later row wins -- as it always has.

    Rows at **finer stamps than the grid**, quarter-hourly readings in an hourly case, are
    measurements of different instants inside the slot, so the slot takes their mean.  A power
    averaged over the hour is what an hourly stamp claims to be; picking one reading instead files
    the measurement taken at HH:15 or HH:30 under HH:00, which shifts the series against the
    weather and calendar inputs aligned to the same stamp.  Anything that is not a number cannot
    be averaged, so for those columns the last row of the slot is kept.

    A row belongs to the slot it falls **in**, which is why the stamp is floored and not rounded:
    rounding to the nearest slot puts HH:45 in the slot after it and makes the averaging window
    swing between five rows and three (``round`` breaks the half-way tie towards the even hour),
    so the hour would be centred on its own label instead of starting at it.

    Slots with no row at all stay empty here; the dataset builder drops them later.
    """
    freq = pandas_freq(freq)
    df = df[~df.index.duplicated(keep='last')].copy()
    df.index = df.index.floor(freq)
    df = _slot_values(df) if df.index.has_duplicates else df.sort_index()
    dates = pd.date_range(df.index[0], df.index[-1], freq=freq)
    df_out = pd.DataFrame(index=dates, columns=df.columns)
    dates_in = dates.intersection(df.index)
    df_out.loc[dates_in] = df
    return df_out


def _slot_values(df):
    """One row per slot: the mean of the numbers in it, the last value of anything else."""
    slots = df.groupby(level=0, sort=True)
    numbers = [name for name in df.columns if is_numeric_dtype(df[name])]
    parts = [slots[numbers].mean()] if numbers else []
    rest = [name for name in df.columns if name not in set(numbers)]
    if rest:
        parts.append(slots[rest].last())
    return pd.concat(parts, axis=1)[df.columns]


def get_slice(data, ind):
    if isinstance(data, dict):
        for key, value in data.items():
            data[key] = get_slice(value, ind)
    elif isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
        return data.iloc[ind].astype(np.float32)
    elif isinstance(data, pd.DatetimeIndex) or isinstance(data, pd.MultiIndex):
        return data[ind]
    else:
        return data[ind].astype(np.float32)
    return data


def get_slice_with_dates(data, data_dates, dates):
    """The rows of ``data`` at ``dates``; a new dict for dictionary inputs (the input is never modified,
    so the callers need no deep copy of the full datasets)."""
    if isinstance(data, dict):
        return {key: get_slice_with_dates(value, data_dates, dates) for key, value in data.items()}
    elif isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
        _, _, _, ind, _ = intersection_index(data.index, dates)
        return data.iloc[ind]
    else:
        dates_, data_dates, index, ind1, ind2 = intersection_index(data_dates, dates)
        return data[ind1]
    return data


def sync_data_with_dates(x, dates, dates_x=None):
    """The rows of ``x`` whose time is among ``dates``, in the order ``x`` has them.

    By position, not by label. The target of a ``long_term`` case is aligned to its
    (prediction time, issue time) rows, so an hour appears in it once per issue day; looking
    those hours up by label returned every row of an hour once per occurrence -- nine rows for
    three -- and the selectors were fitted on inputs and targets of different lengths, which
    CatBoost refused and the selector answered by keeping every feature.
    """
    if dates_x is not None:
        ind1 = intersection_index(dates_x, dates)[3]
        return x[ind1]
    ind1 = intersection_index(x.index, dates)[3]
    return x.iloc[ind1]


def recursive_copy(src, dest):
    """
    Copy each file from src dir to dest dir, including sub-directories.
    """
    for item in os.listdir(src):
        file_path = os.path.join(src, item)

        # if item is a file, copy it
        if os.path.isfile(file_path):
            shutil.copy(file_path, dest)

        # else if item is a folder, recurse
        elif os.path.isdir(file_path):
            new_dest = os.path.join(dest, item)
            os.mkdir(new_dest)
            recursive_copy(file_path, new_dest)

def create_image_batch(static_data, params, path_dataset, batch_size, tag, dates, use_target, api):
    from pathlib import Path
    from eforecast.datasets.image_data.batches import content, write
    _image_classes()
    tag = tag if tag is not None else 'image'
    source = ImageDataset(static_data, dates, params, use_target=use_target, api=api) if not params.get('image_data_type') or use_target else None
    if source is None:
        from eforecast.datasets.image_data.image_dataset_real_time import ImageDatasetRealTime
        reader = ImageDatasetRealTime.__new__(ImageDatasetRealTime)
        reader.static_data, reader.params = copy.deepcopy(static_data), copy.deepcopy(params)
        reader.path_sat_processed = static_data['path_image']
        reader.image_type = params['image_data_type'].split(':')
        reader.final_size, reader.spatial_coord_use = int(params['final_image_size']), params['spatal_coord']
        reader.lags = next(v['lags'] for v in static_data['variables'] if v.get('type') == 'image')
        class Pixels:
            cv_mask = dates
            def get_safe(self, date):
                try:
                    return {'date': pd.DatetimeIndex([date]), 'images': reader.get_image(date)[0]}
                except (OSError, ValueError, KeyError):
                    log.warning('cannot prepare processed image %s', date, exc_info=True)
                    return None
        source = Pixels()
    dataset = ImageDataloader(source,
                              max(1, min(32, int(batch_size))), 1)
    records = []
    while len(dataset.dates) < dataset.length:
        x_batch = dataset.get_batch(randomly=False)
        if 'images' not in x_batch:
            continue
        name = f'{tag}_tensor{len(records)}.pt'
        write(path_dataset, name, x_batch)
        records.append({'name': name, 'content': content(Path(path_dataset) / name)})
    return records


def create_image_batches(static_data, params, batch_size, gpu_id, use_target, tags=('train', 'val', 'test'),
                         api='eumdac', dates=None):
    import torch
    from pathlib import Path
    from eforecast.datasets.image_data.batches import directory, seal
    _image_classes()
    if tags is None and dates is None:
        raise ValueError('create_image_batches needs tags (train, val, test, eval) or explicit dates')
    path_dataset = directory()
    records = {}
    if dates is not None:
        records['image'] = create_image_batch(static_data, params, path_dataset, batch_size, None, dates,
                                               use_target, api)
    else:
        cv_masks = joblib.load(os.path.join(params['cluster_dir'], 'cv_mask.pickle'))
        for tag in tags:
            if tag in ('train', 'val', 'test'):
                selected = cv_masks[('train', 'val', 'test').index(tag)]
            elif tag == 'eval':
                from eforecast.datasets.files_manager import FilesManager
                selected = FilesManager(static_data, train=False, is_online=False).check_if_exists_image_data()
                if isinstance(selected, tuple): selected = selected[0]
                if isinstance(selected, dict):
                    selected = get_dates_from_dict({k: pd.DataFrame(index=v) for k, v in selected.items()})
            else:
                raise ValueError(f'unknown image batch tag {tag}')
            records[tag] = create_image_batch(static_data, params, path_dataset, batch_size, tag, selected,
                                                use_target, api)
    seal(path_dataset, records)
    params['path_image_batches'] = str(path_dataset)
    if 'train' in records and records['train']:
        params['x_sample'] = torch.load(path_dataset / records['train'][0]['name'], map_location='cpu',
                                        weights_only=False)
    return params

def load_data_shallow_models(data_feeder, data_types, data_tag, train, get_lstm_vars=False):
    X = dict()
    metadata = dict()
    dates = None
    merge, compress, image_data_type, scale_nwp_method, scale_row_method, scale_img_method, fs_method = (data_types[data_tag].get('merge'),
                                                           data_types[data_tag].get('compress'),
                                                           data_types[data_tag].get('image_data_type'),
                                                           data_types[data_tag].get('scale_nwp_method'),
                                                           data_types[data_tag].get('scale_row_method'),
                                                           data_types[data_tag].get('scale_img_method'),
                                                           data_types[data_tag].get(
                                                               'feature_selection_method'))
    x, meta_x = data_feeder.feed_inputs(data_tag,
                                        merge=merge, compress=compress,
                                        scale_nwp_method=scale_nwp_method,
                                        scale_row_method=scale_row_method,
                                        scale_img_method=scale_img_method,
                                        image_data_type=image_data_type,
                                        feature_selection_method=fs_method,
                                        get_lstm_vars=get_lstm_vars)
    X.update(x)
    metadata.update(meta_x)
    dates = meta_x[data_tag]['dates'] if dates is None else meta_x[data_tag]['dates'].intersection(dates)
    if train:
        y = data_feeder.feed_target()
        X, y = sync_dict_df(X, y, dates_dict=dates)
        for data_tag in metadata.keys():
            metadata[data_tag]['dates'] = y.index
        return X, y, metadata,
    else:
        return X, metadata

def get_fuzzy_variables(params, columns):
    fuzzy_variables = dict()
    for var_name in params['var_imp']:
        if '$' in var_name:
            var_names = [c for c in columns if var_name.split('$')[0].lower() in c.lower()
                         and var_name.split('$')[1].lower() in c.lower()]
        else:
            var_names = [c for c in columns if var_name.lower() in c.lower()]
        if var_name not in columns:
            if len(var_names) == 0:
                raise ValueError(f'Cannot find variables associated with {var_name}')
            fuzzy_variables[var_name] = var_names
        else:
            fuzzy_variables[var_name] = var_name
    return fuzzy_variables

def create_farm_shards_centers(x, y, params, model_layers, refit):
    from eforecast.common_utils.train_utils import create_centroids

    if refit or 'centroids' not in params.keys():
        X_imp1, y1 = x['clustering'].detach().cpu().numpy(), y.detach().cpu().numpy()
        params = create_centroids(X_imp1, y1, params, abstract=True)

    new_model_layers = dict()
    for data_tag in x.keys():
        if data_tag not in {'clustering', 'farm_id'}:
            new_model_layers[data_tag] = model_layers[data_tag.split('/')[0]]

    for layer in model_layers.keys():
        if 'hidden' in layer or 'output' in layer or layer == 'images':
            new_model_layers[layer] = model_layers[layer]
    model_layers = copy.deepcopy(new_model_layers)
    return params, model_layers


def centroid_pool(x, y, train_dates, train):
    """The rows the fuzzy rule initialiser may see: the canonical training dates, and no others.

    ``create_centroids`` fits RBF-OLS candidates on a chronological 75/25 split of whatever it is
    given, and it was given the whole pre-evaluation intersection -- the canonical validation, test
    and boundary-purged rows with it. The rule centres and widths of every fuzzy model were
    therefore initialised from held-out targets, which is the leak Steps 04 to 06 removed
    everywhere else and never looked for here.
    """
    if train_dates is None:
        if train:
            log.warning('the fuzzy rule initialiser was given no training mask, so it sees every date of '
                        'the case: its centres and widths are fitted with held-out targets. The caller '
                        'should pass train_dates (the first canonical mask).')
        return x, y
    dates = pd.DatetimeIndex(train_dates)
    x = x[get_time_level(x.index).isin(dates)]
    y = y[get_time_level(y.index).isin(dates)]
    log.info('fuzzy rule initialiser: %d rows on the %d canonical training dates', x.shape[0], len(dates))
    return x, y


def load_data_deep_models(data_feeder, data_types, model_layers, params, cluster, train, is_fuzzy, refit=False,
                          return_f_center=True, inverse=False, train_dates=None):
    from eforecast.common_utils.train_utils import create_centroids

    # The legacy image dataset has a separate eager companion-array contract.
    # Keep hybrid image models on that reader until A11-11 owns their batches.
    data_feeder.reader_consumer = {**getattr(data_feeder, 'reader_consumer', {}),
        'image_pixels': any(key.split('/')[0] == 'images' for key in model_layers)}
    X = dict()
    metadata = dict()
    for data_tag_layer in model_layers.keys():
        if '/' in data_tag_layer:
            data_tag = data_tag_layer.split('/')[0]
        else:
            data_tag = data_tag_layer
        if data_tag in data_types.keys():
            merge, compress, image_data_type, scale_nwp_method, scale_row_method, scale_img_method, fs_method = (
                data_types[data_tag].get('merge'),
                data_types[data_tag].get('compress'),
                data_types[data_tag].get('image_data_type'),
                data_types[data_tag].get('scale_nwp_method'),
                data_types[data_tag].get('scale_row_method'),
                data_types[data_tag].get('scale_img_method'),
                data_types[data_tag].get(
                    'feature_selection_method'))
            if data_tag == 'images':
                continue
                ## TODO Fix Image outputs in load_data_deep_models for Image2image case
            else:
                x, meta_x = data_feeder.feed_inputs(data_tag,
                                                   merge=merge, compress=compress,
                                                   scale_nwp_method=scale_nwp_method,
                                                   scale_row_method=scale_row_method,
                                                    scale_img_method=scale_img_method,
                                                    image_data_type=image_data_type,
                                                   feature_selection_method=fs_method,
                                                   cluster=cluster,
                                                   transform_calendar=True, inverse_transform=False)

            if meta_x.get(data_tag) is None or x.get(data_tag) is None:
                raise ValueError(f"no data for input {data_tag!r} with merge={merge!r}, compress={compress!r}, "
                                 f"scale_nwp_method={scale_nwp_method!r}, scale_row_method={scale_row_method!r}: "
                                 f"check the method's merge/scale options against the datasets of the project")
            if 'cal_vars' in meta_x[data_tag].keys():
                params['cal_vars'] = meta_x[data_tag]['cal_vars']
            X.update(x)
            metadata.update(meta_x)
        elif data_tag == 'images' or 'output' in data_tag or data_tag == 'hidden_layer':
            continue
        else:
            print(f'Warning: Net brunch of {data_tag} input tag ignored.')
    if is_fuzzy:
        if 'row_all' not in X.keys():
            raise RuntimeError("Dataset with tag 'row_all' not found to inputs")
        x = X['row_all']
        fuzzy_variables = get_fuzzy_variables(params, list(x.columns))
        X_imp = None
        for var_name, var_cols in fuzzy_variables.items():
            if isinstance(var_cols, list):
                var_mean = x[var_cols].mean(axis=1).to_frame(var_name)
                X_imp =  var_mean if X_imp is None else pd.concat([X_imp, var_mean], axis=1)
            else:
                var_mean = x[var_cols].to_frame(var_name)
                X_imp = var_mean if X_imp is None else pd.concat([X_imp, var_mean], axis=1)
        X['clustering'] = X_imp
        metadata['clustering'] = {'dates': X_imp.index}
        if return_f_center:
            if refit or ('rules' not in params.keys() and 'centroids' not in params.keys()):
                y = data_feeder.feed_target()
                X_imp1, y1 = X_imp.copy(), y.copy()
                X_imp1, y1 = centroid_pool(X_imp1, y1, train_dates, train)
                params = create_centroids(X_imp1, y1, params)
        new_X = dict()
        new_metadata = dict()
        new_model_layers = dict()
        for rule in params['rules']:
            for data_tag in X.keys():
                if data_tag not in {'clustering'}:
                    new_X[f'{data_tag}/{rule}'] = X[data_tag]
                    new_metadata[f'{data_tag}/{rule}'] = metadata[data_tag]
                    if data_tag not in model_layers.keys():
                        data_tag_layer = f'{data_tag}/{rule}'
                    else:
                        data_tag_layer = data_tag
                    new_model_layers[f'{data_tag}/{rule}'] = model_layers[data_tag_layer]
                else:
                    new_X[data_tag] = X[data_tag]
                    new_metadata[data_tag] = metadata[data_tag]
        for layer in model_layers.keys():
            if 'hidden' in layer or 'output' in layer or layer == 'images':
                new_model_layers[layer] = model_layers[layer]
        X = copy.deepcopy(new_X)
        metadata = copy.deepcopy(new_metadata)
        model_layers = copy.deepcopy(new_model_layers)
    else:
        params['rules'] = None
    new_X = dict()
    new_metadata = dict()
    new_model_layers = dict()
    for data_tag, data in X.items():
        if data_tag not in {'clustering'}:
            if data_tag not in model_layers.keys():
                raise ValueError(f'Cannot find {data_tag} tag in network architecture')
            elif isinstance(data, dict) and data_tag not in {'lstm'}:
                for tag, values in data.items():
                    new_X[f'{data_tag}/{tag}'] = X[data_tag][tag]
                    new_metadata[f'{data_tag}/{tag}'] = metadata[data_tag]
                    new_model_layers[f'{data_tag}/{tag}'] = model_layers[data_tag]
            else:
                new_X[data_tag] = X[data_tag]
                new_metadata[data_tag] = metadata[data_tag]
                new_model_layers[data_tag] = model_layers[data_tag]
        else:
            new_X[data_tag] = X[data_tag]
            new_metadata[data_tag] = metadata[data_tag]
    for layer in model_layers.keys():
        if 'hidden' in layer or 'output' in layer or layer == 'images':
            new_model_layers[layer] = model_layers[layer]
    X = copy.deepcopy(new_X)
    metadata = copy.deepcopy(new_metadata)
    model_layers = copy.deepcopy(new_model_layers)
    dates = None
    for data_tag in metadata.keys():
        if data_tag not in {'images'}:
            if dates is None:
                dates = metadata[data_tag]['dates']
            else:
                dates, dates_x, dates_, ind1, ind2 = intersection_index(metadata[data_tag]['dates'], dates)
    for data_tag in metadata.keys():
        if isinstance(X[data_tag], pd.DataFrame):
            X[data_tag] = X[data_tag].loc[dates]
        elif isinstance(X[data_tag], (np.ndarray, TensorView, Rows)):
            indices = np.where(metadata[data_tag]['dates'].isin(dates))[0]
            X[data_tag] = take_rows(X[data_tag], indices)
        else:
            for key in X[data_tag].keys():
                if isinstance(X[data_tag][key], pd.DataFrame):
                    X[data_tag][key] = X[data_tag][key].loc[dates]
                elif isinstance(X[data_tag][key], (np.ndarray, TensorView, Rows)):
                    indices = np.where(metadata[data_tag]['dates'].isin(dates))[0]
                    X[data_tag][key] = take_rows(X[data_tag][key], indices)
                else:
                    raise ValueError(f'Unknown type of data: {type(X[data_tag])}')
        _, dates_x, _, _, _ = intersection_index(metadata[data_tag]['dates'], dates)
        metadata[data_tag]['dates'] = dates_x
    if train:
        y = data_feeder.feed_target()
        X, y = sync_dict_df(X, y, dates_dict=dates)
        for data_tag in metadata.keys():
            metadata[data_tag]['dates'] = y.index
        return X, y, metadata, model_layers, params
    else:
        return X, metadata, model_layers, params


def create_farm_batches(static_data_all, path_batches, data_types, model_layers, params, is_fuzzy, batch_size=2048,
                        farms_per_chunk=10, tag='train',
                        cv_mask_idx=0, rule_tag=None, farm_meta_dict=None, global_scalers=False, inverse=False):
    """Write all 179 farm datasets for one split into three memory-mapped binary files.


    Returns:
        int: total number of rows written.
    """
    import gc
    from tqdm import tqdm
    from eforecast.datasets.data_feeder import DataFeeder
    if rule_tag is not None:
        path_batches = os.path.join(path_batches, rule_tag)
    os.makedirs(path_batches, exist_ok=True)

    y_path         = os.path.join(path_batches, f'{tag}_y.dat')
    id_path        = os.path.join(path_batches, f'{tag}_farm_id.dat')
    farm_meta_path = os.path.join(path_batches, f'{tag}_farm_meta.dat')
    meta_path      = os.path.join(path_batches, f'{tag}_meta.pickle')
    if global_scalers:
        scalers_path   = os.path.join(path_batches, 'global_scalers.pickle')

    if os.path.exists(meta_path):
        meta = joblib.load(meta_path)
        print(f'[create_farm_batches] {tag} memmap already exists with {meta["n_rows"]} rows. Skipping.')
        return meta['n_rows']

    n_farms = len(static_data_all)
    n_features = None
    n_out = None
    schema = None
    target_columns = None
    row_offset = 0
    x_mm = None
    y_mm = None
    id_mm = None
    meta_mm = None

    def _count_rows(static_data_list, cv_mask_idx_, data_types, model_layers, params, is_fuzzy):
        """First pass: count total rows so we can pre-allocate the memmap."""
        total = 0
        print(f'begin to count rows for {rule_tag}....')
        for sd in tqdm(static_data_list):
            if rule_tag is not None:
                cv_masks_file = os.path.join(sd['path_model'], 'cluster_organizer', 'RBF', rule_tag, 'cv_mask.pickle')
            else:
                cv_masks_file = os.path.join(sd['path_data'], 'cv_mask.pickle')
            if not os.path.exists(cv_masks_file):
                continue
            cv_masks_ = joblib.load(cv_masks_file)
            try:
                if rule_tag is not None:
                    cluster = {'cluster_name': rule_tag,
                               'cluster_path': os.path.join(sd['path_model'], 'cluster_organizer', 'RBF', rule_tag)}
                else:
                    cluster = {'cluster_name': 'global',
                               'cluster_path': os.path.join(sd['path_model'],
                                                            'global')}
                data_feeder = DataFeeder(sd, online=False, train=True)
                X, y, metadata, _, _ = load_data_deep_models(data_feeder, data_types, model_layers, params, cluster, True,
                                                          is_fuzzy, return_f_center=False, inverse=inverse)
                _, _, mask = get_slice_for_nets(copy.deepcopy(X), metadata, dates=cv_masks_[cv_mask_idx_], y=y)
                total += len(mask)
            except Exception as e:
                print(f'[create_farm_batches] {tag} for  {rule_tag}: skipping farm {sd["_id"]} with {e}')
                continue
        print(f'end to count rows....{total}')
        return total

    def _fit_global_scalers(static_data_list, cv_mask_idx_, data_types_, model_layers_, params_, is_fuzzy_):
        """Pass 1.5: fit one StandardScaler per data_tag across all farms on raw (unscaled) data."""
        from sklearn.preprocessing import StandardScaler
        from eforecast.datasets.data_feeder import DataFeeder
        all_data = {}
        print(f'[create_farm_batches] fitting global scalers for {rule_tag}...')
        for sd in tqdm(static_data_list):
            if rule_tag is not None:
                cv_masks_file = os.path.join(sd['path_model'], 'cluster_organizer', 'RBF', rule_tag, 'cv_mask.pickle')
            else:
                cv_masks_file = os.path.join(sd['path_data'], 'cv_mask.pickle')
            if not os.path.exists(cv_masks_file):
                continue
            cv_masks_ = joblib.load(cv_masks_file)
            try:
                if rule_tag is not None:
                    cluster_ = {'cluster_name': rule_tag,
                                'cluster_path': os.path.join(sd['path_model'], 'cluster_organizer', 'RBF', rule_tag)}
                else:
                    cluster_ = {'cluster_name': 'global',
                                'cluster_path': os.path.join(sd['path_model'], 'global')}
                df = DataFeeder(sd, online=False, train=True)
                df.reader_consumer = {'kind': 'farm', 'global_scalers': True}
                X_, y_, meta_, _, _ = load_data_deep_models(df, data_types_, model_layers_, params_, cluster_,
                                                            True, is_fuzzy_, return_f_center=False, inverse=inverse)
                x_sl, _, _ = get_slice_for_nets(copy.deepcopy(X_), meta_, dates=cv_masks_[cv_mask_idx_], y=y_)
                if rule_tag is not None:
                    x_sl = x_sl[list(model_layers_.keys())[0]]
                for data_tag, data in x_sl.items():
                    arr = data.values if isinstance(data, pd.DataFrame) else np.asarray(data)
                    if arr.ndim == 1:
                        arr = arr[:, np.newaxis]
                    arr2d = arr.reshape(-1, arr.shape[-1]) if arr.ndim == 3 else arr.reshape(arr.shape[0], -1)
                    if data_tag not in all_data:
                        all_data[data_tag] = []
                    all_data[data_tag].append(arr2d.astype(np.float32))
            except Exception as e:
                print(f'[fit_global_scalers] skipping farm {sd["_id"]}: {e}')
                continue
        scalers = {}
        for data_tag, arrays in all_data.items():
            combined = np.concatenate(arrays, axis=0)
            sc = StandardScaler()
            sc.fit(combined)
            scalers[data_tag] = sc
            print(f'  scaler for {data_tag}: mean={sc.mean_.mean():.4f}, std={sc.scale_.mean():.4f}')
        return scalers

    def _apply_global_scaler(arr, scaler):
        """Apply a fitted StandardScaler to a 2D or 3D array, preserving shape."""
        original_shape = arr.shape
        if arr.ndim == 3:
            arr2d = arr.reshape(-1, arr.shape[-1])
            return scaler.transform(arr2d).reshape(original_shape).astype(np.float32)
        else:
            arr2d = arr.reshape(arr.shape[0], -1)
            return scaler.transform(arr2d).reshape(original_shape).astype(np.float32)

    spatial_pad_mode = 'mean'
    if isinstance(params, dict):
        spatial_pad_mode = params.get('spatial_pad_mode', 'mean')
    allowed_pad_modes = {'mean', 'linear_ramp'}
    if spatial_pad_mode not in allowed_pad_modes:
        raise ValueError(f'Unsupported spatial_pad_mode={spatial_pad_mode}. Allowed: {allowed_pad_modes}')

    def _pad_arrays_for_concat(arrays, data_tag):
        """Pad only spatial dimensions (last two axes) to 17x17 for concat on axis 0."""

        ndims = {arr.ndim for arr in arrays}
        if len(ndims) != 1:
            raise ValueError(f'Inconsistent ndims for {data_tag}: {[arr.shape for arr in arrays]}')

        ndim = arrays[0].ndim
        if ndim <= 3:
            return arrays

        target_lat, target_lon = 17, 17
        padded_arrays = []
        needs_padding = False
        for arr in arrays:
            pad_width = [(0, 0)] * ndim

            if arr.shape[-2] > target_lat or arr.shape[-1] > target_lon:
                raise ValueError(
                    f'Invalid spatial shape for {data_tag}: {arr.shape[-2:]} exceeds {(target_lat, target_lon)}'
                )

            pad_lat = target_lat - arr.shape[-2]
            pad_lon = target_lon - arr.shape[-1]
            pad_width[-2] = (0, pad_lat)
            pad_width[-1] = (0, pad_lon)

            if pad_lat > 0 or pad_lon > 0:
                needs_padding = True
                arr = np.pad(arr, pad_width=pad_width, mode=spatial_pad_mode)
            padded_arrays.append(arr)

        if needs_padding:
            print(f'[create_farm_batches] Padded spatial dims for {data_tag} to (17, 17) using {spatial_pad_mode}')
        return padded_arrays

    print(f'[create_farm_batches] {tag} for {rule_tag}: counting total rows (pass 1 of 2)...')
    n_rows_total = _count_rows(static_data_all, cv_mask_idx, data_types, model_layers, params, is_fuzzy)
    if n_rows_total == 0:
        print(f'[create_farm_batches] {tag} {rule_tag}: no rows found, aborting.')
        return 0
    print(f'[create_farm_batches] {tag} {rule_tag}: {n_rows_total} total rows across {n_farms} farms.')

    # --- global scalers: fit on train split, reuse for val/test ---
    if global_scalers:
        if tag == 'train' or not os.path.exists(scalers_path):
            global_scalers_ = _fit_global_scalers(static_data_all, cv_mask_idx, data_types, model_layers, params, is_fuzzy)
            joblib.dump(global_scalers_, scalers_path)
            print(f'[create_farm_batches] global scalers saved to {scalers_path}')
        else:
            global_scalers_ = joblib.load(scalers_path)
            print(f'[create_farm_batches] global scalers loaded from {scalers_path}')

    n_chunks = (n_farms + farms_per_chunk - 1) // farms_per_chunk
    for chunk_idx in tqdm(range(n_chunks)):
        farm_slice = static_data_all[chunk_idx * farms_per_chunk:(chunk_idx + 1) * farms_per_chunk]
        xs, ys, farm_ids, farm_metas = dict(), [], [], []

        for farm_offset, static_data in enumerate(farm_slice):
            farm_global_idx = chunk_idx * farms_per_chunk + farm_offset
            try:
                if rule_tag is not None:
                    cv_masks_file = os.path.join(static_data['path_model'], 'cluster_organizer', 'RBF', rule_tag,
                                                 'cv_mask.pickle')
                else:
                    cv_masks_file = os.path.join(static_data['path_data'], 'cv_mask.pickle')
                if not os.path.exists(cv_masks_file):
                    print(f'[create_farm_batches] Skipping farm {farm_global_idx}: no cv_mask.pickle')
                    continue
                cv_masks = joblib.load(cv_masks_file)
                dates_mask = cv_masks[cv_mask_idx]

                if rule_tag is not None:
                    cluster = {'cluster_name': rule_tag,
                               'cluster_path': os.path.join(static_data['path_model'], 'cluster_organizer', 'RBF', rule_tag)}
                else:
                    cluster = {'cluster_name': 'global',
                               'cluster_path': os.path.join(static_data['path_model'], 'global')}
                data_feeder = DataFeeder(static_data, online=False, train=True)
                data_feeder.reader_consumer = {'kind': 'farm', 'global_scalers': global_scalers}
                X, y, metadata, _, _ = load_data_deep_models(data_feeder, data_types, model_layers, params, cluster,
                                                             True,
                                                             is_fuzzy, return_f_center=False, inverse=inverse)
                x_farm, y_farm, mask = get_slice_for_nets(copy.deepcopy(X), metadata, dates=dates_mask, y=y)
                if 'lstm' in list(model_layers.keys()):
                    x_farm = x_farm['lstm']
                if schema is None:
                    schema = dict()
                    x_mm = dict()
                    for data_tag, data in x_farm.items():
                        if isinstance(data, pd.DataFrame):
                            columns = list(data.columns)
                            n_feat = len(columns)
                        else:
                            columns = None
                            n_feat = data.shape[1] if data.ndim > 1 else 1
                        if data.ndim == 3:
                            n_var = data.shape[2]
                            schema[data_tag] = {'columns': columns, 'n_features': n_feat, 'n_var': n_var}
                        elif data.ndim == 5:
                            n_var = data.shape[2]
                            lat = data.shape[3]
                            long = data.shape[4]
                            schema[data_tag] = {'columns': columns, 'n_features': n_feat, 'n_var': n_var, 'lats': lat, 'longs': long}
                        else:
                            n_var = None
                            schema[data_tag] = {'columns': columns, 'n_features': n_feat}
                        if data_tag not in x_mm:
                            x_mm[data_tag] = []
                        x_dat_path = os.path.join(path_batches, f'{tag}_x_{data_tag.replace("/", "_")}.dat')
                        if data.ndim < 3:
                            x_mm[data_tag] = np.memmap(x_dat_path, dtype='float32', mode='w+',
                                                   shape=(n_rows_total, n_feat))
                        elif data.ndim == 5:
                            x_mm[data_tag] = np.memmap(x_dat_path, dtype='float32', mode='w+',
                                                   shape=(n_rows_total, n_feat, n_var, lat, long))
                        else:
                            x_mm[data_tag] = np.memmap(x_dat_path, dtype='float32', mode='w+',
                                                   shape=(n_rows_total, n_feat, n_var))
                    target_columns = list(y.columns)
                    n_out = len(target_columns)
                    y_mm  = np.memmap(y_path,  dtype='float32', mode='w+', shape=(n_rows_total, n_out))
                    id_mm = np.memmap(id_path, dtype='int64',   mode='w+', shape=(n_rows_total,))
                    if farm_meta_dict is not None:
                        _sample_meta = next(iter(farm_meta_dict.values()))
                        meta_dim_local = len(_sample_meta)
                        meta_mm = np.memmap(farm_meta_path, dtype='float32', mode='w+',
                                            shape=(n_rows_total, meta_dim_local))

                for data_tag, data in x_farm.items():
                    if data_tag not in xs:
                        xs[data_tag] = []
                    arr = data if isinstance(data, TensorView) else data.values if isinstance(data, pd.DataFrame) else np.asarray(data)
                    if arr.ndim == 1:
                        arr = arr[:, np.newaxis]
                    if global_scalers:
                        if global_scalers_ is not None and data_tag in global_scalers_:
                            arr = _apply_global_scaler(arr, global_scalers_[data_tag])
                    xs[data_tag].append(arr.astype(np.float32))
                ys.append(y_farm)
                farm_ids.append(np.full(len(mask), farm_global_idx, dtype=np.int64))
                if farm_meta_dict is not None and farm_global_idx in farm_meta_dict:
                    meta_vec = farm_meta_dict[farm_global_idx]
                    farm_metas.append(np.tile(meta_vec, (len(mask), 1)).astype(np.float32))

            except Exception as e:
                print(f'[create_farm_batches] Error on farm {farm_global_idx}  {rule_tag}: {e}')
                continue

        if len(xs) == 0:
            continue

        x_chunk = dict()
        for data_tag in xs.keys():
            if any(isinstance(a, TensorView) for a in xs[data_tag]):
                continue  # gather directly into the existing shard after permutation
            arrays_to_concat = _pad_arrays_for_concat(xs[data_tag], data_tag)
            x_chunk[data_tag] = np.concatenate(arrays_to_concat, axis=0)
        y_chunk = np.concatenate(ys, axis=0)
        id_chunk = np.concatenate(farm_ids, axis=0)
        meta_chunk = np.concatenate(farm_metas, axis=0) if farm_metas else None

        perm = np.random.permutation(len(y_chunk))
        for data_tag in x_chunk.keys():
            x_chunk[data_tag] = x_chunk[data_tag][perm]
        y_chunk = y_chunk[perm]
        id_chunk = id_chunk[perm]
        if meta_chunk is not None:
            meta_chunk = meta_chunk[perm]

        n_chunk = len(y_chunk)
        for data_tag in x_mm.keys():
            if data_tag in x_chunk:
                x_mm[data_tag][row_offset:row_offset + n_chunk] = x_chunk[data_tag]
            else:
                from eforecast.datasets.tensor_views import write_shard
                write_shard(x_mm[data_tag], xs[data_tag], perm, offset=row_offset, pad_mode=spatial_pad_mode)
        y_mm[row_offset:row_offset + n_chunk]  = y_chunk
        id_mm[row_offset:row_offset + n_chunk] = id_chunk
        if meta_chunk is not None and meta_mm is not None:
            meta_mm[row_offset:row_offset + n_chunk] = meta_chunk
        for data_tag in x_mm.keys():
            x_mm[data_tag].flush()
        y_mm.flush()
        id_mm.flush()
        if meta_mm is not None:
            meta_mm.flush()
        row_offset += n_chunk

        print(f'[create_farm_batches] Chunk {rule_tag} {chunk_idx + 1}/{n_chunks}: {n_chunk} rows written '
              f'(total so far: {row_offset})')
        del x_chunk, y_chunk, id_chunk, xs, ys, farm_ids
        gc.collect()

    actual_rows = row_offset
    meta_dim_saved = 0
    if farm_meta_dict is not None and meta_mm is not None:
        meta_dim_saved = meta_mm.shape[1]
    schema.update({
        'n_rows': actual_rows,
        'n_out': n_out,
        'target_columns': target_columns,
        'batch_size': batch_size,
        'meta_dim': meta_dim_saved,
    })
    joblib.dump(schema, meta_path)
    print(f'[create_farm_batches] Done. {tag}  {rule_tag}: {actual_rows} rows, '
          f'{n_features} features, {n_out} outputs.')
    return actual_rows


def get_slice_for_nets(x_dict, metadata_dict, dates=None, y=None):
    for data_tag, metadata in metadata_dict.items():
        if dates is None:
            dates = metadata['dates']
        else:
            _, dates, _, _, _ = intersection_index(metadata['dates'], dates)
    if y is not None:
        _, dates, index, _, _ = intersection_index(dates, y.index)
        dates = get_time_level(dates)
        y_slice = y[~y.index.duplicated()].loc[dates].values.astype(np.float32)
    else:
        y_slice = None
    X_slice = dict()
    for data_tag, x in x_dict.items():
        X_slice[data_tag] = get_slice_with_dates(x, metadata_dict[data_tag]['dates'], dates)

    return X_slice, y_slice, dates
