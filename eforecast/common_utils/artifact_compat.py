# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Read retained pickles whose writer used a newer pandas than this interpreter.

pandas 3 pickles an ``NDArrayBacked`` subclass with a two-element state
``(dtype, values)``; pandas 2 expects ``(dtype, values, attributes)`` and raises
``NotImplementedError`` on the shorter form. Both versions represent the values
themselves identically -- pandas 2.0 already supports non-nanosecond datetime units --
so the only incompatibility is the missing third element.

This module adds exactly that element, for exactly that shape, and nothing else. It
never converts a dtype, resamples a timestamp, rewrites a file or relaxes any other
pickle check, and it is inert when the running pandas already understands the state.
A file that fails for any other reason still fails, with its original error.
"""
import pickle

INCOMPATIBLE = 'retained artifact was written by a newer pandas than this interpreter'


def _backed():
    try:
        from pandas._libs.arrays import NDArrayBacked
    except Exception:                                           # noqa: BLE001 - no pandas, no shim
        return None
    return NDArrayBacked


_SHIM_NEEDED = None


def _needs_shim():
    """True when this pandas cannot load the two-element state a newer pandas writes.

    Answered by actually attempting such a load once, rather than by inferring it from
    a version number or from what this pandas happens to write.
    """
    global _SHIM_NEEDED
    if _SHIM_NEEDED is not None:
        return _SHIM_NEEDED
    _SHIM_NEEDED = False
    if _backed() is None:
        return False
    try:
        import io
        import numpy as np
        import pandas as pd
        frame = pd.DataFrame({'v': [1.0]}, index=pd.DatetimeIndex(
            np.array(['2024-01-01T00:00:00.000000'], dtype='datetime64[us]')))

        class Writer(pickle.Pickler):
            def reducer_override(self, obj):
                if isinstance(obj, pd.arrays.DatetimeArray):
                    reduced = obj.__reduce__()
                    if (isinstance(reduced, tuple) and len(reduced) == 3
                            and isinstance(reduced[2], tuple) and len(reduced[2]) == 3):
                        return reduced[0], reduced[1], (reduced[2][0], reduced[2][1])
                return NotImplemented

        buffer = io.BytesIO()
        Writer(buffer, protocol=5).dump(frame)
        try:
            pickle.loads(buffer.getvalue())
        except NotImplementedError:
            _SHIM_NEEDED = True
    except Exception:                                           # noqa: BLE001
        _SHIM_NEEDED = False
    return _SHIM_NEEDED


def unpickler(path, handle):
    """A joblib unpickler that accepts the newer two-element array state."""
    from joblib.numpy_pickle import NumpyUnpickler
    backed = _backed()
    normalized = []

    class Compatible(NumpyUnpickler):
        dispatch = dict(NumpyUnpickler.dispatch)

        def load_build(self):
            state = self.stack[-1]
            instance = self.stack[-2] if len(self.stack) > 1 else None
            if (backed is not None and isinstance(instance, backed)
                    and isinstance(state, tuple) and len(state) == 2):
                # The missing element is the instance attribute mapping, which the
                # newer writer stores as empty. Nothing else about the state changes.
                normalized.append(type(instance).__name__)
                self.stack[-1] = (state[0], state[1], {})
            return NumpyUnpickler.load_build(self)

        dispatch[pickle.BUILD[0]] = load_build

    try:
        reader = Compatible(path, handle, ensure_native_byte_order=True)
    except TypeError:                                           # older joblib signature
        reader = Compatible(path, handle)
    return reader, normalized


def load(path, *, report=None):
    """``joblib.load`` first; the bounded compatibility read only if that fails.

    ``report`` receives {'shim_used': bool, 'normalized': [...]} when supplied, so a
    caller can record that an artifact needed the compatibility read.

    The first attempt deliberately uses the *unwrapped* loader when ``install`` has already
    replaced ``joblib.load`` in this process.  Otherwise the wrapper's own fallback would
    satisfy the read, this function would never see ``NotImplementedError``, and an artifact
    that genuinely needs the compatibility path would be reported as read natively -- a false
    negative in exactly the inventory a reader consults to decide whether it needs the shim.
    """
    import joblib
    plain = getattr(joblib.load, '_eforecast_original', joblib.load)
    try:
        value = plain(path)
        if report is not None:
            report.update(shim_used=False, normalized=[])
        return value
    except NotImplementedError as error:
        if not _needs_shim():
            raise
        with open(path, 'rb') as handle:
            reader, normalized = unpickler(str(path), handle)
            try:
                value = reader.load()
            except Exception:                                   # noqa: BLE001
                raise error from None
        if report is not None:
            report.update(shim_used=True, normalized=normalized)
        return value


def install():
    """Make ``joblib.load`` fall back to the bounded compatibility read.

    Idempotent, and a strict no-op unless the ordinary load raises
    ``NotImplementedError`` -- the error a newer pandas' array state produces here.
    Any other failure, and any successful load, is passed through untouched.
    """
    import joblib
    if getattr(joblib.load, '_eforecast_compat', False):
        return joblib.load
    original = joblib.load

    def load_with_fallback(filename, *args, **kwargs):
        try:
            return original(filename, *args, **kwargs)
        except NotImplementedError as error:
            if not _needs_shim() or not isinstance(filename, (str, bytes)) and not hasattr(filename, '__fspath__'):
                raise
            with open(filename, 'rb') as handle:
                reader, _ = unpickler(str(filename), handle)
                try:
                    return reader.load()
                except Exception:                               # noqa: BLE001
                    raise error from None

    load_with_fallback._eforecast_compat = True
    load_with_fallback._eforecast_original = original
    joblib.load = load_with_fallback
    return load_with_fallback


def inspect(path):
    """Report how one retained artifact loads here, without changing it."""
    import warnings
    record = {'path': str(path)}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        report = {}
        try:
            load(path, report=report)
            record['status'] = 'compatibility_read' if report.get('shim_used') else 'native'
            record['normalized_states'] = report.get('normalized') or []
        except Exception as error:                              # noqa: BLE001
            record['status'] = 'unreadable'
            record['error'] = f'{type(error).__name__}: {str(error)[:200]}'
        versions = sorted({str(w.message)[:200] for w in caught
                           if 'InconsistentVersionWarning' in type(w.message).__name__
                           or 'version' in str(w.message).lower() and 'unpickle' in str(w.message).lower()})
        if versions:
            record['library_version_warnings'] = versions
    return record


def survey(root, *, limit=2000):
    """Every retained pickle under a tree, and how this interpreter reads it."""
    import sys
    from pathlib import Path
    root = Path(root)
    paths = sorted(p for p in root.rglob('*.pickle') if p.is_file())[:limit]
    records = [inspect(p) for p in paths]
    counts = {}
    for record in records:
        counts[record['status']] = counts.get(record['status'], 0) + 1
    import pandas
    return {'root': str(root), 'checked': len(records), 'counts': counts,
            'reader': {'python': sys.version.split()[0], 'pandas': pandas.__version__},
            'needs_compatibility_read': [r['path'] for r in records if r['status'] == 'compatibility_read'],
            'unreadable': [r for r in records if r['status'] == 'unreadable'],
            'library_version_warnings': {r['path']: r['library_version_warnings']
                                         for r in records if r.get('library_version_warnings')},
            'records': records}


def main():
    import argparse
    import json
    parser = argparse.ArgumentParser(description='Check whether this interpreter can read retained artifacts.')
    parser.add_argument('root')
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    value = survey(args.root)
    if args.json:
        print(json.dumps(value, indent=1, sort_keys=True))
        return 0
    print(f"reader: python {value['reader']['python']}, pandas {value['reader']['pandas']}")
    print(f"checked {value['checked']} retained pickles under {value['root']}")
    for status, count in sorted(value['counts'].items()):
        print(f'  {status}: {count}')
    for path in value['needs_compatibility_read']:
        print('  compatibility read required:', path)
    for record in value['unreadable']:
        print('  UNREADABLE:', record['path'], record['error'])
    for path, warnings_ in value['library_version_warnings'].items():
        print('  library version warning:', path)
        for text in warnings_:
            print('     ', text[:160])
    return 1 if value['unreadable'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
