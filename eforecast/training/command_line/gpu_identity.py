# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

"""Resolve CUDA ordinals through the driver, without creating a model/context in the scheduler."""
import ctypes
import uuid


def cuda_uuid(ordinal):
    """UUID of the device CUDA exposes at this ordinal, honoring visibility/order remapping."""
    driver = ctypes.CDLL('libcuda.so.1')
    initialize = driver.cuInit
    initialize.argtypes, initialize.restype = [ctypes.c_uint], ctypes.c_int
    get_device = driver.cuDeviceGet
    get_device.argtypes, get_device.restype = [ctypes.POINTER(ctypes.c_int), ctypes.c_int], ctypes.c_int
    get_uuid = getattr(driver, 'cuDeviceGetUuid_v2', None) or driver.cuDeviceGetUuid
    get_uuid.argtypes, get_uuid.restype = [ctypes.c_void_p, ctypes.c_int], ctypes.c_int
    device, raw = ctypes.c_int(), (ctypes.c_ubyte * 16)()
    for name, function, arguments in [('cuInit', initialize, (0,)),
                                      ('cuDeviceGet', get_device, (ctypes.byref(device), ordinal))]:
        result = function(*arguments)
        if result:
            raise RuntimeError(f'{name} failed with CUDA error {result}')
    result = get_uuid(ctypes.byref(raw), device)
    if result:
        raise RuntimeError(f'cuDeviceGetUuid failed with CUDA error {result}')
    return str(uuid.UUID(bytes=bytes(raw)))
