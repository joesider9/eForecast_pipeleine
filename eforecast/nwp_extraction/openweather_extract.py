# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

import requests

import os
import datetime
import joblib
import pandas as pd

from eforecast.common_utils.date_utils import convert_timezone_dates

file_cred = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'openweather_key.json')


def weather_key():
    """Reload the authoritative source each time; preserve legacy files without rewriting."""
    from eforecast.config.secrets_source import resolve, SecretMissing, path_of
    try:
        return resolve('openweather', environment='OPENWEATHER_API_KEY').reveal()
    except SecretMissing:
        if path_of('openweather').exists() or not os.path.isfile(file_cred):
            raise
    import warnings
    from credentials import Credentials, JsonFileBackend
    warnings.warn('Using the legacy OpenWeather credential file; move the rotated key to the '
                  'openweather secret source. The stored file was not changed.', UserWarning, stacklevel=2)
    return Credentials([JsonFileBackend(file_cred)]).load('cred2')


class OpenWeatherDownloader:

    def __init__(self, path_nwp, date=None):
        self.path_nwp = path_nwp
        if date is None:
            self.date = pd.to_datetime(datetime.datetime.now().strftime('%d%m%y'), format='%d%m%y')
        else:
            self.date = date
        self.dates = pd.date_range(self.date, self.date + pd.DateOffset(hours=42), freq='h')
        self.dates = pd.DatetimeIndex(convert_timezone_dates(self.dates))
        self.lat1, self.long1 = 38.04, 24.35
        self.lat2, self.long2 = 38.06, 24.34
        fname = 'openweather_' + self.date.strftime('%d%m%y') + '.pickle'
        self.filename = os.path.join(path_nwp, fname)

    def download(self):
        key = weather_key()
        url1 = f'https://api.openweathermap.org/data/2.5/onecall?lat={self.lat1}&lon={self.long1}&exclude=daily,minutely' \
               f',current,alerts&appid={key}'
        url2 = f'https://api.openweathermap.org/data/2.5/onecall?lat={self.lat2}&lon={self.long2}&exclude=daily,minutely' \
               f',current,alerts&units=metric&appid={key}'

        try:
            response1 = requests.get(url1, timeout=20)
            response2 = requests.get(url2, timeout=20)
        except requests.RequestException:
            # Requests exceptions can contain the full URL, including appid.
            raise ConnectionError('OpenWeather request failed; check provider access and connectivity') from None
        if response1.status_code == 200 and response2.status_code == 200:
            try:
                nwp1 = response1.json()["hourly"]
                nwp2 = response2.json()["hourly"]
            except:
                raise ValueError('Openweather nwps are not downloaded correctly')
        else:
            raise ConnectionError('Openweather is not respond')
        for nwp in nwp1:
            del nwp['weather']
        for nwp in nwp2:
            del nwp['weather']
        nwp1 = pd.DataFrame().from_dict(nwp1)
        nwp2 = pd.DataFrame().from_dict(nwp2)
        nwp1.dt = pd.to_datetime(nwp1.dt, unit='s')
        nwp2.dt = pd.to_datetime(nwp2.dt, unit='s')
        nwp1 = nwp1.set_index('dt')
        nwp2 = nwp2.set_index('dt')
        columns = ['wind_speed', 'wind_deg']

        nwps = dict()
        for date in nwp1.index:
            if date.strftime('%d%m%y%H%M') not in nwps.keys():
                nwps[date.strftime('%d%m%y%H%M')] = dict()
            nwps[date.strftime('%d%m%y%H%M')]['marmari1'] = nwp1.loc[date, columns].astype(float).to_dict()
            nwps[date.strftime('%d%m%y%H%M')]['marmari2'] = nwp2.loc[date, columns].astype(float).to_dict()

        joblib.dump(nwps, self.filename)
