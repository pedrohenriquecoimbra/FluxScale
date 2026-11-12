"""The Python half of the FluxScale page.

The same functions run in two places: inside the browser, where the page loads
Python with Pyodide, and behind ``fluxscale gui``, where a local server calls
them. Either way the page gets what `fluxscale.process` gives in Python.

The main functions are:
- scan
    function: put a folder of EddyPro raw datasets in order and say what they hold
    call: scan(folder)
- run
    function: decompose the files and compute the chosen covariances
    call: run(folder, work, options)
- write_export
    function: write the last result as a file
    call: write_export(work, 'csv')
"""
# built-in modules
import datetime
import logging
import math
import os
import shutil
from collections import Counter

# 3rd party modules
import numpy as np
import pandas as pd
import xarray as xr

# project modules
from ..core import handlers
from ..core.wavelet_functions import formula_to_vars
from ..extra.partitioning.coimbra_et_al_2025 import _sign_classes
from ..io import READERS
from ..version import __version__

LABEL = '%Y%m%d-%H%M'       # what EddyPro starts the name of a raw dataset with
HEADER_LINE = 9             # line of the column names, counted from 0
READER = 'ep_raw_lvl'
PAGE_READER = 'ep_raw_lvl_page'
MIN_VALID = 0.7             # share of samples a variable needs, as nan_tolerance=.3 in data_run
METHODS = ('dwt', 'moda')   # the methods whose levels add up to the covariance
SIGNS = {1: '+', -1: '-'}
# The classes each partition of data_partition reads, by the gas that conditions w*co2
PARTITIONS = {
    'h2o': dict(suffix='', label='CO2 sampled by H2O', classes=['wco2-wh2o+', 'wco2-wh2o-']),
    'co': dict(suffix='_pCO', label='CO2 sampled by CO',
               classes=['wco2-wco+', 'wco2-wco-', 'wco2+wco+', 'wco2+wco-']),
    'ch4': dict(suffix='_pCH4', label='CO2 sampled by CH4',
                classes=['wco2-wch4+', 'wco2-wch4-', 'wco2+wch4+', 'wco2+wch4-']),
}

state = dict(spectra=None, fluxes=None)     # the last result, for write_export
_listener = dict(notify=None, done=0, total=0, errors=[])


def set_notify(notify):
    """Send progress to *notify(text, done, total)*; total is 0 when a step has no count."""
    _listener['notify'] = notify


def _notify(text, done=0, total=0):
    if _listener['notify'] is not None:
        _listener['notify'](text, int(done), int(total))


class _Progress(logging.Handler):
    """Turns what fluxscale logs into the page's status line, and keeps its errors."""

    def emit(self, record):
        message = record.getMessage()
        if record.levelno >= logging.ERROR:
            _listener['errors'].append(message.splitlines()[0][:300])
        elif message.startswith('cascade_run:') and ' done ' in message:
            _notify(f"Decomposed {message.split(chr(39))[1]}")
        elif message.startswith('Using the decomposition'):
            _notify('Using the decomposition already made')
        elif message.endswith(' reading'):
            _notify(f"Computing covariances of {message.split()[0]}")


_progress = _Progress(level=logging.INFO)


@READERS.register(PAGE_READER)
def _read_and_count(path, **kwargs):
    """The EddyPro reader, telling the page how many files were read."""
    data = READERS.get(READER)(path, **kwargs)
    _listener['done'] += 1
    _notify('Reading files', min(_listener['done'], _listener['total']), _listener['total'])
    return data


def _label(name):
    try:
        return datetime.datetime.strptime(name[:13], LABEL)
    except ValueError:
        return None


def _header(path):
    """Column names of a raw dataset and its number of samples, or a reason it is not one."""
    with open(path, 'r', errors='replace') as file:
        lines = file.readlines()
    if len(lines) <= HEADER_LINE + 1 or not lines[0].lower().startswith('raw dataset'):
        return None, 0, lines[0].strip()[:40] if lines else ''
    return lines[HEADER_LINE].split(), len(lines) - HEADER_LINE - 1, lines[0].strip()


def scan(folder):
    """
    function: put the raw datasets of a folder in time order, and say what the
        page needs to offer a run: the period, the gaps, the sampling rate and
        the variables that hold data
    call: scan(folder)
    Input:
        folder: folder of EddyPro raw datasets, named `YYYYMMDD-HHMM_...`
    Return:
        dict, ready for JSON. `files` are in time order; `rejected` says why a
            file was left out; `variables` are the columns with data
    """
    files, rejected = [], []
    seen = set()
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            continue
        label = _label(name)
        if label is None:
            rejected.append(dict(name=name, reason='its name does not start with a date, as 20220514-1200'))
        elif label in seen:
            rejected.append(dict(name=name, reason='another file has the same date'))
        else:
            seen.add(label)
            files.append(dict(name=name, label=label, size=os.path.getsize(path)))
    files.sort(key=lambda file: file['label'])
    if not files:
        return dict(files=[], rejected=rejected, variables=[], version=__version__)

    columns, samples, title = _header(os.path.join(folder, files[0]['name']))
    if columns is None:
        raise ValueError(f"{files[0]['name']} does not look like an EddyPro raw dataset: it starts with "
                         f"{title!r}, not 'Raw dataset level 6'.")

    labels = [file['label'] for file in files]
    steps = Counter(int((b - a).total_seconds() // 60) for a, b in zip(labels, labels[1:]))
    duration = min(steps.most_common(1)[0][0], 24 * 60) if steps else 30
    expected = int((labels[-1] - labels[0]).total_seconds() // 60 // duration) + 1
    fs = samples / (duration * 60)

    # one file from the middle says which columns hold data
    probe = READERS.get(READER)(os.path.join(folder, files[len(files) // 2]['name']))
    valid = {name: float(np.isfinite(probe[name].values).mean()) for name in probe.data_vars}
    variables = [name for name in columns if valid.get(name, 0) >= MIN_VALID]

    return dict(
        files=[dict(name=file['name'], label=file['label'].isoformat(), size=file['size']) for file in files],
        rejected=rejected, columns=columns, variables=variables, level=title,
        first=labels[0].isoformat(), last=labels[-1].isoformat(), fileduration=duration,
        missing=max(expected - len(files), 0), samples=samples,
        fs=round(fs) if abs(fs - round(fs)) < .05 * fs and round(fs) else round(fs, 3),
        version=__version__)


def _values(array):
    """A numpy array as nested lists, with None where JSON has no number."""
    array = np.asarray(array, dtype=float)
    if array.ndim > 1:
        return [_values(row) for row in array]
    return [float(v) if math.isfinite(v) else None for v in array]


def _series(data, name):
    return _values(data[name].values) if name in data else None


def _spectrum(data, name):
    if name not in data or 'natural_frequency' not in data[name].dims:
        return None
    return _values(data[name].transpose('TIMESTAMP', 'natural_frequency').values)


def _describe(spectra, fluxes, formulas):
    """What the page draws of a result: one entry per product, with the classes of its samplings."""
    products = {}
    for formula in formulas:
        parsed = formula_to_vars(formula)
        name = ''.join(parsed.xy)
        if name not in fluxes:
            continue
        product = products.setdefault(name, dict(
            name=name, formula='*'.join(parsed.xy), variables=parsed.xy,
            flux=_series(fluxes, name), cospectrum=_spectrum(spectra, name),
            covariance=_series(fluxes, f'cov_{name}'), filled=_series(fluxes, f'cov_{name}_filled'),
            low=_series(fluxes, '*'.join(parsed.xy) + '_lf'), samplings=[]))
        if not parsed.condsamp_pair:
            continue
        names = [name] + [''.join(pair) for pair in parsed.condsamp_pair]
        classes = sorted((cls for cls, _ in _sign_classes(names, SIGNS)), reverse=True)
        product['samplings'].append(dict(
            formula=formula, by=['*'.join(pair) for pair in parsed.condsamp_pair],
            classes=[dict(name=cls, flux=_series(fluxes, cls), cospectrum=_spectrum(spectra, cls))
                     for cls in classes if cls in fluxes]))
    return list(products.values())


def _partitions(spectra, fluxes):
    found = []
    for gas, partition in PARTITIONS.items():
        names = {key: key + partition['suffix'] for key in ('NEE', 'GPP', 'Reco')}
        if all(name in fluxes for name in names.values()):
            found.append(dict(
                gas=gas, label=partition['label'],
                parts=[dict(name=key, flux=_series(fluxes, name), cospectrum=_spectrum(spectra, name))
                       for key, name in names.items()]))
    return found


def run(folder, work, options):
    """
    function: decompose the raw datasets of a folder and compute covariances,
        conditional samplings and the partitions they allow, as
        `fluxscale.process` and `data_partition` do. The decomposition is kept
        in *work*, so that asking for other covariances of the same variables
        does not decompose again
    call: run(folder, work, options)
    Input:
        folder: folder of EddyPro raw datasets (c.f. scan)
        work: folder for the decomposition and the result
        options: dict with `formulas` (e.g. ['w*co2', 'w*co2|w*h2o']),
            `variables` to decompose, `fs` (Hz), `fileduration` (minutes),
            `first` and `last` (labels of the files, ISO), and optionally
            `method` ('dwt' or 'moda') and `integration_period` (seconds)
    Return:
        dict, ready for JSON: `time`, `frequency`, `products` (flux, cospectrum
            and the classes of each sampling), `partitions` (NEE, GPP and Reco,
            each with its flux and cospectrum)
    """
    formulas = list(dict.fromkeys(options['formulas']))
    if not formulas:
        raise ValueError('Choose at least one covariance.')
    method = options.get('method') or 'dwt'
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, not {method!r}.")
    fileduration = int(options['fileduration'])
    fs = float(options['fs'])
    fs = int(fs) if fs == int(fs) else fs
    integration_period = float(options.get('integration_period') or fileduration * 60)
    first = datetime.datetime.fromisoformat(options['first']) - datetime.timedelta(minutes=fileduration)
    last = datetime.datetime.fromisoformat(options['last'])
    # every variable is decomposed in the first run: its variance asks for it
    variables = list(dict.fromkeys(
        [v for f in formulas for v in formula_to_vars(f).uniquevars] + list(options.get('variables') or [])))
    asked = formulas + [f'{v}*{v}' for v in variables if f'{v}*{v}' not in formulas]

    # the decomposition is only good for the same files and settings
    key = repr((sorted(os.listdir(folder)), fs, fileduration, method))
    key_path = os.path.join(work, 'run.key')
    if os.path.exists(key_path):
        with open(key_path, encoding='utf-8') as file:
            if file.read() != key:
                shutil.rmtree(work, ignore_errors=True)
    shutil.rmtree(os.path.join(work, 'wavelet_full_cospectra'), ignore_errors=True)
    os.makedirs(work, exist_ok=True)
    with open(key_path, 'w', encoding='utf-8') as file:
        file.write(key)

    _listener.update(done=0, total=len([n for n in os.listdir(folder) if _label(n)]), errors=[])
    logger = logging.getLogger('fluxscale')
    level = logger.level
    logger.addHandler(_progress)
    logger.setLevel(logging.INFO)
    try:
        handlers.process(
            datetimerange=f"{first:%Y%m%d%H%M}-{last:%Y%m%d%H%M}", fileduration=fileduration,
            input_path=folder, acquisition_frequency=fs, covariance=asked, output_folderpath=work,
            processing_time_duration='1D', reader_method=PAGE_READER, method=method,
            logging_kwargs=dict(level=logging.WARNING))
        _notify('Putting the result together')
        try:
            spectra = handlers.open_files_in_folder(
                os.path.join(work, 'wavelet_full_cospectra'), load=True).sortby('TIMESTAMP')
        except OSError:
            reason = _listener['errors'][-1] if _listener['errors'] else 'no period could be computed'
            raise RuntimeError(f"FluxScale gave no result: {reason}") from None
    finally:
        logger.removeHandler(_progress)
        logger.setLevel(level)

    spectra = handlers.data_average_dims(spectra)
    # the partitions are sums of classes, so they hold scale by scale too
    gases = [gas for gas, partition in PARTITIONS.items() if all(c in spectra for c in partition['classes'])]
    if gases:
        spectra = handlers.data_partition(spectra, variables_available=['w', 'co2'] + gases)
    fluxes = handlers.data_integrate_in_frequency(spectra, 1 / integration_period)
    state.update(spectra=spectra, fluxes=fluxes)

    return dict(
        time=[str(t)[:16] for t in spectra.TIMESTAMP.values.astype('datetime64[m]')],
        frequency=_values(spectra.natural_frequency.values),
        products=_describe(spectra, fluxes, asked), partitions=_partitions(spectra, fluxes),
        method=method, rule=spectra.attrs.get(handlers.FLUX_RULE), fs=fs, fileduration=fileduration,
        integration_period=integration_period, variables=variables, formulas=formulas,
        warnings=list(dict.fromkeys(_listener['errors']))[:5], version=__version__)


def write_export(work, kind):
    """Write the last result of run in *work* and return the path: 'csv' for the
    fluxes of each window, 'netcdf' for the same with their cospectra."""
    if state['fluxes'] is None:
        raise RuntimeError('Nothing to export yet: run first.')
    if kind == 'csv':
        path = os.path.join(work, 'fluxscale_fluxes.csv')
        table = state['fluxes'].drop_vars('natural_frequency', errors='ignore').to_dataframe()
        table.index = pd.DatetimeIndex(table.index).strftime('%Y-%m-%d %H:%M')
        table.to_csv(path, index_label='TIMESTAMP', float_format='%.8g')
    elif kind == 'netcdf':
        path = os.path.join(work, 'fluxscale_cospectra.nc')
        spectra = state['spectra']
        with_frequency = [v for v in spectra.data_vars if 'natural_frequency' in spectra[v].dims]
        merged = xr.merge([state['fluxes'], spectra[with_frequency].rename(
            {v: f'csp_{v}' for v in with_frequency})], compat='override', combine_attrs='override')
        if os.path.exists(path):
            os.remove(path)
        merged.to_netcdf(path)
    else:
        raise ValueError(f"unknown export {kind}")
    return path
