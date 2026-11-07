"""
This script is a key part of the following publications:
    - Herig Coimbra, Pedro Henrique and Loubet, Benjamin and Laurent, Olivier and Mauder, Matthias and Heinesch, Bernard and 
    Bitton, Jonathan and Delpierre, Nicolas and Depuydt, Jérémie and Buysse, Pauline, Improvement of Co2 Flux Quality Through 
    Wavelet-Based Eddy Covariance: A New Method for Partitioning Respiration and Photosynthesis. 
    Available at SSRN: https://ssrn.com/abstract=4642939 or http://dx.doi.org/10.2139/ssrn.4642939

The main function is:  
- run_wt
    function: (1) gets data, (2) performs wavelet transform, (3) cross calculate variables, (4) averages by 30 minutes, (5) saves 
    call: run_wt()
    Input:
        a: 
    Return:
        b: 

- conditional_sampling
    function: split an array (n dimensions) into 4 arrays based on signal (+ or -) of itself and 2nd array 
    call: conditional_sampling()
    Input:
        args: arrays to be used as filter 
    Return:
        b: 

- universal_wt
    function: call any wavelet transform
    call: universal_wt()
    Input:
        a: 
    Return:
        b: 
"""

# built-in modules
import os
import re
import logging
import logging.handlers
import queue
import warnings
import time
import datetime
import glob
import shutil
import tempfile

# 3rd party modules
import dask
from dask.utils import parse_bytes
import joblib
import numpy as np
import xarray as xr

# project modules
from . import cascade, commons, grid
from ..io import READERS
# from .read_data import loaddatawithbuffer
from .wavelet_functions import universal_wt, formula_to_vars, prepare_signal
from ..extra.partitioning.coimbra_et_al_2025 import conditional_sampling, conditional_sampling_mean, partition_DWCS, partition_DWCS_CO, partition_DWCS_H2O
from ..extra import eddypro as eddypro


logger = logging.getLogger(__name__)

__all__ = [
    # decomposition & covariance on xarray.Dataset
    "decompose_variables",
    "data_statistics",
    "data_compute_product",
    "data_compute_flux",
    "data_conditional_sampling",
    "data_conditional_sampling_mean",
    "data_partition",
    "data_average_dims",
    "data_integrate_in_frequency",
    "open_files_in_folder",
    # run entry points
    "process",
    "data_run",
    "run_from_eddypro",
    # re-exported analysis API (kept public for backwards compatibility)
    "universal_wt",
    "formula_to_vars",
    "prepare_signal",
    "conditional_sampling",
    "partition_DWCS",
    "partition_DWCS_CO",
    "partition_DWCS_H2O",
]


# attrs[FLUX_RULE] says how a result splits a flux over the levels: CROSS_LEVEL
# or SAME_SCALE (c.f. data_compute_flux). A result without it is SAME_SCALE.
FLUX_RULE = 'flux_rule'
CROSS_LEVEL = 'cross_level'
SAME_SCALE = 'same_scale'
# Methods whose levels add up to the signal, for the cross-level rule
CROSS_LEVEL_METHODS = ('dwt', 'moda')
# Suffix of the gap-filled signal that decompose_variables keeps for the flux.
# It is a helper: _run_decomposed takes it out, and it never reaches an output.
SIGNAL = '_signal'


def _window_dims(data):
    """The dims of an averaging window: all but TIMESTAMP and natural_frequency."""
    return [d for d in data.dims if d not in {'TIMESTAMP', 'natural_frequency'}]


def decompose_variables(data, variables=None,
                        nan_tolerance=.3, keep_signal=False, **kwargs):
    """
    Calculate data decomposed with wavelet transform for xarray.Dataset.

    Parameters:
    - data: xarray.Dataset
    - variables: list of variable names to decompose (defaults to ['w', 'co2'])
    - nan_tolerance: tolerance for NaN values
    - keep_signal: also keep the gap-filled signal of each variable, as
      <var>_signal, for data_compute_flux
    - **kwargs: additional arguments for universal_wt

    Returns:
    - xarray.Dataset with decomposed variables
    """
    if variables is None:
        variables = ['w', 'co2']
    # Initialize output dataset
    result = xr.Dataset()

    # Add the original coordinates
    result = result.assign_coords(data.coords)

    # # Placeholder for wavelet scales and COI
    # sj = None
    # coi = None

    try:
        for var in variables:
            if var not in data.data_vars:
                raise ValueError(f"Variable {var} not found in dataset.")
            
            # Prepare signal (assuming prepare_signal is adapted for xarray)
            ready_signal = prepare_signal(
                data[var], nan_tolerance=nan_tolerance)
            
            # Perform wavelet transform
            wt_signal = universal_wt(
                signal=ready_signal.signal, **kwargs, iwt=True)
            
            # Store results
            wt_signal = wt_signal.rename(
                {'wave': var,
                 'approximation': f'{var}_lf',
                 'coi': f'{var}_qc'}
            )
            if keep_signal:
                wt_signal = wt_signal.rename({'signal': f'{var}{SIGNAL}'})
            else:
                wt_signal = wt_signal.drop_vars('signal')
            wt_signal[f'{var}_qc'] = wt_signal[f'{var}_qc'].where(
                ready_signal.signan != 0, 0)
            result = xr.merge([wt_signal, result], compat='override')

        # # Add COI as a variable
        # result['coi'] = wt_signal.coi

    except Exception as e:
        logger.error(f"Error in decompose_variables: {e}")
        raise

    return result


def data_statistics(data, formula='w*co2'):
    formulavar = formula_to_vars(formula) if isinstance(
        formula, str) else formula
    xy_name = ''.join(formulavar.xy)

    if len(formulavar.xy) == 2:
        window = _window_dims(data)
        data[f'cov_{xy_name}'] = xr.cov(
            data[formulavar.xy[0]], data[formulavar.xy[1]], dim=window)

        data[f'adv_{xy_name}'] = (
            data[formulavar.xy[0]].mean(dim=window) *
            data[formulavar.xy[1]].mean(dim=window))
    return data


def data_compute_product(data, formula='w*co2|w*h2o', name=None):
    """
    Calculate the product of variables specified by a formula for xarray.Dataset.

    The product is taken level by level (same-scale): it gives the signs that
    set the classes of conditional sampling. A name already in *data* is kept
    as it is.

    Parameters:
    - data: xarray.Dataset
    - formula: string or object specifying the formula (e.g., 'w*co2|w*h2o')

    Returns:
    - xarray.Dataset with the calculated products
    """
    formulavar = formula_to_vars(formula) if isinstance(
        formula, str) else formula
    xy_name = name or ''.join(formulavar.xy)

    # Calculate the product for the main variables
    if xy_name not in data.data_vars:
        XY = data[formulavar.xy[0]]
        for c in formulavar.xy[1:]:
            XY = XY * data[c].conj()
        data[xy_name] = XY

    # Calculate the product for conditional sampling pairs
    for cs in formulavar.condsamp_pair:
        cs_name = ''.join(cs)
        if cs_name not in data.data_vars:
            CS = data[cs[0]]
            for c in cs[1:]:
                CS = CS * data[c].conj()
            data[cs_name] = CS

    # Convert the dictionary to an xarray.Dataset
    return data


def _anomaly(values, window):
    return values - values.mean(window)


def _flux_density(parts, whole, window, average):
    """
    (1/n) sum_k part_k' * prod_{i != k} whole_i', with conj on positions >= 1;
    *whole* is already in anomalies and conjugated. With *average*, the mean
    over *window*, one natural_frequency at a time.
    """
    if average and parts[0].sizes.get('natural_frequency', 1) > 1:
        return xr.concat(
            [_flux_density([p.isel(natural_frequency=[i]) for p in parts],
                           whole, window, True)
             for i in range(parts[0].sizes['natural_frequency'])],
            dim='natural_frequency')
    total = None
    for k, part in enumerate(parts):
        term = _anomaly(part, window)
        if k and np.iscomplexobj(term):
            term = term.conj()
        for i, other in enumerate(whole):
            if i != k:
                term = term * other
        total = term if total is None else total + term
    total = total / len(parts)
    return total.mean(window) if average else total


def data_compute_flux(data, formula='w*co2', average=False, levels=True, low=True):
    """
    Cross-level flux of a product of n >= 2 variables, level by level.

    With x_1..x_n the gap-filled signals, L_kj the level j of x_k (levels and
    approximation add up to the signal) and a' = a - mean(a) over the window,
    the density of level j at each sample is

        d_j = (1/n) sum_k L_kj' * prod_{i != k} x_i'

    with conj on the factors at positions >= 1, as in data_compute_product.
    For two variables, the window mean of d_j is
    (cov(x_j, y) + cov(x, y_j)) / 2. The densities of all levels and of the
    approximation add up to prod x_i', sample by sample, so their window
    means add up to cov_<xy>_filled, the covariance of the gap-filled signals
    (ddof=0). A sum over part of the levels is not the covariance of a
    band-passed signal.

    Parameters:
    - data: xarray.Dataset with the levels <c> (natural_frequency, ...), the
      approximations <c>_lf and the gap-filled signals <c>_signal, with a
      window dim besides TIMESTAMP (e.g. 'ns')
    - formula: string or object specifying the formula (e.g., 'w*co2'); only
      its first product is used
    - average: return window means instead of densities
    - levels: compute <xy> from the levels
    - low: compute <x*y>_lf from the approximations, and cov_<xy>_filled

    Returns:
    - xarray.Dataset with <xy>, <x*y>_lf and cov_<xy>_filled (always a window
      mean); empty if the formula has fewer than 2 variables

    Raises:
    - ValueError: if *data* has no window dim
    - KeyError: if a <c>_signal is missing
    """
    formulavar = formula_to_vars(formula) if isinstance(formula, str) else formula
    xy = formulavar.xy
    out = xr.Dataset()
    if len(xy) < 2:
        return out
    window = _window_dims(data)
    if not window:
        raise ValueError("data_compute_flux needs a window dim besides "
                         "TIMESTAMP and natural_frequency.")
    whole = []
    for i, c in enumerate(xy):
        if f'{c}{SIGNAL}' not in data:
            raise KeyError(f"data_compute_flux needs {c}{SIGNAL}, the "
                           f"gap-filled signal of {c}.")
        anomaly = _anomaly(data[f'{c}{SIGNAL}'], window)
        whole.append(anomaly.conj() if i and np.iscomplexobj(anomaly) else anomaly)
    name = ''.join(xy)
    if levels:
        out[name] = _flux_density([data[c] for c in xy], whole, window, average)
    if low:
        out['*'.join(xy) + '_lf'] = _flux_density(
            [data[f'{c}_lf'] for c in xy], whole, window, average)
        product = whole[0]
        for other in whole[1:]:
            product = product * other
        out[f'cov_{name}_filled'] = product.mean(window)
    return out


def _condsamp_names(formula):
    """Names of the variables of a formula: the one sampled, then its conditions."""
    formulavar = formula_to_vars(formula) if isinstance(formula, str) else formula
    return [''.join(formulavar.xy)] + [''.join(cs)
                                       for cs in formulavar.condsamp_pair]


def data_conditional_sampling(data, formula='w*co2|w*h2o', values=None):
    """
    Calculate conditional sampling from a formula for xarray.Dataset.

    The class of a sample comes from the signs of the products of the formula
    (c.f. data_compute_product). The values put in the classes are the
    product itself, or *values*.

    Parameters:
    - data: xarray.Dataset
    - formula: string or object specifying the formula (e.g., 'w*co2|w*h2o')
    - values: optional dict of the sampled name (e.g. 'wco2') to the values to
      put in its classes, such as the flux density of data_compute_flux

    Returns:
    - xarray.Dataset with the conditional sampling results
    """
    # Generate names for variables
    names = _condsamp_names(formula)

    # If name in names not in data, calculate it
    for n in names:
        data = data_compute_product(data, formula=n)

    # Extract data arrays for conditional sampling
    data_arrays = [data[names[0]]] + [data[n] for n in names[1:]]
    sampled = (values or {}).get(names[0], data_arrays[0])

    # Perform conditional sampling
    φc = conditional_sampling(
        sampled, *[da for da in data_arrays[:]], names=names, label={1: "+", -1: "-"}
    ) if names else {}

    return φc


def data_conditional_sampling_mean(data, formulas, dim, values=None):
    """
    Calculate the mean of the conditional sampling of each formula for xarray.Dataset.

    Gives the mean of what data_conditional_sampling returns, without building its
    full-size arrays (see conditional_sampling_mean).

    Parameters:
    - data: xarray.Dataset holding the products named in the formulas
    - formulas: list of strings or objects specifying the formulas (e.g., ['w*co2|w*h2o'])
    - dim: dimension or list of dimensions to average over
    - values: optional dict of a sampled name to the values to put in its
      classes (c.f. data_conditional_sampling)

    Returns:
    - xarray.Dataset with the averaged conditional sampling results
    """
    return conditional_sampling_mean(
        data, [_condsamp_names(f) for f in formulas], dim, values=values)


def data_partition(data, dst=None,
                   id_columns=None,
                   variables_available=None, **kwargs):
    """Apply every applicable wavelet conditional-sampling partition to *data*.

    NEE is ``wco2``. With the cross-level rule (attrs['flux_rule']), the classes
    hold the flux density of the same samples, so they add up to NEE but for
    the samples where a sign is zero, and Reco = NEE - GPP takes that rest.
    With 'moda' these are the gap-filled samples, so in a gappy window the
    rest, and so Reco, can take a large share of NEE.

    Each gas-channel partition is attempted independently ("best effort"): a
    channel whose required variables are absent is skipped and logged at DEBUG,
    while a channel that raises is logged with a full traceback at ERROR and
    skipped, so the remaining channels still run.

    Args:
        data: xarray.Dataset of conditionally-sampled covariances.
        dst: Optional path stem; each channel writes ``f"{dst}.<tag>"`` if set.
        id_columns: Reserved; currently informational only.
        variables_available: Variables present in the source data. Defaults to
            ``['u', 'v', 'w', 'ts', 'co2', 'h2o']`` when ``None``.
        **kwargs: Ignored; accepted for forward compatibility.

    Returns:
        xarray.Dataset: *data* merged with the partition outputs. Channels other
        than the primary H2O one are suffixed (``_pH2O_CO``, ``_pCO``,
        ``_pCH4``) so their NEE/GPP/Reco variables don't collide on merge.
    """
    if variables_available is None:
        variables_available = ['u', 'v', 'w', 'ts', 'co2', 'h2o']
    id_columns = id_columns or ['TIMESTAMP'] + \
        [c for c in ['natural_frequency'] if c in data]

    def _run(label, required, fn, columns, suffix, tag):
        missing = [v for v in required if v not in variables_available]
        if missing:
            logger.debug("Skipping %s: missing variables %s.",
                         label, ', '.join(missing))
            return xr.Dataset()
        try:
            out = fn()[columns]
            if suffix:
                out = out.rename(
                    {var: f"{var}{suffix}" for var in out.data_vars})
            if dst:
                out.to_netcdf(f"{dst}.{tag}")
            return out
        except Exception:
            logger.exception("Partition %s failed; skipping.", label)
            return xr.Dataset()

    ds_pH2O = _run(
        "DWCS_H2O", ['w', 'co2', 'h2o'],
        lambda: partition_DWCS_H2O(
            data, NEE='NEE', GPP='GPP', Reco='Reco', CO2='wco2',
            CO2neg_H2Opos='wco2-wh2o+', CO2neg_H2Oneg='wco2-wh2o-', NIGHT=None),
        ['NEE', 'GPP', 'Reco'], '', "FCO2_condH2O")

    ds_pH2O_CO = _run(
        "DWCS_H2O_CO", ['w', 'co2', 'h2o', 'co'],
        lambda: partition_DWCS_CO(
            data, NEE='NEE', GPP='GPP', Reco='Reco', ffCO2='ffCO2', CO2='wco2',
            CO2neg_H2Opos='wco2-wh2o+', CO2neg_H2Oneg='wco2-wh2o-',
            CO2pos_COpos='wco2+wco+', CO2pos_COneg='wco2+wco-', NIGHT=None),
        ['NEE', 'GPP', 'Reco', 'ffCO2'], '_pH2O_CO', "FCO2_condH2O_CO")

    ds_pCO = _run(
        "DWCS_CO", ['w', 'co2', 'co'],
        lambda: partition_DWCS_CO(
            data, NEE='NEE', GPP='GPP', Reco='Reco', ffCO2='ffCO2', CO2='wco2',
            CO2neg_H2Opos=['wco2-wco+', 'wco2-wco-'], CO2neg_H2Oneg=None,
            CO2pos_COpos='wco2+wco+', CO2pos_COneg='wco2+wco-', NIGHT=None),
        ['NEE', 'GPP', 'Reco', 'ffCO2'], '_pCO', "FCO2_condCO")

    ds_pCH4 = _run(
        "DWCS_CH4", ['w', 'co2', 'ch4'],
        lambda: partition_DWCS_CO(
            data, NEE='NEE', GPP='GPP', Reco='Reco', ffCO2='ffCO2', CO2='wco2',
            CO2neg_H2Opos=['wco2-wch4+', 'wco2-wch4-'], CO2neg_H2Oneg=None,
            CO2pos_COpos='wco2+wch4+', CO2pos_COneg='wco2+wch4-', NIGHT=None),
        ['NEE', 'GPP', 'Reco', 'ffCO2'], '_pCH4', "FCO2_condCH4")

    return xr.merge([data, ds_pH2O, ds_pH2O_CO, ds_pCO, ds_pCH4], compat='override')


# Folders whose files add up to more than this are opened lazily by default.
LOAD_LIMIT = 2**30


def open_files_in_folder(path, load=None, chunks=None):
    """Open the ``*.nc`` files in *path* as one Dataset along ``TIMESTAMP``.

    Files are concatenated in sorted filename order with the merge rules of
    ``xr.open_mfdataset(..., combine='nested', concat_dim='TIMESTAMP')``,
    pinned so the result does not depend on xarray's changing defaults.

    Args:
        path: Folder of per-period NetCDF outputs, e.g.
            ``<output>/wavelet_full_cospectra``.
        load: If True, read each file into memory, then concatenate. For many
            small files this is much faster than a lazy open, which builds one
            dask chunk per file and variable, but peak RAM is a few times the
            size of the files. If False, open lazily with dask, for folders
            that do not fit in memory (slow with thousands of files); peak RAM
            then follows the chunk size and the number of dask threads, not
            the size of the folder. If None (default), read into memory unless
            the files add up to more than ``LOAD_LIMIT`` bytes.
        chunks: Dask chunks for the result, e.g. ``{'TIMESTAMP': 4096}``. If
            None, the result is in memory or, when it is opened lazily, has
            chunks of about dask's ``array.chunk-size`` along ``TIMESTAMP``.

    Returns:
        xarray.Dataset

    Raises:
        OSError: If *path* has no ``*.nc`` file.
        ValueError: If the files were made with different flux rules
            (attrs['flux_rule'], missing meaning 'same_scale'), whose numbers
            cannot be put together.
    """
    files = sorted(glob.glob(os.path.join(
        os.path.abspath(os.path.expanduser(path)), '*.nc')))
    if not files:
        raise OSError(f"no files to open in {path!r}")
    combine_kwargs = dict(
        concat_dim='TIMESTAMP', data_vars='all', coords='different',
        compat='no_conflicts', join='outer', combine_attrs='override')
    if load is None:
        nbytes = sum(os.path.getsize(f) for f in files)
        load = nbytes <= LOAD_LIMIT
        if not load:
            logger.warning(
                "%d files (%.1f GiB): opening them lazily, which is slower. "
                "Use load=True to read them into memory; peak RAM is a few "
                "times their size.", len(files), nbytes / 2**30)
    rules = []

    def note_rule(ds):
        rules.append(ds.attrs.get(FLUX_RULE, SAME_SCALE))
        return ds

    if load:
        datasets = [note_rule(xr.load_dataset(f)) for f in files]
        _check_one_rule(rules, path)
        ds = xr.combine_nested(datasets, **combine_kwargs)
        return ds if chunks is None else ds.chunk(chunks)
    ds = xr.open_mfdataset(files, combine='nested', preprocess=note_rule,
                           **combine_kwargs)
    try:
        _check_one_rule(rules, path)
    except ValueError:
        ds.close()
        raise
    if chunks is None:
        # Small files merge into few chunks and large ones split, so the chunks
        # do not grow with the folder. Not 'auto', which refuses text variables.
        step = max(v.nbytes // v.sizes['TIMESTAMP']
                   for v in ds.variables.values() if 'TIMESTAMP' in v.dims)
        limit = parse_bytes(dask.config.get('array.chunk-size'))
        chunks = {'TIMESTAMP': max(1, limit // step)}
    out = ds.chunk(chunks)
    out.set_close(ds.close)  # chunk() drops the closer of the open files
    return out


def _check_one_rule(rules, path):
    counts = {rule: rules.count(rule) for rule in sorted(set(rules))}
    if len(counts) > 1:
        raise ValueError(
            f"The files in {path!r} were made with different flux rules "
            f"({', '.join(f'{n} {r}' for r, n in counts.items())}), and cannot "
            "be put together. Run them again into a new output folder.")


def data_average_dims(data, id_cols=None):
    if id_cols is None:
        return data.mean(dim=_window_dims(data), keep_attrs=True)
    ds = data.mean(dim=[d for d in data.dims if d not in id_cols], keep_attrs=True)
    return ds


def data_integrate_in_frequency(data, f0, freq='natural_frequency'):
    assert freq in data.dims, f'Dim `{freq}` not found in data.'
    ds = data.sel({freq: slice(f0)}
                  ).sum(dim=freq, keep_attrs=True)
    return ds


def _date_from_yl(date, processing_time_duration):
    date = re.sub('[-: ]', '', str(date))
    if processing_time_duration.endswith("D"):
        date = date[:8]
    if processing_time_duration.endswith("H") or processing_time_duration.endswith("Min"):
        date = date[:12]
    return date


def _validate_run(date, yl, output_path, curoutpath_inprog, overwrite,
                  compare_start=True, compare_end=False):
    # recheck if files exist and overwrite option
    # doesn't save time (maybe only save 5min)
    file_name = os.path.basename(output_path.format(date))
    part_name0 = file_name.rsplit('_', 1)[0] + '_' if compare_start else ''
    part_name1 = file_name.rsplit('.', 1)[-1] if compare_end else ''
    current_files = [p for p in os.listdir(os.path.dirname(
        output_path)) if p.startswith(part_name0) and p.endswith(part_name1)]

    if not overwrite and current_files: #file_name in os.path.exists(output_pathmodel.format(date)):
        logger.warning(
            "UserWarning: Skipping, file already exists ({}).".format(date))
        return False

    # # if all([os.path.exists(output_pathmodel.format(_yl.strftime('%Y%m%d%H%M'))) for _yl in yl[:-1]]):
    # if all([output_pathmodel.format(_yl.strftime('%Y%m%d%H%M')) in current_files for _yl in yl[:-1]]):
    #     logger.warning(
    #         "UserWarning: Skipping, file already exists ({}).".format(date))
    #     return False
    # # elif any([os.path.exists(output_pathmodel.format(_yl.strftime('%Y%m%d%H%M'))) for _yl in yl[:-1]]):
    # if any([output_pathmodel.format(_yl.strftime('%Y%m%d%H%M')) in current_files for _yl in yl[:-1]]):
    #     logger.warning("UserWarning: Continuing but some files already exist ({}), others don't ({}).".format(
    #         ', '.join([_yl.strftime('%Y%m%d%H%M') for _yl in yl[:-1] if output_pathmodel.format(_yl.strftime('%Y%m%d%H%M')) in current_files]),
    #         ', '.join([_yl.strftime('%Y%m%d%H%M') for _yl in yl[:-1] if not output_pathmodel.format(_yl.strftime('%Y%m%d%H%M')) in current_files]),
    #     ))
    #     return True

    if commons.checkifinprogress(curoutpath_inprog):
        return False
    return True


def _load_data(yl, input_path, reader):
    def date2name(date):
        return date.strftime('%Y%m%d-%H%M')

    start_time = time.time()
    paths = [
        file
        for name in [date2name(d) for d in yl]
        for file in glob.glob(f'{os.path.join(input_path, name)}*')
    ]

    data = []

    for p in paths:
        try:
            data += [reader(p)]
        except Exception as e:
            logger.error(f"Error when reading {p}: {e}")

    try:
        data = xr.concat(data, dim='TIMESTAMP')
    except Exception as e:
        data = xr.Dataset()
        logger.error(str(e))
    # if data.size == 0:
    #     logger.warning(
    #             f"UserWarning: No file found ({date}, path: {input_path}).")
    #     return None
    logger.debug(f'\tLoading data took {round(time.time() - start_time)} s.')
    return data


def _exit(curoutpath_inprog):
    if os.path.exists(curoutpath_inprog):
        os.remove(curoutpath_inprog)


def _clear_unfinished(output_folderpath, run_time):
    """
    Remove what a stopped run left: the markers of the periods it had in
    progress and their incomplete files. The next run would skip those periods
    as already done. Stopping one worker stops them all, at any point.
    """
    suffix = '.inprogress'
    folder = glob.escape(os.path.join(output_folderpath, "wavelet_full_cospectra"))
    for marker in glob.glob(os.path.join(folder, f"*_{run_time}.nc{suffix}")):
        for path in (marker[:-len(suffix)], marker):
            if os.path.exists(path):
                os.remove(path)


def _apply_filter(data, filter_criteria):
    for var, criteria in filter_criteria.items():
        if var in data:
            data[var] = data[var].where(
                (data[var] >= criteria.start) & (data[var] < criteria.stop))
        else:
            logger.warning(f"Trying to filter {var}, but variable not found in data.")
    return data


def _process_period(yl, processing_time_duration, fileduration, input_path, reader,
                    output_folderpath, identifier, run_time, overwrite,
                    filter_criteria, run_kwargs, store=None):
    """
    Read, decompose, average and save one period of `process`. Returns the
    averaged dataset, or None if the period was skipped or failed.
    """
    info_t_yl_ymd = time.time()
    output_path = None
    curoutpath_inprog = ""
    result = None

    try:
        date = _date_from_yl(yl[0], processing_time_duration)

        logger.info("%s reading", date)

        if output_folderpath is not None:
            output_path = str(os.path.join(
                output_folderpath,
                "wavelet_full_cospectra",
                f"{identifier}_full_cospectra_{date}_{run_time}.nc"))
            commons.mkdirs(output_path)
            curoutpath_inprog = f"{output_path}.inprogress"
            logger.debug(f'In progress file: {curoutpath_inprog}.')
            if not _validate_run(date, yl, output_path, curoutpath_inprog, overwrite):
                return None
    except Exception as e:
        logger.critical(e)
        warnings.warn(str(e))
        return None

    try:
        if store is None:
            data = _load_data(yl, input_path, reader)
        else:
            data = grid.read_windows(store, yl)
    except Exception as e:
        logger.critical(e)
        raise

    if data is None or (store is not None and not data.sizes['TIMESTAMP']):
        _exit(curoutpath_inprog)
        return None

    try:
        # main run
        if store is None:
            data = _apply_filter(data, filter_criteria)

        ds_average = data_run(data, sel={'TIMESTAMP': slice(min(yl), max(yl) + datetime.timedelta(minutes=fileduration))},
                              dst=None, average=True, store=store,
                              window=int(round(fileduration * 60 * run_kwargs['fs'])),
                              **run_kwargs)
        if output_path is not None:
            ds_average.to_netcdf(output_path)
        result = ds_average

        # allvars = run_kwargs['varstorun']
        # saved_files = []
        # for f in allvars:
        #     run_kwargs['varstorun'] = [f]
        #     output = data_run(data, period=[min(yl), max(yl)],
        #                   meta=meta,
        #                   dst=output_path, **run_kwargs)
        #     saved_files.append(output.saved)
        #     fulldata = pd.concat([fulldata, output.data], axis=0)

        # for f in [s for s_ in saved_files for s in s_]:
        #     if os.path.exists(f):
        #         os.rename(f, f.replace('.part', ''))

    except Exception as e:
        logger.critical(e)
        # a file left here would make the next run skip this period as done
        if output_path is not None and result is None and os.path.exists(output_path):
            os.remove(output_path)

    logger.debug(
        f"Date loop ({yl[0].strftime('%Y%m%d-%H%M')}:{yl[-1].strftime('%Y%m%d-%H%M')}) took {round(time.time() - info_t_yl_ymd)} s.")
    _exit(curoutpath_inprog)
    return result


def _process_period_in_worker(log_level, yl, **kwargs):
    """
    Run `_process_period` in a worker process, which has none of the log handlers
    of the main one. Returns its result and its log records.
    """
    records = queue.SimpleQueue()
    handler = logging.handlers.QueueHandler(records)
    root = logging.getLogger()
    root.setLevel(log_level)
    root.addHandler(handler)
    logging.captureWarnings(True)
    try:
        result = _process_period(yl, **kwargs)
    finally:
        root.removeHandler(handler)
    return result, [records.get() for _ in range(records.qsize())]


def _process_periods_in_parallel(ymd, n_workers, **kwargs):
    """
    Run `_process_period` on each period of *ymd* in *n_workers* processes.
    Returns the results in the order of *ymd*, and logs each period as it ends.
    """
    logger.info("Running %d workers: each period logs when it ends.", n_workers)
    log_level = logger.getEffectiveLevel()
    tasks = (joblib.delayed(_process_period_in_worker)(log_level, yl, **kwargs)
             for yl in ymd)
    results = []
    for result, records in joblib.Parallel(n_jobs=n_workers, return_as='generator')(tasks):
        for record in records:
            logging.getLogger(record.name).handle(record)
        results.append(result)
    return results


def _warn_if_mixed_rules(output_folderpath):
    """
    Warn when the output folder holds files made with different flux rules.

    It looks at the files after the run, because the rule of a file depends on
    the data too (no window dim gives same_scale), not only on the arguments.
    """
    files = glob.glob(os.path.join(glob.escape(os.path.join(
        output_folderpath, "wavelet_full_cospectra")), "*.nc"))
    found = set()
    for path in files:
        try:
            with xr.open_dataset(path) as ds:
                found.add(ds.attrs.get(FLUX_RULE, SAME_SCALE))
        except Exception as e:
            logger.debug(f"Could not read the flux rule of {path}: {e}")
    if len(found) > 1:
        logger.warning(
            "%s holds files made with the %s flux rules: open_files_in_folder "
            "will refuse to put them together. Use a new output_folderpath.",
            output_folderpath, ' and '.join(sorted(found)))


def _window_means(data, formulas):
    """What a run on the fixed grid keeps of a file besides its samples."""
    covariances = {c for f in formulas for c in formula_to_vars(f).combinations}
    statistics = xr.merge([data_statistics(data, formula=f) for f in covariances])
    return data_average_dims(xr.merge([data, statistics], compat='override'))


def _build_grid_store(ymd, folder, identifier, input_path, reader, fileduration,
                      filter_criteria, run_kwargs, overwrite, n_jobs):
    """
    Decompose the whole run on the fixed grid (see `grid.build_store`). Returns
    the path of the store, or None if the run can't use the grid.
    """
    if run_kwargs['method'] not in ('dwt', 'modwt', 'moda'):
        logger.info("The fixed grid is for discrete methods: using regular blocks.")
        return None
    labels = [t for yl in ymd for t in yl]
    if not labels:
        return None
    fs = run_kwargs['fs']
    start = min(labels) - datetime.timedelta(minutes=fileduration)

    def read(path):
        return _apply_filter(reader(path), filter_criteria)

    reason = grid.why_not_on_grid(input_path, read, start, max(labels), fs, fileduration)
    if reason:
        logger.warning(f"Using regular blocks instead of the fixed grid: {reason}.")
        return None
    formulas = run_kwargs['varstorun']
    return grid.build_store(
        folder, identifier, input_path, read,
        lambda data: _window_means(data, formulas),
        sorted({var for f in formulas for var in formula_to_vars(f).uniquevars}),
        start, max(labels), fs, fileduration,
        run_kwargs.get('level', grid.deepest_level(fs, run_kwargs.get('f0', grid.DEFAULT_F0))),
        run_kwargs.get('wavelet', cascade.DEFAULT_WAVELET), overwrite, n_jobs,
        run_kwargs['method'])


def process(datetimerange, fileduration, input_path, acquisition_frequency,
            covariance=None, output_folderpath=None, verbosity=1,
            overwrite=False, processing_time_duration="1D",
            reader_method='ep_raw_lvl',
            internal_averaging=None, dt=0.05,
            integration_period=None,
            identifier=None,
            filter_criteria=None,
            method="dwt", averaging=30, n_jobs=1, fixed_grid=True, cross_level=True,
            **kwargs):
    """
    Decompose and average the raw files of *datetimerange*, saving one file per
    *processing_time_duration*.

    Parameters:
    - n_jobs: periods to process at once (c.f. joblib.Parallel; -1 is one per
      CPU). Each runs in its own process, with a whole period in memory.
    - fixed_grid: decompose the whole run once on a calendar grid (see grid), so
      that a period gets the same result however the run is cut and wherever it
      starts. Memory then follows the period but not the number of levels, and
      the decomposition is kept in <output_folderpath>/decomposition for the
      next run of the same period. With method='modwt', which has no grid, it
      gives each period its neighbours in the same way, and keeps the samples;
      so with 'moda'. Only for discrete methods and for files that are one
      window each: anything else uses regular blocks, with a warning. False
      decomposes each block on its own.
    - method: 'dwt', 'modwt', 'moda', 'cwt' or 'fcwt' (see universal_wt)
    - cross_level: split each flux over the levels with the cross-level rule
      (see data_compute_flux), so that the levels and <x*y>_lf add up to
      cov_<xy>_filled in each window. For 'dwt' and 'moda', and for files that
      are windows (an 'ns' dim); 'modwt', 'cwt' and 'fcwt' keep same-scale
      products. False gives the same-scale products of earlier versions.
      attrs['flux_rule'] says which rule a result has

    Returns:
    - xarray.Dataset with the averaged periods
    """
    if filter_criteria is None:
        filter_criteria = {}
    logger.debug('--- Starting process ---')
    local_args = locals()
    info_t_start = time.time()

    # raw_kwargs = {'path': input_path, 'fkwargs': {
    #     'dt': 1/acquisition_frequency}}
    # raw_kwargs.update({k: v for k, v in kwargs.items() if k in ['fmt']})

    run_kwargs = {
        'fs': acquisition_frequency,
        'method': method,
        'cross_level': cross_level,
        # 'averaging': averaging,
        'varstorun': covariance or commons.available_combinations(commons.DEFAULT_COVARIANCE),
        **kwargs.get("run_kwargs", {}),
        **kwargs.get("wt_kwargs", {})
    }

    logging_kwargs = kwargs.get("logging_kwargs", {})

    identifier = identifier or kwargs.get('sitename', '00000')
    
    run_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
    # output_path = ""
    # output_pathmodel = ""

    
    if output_folderpath is not None:
        commons.start_logging(output_folderpath, **logging_kwargs)
        try:
            commons.save_locals(local_args, os.path.join(output_folderpath, f'log/setup_{run_time}.yml'))
        except Exception as e:
            logger.warning(f"Could not save local arguments to file: {e}")
        
    ymd = [datetimerange.split(
        '-')[0], datetimerange.split('-')[1], f'{fileduration}min']
    
    # if isinstance(averaging, (list, tuple)):
    #     averaging = averaging[-1]
    # if internal_averaging is None:
    #     internal_averaging = averaging
    
    # fulldata = pd.DataFrame()
    
    _, _, _f = ymd
    ymd = commons.list_time_in_period(*ymd, processing_time_duration, include='right')

    # ymd = {y[-1]: y for y in ymd}
    
    logger.debug(
        f'Start date loop at {round(time.time() - info_t_start)} s.')

    period_kwargs = {
        'processing_time_duration': processing_time_duration,
        'fileduration': fileduration,
        'input_path': input_path,
        # looked up here: workers lack readers registered while running
        'reader': READERS.get(reader_method),
        'output_folderpath': output_folderpath,
        'identifier': identifier,
        'run_time': run_time,
        'overwrite': overwrite,
        'filter_criteria': filter_criteria,
        'run_kwargs': run_kwargs,
    }
    n_workers = min(joblib.effective_n_jobs(n_jobs), len(ymd))

    temporary = None
    try:
        if fixed_grid:
            if output_folderpath is None:
                folder = temporary = tempfile.mkdtemp(prefix='fluxscale_grid_')
            else:
                folder = os.path.join(output_folderpath, 'decomposition')
            period_kwargs['store'] = _build_grid_store(
                ymd, folder, identifier, input_path, period_kwargs['reader'],
                fileduration, filter_criteria, run_kwargs, overwrite,
                joblib.effective_n_jobs(n_jobs))
        if n_workers > 1:
            ds_collection = _process_periods_in_parallel(ymd, n_workers, **period_kwargs)
        else:
            ds_collection = [_process_period(yl, **period_kwargs) for yl in ymd]
    finally:
        if output_folderpath is not None:
            _clear_unfinished(output_folderpath, run_time)
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)
    ds_collection = [ds for ds in ds_collection if ds is not None]
    if output_folderpath is not None:
        _warn_if_mixed_rules(output_folderpath)
    
    logger.debug(f'End date loop at {round(time.time() - info_t_start)} s.')
    logger.debug(f"integration_period: {integration_period}.")

    # if output_pathmodel and not fulldata.empty:
    #     # timestamp = pd.Timestamp.now().strftime('%Y%m%dT%H%M%S_%f')
    #     dst_path = os.path.join(output_folderpath, os.path.basename(
    #         output_pathmodel.format(run_time)))
    #     if integration_period:
    #         fulldata = integrate_cospectra(fulldata, 1/integration_period, dst_path=None)
    #     fulldata.to_csv(dst_path, index=False)

    # Concatenate all data
    if ds_collection:
        ds_collection = xr.concat(ds_collection, dim='TIMESTAMP')
        ds_collection = ds_collection.rename(
            {var: f"csp_{var}" for var in ds_collection.data_vars if 'natural_frequency' in ds_collection[var].dims})

        if integration_period:
            ds_collection_i = data_integrate_in_frequency(
                ds_collection, 1/(integration_period*60))
            ds_collection_i = ds_collection_i.rename(
                {var: f"{var}_int" for var in ds_collection_i.data_vars})

            ds_collection = ds_collection.merge(ds_collection_i)
    else:
        logger.error('No data was collected.')
        ds_collection = xr.Dataset()

    # Include metadata
    meta = {
        'identifier': identifier,
        'processing_time_duration': processing_time_duration,
        'acquisition_frequency': acquisition_frequency,
    }
    meta.update(ds_collection.attrs)
    ds_collection.attrs.update(meta)

    logger.debug(
        f'\t\tFull process took {round(time.time() - info_t_start)} s (run_wt).')
    return ds_collection


def _run_decomposed(data_decomposed, varstorun, sel, average_period, average, kwargs,
                    cross_level=True, low=True):
    """
    Products, conditional sampling and low frequencies of decomposed data.

    With *cross_level*, each product of two variables or more is the flux
    density of data_compute_flux, from the levels and the <var>_signal in
    *data_decomposed*; the classes take their signs from the same-scale
    products and their values from that density. *low* adds <x*y>_lf and
    cov_<xy>_filled (c.f. data_compute_flux). Without a window dim there is
    no window to split the covariance of, and the products stay same-scale.
    """
    data_decomposed.attrs.update({
        'averaging': average_period,
        'method': f"{kwargs.get('method', '')} ~{kwargs.get('mother_wavelet', '')}",
        'dt': kwargs.get('dt', np.nan)
    })

    # Select valid dates
    if sel:
        data_decomposed = data_decomposed.sel(sel)

    # The gap-filled signals are only for the flux: take them out, so that
    # they do not reach the output with the rest of data_decomposed
    uniquevars = {v for f in varstorun for v in formula_to_vars(f).uniquevars}
    helpers = [f'{v}{SIGNAL}' for v in sorted(uniquevars)
               if f'{v}{SIGNAL}' in data_decomposed.data_vars]
    signals = data_decomposed[helpers]
    if helpers:
        data_decomposed = data_decomposed.drop_vars(helpers)

    window = _window_dims(data_decomposed)
    if cross_level and not window:
        logger.warning(
            "No window dim (only TIMESTAMP): the products stay same-scale, as "
            "there is no window covariance to split over the levels.")
        cross_level = False

    # Calculate covariance
    info_t_calc_product = time.time()
    uniquecovs = list(
        set([c for f in varstorun for c in formula_to_vars(f).combinations]))

    # Calculate product from formula for each unique covariance
    data_product = [
        data_compute_product(data_decomposed, formula=f)
        for f in uniquecovs]
    logger.debug(
        f'\tCalculate product from formula took {round(time.time() - info_t_calc_product)} s.')
    # every entry is data_decomposed itself, so there is nothing to compare
    data_product = xr.merge(data_product, compat='override')
    data_product = xr.merge([data_decomposed, data_product], compat='override')

    # Cross-level fluxes, from the levels and the gap-filled signals. They are
    # kept apart from the products until the classes are made, which take
    # their signs from the same-scale products (data_compute_product keeps a
    # name already in data, so writing the flux first would change the signs).
    flux = xr.Dataset()
    values = None
    if cross_level:
        flux_input = data_decomposed.assign(
            {v: signals[v] for v in signals.data_vars})
        multi = sorted(c for c in uniquecovs if len(formula_to_vars(c).xy) > 1)
        flux = xr.merge([data_compute_flux(flux_input, c, average=False, low=low)
                         for c in multi], compat='override')
        values = {n: flux[n] for n in (''.join(formula_to_vars(f).xy) for f in varstorun)
                  if n in flux}

    # Calculate conditional sampling
    info_t_calc_cond_samp = time.time()
    if average:
        data_condsamp = data_conditional_sampling_mean(
            data_product, varstorun, dim=_window_dims(data_product), values=values)
    else:
        data_condsamp = xr.merge([
            data_conditional_sampling(data_product, f, values=values)
            for f in varstorun
        ])
    logger.debug(
        f'\tCalculate conditional sampling took {round(time.time() - info_t_calc_cond_samp)} s.')

    # Low-frequency
    if cross_level:
        # data_decomposed carries the plain products and <var>_lf to the output
        data_lf = data_decomposed.assign({v: flux[v] for v in flux.data_vars})
    else:
        xycovs = list(
            set(["*".join([f"{c}_lf" for c in formula_to_vars(f).xy]) for f in varstorun]))
        data_lf = [
            data_compute_product(data_decomposed, formula=f, name=f.replace('_lf', '') + "_lf")
            for f in xycovs]
        logger.debug(
            f'\tCalculate product from formula took {round(time.time() - info_t_calc_product)} s.')
        data_lf = xr.merge(data_lf)

    # Average data
    # data_averaged = data_product.groupby('TIMESTAMP').sum('natural_frequency')
    # data_integrated = data_product.where(data_product.natural_frequency >=
    #                                      1/(30*60), drop=True).sum('natural_frequency')

    # Rename variables
    data_product = data_product.rename(
        {var: f"{var}~" for var in data_product.data_vars})
    if cross_level:
        data_product = data_product.assign(
            {f"{v}~": flux[v] for v in flux.data_vars if f"{v}~" in data_product})

    # Merge data
    response = xr.merge([data_product, data_lf, data_condsamp], compat='override')
    for ds in [data_product, data_lf, data_condsamp]:
        response.attrs.update(ds.attrs)
    response.attrs[FLUX_RULE] = CROSS_LEVEL if cross_level else SAME_SCALE
    return response


def _run_on_grid(data, store, window, variables, varstorun, sel, average_period, kwargs,
                 cross_level=True):
    """_run_decomposed of a period read back from *store*, one level at a time."""
    block = grid.open_block(store, data.TIMESTAMP.values, window, kwargs.get('method'))
    low = {var: block.approximation(var) for var in variables}
    signal = {var: block.signal(var) for var in variables} if cross_level else None
    levels = [
        data_average_dims(_run_decomposed(
            block.decomposed(variables, level, low, signal), varstorun, sel,
            average_period, True, kwargs, cross_level=cross_level, low=False))
        for level in block.levels]
    response = xr.concat(levels, dim='natural_frequency', data_vars='minimal',
                         coords='minimal', compat='override')
    response.attrs.update(levels[0].attrs)
    if response.attrs.get(FLUX_RULE) == CROSS_LEVEL:
        # the low terms do not depend on the level: once per block
        dims = ('TIMESTAMP', 'ns')
        whole = xr.Dataset(
            {**{f'{v}_lf': (dims, low[v]) for v in variables},
             **{f'{v}{SIGNAL}': (dims, signal[v]) for v in variables}},
            coords={'TIMESTAMP': block.timestamps, 'ns': block.ns})
        if sel:
            whole = whole.sel(sel)
        multi = sorted({c for f in varstorun for c in formula_to_vars(f).combinations
                        if len(formula_to_vars(c).xy) > 1})
        response = xr.merge(
            [response] + [data_compute_flux(whole, c, average=True, levels=False)
                          for c in multi],
            compat='override', combine_attrs='override')
    return response


def data_run(data, varstorun, sel=None, average_period='30min', dst=None, average=False,
             store=None, window=None, cross_level=True, **kwargs):
    """
    Main function to decompose, calculate covariance, and average data using xarray.Dataset.

    Parameters:
    - data: xarray.Dataset
    - varstorun: list of formulas to run
    - sel: dict with keys matching dimensions and values (c.f. xarray.Dataset.sel)
    - average_period: period for averaging
    - dst: path to save the result to
    - average: return the result of data_average_dims. Much faster and lighter than
      averaging the full result, whose conditional sampling is the bulk of it.
      With average=False, the products and <x*y>_lf are per-sample densities,
      like the levels, and cov_<xy>_filled is a window mean: they add up only
      after data_average_dims
    - store: path of a decomposition from grid.build_store to read instead of
      decomposing *data*, which then is what grid.read_windows gives. Needs
      average=True
    - window: samples per window of *data*, with store
    - cross_level: products of two variables or more are cross-level fluxes
      (see data_compute_flux), and the classes hold their values. Only for
      'dwt' and 'moda', whose levels add up to the signal: 'modwt' keeps
      same-scale products, and so do 'cwt' and 'fcwt' for now, with a warning
    - output_kwargs: output configuration
    - meta: metadata dictionary
    - **kwargs: additional arguments for decompose_data

    Returns:
    - xarray.Dataset with processed data. attrs['flux_rule'] is 'cross_level'
      or 'same_scale'
    """
    logger.debug(
        'Start data_run.')
    info_t_main = time.time()
    vars_unique = list(
        set([var for f in varstorun for var in formula_to_vars(f).uniquevars]))

    if store is not None and not average:
        raise ValueError("data_run with a store needs average=True.")

    method = kwargs.get('method', 'dwt')
    if cross_level and method in ('cwt', 'fcwt'):
        logger.warning(
            "UserWarning: The cross-level flux is not available for %s yet: "
            "products stay same-scale.", method)
        cross_level = False
    elif cross_level and method not in CROSS_LEVEL_METHODS:
        logger.info(
            "The levels of %s do not add up to the signal: products stay "
            "same-scale (use 'moda' for the cross-level flux).", method)
        cross_level = False

    # Decompose all required variables
    if store is None:
        results = _run_decomposed(
            decompose_variables(
                data.stack(z=data.dims), vars_unique, nan_tolerance=.3,
                keep_signal=cross_level, **kwargs).unstack(),
            varstorun, sel, average_period, average, kwargs, cross_level=cross_level)

        # Calculate statistics
        uniquecovs = list(
            set([c for f in varstorun for c in formula_to_vars(f).combinations]))
        data_stats = [
            data_statistics(data, formula=f)
            for f in uniquecovs]
        data_stats = xr.merge(data_stats)
    else:
        results = _run_on_grid(data, store, window, vars_unique, varstorun, sel,
                               average_period, kwargs, cross_level=cross_level)
        data_stats = xr.Dataset()   # data has them already

    # Merge data
    response = xr.merge([data, data_stats, results], compat='override')

    # Merge attrs
    for ds in [data, data_stats, results]:
        response.attrs.update(ds.attrs)

    if average:
        response = data_average_dims(response)

    # Save data
    if dst:
        response.to_netcdf(dst)
        response.attrs.setdefault('pipeout', []).append(f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}    Saved in: {dst}.")

    logger.debug(
        f'\tMain took {round(time.time() - info_t_main)} s.')
    return response


def run_from_eddypro(path,
                     # ="input/EP/FR-Gri_sample.eddypro",
                     #  covariance=["w*co2|w|co2|h2o", "w*co2|w*h2o", "w*h2o",],
                     #  processduration='6H',
                     **kwargs):
    c = eddypro.extract_info_from_eddypro_setup(eddypro=path)
    c.update(**kwargs)

    for path in ['input_path', 'output_folderpath']:
        if c.get(path, None) is not None:
            c[path] = os.path.abspath(c[path])

    # if covariance is None:
    #     # TODO: get variables available from eddypro
    #     covariance = commons.available_combinations(commons.DEFAULT_COVARIANCE)

    return process(**c)
