# built-in modules
import logging
import math

# 3rd party modules
import numpy as np
import pandas as pd
import xarray as xr
import itertools

# project modules
from fluxscale.core.addons import read_file

logger = logging.getLogger(__name__)

__all__ = [
    "conditional_sampling",
    "conditional_sampling_mean",
    "partition_DWCS",
    "partition_DWCS_H2O",
    "partition_DWCS_CH4",
    "partition_DWCS_CO",
]


def _as_series(data, x):
    """Resolve a partition input into a numeric array/series.

    Args:
        data: Mapping (DataFrame/Dataset) used to look up column names.
        x: ``None`` (treated as additive identity ``0``), a column name,
            a list/tuple of column names (summed), or an already-resolved array.

    Returns:
        The resolved value: ``0`` for ``None``, ``data[x]`` for a string, the
        element-wise sum for a list/tuple, otherwise ``x`` unchanged.
    """
    if x is None:
        return 0
    if isinstance(x, str):
        return data[x]
    if isinstance(x, (list, tuple)):
        return sum(data[c] for c in x)
    return x


def _sign_classes(names, label):
    """Yield ``(class name, signs)`` for every combination of signs over *names*.

    Args:
        names: Variable names, one per condition.
        label: Dictionary mapping condition values to labels, e.g. {1: "+", -1: "-"}.

    Yields:
        The class name (e.g. ``"wco2+wh2o-"``) and its tuple of condition values.
    """
    n = len(names)
    # run for all unique combinations of + and - for groups of size n
    # (e.g., n=2: ++, +-, -+, --, n=3 : +++, ++-, ...)
    for co in set(itertools.combinations(list(label.keys())*n, n)):
        if not co:
            continue
        sign = ''.join([label[c] for c in co])
        yield ''.join([c for cs in zip(names, sign) for c in cs]), co


def conditional_sampling(Y12, *args, names=None, label=None, false=0):
    """
    Perform conditional sampling on xarray.DataArray objects.

    Parameters:
    - Y12: xarray.DataArray (main variable)
    - *args: xarray.DataArray objects for conditional sampling
    - names: list of names for each variable (defaults to ['xy', 'a'])
    - label: dictionary mapping condition values to labels, e.g. {1: "+", -1: "-", 0: "·"}
      (defaults to {1: "+", -1: "-"})
    - false: value to use for false conditions

    Returns:
    - xarray.Dataset with conditionally sampled variables
    """
    if names is None:
        names = ['xy', 'a']
    if label is None:
        label = {1: "+", -1: "-"}
    nargs = len(args)
    if nargs < len(names):
        names = names[:nargs]
    if nargs > len(names):
        names = names + ['b'] * (nargs-len(names))
        
    YS = list(args)
    Ys = xr.Dataset()  # Initialize as an empty xarray.Dataset

    for name, co in _sign_classes(names, label):
        Ys[name] = Y12
        logger.debug(f"name: {name}, co: {co}")

        # Apply conditions by sign
        for i, c in enumerate(co):
            if c:
                mask = xr.where(c * YS[i] > 0, 1, false)
            else:
                mask = xr.where(YS[i] == 0, 1, false)

            mask = np.where(mask == 0, false, mask)
            Ys[name] = Ys[name] * mask

    return Ys


def _base3(digits):
    """Read *digits* (integers or integer arrays) as one base-3 number."""
    code = 0
    for digit in digits:
        code = 3 * code + digit
    return code


def _sum_by_code(code, x, n_codes):
    """Sum *x* by integer *code*. np.bincount takes no complex weights, which the CWT gives."""
    if np.iscomplexobj(x):
        return (_sum_by_code(code, x.real, n_codes)
                + 1j * _sum_by_code(code, x.imag, n_codes))
    return np.bincount(code, weights=x, minlength=n_codes)


@np.errstate(invalid='ignore')  # complex NaNs in comparisons, 0/0 with nothing to average
def conditional_sampling_mean(data, groups, dim, values=None):
    """
    Mean over *dim* of each sign class, without building the classes.

    :func:`conditional_sampling` returns one full-size array per class. When only
    their means are wanted, it is much cheaper to work out the class of each
    sample and add the sample to a running total for that class, one row (e.g.
    one scale in one averaging period) at a time. The result is the mean of what
    conditional_sampling returns with ``label={1: "+", -1: "-"}``, up to
    floating-point summation order: every sample whose sampled value is not NaN
    counts in the denominator, even when zero or NaN in a condition puts it in no
    class.

    Parameters:
    - data: xarray.Dataset holding every variable named in *groups*
    - groups: list of lists of variable names, one list per conditional sampling.
      The first name is the variable sampled; every name, the first included, is
      a condition on its sign (c.f. data_conditional_sampling)
    - dim: dimension or list of dimensions to average over
    - values: optional dict of a sampled name to the values to sum into its
      classes instead of data[name], whose sign still sets the class (e.g. the
      cross-level flux density, c.f. handlers.data_compute_flux). It must have
      the dims of data[name]

    Returns:
    - xarray.Dataset with one variable per class, named as in conditional_sampling
    """
    if not groups:
        return xr.Dataset()

    values = values or {}
    dims = [dim] if isinstance(dim, str) else dim
    conditions = list(dict.fromkeys(name for group in groups for name in group))
    sampled = list(dict.fromkeys(group[0] for group in groups if group[0] in values))
    arrays = xr.broadcast(*[data[name] for name in conditions],
                          *[values[name] for name in sampled])

    kept = [d for d in arrays[0].dims if d not in dims]
    averaged = [d for d in arrays[0].dims if d in dims]
    sizes = arrays[0].sizes
    n_rows = math.prod(sizes[d] for d in kept)
    n_samples = math.prod(sizes[d] for d in averaged)
    signs = {name: array.transpose(*kept, *averaged).values.reshape(n_rows, n_samples)
             for name, array in zip(conditions, arrays)}
    summed = dict(signs)
    summed.update({name: array.transpose(*kept, *averaged).values.reshape(n_rows, n_samples)
                   for name, array in zip(sampled, arrays[len(conditions):])})

    # A class is a base-3 code with one digit per condition: 1 for positive, 0 for
    # negative, 2 for neither. A NaN in the sampled variable only reaches codes
    # where its own digit is 2, which no class reads, so it needs no cleaning.
    digit = {1: 1, -1: 0}
    totals = [np.empty((n_rows, 3 ** len(group)),
                       dtype=np.result_type(summed[group[0]], float))
              for group in groups]
    for row in range(n_rows):
        digits = {name: np.where(x[row] > 0, 1, np.where(x[row] < 0, 0, 2))
                  for name, x in signs.items()}
        for group, total in zip(groups, totals):
            total[row] = _sum_by_code(_base3(digits[name] for name in group),
                                      summed[group[0]][row], total.shape[1])

    means = {}
    for group, total in zip(groups, totals):
        count = np.count_nonzero(~np.isnan(summed[group[0]]), axis=1)
        for name, signs in _sign_classes(group, {1: "+", -1: "-"}):
            mean = total[:, _base3(digit[s] for s in signs)] / count
            means[name] = (kept, mean.reshape([sizes[d] for d in kept]))
    return xr.Dataset(means, coords=arrays[0].isel({d: 0 for d in averaged}, drop=True).coords)


def partition_DWCS(data, labelpositive='GPP', labelnegative='Reco', all='wco2',
                  positive='wco2-wh2o+', negative='wco2-wh2o-', NIGHT=None):
    """
    Partition *all* into *labelpositive* = positive + negative/2 (0 at night)
    and *labelnegative* = *all* - *labelpositive*.

    *all* is the cross-level flux when the data have attrs['flux_rule'] ==
    'cross_level' (c.f. handlers.data_compute_flux). The classes then hold the
    same flux, sample by sample, and add up to *all* but for samples where a
    sign is zero. *labelnegative* takes that rest.
    """
    if isinstance(data, str):
        data = read_file(data)
    else:
        data = data.copy()

    all_ = _as_series(data, all)
    positive_ = _as_series(data, positive)
    negative_ = _as_series(data, negative)

    if NIGHT is not None:
        night = data[NIGHT]
        islight = xr.where((~night.isnull()) & night.astype(bool), 0, 1)
    else:
        islight = xr.ones_like(positive_)
    data[labelpositive] = islight * (positive_ + 0.5 * negative_)
    data[labelnegative] = (all_ - data[labelpositive])
    return data

def partition_DWCS_H2O(data=None, NEE='NEE', GPP='GPP', Reco='Reco', CO2='wco2',
                  CO2neg_H2Opos='wco2-wh2o+', CO2neg_H2Oneg='wco2-wh2o-', NIGHT=None):
    """
    Partition with H2O: GPP = CO2-H2O+ + CO2-H2O-/2 (0 at night), Reco = NEE - GPP.

    NEE is *CO2* (``wco2``), the cross-level flux when the data have
    attrs['flux_rule'] == 'cross_level' (c.f. handlers.data_compute_flux). The
    classes then hold the same flux, sample by sample, and add up to NEE but
    for samples where a sign is zero (with 'moda', the gap-filled samples).
    Reco takes that rest.
    """
    if isinstance(data, str):
        data = read_file(data)
    elif data is None:
        data = pd.DataFrame()
    else:
        data = data.copy()

    CO2 = _as_series(data, CO2)
    CO2neg_H2Opos = _as_series(data, CO2neg_H2Opos)
    CO2neg_H2Oneg = _as_series(data, CO2neg_H2Oneg)

    if NIGHT is not None:
        night = data[NIGHT]
        islight = xr.where((~night.isnull()) & night.astype(bool), 0, 1)
    else:
        islight = xr.ones_like(CO2)

    data[GPP] = islight * (CO2neg_H2Opos + 0.5 * CO2neg_H2Oneg)
    data[Reco] = (CO2 - data[GPP])
    data[NEE] = CO2
    return data

def partition_DWCS_CH4(data=None, NEE='NEE', GPP='GPP', Reco='Reco', CO2='wco2', 
                  CO2pos_CH4pos='wco2+wch4+', CO2pos_CH4neg='wco2+wch4-', 
                  CO2neg_CH4pos='wco2-wch4+', CO2neg_CH4neg='wco2-wch4-', NIGHT=None):
    """
    Partition with CH4: Reco = CO2+CH4+ + CO2+CH4-/2, GPP = NEE - Reco (0 at
    night).

    NEE is *CO2* (``wco2``), the cross-level flux when the data have
    attrs['flux_rule'] == 'cross_level' (c.f. handlers.data_compute_flux). The
    classes then hold the same flux, sample by sample, and add up to NEE but
    for samples where a sign is zero (with 'moda', the gap-filled samples).
    GPP takes that rest by day. At night GPP is 0, so GPP + Reco is not NEE.
    """
    if isinstance(data, str):
        data = read_file(data)
    elif data is None:
        data = pd.DataFrame()
    else:
        data = data.copy()

    CO2 = _as_series(data, CO2)
    CO2pos_CH4pos = _as_series(data, CO2pos_CH4pos)
    CO2pos_CH4neg = _as_series(data, CO2pos_CH4neg)
    CO2neg_CH4pos = _as_series(data, CO2neg_CH4pos)
    CO2neg_CH4neg = _as_series(data, CO2neg_CH4neg)

    if NIGHT is not None:
        night = data[NIGHT]
        islight = xr.where((~night.isnull()) & night.astype(bool), 0, 1)
    else:
        islight = xr.ones_like(CO2)

    data[Reco] = CO2pos_CH4pos + 0.5 * CO2pos_CH4neg
    # GPP (photosynthetic uptake) only occurs in daylight; islight gates it.
    data[GPP] = islight * (CO2 - data[Reco])
    data[NEE] = CO2
    return data

def partition_DWCS_CO(data=None, NEE='NEE', GPP='GPP', Reco='Reco', ffCO2='ffCO2',
                     CO2='wco2', 
                     CO2neg_H2Opos='wco2-wh2o+', 
                     CO2neg_H2Oneg='wco2-wh2o-', 
                     CO2pos_COpos='wco2+wco+',
                     CO2pos_COneg='wco2+wco-',
                     NIGHT=None):
    """
    Partition with H2O and CO: GPP = CO2-H2O+ + CO2-H2O-/3 (0 at night),
    ffCO2 = CO2+CO+ and Reco = CO2+CO-, and what is left of NEE goes half to
    Reco, half to ffCO2.

    NEE is *CO2* (``wco2``), the cross-level flux when the data have
    attrs['flux_rule'] == 'cross_level' (c.f. handlers.data_compute_flux). The
    classes then hold the same flux, sample by sample, and add up to NEE but
    for samples where a sign is zero (with 'moda', the gap-filled samples).
    Reco and ffCO2 take that rest, half each.
    """
    if isinstance(data, str):
        data = read_file(data)
    elif data is None:
        data = pd.DataFrame()
    else:
        data = data.copy()

    CO2 = _as_series(data, CO2)
    CO2neg_H2Opos = _as_series(data, CO2neg_H2Opos)
    CO2neg_H2Oneg = _as_series(data, CO2neg_H2Oneg)
    CO2pos_COpos = _as_series(data, CO2pos_COpos)
    CO2pos_COneg = _as_series(data, CO2pos_COneg)

    if NIGHT is not None:
        night = data[NIGHT]
        islight = xr.where((~night.isnull()) & night.astype(bool), 0, 1)
    else:
        islight = xr.ones_like(CO2)

    data[NEE] = CO2
    data[GPP] = islight * (CO2neg_H2Opos + CO2neg_H2Oneg / 3)

    data[ffCO2] = CO2pos_COpos
    data[Reco]  = CO2pos_COneg

    remaining   = CO2 - data[GPP] - data[Reco] - data[ffCO2]
    data[Reco]  = data[Reco]  + remaining / 2
    data[ffCO2] = data[ffCO2] + remaining / 2

    return data