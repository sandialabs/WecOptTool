"""Provide the wave definition structure and provide
functions for creating different common types of waves.

This module provides the (empty) dataset structure for waves in
``wecopttool``.
It also provides functions for creating common types of waves such as
regular waves and irregular waves.
The dataset structure is an xarray.Dataset containing the following two
2D xarray.DataArrray: (1)  the amplitude spectrum  magnitude ``S``
(m^2*s) and (2) the phase  ``phase`` (rad). The 2D coordinates are:
wave frequency ``omega`` (rad/s)  and direction ``wave_direction`` (rad).
"""


from __future__ import annotations  # TODO: delete after python 3.10
import warnings
from typing import Callable

import numpy as np
import numpy.typing as npt
import xarray as xr
from scipy.special import gamma

from wecopttool.core import freq_array


def wave_dataset(f0: float, nfreq: int,
                 directions: float | npt.ArrayLike) -> xr.Dataset:
    """Create an empty wave dataset with correct dimensions and
    coordinates.

    Parameters
    ----------
    f0: float
        Initial frequency (in Hz) for frequency array.
        Frequency array given as
        [``f0``, 2 x ``f0``, ..., ``nfreq`` x ``f0``].
    nfreq: int
        Number of frequencies in frequency array. See ``f0``.
    directions: np.ndarray
        Wave directions in degrees. 1D array.

    Returns
    -------
    xr.Dataset
        Empty wave dataset.
    """
    directions = np.atleast_1d(_degrees_to_radians(directions))
    ndirections = len(directions)
    freqs = freq_array(f0, nfreq)
    omega = freqs*2*np.pi

    dims = ('omega', 'wave_direction')
    freq_units = {'units': 'rad/s'}
    rad_units = {'units': 'rad'}
    coords = [(dims[0], omega, freq_units), (dims[1], directions, rad_units)]
    tmp = np.zeros([nfreq, ndirections])

    attrs = {'units': 'm^2*s', 'long_name': 'wave amplitude'}
    spectrum = xr.DataArray(tmp, dims=dims, coords=coords, attrs=attrs)

    attrs = {'units': 'rad', 'long_name': 'wave phase'}
    phase = xr.DataArray(tmp.copy(), dims=dims, coords=coords, attrs=attrs)

    return xr.Dataset({'S': spectrum, 'phase': phase}, attrs={})


def regular_wave(f0: float, nfreq: int, freq: float, amplitude: float,
                 phase: float | None = None, direction: float = 0.0
                 ) -> xr.Dataset:
    """Create the dataset for a regular wave.

    Parameters
    ----------
    f0: float
        Initial frequency (in Hz) for frequency array.
        Frequency array given as
        [``f0``, 2 x ``f0``, ..., ``nfreq`` x ``f0``].
    nfreq: int
        Number of frequencies in frequency array. See ``f0``.
    freq: float
        Frequency (in Hz) of the regular wave. If ``freq`` not in the
        frequency array, the closest value is used and a warning is
        displayed.
    amplitude: float
        Amplitude (in m) of the regular wave.
    phase: float, optional
        Phase (in degrees) of the regular wave.
    direction: float, optional
        Direction (in degrees) of the regular wave.

    Returns
    -------
     xr.Dataset
        Wave dataset.
    """
    # empty dataset
    waves = wave_dataset(f0, nfreq, direction)

    # get index
    omega = freq*2*np.pi
    iomega = waves.sel(omega=omega, method='nearest').omega.values
    ifreq = iomega/(2*np.pi)
    if not np.isclose(iomega, omega):
        warnings.warn(f"Requested frequency {freq} Hz is not in array. " +
                      f"Using nearest value of {ifreq} Hz.")

    # amplitude
    waves['S'].loc[{'omega': iomega}] = 0.5 * amplitude**2 / f0

    # phase
    if phase is None:
        rphase = _random_phase()
        phase = np.degrees(rphase)
    else:
        rphase = _degrees_to_radians(phase)
    waves['phase'].loc[{'omega': iomega}] = rphase

    # attributes
    waves.attrs = {'Wave type': 'Regular',
                   'Frequency (Hz)': ifreq,
                   'Amplitude (m)': amplitude,
                   'Phase (degrees)': phase,
                   'Direction (degrees)': direction,
                   }

    return waves


def long_crested_wave(f0: float, nfreq: int, spectrum_func: Callable,
                      direction: float = 0.0, spectrum_name: str = '',
                      seed: int | None = None) -> xr.Dataset:
    """Create the dataset for a long-crested irregular wave.

    Parameters
    ----------
    f0: float
        Initial frequency (in Hz) for frequency array.
        Frequency array given as
        [``f0``, 2 x ``f0``, ..., ``nfreq`` x ``f0``].
    nfreq: int
        Number of frequencies in frequency array. See ``f0``.
    spectrum_func: function
        Wave spectrum function. Maps frequencies to amplitude spectrum.
        float | npt.ArrayLike -> float | np.ndarray
    direction: float, optional
        Direction (in degrees) of the regular wave.
    spectrum_name: str, optional
        Name of the spectrum function.
    seed: int, optional
        Random seed for reproducing the same results.

    Returns
    -------
     xr.Dataset
        Wave dataset.

    Examples
    --------
    Define wave parameters.

    >>> from wecopttool.waves import long_crested_wave as lcw
    >>> from wecopttool.waves import pierson_moskowitz_spectrum as pm
    >>> Hs = 5
    >>> Tp = 6
    >>> fp = 1/Tp

    Generate the wave using a Pierson-Moskowitz idealized spectrum.

    >>> wave = lcw(f0=fp/10,
    ...            nfreq=30,
    ...            spectrum_func=lambda f: pm(freq=f, fp=fp, hs=Hs))
    """
    # empty dataset
    waves = wave_dataset(f0, nfreq, direction)

    # amplitude & phase
    freqs = freq_array(f0, nfreq)
    waves['S'].values = spectrum_func(freqs).reshape(nfreq, 1)
    waves['phase'].values = _random_phase([nfreq, 1], seed)

    # attributes
    waves.attrs['Wave type'] = 'Long-crested irregular'
    waves.attrs['Direction (degrees)'] = direction
    waves.attrs['Spectrum'] = spectrum_name

    return waves


def irregular_wave(f0: float, nfreq: int,
                   directions: float | npt.ArrayLike,
                   spectrum_func: Callable, spread_func: Callable,
                   spectrum_name: str = '', spread_name: str = '',
                   seed: int | None = None) -> xr.Dataset:
    """Create the dataset for an irregular wave field.

    Parameters
    ----------
    f0: float
        Initial frequency (in Hz) for frequency array.
        Frequency array given as
        [``f0``, 2 x ``f0``, ..., ``nfreq`` x ``f0``].
    nfreq: int
        Number of frequencies in frequency array. See ``f0``.
    directions: np.ndarray
        Wave directions in degrees. 1D array.
    spectrum_func: function
        Wave spectrum function. Maps frequencies to amplitude spectrum.
        float | npt.ArrayLike -> float | np.ndarray
    spread_func: function
        Wave spreading function. Maps wave frequencies and directions to
        spread value.
        tuple[float | npt.ArrayLike, float | npt.ArrayLike
        ] -> np.ndarray.
    spectrum_name: str, optional
        Name of the spectrum function.
    spread_name: str, optional
        Name of the spread function.
    seed: int, optional
        Random seed for reproducing the same results.

    Returns
    -------
     xr.Dataset
        Wave dataset.

    Examples
    --------
    Define the desired wave parameters.

    >>> import wecopttool as wot
    >>> import numpy as np
    >>> Hs = 5
    >>> Tp = 6
    >>> fp = 1/Tp
    >>> directions = np.linspace(0, 360, 36, endpoint=False)

    Create a function handle to define the spectral density,

    >>> def spectrum_func(f):
    ...    return wot.waves.pierson_moskowitz_spectrum(freq=f,
    ...                                                fp=fp,
    ...                                                hs=Hs)

    and a spreading function handle for spreading.

    >>> def spread_func(f, d):
    ...     return wot.waves.spread_cos2s(freq=f,
    ...                                   directions=d,
    ...                                   dm=10,
    ...                                   fp=fp,
    ...                                   s_max=10)

    Generate the wave.

    >>> wave = wot.waves.irregular_wave(f0=fp/10,
    ...                                 nfreq=20,
    ...                                 directions=directions,
    ...                                 spectrum_func=spectrum_func,
    ...                                 spread_func=spread_func)
    """
    # empty dataset
    waves = wave_dataset(f0, nfreq, directions)

    # amplitude & phase
    ndirections = len(directions)
    freqs = freq_array(f0, nfreq)
    spectrum = spectrum_func(freqs).reshape(nfreq, 1)
    spread = spread_func(freqs, directions)
    assert spread.shape == (nfreq, ndirections)
    waves['S'].values = spectrum * spread
    waves['phase'].values = _random_phase([nfreq, ndirections], seed)

    # attributes
    waves.attrs['Wave type'] = 'Irregular'
    waves.attrs['Spectrum'] = spectrum_name
    waves.attrs['Spreading function'] = spread_name

    return waves


def _degrees_to_radians(degrees: float | npt.ArrayLike
                       ) -> float | np.ndarray:
    """Convert degrees to radians in range -π to π and sort.
    """
    radians = np.asarray(np.remainder(np.deg2rad(degrees), 2*np.pi))
    radians[radians > np.pi] -= 2*np.pi
    radians = radians.item() if (radians.size == 1) else np.sort(radians)
    return radians


def _random_phase(shape: list[int] | int | None = None,
                 seed: float | None = None) -> float | np.ndarray:
    """Generate random phases in range -π to π radians.
    """
    rng = np.random.default_rng(seed)
    return rng.random(shape)*2*np.pi - np.pi


def pierson_moskowitz_spectrum(
    freq: float | npt.ArrayLike, fp: float, hs: float) -> float | np.ndarray:
    """Calculate the Pierson-Moskowitz omni-directional wave spectrum
    for the specified frequencies  and parameters.

    This is included as one example of a spectrum function.

    Parameters
    ----------
    freq: np.ndarray
        Wave frequencies.
    fp: float
        Peak frequency of the sea-state in :math:`Hz`.
    hs: float
        Significant wave height (zeroth moment) of the sea-state in
        :math:`m`.
    """
    b = (1.057*fp)**4
    a = hs**2 / 4 * b
    return a * freq**(-5) * np.exp(-b * freq**(-4))


def jonswap_spectrum(freq:float | npt.ArrayLike, fp:float, hs:float,
    gamma:float=3.3) -> float | np.ndarray:
    """Calculate the Joint North Sea Wave Project (JONSWAP)
    omni-directional wave spectrum for the specified frequencies and
    parameters.

    See, e.g., DNV-RP-C205

    Parameters
    ----------
    freq : float | npt.ArrayLike
        Wave frequencies in :math:`Hz`.
    fp : float
        Peak frequency in :math:`Hz`.
    hs : float
        Significant wave height in :math:`m`.
    gamma : float, optional
        Peakedness factor. The default is 3.3.

    Returns
    -------
    Sf : np.array
        Spectral density in :math:`m^2/Hz`

    Notes
    -----
    For ``gamma = 1``, the JONSWAP spectrum reduces to a Pierson-Moskowitz
    spectrum.

    """
    # spectral width parameter
    sigma_a = 0.07
    sigma_b = 0.09
    sigma = np.piecewise(freq,
                         condlist=[freq <= fp, freq > fp],
                         funclist=[sigma_a, sigma_b])


    # Pieson-Moskowitz spectrum
    Spm = 5/16 * hs**2 * fp**4 * freq**-5 * np.exp(-5/4 * (freq / fp)**-4)

    # Modify PM to get JONSWAP spectrum
    G = gamma**np.exp(-1*(freq - fp)**2/(2*sigma**2 * fp**2))
    alpha = 1/16*hs**2 * np.trapz(Spm*G, freq)**-1
    Sj = alpha * Spm * G

    return Sj


def spread_cos2s(freq: float | npt.ArrayLike,
                 directions: float | npt.ArrayLike,
                 dm: float, fp: float, s_max: float) -> float | np.ndarray:
    """Calculate the Cosine-2s spreading function for the specified
    frequencies and wave directions.

    This is included as one example of a spreading function.

    Parameters
    ----------
    freq: np.ndarray
        Wave frequencies.
    directions: np.ndarray
        Wave directions relative to mean/wind direction in degrees.
    dm: float
        Mean wave direction in degrees.
    fp: float
        Peak frequency of the sea-state in :math:`Hz`.
    s_max: float
        The spreading parameter. Larger values corresponds to less
        spread. For fully developed seas a value of 10 is a good choice.

    Returns
    -------
    np.ndarray
        Matrix of values of the spread function for each combination of
        frequency and wave direction.
    """
    freq = np.atleast_1d(freq)
    rdir = _degrees_to_radians(directions-dm)
    pow = np.ones(len(freq)) * 5.0
    pow[freq > fp] = -2.5
    s = s_max * (freq/fp)**pow
    cs = 2**(2*s-1)/np.pi * (gamma(s+1))**2/gamma(2*s+1)
    return (cs * np.power.outer(np.cos(rdir/2), 2*s)).T
