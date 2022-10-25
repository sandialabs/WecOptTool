"""Provide the wave definition structure and provide
functions for creating different common types of waves.

This module provides the (empty) dataset structure for waves in
:python:`wecopttool`.
It also provides functions for creating common types of waves such as
regular waves and irregular waves.
The dataset structure is an xarray.Dataset containing the following two
2D xarray.DataArrray: (1) the amplitude :python:`amplitude` (m^2) and
(2) the phase :python:`phase` (rad).
The 2D coordinates are: wave angular frequency :python:`omega` (rad/s)
and direction :python:`wave_direction` (rad).

Two omni-directional spectra and one spread function are included.
These serve as examples for creating your own omni-directional spectra
and spread functions, possibly leveraging other libraries (e.g., MHKiT).
"""


from __future__ import annotations


import logging
from typing import Callable, Mapping, Union, Optional, Iterable

import numpy as np
from numpy.typing import ArrayLike
from numpy import ndarray
from xarray import DataArray, Dataset
from scipy.special import gamma

from wecopttool.core import frequency, degrees_to_radians, frequency_parameters


# logger
_log = logging.getLogger(__name__)


def elevation_fd(
    f1: float,
    nfreq: int,
    directions: Union[float, ArrayLike],
    amplitudes: Optional[ArrayLike] = None,
    phases: Optional[ArrayLike] = None,
    attr: Optional[Mapping] = None
) -> DataArray:
    """Construct the complex wave elevation
    :py:class:`xarray.DataArray`.

    This is the complex wave elevation (m) indexed by radial frequency
    (rad/s) and wave direction (rad).
    The coordinate units and names match those from *Capytaine*.

    Parameters
    ----------
    f1
        Fundamental frequency :python:`f1` [Hz].
    nfreq
        Number of frequencies (not including zero frequency),
        i.e., :python:`freq = [0, f1, 2*f1, ..., nfreq*f1]`.
    directions
        Wave directions in degrees. 1D array.
    amplitudes:
        Wave elevation amplitude in meters.
    phases:
        Wave phases in degrees.
    attr:
        Additional attributes (metadata) to include in the
        :py:class:`xarray.DataArray`.
    """
    directions = np.atleast_1d(degrees_to_radians(directions, sort=False))
    ndirections = len(directions)
    freq = frequency(f1, nfreq, False)
    omega = freq*2*np.pi

    dims = ('omega', 'wave_direction')
    freq_attr = {'long_name': 'Wave frequency', 'units': 'rad/s'}
    dir_attr = {'long_name': 'Wave direction', 'units': 'rad'}
    coords = [(dims[0], omega, freq_attr), (dims[1], directions, dir_attr)]

    if amplitudes is None:
        amplitudes = np.zeros([nfreq, ndirections])

    if phases is None:
        phases = random_phase([nfreq, ndirections])
    else:
        phases = degrees_to_radians(phases, False)

    camplitude = amplitudes * np.exp(1j*phases)

    attr = {} if attr is None else attr
    attrs = {'units': 'm', 'long_name': 'Wave elevation'} | attr

    waves = DataArray(camplitude, dims=dims, coords=coords,
                         attrs=attrs, name='wave_elev')

    return waves.sortby(waves.wave_direction)


def regular_wave(
    f1: float,
    nfreq: int,
    freq: float,
    amplitude: float,
    phase: Optional[float] = None,
    direction: float = 0.0,
) -> DataArray:
    """Create the dataset for a regular wave.

    Parameters
    ----------
    f1
        Fundamental frequency :python:`f1` [Hz].
    nfreq
        Number of frequencies (not including zero frequency),
        i.e., :python:`freq = [0, f1, 2*f1, ..., nfreq*f1]`.
    freq
        Frequency (in Hz) of the regular wave. If :python:`freq` is not
        in the frequency array, the closest value is used and a warning
        is displayed.
    amplitude
        Amplitude (in meters) of the regular wave.
    phase
        Phase (in degrees) of the regular wave.
    direction
        Direction (in degrees) of the regular wave.
    """

    # attributes & index
    omega = freq*2*np.pi
    tmp_waves = elevation_fd(f1, nfreq, direction)
    iomega = tmp_waves.sel(omega=omega, method='nearest').omega.values
    ifreq = iomega/(2*np.pi)

    if not np.isclose(iomega, omega):
        _log.warning(
            f"Requested frequency {freq} Hz is not in array. " +
            f"Using nearest value of {ifreq} Hz."
        )

    attrs = {
        'Wave type': 'Regular',
        'Frequency (Hz)': ifreq,
        'Amplitude (m)': amplitude,
        'Phase (degrees)': phase,
        'Direction (degrees)': direction,
    }

    # phase
    if phase is None:
        rphase = random_phase()
        phase = np.degrees(rphase)
    else:
        rphase = degrees_to_radians(phase)

    # wave elevation
    tmp = np.zeros([nfreq, 1])
    waves = elevation_fd(f1, nfreq, direction, tmp, tmp, attrs)
    waves.loc[{'omega': iomega}] = amplitude  * np.exp(1j*rphase)

    return waves


def long_crested_wave(
    efth: DataArray,
    direction: Optional[float] = 0.0,
) -> DataArray:
    """Create a complex frequency-domain wave elevation from an
    omnidirectional spectrum.

    The omnidirectional spectrum is in the :python:`wavespectra` format.

    .. note:: The frequencies must be evenly-spaced with spacing equal
              to the first frequency. This is not always the case when
              e.g. reading from buoy data. Use interpolation as
              :python:`da.interp(freq=[...])`.

    Parameters
    ----------
    efth
        Omnidirection wave spectrum in units of m^2/Hz, in the format
        used by :python:`wavespectra`.
    direction
        Direction (in degrees) of the long-crested wave.

    """
    f1, nfreq = frequency_parameters(efth.freq.values, False)
    df = f1

    amplitudes = np.sqrt(2*efth.values * df)

    attr = {
        'Wave type': 'Long-crested irregular',
        'Direction (degrees)': direction,
    }

    return elevation_fd(f1, nfreq, direction, amplitudes, None, attr)


def irregular_wave(efth: DataArray) -> DataArray:
    """Create a complex frequency-domain wave elevation from a spectrum.

    The omnidirectional spectrum is in the :python:`wavespectra` format.

    .. note:: The frequencies must be evenly-spaced with spacing equal
              to the first frequency. This is not always the case when
              e.g. reading from buoy data. Use interpolation as
              :python:`da.interp(freq=[...])`.

    .. note:: The wave directions must also be evenly spaced.

    Parameters
    ----------
    efth
        Wave spectrum in units of m^2/Hz/deg, in the format used by
        :python:`wavespectra`.
    """
    f1, nfreq = frequency_parameters(efth.freq.values, False)
    directions = efth.dir.values
    df = f1
    dd = np.sort(directions)[1]-np.sort(directions)[0]

    amplitudes = np.sqrt(2*efth.values * df * dd)

    attr = {'Wave type': 'Irregular'}

    return elevation_fd(f1, nfreq, directions, amplitudes, None, attr)


def random_phase(
    shape: Optional[Union[Iterable[int], int]] = None,
    seed: Optional[float] = None,
) -> Union[float , ndarray]:
    """Generate random phases in range [-π, π) radians.

    Parameters
    ----------
    shape
        Shape of the output array of random phases.
    seed
        Seed for random number generator. Used for reproducibility.
        Generally should not be used except for testing.
    """
    rng = np.random.default_rng(seed)
    return rng.random(shape)*2*np.pi - np.pi


# TODO: Move everything below to wavespectra.construct
#       wavespectra is good at reading from simulations or measurements
#       but not good yet at constructing from parametric models.
def omnidirectional_spectrum(
    f1: float,
    nfreq: int,
    spectrum_func: Callable,
    spectrum_name: str = '',
) -> Dataset:
    """Create the dataset for an omnidirectional wave spectrum in the
    :python:`wavespectra` format.

    Examples
    --------
    Define wave parameters.

    >>> from wecopttool.waves import omnidirectional_spectrum as omnispec
    >>> from wecopttool.waves import pierson_moskowitz_spectrum as pm
    >>> Hs = 5
    >>> Tp = 6
    >>> fp = 1/Tp

    Generate the wave using a Pierson-Moskowitz idealized spectrum.

    >>> omnidirectional_spectrum = omnispec(
    ...     f1=fp/10,
    ...     nfreq=30,
    ...     spectrum_func=lambda f: pm(freq=f, fp=fp, hs=Hs),
    ...     spectrum_name="Pierson-Moskowitz",
    ...     )

    Parameters
    ----------
    f1
        Fundamental frequency :python:`f1` [Hz].
    nfreq
        Number of frequencies (not including zero frequency),
        i.e., :python:`freq = [0, f1, 2*f1, ..., nfreq*f1]`.
    spectrum_func
        Wave spectrum function. Maps frequencies to amplitude spectrum.
        :python:`Union[float, ArrayLike] -> Union[float, ndarray]`.
        Units: :math:`m^2/Hz`.
    spectrum_name
        Name of the spectrum function.
    """
    # dimensions & coordinates
    freq = frequency(f1, nfreq, False)
    dims = ('freq', 'dir')
    freq_attr = {
        'long_name': 'wave frequency',
        'units': 'Hz'
    }
    dir_attr = {
        'long_name': 'wave direction',
        'units': 'deg'
    }
    dir = [0.0,]
    coords = [(dims[0], freq, freq_attr), (dims[1], dir, dir_attr)]

    # spectrum
    efth = spectrum_func(freq).reshape(nfreq, 1)
    efth_attr = {
        'long_name': 'omnidirectional spectrum',
        'units': 'm^2/Hz',
        'Omnidirectional Spectrum': spectrum_name,
    }

    return DataArray(efth, dims=dims, coords=coords, attrs=efth_attr)


def spectrum(
    f1: float,
    nfreq: int,
    directions: Union[float, ArrayLike],
    spectrum_func: Callable,
    spread_func: Callable,
    spectrum_name: str = '',
    spread_name: str = '',
) -> Dataset:
    """Create the dataset for an irregular wave in the
    :python:`wavespectra` format.

    Examples
    --------
    Define the desired spectrum parameters.

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

    Generate the spectrum.

    >>> wave = wot.waves.spectrum(f1=fp/10,
    ...                           nfreq=20,
    ...                           directions=directions,
    ...                           spectrum_func=spectrum_func,
    ...                           spread_func=spread_func,
    ...                           spectrum_name="Pierson-Moskowitz",
    ...                           spread_name="cosine-2s",)

    Parameters
    ----------
    f1
        Fundamental frequency :python:`f1` [Hz].
    nfreq
        Number of frequencies (not including zero frequency),
        i.e., :python:`freq = [0, f1, 2*f1, ..., nfreq*f1]`.
    directions
        Wave directions in degrees. 1D array, evenly spaced.
    spectrum_func
        Wave spectrum function. Maps frequencies to amplitude spectrum.
        :python:`Union[float, ArrayLike] -> Union[float, np.ndarray]`.
        Units: :math:`m^2/Hz`.
    spread_func
        Wave spreading function.
        Maps wave frequencies and directions to spread value.
        :python:`tuple[ Union[float, ArrayLike], Union[float, ArrayLike]] -> ndarray`.
        Units: :math:`1/degree`.
    spectrum_name
        Name of the spectrum function.
    spread_name
        Name of the spread function.
    """
    # dimensions & coordinates
    freq = frequency(f1, nfreq, False)
    dims = ('freq', 'dir')
    freq_attr = {
        'long_name': 'wave frequency',
        'units': 'Hz'
    }
    dir_attr = {
        'long_name': 'wave direction',
        'units': 'deg'
    }
    dir = directions
    coords = [(dims[0], freq, freq_attr), (dims[1], dir, dir_attr)]

    # spectrum
    efth = spectrum_func(freq).reshape(nfreq, 1) * spread_func(freq, dir)
    efth_attr = {
        'long_name': 'spectrum',
        'units': 'm^2/Hz/deg',
        'Omnidirectional Spectrum': spectrum_name,
        'Spreading function': spread_name,
    }

    return DataArray(efth, dims=dims, coords=coords, attrs=efth_attr)


def pierson_moskowitz_spectrum(
    freq: Union[float, ArrayLike],
    fp: float,
    hs: float,
) -> Union[float, ndarray]:
    """Calculate the Pierson-Moskowitz omni-directional wave spectrum
    for the specified frequencies  and parameters.

    This is included as one example of a spectrum function.

    Return is in units of :math:`m^2/Hz`.

    Parameters
    ----------
    freq
        Wave frequencies.
    fp
        Peak frequency of the sea-state in :math:`Hz`.
    hs
        Significant wave height of the sea-state in :math:`m`.
    """
    a_param, b_param = pierson_moskowitz_params(fp, hs)
    return general_spectrum(a_param, b_param, freq)


def jonswap_spectrum(
    freq: Union[float, ArrayLike],
    fp: float,
    hs: float,
    gamma: float = 3.3,
) -> Union[float, ndarray]:
    """Calculate the Joint North Sea Wave Project (JONSWAP)
    omni-directional wave spectrum for the specified frequencies and
    parameters.

    See, e.g., :title:`DNV-RP-C205`

    For :python:`gamma = 1`, the JONSWAP spectrum reduces to a
    Pierson-Moskowitz spectrum.

    Return is in units of :math:`m^2/Hz`.

    Parameters
    ----------
    freq
        Wave frequencies in :math:`Hz`.
    fp
        Peak frequency in :math:`Hz`.
    hs
        Significant wave height in :math:`m`.
    gamma
        Peakedness factor.
    """
    # Pierson-Moskowitz parameters
    a_param_pm, b_param = pierson_moskowitz_params(fp, hs)

    # JONSWAP parameters
    sigma_a = 0.07
    sigma_b = 0.09
    sigma = np.piecewise(freq,
                         condlist=[freq <= fp, freq > fp],
                         funclist=[sigma_a, sigma_b])
    alpha = np.exp(-1*((freq/fp - 1)/(np.sqrt(2)*sigma))**2)
    c_param = 1-0.287*np.log(gamma)
    a_param = a_param_pm * c_param * gamma**alpha

    return general_spectrum(a_param, b_param, freq)


def spread_cos2s(
    freq: Union[float, ArrayLike],
    directions: Union[float, ArrayLike],
    dm: float,
    fp: float,
    s_max: float,
) -> Union[float, np.ndarray]:
    """Calculate the Cosine-2s spreading function for the specified
    frequencies and wave directions.

    This is included as one example of a spreading function.

    Return is in units of :math:`1/degrees`.

    Parameters
    ----------
    freq
        Wave frequencies in Hz.
    directions
        Wave directions relative to mean/wind direction in degrees.
    dm: float
        Mean wave direction in degrees.
    fp
        Peak frequency of the sea-state in :math:`Hz`.
    s_max
        The spreading parameter. Larger values corresponds to less
        spread. For fully developed seas a value of 10 is a good choice.
    """
    freq = np.atleast_1d(freq)
    rdir = degrees_to_radians(directions-dm, False)
    pow = np.ones(len(freq)) * 5.0
    pow[freq > fp] = -2.5
    s_param = s_max * (freq/fp)**pow
    cs = 2**(2*s_param-1)/np.pi * (gamma(s_param+1))**2/gamma(2*s_param+1)
    return np.pi/180 * (cs * np.power.outer(np.cos(rdir/2), 2*s_param)).T


def general_spectrum(
    a_param: Union[float, ArrayLike],
    b_param: Union[float, ArrayLike],
    freq: Union[float, ArrayLike],
) -> Union[float, ArrayLike]:
    """Create a spectrum function.

    The general omni-directional spectrum formulation is
    :math:`S(f) = A f^(-5) e^(-B f^(-4))`.

    Return is in units of :math:`m^2/Hz`.

    Parameters
    ----------
    a_param
        Parameter :math:`A` in the general spectrum equation.
    b_param
        Parameter :math:`B` in the general spectrum equation.
    freq
        Wave frequencies in Hz.
    """
    spectrum = a_param * freq**(-5) * np.exp(-b_param * freq**(-4))
    # if scalar, return scalar
    spectrum = spectrum.item() if (spectrum.size == 1) else spectrum
    return spectrum


def pierson_moskowitz_params(fp: float, hs: float) -> Union[float, ndarray]:
    """Return the two PM parameters for the general spectrum formulation.

    Parameters
    ----------
    fp
        Peak frequency in :math:`Hz`.
    hs
        Significant wave height in :math:`m`.

    Returns
    -------
    a_param
        Parameter :math:`A` in the general spectrum equation.
    b_param
        Parameter :math:`B` in the general spectrum equation.

    See Also
    --------
    general_spectrum,
    """
    b_param = (1.057*fp)**4
    a_param = hs**2 / 4 * b_param
    return a_param, b_param