"""Provide the wave definition structure and provide
functions for creating different common types of waves.

This module provides the (empty) data structure for waves in
:python:`wecopttool`.
It also provides functions for creating common types of waves such as
regular waves and irregular waves.
The data structure is a 2D complex :py:class:`xarray.DataArray`
containing the complex amplitude.
The 2D coordinates are: wave angular frequency :python:`omega` (rad/s)
and direction :python:`wave_direction` (rad).

This module uses wave spectrum data in the
:py:class:`wavespectra.SpecArray` format, but does not require that you
use :py:class:`wavespectra.SpecArray` objects.
"""


from __future__ import annotations


__all__ = [
    "elevation_fd",
    "regular_wave",
    "long_crested_wave",
    "irregular_wave",
    "random_phase",
]


import logging
from typing import Callable, Mapping, Union, Optional, Iterable

import numpy as np
from numpy.typing import ArrayLike
from numpy import ndarray
from xarray import DataArray
from scipy.special import gamma

from wecopttool.core import frequency, degrees_to_radians, frequency_parameters


# logger
_log = logging.getLogger(__name__)


def elevation_fd(
    f1: float,
    nfreq: int,
    directions: Union[float, ArrayLike],
    nrealizations: int,
    amplitudes: Optional[ArrayLike] = None,
    phases: Optional[ArrayLike] = None,
    attr: Optional[Mapping] = None,
    seed: Optional[float] = None,
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
    nrealizations
        Number of wave phase realizations.
    amplitudes:
        Wave elevation amplitude in meters.
    phases:
        Wave phases in degrees.
    attr:
        Additional attributes (metadata) to include in the
        :py:class:`xarray.DataArray`.
    seed
        Seed for random number generator. Used for reproducibility.
        Generally should not be used except for testing.
    """
    directions = np.atleast_1d(degrees_to_radians(directions, sort=False))
    ndirections = len(directions)
    realization = range(nrealizations)
    freq = frequency(f1, nfreq, False)
    omega = freq*2*np.pi

    dims = ('omega', 'wave_direction', 'realization')
    omega_attr = {'long_name': 'Radial frequency', 'units': 'rad/s'}
    freq_attr = {'long_name': 'Frequency', 'units': 'Hz'}
    dir_attr = {'long_name': 'Wave direction', 'units': 'rad'}
    real_attr = {'long_name': 'Phase realization', 'units': ''}
    coords = {'omega': (dims[0], omega, omega_attr),
              'freq': (dims[0], freq, freq_attr),
              'wave_direction': (dims[1], directions, dir_attr),
              'realization': (dims[2], realization, real_attr)}

    if amplitudes is None:
        amplitudes = np.zeros([nfreq, ndirections, nrealizations])
    else:
        if amplitudes.shape == (nfreq, ndirections):
            amplitudes = np.expand_dims(amplitudes,axis=2)
        assert amplitudes.shape == (nfreq, ndirections, 1) or \
                amplitudes.shape == (nfreq, ndirections, nrealizations)

    if phases is None:
        phases = random_phase([nfreq, ndirections, nrealizations], seed)
    else:
        phases = degrees_to_radians(phases, False)
        assert phases.shape == (nfreq, ndirections, nrealizations)

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
    tmp_waves = elevation_fd(f1, nfreq, direction, 1)
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
    tmp = np.zeros([nfreq, 1, 1])
    waves = elevation_fd(f1, nfreq, direction, 1, tmp, tmp, attrs)
    waves.loc[{'omega': iomega}] = amplitude  * np.exp(1j*rphase)

    return waves


def long_crested_wave(
    efth: DataArray,
    nrealizations: int,
    direction: Optional[float] = 0.0,
    seed: Optional[float] = None,
) -> DataArray:
    """Create a complex frequency-domain wave elevation from an
    omnidirectional spectrum.

    The spectrum is a :py:class:`xarray.DataArray` in the format used
    by :py:class:`wavespectra.SpecArray`.

    .. note:: The frequencies must be evenly-spaced with spacing equal
              to the first frequency. This is not always the case when
              e.g. reading from buoy data. Use interpolation as
              :python:`da.interp(freq=[...])`.

    Parameters
    ----------
    efth
        Omnidirection wave spectrum in units of m^2/Hz, in the format
        used by :py:class:`wavespectra.SpecArray`.
    nrealizations
        Number of wave phase realizations to be created for the
        long-crested wave.
    direction
        Direction (in degrees) of the long-crested wave.
    seed
        Seed for random number generator. Used for reproducibility.
        Generally should not be used except for testing.
    """
    f1, nfreq = frequency_parameters(efth.freq.values, False)
    df = f1

    if 'dir' not in efth.dims:
        efth = DataArray(
            data=np.expand_dims(efth, 1),
            dims=["freq", "dir"],
            coords=dict(freq=efth['freq'], dir=np.array([direction])),
            name="efth",
        )

    values = efth.values
    values[values<0] = np.nan
    amplitudes = np.sqrt(2 * values * df)

    attr = {
        'Wave type': 'Long-crested irregular',
        'Direction (degrees)': direction,
    }

    return elevation_fd(f1, nfreq, direction, nrealizations, amplitudes, None, attr, seed)


def irregular_wave(efth: DataArray,
                   nrealizations: int,
                   seed: Optional[float] = None,) -> DataArray:
    """Create a complex frequency-domain wave elevation from a spectrum. 

    The spectrum is a :py:class:`xarray.DataArray` in the format used
    by :py:class:`wavespectra.SpecArray`.
    The spectrum is a :py:class:`xarray.DataArray` in the format used 
    by :py:class:`wavespectra.SpecArray`. For generating wave spectra 
    with directional spreading, see 
    :py:class:`wavespectra.construction.direction`.
    
    .. note:: The frequencies must be evenly-spaced with spacing equal
              to the first frequency. This is not always the case when
              e.g. reading from buoy data. Use interpolation as
              :python:`da.interp(freq=[...])`.

    .. note:: The wave directions must also be evenly spaced. 

    Parameters
    ----------
    efth
        Wave spectrum in units of m^2/Hz/deg, in the format used by
        :py:class:`wavespectra.SpecArray`.
    nrealizations
        Number of wave phase realizations to be created for the
        irregular wave.
    seed
        Seed for random number generator. Used for reproducibility.
        Generally should not be used except for testing.
    """
    f1, nfreq = frequency_parameters(efth.freq.values, False)
    directions = efth.dir.values
    df = f1
    dd = np.sort(directions)[1]-np.sort(directions)[0]

    values = efth.values
    values[values<0] = np.nan
    amplitudes = np.sqrt(2 * values * df * dd)

    attr = {'Wave type': 'Irregular'}

    return elevation_fd(f1, nfreq, directions, nrealizations, amplitudes, None, attr, seed)


def random_phase(
    shape: Optional[Union[Iterable[int], int, int]] = None,
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
