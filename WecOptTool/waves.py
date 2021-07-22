
import warnings

import autograd.numpy as np
import numpy as npo
import xarray as xr
from scipy.special import gamma

from WecOptTool.core import freq_array


def regular_wave(f0, num_freq, freq, amplitude, phase=None, direction=0.0):
    """

    Parameters
    ----------
    f0: float
        Initial frequency (in Hz) for frequency array.
        Frequency array given as [f0, 2*f0, ..., num_freq*f0].
    num_freq: int
        Number of frequencies in frequency array. See ``f0``.
    freq: float
        Frequency (in Hz) of the regular wave. If ``freq`` not in the
        frequency array, the closest value is used and a warning is
        displayed.
    amplitude: float
        Amplitude (in m) of the regular wave.
    phase: float
        Phase (in degrees) of the regular wave.
    direction: float
        Direction (in degrees) of the regular wave.

    Returns
    -------
     xr.Dataset
        2D Dataset containing the amplitude spectrum  magnitude ``S``
        (m^2*s) and phase ``phase`` (rad) with coordinates: wave
        frequency ``omega`` (rad) and direction ``wave_direction``
        (rad).
    """
    # empty dataset
    waves = empty_wave_dataset(f0, num_freq, direction)

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
        rphase = random_phase()
        phase = np.degrees(rphase)
    else:
        rphase = degrees_to_radians(phase)
    waves['phase'].loc[{'omega': iomega}] = rphase

    # attributes
    waves.attrs = {'Wave type': 'Regular',
                   'Frequency (Hz)': ifreq,
                   'Amplitude (m)': amplitude,
                   'Phase (degrees)': phase,
                   'Direction (degrees)': direction,
                  }

    return waves


def long_crested_wave(f0, num_freq, spectrum_func,
                      direction=0.0, spectrum_name='', seed=None):
    """
    """
    # empty dataset
    waves = empty_wave_dataset(f0, num_freq, direction)

    # amplitude & phase
    freqs = freq_array(f0, num_freq)
    waves['S'].values = spectrum_func(freqs).reshape(num_freq, 1)
    waves['phase'].values = random_phase([num_freq, 1], seed)

    # attributes
    waves.attrs['Wave type'] = 'Long-crested irregular'
    waves.attrs['Direction (degrees)'] = direction
    waves.attrs['Spectrum'] = spectrum_name

    return waves


def irregular_wave(f0, num_freq, spectrum_func, spread_func, directions,
                   spectrum_name='', spread_name='', seed=None):
    """
    """
    # empty dataset
    waves = empty_wave_dataset(f0, num_freq, directions)

    # amplitude & phase
    num_directions = len(directions)
    freqs = freq_array(f0, num_freq)
    spectrum = spectrum_func(freqs).reshape(num_freq, 1)
    spread = spread_func(freqs, directions)
    assert spread.shape == (num_freq, num_directions)
    waves['S'].values = spectrum * spread
    waves['phase'].values = random_phase([num_freq, num_directions], seed)

    # attributes
    waves.attrs['Wave type'] = 'Irregular'
    waves.attrs['Spectrum'] = spectrum_name
    waves.attrs['Spreading function'] = spread_name

    return waves


def empty_wave_dataset(f0, num_freq, directions):
    """ Create an empty dataset with correct dimensions.
    """
    directions = np.atleast_1d(degrees_to_radians(directions))
    num_directions = len(directions)
    freqs = freq_array(f0, num_freq)
    omega = freqs*2*np.pi

    dims = ('omega', 'wave_direction')
    rad_units = {'units': '(radians)'}
    coords = [(dims[0], omega, rad_units), (dims[1], directions, rad_units)]
    tmp = np.zeros([num_freq, num_directions])

    attrs = {'units': 'm^2*s', 'long_name': 'wave amplitude'}
    S = xr.DataArray(tmp, dims=dims, coords=coords, attrs=attrs)

    attrs = {'units': '(radians)', 'long_name': 'wave phase'}
    phase = xr.DataArray(tmp.copy(), dims=dims, coords=coords, attrs=attrs)

    return xr.Dataset({'S': S, 'phase': phase}, attrs={})


def degrees_to_radians(degrees):
    """ Convert degrees to radians in range -π to π and sort.
    """
    radians = np.asarray(np.remainder(np.deg2rad(degrees), 2*np.pi))
    radians[radians > np.pi] -= 2*np.pi
    radians = radians.item() if radians.size==1 else np.sort(radians)
    return radians


def random_phase(shape=None, seed=None):
    """ Generate random phases in range -π to π radians.
    """
    rng = np.random.default_rng(seed)
    return rng.random(shape)*2*np.pi - np.pi


def spread_cos2s(freq, dir, fp, s_max):
    freq = np.atleast_1d(freq)
    rdir = degrees_to_radians(dir)
    pow = np.ones(len(freq)) * 5.0
    pow[freq>fp] = -2.5
    s = s_max * (freq/fp)**pow
    Cs = 2**(2*s-1)/np.pi * (gamma(s+1))**2/gamma(2*s+1)
    return (Cs * npo.power.outer(np.cos(rdir/2), 2*s)).T
