
import autograd.numpy as np
import xarray as xr

from WecOptTool.core import freq_array


def regular_wave(f0, num_freq, freq, amplitude, phase=0.0, direction=0.0):
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
    num_directions = 1
    freqs = freq_array(f0, num_freq)
    index = _find_index(freqs, freq)

    dims = ('omega', 'wave_direction')
    omega = freqs*2*np.pi
    wavedir = np.array([np.radians(direction)])
    rad_units = {'units': '(radians)'}
    coords = [(dims[0], omega, rad_units), (dims[1], wavedir, rad_units)]

    S = np.zeros([num_freq, num_directions])
    S[index] = 0.5 * amplitude**2 / f0
    attrs = {'units': 'm^2*s',  # TODO: verify
             'long_name': 'wave amplitude'}
    S = xr.DataArray(S, dims=dims, coords=coords, attrs=attrs)

    iphase = phase
    phase = np.zeros([num_freq, num_directions])
    phase[index] = np.radians(iphase)
    attrs = {'units': '(radians)',
             'long_name': 'wave phase'}
    phase = xr.DataArray(phase, dims=dims, coords=coords, attrs=attrs)

    attrs = {'Wave type': 'Regular',
             'Frequency (Hz)': freq,
             'Amplitude (m)': amplitude,
             'Phase (degrees)': iphase,
             'Direction (degrees)': direction,
             }

    return xr.Dataset({'S': S, 'phase': phase}, attrs=attrs)


def irregular_wave():
    """Using MHKiT """
    # TODO: implement
    raise NotImplementedError


def _find_index(array, value):
    # TODO: use xarray index?
    diff = np.abs(array - value)
    index = diff.argmin()
    if not np.isclose(diff[index], 0.0):
        msg = f'WARNING: requested value {value} is not in array. ' + \
            f'Using nearest value of {array[index]}.'
        print(msg)  # TODO: use `logging` or `warnings` library
    return index
