""" Unit tests for functions in the :python:`waves.py` module.
"""

import os

import pytest
import capytaine as cpy
import numpy as np
import wavespectra as ws

import wecopttool as wot
from wecopttool.core import _default_parameters


@pytest.fixture()
def f1(): return 0.12


@pytest.fixture()
def nfreq(): return 5


@pytest.fixture()
def fp(): return 0.25


@pytest.fixture()
def hs(): return 0.1


@pytest.fixture()
def ndir(): return 90


@pytest.fixture()
def ndbc_spectrum():
    f1 = 0.02
    nfreq = 24
    time = '2020-01-01T01:40:00.000000000'
    freq = wot.frequency(f1, nfreq, False)
    markers = ('w', 'd', 'i', 'j', 'k')
    dir = os.path.join(os.path.dirname(__file__), 'data', 'ndbc')
    files = [f'41013{i}2020.txt' for i in markers]
    spec = ws.read_ndbc([os.path.join(dir, file) for file in files])
    return spec.sel(time=time).interp(freq=freq)


@pytest.fixture()
def ndbc_omnidirectional():
    f1 = 0.02
    nfreq = 24
    time = '2020-01-01T01:40:00.000000000'
    freq = wot.frequency(f1, nfreq, False)
    dir = os.path.join(os.path.dirname(__file__), 'data', 'ndbc')
    spec = ws.read_ndbc(os.path.join(dir, '41013w2020.txt'))
    return spec.sel(time=time).interp(freq=freq)


def test_elevation_fd(f1, nfreq):
    ndir = 90
    directions = np.linspace(0, 360, ndir, endpoint=False)
    elev = wot.waves.elevation_fd(f1, nfreq, directions)

    assert 'wave_direction' in elev.coords
    assert 'omega' in elev.coords
    assert np.squeeze(elev.values).shape == (nfreq, ndir)
    assert np.iscomplexobj(elev)
    assert np.allclose(np.abs(elev), 0.0)


def test_regular_wave(f1, nfreq):
    freq = f1*np.random.randint(1, nfreq)
    amp = 2.5 * np.random.random()
    phase = np.random.random() * 360
    dir = np.random.random() * 360
    elev = wot.waves.regular_wave(f1, nfreq, freq, amp, phase, dir)

    elev0 = elev.copy()
    idx = np.where(
        elev.omega.values==elev.sel(omega=2*np.pi*freq).omega.values)
    elev0.values[idx] = 0

    assert 'wave_direction' in elev.coords
    assert 'omega' in elev.coords
    assert np.squeeze(elev.values).shape == (nfreq,)
    assert np.iscomplexobj(elev)
    assert np.isclose(
        elev.wave_direction.values.item(), wot.degrees_to_radians(dir))
    assert np.isclose(
        elev.sel(omega=freq*2*np.pi).values,
        amp*np.exp(1j*wot.degrees_to_radians(phase)))
    assert np.allclose(elev0.values, 0.0+0.0j)


def test_long_crested_wave(ndbc_omnidirectional):
    nfreq = len(ndbc_omnidirectional.freq)
    ndir= len(ndbc_omnidirectional.dir)

    direction = ndbc_omnidirectional.dir.values[np.random.randint(0, ndir)]
    elev = wot.waves.long_crested_wave(ndbc_omnidirectional.efth, direction)

    assert 'wave_direction' in elev.coords
    assert 'omega' in elev.coords
    assert np.squeeze(elev.values).shape == (nfreq,)
    assert np.iscomplexobj(elev)
    assert len(elev.wave_direction) == 1
    assert elev.wave_direction.values.item() == direction


def test_irregular_wave(ndbc_spectrum):
    nfreq = len(ndbc_spectrum.freq)
    ndir= len(ndbc_spectrum.dir)

    elev = wot.waves.irregular_wave(ndbc_spectrum.efth)

    assert 'wave_direction' in elev.coords
    assert 'omega' in elev.coords
    assert np.squeeze(elev.values).shape == (nfreq, ndir)
    assert np.iscomplexobj(elev)


def test_random_phase():
    shape = (np.random.randint(10, 100), np.random.randint(10, 100))
    phase = wot.waves.random_phase(shape)
    phase1 = wot.waves.random_phase()

    assert phase.shape == shape
    assert np.max(phase) < np.pi
    assert np.min(phase) >= -np.pi
    assert (phase1 < np.pi) and (phase1 >= -np.pi)
    assert isinstance(phase1, float)


def test_omnidirectional_spectrum(f1, nfreq, fp, hs):
    spectrum_func = lambda f: wot.waves.pierson_moskowitz_spectrum(f, fp, hs)
    wave_spec = wot.waves.omnidirectional_spectrum(
        f1, nfreq, spectrum_func, "Pierson-Moskowitz")

    # the values should be the same as calling the spectrum function
    freq = wot.frequency(f1, nfreq, False)
    spec_test = spectrum_func(freq)

    assert np.allclose(spec_test, wave_spec.values.flatten())


def test_spectrum(f1, nfreq, fp, hs, ndir):
    s_max = 10
    directions = np.linspace(0, 360, ndir, endpoint=False)
    dm = directions[np.random.randint(0, ndir)]

    spectrum_func = lambda f: wot.waves.pierson_moskowitz_spectrum(f, fp, hs)
    spread_func = lambda f,d: wot.waves.spread_cos2s(f, d, dm, fp, s_max)
    spectrum_name, spread_name = "Pierson-Moskowitz", "Cos2s"
    wave_spec = wot.waves.spectrum(
        f1, nfreq, directions, spectrum_func, spread_func,
        spectrum_name, spread_name)

    # integral over all angles should be equal to omnidirectional
    spec_omni = wot.waves.omnidirectional_spectrum(
        f1, nfreq, spectrum_func, spectrum_name)
    spec_omni = spec_omni.values.flatten()
    ddir = (wave_spec.dir[1] - wave_spec.dir[0]).values
    integral_d = wave_spec.sum(dim = 'dir').values * ddir

    # mean direction
    dfreq = (wave_spec.freq[1] - wave_spec.freq[0]).values
    integral_f = wave_spec.sum(dim = 'freq').values * dfreq

    assert wave_spec.shape == (nfreq, ndir)  # shape
    assert np.allclose(integral_d, spec_omni, rtol=0.01)
    assert directions[np.argmax(integral_f)] == dm


def test_pierson_moskowitz_spectrum(f1, nfreq, fp, hs):
    spectrum = wot.waves.pierson_moskowitz_spectrum

    # scalar
    freq_1 = 0.4
    spec1 = spectrum(freq_1, fp, hs)

    # vector
    freqs = wot.frequency(f1, nfreq, False)
    spec = spectrum(freqs, fp, hs)

    # total elevation variance
    freqs_int = np.linspace(0, 10, 1000)[1:]
    total_variance_calc = np.trapz(spectrum(freqs_int, fp, hs), freqs_int)
    a_param, b_param = wot.waves.pierson_moskowitz_params(fp, hs)
    total_variance_theory = a_param/(4*b_param)

    assert isinstance(spec1, float)  # scalar
    assert spec.shape == freqs.shape  # vector shape
    assert np.isclose(total_variance_calc, total_variance_theory)  # integral


def test_jonswap_spectrum(f1, nfreq, fp, hs):
    spectrum = wot.waves.jonswap_spectrum

    # scalar
    freq_1 = 0.4
    spec1 = spectrum(freq_1, fp, hs)

    # vector
    freqs = wot.frequency(f1, nfreq, False)
    spec = spectrum(freqs, fp, hs)

    # reduces to PM
    spec_gamma1 = spectrum(freqs, fp, hs, gamma=1.0)
    spec_pm = wot.waves.pierson_moskowitz_spectrum(freqs, fp, hs)

    assert isinstance(spec1, float)  # scalar
    assert spec.shape == freqs.shape  # vector shape
    assert np.allclose(spec_gamma1, spec_pm)  # reduces to PM


def test_spread_cos2s(f1, nfreq, fp, ndir):
    """Confirm that energy is spread correctly accross wave directions.
    Integral over all directions of the spread function gives (vector)
    1.
    """
    directions = np.linspace(0, 360, ndir, endpoint=False)
    wdir_mean = directions[np.random.randint(0, ndir)]
    freqs = wot.frequency(f1, nfreq, False)
    s_max = round(np.random.random()*10)
    spread = wot.waves.spread_cos2s(freq = freqs,
                                    directions = directions,
                                    dm = wdir_mean,
                                    fp = fp,
                                    s_max = s_max)
    ddir = directions[1]-directions[0]
<<<<<<< HEAD
    dfreq = freqs[1] - freqs[0]
    integral_d = np.sum(spread, axis=1)*ddir
    integral_f = np.sum(spread, axis=0)*dfreq

    assert directions[np.argmax(integral_f)] == wdir_mean  # mean dir
    assert np.allclose(integral_d, np.ones((1, nfreq)), rtol=0.01) # omnidir


def test_general_spectrum(f1, nfreq):
    freq = wot.frequency(f1, nfreq, False)
    a_param = np.random.random()*10
    b_param = np.random.random()*10
    spec_f1 = wot.waves.general_spectrum(a_param, b_param, 1.0)
    spec_a0 = wot.waves.general_spectrum(0, b_param, freq)
    spec_b0 = wot.waves.general_spectrum(a_param, 0, freq)

    a_vec = np.random.random(freq.shape)*10
    b_vec = np.random.random(freq.shape)*10
    spec_vec = wot.waves.general_spectrum(a_vec, b_vec, freq)

    # types and shapes
    assert isinstance(spec_f1, float)
    assert spec_a0.shape == spec_b0.shape == freq.shape
    assert spec_vec.shape == freq.shape
    # values
    assert np.isclose(spec_f1, a_param * np.exp(-b_param))
    assert np.allclose(spec_a0, 0.0)
    assert np.allclose(spec_b0, a_param * freq**(-5))


def test_pierson_moskowitz_params(fp, hs):
    params = wot.waves.pierson_moskowitz_params(fp, hs)

    assert len(params) == 2  # returns two floats
    for iparam in params:
        assert isinstance(iparam, float)
=======

    rtol = 0.01
    assert np.allclose(np.sum(spread, axis = 1)*ddir,
                       np.ones((1, nfreq)),
                       rtol
                       )


def test_spectrum_energy(irregular_wave, long_crested_wave):
    """Confirm that energy is spread correctly accross wave directions.

    Integral (sum) over all directions of the long crested irregular
    wave (2D) spectrum gives the omni-direction spectrum (vector).
    """
    wdir_step = (irregular_wave.wave_direction[1]
                - irregular_wave.wave_direction[0])
    # TODO: make units consistent in the wave xarray
    rtol= 0.01
    w1 = wdir_step.values * irregular_wave.S.sum(dim = 'wave_direction').values
    w2 = (long_crested_wave.S.values).T
    assert np.allclose(w1, w2, rtol)
>>>>>>> main
