""" Unit tests for functions in the :python:`waves.py` module.
"""

import os

import pytest
import capytaine as cpy
import numpy as np
import wavespectra as ws
from scipy import signal

import wecopttool as wot
from wecopttool.core import _default_parameters


# setup wave structure
@pytest.fixture(scope="module")
def f1():
    """Fundamental frequency [Hz]."""
    return 0.12


@pytest.fixture(scope="module")
def nfreq():
    """Number of frequencies in frequency vector."""
    return 5


@pytest.fixture(scope="module")
def fp():
    """Peak frequency [Hz] of the sea state."""
    return 0.25


@pytest.fixture(scope="module")
def hs():
    """Significan wave height [m] of the sea state."""
    return 0.1


@pytest.fixture(scope="module")
def ndir():
    """Number of directions to consider in the range [0, 360)
    degrees.
    """
    return 90


@pytest.fixture(scope="module")
def directions(ndir):
    """Vector of wave directions."""
    return np.linspace(0, 360, ndir, endpoint=False)


class TestElevationFD:
    """Test function :python:`waves.elevation_fd`."""

    @pytest.fixture(scope="class")
    def elevation(self, f1, nfreq, directions):
        """Complex sea state elevation amplitude [m] indexed by
        frequency and direction."""
        return wot.waves.elevation_fd(f1, nfreq, directions)

    def test_coordinates(self, elevation):
        """Test that the elevation dataArray has the correct
        coordinates.
        """
        coordinates = ['wave_direction', 'omega', 'freq']
        for icoord in coordinates:
            assert icoord in elevation.coords, f'missing coordinate: {icoord}'

    def test_shape(self, elevation, nfreq, ndir):
        """Test that the elevation dataArray has the correct shape."""
        assert np.squeeze(elevation.values).shape == (nfreq, ndir)

    def test_type(self, elevation):
        """Test that the elevation dataArray has the correct type."""
        assert np.iscomplexobj(elevation)

    def test_empty(self, elevation):
        """Test that the elevation dataArray is initialized with zeros.
        """
        assert np.allclose(np.abs(elevation), 0.0)


class TestRegularWave:
    """Test function :python:`waves.regular_wave`."""

    @pytest.fixture(scope="class")
    def dir(self,):
        """Wave direction [degrees]."""
        return np.random.random()*360

    @pytest.fixture(scope="class")
    def freq(self, f1, nfreq):
        """Wave frequency [Hz]."""
        return f1*np.random.randint(1, nfreq)

    @pytest.fixture(scope="class")
    def amp(self,):
        """Wave amplitude [m]."""
        return 2.5 * np.random.random()

    @pytest.fixture(scope="class")
    def phase(self,):
        """Wave phase [degrees]."""
        return np.random.random() * 360

    @pytest.fixture(scope="class")
    def elevation(self, f1, nfreq, freq, amp, phase, dir):
        """Complex sea state elevation amplitude [m] indexed by
        frequency and direction."""
        return wot.waves.regular_wave(f1, nfreq, freq, amp, phase, dir)

    def test_coordinates(self, elevation):
        """Test that the elevation dataArray has the correct
        coordinates.
        """
        coordinates = ['wave_direction', 'omega', 'freq']
        for icoord in coordinates:
            assert icoord in elevation.coords, f'missing coordinate: {icoord}'

    def test_shape(self, elevation, nfreq, ndir):
        """Test that the elevation dataArray has the correct shape."""
        assert np.squeeze(elevation.values).shape == (nfreq, )

    def test_type(self, elevation):
        """Test that the elevation dataArray has the correct type."""
        assert np.iscomplexobj(elevation)

    def test_direction(self, elevation, dir):
        """Test that the wave direction is correct."""
        dir_out = elevation.wave_direction.values.item()
        assert np.isclose(dir_out, wot.degrees_to_radians(dir))

    def test_value(self, elevation, freq, amp, phase):
        """Test that the complex amplitude [m] has the correct value."""
        amp_out = elevation.sel(omega=freq*2*np.pi).values
        amp_correct = amp*np.exp(1j*wot.degrees_to_radians(phase))
        assert np.isclose(amp_out, amp_correct)

    def test_only_one_value(self, elevation, freq):
        """Test that all other values are zero."""
        # set the single entry to zero
        elev0 = elevation.copy()
        omega = elevation.sel(omega=2*np.pi*freq).omega.values
        idx = np.where(elevation.omega.values == omega)
        elev0.values[idx] = 0
        # check that now the entire array is zero.
        assert np.allclose(elev0.values, 0.0+0.0j)


class TestLongCrestedWave:
    """Test function :python:`waves.long_crested_wave`."""

    @pytest.fixture(scope="class")
    def ndbc_omnidirectional(self,):
        """Omnidirectional spectrum from NDBC data interpolated at
        desired frequencies.
        """
        f1 = 0.02
        time = '2020-01-01T01:40:00.000000000'
        nfreq = 24
        freq = wot.frequency(f1, nfreq, False)
        dir = os.path.join(os.path.dirname(__file__), 'data', 'ndbc')
        spec = ws.read_ndbc_ascii(os.path.join(dir, '41013w2020.txt'))
        return spec.sel(time=time).interp(freq=freq)

    @pytest.fixture(scope="class")
    def nfreq(self, ndbc_omnidirectional):
        """Number of wave frequencies."""
        return len(ndbc_omnidirectional.freq)

    @pytest.fixture(scope="class")
    def ndir(self, ndbc_omnidirectional):
        """Number of wave directions."""
        return len(ndbc_omnidirectional.dir)

    @pytest.fixture(scope="class")
    def direction(self, ndbc_omnidirectional, ndir):
        """Wave direction."""
        return ndbc_omnidirectional.dir.values[np.random.randint(0, ndir)]

    @pytest.fixture(scope="class")
    def elevation(self, ndbc_omnidirectional, direction):
        """Complex sea state elevation amplitude [m] indexed by
        frequency and direction."""
        elev = wot.waves.long_crested_wave(
            ndbc_omnidirectional.efth, direction)
        return elev

    @pytest.fixture(scope="class")
    def pm_f1(self,):
        """Fundamental frequency for the Pierson-Moskowitz spectrum."""
        return 0.05

    @pytest.fixture(scope="class")
    def pm_nfreq(self,):
        """Number of frequencies for the Pierson-Moskowitz spectrum."""
        return 100

    @pytest.fixture(scope="class")
    def pm_hs(self,):
        """Significant wave height for Pierson-Moskowitz spectrum."""
        return 5.0

    @pytest.fixture(scope="class")
    def pm_spectrum(self, pm_f1, pm_nfreq, pm_hs):
        """Pierson-Moskowitz spectrum."""
        Tp = 1.2
        Hs = pm_hs

        def spectrum_func(f):
            return wot.waves.pierson_moskowitz_spectrum(f, fp=1/Tp, hs=Hs)
        spectrum_name = f"Pierson-Moskowitz ({Tp}s, {Hs}m)"

        efth_xr = wot.waves.omnidirectional_spectrum(
            pm_f1, pm_nfreq, spectrum_func, spectrum_name
        )
        return efth_xr

    def test_coordinates(self, elevation):
        """Test that the elevation dataArray has the correct
        coordinates.
        """
        coordinates = ['wave_direction', 'omega', 'freq']
        for icoord in coordinates:
            assert icoord in elevation.coords, f'missing coordinate: {icoord}'

    def test_shape(self, elevation, nfreq, ndir):
        """Test that the elevation dataArray has the correct shape."""
        assert np.squeeze(elevation.values).shape == (nfreq, )

    def test_type(self, elevation):
        """Test that the elevation dataArray has the correct type."""
        assert np.iscomplexobj(elevation)

    def test_direction(self, elevation, direction):
        """Test that the wave direction is correct."""
        dir_out = elevation.wave_direction.values.item()
        assert np.isclose(dir_out, wot.degrees_to_radians(direction))

    def test_spectrum(self, pm_spectrum, pm_hs):
        """Test that the constructed spectrum has the expected Hs."""
        efth = ws.SpecArray(pm_spectrum)
        assert np.isclose(pm_hs, efth.hs().values)

    def test_time_series(self, pm_spectrum, pm_f1, pm_nfreq):
        """Test that the created time series has the desired spectrum."""
        # create time-series
        direction = 0.0
        wave = wot.waves.long_crested_wave(pm_spectrum, direction)
        wave_ts = wot.fd_to_td(wave.values, pm_f1, pm_nfreq, False)
        # calculate the spectrum from the time-series
        t = wot.time(pm_f1, pm_nfreq)
        fs = 1/t[1]
        nnft = len(t)
        [_, S_data] = signal.welch(
            wave_ts.squeeze(), fs=fs, window='boxcar', nperseg=nnft, nfft=nnft,
            noverlap=0
        )
        # check it is equal to the original spectrum
        assert np.allclose(S_data[1:-1], pm_spectrum.values.squeeze()[:-1])


class TestIrregularWave:
    """Test function :python:`waves.irregular_wave`."""

    @pytest.fixture(scope="class")
    def ndbc_spectrum(self,):
        """Spectrum from NDBC data interpolated at desired frequencies.
        """
        f1 = 0.02
        nfreq = 24
        time = '2020-01-01T01:40:00.000000000'
        freq = wot.frequency(f1, nfreq, False)
        markers = ('w', 'd', 'i', 'j', 'k')
        dir = os.path.join(os.path.dirname(__file__), 'data', 'ndbc')
        files = [f'41013{i}2020.txt' for i in markers]
        spec = ws.read_ndbc_ascii([os.path.join(dir, file) for file in files])
        return spec.sel(time=time).interp(freq=freq)

    @pytest.fixture(scope="class")
    def elevation(self, ndbc_spectrum):
        """Complex sea state elevation amplitude [m] indexed by
        frequency and direction."""
        return wot.waves.irregular_wave(ndbc_spectrum.efth)

    def test_coordinates(self, elevation):
        """Test that the elevation dataArray has the correct
        coordinates.
        """
        coordinates = ['wave_direction', 'omega', 'freq']
        for icoord in coordinates:
            assert icoord in elevation.coords, f'missing coordinate: {icoord}'

    def test_shape(self, ndbc_spectrum, elevation):
        """Test that the elevation dataArray has the correct shape."""
        nfreq = len(ndbc_spectrum.freq)
        ndir = len(ndbc_spectrum.dir)
        assert np.squeeze(elevation.values).shape == (nfreq, ndir)

    def test_type(self, elevation):
        """Test that the elevation dataArray has the correct type."""
        assert np.iscomplexobj(elevation)


class TestRandomPhase:
    """Test function :python:`waves.random_phase`."""

    @pytest.fixture(scope="class")
    def shape(self,):
        """Shape of the phase matrix, randomized each time the test is
        run.
        """
        return (np.random.randint(10, 100), np.random.randint(10, 100))

    @pytest.fixture(scope="class")
    def phase_mat(self, shape):
        """Random phase matrix."""
        return wot.waves.random_phase(shape)

    def test_mat_shape(self, phase_mat, shape):
        """Test correct shape of random phase matrix."""
        assert phase_mat.shape == shape

    def test_mat_values(self, phase_mat):
        """Test all phases are between [-pi, pi])."""
        assert (np.max(phase_mat) < np.pi) and (np.min(phase_mat) >= -np.pi)

    def test_float(self,):
        """Test that the function works for a single float instead of a
        matrix.
        """
        phase = wot.waves.random_phase()
        assert (phase < np.pi) and (phase >= -np.pi)


# TODO: Move everything below to wavespectra.construct
class TestWaveSpectra:
    def test_omnidirectional_spectrum(self, f1, nfreq, fp, hs):
        def spectrum_func(
            f): return wot.waves.pierson_moskowitz_spectrum(f, fp, hs)
        wave_spec = wot.waves.omnidirectional_spectrum(
            f1, nfreq, spectrum_func, "Pierson-Moskowitz")

        # the values should be the same as calling the spectrum function
        freq = wot.frequency(f1, nfreq, False)
        spec_test = spectrum_func(freq)

        assert np.allclose(spec_test, wave_spec.values.flatten())

    def test_spectrum(self, f1, nfreq, fp, hs, ndir):
        s_max = 10
        directions = np.linspace(0, 360, ndir, endpoint=False)
        dm = directions[np.random.randint(0, ndir)]

        def spectrum_func(f): 
            return wot.waves.pierson_moskowitz_spectrum(f, fp, hs)

        def spread_func(f, d): 
            return wot.waves.spread_cos2s(f, d, dm, fp, s_max)
        spectrum_name, spread_name = "Pierson-Moskowitz", "Cos2s"
        wave_spec = wot.waves.spectrum(
            f1, nfreq, directions, spectrum_func, spread_func,
            spectrum_name, spread_name)

        # integral over all angles should be equal to omnidirectional
        spec_omni = wot.waves.omnidirectional_spectrum(
            f1, nfreq, spectrum_func, spectrum_name)
        spec_omni = spec_omni.values.flatten()
        ddir = (wave_spec.dir[1] - wave_spec.dir[0]).values
        integral_d = wave_spec.sum(dim='dir').values * ddir

        # mean direction
        dfreq = (wave_spec.freq[1] - wave_spec.freq[0]).values
        integral_f = wave_spec.sum(dim='freq').values * dfreq

        assert wave_spec.shape == (nfreq, ndir)  # shape
        assert np.allclose(integral_d, spec_omni, rtol=0.01)
        assert directions[np.argmax(integral_f)] == dm

    def test_pierson_moskowitz_spectrum(self, f1, nfreq, fp, hs):
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
        assert np.isclose(total_variance_calc,
                          total_variance_theory)  # integral

    def test_jonswap_spectrum(self, f1, nfreq, fp, hs):
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

    def test_spread_cos2s(self, f1, nfreq, fp, ndir):
        """Confirm that energy is spread correctly accross wave directions.
        Integral over all directions of the spread function gives (vector)
        1.
        """
        directions = np.linspace(0, 360, ndir, endpoint=False)
        wdir_mean = directions[np.random.randint(0, ndir)]
        freqs = wot.frequency(f1, nfreq, False)
        s_max = 10
        spread = wot.waves.spread_cos2s(freq=freqs,
                                        directions=directions,
                                        dm=wdir_mean,
                                        fp=fp,
                                        s_max=s_max)
        ddir = directions[1]-directions[0]
        dfreq = freqs[1] - freqs[0]
        integral_d = np.sum(spread, axis=1)*ddir
        integral_f = np.sum(spread, axis=0)*dfreq

        assert directions[np.argmax(integral_f)] == wdir_mean  # mean dir
        assert np.allclose(integral_d, np.ones(
            (1, nfreq)), rtol=0.01)  # omnidir

    def test_general_spectrum(self, f1, nfreq):
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

    def test_pierson_moskowitz_params(self, fp, hs):
        params = wot.waves.pierson_moskowitz_params(fp, hs)

        assert len(params) == 2  # returns two floats
        for iparam in params:
            assert isinstance(iparam, float)
