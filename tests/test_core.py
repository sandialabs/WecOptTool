""" Unit tests for functions in the :python:`core.py` module.
Does not include the :python:`WEC` class.
"""

import os
import random

import pytest
from pytest import approx
import numpy as np
import xarray as xr
import capytaine as cpy

import wecopttool as wot


# small problem setup with synthetic BEM data
@pytest.fixture(scope='module')
def f1():
    """Fundamental frequency [Hz]."""
    return 0.12


@pytest.fixture(scope='module')
def nfreq():
    """Number of frequencies in frequency vector."""
    return 5


@pytest.fixture(scope='module')
def nsubsteps():
    """Number of sub-steps between default step size for the time
    vector.
    """
    return random.randint(2, 10)


@pytest.fixture(scope="module")
def ncomponents(nfreq):
    """Number of components in the WEC state."""
    return wot.ncomponents(nfreq)


@pytest.fixture(scope='module')
def bem_data(f1, nfreq):
    """Synthetic BEM data."""
    # TODO - start using single BEM solution across entire test suite
    coords = {
        'omega': [2*np.pi*(ifreq+1)*f1 for ifreq in range(nfreq)],
        'influenced_dof': ['DOF_1', 'DOF_2'],
        'radiating_dof': ['DOF_1', 'DOF_2'],
        'wave_direction': [0.0, 1.5, 2.1],
    }

    ndof = 2; ndir = 3;
    radiation_dims = ['radiating_dof', 'influenced_dof', 'omega']
    excitation_dims = ['influenced_dof', 'wave_direction', 'omega']

    added_mass = np.ones([ndof, ndof, nfreq])
    radiation_damping = np.ones([ndof, ndof, nfreq])
    diffraction_force = np.ones([ndof, ndir, nfreq], dtype=complex) + 1j
    Froude_Krylov_force = np.ones([ndof, ndir, nfreq], dtype=complex) + 1j

    data_vars = {
        'added_mass': (radiation_dims, added_mass),
        'radiation_damping': (radiation_dims, radiation_damping),
        'diffraction_force': (excitation_dims, diffraction_force),
        'Froude_Krylov_force': (excitation_dims, Froude_Krylov_force)
    }
    return xr.Dataset(data_vars=data_vars, coords=coords)


@pytest.fixture(scope='module')
def hydro_data(bem_data):
    """Synthetic hydro-data containing inertia, stiffness, and friction
    in addition to the coefficients in `bem_data`."""
    ndof = len(bem_data.influenced_dof)
    inertia_matrix = np.ones([ndof, ndof])
    stiffness = np.ones([ndof, ndof])
    friction = np.ones([ndof, ndof])
    data = wot.linear_hydrodynamics(
        bem_data, inertia_matrix, stiffness, friction
    )
    return data


# fixture: regular wave
@pytest.fixture(scope="module")
def wave_regular(f1, nfreq):
    """Wave structure consisting of a single regular wave."""
    n = np.random.randint(1, nfreq)
    freq = n*f1  # Hz
    amp = 1.1  # m
    phase = 24.5  # degrees
    wave = wot.waves.regular_wave(f1, nfreq, freq, amp, phase)
    params = {'n': n, 'amp': amp, 'phase': phase}
    return wave, params


@pytest.fixture(scope="module")
def ndof_waves():
    """Number of WEC degrees of freedom to consider for the wave tests.
    """
    return 3


@pytest.fixture(scope="module")
def exc_coeff(wave_regular, waves_multi, ndof_waves, nfreq, f1):
    """Excitation coefficients for the WEC."""
    # regular wave parameter
    wave_regular, params_reg = wave_regular
    n_r = params_reg['n']
    # multi-directional wave parameters
    waves_multi, params_multi = waves_multi
    directions = np.deg2rad(params_multi['directions'])
    ndir = len(directions)
    n_m = params_multi['n']
    # excitation force coefficients
    exc_coeff = np.zeros([nfreq, ndir, ndof_waves], dtype=complex)
    exc_coeff[n_m-1, 0, :] = [1+11j, 2+22j, 3+33j]
    exc_coeff[n_m-1, 1, :] = [4+44j, 5+55j, 6+66j]
    exc_coeff[n_r-1, 0, :] = [1+11j, 2+22j, 3+33j]
    exc_coeff[n_r-1, 1, :] = [4+44j, 5+55j, 6+66j]
    omega = 2*np.pi*wot.frequency(f1, nfreq, False)
    coords = {
        'omega': (['omega'], omega, {'units': 'rad/s'}),
        'wave_direction': (['wave_direction'], directions, {'units': 'rad'}),
        'influenced_dof': (['influenced_dof'], ['DOF_1', 'DOF_2', 'DOF_3'], {})
        }
    return xr.DataArray(exc_coeff, coords=coords, attrs={})


@pytest.fixture(scope="module")
def fexc_regular(nfreq, wave_regular, exc_coeff, ndof_waves):
    """Excitation force for the regular wave."""
    # wave elevation
    wave_regular, params_reg = wave_regular
    n_r = params_reg['n']
    wave_elev_regular = (
        params_reg['amp'] *
        np.exp(1j*wot.degrees_to_radians(params_reg['phase'])))
    # excitation force
    fexc = np.zeros([nfreq, ndof_waves], dtype=complex)
    for idof in range(ndof_waves):
        idir = 0
        fexc[n_r-1, idof] = (
            exc_coeff[n_r-1, idir, idof] * wave_elev_regular)
    return fexc


# fixture: sea-state composed of two wave components with different
# frequencies and directions.
@pytest.fixture(scope="module")
def waves_multi(f1, nfreq):
    """Waves structure composed of two sinusoidal waves in different
    directions.
    """
    n = np.random.randint(1, nfreq)
    directions = [0.0, 30.0]
    ndir = len(directions)

    amplitudes = np.zeros([nfreq, ndir])
    phases = np.zeros([nfreq, ndir])
    amp0, amp1 = 1.2, 2.1
    phase0, phase1 = 26, -13
    amplitudes[n-1, :] = [amp0, amp1]
    phases[n-1, :] = [phase0, phase1]

    waves = wot.waves.elevation_fd(f1, nfreq, directions, amplitudes, phases)

    params = {'n': n, 'directions': directions, 'amp0': amp0, 'amp1': amp1,
              'phase0': phase0, 'phase1': phase1}

    return waves, params


@pytest.fixture(scope="module")
def fexc_multi(nfreq, waves_multi, exc_coeff, ndof_waves):
    """Excitation force for the multi-directional wave."""
    # wave elevation
    waves_multi, params_multi = waves_multi
    n_m = params_multi['n']
    directions = np.deg2rad(params_multi['directions'])
    ndir = len(directions)
    waves_elev_multi = [
        params_multi['amp0'] * np.exp(1j*np.deg2rad(params_multi['phase0'])),
        params_multi['amp1'] * np.exp(1j*np.deg2rad(params_multi['phase1']))]
    # excitation force
    fexc = np.zeros([nfreq, ndof_waves], dtype=complex)
    for idof in range(ndof_waves):
        for idir in range(ndir):
            fexc[n_m-1, idof] += \
                exc_coeff[n_m-1, idir, idof] * waves_elev_multi[idir]
    return fexc


# fixture: synthetic impedance
@pytest.fixture(scope="module")
def ndof_imp():
    """Number of degrees of freedom in synthetic impedance."""
    return 2


@pytest.fixture(scope="module")
def nfreq_imp():
    """Number of frequencies in synthetic impedance."""
    return 2


@pytest.fixture(scope="module")
def rao(ndof_imp, nfreq_imp):
    """Synthetic RAO transfer matrix."""
    rao = np.empty([ndof_imp, ndof_imp, nfreq_imp], dtype=complex)
    rao[0, 0, :] = [0+1j, 0+2j]
    rao[1, 0, :] = [1+1j, 11+2j]
    rao[0, 1, :] = [2+1j, 22+2j]
    rao[1, 1, :] = [3+1j, 33+2j]
    return rao


@pytest.fixture(scope="module")
def mimo(nfreq_imp):
    """Correct MIMO matrix corresponding to the synthetic RAO
    transfer matrix.
    """
    ncomponents = wot.ncomponents(nfreq_imp)
    mimo = np.zeros([ncomponents*2, ncomponents*2])
    mimo[:ncomponents, :ncomponents] = np.array([
        [0, 0,  0, 0], #  0
        [0, 0, -1, 0], #  0
        [0, 1,  0, 0], #  0
        [0, 0,  0, 0], # -2
    #   [0, 0,  0, 2], #  0
    ])
    mimo[ncomponents:, :ncomponents] = np.array([
        [0, 0,  0,  0], #  0
        [0, 1, -1,  0], #  0
        [0, 1,  1,  0], #  0
        [0, 0,  0, 11], # -2
    #   [0, 0,  0,  2], # 11
    ])
    mimo[:ncomponents, ncomponents:] = np.array([
        [0, 0,  0,  0], #  0
        [0, 2, -1,  0], #  0
        [0, 1,  2,  0], #  0
        [0, 0,  0, 22], # -2
    #   [0, 0,  0,  2], # 22
    ])
    mimo[ncomponents:, ncomponents:] = np.array([
        [0, 0,  0,  0], #  0
        [0, 3, -1,  0], #  0
        [0, 1,  3,  0], #  0
        [0, 0,  0, 33], # -2
    #   [0, 0,  0,  2], # 33
    ])
    return mimo


class TestNComponents:
    """Test function :python:`ncomponents`."""

    def test_default(self, ncomponents, nfreq):
        """Test default, which will include a zero-frequency component.
        """
        assert ncomponents == 2*nfreq

    def test_nozero(self, nfreq):
        """Test without a zero-frequency component."""
        assert wot.ncomponents(nfreq, False) == 2*nfreq - 1


class TestFrequency:
    """Test function :python:`frequency`."""

    @pytest.fixture(scope="class")
    def freqs(self, f1, nfreq):
        """Frequency vector including the zero-frequency (default)."""
        return wot.frequency(f1, nfreq)

    @pytest.fixture(scope="class")
    def freqs_nm(self, f1, nfreq):
        """Frequency vector without the zero-frequency (mean)."""
        return wot.frequency(f1, nfreq, False)

    def test_shape(self, freqs, nfreq):
        """Test that the default frequency vector has the correct
        shape.
        """
        assert ((freqs.ndim==1) and (len(freqs)==nfreq+1))

    def test_first_last(self, freqs, nfreq, f1):
        """Test the first and last frequencies in the default vector."""
        assert ((freqs[0]==0.0) and (freqs[-1]==approx(f1*nfreq)))

    def test_evenly_spaced(self, freqs):
        """Test that the default frequency vector is evenly-spaced."""
        assert np.diff(freqs)==approx(np.diff(freqs)[0])

    def test_shape_nm(self, freqs_nm, nfreq):
        """Test that the "no-mean" frequency vector has the correct
        shape.
        """
        assert ((freqs_nm.ndim==1) and (len(freqs_nm)==nfreq))

    def test_first_last_nm(self, freqs_nm, nfreq, f1):
        """Test the first and last frequencies in the "no-mean" vector.
        """
        assert ((freqs_nm[0]==approx(f1)) and (freqs_nm[-1]==approx(f1*nfreq)))

    def test_evenly_spaced_nm(self, freqs_nm):
        """Test that the "no-mean" frequency vector is evenly-spaced."""
        assert np.diff(freqs_nm)==approx(np.diff(freqs_nm)[0])


class TestTime:
    """Test function :python:`time`."""

    @pytest.fixture(scope="class")
    def time(self, f1, nfreq):
        """Time vector with default :python:`nsubstep=1`."""
        return wot.time(f1, nfreq)

    @pytest.fixture(scope="class")
    def time_sub(self, f1, nfreq, nsubsteps):
        """Time vector with sub-steps."""
        return wot.time(f1, nfreq, nsubsteps)

    def test_shape(self, time, ncomponents):
        """Test that the default time vector has the correct shape."""
        t = time
        n = 1
        assert ((t.ndim==1) and (len(t)==ncomponents*n))

    def test_first_last(self, time, f1):
        """Test the first and last time steps in the default vector."""
        t = time
        t_last = 1.0/f1-np.diff(t)[0]
        assert ((t[0]==0.0) and (t[-1]==approx(t_last)))

    def test_evenly_spaced(self, time):
        """Test that the default time vector is evenly-spaced."""
        t = time
        assert np.diff(t)==approx(np.diff(t)[0])

    def test_shape_sub(self, time_sub, ncomponents, nsubsteps):
        """Test that the time vector with sub-steps has the correct
        shape.
        """
        t = time_sub
        n = nsubsteps
        assert ((t.ndim==1) and (len(t)==ncomponents*n))

    def test_first_last_sub(self, time_sub, f1):
        """Test the first and last time steps in the vector with
        sub-stpes.
        """
        t = time_sub
        t_last = 1.0/f1-np.diff(t)[0]
        assert ((t[0]==0.0) and (t[-1]==approx(t_last)))

    def test_evenly_spaced_sub(self, time_sub):
        """Test that the time vector with sub-steps is evenly-spaced."""
        t = time_sub
        assert np.diff(t)==approx(np.diff(t)[0])


class TestTimeMat:
    """Test function :python:`time_mat`."""

    @pytest.fixture(scope="class")
    def f1_tm(self,):
        """Fundamental frequency [Hz] for the synthetic time matrix."""
        return 0.5

    @pytest.fixture(scope="class")
    def nfreq_tm(self,):
        """Number of frequencies (harmonics) for the synthetic time
        matrix.
        """
        return 2

    @pytest.fixture(scope="class")
    def time_mat(self, f1_tm, nfreq_tm):
        """Correct/expected time matrix."""
        f = np.array([0, 1, 2])*f1_tm
        w = 2*np.pi * f
        t = 1/(2*nfreq_tm) * 1/f1_tm * np.arange(0, 2*nfreq_tm)
        c, s = np.cos, np.sin
        mat = np.array([
            [1,            1,             0,            1],
            [1, c(w[1]*t[1]), -s(w[1]*t[1]), c(w[2]*t[1])],
            [1, c(w[1]*t[2]), -s(w[1]*t[2]), c(w[2]*t[2])],
            [1, c(w[1]*t[3]), -s(w[1]*t[3]), c(w[2]*t[3])],
        ])
        return mat

    @pytest.fixture(scope="class")
    def time_mat_sub(self, f1, nfreq, nsubsteps):
        """Time matrix with sub-steps."""
        return wot.time_mat(f1, nfreq, nsubsteps)

    def test_time_mat(self, time_mat, f1_tm, nfreq_tm):
        """Test the default created time matrix."""
        calculated = wot.time_mat(f1_tm, nfreq_tm)
        assert calculated==approx(time_mat)

    def test_shape(self, time_mat_sub, ncomponents, nsubsteps):
        """Test the shape of the time matrix with sub-steps."""
        assert time_mat_sub.shape==(nsubsteps*ncomponents, ncomponents)

    def test_zero_freq(self, time_mat_sub):
        """Test the zero-frequency components of the time matrix with
        sub-steps.
        """
        assert all(time_mat_sub[:, 0]==1.0)

    def test_time_zero(self, time_mat_sub, nfreq):
        """Test the components at time zero of the time matrix with
        sub-steps.
        """
        assert all(time_mat_sub[0, 1:]==np.array([1, 0]*nfreq)[:-1])

    def test_behavior(self,):
        """Test that when the time matrix multiplies a state-vector it
        results in the correct response time-series.
        """
        f = 0.1
        w = 2*np.pi*f
        time_mat = wot.time_mat(f, 1)
        x = 1.2 + 3.4j
        X = np.reshape([0, np.real(x), np.imag(x)], [-1,1])[:-1]
        x_t = time_mat @ X
        t = wot.time(f, 1)
        assert np.allclose(x_t.squeeze(), np.real(x*np.exp(1j*w*t)))


class TestDerivativeMats:
    """Test functions :python:`derivative_mat`
    and :python:`derivative2_mat`.
    """

    @pytest.fixture(scope="class")
    def f1_dm(self,):
        """Fundamental frequency [Hz] for the synthetic derivative
        matrix.
        """
        return 0.5

    @pytest.fixture(scope="class")
    def nfreq_dm(self,):
        """Number of frequencies (harmonics) for the synthetic
        derivative matrix.
        """
        return 2

    @pytest.fixture(scope="class")
    def derivative_mat(self, f1_dm):
        """Correct/expected derivative matrix."""
        w0 = 2*np.pi*f1_dm
        mat = np.array([
            [0,  0,   0,    0],
            [0,  0, -w0,    0],
            [0, w0,   0,    0],
            [0,  0,   0,    0],
        ])
        return mat
    
    @pytest.fixture(scope="class")
    def derivative2_mat(self, f1_dm):
        """Correct/expected second derivative matrix."""
        w0 = 2*np.pi*f1_dm
        mat = np.array([
            [0,      0,      0,          0],
            [0, -w0**2,      0,          0],
            [0,      0, -w0**2,          0],
            [0,      0,      0, -(2*w0)**2],
        ])
        return mat

    def test_derivative_mat(self, derivative_mat, f1_dm, nfreq_dm):
        """Test the default created derivative matrix."""
        calculated = wot.derivative_mat(f1_dm, nfreq_dm)
        assert calculated==approx(derivative_mat)

    def test_derivative2_mat(self, derivative2_mat, f1_dm, nfreq_dm):
        """Test the default created second derivative matrix."""
        calculated = wot.derivative2_mat(f1_dm, nfreq_dm)
        assert calculated==approx(derivative2_mat)

    def test_dmat_no_mean(self, derivative_mat, f1_dm, nfreq_dm):
        """Test the derivative matrix without the mean component."""
        calculated = wot.derivative_mat(f1_dm, nfreq_dm, False)
        assert calculated==approx(derivative_mat[1:, 1:])

    def test_d2mat_no_mean(self, derivative2_mat, f1_dm, nfreq_dm):
        """Test the second derivative matrix without the mean component."""
        calculated = wot.derivative2_mat(f1_dm, nfreq_dm, False)
        assert calculated==approx(derivative2_mat[1:, 1:])

    def test_dmat_behavior(self,):
        """Test that when the derivative matrix multiplies a
        state-vector it results in the correct state-vector for the
        derivative of the input response.
        """
        f = 0.1
        w = 2*np.pi*f
        x = np.array([1 + 2j,
                      3 + 4j,
                      5 + 6j])
        derivative_mat = wot.derivative_mat(f, np.size(x))
        X = np.concatenate([
            [0.],
            np.reshape([[np.real(i), np.imag(i)] for i in x[:-1]], -1),
            [np.real(x[-1])]
        ])
        V = derivative_mat @ X
        v = np.sum(V[1::2]) + 1j*np.sum(V[2::2])
        expected = np.sum(
            [[(i+1) * 1j * w * x[i]] for i in range(np.size(x)-1)])
        assert np.allclose(v, expected)

    def test_d2mat_behavior(self,):
        """Test that when the second derivative matrix multiplies a
        state-vector it results in the correct state-vector for the
        second derivative of the input response.
        """
        f = 0.1
        w = 2*np.pi*f
        x = np.array([1 + 2j,
                        3 + 4j,
                        5 + 6j])
        derivative2_mat = wot.derivative2_mat(f, np.size(x))
        X = np.concatenate([
            [0.],
            np.reshape([[np.real(i), np.imag(i)] for i in x[:-1]], -1),
            [np.real(x[-1])]
        ])
        V = derivative2_mat @ X
        v = np.sum(V[1::2]) + 1j*np.sum(V[2::2])
        expected = np.sum(
            [[-(i * w)**2 * x[i]] for i in range(np.size(x)-1)]
            + np.real(np.size(x) * w)**2 * x[-1])


class TestMIMOTransferMat:
    """Test function :python:`mimo_transfer_mat`."""

    def test_mimo_transfer_mat(self, rao, mimo):
        """Test the function produces the correct MIMO transfer matrix.
        """
        calculated = wot.mimo_transfer_mat(rao, False)
        assert np.all(calculated == mimo)

    def test_behavior(self,):
        """Test that the MIMO transfer matrix applied to a state vector
        produces the expected output state vector, based on the
        synthetic RAO transfer function.
        """
        # test use/behavior
        x = np.array([1 + 2j,
                      3 + 4j])
        X = np.concatenate([
            [0.],
            np.reshape([[np.real(i), np.imag(i)] for i in x[:-1]], -1),
            [np.real(x[-1])]
        ])
        z = np.array([1.5+2.5j,
                      3.5+4.5j])
        F = np.concatenate([
            [0.],
            np.reshape(
                [[np.real(z[i]*x[i]), np.imag(z[i]*x[i])] for 
                i in range(np.size(x)-1)], -1),
            [np.real(z[-1]) * np.real(x[-1])],
        ])
        Z_mimo = wot.mimo_transfer_mat(np.reshape([z], [1,1,-1]), False)
        assert np.allclose(Z_mimo @ X, F)


class TestVecToDOFMatToVec:
    """Test functions :python:`vec_to_dofmat` and
    :python:`dofmat_to_vec`.
    """

    @pytest.fixture(scope="class")
    def vec(self,):
        """Sample vector."""
        return np.array([1,2,3,4,5,6])

    @pytest.fixture(scope="class")
    def dofmats(self, vec):
        """Correct sample matrices from the sample vector."""
        m1 = np.array([[1], [2], [3], [4], [5], [6]])
        m2 = np.array([
            [1, 4],
            [2, 5],
            [3, 6],
        ])
        m3 = np.array([
            [1, 3, 5],
            [2, 4, 6],
        ])
        return (m1, m2, m3)

    @pytest.fixture(scope="class")
    def dofmats_calc(self, vec):
        """Calculate sample matrices from the sample vector."""
        m1 = wot.vec_to_dofmat(vec, 1)
        m2 = wot.vec_to_dofmat(vec, 2)
        m3 = wot.vec_to_dofmat(vec, 3)
        return (m1, m2, m3)

    def test_shapes(self, dofmats_calc):
        """Test the shapes of the calculated matrices."""
        (m1, m2, m3) = dofmats_calc
        assert m1.shape==(6, 1) and m2.shape==(3, 2) and m3.shape==(2, 3)

    def test_values(self, dofmats, dofmats_calc):
        """Test the values of the calculated matrices."""
        (m1c, m2c, m3c) = dofmats_calc
        (m1, m2, m3) = dofmats
        assert np.all(m1c==m1) and np.all(m2c==m2) and np.all(m3c==m3)

    def test_recover_vec(self, vec, dofmats_calc):
        """Test that :python:`dofmat_to_vec` recovers the original
        vector.
        """
        (m1, m2, m3) = dofmats_calc
        v1 = wot.dofmat_to_vec(m1)
        v2 = wot.dofmat_to_vec(m2)
        v3 = wot.dofmat_to_vec(m3)
        assert np.all(v1==vec) and np.all(v2==vec) and np.all(v3==vec)

    def test_error(self, vec):
        """Test that the function raises a ValueError if an incompatible
        number of degrees of freedom are specified.
        """
        with pytest.raises(ValueError):
            wot.vec_to_dofmat(vec, 4)


class TestRealToComplexToReal:
    """Test functions :python:`real_to_complex` and
    :python:`complex_to_real`.
    """

    @pytest.fixture(scope="class")
    def complex_response(self,):
        """Sample complex response with 2 degrees of freedom and 2
        frequencies components plus the mean (zero-frequency) component.
        """
        response = np.array([
            # DOF 1,  DOF 2
            [  1+0j, 11+ 0j],  # f0
            [  2+3j, 12+13j],  # f1
            [  4   , 14    ],  # f2 (real component only)
        ])
        return response

    @pytest.fixture(scope="class")
    def real_response(self,):
        """Sample response as real coefficients (state vector used in
        WecOptTool).
        """
        response = np.array([
            # D1, D2
            [  1, 11],  # f0
            [  2, 12],  # f1 real
            [  3, 13],  # f1 imag
            [  4, 14],  # f2 real
        ])
        return response

    def test_complex_to_real(self, complex_response, real_response):
        """Test converting from complex to real."""
        calculated = wot.complex_to_real(complex_response)
        assert np.allclose(calculated, real_response)

    def test_real_to_complex(self, complex_response, real_response):
        """Test converting from real to complex."""
        calculated = wot.real_to_complex(real_response)
        assert np.allclose(calculated, complex_response)

    def test_cycle_real(self, real_response):
        """Test converting from real to complex and back to real."""
        calculated = wot.real_to_complex(real_response)
        calculated = wot.complex_to_real(calculated)
        assert np.allclose(calculated, real_response)

    def test_cycle_complex(self, complex_response):
        """Test converting from complex to real and back to complex."""
        calculated = wot.complex_to_real(complex_response)
        calculated = wot.real_to_complex(calculated)
        assert np.allclose(calculated, complex_response)

    def test_shapes(self, complex_response, real_response):
        """Test output shapes."""
        c_calc = wot.real_to_complex(real_response)
        r_calc = wot.complex_to_real(c_calc)
        c_shape = complex_response.shape
        r_shape = real_response.shape
        assert (c_calc.shape==c_shape) and (r_calc.shape==r_shape)

    def test_1d(self,):
        """Test the function with a 1-dimensional complex array as
        input.
        """
        complex_1d = np.array([1+0j, 2+3j])
        real_1d_calculated = wot.complex_to_real(complex_1d)
        real_1d = np.array([[1], [2]])
        assert np.allclose(real_1d_calculated, real_1d)


class TestFDToTDToFD:
    """Test functions :python:`fd_to_td` and :python:`td_to_fd`."""

    @pytest.fixture(scope="class")
    def components(self):
        """Values of the two non-zero components of the response."""
        a0r, a0i = 1, 2
        a1r, a1i = 3, 4
        return (a0r, a0i, a1r, a1i)

    @pytest.fixture(scope="class")
    def idx(self, nfreq):
        return np.random.randint(1, nfreq-1)
    
    @pytest.fixture(scope="class")
    def dc(self):
        return np.random.randint(1, 5)

    @pytest.fixture(scope="class")
    def fd(self, nfreq, components, idx):
        """Sample frequency domain response. There are two degrees of
        freedom and the response is zero at all but one (random)
        frequency. The non-zero values are set based on
        :python:`components`.
        """
        (a0r, a0i, a1r, a1i) = components
        fd = np.zeros([nfreq+1, 2], dtype=complex)
        fd[idx, 0] = a0r + a0i*1j
        fd[idx, 1] = a1r + a1i*1j
        return fd

    @pytest.fixture(scope="class")
    def td(self, nfreq, f1, components, idx):
        """Corresponding sample time domain response."""
        freq = wot.frequency(f1, nfreq)
        time = wot.time(f1, nfreq)
        wt  = 2*np.pi*freq[idx]*time
        (a0r, a0i, a1r, a1i) = components
        td = np.zeros([len(time), 2])
        td[:, 0] = a0r*np.cos(wt) - a0i*np.sin(wt)
        td[:, 1] = a1r*np.cos(wt) - a1i*np.sin(wt)
        return td

    @pytest.fixture(scope="class")
    def fd_1dof(self, nfreq, components, idx):
        """Sample frequency domain response with only one degree of
        freedom.
        """
        (a0r, a0i, _, _) = components
        fd = np.zeros(nfreq+1, dtype=complex)
        fd[idx] = a0r + a0i*1j
        return fd

    @pytest.fixture(scope="class")
    def td_1dof(self, nfreq, f1, components, idx):
        """Corresponding sample time domain response with only one
        degree of freedom.
        """
        (a0r, a0i, _, _) = components
        freq = wot.frequency(f1, nfreq)
        time = wot.time(f1, nfreq)
        wt  = 2*np.pi*freq[idx]*time
        return a0r*np.cos(wt) - a0i*np.sin(wt)

    @pytest.fixture(scope="class")
    def fd_topfreq(self, nfreq, components):
        """Sample frequency domain response with the nonzero components
        in the highest (Nyquist) frequency."""
        (a0r, _, a1r, _) = components
        fd = np.zeros([nfreq+1, 2], dtype=complex)
        fd[-1, 0] = a0r + 0j
        fd[-1, 1] = a1r + 0j
        return fd
    
    @pytest.fixture(scope="class")
    def td_topfreq(self, nfreq, f1, components):
        """Corresponding sample time domain response for the frequency
        vector with a nonzero top (Nyquist) frequency."""
        freq = wot.frequency(f1, nfreq)
        time = wot.time(f1, nfreq)
        wt  = 2*np.pi*freq[-1]*time
        (a0r, a0i, a1r, a1i) = components
        td = np.zeros([len(time), 2])
        td[:, 0] = a0r*np.cos(wt) - a0i*np.sin(wt)
        td[:, 1] = a1r*np.cos(wt) - a1i*np.sin(wt)
        return td
    
    @pytest.fixture(scope="class")
    def fd_nzmean(self, nfreq, components, idx, dc):
        """Sample frequency domain response with a nonzero mean."""
        (a0r, a0i, a1r, a1i) = components
        fd = np.zeros([nfreq+1, 2], dtype=complex)
        fd[0, :] = dc
        fd[idx, 0] = a0r + a0i*1j
        fd[idx, 1] = a1r + a1i*1j
        return fd

    @pytest.fixture(scope="class")
    def td_nzmean(self, nfreq, f1, components, idx, dc):
        """Corresponding sample time domain response with a nonzero mean."""
        freq = wot.frequency(f1, nfreq)
        time = wot.time(f1, nfreq)
        wt  = 2*np.pi*freq[idx]*time
        (a0r, a0i, a1r, a1i) = components
        td = np.zeros([len(time), 2])
        td[:, 0] = a0r*np.cos(wt) - a0i*np.sin(wt) + dc
        td[:, 1] = a1r*np.cos(wt) - a1i*np.sin(wt) + dc
        return td

    def test_fd_to_td(self, fd, td, f1, nfreq):
        """Test the :python:`fd_to_td` function outputs."""
        calculated = wot.fd_to_td(fd, f1, nfreq)
        assert calculated.shape==(2*nfreq, 2) and np.allclose(calculated, td)

    def test_td_to_fd(self, fd, td, nfreq):
        """Test the :python:`td_to_fd` function outputs."""
        calculated = wot.td_to_fd(td)
        assert calculated.shape==(nfreq+1, 2) and np.allclose(calculated, fd)

    def test_fft(self, fd, td, nfreq):
        """Test the :python:`fd_to_td` function outputs when using FFT.
        """
        calculated = wot.fd_to_td(fd)
        assert calculated.shape==(2*nfreq, 2) and np.allclose(calculated, td)

    def test_fd_to_td_1dof(self, fd_1dof, td_1dof, f1, nfreq):
        """Test the :python:`fd_to_td` function outputs for the 1 DOF
        case.
        """
        calculated = wot.fd_to_td(fd_1dof, f1, nfreq)
        shape = (2*nfreq, 1)
        calc_flat = calculated.squeeze()
        assert calculated.shape==shape and np.allclose(calc_flat, td_1dof)

    def test_td_to_fd_1dof(self, fd_1dof, td_1dof, nfreq):
        """Test the :python:`td_to_fd` function outputs for the 1 DOF
        case.
        """
        calculated = wot.td_to_fd(td_1dof.squeeze())
        shape = (nfreq+1, 1)
        calc_flat = calculated.squeeze()
        assert calculated.shape==shape and np.allclose(calc_flat, fd_1dof)

    def test_fft_1dof(self, fd_1dof, td_1dof, nfreq):
        """Test the :python:`fd_to_td` function outputs when using FFT
        for the 1 DOF.
        """
        calculated = wot.fd_to_td(fd_1dof)
        shape = (2*nfreq, 1)
        calc_flat = calculated.squeeze()
        assert calculated.shape==shape and np.allclose(calc_flat, td_1dof)

    def test_fd_to_td_nzmean(self, fd_nzmean, td_nzmean, f1, nfreq):
        """Test the :python: `td_to_fd` function outputs with a 
        nonzero mean value.
        """
        calculated = wot.fd_to_td(fd_nzmean, f1, nfreq)
        assert calculated.shape==(2*nfreq, 2) and np.allclose(calculated, td_nzmean)

    def test_td_to_fd_nzmean(self, fd_nzmean, td_nzmean, nfreq):
        """Test the :python: `td_to_fd` function outputs with a
        nonzero mean value.
        """
        calculated = wot.td_to_fd(td_nzmean)
        assert calculated.shape==(nfreq+1, 2) and np.allclose(calculated, fd_nzmean)

    def test_fd_to_td_nzmean(self, fd_nzmean, td_nzmean, f1, nfreq):
        """Test the :python: `td_to_fd` function outputs with the top (Nyquist)
        frequency vector.
        """
        calculated = wot.fd_to_td(fd_nzmean, f1, nfreq)
        assert calculated.shape==(2*nfreq, 2) and np.allclose(calculated, td_nzmean)

    def test_td_to_fd_topfreq(self, fd_topfreq, td_topfreq, nfreq):
        """Test the :python: `td_to_fd` function outputs for the
        Nyquist frequency.
        """
        calculated = wot.td_to_fd(td_topfreq)
        assert calculated.shape==(nfreq+1, 2) and np.allclose(calculated, fd_topfreq)


class TestReadWriteNetCDF:
    """Test functions :python:`read_netcdf` and :python:`write_netcdf`.
    """

    @pytest.fixture(scope="class")
    def hydro_data_new(self, hydro_data):
        """Hydrodynamic data structure created by writing and then
        reading data to a NetCDF file.
        """
        cwd = os.path.dirname(__file__)
        bem_file_tmp = os.path.join(cwd, 'bem_tmp.nc')
        wot.write_netcdf(bem_file_tmp, hydro_data)
        hydro_data_new = wot.read_netcdf(bem_file_tmp)
        os.remove(bem_file_tmp)
        return hydro_data_new

    def test_read_write_netcdf(self, hydro_data, hydro_data_new):
        """Test that writing to, followed by reading from, a NetCDF file
        results in the same original hydrodynamic data structure.
        """
        assert hydro_data.equals(hydro_data_new)


class TestCheckLinearDamping:
    """Test function :python:`check_linear_damping`."""

    @pytest.fixture(scope="class")
    def data(self, hydro_data):
        """Hydrodynamic data structure for this test set."""
        data = hydro_data.copy(deep=True)
        data['radiation_damping'] *= 0
        data['friction'] *= 0
        data['friction'] += -0.1
        return data

    @pytest.fixture(scope="class")
    def tol(self, data):
        """Tolerance for function :python:`check_linear_damping`."""
        return 0.01

    @pytest.fixture(scope="class")
    def data_new_uniform(self, data, tol):
        """Hydrodynamic data structure for which the function
        :python:`check_linear_damping` has been called.
        """
        return wot.check_linear_damping(data, tol)

    @pytest.fixture(scope="class")
    def data_new_nonuniform(self, data, tol):
        """Hydrodynamic data structure for which the function
        :python:`check_linear_damping` has been called.
        """
        # TODO: clean this up when fixing the dim order discrepancy
        data['radiation_damping'] = data['radiation_damping'].transpose('omega', ...)
        return wot.check_linear_damping(data, tol, False)

    def test_friction(self, data_new_uniform, tol):
        """Test that the modified friction diagonal has the expected
        value for a uniform shift.
        """
        assert np.allclose(np.diagonal(data_new_uniform.friction.values), tol)

    def test_only_diagonal_friction(self, data, data_new_uniform):
        """Test that only the diagonal was changed for a uniform shift."""
        data_org = data.copy(deep=True)
        def nodiag(x):
            return x.friction.values - np.diag(np.diagonal(x.friction.values))
        assert np.allclose(nodiag(data_new_uniform), nodiag(data_org))

    def test_only_friction(self, data, data_new_uniform):
        """Test that only the friction is changed in the hydrodynamic
        data for a uniform shift.
        """
        data_new_nofric = data_new_uniform.copy(deep=True
                                                ).drop_vars('friction')
        data_org_nofric = data.copy(deep=True).drop_vars('friction')
        assert data_new_nofric.equals(data_org_nofric)

    def test_damping(self, data_new_nonuniform, tol):
        """Test that the modified radiation damping diagonal has the expected
        value for a non-uniform shift.
        """
        assert np.allclose(
            np.diagonal(data_new_nonuniform.radiation_damping.values,
                        axis1=1, axis2=2),
                        tol)

    def test_only_diagonal_damping(self, data_new_nonuniform):
        """Test that no off-diagonal radiation damping terms are nonzero
        for a non-uniform shift.
        """
        assert (
            np.prod(
            np.shape(data_new_nonuniform.radiation_damping.values)[:-1]) == 
            np.count_nonzero(data_new_nonuniform.radiation_damping.values)
        )

    def test_only_rd(self, data, data_new_nonuniform):
        """Test that only the radiation damping is changed in the hydrodynamic
        data for a non-uniform shift.
        """
        data_new_nord = data_new_nonuniform.copy(
            deep=True).drop_vars('radiation_damping')
        data_org_nord = data.copy(deep=True).drop_vars('radiation_damping')
        assert data_new_nord.equals(data_org_nord)
        

class TestCheckImpedance:
    """Test functions :python:`hydrodynamic_impedance` and 
    :python:`check_impedance`."""

    @pytest.fixture(scope="class")
    def data(self, hydro_data):
        """Hydrodynamic data structure for this test set."""
        data = hydro_data.copy(deep=True)
        data['radiation_damping'] *= 0
        data['friction'] *= 0
        data['friction'] += -0.1
        Zi = wot.hydrodynamic_impedance(data)
        return Zi
    
    def test_hydrodynamic_impedance(self, data, hydro_data):
        """"Check that shape of impedance is as expected"""
        assert data.shape == hydro_data.added_mass.shape

    @pytest.fixture(scope="class")
    def tol(self, data):
        """Tolerance for function :python:`check_impedance`."""
        return 0.01

    @pytest.fixture(scope="class")
    def data_new(self, data, tol):
        """Hydrodynamic data structure for which the function
        :python:`check_impedance` has been called.
        """
        return wot.check_impedance(data, tol)

    def test_friction(self, data_new, tol):
        """Test that the modified impedance diagonal has the expected
        value.
        """
        assert np.allclose(np.real(np.diagonal(data_new)), tol)

    def test_only_diagonal_friction(self, data, data_new):
        """Test that only the diagonal was changed."""
        data_org = data.copy(deep=True)

        def offdiags(x):
            return x.values[np.invert(np.eye(x.shape[0], dtype=bool))]
        assert np.allclose(offdiags(data_new), offdiags(data_org))

    def test_only_friction(self, data, data_new):
        """Test that only the real part of the impedance was changed.
        """
        assert np.allclose(np.imag(data), np.imag(data_new))


class TestForceFromImpedanceOrTransferFunction:
    """Test both functions :python:`force_from_rao_transfer_function`
    and :python:`force_from_impedance`.
    """

    @pytest.fixture(scope="class")
    def omega(self, f1, nfreq_imp):
        """Radial frequency vector."""
        return wot.frequency(f1, nfreq_imp) * 2*np.pi

    @pytest.fixture(scope="class")
    def x_wec(self,):
        """WEC position state vector for a simple synthetic case."""
        return [0, 1, 1, 1, 0, 2, 2, 2]

    @pytest.fixture(scope="class")
    def force(self, f1, nfreq_imp, ndof_imp):
        """Correct forces for the synthetic example."""
        # amplitude: A  = mimo @ x_wec, calculated manually
        #   reshaped for convenience
        A = np.array([
            [0, 1,  7, 44],
            [0, 4, 10, 77],
        ])
        force = np.zeros((wot.ncomponents(nfreq_imp), ndof_imp))
        w = wot.frequency(f1, nfreq_imp) * 2*np.pi
        t = wot.time(f1, nfreq_imp)
        force[:, 0] = (
            A[0, 0] +
            A[0, 1]*np.cos(w[1]*t) - A[0, 2]*np.sin(w[1]*t) +
            A[0, 3]*np.cos(w[2]*t)
        )
        force[:, 1] = (
            A[1, 0] +
            A[1, 1]*np.cos(w[1]*t) - A[1, 2]*np.sin(w[1]*t) +
            A[1, 3]*np.cos(w[2]*t)
        )
        return force

    def test_from_transfer(
            self, rao, f1, nfreq_imp, ndof_imp, force, x_wec
        ):
        """Test the function :python:`force_from_rao_transfer_function`
        for a small synthetic problem.
        """
        force_func = wot.force_from_rao_transfer_function(rao, False)
        wec = wot.WEC(f1, nfreq_imp, {}, ndof=ndof_imp, inertia_in_forces=True)
        force_calculated = force_func(wec, x_wec, None, None)
        assert np.allclose(force_calculated, force)

    def test_from_impedance(
            self, rao, f1, nfreq_imp, ndof_imp, force, x_wec, omega
        ):
        """Test the function :python:`force_from_impedance` for a small
        synthetic problem.
        """
        force_func = wot.force_from_impedance(omega[1:], rao/(1j*omega[1:]))
        wec = wot.WEC(f1, nfreq_imp, {}, ndof=ndof_imp, inertia_in_forces=True)
        force_calculated = force_func(wec, x_wec, None, None)
        assert np.allclose(force_calculated, force)


class TestForceFromWaves:
    """Test function :python:`force_from_waves`."""

    def test_regular(
            self, exc_coeff, f1, nfreq, ndof_waves, wave_regular, fexc_regular
        ):
        """Test regular wave forces."""
        # correct
        A = fexc_regular
        force = np.zeros((wot.ncomponents(nfreq), ndof_waves))
        w = wot.frequency(f1, nfreq) * 2*np.pi
        w = w[1:]
        t = wot.time(f1, nfreq)
        for i in range(ndof_waves):
            for j in range(nfreq):
                Ar, Ai = np.real(A[j, i]), np.imag(A[j, i])
                force[:, i] += Ar*np.cos(w[j]*t) - Ai*np.sin(w[j]*t)
        # calculated
        force_func = wot.force_from_waves(exc_coeff)
        wec = wot.WEC(f1, nfreq, {}, ndof=ndof_waves, inertia_in_forces=True)
        waves, _ = wave_regular
        force_calculated = force_func(wec, None, None, waves)
        # test
        assert np.allclose(force_calculated, force)

    def test_multi(
            self, exc_coeff, f1, nfreq, ndof_waves, waves_multi, fexc_multi
        ):
        """Test iregular wave forces."""
        # correct
        A = fexc_multi
        force = np.zeros((wot.ncomponents(nfreq), ndof_waves))
        w = wot.frequency(f1, nfreq) * 2*np.pi
        w = w[1:]
        t = wot.time(f1, nfreq)
        for i in range(ndof_waves):
            for j in range(nfreq):
                Ar, Ai = np.real(A[j, i]), np.imag(A[j, i])
                force[:, i] += Ar*np.cos(w[j]*t) - Ai*np.sin(w[j]*t)
        # calculated
        force_func = wot.force_from_waves(exc_coeff)
        wec = wot.WEC(f1, nfreq, {}, ndof=ndof_waves, inertia_in_forces=True)
        waves, _ = waves_multi
        force_calculated = force_func(wec, None, None, waves)
        # test
        assert np.allclose(force_calculated, force)


class TestInertiaStandardForces:
    """Test functions :python:`inertia` and :python:`standard_forces`.
    """

    @pytest.fixture(scope="class")
    def forces(self, hydro_data):
        data = hydro_data.copy(deep=True)
        nfreq = len(data.omega)
        f1 = data.omega.values[0] / (2*np.pi)
        ndof = len(data.influenced_dof)
        ndir = len(data.wave_direction)
        # time series of responses
        index_freq = np.random.randint(1, nfreq)
        amplitude = 1.3
        w = data.omega.values[index_freq-1]
        t = wot.time(f1, nfreq)
        pos = amplitude * np.cos(w*t)
        vel = -1*w*amplitude * np.sin(w*t)
        acc = -1*w**2*amplitude * np.cos(w*t)
        # parameters
        mass = 2.1
        rad = 1.2
        addmass = 0.7
        hstiff = 0.9
        fric = 3.4
        wave_freq = w/(2*np.pi)
        wave_amp = 1.1
        wave_phase = 0.46
        wave_phase_deg = np.rad2deg(wave_phase)
        diff = 4 + 5j
        fk_coeff = -2 + 1.2j
        # build hydrodynamic dataset
        data['inertia_matrix'][:, :] = np.eye(ndof)*mass
        data['hydrostatic_stiffness'][:, :] = np.eye(ndof)*hstiff
        data['friction'][:, :] = np.eye(ndof)*fric
        data['radiation_damping'].values[:, :, index_freq-1] = np.eye(ndof)*rad
        data['added_mass'].values[:, :, index_freq-1] = np.eye(ndof)*addmass
        data['diffraction_force'].values[:, :, index_freq-1] = (
            np.zeros([ndof, ndir], dtype=complex))
        data['diffraction_force'].values[0, 0, index_freq-1] = diff
        data['Froude_Krylov_force'].values[:, :, index_freq-1] = (
            np.zeros([ndof, ndir], dtype=complex))
        data['Froude_Krylov_force'].values[0, 0, index_freq-1] = fk_coeff
        # standard forces
        forces = wot.standard_forces(data)
        # calculated inertia and forces
        wec = wot.WEC(
            f1, nfreq, {}, inertia_matrix=data['inertia_matrix'].values)
        waves = wot.waves.regular_wave(
            f1, nfreq, wave_freq, wave_amp, wave_phase_deg)
        x_wec = np.zeros((nfreq*2)*ndof)
        x_wec[(index_freq*2-1)*1] = amplitude
        inertia_func = wot.inertia(
            f1, nfreq, inertia_matrix=data['inertia_matrix'].values)
        inertia = inertia_func(wec, x_wec, None, None)
        radiation = forces['radiation'](wec, x_wec, None, None)
        hydrostatics = forces['hydrostatics'](wec, x_wec, None, None)
        friction = forces['friction'](wec, x_wec, None, None)
        fk = forces['Froude_Krylov'](wec, None, None, waves)
        diffraction = forces['diffraction'](wec, None, None, waves)
        # true/expected inertia and forces
        inertia_truth = mass * acc
        radiation_truth = -(rad*vel + addmass*acc)
        hydrostatics_truth = -hstiff*pos
        friction_truth = -fric*vel
        diff_comp = wave_amp*np.exp(1j*wave_phase) * diff
        diffraction_truth =  (np.real(diff_comp) * np.cos(w*t) -
                                np.imag(diff_comp) * np.sin(w*t))
        fk_comp = wave_amp*np.exp(1j*wave_phase) * fk_coeff
        fk_truth =  np.real(fk_comp) * np.cos(w*t) - np.imag(fk_comp) * np.sin(w*t)
        # output dictionary
        forces = [
            (inertia, inertia_truth, "inertia"),
            (radiation, radiation_truth, "radiation"),
            (hydrostatics, hydrostatics_truth, "hydrostatics"),
            (friction, friction_truth, "friction"),
            (diffraction, diffraction_truth, "diffraction"),
            (fk, fk_truth, "")
        ]
        return forces

    def test_values(self, forces):
        """Test the values of inertia and all standard forces."""
        for f_calc, f_truth, name in forces:
            assert np.allclose(f_calc[:, 0], f_truth), f"FAILED: {name}!"

    def test_mean(self, forces):
        """Test that the mean component of inertia and all standard
        forces is zero.
        """
        for f_calc, _, name in forces:
            assert np.allclose(f_calc[:, 1], 0), f"FAILED: {name}!"


class TestRunBEM:
    """Test function :python:`run_bem`."""

    def test_it_runs(self,):
        """Test that the function at least runs and returns correct
        data type.
        """
        rect = cpy.RectangularParallelepiped(
            size=(5.0, 5.0, 2.0), resolution=(10, 10, 10), center=(0.0, 0.0, 0.0,)
        )
        rect.add_translation_dof(name="Heave")
        bem_data = wot.run_bem(fb=rect, freq=[0.1, 0.2], wave_dirs=[0,])
        assert type(bem_data) == xr.Dataset


class TestChangeBEMConvention:
    """Test function :python:`change_bem_convention`."""

    @pytest.fixture(scope="class")
    def data_there(self, bem_data):
        """Modified BEM data."""
        data_here = bem_data.copy(deep=True)
        return wot.change_bem_convention(data_here)

    @pytest.fixture(scope="class")
    def data_back(self, data_there):
        """Twice modified BEM data. Should be equal to original."""
        return wot.change_bem_convention(data_there)

    def test_fk(self, bem_data, data_there):
        """Test that the Froude-Krylov force was changed correctly."""
        calculated = data_there['Froude_Krylov_force'].values
        correct = np.conjugate(bem_data['Froude_Krylov_force'].values)
        assert np.allclose(calculated, correct)

    def test_diff(self, bem_data, data_there):
        """Test that the diffraction force was changed correctly."""
        calculated = data_there['diffraction_force'].values
        correct = np.conjugate(bem_data['diffraction_force'].values)
        assert np.allclose(calculated, correct)

    def test_round_trip(self, bem_data, data_back):
        """Test that a round trip returns the original dataset."""
        xr.testing.assert_allclose(data_back, bem_data)


class TestLinearHydrodynamics:
    """Test function :python:`linear_hydrodynamics`."""

    def test_values(self, bem_data, hydro_data):
        """Test the function returns expected values."""
        mat = np.array([[1, 1], [1, 1]])
        calculated = wot.linear_hydrodynamics(bem_data, mat, mat, mat)
        xr.testing.assert_allclose(calculated, hydro_data)


class TestWaveExcitation:
    """Test function :python:`wave_excitation`."""

    def test_regular_value(
            self, exc_coeff, wave_regular, fexc_regular, nfreq, ndof_waves
        ):
        """Test the value of the regular wave excitation force."""
        wave_regular, _ = wave_regular
        calculated = wot.wave_excitation(exc_coeff, wave_regular)
        calc_shape = calculated.shape
        shape = (nfreq, ndof_waves)
        assert calc_shape==shape and np.allclose(calculated, fexc_regular)

    def test_multi_value(
            self, exc_coeff, waves_multi, fexc_multi, nfreq, ndof_waves
        ):
        """Test the value of the multi-directional wave excitation
        force.
        """
        waves_multi, _ = waves_multi
        calculated = wot.wave_excitation(exc_coeff, waves_multi)
        calc_shape = calculated.shape
        shape = (nfreq, ndof_waves)
        assert calc_shape==shape and np.allclose(calculated, fexc_multi)

    def test_error_directions_not_subset(self, f1, nfreq, exc_coeff):
        """Test that an error is raised if the wave directions are not
        a subset of the hydrodata directions.
        """
        with pytest.raises(ValueError):
            waves = wot.waves.elevation_fd(f1, nfreq, [0.0, 25])
            wot.wave_excitation(exc_coeff, waves)

    def test_error_different_frequencies(
            self, f1, nfreq, waves_multi, exc_coeff
        ):
        """Test that an error is raised if the wave and hydrodata do not
        have the same frequencies.
        """
        _, params_multi = waves_multi
        directions = np.deg2rad(params_multi['directions'])
        with pytest.raises(ValueError):
            waves = wot.waves.elevation_fd(f1+0.01, nfreq, directions)
            wot.wave_excitation(exc_coeff, waves)


class TestAtleast2D:
    """Test function :python:`atleast_2d`."""

    def test_1d(self,):
        """Test function creates extra trailing dimension when a 1D array
        is passed.
        """
        a = np.array([1.1, 2.2, 3.3])
        b = wot.atleast_2d(a)
        assert a.shape!=b.shape and b.shape==(len(a), 1)

    def test_2d(self,):
        """Test function is identity when a 2D array is passed.
        """
        a = np.array([[1.1, 2.2, 3.3]])
        b = wot.atleast_2d(a)
        assert b.shape==a.shape and np.allclose(b, a)


class TestDegreesToRadians:
    """Test function :python:`degrees_to_radians`."""

    @pytest.fixture(scope="class")
    def degree(self,):
        """Single angle in degrees."""
        return 44.26

    @pytest.fixture(scope="class")
    def degrees(self,):
        """List of several angles in degrees."""
        return [0.0, -1000, 2.53, 9.99*np.pi]

    @pytest.fixture(scope="class")
    def rad(self, degree):
        """Single angle in radians."""
        return wot.degrees_to_radians(degree)

    @pytest.fixture(scope="class")
    def rads(self, degrees):
        """List of several angles in radians."""
        return wot.degrees_to_radians(degrees)


    def test_default_sort(self, degrees, rads):
        """Test default sorting behavior."""
        rads_sorted = wot.degrees_to_radians(degrees, sort=True)
        assert rads_sorted==approx(rads)

    def test_unsorted(self, degrees, rads):
        """Test unsorted behavior."""
        rads_us = wot.degrees_to_radians(degrees, sort=False)
        assert rads_us!=approx(rads) and np.sort(rads_us)==approx(rads)

    def test_numpy_array(self, degrees, rads):
        """Test function with a NumPy array."""
        rads_np = wot.degrees_to_radians(np.array(degrees))
        assert rads_np==approx(rads)

    def test_scalar(self, degree, rad):
        """Test function with a scalar."""
        rads_s = wot.degrees_to_radians(degree)
        assert rads_s==approx(rad)

    def test_special_cases(self,):
        """Test edge angles."""
        for r,d in [(0, 0), (np.pi, 180), (0, 360), (np.pi, -180)]:
            assert r==approx(wot.degrees_to_radians(d)), f"Failed: ({r}, {d})"

    def test_cyclic(self, degree, rad):
        """Test that cyclic permutations give same answer."""
        rad_cyc = wot.degrees_to_radians(degree+random.randint(-10,10)*360)
        assert rad_cyc==approx(rad)

    def test_range(self, rads):
        """Test that the outputs are in the range [-π, π) radians."""
        assert (all(rads>=-np.pi) and all(rads<np.pi))


class TestSubsetClose:
    """Test function :python:`subset_close`."""

    @pytest.fixture(scope="class")
    def array_a(self,):
        """An array of numbers."""
        return np.array([1, 2.35, 0.3])

    @pytest.fixture(scope="class")
    def array_b(self, array_a):
        """A second array that is longer and contains all elements of
        the first array.
        """
        n = 10 - len(array_a)
        b = np.concatenate([array_a, np.random.random(n)])
        np.random.shuffle(b)
        return b

    def test_subset(self, array_a, array_b):
        """Test subset identified correctly."""
        subset, _ = wot.subset_close(array_a, array_b)
        assert subset

    def test_not_subset(self, array_a, array_b):
        """Test `False` when it is not a subset."""
        subset, _ = wot.subset_close(array_b, array_a)
        assert not subset

    def test_atol(self, array_a, array_b):
        """Test the tolerance is used correctly."""
        tol = 0.001
        subset_t, _ = wot.subset_close(array_a, array_b+0.9*tol, atol=tol)
        subset_f, _ = wot.subset_close(array_a, array_b+1.1*tol, atol=tol)
        assert subset_t and not subset_f

    def test_indices(self,):
        """Test the correct indices are returned."""
        a = np.array([1.1, 2.2, 3.3])
        b = np.arange(10, dtype=float)
        idx = [1, 4, 6]
        for i,id in enumerate(idx):
            b[id] = a[i]
        subset, idx_calculated = wot.subset_close(a, b)
        assert subset and idx_calculated==idx

    def test_scalar(self,):
        """Test function when array_a is a scalar."""
        a = 2.2
        b = np.array([1.1, a, 3.3])
        subset, _ = wot.subset_close(a, b)
        subset_a, _ = wot.subset_close(a, a)
        assert subset and subset_a


class TestScaleDOFs:
    """Test function :python:`scale_dofs`."""

    def test_function(self,):
        """Test that the function returns expected results."""
        scales = [1.1, 2.2, 3.3]
        ncomponents = 3
        scale_vec_calculated = wot.scale_dofs(scales, ncomponents)
        scale_vec = [1.1, 1.1, 1.1, 2.2, 2.2, 2.2, 3.3, 3.3, 3.3]
        assert np.allclose(scale_vec_calculated, scale_vec)


class TestDecomposeState:
    """Test function :python:`decompose_state`."""

    def test_function(self,):
        """Test that the function returns expected results."""
        ndof, nfreq = 1, 2  # ncomponents = ndof*(2*nfreq-1)+1 = 4
        state = [1, 1, 1, 1, 3.4]
        x_wec = [1, 1, 1, 1]
        x_opt = [3.4]
        x_w_calc, x_o_calc = wot.decompose_state(state, ndof, nfreq)
        assert np.allclose(x_w_calc, x_wec) and np.allclose(x_o_calc, x_opt)


class TestFrequencyParameters:
    """Test function :python:`frequency_parameters`."""

    def test_default(self,):
        """Test that the function returns the same parameters used to
        create the frequency array.
        """
        f1 = np.random.random()
        nfreq = np.random.randint(2,10)
        freq = wot.frequency(f1, nfreq)
        f1_calc, nfreq_calc = wot.frequency_parameters(freq)
        assert f1_calc==approx(f1) and nfreq_calc==nfreq

    def test_zero_freq(self,):
        """Test with a frequency array not containing the zero frequency.
        """
        f1 = 0.1
        nfreq = 3
        freq = [0.1, 0.2, 0.3]
        f1_calc, nfreq_calc = wot.frequency_parameters(freq, False)
        assert f1_calc==approx(f1) and nfreq_calc==nfreq

    def test_error_spacing(self,):
        """Test that it throws an error if the frequency array is not
        evenly-spaced.
        """
        with pytest.raises(ValueError):
            freq = [0, 0.1, 0.2, 0.4]
            wot.frequency_parameters(freq)

    def test_error_zero(self,):
        """Test that it throws an error if the frequency array does not
        contain the zero-frequency.
        """
        with pytest.raises(ValueError):
            freq = [0.1, 0.2, 0.3]
            wot.frequency_parameters(freq)

    def test_error_zero_false(self,):
        """Test that it throws an error if the frequency array contains
        the zero-frequency but :python:`zero_freq` is :python:`False`.
        """
        with pytest.raises(ValueError):
            freq = [0, 0.1, 0.2, 0.3]
            wot.frequency_parameters(freq, False)


class TestTimeResults:
    """Test function :python:`time_results`."""

    @pytest.fixture(scope="class")
    def f1(self,):
        """Fundamental frequency [Hz]."""
        return 0.1

    @pytest.fixture(scope="class")
    def nfreq(self,):
        """Number of frequencies."""
        return 2

    @pytest.fixture(scope="class")
    def time(self, f1, nfreq):
        """Time vector [s]."""
        time = wot.time(f1, nfreq)
        return xr.DataArray(data=time, name='time', dims='time', coords=[time])

    @pytest.fixture(scope="class")
    def components(self,):
        """Real and imaginary components of the complex response."""
        return {'re1': 1.3, 'im1': -2.1, 're2': 0.5, 'im2': 0.4}

    @pytest.fixture(scope="class")
    def fd(self, f1, nfreq, components):
        """Frequency domain response."""
        omega = wot.frequency(f1, nfreq) * 2*np.pi
        re1 = components['re1']
        im1 = components['im1']
        re2 = components['re2']
        im2 = components['im2']
        mag = np.array([0.0, re1+im1*1j, re2+im2*1j])
        mag = xr.DataArray(
            data=mag, name='response', dims='omega', coords=[omega])
        return mag

    def test_values(self, f1, nfreq, time, fd, components):
        """Test that the function returns the correct time domain
        response.
        """
        td = wot.time_results(fd, time)
        re1 = components['re1']
        im1 = components['im1']
        re2 = components['re2']
        im2 = components['im2']
        w = wot.frequency(f1, nfreq) * 2*np.pi
        t = td.time.values
        response = (
            re1*np.cos(w[1]*t) - im1*np.sin(w[1]*t) +
            re2*np.cos(w[2]*t) - im2*np.sin(w[2]*t)
        )
        assert np.allclose(td.values, response)
