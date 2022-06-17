""" Unit tests for functions in the `core.py` module.
Does not include the WEC class.
"""
import os
import random

import pytest
from pytest import approx
import numpy as np
import xarray as xr
import capytaine as cpy

import wecopttool as wot


@pytest.fixture()
def f1(): return 0.12


@pytest.fixture()
def nfreq(): return 5


@pytest.fixture(scope='function')
def nsubsteps(): return random.randint(2, 10)


@pytest.fixture()
def ncomponents(nfreq): return wot.ncomponents(nfreq)


@pytest.fixture(scope='module')
def bem_data():
    coords = {
        'omega': [2*np.pi*ifreq for ifreq in [0.0, 0.1, 0.2, 0.3, 0.4]],
        'influenced_dof': ['DOF_1', 'DOF_2'],
        'radiating_dof': ['DOF_1', 'DOF_2'],
        'wave_direction': [0.0, 1.5, 2.1],
    }

    ndof = 2; nfreq = 4; ndir = 3;
    radiation_dims = ['radiating_dof', 'influenced_dof', 'omega']
    excitation_dims = ['influenced_dof', 'wave_direction', 'omega']

    added_mass = np.ones([ndof, ndof, nfreq+1])
    radiation_damping = np.ones([ndof, ndof, nfreq+1])
    diffraction_force = np.ones([ndof, ndir, nfreq+1], dtype=complex) + 1j
    Froude_Krylov_force = np.ones([ndof, ndir, nfreq+1], dtype=complex) + 1j

    data_vars = {'added_mass': (radiation_dims, added_mass),
                 'radiation_damping': (radiation_dims, radiation_damping),
                 'diffraction_force': (excitation_dims, diffraction_force),
                 'Froude_Krylov_force': (excitation_dims, Froude_Krylov_force)
    }
    return xr.Dataset(data_vars=data_vars, coords=coords)


@pytest.fixture(scope='module')
def hydro_data(bem_data):
    ndof = len(bem_data.influenced_dof)
    mass = np.ones([ndof, ndof])
    stiffness = np.ones([ndof, ndof])
    friction = np.ones([ndof, ndof])
    return wot.linear_hydrodynamics(bem_data, mass, stiffness, friction)


@pytest.fixture
def wave_regular(f1, nfreq):
    n = np.random.randint(1, nfreq)
    freq = n*f1  # Hz
    amp = 1.1  # m
    phase = 24.5  # degrees
    wave = wot.waves.regular_wave(f1, nfreq, freq, amp, phase)
    params = {'n': n, 'amp': amp, 'phase': phase}
    return wave, params


@pytest.fixture
def waves_multi(f1, nfreq):
    n = np.random.randint(1, nfreq)
    freq = n*f1  # Hz
    directions = [0.0, 30.0]
    waves = wot.waves.wave_dataset(f1, nfreq, directions)
    loc0 = {'omega': freq*2*np.pi, 'wave_direction': np.deg2rad(directions[0])}
    loc1 = {'omega': freq*2*np.pi, 'wave_direction': np.deg2rad(directions[1])}
    amp0, amp1 = 1.2, 2.1
    phase0, phase1 = 26, -13
    waves['S'].loc[loc0] = 0.5 * amp0**2 / f1
    waves['S'].loc[loc1] = 0.5 * amp1**2 / f1
    waves['phase'].loc[loc0] = np.deg2rad(phase0)
    waves['phase'].loc[loc1] = np.deg2rad(phase1)
    params = {'n': n, 'directions': directions, 'amp0': amp0, 'amp1': amp1,
              'phase0': phase0, 'phase1': phase1}
    return waves, params


def test_ncomponents(ncomponents, nfreq):
    assert ncomponents==2*nfreq+1


def test_frequency(f1, nfreq):
    freqs = wot.frequency(f1, nfreq)
    assert ((freqs.ndim==1) and (len(freqs)==nfreq+1))  # shape
    assert ((freqs[0]==0.0) and (freqs[-1]==approx(f1*nfreq))) # first & last
    assert np.diff(freqs)==approx(np.diff(freqs)[0])  # evenly spaced


def test_time(f1, nfreq, nsubsteps, ncomponents):
    time = wot.time(f1, nfreq)
    time_1 = wot.time(f1, nfreq, 1)
    time_sub = wot.time(f1, nfreq, nsubsteps)
    assert time==approx(time_1)  # default behavior
    for t,n in [(time, 1), (time_sub, nsubsteps)]:
        assert ((t.ndim==1) and (len(t)==ncomponents*n))  # shape
        assert np.diff(t)==approx(np.diff(t)[0])  # evenly spaced
        t_last = 1.0/f1-np.diff(t)[0]
        assert ((t[0]==0.0) and (t[-1]==approx(t_last)))  # first & last


def test_time_mat(f1, nfreq, nsubsteps, ncomponents):
    time_mat = wot.time_mat(f1, nfreq)
    time_mat_1 = wot.time_mat(f1, nfreq, 1)
    time_mat_sub = wot.time_mat(f1, nfreq, nsubsteps)
    # synthetic case
    f1_s = 0.5
    nfreq_s = 2
    time_mat_synthetic = wot.time_mat(f1_s, nfreq_s)
    f = np.array([0, 1, 2])*f1_s
    w = 2*np.pi * f
    t = 1/(2*nfreq_s+1) * 1/f1_s * np.arange(0, 2*nfreq_s+1)
    cos, sin = np.cos, np.sin
    expected_synthetic = np.array([
        [1, 1, 0, 1, 0],
        [1, cos(w[1]*t[1]), -sin(w[1]*t[1]), cos(w[2]*t[1]), -sin(w[2]*t[1])],
        [1, cos(w[1]*t[2]), -sin(w[1]*t[2]), cos(w[2]*t[2]), -sin(w[2]*t[2])],
        [1, cos(w[1]*t[3]), -sin(w[1]*t[3]), cos(w[2]*t[3]), -sin(w[2]*t[3])],
        [1, cos(w[1]*t[4]), -sin(w[1]*t[4]), cos(w[2]*t[4]), -sin(w[2]*t[4])]
    ])
    # assert
    assert time_mat==approx(time_mat_1)  # default behavior
    for tm,n in [(time_mat, 1), (time_mat_sub, nsubsteps)]:
        assert tm.shape==(n*ncomponents, ncomponents)  # shape
        assert all(tm[:, 0]==1.0) # zero-frequency components
        assert all(tm[0, 1:]==np.array([1, 0]*nfreq))  # time zero
    assert time_mat_synthetic==approx(expected_synthetic) # full matrix

    # test use/behavior
    f = 0.1
    w = 2*np.pi*f
    time_mat = wot.time_mat(f, 1)
    x = 1.2 + 3.4j
    X = np.reshape([0, np.real(x), np.imag(x)], [-1,1])
    x_t = time_mat @ X
    t = wot.time(f, 1)
    assert np.allclose(x_t.squeeze(), np.real(x*np.exp(1j*w*t)))


def test_derivative_mat(f1, nfreq):
    der = wot.derivative_mat(f1, nfreq)
    # synthetic case
    f1_s = 0.5
    nfreq_s = 2
    der_synthetic = wot.derivative_mat(f1_s, nfreq_s)
    w0 = 2*np.pi*f1_s
    expected_synthetic = np.array([
        [0,  0,   0,    0,     0],
        [0,  0, -w0,    0,     0],
        [0, w0,   0,    0,     0],
        [0,  0,   0,    0, -2*w0],
        [0,  0,   0, 2*w0,     0]
    ])
    assert der.shape==(2*nfreq+1, 2*nfreq+1)  # shape
    assert ((all(der[:, 0]==0.0)) and (all(der[0, :]==0.0)))  # first row & col
    assert der_synthetic==approx(expected_synthetic)  # full matrix

    # test use/behavior
    f = 0.1
    w = 2*np.pi*f
    derivative_mat = wot.derivative_mat(f, 1)
    x = 1.2 + 3.4j
    X = np.reshape([0, np.real(x), np.imag(x)], [-1,1])
    V = derivative_mat @ X
    v = V[1] + 1j*V[2]
    assert np.allclose(v, 1j*w*x)


def test_degrees_to_radians():
    degree = 44.26
    degrees = [0.0, -1000, 2.53, 9.99*np.pi]
    rads = wot.degrees_to_radians(degrees)
    rads_st = wot.degrees_to_radians(degrees, sort=True)
    rads_us = wot.degrees_to_radians(degrees, sort=False)
    rads_np = wot.degrees_to_radians(np.array(degrees))
    rad = wot.degrees_to_radians(degree)
    rad_cyc = wot.degrees_to_radians(degree+random.randint(-10,10)*360)
    assert rads==approx(rads_st)  # default behavior
    assert rads==approx(rads_np)  # numpy arrays
    assert (all(rads>=-np.pi) and all(rads<np.pi))  # range (-π, π]
    assert (all(rads_us>=-np.pi) and all(rads_us<np.pi))  # range (-π, π]
    assert np.sort(rads)==approx(np.sort(rads_us))  # sorting
    assert ((rad>=-np.pi) and (rad<np.pi))  # scalar
    assert rad==approx(rad_cyc)  # cyclic
    for r,d in [(0, 0), (np.pi, 180), (0, 360), (np.pi, -180)]:
        assert r==approx(wot.degrees_to_radians(d))  # special cases


def test_vec_to_dofmat_to_vec():
    """Test both `vec_to_dofmat` and `dofmat_to_vec`."""
    vec = np.array([1,2,3,4,5,6])
    m1 = wot.vec_to_dofmat(vec, 1)
    m2 = wot.vec_to_dofmat(vec, 2)
    m3 = wot.vec_to_dofmat(vec, 3)
    m2_expected = np.array([[1, 4],
                       [2, 5],
                       [3, 6]])
    m3_expected = np.array([[1, 3, 5],
                       [2, 4, 6]])
    assert m1.shape==(6, 1) and m2.shape==(3, 2) and m3.shape==(2, 3)  # shape
    assert all(m1.squeeze()==vec) and np.all(m2==m2_expected) and \
           np.all(m3==m3_expected)  # values
    assert all(wot.dofmat_to_vec(m1) == vec) and \
           all(wot.dofmat_to_vec(m2) == vec) and \
           all(wot.dofmat_to_vec(m3) == vec)  # reverse
    with pytest.raises(ValueError):
        wot.vec_to_dofmat(vec, 4)  # error


def test_mimo_transfer_mat():
    ndof = 2
    nfreq = 2
    imp = np.empty([ndof, ndof, nfreq+1], dtype=complex)
    imp[0, 0, :] = [0+0j, 0+1j, 0+2j]
    imp[1, 0, :] = [1+0j, 1+1j, 11+2j]
    imp[0, 1, :] = [2+0j, 2+1j, 22+2j]
    imp[1, 1, :] = [3+0j, 3+1j, 33+2j]
    mimo = wot.mimo_transfer_mat(imp)
    expected_11 = np.array([[0,  0, 0,  0,  0],
                            [0,  0, -1,  0,  0],
                            [0, 1, 0,  0,  0],
                            [0,  0, 0,  0,  -2],
                            [0,  0, 0, 2,  0]])
    expected_21 = np.array([[1,  0, 0,  0,  0],
                            [0,  1, -1,  0,  0],
                            [0, 1, 1,  0,  0],
                            [0,  0, 0, 11,  -2],
                            [0,  0, 0, 2, 11]])
    expected_12 = np.array([[2,  0, 0,  0,  0],
                            [0,  2, -1,  0,  0],
                            [0, 1, 2,  0,  0],
                            [0,  0, 0, 22,  -2],
                            [0,  0, 0, 2, 22]])
    expected_22 = np.array([[3,  0, 0,  0,  0],
                            [0,  3, -1,  0,  0],
                            [0, 1, 3,  0,  0],
                            [0,  0, 0, 33,  -2],
                            [0,  0, 0, 2, 33]])
    ncomponents = 2*nfreq + 1
    nmat = ndof*ncomponents
    assert mimo.shape==(nmat, nmat) # shape
    assert np.all(mimo[:ncomponents, :ncomponents] == expected_11)  # values
    assert np.all(mimo[ncomponents:, :ncomponents] == expected_21)
    assert np.all(mimo[:ncomponents, ncomponents:] == expected_12)
    assert np.all(mimo[ncomponents:, ncomponents:] == expected_22)
    with pytest.raises(AssertionError):
        imp[0,0] += 1j
        wot.mimo_transfer_mat(imp)  # error

    # test use/behavior
    x = 1.2+3.4j
    X = np.reshape([0, np.real(x), np.imag(x)], [-1,1])
    z = 2.1+4.3j
    f = z*x
    F = np.reshape([0, np.real(f), np.imag(f)], [-1,1])
    Z_mimo = wot.mimo_transfer_mat(np.reshape([0j, z], [1,1,-1]))
    assert np.allclose(Z_mimo @ X, F)


def test_real_to_complex_to_real():
    comp_amp = np.array([[1+0j, 11+0j],  # f0
                         [2+3j, 12+13j],  # f1
                         [4+5j, 14+15j]])  # f2
    comp_1d = np.array([1+0j, 2+3j])
    real_1d = wot.complex_to_real(comp_1d)
    comp_1d_calc = wot.real_to_complex(real_1d)
    nfreq = comp_amp.shape[0]-1
    ndof = comp_amp.shape[1]
    real_amp_exp = np.array([[1, 11],  # f0
                             [2, 12],  # f1 real
                             [3, 13],  # f1 imag
                             [4, 14],  # f2 real
                             [5, 15]])  # f2 imag
    real_amp = wot.complex_to_real(comp_amp)
    comp_amp_calc = wot.real_to_complex(real_amp)
    assert real_amp.shape == (1+2*nfreq, ndof)  # to real
    assert np.allclose(real_amp, real_amp_exp)
    assert comp_amp.shape == comp_amp_calc.shape  # back to complex
    assert np.allclose(comp_amp, comp_amp_calc)
    assert real_1d.shape == (3, 1)  # 1D array
    assert np.allclose(real_1d.squeeze(), [1, 2, 3])
    assert comp_1d_calc.shape == (2, 1)
    assert np.allclose(comp_1d, comp_1d_calc.squeeze())


def test_fd_to_td_to_fd(f1, nfreq):
    fd = np.zeros([nfreq+1, 2], dtype=complex)
    freq = wot.frequency(f1, nfreq)
    time = wot.time(f1, nfreq)
    idx = np.random.randint(1, nfreq)
    wt  = 2*np.pi*freq[idx]*time
    a0r, a0i = 1, 2
    a1r, a1i = 3, 4
    fd[idx, 0] = a0r + a0i*1j
    fd[idx, 1] = a1r + a1i*1j
    td = wot.fd_to_td(fd, f1, nfreq)
    td_calc = np.zeros([len(time), 2])
    td_calc[:, 0] = a0r*np.cos(wt) - a0i*np.sin(wt)
    td_calc[:, 1] = a1r*np.cos(wt) - a1i*np.sin(wt)
    fd_calc = wot.td_to_fd(td)

    fd_1d = np.zeros(nfreq+1, dtype=complex)
    fd_1d[idx] = a0r + a0i*1j
    td_1d = wot.fd_to_td(fd_1d, f1, nfreq)
    td_1d_calc = a0r*np.cos(wt) - a0i*np.sin(wt)
    fd_calc_1d = wot.td_to_fd(td_1d.squeeze())

    td_fft = wot.fd_to_td(fd)
    td_1d_fft = wot.fd_to_td(fd_1d)

    assert td.shape == (1+2*nfreq, 2)  # shape, td
    assert np.allclose(td, td_calc)  # values, td
    assert fd_calc.shape == (1+nfreq, 2)  # shape, fd
    assert np.allclose(fd, fd_calc)  # same values back, fd
    assert td_1d.shape == (1+2*nfreq, 1)  # 1D array
    assert np.allclose(td_1d.squeeze(), td_1d_calc)
    assert fd_calc_1d.shape == (1+nfreq, 1)
    assert np.allclose(fd_1d, fd_calc_1d.squeeze())
    assert td_fft.shape==td.shape  # FFT
    assert td_1d_fft.shape==td_1d.shape
    assert np.allclose(td_fft, td)
    assert np.allclose(td_1d_fft, td_1d)


def test_wave_elevation(f1, nfreq, wave_regular, waves_multi):
    wave_regular, params_reg = wave_regular
    n = params_reg['n']
    amp = params_reg['amp']
    phase = params_reg['phase']
    freq = n*f1
    elev_fd = wot.wave_elevation(wave_regular)
    elev_fd_exp = np.zeros([nfreq+1, 1], dtype=complex)
    elev_fd_exp[n] = amp*np.exp(1j*np.deg2rad(phase))
    elev_td = wot.fd_to_td(elev_fd, f1, nfreq)
    time = wot.time(f1, nfreq)
    elev_td_exp = amp * np.cos((2*np.pi*freq)*time + np.deg2rad(phase))

    waves_multi, params_multi = waves_multi
    n = params_multi['n']
    freq = n*f1
    amp0 = params_multi['amp0']
    amp1 = params_multi['amp1']
    phase0 = params_multi['phase0']
    phase1 = params_multi['phase1']
    elev_multi_fd = wot.wave_elevation(waves_multi)
    elev_multi_td = wot.fd_to_td(elev_multi_fd, f1, nfreq)
    elev_multi_fd_exp = np.zeros([nfreq+1, 2], dtype=complex)
    elev_multi_fd_exp[n, 0] = amp0*np.exp(1j*np.deg2rad(phase0))
    elev_multi_fd_exp[n, 1] = amp1*np.exp(1j*np.deg2rad(phase1))
    elev_multi_td_exp = np.zeros([1+2*nfreq, 2])
    elev_multi_td_exp[:, 0] = amp0 * np.cos(
        (2*np.pi*freq)*time + np.deg2rad(phase0))
    elev_multi_td_exp[:, 1] = amp1 * np.cos(
        (2*np.pi*freq)*time + np.deg2rad(phase1))

    assert elev_fd.shape==(nfreq+1, 1)  # shape, fd
    assert np.allclose(elev_fd, elev_fd_exp)  # values, fd
    assert elev_td.shape==(1+2*nfreq, 1)  # shape, td
    assert np.allclose(elev_td.squeeze(), elev_td_exp)  # values
    assert elev_multi_fd.shape==(nfreq+1, 2)  # multiple directions
    assert np.allclose(elev_multi_fd, elev_multi_fd_exp)
    assert elev_multi_td.shape==(1+2*nfreq, 2)
    assert np.allclose(elev_multi_td, elev_multi_td_exp)


def test_wave_excitation(f1, nfreq, wave_regular, waves_multi):
    wave_regular, params_reg = wave_regular
    waves_multi, params_multi = waves_multi
    directions = np.deg2rad(params_multi['directions'])
    omega = 2*np.pi*wot.frequency(f1, nfreq)
    ndir = len(directions); assert ndir==2; assert directions[0]==0.0
    ndof = 3
    n_m = params_multi['n']
    n_r = params_reg['n']
    wave_elev_regular = \
        params_reg['amp'] * np.exp(1j*np.deg2rad(params_reg['phase']))
    waves_elev_multi = [
        params_multi['amp0'] * np.exp(1j*np.deg2rad(params_multi['phase0'])),
        params_multi['amp1'] * np.exp(1j*np.deg2rad(params_multi['phase1']))]

    exc_coeff = np.zeros([nfreq+1, ndir, ndof], dtype=complex)
    exc_coeff[n_m, 0, :] = [1+11j, 2+22j, 3+33j]
    exc_coeff[n_m, 1, :] = [4+44j, 5+55j, 6+66j]
    exc_coeff[n_r, 0, :] = [1+11j, 2+22j, 3+33j]
    exc_coeff[n_r, 1, :] = [4+44j, 5+55j, 6+66j]
    coords = {
        'omega': (['omega'], omega, {'units': 'rad/s'}),
        'wave_direction': (['wave_direction'], directions, {'units': 'rad'}),
        'influenced_dof': (['influenced_dof'], ['DOF_1', 'DOF_2', 'DOF_3'], {})
        }
    exc_coeff = xr.DataArray(exc_coeff, coords=coords, attrs={})
    fexc_reg = wot.wave_excitation(exc_coeff, wave_regular)
    fexc_multi = wot.wave_excitation(exc_coeff, waves_multi)
    fexc_reg_exp = np.zeros([nfreq+1, ndof], dtype=complex)
    for idof in range(ndof):
        idir = 0
        fexc_reg_exp[n_r, idof] = exc_coeff[n_r, idir, idof] * wave_elev_regular
    fexc_multi_exp = np.zeros([nfreq+1, ndof], dtype=complex)
    for idof in range(ndof):
        for idir in range(ndir):
            fexc_multi_exp[n_m, idof] += \
                exc_coeff[n_m, idir, idof] * waves_elev_multi[idir]

    assert fexc_reg.shape==(nfreq+1, ndof)  # shape
    assert fexc_multi.shape==(nfreq+1, ndof)
    assert np.allclose(fexc_reg, fexc_reg_exp)  # values
    assert np.allclose(fexc_multi, fexc_multi_exp)
    with pytest.raises(ValueError):  # directions not subset raises error
        waves = wot.waves.wave_dataset(f1, nfreq, [0.0, 25])
        wot.wave_excitation(exc_coeff, waves)
    with pytest.raises(ValueError):  # frequencies not same raises error
        waves = wot.waves.wave_dataset(f1+0.01, nfreq, directions)
        wot.wave_excitation(exc_coeff, waves)


def test_read_write_netcdf(hydro_data):
    cwd = os.path.dirname(__file__)
    bem_file_tmp = os.path.join(cwd, 'bem_tmp.nc')
    wot.write_netcdf(bem_file_tmp, hydro_data)
    hydro_data_new = wot.read_netcdf(bem_file_tmp)
    os.remove(bem_file_tmp)
    assert hydro_data.equals(hydro_data_new)


def test_check_linear_damping(hydro_data):
    tol = 0.01
    data = hydro_data.copy(deep=True)
    data['radiation_damping'] *= 0
    data['friction'] *= 0
    data['friction'] += -0.1
    data_org = data.copy(deep=True)
    data_new = wot.check_linear_damping(data, tol)
    data_new_nofric = data_new.copy(deep=True).drop_vars('friction')
    data_org_nofric = data_org.copy(deep=True).drop_vars('friction')
    nodiag = lambda x: x.friction.values - np.diag(np.diagonal(x.friction.values))
    assert data.equals(data_org)  # no side effects
    assert np.allclose(np.diagonal(data_new.friction.values), tol)  # values
    assert np.allclose(nodiag(data_new), nodiag(data_org))  # only diagonal changed
    assert data_new_nofric.equals(data_org_nofric) # only friction is changed


def test_standard_forces_inertia(hydro_data):
    data = hydro_data.copy(deep=True)
    nfreq = len(data.omega) - 1
    f1 = data.omega.values[1] / (2*np.pi)
    ndof = len(data.influenced_dof)
    ndir = len(data.wave_direction)

    x_wec = np.zeros((nfreq*2+1)*ndof)
    index_freq = np.random.randint(1, nfreq)
    amplitude = 1.3
    x_wec[(index_freq*2-1)*1] = amplitude

    w = data.omega.values[index_freq]
    t = wot.time(f1, nfreq)
    pos = amplitude * np.cos(w*t)
    vel = -1*w*amplitude * np.sin(w*t)
    acc = -1*w**2*amplitude * np.cos(w*t)

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
    data['mass'][:, :] = np.eye(ndof)*mass
    data['hydrostatic_stiffness'][:, :] = np.eye(ndof)*hstiff
    data['friction'][:, :] = np.eye(ndof)*fric
    data['radiation_damping'].values[:, :, index_freq] = np.eye(ndof)*rad
    data['added_mass'].values[:, :, index_freq] = np.eye(ndof)*addmass
    data['diffraction_force'].values[:, :, index_freq] = np.zeros([ndof, ndir], dtype=complex)
    data['diffraction_force'].values[0, 0, index_freq] = diff
    data['Froude_Krylov_force'].values[:, :, index_freq] = np.zeros([ndof, ndir], dtype=complex)
    data['Froude_Krylov_force'].values[0, 0, index_freq] = fk_coeff

    forces = wot.standard_forces(data)

    wec = wot.WEC(f1, nfreq, {}, mass=data['mass'].values)
    waves = wot.waves.regular_wave(f1, nfreq, wave_freq, wave_amp, wave_phase_deg)
    inertia_func = wot.inertia(f1, nfreq, mass=data['mass'].values)
    inertia = inertia_func(wec, x_wec, None, None)
    radiation = forces['radiation'](wec, x_wec, None, None)
    hydrostatics = forces['hydrostatics'](wec, x_wec, None, None)
    friction = forces['friction'](wec, x_wec, None, None)
    fk = forces['Froude_Krylov'](wec, None, None, waves)
    diffraction = forces['diffraction'](wec, None, None, waves)

    inertia_truth = mass * acc
    radiation_truth = rad*vel + addmass*acc
    hydrostatics_truth = hstiff*pos
    friction_truth = fric*vel
    diff_comp = wave_amp*np.exp(1j*wave_phase) * diff
    diffraction_truth =  (np.real(diff_comp) * np.cos(w*t) -
                          np.imag(diff_comp) * np.sin(w*t))
    fk_comp = wave_amp*np.exp(1j*wave_phase) * fk_coeff
    fk_truth =  np.real(fk_comp) * np.cos(w*t) - np.imag(fk_comp) * np.sin(w*t)

    forces = [
        (inertia, inertia_truth),
        (radiation, radiation_truth),
        (hydrostatics, hydrostatics_truth),
        (friction, friction_truth),
        (diffraction, diffraction_truth),
        (fk, fk_truth)
    ]

    for f_calc, f_truth in forces:
        assert np.allclose(f_calc[:, 0], f_truth)
        assert np.allclose(f_calc[:, 1], 0)