#!/usr/bin/env python3
#  -*- coding: utf-8 -*-

import os

import pytest
import autograd.numpy as np
from autograd.builtins import isinstance, tuple, list, dict
import capytaine as cpy
import meshio
from scipy.optimize import Bounds

import wecopttool as wot
from wecopttool.geom import WaveBot
from wecopttool.core import power_limit


@pytest.fixture()
def wec():
    # water properties
    rho = 1000.0

    # frequencies
    f0 = 0.05
    nfreq = 18

    #  mesh
    meshfile = os.path.join(os.path.dirname(__file__), 'data', 'wavebot.stl')

    # capytaine floating body
    fb = cpy.FloatingBody.from_file(meshfile, name="WaveBot")
    fb.add_translation_dof(name="HEAVE")

    # mass and hydrostativ stiffness
    hs_data = wot.hydrostatics.hydrostatics(fb, rho=rho)
    mass_33 = wot.hydrostatics.mass_matrix_constant_density(hs_data)[2, 2]
    mass = np.atleast_2d(mass_33)
    stiffness_33 = wot.hydrostatics.stiffness_matrix(hs_data)[2, 2]
    stiffness = np.atleast_2d(stiffness_33)

    # PTO
    kinematics = np.eye(fb.nb_dofs)
    pto = wot.pto.PseudoSpectralPTO(nfreq, kinematics)

    # constraints
    nsubsteps = 4
    f_max = 2000.0


    def const_f_pto(wec, x_wec, x_opt):
        f = pto.force_on_wec(wec, x_wec, x_opt, nsubsteps)
        return f_max - np.abs(f.flatten())


    ineq_cons = {'type': 'ineq',
                'fun': const_f_pto,
                }

    constraints = [ineq_cons]

    # WEC
    f_add = {'PTO': pto.force_on_wec}

    wec = wot.WEC(fb, mass, stiffness, f0, nfreq,  rho=rho,
                    f_add=f_add, constraints=constraints)

    # BEM
    wec.run_bem()

    return wec


@pytest.fixture()
def regular_wave(wec):
    freq = 0.5
    amplitude = 0.25
    phase = 0.0
    wave = wot.waves.regular_wave(wec.f0, wec.nfreq, freq, amplitude, phase)
    return wave


@pytest.fixture()
def resonant_wave(wec):
    freq = wec.natural_frequency()[0].squeeze().item()
    amplitude = 0.25
    phase = 0.0
    wave = wot.waves.regular_wave(wec.f0, wec.nfreq, freq, amplitude, phase)
    return wave


@pytest.fixture()
def pto(wec):
    kinematics = np.eye(wec.ndof)
    pto = wot.pto.PseudoSpectralPTO(wec.nfreq, kinematics)
    return pto


def test_natural_frequency(wec):
    freq = wec.natural_frequency()[0].squeeze().item()
    expected = 0.65 # Based on v1.0.0 (also agrees with experimental results)
    assert freq == expected


def test_bem_io(wec):
    # write BEM file
    bem_file = 'bem.nc'
    wec.write_bem(bem_file)

    # read BEM file
    wec.read_bem(bem_file)

    # cleanup
    os.remove(bem_file)


def test_wave_excitation(wec, regular_wave):
    # wave excitation
    _, _ = wot.wave_excitation(wec.hydro, regular_wave)


def test_solve(wec, regular_wave, pto):
    # solve
    options = {}
    obj_fun = pto.average_power
    nstate_opt = pto.nstate
    _, _, x_wec, x_opt, avg_pow, _ = wec.solve(regular_wave, obj_fun,
        nstate_opt,
        scale_x_wec = 1.0,
        scale_x_opt = 0.01,
        scale_obj = 1e-1,
        optim_options=options)

    avg_pow_exp = -474.133896724959
    assert pytest.approx(avg_pow, 1e-5) == avg_pow_exp

    # post-process
    _, _ = pto.post_process(wec, x_wec, x_opt)


def test_solve_initial_guess(wec, regular_wave, pto):

    x_wec_0 = wec.initial_x_wec_guess(regular_wave)

    nits = []
    x_wecs = []
    for x_wec_0i in [x_wec_0*0.1, x_wec_0]:
        *_, x_wec, _, _, res = wec.solve(regular_wave,
                            obj_fun=pto.average_power,
                            nstate_opt=pto.nstate,
                            scale_x_wec=1.0,
                            scale_x_opt=0.01,
                            scale_obj=1e-1,
                            x_wec_0=x_wec_0i,
                            )
        nits.append(res['nit'])
        x_wecs.append(x_wec)

    assert nits[0] > nits[1]
    assert pytest.approx(x_wecs[1],1e0) == x_wec_0
    
    
def test_complex_to_real_amplitudes(wec, regular_wave):
    x_wec = wec.initial_x_wec_guess(regular_wave)
    fd_wec = wot.real_to_complex_amplitudes(np.atleast_2d(x_wec))
    x_wec_1 = wot.complex_to_real_amplitudes(fd_wec)
    
    assert np.all(x_wec == x_wec_1)


def test_solve_constraints(wec, regular_wave, pto):
    """Checks that two constraints on PTO force can be enforced
    """

    f_max = 1.85e3
    f_min = 1.8e3
    nsubsteps = 4

    def const_f_pto_max(wec, x_wec, x_opt):
        f = pto.force_on_wec(wec, x_wec, x_opt, nsubsteps)
        return f_max - f.flatten()

    def const_f_pto_min(wec, x_wec, x_opt):
        f = pto.force_on_wec(wec, x_wec, x_opt, nsubsteps)
        return f.flatten() + f_min

    ineq_cons_max = {'type': 'ineq',
                'fun': const_f_pto_max,
                }

    ineq_cons_min = {'type': 'ineq',
                'fun': const_f_pto_min,
                }

    wec.constraints = [ineq_cons_max, ineq_cons_min]

    options = {}
    obj_fun = pto.average_power
    nstate_opt = pto.nstate

    *_, x_wec, x_opt, _, res1 = wec.solve(regular_wave, 
                                          obj_fun,
                                          nstate_opt,
                                          scale_x_wec=1.0,
                                          scale_x_opt=0.01,
                                          scale_obj=1e-1,
                                          unconstrained_first=False,
                                          optim_options=options)

    pto_tdom, _ = pto.post_process(wec, x_wec, x_opt)

    assert pytest.approx(-1*f_min,
                         1e-5) == pto_tdom['force'].min().values.item()
    assert pytest.approx(f_max, 1e-5) == pto_tdom['force'].max().values.item()

    *_, res2 = wec.solve(regular_wave,
                         obj_fun,
                         nstate_opt,
                         unconstrained_first=True,
                         optim_options=options)

    assert pytest.approx(res1['fun'], 1e-5) == res2['fun']
    assert res2['nit'] < res1['nit']


def test_plot(wec):
    _, _ = wec.plot_impedance(show=False)
    _, _ = wec.plot_impedance(style='complex', show=False)


def test_core(wec):
    # set_attr
    hydrostatic_stiffness = wec.hydrostatic_stiffness * 1
    mass_matrix = wec.mass * 1
    fb = wec.fb.copy()
    wec.fb = fb
    wec.mass = mass_matrix
    wec.hydrostatic_stiffness = hydrostatic_stiffness
    wec.run_bem()
    # repr
    print(wec)
    # unused properties
    _ = wec.period
    # unused additional calculations
    wec.bem_calc_rao()
    wec.bem_calc_inf_added_mass()


def test_waves_module(wec):
    # irregular waves
    hs = 1.5
    fp = 1.0/8.0
    dm = 10.0
    s_max = 10.0

    def spectrum_func(f):
        return wot.waves.pierson_moskowitz_spectrum(f, fp, hs)

    def spread_func(f, d):
        return wot.waves.spread_cos2s(f, d, dm, fp, s_max)

    spectrum_name = 'Pierson Moskowitz'
    spread_name = 'Cosine-2S'

    # multidirection
    directions = np.linspace(0, 360, 36, endpoint=False)
    _ = wot.waves.irregular_wave(
        wec.f0, wec.nfreq, directions, spectrum_func, spread_func,
                   spectrum_name, spread_name)

    # long-crested
    direction = 0.0
    _ = wot.waves.long_crested_wave(
        wec.f0, wec.nfreq, spectrum_func, direction, spectrum_name)


def test_jonswap_spectrum(wec):
    tp = 8
    hs = 1.5
    freq = wec.f0 * np.arange(1, wec.nfreq)
    Sj = wot.waves.jonswap_spectrum(freq=freq, fp=1/tp, hs=hs, gamma=1)
    Spm = wot.waves.pierson_moskowitz_spectrum(freq=freq, fp=1/tp, hs=hs)

    assert pytest.approx(Sj, 1e-2) == Spm


def test_pto(wec):
    kinematics = np.eye(wec.ndof)
    pto = wot.pto.ProportionalPTO(kinematics, ['DOF_1'])
    _ = pto.ndof_wec
    x_wec = np.zeros(wec.nstate_wec)
    x_pto = np.zeros(pto.nstate)
    _ = pto.position(wec, x_wec, x_pto)
    _ = pto.acceleration(wec, x_wec, x_pto)
    _ = pto.energy(wec, x_wec, x_pto)


def test_wavebot_ps_theoretical_limit(wec,regular_wave,pto):
    """Check that power obtained using pseudo-spectral with no constraints
    equals theoretical limit.
    """
    wec.constraints = []
    obj_fun = pto.average_power
    nstate_opt = pto.nstate
    _, fdom, _, _, average_power, _ = wec.solve(regular_wave, obj_fun, nstate_opt,
        optim_options={'maxiter': 1000, 'ftol': 1e-8}, scale_x_opt=1e3)
    
    plim = power_limit(fdom['excitation_force'], wec.hydro.Zi)
    assert pytest.approx(average_power, 1e-5) == plim
    
    opt_vel = wec.optimal_velocity(regular_wave)
    assert pytest.approx(fdom.vel.data[1:],1e0) == opt_vel
    
    opt_pos = wec.optimal_position(regular_wave)
    assert pytest.approx(fdom.pos.data[1:],1e0) == opt_pos


def test_wavebot_p_cc(wec,resonant_wave):

    # remove constraints
    wec.constraints = []

    # update PTO
    kinematics = np.eye(wec.ndof)
    pto = wot.pto.ProportionalPTO(kinematics)
    obj_fun = pto.average_power
    nstate_opt = pto.nstate
    wec.f_add = {'PTO': pto.force_on_wec}

    # set bounds such that damping must be negative
    lb = np.concatenate([-1 * np.inf * np.ones(wec.nstate_wec), 
                         -1 * np.inf * np.ones(nstate_opt)])
    ub = np.concatenate([1 * np.inf * np.ones(wec.nstate_wec), 
                         0 * np.ones(nstate_opt)])
    bounds = Bounds(lb, ub)
    
    _, fdom, _, xopt, average_power, _ = wec.solve(resonant_wave, obj_fun, nstate_opt,
        optim_options={'maxiter': 1000, 'ftol': 1e-8}, scale_x_opt=1e3,
        bounds=bounds)
    
    # P controller power matches theoretical limit at resonance
    plim = power_limit(fdom['excitation_force'],
                       wec.hydro.Zi).item()

    assert pytest.approx(average_power, 0.03) == plim
    
    # optimal gain matches real part of impedance
    omega_wave_ind = np.where((resonant_wave.S > 0).squeeze())[0].item()
    optimal_kp_expected = wec.hydro.Zi[omega_wave_ind].real
    
    assert pytest.approx(optimal_kp_expected, 1e-1) == -1*xopt.item()


def test_wavebot_pi_cc(wec,regular_wave):

    # remove constraints
    wec.constraints = []

    # update PTO
    kinematics = np.eye(wec.ndof)
    pto = wot.pto.ProportionalIntegralPTO(kinematics)

    wec.f_add = {'PTO': pto.force_on_wec}
    
    # set bounds such that damping must be negative
    lb = np.concatenate([-1 * np.inf * np.ones(wec.nstate_wec), 
                         -1 * np.inf * np.ones(1),
                         -1 * np.inf * np.ones(1)])
    ub = np.concatenate([1 * np.inf * np.ones(wec.nstate_wec), 
                         0 * np.ones(1),
                         1 * np.inf * np.ones(1)])
    bounds = Bounds(lb, ub)

    _, fdom, _, xopt, avg_power, _ = wec.solve(regular_wave,
                                               obj_fun=pto.average_power, 
                                               nstate_opt=pto.nstate,
                                               optim_options={
                                                   'maxiter': 1000, 
                                                   'ftol': 1e-8},
                                               scale_x_opt=1e3,
                                               bounds=bounds)

    # PI controller power matches theoretical limit for an single freq
    plim = power_limit(fdom['excitation_force'],
                       wec.hydro.Zi).item()

    assert pytest.approx(avg_power, 0.03) == plim
    
    # optimal gain matches complex conjugate of impedance
    omega_wave_ind = np.where((regular_wave.S > 0).squeeze())[0].item()
    omega_wave = regular_wave.omega[omega_wave_ind].data.item()
    tmp1 = wec.hydro.Zi[omega_wave_ind].conj().data.item()
    optimal_gains_expected = -1*tmp1.real + 1j * omega_wave * tmp1.imag
    
    assert pytest.approx(optimal_gains_expected, 1e-6) == xopt[0] + 1j*xopt[1]


def test_examples_device_wavebot_mesh():
    wb = WaveBot()
    mesh = wb.mesh(mesh_size_factor=0.5)
    assert type(mesh) is meshio._mesh.Mesh


def test_examples_device_wavebot_plot_cross_section():
    wb = WaveBot()
    wb.plot_cross_section()


def test_multiple_dof(regular_wave):
    """When defined as uncoupled, surge and heave computed seperately should
    give same solution as computed together"""

    # frequencies
    f0 = 0.05
    nfreq = 18

    #  mesh
    meshfile = os.path.join(os.path.dirname(__file__), 'data', 'wavebot.stl')

    # capytaine floating body
    fb = cpy.FloatingBody.from_file(meshfile, name="WaveBot")
    fb.add_translation_dof(name="SURGE")
    fb.add_translation_dof(name="HEAVE")

    # hydrostatic
    hs_data = wot.hydrostatics.hydrostatics(fb)
    mass_11 = wot.hydrostatics.mass_matrix_constant_density(hs_data)[0, 0]
    mass_13 = wot.hydrostatics.mass_matrix_constant_density(hs_data)[0, 2] # will be 0
    mass_31 = wot.hydrostatics.mass_matrix_constant_density(hs_data)[2, 0] # will be 0
    mass_33 = wot.hydrostatics.mass_matrix_constant_density(hs_data)[2, 2]
    stiffness_11 = wot.hydrostatics.stiffness_matrix(hs_data)[0, 0]
    stiffness_13 = wot.hydrostatics.stiffness_matrix(hs_data)[0, 2] # will be 0
    stiffness_31 = wot.hydrostatics.stiffness_matrix(hs_data)[2, 0] # will be 0
    stiffness_33 = wot.hydrostatics.stiffness_matrix(hs_data)[2, 2]
    mass = np.array([[mass_11, mass_13],
                     [mass_31, mass_33]])
    stiffness = np.array([[stiffness_11, stiffness_13],
                          [stiffness_31, stiffness_33]])

    # PTO
    kinematics = np.eye(fb.nb_dofs)
    names = ["SURGE", "HEAVE"]
    pto = wot.pto.PseudoSpectralPTO(nfreq, kinematics, names=names)

    # WEC
    f_add = pto.force_on_wec
    wec = wot.WEC(fb, mass, stiffness, f0, nfreq, f_add=f_add)

    # BEM
    wec.run_bem()

    # set diagonal (coupling) components to zero
    wec.hydro.added_mass[:, 0, 1] = 0.0
    wec.hydro.added_mass[:, 1, 0] = 0.0
    wec.hydro.radiation_damping[:, 0, 1] = 0.0
    wec.hydro.radiation_damping[:, 1, 0] = 0.0
    wec._del_impedance()
    wec.bem_calc_impedance()

    # solve
    obj_fun = pto.average_power
    nstate_opt = pto.nstate
    options = {'maxiter': 250, 'ftol': 1e-8}
    scale_x_wec = 100.0
    scale_x_opt = 0.1
    scale_obj = 1.0
    _, _, _, _, avg_pow_sh_sh, _ = wec.solve(
        regular_wave, obj_fun, nstate_opt, optim_options=options,
        scale_x_wec=scale_x_wec, scale_x_opt=scale_x_opt, scale_obj=scale_obj)

    # only surge PTO
    kinematics = np.array([[1.0, 0.0]])
    pto = wot.pto.PseudoSpectralPTO(nfreq, kinematics, names=["SURGE"])
    f_add = pto.force_on_wec
    wec = wot.WEC(fb, mass, stiffness, f0, nfreq, f_add=f_add)
    wec.run_bem()
    obj_fun = pto.average_power
    nstate_opt = pto.nstate
    _, _, _, _, avg_pow_sh_s, _ = wec.solve(
        regular_wave, obj_fun, nstate_opt, optim_options=options,
        scale_x_wec=scale_x_wec, scale_x_opt=scale_x_opt, scale_obj=scale_obj)

    # only heave PTO
    kinematics = np.array([[0.0, 1.0]])
    pto = wot.pto.PseudoSpectralPTO(nfreq, kinematics, names=["HEAVE"])
    f_add = pto.force_on_wec
    wec = wot.WEC(fb, mass, stiffness, f0, nfreq, f_add=f_add)
    wec.run_bem()
    obj_fun = pto.average_power
    nstate_opt = pto.nstate
    _, _, _, _, avg_pow_sh_h, _ = wec.solve(
        regular_wave, obj_fun, nstate_opt, optim_options=options,
        scale_x_wec=scale_x_wec, scale_x_opt=scale_x_opt, scale_obj=scale_obj)

    # WEC only surge
    mass = np.array([[mass_11]])
    stiffness = np.array([[stiffness_11]])
    fb = cpy.FloatingBody.from_file(meshfile, name="WaveBot")
    fb.add_translation_dof(name="SURGE")
    kinematics = np.array([[1.0]])
    pto = wot.pto.PseudoSpectralPTO(nfreq, kinematics, names=["SURGE"])
    f_add = pto.force_on_wec
    wec = wot.WEC(fb, mass, stiffness, f0, nfreq, f_add=f_add)
    wec.run_bem()
    obj_fun = pto.average_power
    nstate_opt = pto.nstate
    _, _, _, _, avg_pow_s_s, _ = wec.solve(
        regular_wave, obj_fun, nstate_opt, optim_options=options,
        scale_x_wec=scale_x_wec, scale_x_opt=scale_x_opt, scale_obj=scale_obj)

    # WEC only heave
    mass = np.array([[mass_33]])
    stiffness = np.array([[stiffness_33]])
    fb = cpy.FloatingBody.from_file(meshfile, name="WaveBot")
    fb.add_translation_dof(name="HEAVE")
    pto = wot.pto.PseudoSpectralPTO(nfreq, kinematics, names=["HEAVE"])
    f_add = pto.force_on_wec
    wec = wot.WEC(fb, mass, stiffness, f0, nfreq, f_add=f_add)
    wec.run_bem()
    obj_fun = pto.average_power
    nstate_opt = pto.nstate
    scale_x_wec = 10000.0
    scale_x_opt = 1.0
    scale_obj = 1.0
    _, _, _, _, avg_pow_h_h, _ = wec.solve(
        regular_wave, obj_fun, nstate_opt, optim_options=options,
        scale_x_wec=scale_x_wec, scale_x_opt=scale_x_opt, scale_obj=scale_obj)

    # checks
    tol = 1e0
    assert pytest.approx(avg_pow_sh_sh, tol) == avg_pow_sh_s + avg_pow_sh_h
    assert pytest.approx(avg_pow_sh_s, tol) == avg_pow_s_s
    assert pytest.approx(avg_pow_sh_h, tol) == avg_pow_h_h


@pytest.fixture
def surge_heave_wavebot():
    # frequencies
    f0 = 0.05
    nfreq = 18

    #  mesh
    meshfile = os.path.join(os.path.dirname(__file__), 'data', 'wavebot.stl')

    # capytaine floating body
    fb = cpy.FloatingBody.from_file(meshfile, name="WaveBot")
    fb.add_translation_dof(name="SURGE")
    fb.add_translation_dof(name="HEAVE")

    # hydrostatic
    hs_data = wot.hydrostatics.hydrostatics(fb)
    mass_11 = wot.hydrostatics.mass_matrix_constant_density(hs_data)[0, 0]
    mass_13 = wot.hydrostatics.mass_matrix_constant_density(hs_data)[0, 2] # will be 0
    mass_31 = wot.hydrostatics.mass_matrix_constant_density(hs_data)[2, 0] # will be 0
    mass_33 = wot.hydrostatics.mass_matrix_constant_density(hs_data)[2, 2]
    stiffness_11 = wot.hydrostatics.stiffness_matrix(hs_data)[0, 0]
    stiffness_13 = wot.hydrostatics.stiffness_matrix(hs_data)[0, 2] # will be 0
    stiffness_31 = wot.hydrostatics.stiffness_matrix(hs_data)[2, 0] # will be 0
    stiffness_33 = wot.hydrostatics.stiffness_matrix(hs_data)[2, 2]
    mass = np.array([[mass_11, mass_13],
                     [mass_31, mass_33]])
    stiffness = np.array([[stiffness_11, stiffness_13],
                          [stiffness_31, stiffness_33]])

    # WEC
    wec = wot.WEC(fb, mass, stiffness, f0, nfreq)

    # BEM
    wec.run_bem()

    # set diagonal (coupling) components to zero
    wec.hydro.added_mass[:, 0, 1] = 0.0
    wec.hydro.added_mass[:, 1, 0] = 0.0
    wec.hydro.radiation_damping[:, 0, 1] = 0.0
    wec.hydro.radiation_damping[:, 1, 0] = 0.0
    wec._del_impedance()
    wec.bem_calc_impedance()
    
    return wec


def test_multiple_dof_ps_theoretical_limit(regular_wave, surge_heave_wavebot):

    # PTO
    kinematics = np.eye(2)
    names = ["SURGE", "HEAVE"]
    pto = wot.pto.PseudoSpectralPTO(surge_heave_wavebot.nfreq,
                                    kinematics,
                                    names=names)

    # WEC
    surge_heave_wavebot.f_add = {'pto': pto.force_on_wec}

    _, fdom, _, _, obj, _ = surge_heave_wavebot.solve(regular_wave,
                                                      obj_fun=pto.average_power,
                                                      nstate_opt=pto.nstate,
                                                      optim_options={
                                                          'maxiter': 250, 
                                                          'ftol': 1e-8},
                                                      scale_x_wec=1e2,
                                                      scale_x_opt=1e-2,
                                                      scale_obj=1,
                                                      )

    plim = power_limit(fdom['excitation_force'], surge_heave_wavebot.hydro.Zi)
    assert pytest.approx(obj, 1e-1) == plim


def test_multiple_dof_fixed_structure_P(surge_heave_wavebot):
    
    kinematics = np.eye(surge_heave_wavebot.ndof)
    names = ["SURGE", "HEAVE"]
    pto = wot.pto.ProportionalPTO(kinematics, names=names)
    
    x_wec = np.random.randn(surge_heave_wavebot.nstate_wec)
    x_opt = np.random.randn(pto.nstate)
    
    pto_force = pto.force_on_wec(surge_heave_wavebot, x_wec, x_opt)
    pto_vel = pto.velocity(surge_heave_wavebot, x_wec, x_opt)
    
    assert np.all(x_opt[0] * pto_vel[:,0] == pto_force[:,0])
    assert np.all(x_opt[1] * pto_vel[:,1] == pto_force[:,1])


def test_multiple_dof_fixed_structure_PI(surge_heave_wavebot):

    kinematics = np.eye(surge_heave_wavebot.ndof)
    names = ["SURGE", "HEAVE"]
    pto = wot.pto.ProportionalIntegralPTO(kinematics, names=names)

    x_wec = np.random.randn(surge_heave_wavebot.nstate_wec)
    x_opt = np.random.randn(pto.nstate)
    
    pto_force = pto.force_on_wec(surge_heave_wavebot, x_wec, x_opt)
    pto_vel = pto.velocity(surge_heave_wavebot, x_wec, x_opt)
    pto_pos = pto.position(surge_heave_wavebot, x_wec, x_opt)
    
    assert np.all(x_opt[0] * pto_vel[:,0] + x_opt[2] * pto_pos[:,0] 
                  == pto_force[:,0])
    assert np.all(x_opt[1] * pto_vel[:,1] + x_opt[3] * pto_pos[:,1] 
                  == pto_force[:,1])


def test_buoyancy_excess(wec, pto, regular_wave):
    """Give too much buoyancy and check that equilibrium point found matches
    that given by the hydrostatic stiffness"""

    delta = np.random.randn() # excess buoyancy factor

    # remove constraints
    wec.constraints = []

    def f_b(wec, x_wec, x_opt):
        V = wec.fb.keep_immersed_part(inplace=False).mesh.volume
        rho = wec.rho
        g = 9.81
        return (1+delta) * rho * V * g * np.ones([wec.ncomponents, wec.ndof])

    def f_g(wec, x_wec, x_opt):
        g = 9.81
        m = wec.mass.item()
        return -1 * m * g * np.ones([wec.ncomponents, wec.ndof])

    wec.f_add = {**wec.f_add,
                 'Fb':f_b,
                 'Fg':f_g,
                 }

    tdom, fdom, x_wec, x_opt, avg_pow, _ = wec.solve(regular_wave,
                                                obj_fun = pto.average_power,
                                                nstate_opt = pto.nstate,
                                                scale_x_wec = 1.0,
                                                scale_x_opt = 0.01,
                                                scale_obj = 1e-1,
                                                optim_options={})

    mean_pos = tdom.pos.squeeze().mean().item()
    expected = (wec.rho * wec.fb.mesh.volume * wec.g * delta) \
        / wec.hydrostatic_stiffness.item()
    assert pytest.approx (expected, 1e-1) == mean_pos
