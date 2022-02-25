#!/usr/bin/env python3
#  -*- coding: utf-8 -*-

import os

import pytest
import autograd.numpy as np
from autograd.builtins import isinstance, tuple, list, dict
import capytaine as cpy
import meshio

import wecopttool as wot
from wecopttool.geom import WaveBot
from wecopttool.core import power_limit


@pytest.fixture(scope="module")
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


@pytest.fixture(scope="module")
def regular_wave(wec):
    freq = 0.5
    amplitude = 0.25
    phase = 0.0
    wave = wot.waves.regular_wave(wec.f0, wec.nfreq, freq, amplitude, phase)
    return wave


@pytest.fixture(scope="module")
def resonant_wave(wec):
    freq = wec.natural_frequency()[0].squeeze().item()
    amplitude = 0.25
    phase = 0.0
    wave = wot.waves.regular_wave(wec.f0, wec.nfreq, freq, amplitude, phase)
    return wave


@pytest.fixture(scope="module")
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
    _, _, x_wec, x_opt, avg_pow, _ = wec.solve(regular_wave, obj_fun,
        nstate_opt,
        scale_x_wec = 1.0,
        scale_x_opt = 0.01,
        scale_obj = 1e-1,
        optim_options=options)
    
    pto_tdom, _ = pto.post_process(wec, x_wec, x_opt)
    
    assert pytest.approx(-1*f_min, 1e-5) == pto_tdom['force'].min().values.item()
    assert pytest.approx(f_max, 1e-5) == pto_tdom['force'].max().values.item()


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
    plim = power_limit(fdom['excitation_force'][1:, 0], wec.hydro.Zi[:, 0, 0])

    assert pytest.approx(average_power, 1e-5) == plim


def test_wavebot_p_cc(wec,resonant_wave):
    """Check that power from proportional damping controller can match
    theorectical limit at the natural resonance.
    """

    # remove constraints
    wec.constraints = []

    # update PTO
    kinematics = np.eye(wec.ndof)
    pto = wot.pto.ProportionalPTO(kinematics)
    obj_fun = pto.average_power
    nstate_opt = pto.nstate
    wec.f_add = {'PTO': pto.force_on_wec}

    _, fdom, _, xopt, average_power, _ = wec.solve(resonant_wave, obj_fun, nstate_opt,
        optim_options={'maxiter': 1000, 'ftol': 1e-8}, scale_x_opt=1e3)
    plim = power_limit(fdom['excitation_force'][1:, 0],
                       wec.hydro.Zi[:, 0, 0]).item()

    assert pytest.approx(average_power, 0.03) == plim


def test_wavebot_pi_cc(wec,regular_wave):
    """Check that power from proportional integral (PI) controller can match
    theorectical limit at any single wave frequency (i.e., regular wave).
    """

    # remove constraints
    wec.constraints = []

    # update PTO
    kinematics = np.eye(wec.ndof)
    pto = wot.pto.ProportionalIntegralPTO(kinematics)
    obj_fun = pto.average_power
    nstate_opt = pto.nstate
    wec.f_add = {'PTO': pto.force_on_wec}

    tdom, fdom, xwec, xopt, average_power, res = wec.solve(regular_wave,
        obj_fun, nstate_opt,
        optim_options={'maxiter': 1000, 'ftol': 1e-8}, scale_x_opt=1e3)
    plim = power_limit(fdom['excitation_force'][1:, 0],
                       wec.hydro.Zi[:, 0, 0]).item()

    assert pytest.approx(average_power, 0.03) == plim


def test_examples_device_wavebot_mesh():
    wb = WaveBot()
    mesh = wb.mesh(mesh_size_factor=0.5)
    assert type(mesh) is meshio._mesh.Mesh


def test_examples_device_wavebot_plot_cross_section():
    wb = WaveBot()
    wb.plot_cross_section()


def test_buoyancy_excess(wec, regular_wave, pto):
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
    