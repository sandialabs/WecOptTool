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


# TODO: Currently just testing that it runs.
#       Should test that it behaves correctly.
#       E.g. check types, check dimensions, compare to stored data, etc.
#       Use ``assert ...``.


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
    f_add = pto.force_on_wec

    wec = wot.WEC(fb, mass, stiffness, f0, nfreq,  rho=rho,
                    f_add=f_add, constraints=constraints)

    # BEM
    wec.run_bem()

    return wec


@pytest.fixture(scope="module")
def wave(wec):
    freq = 0.2
    amplitude = 0.25
    phase = 0.0
    wave = wot.waves.regular_wave(wec.f0, wec.nfreq, freq, amplitude, phase)
    return wave


@pytest.fixture(scope="module")
def pto(wec):
    kinematics = np.eye(wec.ndof)
    pto = wot.pto.PseudoSpectralPTO(wec.nfreq, kinematics)
    return pto


def test_bem_io(wec):
    # write BEM file
    bem_file = 'bem.nc'
    wec.write_bem(bem_file)

    # read BEM file
    wec.read_bem(bem_file)

    # cleanup
    os.remove(bem_file)


def test_wave_excitation(wec, wave):
    # wave excitation
    _, _ = wot.wave_excitation(wec.hydro, wave)


def test_solve(wec, wave, pto):
    # solve
    options = {}
    obj_fun = pto.energy
    nstate_opt = pto.nstate
    maximize = True
    _, _, x_wec, x_opt, _, _ = wec.solve(
        wave, obj_fun, nstate_opt, optim_options=options, maximize=maximize)

    # post-process
    _, _ = pto.post_process(wec, x_wec, x_opt)


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


def test_pto(wec):
    kinematics = np.eye(wec.ndof)
    pto = wot.pto.ProportionalPTO(kinematics, ['DOF_1'])
    _ = pto.ndof_wec
    x_wec = np.zeros(wec.nstate_wec)
    x_pto = np.zeros(pto.nstate)
    _ = pto.position(wec, x_wec, x_pto)
    _ = pto.acceleration(wec, x_wec, x_pto)
    _ = pto.energy(wec, x_wec, x_pto)


def test_wavebot_ps_cc(wec,wave,pto):
    """Check that power obtained using pseudo-spectral with no constraints
    equals theoretical limit.
    """
    wec.constraints = []
    obj_fun = pto.average_power
    nstate_opt = pto.nstate
    wec_tdom, wec_fdom, x_wec, x_opt, average_power, _ = wec.solve(
    wave, obj_fun, nstate_opt, optim_options={'maxiter': 1000, 'ftol': 1e-8},
    scale_x_opt=1e3)
    idof = 0
    Fe = wec_fdom['excitation_force'][1:, idof]
    Zi = wec.hydro.Zi[:, idof, idof]
    power_limit = -1*np.sum(np.abs(Fe)**2 / (8*np.real(Zi))).values.item()

    assert pytest.approx(average_power, 1e-5) == power_limit

def test_examples_device_wavebot_mesh():
    wb = WaveBot()
    mesh = wb.mesh(mesh_size_factor=0.5)
    assert type(mesh) is meshio._mesh.Mesh


def test_examples_device_wavebot_plot_cross_section():
    wb = WaveBot()
    wb.plot_cross_section()
