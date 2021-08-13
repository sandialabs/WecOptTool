#!/usr/bin/env python3
#  -*- coding: utf-8 -*-

import os

import pytest
import autograd.numpy as np
import capytaine as cpy

import wecopttool as wot
from wecopttool.examples import WaveBot


# TODO: Currently just testing that it runs.
#       Should test that it behaves correctly.
#       E.g. check types, check dimensions, compare to stored data, etc.
#       Use ``assert ...``.


@pytest.fixture(scope="module")
def wec():
    # water properties
    rho = 1e3

    # frequencies
    f0 = 0.05
    num_freq = 18
    mesh = WaveBot.mesh()
    mesh_file = 'tmp_mesh.stl'
    mesh.write(mesh_file)
    fb = cpy.FloatingBody.from_file(mesh_file, name='WaveBot')
    os.remove(mesh_file)
    fb.add_translation_dof(name="HEAVE")

    # mass and hydrostatic stiffness
    hs_data = wot.hydrostatics.hydrostatics(fb, rho=rho)
    mass_33 = wot.hydrostatics.mass_matrix_constant_density(hs_data)[2, 2]
    mass = np.atleast_2d(mass_33)
    stiffness_33 = wot.hydrostatics.stiffness_matrix(hs_data)[2, 2]
    stiffness = np.atleast_2d(stiffness_33)

    # PTO: state, force, power (objective function)
    kinematics = np.eye(fb.nb_dofs)
    _, f_pto, _, _ = wot.pto.pseudospectral_pto(num_freq, kinematics)

    # create WEC
    wec = wot.WEC(fb, mass, stiffness, f0, num_freq, f_add=f_pto, rho=rho)

    # run BEM
    wec.run_bem()

    return wec


def test_bem_io(wec):
    # write BEM file
    bem_file = 'bem.nc'
    wec.write_bem(bem_file)

    # read BEM file
    wec.read_bem(bem_file)

    # cleanup
    os.remove(bem_file)


def test_wave_excitation(wec):
    # wave
    freq = 0.2
    amplitude = 0.25
    phase = 0.0
    waves = wot.waves.regular_wave(
        wec.f0, wec.num_freq, freq, amplitude, phase)

    # wave excitation
    _ = wot.wave_excitation(wec.hydro, waves)
