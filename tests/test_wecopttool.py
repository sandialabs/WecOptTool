#!/usr/bin/env python3
#  -*- coding: utf-8 -*-

import os

import pytest
import autograd.numpy as np
from autograd.builtins import isinstance, tuple, list, dict
import capytaine as cpy
import pygmsh
import gmsh

import wecopttool as wot


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
    h1 = 0.17
    h2 = 0.37
    r1 = 0.88
    r2 = 0.35
    freeboard = 0.01
    mesh_size_factor = 0.1

    with pygmsh.occ.Geometry() as geom:
        gmsh.option.setNumber('Mesh.MeshSizeFactor', mesh_size_factor)
        cyl = geom.add_cylinder([0, 0, 0], [0, 0, -h1], r1)
        cone = geom.add_cone([0, 0, -h1], [0, 0, -h2], r1, r2)
        geom.translate(cyl, [0, 0, freeboard])
        geom.translate(cone, [0, 0, freeboard])
        geom.boolean_union([cyl, cone])
        mesh = geom.generate_mesh()

    # capytaine floating body
    fb = cpy.FloatingBody.from_meshio(mesh, name="WaveBot")
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
    constraints = []

    # WEC
    f_add = pto.force_on_wec

    wec = wot.WEC(fb, mass, stiffness, f0, nfreq,  rho=rho,
                  f_add=f_add, constraints=constraints)

    # BEM
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
        wec.f0, wec.nfreq, freq, amplitude, phase)

    # wave excitation
    _, _ = wot.wave_excitation(wec.hydro, waves)
