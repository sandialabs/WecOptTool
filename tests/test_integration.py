""" Integration tests spanning WecOptTool.
"""
import pytest
from pytest import approx
import wecopttool as wot
import capytaine as cpy
import numpy as np


def test_from_floatingbody():
    wb = wot.geom.WaveBot()  # use standard dimensions
    mesh_size_factor = 0.5  # 1.0 for default, smaller to refine mesh
    mesh = wb.mesh(mesh_size_factor)
    fb = cpy.FloatingBody.from_meshio(mesh, name="WaveBot")
    fb.add_translation_dof(name="HEAVE")
    hs_data = wot.hydrostatics.hydrostatics(fb)
    mass_33 = wot.hydrostatics.mass_matrix_constant_density(hs_data)[2, 2]
    mass = np.atleast_2d(mass_33)
    stiffness_33 = wot.hydrostatics.stiffness_matrix(hs_data)[2, 2]
    stiffness = np.atleast_2d(stiffness_33)

    wec = wot.WEC.from_floating_body(fb=fb,
                               mass=mass,
                               hydrostatic_stiffness=stiffness,
                               f1=0.05,
                               nfreq=50,
                               wave_directions=[0],
                               friction=None,
                               constraints=None,
                               rho=1e3,
                               depth=6.1,
                               )

    print(repr(wec))
