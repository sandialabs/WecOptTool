""" Integration tests spanning WecOptTool.
"""
import pytest
from pytest import approx
import wecopttool as wot
import capytaine as cpy
import numpy as np


@pytest.fixture()
def f1():
    return 0.05


@pytest.fixture()
def nfreq():
    return 18


@pytest.fixture()
def pto():
    """Basic PTO: unstructured, 1 DOF, mechanical power."""
    ndof = 1
    kinematics = np.eye(ndof)
    pto = wot.pto.PTO(ndof, kinematics)
    return pto


@pytest.fixture
def fb():
    #  mesh
    mesh_size_factor = 0.5 # 1.0 for default, smaller to refine mesh
    wb = wot.geom.WaveBot()  # use standard dimensions
    mesh = wb.mesh(mesh_size_factor)

    # capytaine floating body
    fb = cpy.FloatingBody.from_meshio(mesh, name="WaveBot")
    fb.add_translation_dof(name="Heave")
    return fb


@pytest.fixture()
def bem(f1, nfreq, fb):
    freq = wot.frequency(f1, nfreq, False)
    return wot.run_bem(fb, freq)


@pytest.fixture()
def wec_from_bem(f1, nfreq, bem, fb, pto):
    """Simple WEC: 1 DOF, no constraints."""
    mass = wot.hydrostatics.inertia_matrix(fb).values
    hstiff = wot.hydrostatics.stiffness_matrix(fb).values
    f_add = {"PTO": pto.force_on_wec}
    return wot.WEC.from_bem(bem, f1, nfreq, mass, hstiff, f_add=f_add)


@pytest.fixture()
def wec_from_floatingbody(f1, nfreq, bem, fb, pto):
    """Simple WEC: 1 DOF, no constraints."""
    mass = wot.hydrostatics.inertia_matrix(fb).values
    hstiff = wot.hydrostatics.stiffness_matrix(fb).values
    f_add = {"PTO": pto.force_on_wec}
    wec = wot.WEC.from_floating_body(bem, f1, nfreq, mass, hstiff, f_add=f_add)
    return wec


@pytest.fixture()
def wec_from_rao_transfer_function(bem, pto, fb):
    """Simple WEC: 1 DOF, no constraints."""

    w = bem['omega'].values

    w = np.expand_dims(w, [1, 2])
    A = bem['added_mass'].values
    B = bem['radiation_damping'].values
    K = wot.hydrostatics.stiffness_matrix(fb).values

    freqs = w * 2 * np.pi
    transfer_func = K + 1j*w*B + -1*w**2*A
    transfer_func = np.concatenate([np.expand_dims(K, 0), transfer_func])
    transfer_func = -1*transfer_func
    exc_coeff = bem['Froude_Krylov_force'] + bem['diffraction_force']
    f_add = {"PTO": pto.force_on_wec}

    wec = wot.WEC.from_rao_transfer_function(
        freqs, transfer_func, exc_coeff, f_add)
    return wec



# def test_from_floatingbody():
#     wb = wot.geom.WaveBot()  # use standard dimensions
#     mesh_size_factor = 0.5  # 1.0 for default, smaller to refine mesh
#     mesh = wb.mesh(mesh_size_factor)
#     fb = cpy.FloatingBody.from_meshio(mesh, name="WaveBot")
#     fb.add_translation_dof(name="HEAVE")
#     hs_data = wot.hydrostatics.hydrostatics(fb)
#     mass_33 = wot.hydrostatics.mass_matrix_constant_density(hs_data)[2, 2]
#     mass = np.atleast_2d(mass_33)
#     stiffness_33 = wot.hydrostatics.stiffness_matrix(hs_data)[2, 2]
#     stiffness = np.atleast_2d(stiffness_33)

#     wec = wot.WEC.from_floating_body(fb=fb,
#                                mass=mass,
#                                hydrostatic_stiffness=stiffness,
#                                f1=0.05,
#                                nfreq=50,
#                                wave_directions=[0],
#                                friction=None,
#                                constraints=None,
#                                rho=1e3,
#                                depth=6.1,
#                                )

#     print(repr(wec))
