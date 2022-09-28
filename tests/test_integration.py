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
    wec = wot.WEC.from_bem(bem, mass, hstiff, f_add=f_add)
    return wec

@pytest.fixture()
def wec_from_floatingbody(f1, nfreq, fb, pto):
    """Simple WEC: 1 DOF, no constraints."""
    mass = wot.hydrostatics.inertia_matrix(fb).values
    hstiff = wot.hydrostatics.stiffness_matrix(fb).values
    f_add = {"PTO": pto.force_on_wec}
    wec = wot.WEC.from_floating_body(fb, f1, nfreq, mass, hstiff, f_add=f_add)
    return wec


@pytest.fixture()
def wec_from_impedance(bem, pto, fb):
    """Simple WEC: 1 DOF, no constraints."""
    bemc = bem.copy().transpose(
        "radiating_dof", "influenced_dof", "omega", "wave_direction")
    omega = bemc['omega'].values
    w = np.expand_dims(omega, [0, 1])
    A = bemc['added_mass'].values
    B = bemc['radiation_damping'].values
    mass = wot.hydrostatics.inertia_matrix(fb).values
    hstiff = wot.hydrostatics.stiffness_matrix(fb).values
    K = np.expand_dims(hstiff, 2)

    freqs = omega / (2 * np.pi)
    impedance = (A + mass)*(1j*w) + B + K/(1j*w)
    exc_coeff = bem['Froude_Krylov_force'] + bem['diffraction_force']
    f_add = {"PTO": pto.force_on_wec}

    wec = wot.WEC.from_impedance(freqs, impedance, exc_coeff, hstiff, f_add)
    return wec


def test_same_wec_init(
    wec_from_bem,
    wec_from_floatingbody,
    wec_from_impedance,
    pto,
    f1,
    nfreq,
):
    waves = wot.waves.regular_wave(f1, nfreq, 0.3, 0.0625)
    obj_fun = pto.average_power
    _, _, bem_res = wec_from_bem.solve(waves, obj_fun, 2*nfreq+1)
    _, _, fb_res = wec_from_floatingbody.solve(waves, obj_fun, 2*nfreq+1)
    _, _, imp_res = wec_from_impedance.solve(waves, obj_fun, 2*nfreq+1)

    assert fb_res.fun == approx(bem_res.fun, rel=0.01)
    assert imp_res.fun == approx(bem_res.fun, rel=0.01)
