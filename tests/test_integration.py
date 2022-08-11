""" Integration tests spanning WecOptTool.
"""
import wave
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
    pto = wot.pto.PTO(ndof, kinematics, names=['test PTO'])
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
def wec_from_bem(bem, fb, pto):
    """Simple WEC: 1 DOF, no constraints."""
    mass = wot.hydrostatics.inertia_matrix(fb).values
    hstiff = wot.hydrostatics.stiffness_matrix(fb).values
    f_add = {"PTO": pto.force_on_wec}
    return wot.WEC.from_bem(bem, mass, hstiff, f_add=f_add)


@pytest.fixture
def regular_wave(f1, nfreq):
    wfreq = 0.3
    wamp = 0.0625 
    wphase = 30
    wdir = 0
    waves = wot.waves.regular_wave(f1, nfreq, wfreq, wamp, wphase, wdir)
    return waves


def test_ryan(wec_from_bem, regular_wave, pto, nfreq):

    res_fd, res_td, res = wec_from_bem.solve(waves=regular_wave,
                                             obj_fun=pto.average_power,
                                             nstate_opt=2*nfreq+1,
                                             )
    
    results_fd, results_td = pto.post_process(wec_from_bem,res,
                                              nsubsteps=4)

    print(res_fd)

    pass


# def test_from_floatingbody():
    
#     wb = wot.geom.WaveBot()  # use standard dimensions
#     mesh_size_factor = 0.5  # 1.0 for default, smaller to refine mesh
#     mesh = wb.mesh(mesh_size_factor)
    
#     fb = cpy.FloatingBody.from_meshio(mesh, name="WaveBot")
#     fb.add_translation_dof(name="HEAVE")
    
#     mass = wot.hydrostatics.inertia_matrix(fb).values
#     stiffness = wot.hydrostatics.stiffness_matrix(fb).values

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
