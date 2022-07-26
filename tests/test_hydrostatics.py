""" Unit tests for functions in the :python:`hydrostatics.py` module.
"""

import pytest
import capytaine as cpy
import numpy as np

import wecopttool as wot
from wecopttool.core import _default_parameters


@pytest.fixture()
def rho(): return _default_parameters['rho']


@pytest.fixture()
def g(): return _default_parameters['g']


@pytest.fixture()
def lx(): return 4.0


@pytest.fixture()
def ly(): return 5.0


@pytest.fixture()
def lz(): return 2.0


@pytest.fixture()
def fb(lx, ly, lz):
    """Simple constant density recatangular barge."""
    rect = cpy.RectangularParallelepiped(
        (lx, ly, lz), resolution=(100, 100, 10), center=(0.0, 0.0, 0.0,))
    rect.add_all_rigid_body_dofs()
    return rect


def test_stiffness_matrix(fb, lx, ly, lz, rho, g):
    cog = (0.0, 0.0, 0.0)

    fbd = fb.copy()
    fbd.center_of_mass = None
    calc_default = wot.hydrostatics.stiffness_matrix(fbd, rho, g)

    fbc = fb.copy()
    fbc.center_of_mass = cog
    calc = wot.hydrostatics.stiffness_matrix(fbc, rho, g)
    calc_redundant = wot.hydrostatics.stiffness_matrix(fbc, rho, g, cog)

    truth = np.zeros([6, 6])

    truth[2, 2] = rho*g * lx*ly
    truth[2, 3] = 0.0
    truth[2, 4] = 0.0
    truth[3, 3] = (rho*g * lx*ly**3 /12) + rho*g*(lx*ly*lz/2)*(-lz/4)
    truth[3, 4] = 0.0
    truth[3, 5] = 0.0
    truth[4, 4] = (rho*g * ly*lx**3 /12) + rho*g*(lx*ly*lz/2)*(-lz/4)
    truth[4, 5] = 0.0

    truth[3, 2] = truth[2, 3]
    truth[4, 2] = truth[2, 4]
    truth[4, 3] = truth[3, 4]

    assert np.allclose(truth, calc_default, rtol=0.01) # infer COG
    assert np.allclose(truth, calc, rtol=0.01) # given COG
    assert np.allclose(truth, calc_redundant, rtol=0.01) # COG given twice
    with pytest.raises(ValueError):
        wot.hydrostatics.stiffness_matrix(fbc, rho, g, (0, 0, -0.1))  # error


def test_inertia_matrix(fb, lx, ly, lz, rho, g):
    mass = lx*ly*lz/2 * rho
    cog = (0.0, 0.0, 0.0)

    fbd = fb.copy()
    fbd.center_of_mass = None
    fbd.mass = None
    calc_default = wot.hydrostatics.inertia_matrix(fbd, rho)

    fbc = fb.copy()
    fbc.center_of_mass = cog
    fbc.mass = mass
    calc = wot.hydrostatics.inertia_matrix(fbc, rho)
    calc_redundant = wot.hydrostatics.inertia_matrix(fbc, rho, cog, mass)

    truth = np.zeros([6, 6])

    truth[0, 0] = mass
    truth[1, 1] = mass
    truth[2, 2] = mass

    truth[3, 3] = mass/12 * (ly**2 + lz**2)
    truth[3, 4] = 0.0
    truth[3, 5] = 0.0
    truth[4, 3] = 0.0
    truth[4, 4] = mass/12 * (lx**2 + lz**2)
    truth[4, 5] = 0.0
    truth[5, 3] = 0.0
    truth[5, 4] = 0.0
    truth[5, 5] = mass/12 * (lx**2 + ly**2)


    # assert np.allclose(truth, calc_default, rtol=0.01) # infer mass  # TODO: Uncomment after Capytaine issue # 182 is fixed.
    assert np.allclose(truth, calc, rtol=0.01) # given mass
    assert np.allclose(truth, calc_redundant, rtol=0.01) # mass given twice
    with pytest.raises(ValueError):
        wot.hydrostatics.inertia_matrix(fbc, rho, mass=0.9*mass)  # error
