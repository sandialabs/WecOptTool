""" Unit tests for functions in the :python:`hydrostatics.py` module.
"""

import pytest
import capytaine as cpy
import numpy as np

import wecopttool as wot
from wecopttool.core import _default_parameters


# setup simple constant density rectangular barge
@pytest.fixture()
def rho():
    """Water density [kg/m^3]."""
    return _default_parameters['rho']


@pytest.fixture()
def g():
    """Gravitational acceleration [m/s^2]."""
    return _default_parameters['g']


@pytest.fixture()
def lx():
    """Barge length [m]."""
    return 4.0


@pytest.fixture()
def ly():
    """Barge width [m]."""
    return 5.0


@pytest.fixture()
def lz():
    """Barge height [m]."""
    return 2.0


@pytest.fixture()
def cog():
    """Center of gravity of the barge."""
    return (0.0, 0.0, 0.0)


@pytest.fixture()
def mass(lx, ly, lz, rho):
    """Mass of the barge."""
    return lx*ly*lz/2 * rho


@pytest.fixture()
def fb(lx, ly, lz):
    """Simple constant density rectangular barge."""
    rect = cpy.RectangularParallelepiped(
        (lx, ly, lz), resolution=(100, 100, 10), center=(0.0, 0.0, 0.0,))
    rect.add_all_rigid_body_dofs()
    rect.center_of_mass = None
    fb.mass = None
    return rect


@pytest.fixture()
def fb_mass(fb, cog, mass):
    """Simple constant density rectangular barge with the mass
    properties specified.
    """
    fbc = fb.copy()
    fbc.center_of_mass = cog
    fbc.mass = mass
    return fbc


@pytest.fixture()
def stiffness(lx, ly, lz, rho, g):
    """True (theoretical/calculated) value of the stiffness matrix for
    the barge.
    """
    stiffness = np.zeros([6, 6])
    stiffness[2, 2] = rho*g * lx*ly
    stiffness[2, 3] = 0.0
    stiffness[2, 4] = 0.0
    stiffness[3, 3] = (rho*g * lx*ly**3 /12) + rho*g*(lx*ly*lz/2)*(-lz/4)
    stiffness[3, 4] = 0.0
    stiffness[3, 5] = 0.0
    stiffness[4, 4] = (rho*g * ly*lx**3 /12) + rho*g*(lx*ly*lz/2)*(-lz/4)
    stiffness[4, 5] = 0.0
    stiffness[3, 2] = stiffness[2, 3]
    stiffness[4, 2] = stiffness[2, 4]
    stiffness[4, 3] = stiffness[3, 4]
    return stiffness


@pytest.fixture()
def inertia(lx, ly, lz, mass):
    """True (theoretical/calculated) value of the inertia matrix for
    the barge.
    """
    inertia = np.zeros([6, 6])
    inertia[0, 0] = mass
    inertia[1, 1] = mass
    inertia[2, 2] = mass
    inertia[3, 3] = mass/12 * (ly**2 + lz**2)
    inertia[3, 4] = 0.0
    inertia[3, 5] = 0.0
    inertia[4, 3] = 0.0
    inertia[4, 4] = mass/12 * (lx**2 + lz**2)
    inertia[4, 5] = 0.0
    inertia[5, 3] = 0.0
    inertia[5, 4] = 0.0
    inertia[5, 5] = mass/12 * (lx**2 + ly**2)
    return inertia


class TestStiffnessMatrix:
    """Test function :python:`hydrostatics.stiffness_matrix`."""

    def test_inferred_cog(self, fb, rho, g, stiffness):
        """Test the function with the center of gravity not provided,
        but inferred.
        """
        stiffness_calc = wot.hydrostatics.stiffness_matrix(fb, rho, g)
        assert np.allclose(stiffness, stiffness_calc, rtol=0.01)


    def test_given_cog(self, fb, rho, g, cog, stiffness):
        """Test the function with the center of gravity provided as a
        function input and not in the floating body.
        """
        stiffness_calc = wot.hydrostatics.stiffness_matrix(fb, rho, g, cog)
        assert np.allclose(stiffness, stiffness_calc, rtol=0.01)


    def test_given_cog_fb(self, fb_mass, rho, g, stiffness):
        """Test the function with the center of gravity provided in the
        floating body and not as a function input.
        """
        stiffness_calc = wot.hydrostatics.stiffness_matrix(fb_mass, rho, g)
        assert np.allclose(stiffness, stiffness_calc, rtol=0.01)


    def test_given_cog_redundant(self, fb_mass, rho, g, cog, stiffness):
        """Test the function with the center of gravity provided in both
        the floating body and the function inputs.
        """
        stiffness_calc = wot.hydrostatics.stiffness_matrix(fb_mass, rho, g, cog)
        assert np.allclose(stiffness, stiffness_calc, rtol=0.01)


    def test_cog_mismatch(self, fb_mass, rho, g):
        """Test that the function fails if the center of gravity is
        provided in both the floating body and the function inputs but
        have different values.
        """
        cog_wrong = (0, 0, -0.1)
        with pytest.raises(ValueError):
            wot.hydrostatics.stiffness_matrix(fb_mass, rho, g, cog_wrong)


class TestInertiaMatrix:
    """Test function :python:`hydrostatics.test_inertia_matrix`."""

    def test_inferred_mass(self, fb, rho, inertia):
        """Test the function with the mass not provided, but inferred.
        """
        inertia_calc = wot.hydrostatics.inertia_matrix(fb, rho)
        assert np.allclose(inertia, inertia_calc, rtol=0.01)


    def test_given_mass(self, fb, rho, cog, mass, inertia):
        """Test the function with the mass provided as a function input
        and not in the floating body.
        """
        inertia_calc = wot.hydrostatics.inertia_matrix(fb, rho, cog, mass)
        assert np.allclose(inertia, inertia_calc, rtol=0.01)


    def test_given_mass_fb(self, fb_mass, rho, cog, mass, inertia):
        """Test the function with the mass provided in the floating body
        and not as a function input.
        """
        inertia_calc = wot.hydrostatics.inertia_matrix(fb_mass, rho)
        assert np.allclose(inertia, inertia_calc, rtol=0.01)


    def test_given_mass_redundant(self, fb_mass, rho, cog, mass, inertia):
        """Test the function with the mass provided in both the floating
        body and the function inputs.
        """
        inertia_calc = wot.hydrostatics.inertia_matrix(fb_mass, rho, cog, mass)
        assert np.allclose(inertia, inertia_calc, rtol=0.01)


    def test_mass_mismatch(self, fb_mass, rho, cog, mass):
        """Test that the function fails if the mass is provided in both
        the floating body and the function inputs but have different
        values.
        """
        mass_wrong = 0.9*mass
        with pytest.raises(ValueError):
            wot.hydrostatics.inertia_matrix(fb_mass, rho, cog, mass_wrong)
