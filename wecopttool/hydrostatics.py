"""Functions for calculating hydrostatic and mass properties for
floating bodies.
"""


from __future__ import annotations


from typing import Iterable, Optional
import logging

import numpy as np
from capytaine import FloatingBody
from xarray import DataArray

from wecopttool.core import _default_parameters


# logger
_log = logging.getLogger(__name__)

def stiffness_matrix(
    fb: FloatingBody,
    rho: float = _default_parameters['rho'],
    g: float = _default_parameters['g'],
    center_of_mass: Optional[Iterable[float]] = None,
    rotation_center: Optional[Iterable[float]] = None
) -> DataArray:
    """Compute the hydrostatic stiffness of a Capytaine floating body.

    .. note:: Only works for rigid body DOFs which must be named
              according to the Capytaine convention (e.g.,
              :python:`"Heave"`).

    Uses :python:`capytaine.FloatingBody.compute_hydrostatic_stiffness`
    on the immersed part of the mesh.

    Parameters
    ----------
    fb
        A capytaine floating body (mesh + DOFs, and optionally center of
        mass).
    rho
        Water density in :math:`kg/m^3`.
    g
        Gravitational acceleration in :math:`m/s^2`.
    center_of_mass
        Center of gravity/mass :python:`(cx, cy, cz)`.
    rotation_center
        Center of rotation :python:`(rx, ry, rz)` for hydrostatics
        calculations.

    Raises
    ------
    ValueError
        If :python:`fb.center_of_mass is not None` and
        :python:`center_of_mass` is provided with a different value.
    """
    fb = _set_center_of_mass(fb, center_of_mass)
    fb = _set_rotation_center(fb, rotation_center)
    fb_im = fb.copy(name=f"{fb.name}_immersed").keep_immersed_part()
    return fb_im.compute_hydrostatic_stiffness(rho=rho, g=g)


def inertia_matrix(
    fb: FloatingBody,
    rho: Optional[float] = _default_parameters['rho'],
    center_of_mass: Optional[Iterable[float]] = None,
    mass: Optional[float] = None,
    rotation_center: Optional[Iterable[float]] = None,
) -> DataArray:
    """Compute the inertia (mass) matrix assuming a constant density for
    the WEC.

    .. note:: This function assumes a constant density WEC.

    Uses :python:`capytaine.FloatingBody.compute_rigid_body_inertia` on
    the full mesh.

    Parameters
    ----------
    fb
        A capytaine floating body (mesh + DOFs, and optionally center of
        mass and mass).
    rho
        Water density in :math:`kg/m^3`.
    center_of_mass
        Center of gravity/mass.
    mass
        Rigid body mass.
    rotation_center
        Center of rotation :python:`(rx, ry, rz)` for hydrostatics
        calculations.

    Raises
    ------
    ValueError
        If :python:`fb.center_of_mass is not None` and
        :python:`center_of_mass` is provided with a different value.
    ValueError
        If :python:`fb.mass is not None` and :python:`mass` is provided
        with a different value.
    """
    fb = _set_center_of_mass(fb, center_of_mass)
    fb = _set_rotation_center(fb, rotation_center)
    fb = _set_mass(fb, mass, rho)
    return fb.compute_rigid_body_inertia(rho=rho)


# TODO: combine the three hidden methods below into a single "_set_property" method
def _set_center_of_mass(
    fb: FloatingBody,
    center_of_mass: Optional[Iterable[float]],
) -> FloatingBody:
    """If COG not provided, set to geometric centroid."""
    cog_org = fb.center_of_mass is not None
    cog_new = center_of_mass is not None

    if not cog_org and not cog_new:
        fb.center_of_mass = fb.center_of_buoyancy
        _log.info(
            "Using the geometric centroid as the center of gravity (COG).")
    elif cog_org and cog_new:
        if not np.allclose(fb.center_of_mass, center_of_mass):
            raise ValueError(
                "Both :python:`fb.center_of_mass` and " +
                ":python:`center_of_mass` where provided but have " +
                "different values."
            )
    elif cog_new:
        fb.center_of_mass = center_of_mass

    return fb


def _set_mass(
    fb: FloatingBody,
    mass: Optional[float]=None,
    rho: float = _default_parameters["rho"],
) -> FloatingBody:
    """If mass is not provided, set to displaced mass."""
    mass_org = fb.mass is not None
    mass_new = mass is not None

    if not mass_org and not mass_new:
        vol = fb.copy(name=f"{fb.name}_immersed").keep_immersed_part().volume
        fb.mass = rho * vol
        _log.info("Setting the mass to the displaced mass.")
    elif mass_org and mass_new:
        if not np.isclose(fb.mass, mass):
            raise ValueError(
                "Both :python:`fb.mass` and :python:`mass` where provided " +
                "but have different values."
            )
    elif mass_new:
        fb.mass = mass

    return fb


def _set_rotation_center(
    fb: FloatingBody,
    rotation_center: Optional[Iterable[float]],
) -> FloatingBody:
    """If rotation center not provided, set to center of mass."""
    if not hasattr(fb, 'rotation_center'):
        setattr(fb, 'rotation_center', None)
    rc_org = fb.rotation_center is not None
    rc_new = rotation_center is not None

    if not rc_org and not rc_new:
        fb.rotation_center = fb.center_of_mass
        _log.info(
            "Using the center of gravity (COG) as the rotation center for hydrostatics.")
    elif rc_org and rc_new:
        if not np.allclose(fb.rotation_center, rotation_center):
            raise ValueError(
                "Both :python:`fb.rotation_center` and " +
                ":python:`rotation_center` where provided but have " +
                "different values."
            )
    elif rc_new:
        fb.rotation_center = rotation_center

    return fb
