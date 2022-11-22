"""Functions for calculating hydrostatic and mass properties for
floating bodies.
"""


from __future__ import annotations


__all__ = [
    "stiffness_matrix",
    "inertia_matrix"
]


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

    Uses
    :py:meth:`capytaine.bodies.bodies.FloatingBody.compute_hydrostatic_stiffness`
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
    fb = _set_property(fb, 'center_of_mass', center_of_mass)
    fb = _set_property(fb, 'rotation_center', rotation_center)
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

    Uses
    :py:meth:`capytaine.bodies.bodies.FloatingBody.compute_rigid_body_inertia`
    on the full mesh.

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
    fb = _set_property(fb, 'center_of_mass', center_of_mass)
    fb = _set_property(fb, 'rotation_center', rotation_center)
    fb = _set_property(fb, 'mass', mass, rho)
    return fb.compute_rigid_body_inertia(rho=rho)


def _set_property(
    fb: FloatingBody,
    property: str,
    value: Optional[Iterable[float]],
    rho: float = _default_parameters["rho"],
) -> FloatingBody:
    """Sets default properties if not provided by the user:
        - `center_of_mass` is set to the geometric centroid
        - `mass` is set to the displaced mass
        - `rotation_center` is set to the center of mass
    """
    valid_properties = ['mass', 'center_of_mass', 'rotation_center']
    if property not in valid_properties:
        raise ValueError(
            "`property` is not a recognized property. Valid properties are " +
           f"{valid_properties}."
        )
    if not hasattr(fb, property):
        setattr(fb, property, None)
    prop_org = getattr(fb, property) is not None
    prop_new = value is not None

    if not prop_org and not prop_new:
        if property == 'mass':
            vol = fb.copy(
                name=f"{fb.name}_immersed"
                ).keep_immersed_part().volume
            def_val = rho * vol
            log_str = (
                "Setting the mass to the displaced mass.")
        elif property == 'center_of_mass':
            def_val = fb.center_of_buoyancy
            log_str = (
                "Using the geometric centroid as the center of gravity (COG).")
        elif property == 'rotation_center':
            def_val = fb.center_of_mass
            log_str = (
                "Using the center of gravity (COG) as the rotation center " +
                "for hydrostatics.")
        setattr(fb, property, def_val)
        _log.info(log_str)
    elif prop_org and prop_new:
        if not np.allclose(getattr(fb, property), value):
            raise ValueError(
               f"Both :python:`fb.{property}` and " +
               f":python:`{property}` were provided but have " +
                "different values."
            )
    elif prop_new:
        setattr(fb, property, value)

    return fb
