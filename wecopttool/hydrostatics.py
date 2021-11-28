"""Provide functions for calculating hydrostatic and mass
properties for floating bodies.
"""


from __future__ import annotations  # TODO: delete after python 3.10
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.linalg import block_diag
from meshmagick.hydrostatics import compute_hydrostatics
import capytaine as cpy


def hydrostatics(wec: cpy.FloatingBody, rho: float = 1025, g: float = 9.81,
                 cog: npt.ArrayLike = [0.0, 0.0, 0.0]) -> dict[str, Any]:
    """Compute the hydrostatic properties of a Capytaine floating body
    using MeshMagick

    Parameters
    ----------
        wec: capytaine.FloatingBody
            The WEC as a capytaine floating body (mesh + DOFs).
        rho: float, optional
            Water density in :math:`kg/m^3`.
        g: float, optional
            Gravitational acceleration in :math:`m/s^2`.
        cog: list, optional
            WEC's center of gravity: :math:`[C_x, C_y, C_z]`

    Returns
    -------
        dict
            MeshMagick hydrostatic data
    """
    mesh = wec.mesh.merged().to_meshmagick()
    return compute_hydrostatics(mesh, cog, rho, g, at_cog=True)


def stiffness_matrix(hs_data: dict[str, Any]) -> np.ndarray:
    """Get 6x6 hydrostatic stiffness matrix from MeshMagick
    hydrostatic data.

    Parameters
    ----------
        hs_data: dict
            MeshMagick hydrostatic data

    Returns
    -------
    np.ndarray
        Hydrostatic stiffness matrix. Shape: 6x6.
    """
    return block_diag(0, 0, hs_data['stiffness_matrix'], 0)


def mass_matrix_constant_density(hs_data: dict[str, Any],
                                 mass: float | None = None
                                 ) -> np.ndarray:
    """Create the 6x6 mass matrix assuming a constant density for the
    WEC.

    Parameters
    ----------
    hs_data: dict
        Hydrostatic data from MeshMagick
    mass: float, optional
        Mass of the floating object, if ``None`` use displaced mass

    Returns
    -------
     np.array
        The mass matrix. Shape: (6, 6).
    """
    if mass is None:
        mass = hs_data['disp_mass']
    rho_wec = mass / hs_data['mesh'].volume
    rho_ratio = rho_wec / hs_data['rho_water']
    mom_inertia = np.array([
        [hs_data['Ixx'], -1*hs_data['Ixy'], -1*hs_data['Ixz']],
        [-1*hs_data['Ixy'], hs_data['Iyy'], -1*hs_data['Iyz']],
        [-1*hs_data['Ixz'], -1*hs_data['Iyz'], hs_data['Izz']]])
    mom_inertia *= rho_ratio
    return block_diag(mass, mass, mass, mom_inertia)
