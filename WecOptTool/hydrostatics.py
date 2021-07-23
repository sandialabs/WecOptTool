
import autograd.numpy as np
from scipy.linalg import block_diag
from meshmagick.hydrostatics import compute_hydrostatics


# hydrostatics & mass (MeshMagick)

def hydrostatics(wec, rho=1025, grav=9.81, cog=[0.0, 0.0, 0.0]):
    """ Compute the hydrostatic properties of a Capytaine floating body
    using MeshMagick

    Parameters
    ----------
        wec: capytaine floating body
            Capytaine representation of the WEC
        rho: float
            density of water
        grav: float
            gravitational acceleration
        cog: list, length 3
            WEC's center of gravity

    Returns
    -------
        dict
            MeshMagick hydrostatic data
    """
    mesh = wec.mesh.to_meshmagick()
    hs_data = compute_hydrostatics(mesh, cog, rho, grav, at_cog=True)
    return hs_data


def stiffness_matrix(hs_data):
    """ Get 6x6 hydrostatic stiffness matrix from MeshMagick
    hydrostatic data. """
    return block_diag(0, 0, hs_data['stiffness_matrix'], 0)


def mass_matrix_constant_density(hs_data, mass=None):
    """ Create the 6x6 mass matrix assuming a constant density for the
    WEC.

    Parameters
    ----------
    hs_data: dict
        hydrostatic data from MeshMagick
    mass: float, optional
        mass of the floating object, if `None` use displaced mass

    Returns
    -------
     np.array, shape (6, 6)
        the mass matrix
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
