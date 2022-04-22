"""Provide core functionality for solving the pseudo-spectral problem.
"""


from __future__ import annotations
import stat  # required for Python 3.8 & 3.9 support

# TODO
# __all__ = [
#     'WEC', 'real_to_complex_amplitudes', 'freq_array', '_degrees_to_radians',
#     ]


import logging
import copy
from typing import Iterable, Callable, Any, Optional, Mapping
from pathlib import Path
from numpy import isin

import numpy.typing as npt
import autograd.numpy as np
from autograd.builtins import isinstance, tuple, list, dict
from autograd import grad, jacobian
from pyparsing import null_debug_action
import xarray as xr
import capytaine as cpy
from scipy.optimize import minimize, OptimizeResult, Bounds
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


_log = logging.getLogger(__name__)


# Default values
_default_parameters = {'rho': 1025.0, 'g': 9.81, 'depth': np.infty}

class WEC:
    """Class representing a specific wave energy converter (WEC).
    An instance contains the  following information about the WEC,
    environment, and dynamic modeling:

    * Geometry
    * Degrees of freedom
    * Mass properties
    * Hydrostatic properties
    * Linear frequency domain hydrodynamic coefficients
    * Water properties
    * Additional dynamic forces (power take-off, mooring, nonlinear
      hydrodynamics, etc.)
    * Constraints
    """

    def __init__(self, f0, nfreq, ndof, forces, constraints, wave_direction):
        self._freq = _make_freq_array(f0, nfreq)
        self._ndof = ndof
        self._time = make_time_vec(f0, self.ncomponents) #TODO method referencing function
        self._time_mat = make_time_mat(self.omega, self.ncomponents) #TODO method referencing function
        #TODO: wave_direction

        # derivative matrix
        def block(n): return np.array([[0, 1], [-1, 0]]) * n * self.w0
        blocks = [block(n+1) for n in range(self.nfreq)]
        self.derivative_mat = block_diag(*blocks)

        self.forces = forces
        self.constraints = constraints

        # f(wec, x_wec, x_opt, wave)




    # f = 0
    # for force in forces.items()
    #     f += force(wec, x_wec, x_opt, wave)


    @staticmethod
    def from_bem(bem_data: xr.Dataset, mass: np.ndarray,
                 hydrostatic_stiffness: np.ndarray,
                 friction: Optional[np.ndarray] = None,
                 f_add: Optional[Mapping[str, Callable[[
                     WEC, np.ndarray, np.ndarray], np.ndarray]]] = None,
                 constraints: list[dict] = [],
                 nsubsteps: int=1):
        dims = ['radiating_dof', 'influenced_dof']

        if 'mass' not in list(bem_data.variables.keys()):
            bem_data['mass'] = (dims, mass)
        if 'hydrostatic_stiffness' not in list(bem_data.variables.keys()):
            bem_data['hydrostatic_stiffness'] = (dims, hydrostatic_stiffness)
        if 'friction' not in list(bem_data.variables.keys()):
            if friction is None:
                friction = bem_data['hydrostatic_stiffness']*0
            bem_data['friction'] = friction

        bem_data = _check_damping(bem_data)
        
        if f_add is None:
            f_add = dict()

        # forces in the dynamics equations
        linear_force__functions, _ = _create_standard_forces(bem_data)
        forces = linear_force__functions | f_add

        ndof = len(bem_data["influenced_dof"])
        f0 = bem_data["omega"].values[0] / (2*np.pi)
        nfreq = len(bem_data["omega"])
        return WEC(f0, nfreq, ndof, forces, constraints, wave_direction=bem_data['wave_direction'])

    @staticmethod
    def from_bem_file(file, mass: np.ndarray,
                      hydrostatic_stiffness: np.ndarray,
                      friction: Optional[np.ndarray] = None,
                      f_add: Optional[Mapping[str, StateFunction]] = None,
                      constraints: list[dict] = [],
                      nsubsteps: int=1):
        bem_data = read_file(file)  # TODO
        wec = WEC.from_bem(bem_data, mass, hydrostatic_stiffness,
                           friction, f_add, constraints,
                           nsubsteps=nsubsteps)
        return wec

    @staticmethod
    def from_floating_body(self, fb: cpy.FloatingBody, mass: np.ndarray,
                 hydrostatic_stiffness: np.ndarray, f0: float, nfreq: int,
                 wave_directions: npt.ArrayLike = np.array([0.0,]),
                 friction: Optional[np.ndarray] = None,
                 f_add: Optional[Mapping[str, StateFunction]] = None,
                 constraints: list[dict] = [],
                 rho: float = _default_parameters['rho'],
                 depth: float = _default_parameters['depth'],
                 g: float = _default_parameters['g'],
                 nsubsteps: int = 1) -> None:
        # TODO: log.info saying that the bem_data is returned and should be saved for quicker initialization later
        # RUN BEM
        _log.info(f"Running Capytaine (BEM): {nfreq} frequencies x " +
                 f"{len(wave_directions)} wave directions.")
        freq = _make_freq_array(f0, nfreq)
        write_info = ['hydrostatics', 'mesh', 'wavelength', 'wavenumber']
        bem_data = run_bem(fb, freq, wave_directions,
                        rho=rho, g=g, depth=depth, write_info=write_info)
        wec = WEC.from_bem(bem_data, mass, hydrostatic_stiffness,
                           friction, f_add, constraints,
                           nsubsteps=nsubsteps)
        return wec, bem_data


    @staticmethod
    def from_impedance(f0, nfreq, impedance, f_add, constraints):
        ndof = impedance.shape[0]
        # force_impedance =
        transfer_mat = _make_mimo_transfer_mat(impedance)  # TODO

        def force_impedance(wec, x_wec, x_opt, nsubsteps=nsubsteps):
            f_fd = vec_to_dofmat(np.dot(transfer_mat, x_wec))  # TODO
            return np.dot(time_mat, f_fd)   # TODO

        forces =  force_impedance | f_add
        WEC(f0, nfreq, ndof, forces, constraints, nsubsteps=nsubsteps)


        

    def _add_to_bem(bem_data, mass, stiffness, friction):
        dims = ['radiating_dof', 'influenced_dof']
        bem_data['mass'] = (dims, mass)
        bem_data['hydrostatic_stiffness'] = (dims, stiffness)
        bem_data = bem_data.assign_coords({'friction': friction})
        return bem_data



    # properties: frequency
    @property
    def freq(self):
        return self._freq

    @property
    def nfreq(self):
        return len(self.freq)

    @property
    def f0(self):
        return self.freq[0]

    @property
    def period(self):
        """Period :math:T=1/f in seconds."""
        return 1/self.freq

    @property
    def omega(self):
        """Frequency array in radians per second ω=2πf."""
        return self.freq * 2 * np.pi

    @property
    def w0(self):
        """Initial frequency (and spacing) in rad/s. See ``freq``."""
        return self.freq[0] * 2 * np.pi

    # properties: time
    @property
    def time(self):
        """Time array."""
        return self._time

    @property
    def dt(self):
        """Time spacing."""
        return self._time[1]

    @property
    def tf(self):
        """Final time (period)."""
        return 1/self.f0

    @property
    def time_mat(self):
        """Matrix to convert from the state vector to a time-series.
        """
        return self._time_mat

    @property
    def derivative_mat(self):
        """Derivative matrix for the state vector."""
        return self._derivative_mat

    # properties: problem size
    @property
    def ndof(self):
        """Number of degrees of freedom of the WEC."""
        return self._ndof

    @property
    def ncomponents(self):
        """Number of state values for each DOF in the WEC dynamics."""
        return 2 * self.nfreq + 1

    @property
    def nstate_wec(self):
        """Length of the  WEC dynamics state vector."""
        return self.ndof * self.ncomponents

    # methods: state vector
    def decompose_decision_var(self, state: np.ndarray
                               ) -> tuple[np.ndarray, np.ndarray]:
        """Split the state vector into the WEC dynamics state and the
        optimization (control) state. x = [x_wec, x_opt].
        """
        return state[:self.nstate_wec], state[self.nstate_wec:]

    def vec_to_dofmat(self, vec: np.ndarray) -> np.ndarray:
        """Convert a vector back to a matrix with one column per DOF.
        Opposite of ``dofmat_to_vec``. """
        return np.reshape(vec, (-1, self.ndof), order='F')

    def dofmat_to_vec(self, mat: np.ndarray) -> np.ndarray:
        """Flatten a matrix that has one column per DOF.
        Opposite of ``vec_to_dofmat``. """
        return np.reshape(mat, -1, order='F')



def _make_freq_array(f0: float, nfreq: int) -> np.ndarray:
    """Construct equally spaced frequency array.
    """
    return np.arange(1, nfreq+1)*f0


def real_to_complex_amplitudes(fd: np.ndarray, first_row_is_mean: bool = True
                               ) -> np.ndarray:
    """Convert from two real amplitudes to one complex amplitude per
    frequency. """
    fd = np.atleast_2d(fd)
    if first_row_is_mean:
        mean = fd[0:1, :]
        fd = fd[1:, :]
    else:
        ndof = fd.shape[1]
        mean = np.zeros([1, ndof])
    return np.concatenate((mean, fd[0::2, :] - 1j*fd[1::2, :]), axis=0)


def degrees_to_radians(degrees: float | npt.ArrayLike
                        ) -> float | np.ndarray:
    """Convert degrees to radians in range -π to π and sort.
    """
    radians = np.asarray(np.remainder(np.deg2rad(degrees), 2*np.pi))
    radians[radians > np.pi] -= 2*np.pi
    radians = radians.item() if (radians.size == 1) else np.sort(radians)
    return radians

def _check_damping(bem_data, tol=1e-6) -> xr.Dataset:
    damping = bem_data['radiation_damping'] + bem_data['friction']
    dmin = np.diagonal(damping,axis1=1,axis2=2).min()
    if dmin <= 0.0 + tol:
        _log.warning(f'Linear damping has negative' +
                    ' or close to zero terms; shifting up via linear friction.')
        bem_data['friction'] = bem_data['friction'] + tol-dmin
    
    return bem_data

# methods: time
def make_time_vec(f0: float, ncomponents: int, nsubsteps: int = 1) -> np.ndarray:
    """Assemble the time vector with n subdivisions.

    Parameters
    ----------
    nsubsteps: int
        Number of subdivisions between the default (implied) time
        steps.

    Returns
    -------
    time_vec: np.ndarray
    """
    nsteps = nsubsteps * ncomponents
    return np.linspace(0, 1/f0, nsteps, endpoint=False)

def make_time_mat(omega: np.ndarray, 
                  ncomponents: int, nsubsteps: int = 1) -> np.ndarray:
    """Assemble the time matrix that converts the state to
    time-series.

    Parameters
    ---------
    nsubsteps: int
        Number of subdivisions between the default (implied) time
        steps.

    Returns
    -------
    time_mat: np.ndarray
    """
    f0 = omega[0]/(2*np.pi)
    time = make_time_vec(f0, ncomponents, nsubsteps)
    wt = np.outer(time, omega)
    time_mat = np.empty((nsubsteps*ncomponents, ncomponents))
    time_mat[:, 0::2] = np.cos(wt)
    time_mat[:, 1::2] = np.sin(wt)
    return time_mat

def vec_to_dofmat(vec: np.ndarray, ndof: int) -> np.ndarray:
        """Convert a vector back to a matrix with one column per DOF.
        Opposite of ``dofmat_to_vec``. """
        return np.reshape(vec, (-1, ndof), order='F')

def _make_mimo_transfer_mat(imp: np.ndarray, ndof:int) -> np.ndarray:
    """Create a block matrix of the MIMO transfer function.
    """
    elem = [[None]*ndof for _ in range(ndof)]
    def block(re, im): return np.array([[re, im], [-im, re]])
    for idof in range(ndof):
        for jdof in range(ndof):
            Zp = imp[:, idof, jdof]
            re = np.real(Zp)
            im = np.imag(Zp)
            blocks = [block(ire, iim) for (ire, iim) in zip(re, im)]
            elem[idof][jdof] = block_diag(*blocks)
    return np.block(elem)

def _create_standard_forces(bem_data: xr.Dataset):
    w = bem_data['omega']
    A = bem_data['added_mass']
    B = bem_data['radiation_damping']
    K = bem_data['hydrostatic_stiffness']
    m = bem_data['mass']
    Bf = bem_data['friction']
    
    ndof = len(bem_data.influenced_dof)

    # TODO: m, Bf, K are not the right size. Options:
    #       1 - make them the right size: N_DOF x N_DOF x N_freq
    #       2 - calculate without using the transfer matrix
    #       3 - modify the _make_mimo_... function to accept different sizes
    
    impedance_components = dict()
    impedance_components['inertia'] = 1j*w*m
    impedance_components['radiation'] = -(B + 1j*w*A)
    impedance_components['hydrostatics'] = -1j/w*K
    impedance_components['friction'] = -Bf + B*0  #TODO: this is my way of getting the shape right, kind of sloppy?
    
    def f_from_imp(transfer_mat):
            def f(wec, x_wec, x_opt, nsubsteps=1):
                f_fd = vec_to_dofmat(np.dot(transfer_mat, x_wec), ndof)
                return np.dot(self.time_mat, f_fd)
            return f
    
    linear_force_mimo_matrices = dict()
    linear_force_functions = dict()
    for k, v in impedance_components.items():
        linear_force_mimo_matrices[k] = _make_mimo_transfer_mat(v,ndof)
        linear_force_functions[k] = f_from_imp(v)

    return linear_force_functions, linear_force_mimo_matrices

def run_bem(fb: cpy.FloatingBody, freq: Iterable[float] = [np.infty],
            wave_dirs: Iterable[float] = [0],
            rho: float = _default_parameters['rho'],
            g: float = _default_parameters['g'],
            depth: float = _default_parameters['depth'],
            write_info: Iterable[str] = []
            ) -> xr.Dataset:
    """Run Capytaine for a range of frequencies and wave directions.

    Parameters
    ----------
    fb: capytaine.FloatingBody
        The WEC as a Capytaine floating body (mesh + DOFs).
    freq: list[float]
        List of frequencies to evaluate BEM at.
    wave_dirs: list[float]
        List of wave directions to evaluate BEM at (degrees).
    rho: float, optional
        Water density in :math:`kg/m^3`.
    g: float, optional
        Gravitational acceleration in :math:`m/s^2`.
    depth: float, optional
        Water depth in :math:`m`.
    write_info: list[str], optional
        List of information to keep, passed to `capytaine` solver.
        Options are: `wavenumber`, `wavelength`, `mesh`, `hydrostatics`.

    Returns
    -------
    xarray.Dataset
        BEM results from capytaine.
    """
    if wave_dirs is not None:
        wave_dirs = np.atleast_1d(_degrees_to_radians(wave_dirs))
    solver = cpy.BEMSolver()
    test_matrix = xr.Dataset(coords={
        'rho': [rho],
        'water_depth': [depth],
        'omega': [ifreq*2*np.pi for ifreq in freq],
        'wave_direction': wave_dirs,
        'radiating_dof': list(fb.dofs.keys()),
        'g': [g],
    })
    if wave_dirs is None:
        # radiation only problem, no diffraction or excitation
        test_matrix = test_matrix.drop_vars('wave_direction')
    write_info = {key: True for key in write_info}
    wec_im = fb.copy(name=f"{fb.name}_immersed").keep_immersed_part()
    return solver.fill_dataset(test_matrix, [wec_im], **write_info)
