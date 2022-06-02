"""Provide core functionality for solving the pseudo-spectral problem
for wave energy converters.
"""


from __future__ import annotations  # required for Python 3.8 & 3.9 support


__all__ = [
    'WEC', 'TWEC', 'TStateFunction',
    'frequency', 'time', 'time_mat', 'derivative_mat', 'degrees_to_radians',
    'ncomponents', 'standard_forces', 'mimo_transfer_mat', 'wave_excitation',
    'fd_to_td', 'td_to_fd', 'read_netcdf', 'write_netcdf',
    'vec_to_dofmat', 'dofmat_to_vec', 'run_bem',
    '_add_zerofreq_to_xr',
]

import logging
# import copy
from typing import Iterable, Callable, Any, Optional, Mapping, TypeVar
from pathlib import Path
# from numpy import isin

import numpy.typing as npt
import autograd.numpy as np
from autograd.builtins import isinstance, tuple, list, dict
from autograd import grad, jacobian
# from pyparsing import null_debug_action
import xarray as xr
import capytaine as cpy
# from scipy.optimize import minimize, OptimizeResult, Bounds
from scipy.linalg import block_diag
# import matplotlib.pyplot as plt
# from matplotlib.figure import Figure


# logger
_log = logging.getLogger(__name__)

# default values
_default_parameters = {'rho': 1025.0, 'g': 9.81, 'depth': np.infty}

# type aliases
TWEC = TypeVar("TWEC", bound="WEC")
TStateFunction = Callable[
    [TWEC, np.ndarray, np.ndarray, xr.Dataset], np.ndarray]


class WEC:
    """
    """
    def __init__(self, f0, nfreq, ndof, forces, constraints, wave_direction
                 ) -> TWEC:
        """
        f0 in Hz
        wave_direction (list/array) in degrees
        forces: f(wec, x_wec, x_opt, waves) -> array ndof x ntimes
        """
        self._freq = frequency(f0, nfreq)
        self._time = time(f0, nfreq)
        self._time_mat = time_mat(f0, nfreq)
        self._derivative_mat = derivative_mat(f0, nfreq)
        self._ndof = ndof
        self.forces = forces
        self.constraints = constraints if (constraints is not None) else []
        self._wave_direction = degrees_to_radians(wave_direction)

    # TODO: non-hiden properties with setters/getters?
    # TODO: additional useful properties?
    # TODO: copies of outer functions with less arguments?

    @staticmethod
    def from_bem(bem_data: xr.Dataset, mass: Optional[np.ndarray] = None,
                 hydrostatic_stiffness: Optional[np.ndarray] = None,
                 friction: Optional[np.ndarray] = None,
                 f_add: Optional[Mapping[str, TStateFunction]] = None,
                 constraints: Optional[list[dict]] = None,
                 damping_tol: float = 1e-6,
                 ) -> TWEC:
        """
        """
        # TODO: Capytaine: excitation forces phases, flip? convention?
        ndof = len(bem_data["influenced_dof"])
        wave_direction = bem_data['wave_direction']

        # frequency array
        f0 = bem_data["omega"].values[1] / (2*np.pi)
        nfreq = len(bem_data["omega"]) - 1
        assert np.allclose(np.arange(0, f0*(nfreq+0.5), f0)*2*np.pi,
                           bem_data["omega"].values)

        # add mass, hydrostatic stiffness, and friction to bem_data
        bem_data = _add_to_bem(bem_data, mass, hydrostatic_stiffness, friction)

        # add zero frequency if not included
        bem_data = _add_zerofreq_to_bem(bem_data)

        # check real part of damping diagonal > 0
        bem_data = _check_damping_bem(bem_data, damping_tol)

        # forces in the dynamics equations
        linear_force_functions, _ = standard_forces(bem_data)
        f_add = f_add if (f_add is not None) else {}
        forces = linear_force_functions | f_add

        # constraints
        constraints = constraints if (constraints is not None) else []

        return WEC(f0, nfreq, ndof, forces, constraints, wave_direction)

    @staticmethod
    def from_bem_file(file, mass: np.ndarray,
                      hydrostatic_stiffness: np.ndarray,
                      friction: Optional[np.ndarray] = None,
                      f_add: Optional[Mapping[str, TStateFunction]] = None,
                      constraints: list[dict] = None,
                      ) -> TWEC:
        bem_data = read_netcdf(file)
        wec = WEC.from_bem(bem_data, mass, hydrostatic_stiffness, friction,
                           f_add, constraints)
        return wec

    @staticmethod
    def from_floating_body(fb: cpy.FloatingBody, mass: np.ndarray,
                 hydrostatic_stiffness: np.ndarray, f0: float, nfreq: int,
                 wave_direction: npt.ArrayLike = np.array([0.0,]),
                 friction: Optional[np.ndarray] = None,
                 f_add: Optional[Mapping[str, TStateFunction]] = None,
                 constraints: list[dict] = [],
                 rho: float = _default_parameters['rho'],
                 depth: float = _default_parameters['depth'],
                 g: float = _default_parameters['g'],
                 ) -> tuple[TWEC, xr.Dataset]:
        # TODO: _log.info saying that the bem_data is returned and should be saved for quicker initialization later
        # RUN BEM
        _log.info(f"Running Capytaine (BEM): {nfreq} frequencies x " +
                 f"{len(wave_direction)} wave directions.")
        freq = frequency(f0, nfreq)
        write_info = ['hydrostatics', 'mesh', 'wavelength', 'wavenumber']
        bem_data = run_bem(fb, freq, wave_direction,
                        rho=rho, g=g, depth=depth, write_info=write_info)
        wec = WEC.from_bem(bem_data, mass, hydrostatic_stiffness,
                           friction, f_add, constraints)
        return wec, bem_data

    @staticmethod
    def from_impedance(f0, nfreq, impedance, f_add, constraints):
        # TODO: finish implementing this!
        ndof = impedance.shape[0]
        transfer_mat = mimo_transfer_mat(impedance)

        def force_impedance(wec, x_wec, x_opt, waves):
            f_fd = vec_to_dofmat(np.dot(transfer_mat, x_wec))
            return np.dot(wec.time_mat, f_fd)

        forces =  force_impedance | f_add
        return WEC(f0, nfreq, ndof, forces, constraints)

    # TODO: solve


def ncomponents(nfreq : int) -> int:
    return 2*nfreq + 1


def frequency(f0: float, nfreq: int) -> np.ndarray:
    """Construct equally spaced frequency array.

    The array includes 0 and has length of `nfreq+1`.
    """
    return np.arange(0, nfreq+1)*f0


def time(f0: float, nfreq: int, nsubsteps: int = 1) -> np.ndarray:
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
    nsteps = nsubsteps * ncomponents(nfreq)
    return np.linspace(0, 1/f0, nsteps, endpoint=False)


def time_mat(f0: float, nfreq: int, nsubsteps: int = 1) -> np.ndarray:
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
    t = time(f0, nfreq, nsubsteps)
    omega = frequency(f0, nfreq) * 2*np.pi
    wt = np.outer(t, omega[1:])
    ncomp = ncomponents(nfreq)
    time_mat = np.empty((nsubsteps*ncomp, ncomp))
    time_mat[:, 0] = 1.0
    time_mat[:, 1::2] = np.cos(wt)
    time_mat[:, 2::2] = -np.sin(wt)
    return time_mat


def derivative_mat(f0: float, nfreq: int) -> np.ndarray:
    def block(n): return np.array([[0, -1], [1, 0]]) * n*f0 * 2*np.pi
    blocks = [block(n+1) for n in range(nfreq)]
    blocks = [0.0] + blocks
    return block_diag(*blocks)


def degrees_to_radians(degrees: float | npt.ArrayLike, sort: bool = True,
                       ) -> float | np.ndarray:
    """Convert degrees to radians in range (-π, π] and sort.
    """
    radians = np.asarray(np.remainder(np.deg2rad(degrees), 2*np.pi))
    radians[radians > np.pi] -= 2*np.pi
    if radians.size == 1:
        radians = radians.item()
    elif sort:
        radians = np.sort(radians)
    return radians


# for `WEC.from_bem()`
def _add_to_bem(bem_data, mass=None, hydrostatic_stiffness=None, friction=None):
    """Add mass, hydrostatic stiffness, and linear friction to BEM."""
    vars = {'mass': mass, 'friction': friction,
            'hydrostatic_stiffness': hydrostatic_stiffness}

    dims = ['radiating_dof', 'influenced_dof']

    for name, data in vars.items():
        org = name in bem_data.variables.keys()
        new = data is not None
        if new and org:
            if not np.allclose(data, bem_data.variables[name]):
                raise ValueError(
                    f'BEM data already has variable "{name}" ' +
                    'with diferent values')
            else :
                _log.warning(
                    f'Variable "{name}" is already in BEM data ' +
                    'with same value.')
        elif (not new) and (not org):
            if name=='friction':
                ndof = len(bem_data["influenced_dof"])
                bem_data[name] = (dims, np.zeros([ndof, ndof]))
            raise ValueError(
                f'Variable "{name}" is not in BEM data and was not provided.')
        elif new:
            bem_data[name] = (dims, data)

    return bem_data


def _add_zerofreq_to_xr(data):
    if not np.isclose(data.coords['omega'][0].values, 0):
        tmp = data.isel(omega=0).copy(deep=True)
        tmp['omega'] = tmp['omega'] * 0
        vars = [var for var in list(data.keys()) if 'omega' in data[var].dims]
        print(vars)
        for var in vars:
            tmp[var] = tmp[var] * 0
        data = xr.concat([tmp, data], dim='omega', data_vars='minimal')
    return data


def _add_zerofreq_to_bem(bem_data):
    if not np.isclose(bem_data.coords['omega'][0].values, 0):
        bem_data = _add_zerofreq_to_xr(bem_data)
        bem_data['added_mass'][dict(omega=0)] *= np.NaN
        bem_data['Froude_Krylov_force'][dict(omega=0)] = \
            bem_data['hydrostatic_stiffness']
        # TODO: zero-frequency radiation damping and diffraction both zero?
    return bem_data


def _check_damping_bem(bem_data, tol=1e-6) -> xr.Dataset:
    radiation = bem_data['radiation_damping']
    friction = bem_data['friction']
    ndof = bem_data.shape[0]
    assert ndof == bem_data.shape[1]
    for idof in range(ndof):
        iradiation = radiation.isel(radiating_dof=idof, influenced_dof=idof)
        ifriction = friction.isel(radiating_dof=idof, influenced_dof=idof)
        dmin = np.diagonal(iradiation+ifriction, axis1=1, axis2=2).min()
        if dmin <= 0.0 + tol:
            dof = bem_data.influenced_dof.values[idof]
            _log.warning(
                f'Linear damping for DOF "{dof}" has negative or close to ' +
                'zero terms. Shifting up via linear friction.')
            ifriction[:] = ifriction[:] + (tol-dmin)
    return bem_data


def standard_forces(bem_data: xr.Dataset):
    """
    """
    bem_data = bem_data.transpose(
        "omega", "radiating_dof", "influenced_dof", "wave_direction")

    # intrinsic impedance
    w = bem_data['omega']
    A = bem_data['added_mass']
    B = bem_data['radiation_damping']
    K = bem_data['hydrostatic_stiffness']
    m = bem_data['mass']
    Bf = bem_data['friction']

    position_transfer_functions = dict()
    position_transfer_functions['inertia'] = -1*w**2*m + 0j
    position_transfer_functions['radiation'] = 1j*w*B + -1*w**2*A
    position_transfer_functions['hydrostatics'] = (
        (K + 0j).expand_dims({"omega": B.omega}) )
    position_transfer_functions['friction'] = 1j*w*Bf

    def f_from_imp(transfer_mat):
        def f(wec, x_wec, x_opt, waves):
            f_fd = wec.vec_to_dofmat(np.dot(transfer_mat, x_wec))
            return np.dot(wec.time_mat, f_fd)
        return f

    linear_force_functions = dict()
    ndof = len(bem_data.influenced_dof)
    for name, value in position_transfer_functions.items():
        linear_force_functions[name] = (
            f_from_imp(mimo_transfer_mat(value, ndof)) )

    # wave excitation
    def f_exc_fk(wec, x_wec, x_opt, waves):
        return wave_excitation(bem_data['Froude_Krylov_force'], waves)

    def f_exc_diff(wec, x_wec, x_opt, waves):
        return wave_excitation(bem_data['diffraction'], waves)

    linear_force_functions['Froude_Krylov'] = f_exc_fk
    linear_force_functions['diffraction'] = f_exc_diff

    return linear_force_functions


def mimo_transfer_mat(imp: np.ndarray, ndof:int) -> np.ndarray:
    """Create a block matrix of the MIMO transfer function."""
    # TODO: ndof from imp? veify square mat x nfreq?
    elem = [[None]*ndof for _ in range(ndof)]
    def block(re, im): return np.array([[re, im], [-im, re]])
    for idof in range(ndof):
        for jdof in range(ndof):
            Zp0 = imp[0, idof, jdof]
            assert np.all(np.isreal(Zp0))
            Zp0 = np.real(Zp0)
            Zp = imp[1:, idof, jdof]
            re = np.real(Zp)
            im = np.imag(Zp)
            blocks = [block(ire, iim) for (ire, iim) in zip(re, im)]
            blocks =[Zp0] + blocks
            elem[idof][jdof] = block_diag(*blocks)
    return np.block(elem)


def wave_excitation(exc_coeff, waves):
    dw = waves.omega[1] - waves.omega[0]
    wave_elev_fd = \
        np.sqrt(2*waves['S'] / (2*np.pi) * dw) * np.exp(1j*waves['phase'])
    wave_elev_fd = wave_elev_fd.transpose('omega', 'wave_direction')
    f_exc_fd = xr.dot(exc_coeff, wave_elev_fd, dims=["wave_direction"])  # TODO: Deal with NaN's in the zero-frequency components
    f_exc_fd = f_exc_fd.transpose('omega', 'influenced_dof')
    nfd = 2 * len(waves['omega']) + 1
    return fd_to_td(f_exc_fd, nfd)


def fd_to_td(fd: np.ndarray, n: Optional[int] = None) -> np.ndarray:
    return np.fft.irfft(fd/2, n=n, axis=0, norm='forward')
    # TODO: check that this is correct
    # TODO: use time matrix instead?
    # TODO: allow xarrays?


# for `WEC.from_bem_file()`
def read_netcdf(fpath: str | Path) -> xr.Dataset:
    """Read a NetCDF file with possibly complex entries as an xarray
    DataSet.
    """
    with xr.open_dataset(fpath) as ds:
        ds.load()
    return cpy.io.xarray.merge_complex_values(ds)


# for `from_floating_body()`
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
        wave_dirs = np.atleast_1d(degrees_to_radians(wave_dirs))
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


def vec_to_dofmat(vec: np.ndarray, ndof: int) -> np.ndarray:
        """Convert a vector back to a matrix with one column per DOF.
        Opposite of ``dofmat_to_vec``. """
        return np.reshape(vec, (-1, ndof), order='F')


# Not needed, but for "symmetry"
def td_to_fd(td: np.ndarray, n: Optional[int] = None) -> np.ndarray:
    return np.fft.rfft(td*2, n=n, axis=0, norm='forward')
    # TODO: check that this is correct. composition of these == identity
    # TODO: allow xarrays?


def write_netcdf(fpath: str | Path, data: xr.Dataset) -> None:
    """Save an xarray dataSet with possibly complex entries as a NetCDF
    file.
    """
    cpy.io.xarray.separate_complex_values(data).to_netcdf(fpath)


def dofmat_to_vec(mat: np.ndarray) -> np.ndarray:
        """Flatten a matrix that has one column per DOF.
        Opposite of ``vec_to_dofmat``. """
        return np.reshape(mat, -1, order='F')