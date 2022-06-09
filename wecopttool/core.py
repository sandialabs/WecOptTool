"""Provide core functionality for solving the pseudo-spectral problem
for wave energy converters.
"""


from __future__ import annotations
from argparse import ArgumentError  # TODO: for Python 3.8 & 3.9 support


__all__ = [
    'WEC', 'TWEC', 'TStateFunction',
    'frequency', 'time', 'time_mat', 'derivative_mat', 'degrees_to_radians',
    'ncomponents', 'standard_forces', 'mimo_transfer_mat', 'wave_excitation',
    'fd_to_td', 'td_to_fd', 'read_netcdf', 'write_netcdf',
    'vec_to_dofmat', 'dofmat_to_vec', 'run_bem',
    'add_zerofreq_to_xr', 'complex_to_real', 'real_to_complex', 'wave_elevation',
    'linear_hydrodynamics', 'check_linear_damping',
]   # TODO

import logging
# import copy  # TODO
from typing import Iterable, Callable, Any, Optional, Mapping, TypeVar
from pathlib import Path
# from numpy import isin  # TODO

import numpy.typing as npt
import autograd.numpy as np
from autograd.builtins import isinstance, tuple, list, dict
from autograd import grad, jacobian
# from pyparsing import null_debug_action  # TODO
import xarray as xr
import capytaine as cpy
# from scipy.optimize import minimize, OptimizeResult, Bounds  # TODO
from scipy.linalg import block_diag
# import matplotlib.pyplot as plt  # TODO
# from matplotlib.figure import Figure  # TODO


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
    def __init__(self, f1, nfreq, ndof, forces, constraints=None) -> TWEC:
        """
        """
        self._freq = frequency(f1, nfreq)
        self._time = time(f1, nfreq)
        self._time_mat = time_mat(f1, nfreq)
        self._derivative_mat = derivative_mat(f1, nfreq)
        self._ndof = ndof
        self.forces = forces
        self.constraints = constraints if (constraints is not None) else []

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
        # add mass, hydrostatic stiffness, and friction
        hydro_data = linear_hydrodynamics(
            bem_data, mass, hydrostatic_stiffness, friction)

        # TODO: option to not correct damping
        # TODO: Capytaine: excitation forces phases, flip? convention?
        ndof = len(hydro_data["influenced_dof"])
        wave_directions = hydro_data['wave_direction']

        # frequency array
        f1 = hydro_data["omega"].values[1] / (2*np.pi)
        nfreq = len(hydro_data["omega"]) - 1
        assert np.allclose(np.arange(0, f1*(nfreq+0.5), f1)*2*np.pi,
                           hydro_data["omega"].values) # TODO raise error w message

        # add zero frequency if not included
        if not np.isclose(hydro_data.coords['omega'][0].values, 0):
            _log.warning(
                "Provided BEM data does not include the zero-frequency " +
                "components. Setting the zero-frequency comoponents for all " +
                "coefficients (radiation and excitation) to zero.")
            hydro_data = add_zerofreq_to_xr(hydro_data)

        # check real part of damping diagonal > 0
        hydro_data = check_linear_damping(hydro_data, damping_tol)

        # forces in the dynamics equations
        linear_force_functions, _ = standard_forces(hydro_data)
        f_add = f_add if (f_add is not None) else {}
        forces = linear_force_functions | f_add

        # constraints
        constraints = constraints if (constraints is not None) else []

        return WEC(f1, nfreq, ndof, forces, constraints, wave_directions)

    @staticmethod
    def from_bem_file(file, mass: Optional[np.ndarray] = None,
                      hydrostatic_stiffness: Optional[np.ndarray] = None,
                      friction: Optional[np.ndarray] = None,
                      f_add: Optional[Mapping[str, TStateFunction]] = None,
                      constraints: list[dict] = None,
                      ) -> TWEC:
        # TODO: option to not correct damping
        bem_data = read_netcdf(file)
        wec = WEC.from_bem(bem_data, mass, hydrostatic_stiffness, friction,
                           f_add, constraints)
        return wec

    @staticmethod
    def from_floating_body(fb: cpy.FloatingBody, mass: np.ndarray,
                 hydrostatic_stiffness: np.ndarray, f1: float, nfreq: int,
                 wave_directions: npt.ArrayLike = np.array([0.0,]),
                 friction: Optional[np.ndarray] = None,
                 f_add: Optional[Mapping[str, TStateFunction]] = None,
                 constraints: list[dict] = [],
                 rho: float = _default_parameters['rho'],
                 depth: float = _default_parameters['depth'],
                 g: float = _default_parameters['g'],
                 ) -> tuple[TWEC, xr.Dataset]:
        # TODO: option to not correct damping
        # TODO: _log.info saying that the hydro_data is returned and should be saved for quicker initialization later
        # RUN BEM
        _log.info(f"Running Capytaine (BEM): {nfreq+1} frequencies x " +
                 f"{len(wave_directions)} wave directions.")  # TODO: run the zero-frequency?
        freq = frequency(f1, nfreq)
        bem_data = run_bem(
            fb, freq, wave_directions, rho=rho, g=g, depth=depth)
        wec = WEC.from_bem(bem_data, mass, hydrostatic_stiffness,
                           friction, f_add, constraints)
        hydro_data = linear_hydrodynamics(
            bem_data, mass, hydrostatic_stiffness, friction)
        return wec, hydro_data

    @staticmethod
    def from_impedance(f1, nfreq, impedance, f_add, constraints):
        # TODO: finish implementing this!
        ndof = impedance.shape[0]
        transfer_mat = mimo_transfer_mat(impedance)

        def force_impedance(wec, x_wec, x_opt, waves):
            f_fd = vec_to_dofmat(np.dot(transfer_mat, x_wec))
            return np.dot(wec.time_mat, f_fd)

        forces =  force_impedance | f_add
        return WEC(f1, nfreq, ndof, forces, constraints)

    # TODO: solve


    # public properties, for convenience
    @property
    def ndof(self):
        return self._ndof

    @property
    def frequency(self):
        return self._freq

    @property
    def f1(self):
        return self._freq[1]

    @property
    def nfreq(self):
        return len(self._freq)-1

    @property
    def omega(self):
        return self._freq / (2*np.pi)

    @property
    def w1(self):
        return self.omega[1]

    @property
    def time(self):
        return self._time

    @property
    def time_mat(self):
        return self._time_mat

    @property
    def derivative_mat(self):
        return self._derivative_mat

    @property
    def ncomponents(self):
        return ncomponents(self.nfreq)

    # copies of outer functions with less arguments, for convinience
    def time_nsubsteps(self, nsubsteps: int):
        return time(self.f1, self.nfreq, nsubsteps)

    def time_mat_nsubsteps(self, nsubsteps: int):
        return time_mat(self.f1, self.nfreq, nsubsteps)

    def vec_to_dofmat(self, vec: np.ndarray):
        return vec_to_dofmat(vec, self.ndof)

    def dofmat_to_vec(self, mat: np.ndarray):
        return dofmat_to_vec(mat)

    # TODO: td_to_fd & fd_to_td


def ncomponents(nfreq : int) -> int:
    return 2*nfreq + 1


def frequency(f1: float, nfreq: int) -> np.ndarray:
    """Construct equally spaced frequency array.

    The array includes 0 and has length of `nfreq+1`.
    f1 is fundamental frequency (1st harmonic).
    f = [0, f1, 2*f1, ..., nfreq*f1]
    """
    return np.arange(0, nfreq+1)*f1


def time(f1: float, nfreq: int, nsubsteps: int = 1) -> np.ndarray:
    """Assemble the time vector with n subdivisions.

    Parameters
    ----------
    nsubsteps: int
        Number of steps between the default (implied) time steps.
        A value of `1` corresponds to the default step length.
        dt = dt_default * 1/nsubsteps.

    Returns
    -------
    time_vec: np.ndarray
    """
    nsteps = nsubsteps * ncomponents(nfreq)
    return np.linspace(0, 1/f1, nsteps, endpoint=False)


def time_mat(f1: float, nfreq: int, nsubsteps: int = 1) -> np.ndarray:
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
    t = time(f1, nfreq, nsubsteps)
    omega = frequency(f1, nfreq) * 2*np.pi
    wt = np.outer(t, omega[1:])
    ncomp = ncomponents(nfreq)
    time_mat = np.empty((nsubsteps*ncomp, ncomp))
    time_mat[:, 0] = 1.0
    time_mat[:, 1::2] = np.cos(wt)
    time_mat[:, 2::2] = -np.sin(wt)
    return time_mat


def derivative_mat(f1: float, nfreq: int) -> np.ndarray:
    def block(n): return np.array([[0, -1], [1, 0]]) * n*f1 * 2*np.pi
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


def vec_to_dofmat(vec: np.ndarray, ndof: int) -> np.ndarray:
        """Convert a vector back to a matrix with one column per DOF.
        Opposite of ``dofmat_to_vec``. """
        return np.reshape(vec, (-1, ndof), order='F')


def dofmat_to_vec(mat: np.ndarray) -> np.ndarray:
        """Flatten a matrix that has one column per DOF.
        Opposite of ``vec_to_dofmat``. """
        return np.reshape(mat, -1, order='F')


def mimo_transfer_mat(imp: np.ndarray) -> np.ndarray:
    """Create a block matrix of the MIMO transfer function.

    imp: ndof x ndof x nfreq+1
    """
    ndof = imp.shape[0]
    assert imp.shape[1] == ndof
    elem = [[None]*ndof for _ in range(ndof)]
    def block(re, im): return np.array([[re, -im], [im, re]])
    for idof in range(ndof):
        for jdof in range(ndof):
            Zp0 = imp[idof, jdof, 0]
            assert np.all(np.isreal(Zp0))
            Zp0 = np.real(Zp0)
            Zp = imp[idof, jdof, 1:]
            re = np.real(Zp)
            im = np.imag(Zp)
            blocks = [block(ire, iim) for (ire, iim) in zip(re, im)]
            blocks =[Zp0] + blocks
            elem[idof][jdof] = block_diag(*blocks)
    return np.block(elem)


def real_to_complex(fd: np.ndarray) -> np.ndarray:
    """Convert from two real amplitudes to one complex amplitude per
    frequency. """
    fd= atleast_2d(fd)
    assert np.all(np.isreal(fd[0, :]))
    assert fd.shape[0]%2==1
    mean = fd[0:1, :]
    fd = fd[1:, :]
    return np.concatenate((mean, fd[0::2, :] + 1j*fd[1::2, :]), axis=0)


def complex_to_real(fd: np.ndarray) -> np.ndarray:
    """Convert from one complex amplitude to two real amplitudes per
    frequency."""
    fd= atleast_2d(fd)
    nfreq = fd.shape[0] - 1
    ndof = fd.shape[1]
    out = np.zeros((1+2*nfreq, ndof))
    assert np.all(np.isreal(fd[0, :]))
    out[0, :] = fd[0, :].real
    out[1::2, :] = fd[1:].real
    out[2::2, :] = fd[1:].imag
    return out


def fd_to_td(fd: np.ndarray, f1=None, nfreq=None) -> np.ndarray:
    fd= atleast_2d(fd)
    if (f1 is not None) and (nfreq is not None):
        tmat = time_mat(f1, nfreq)
        td = tmat @ complex_to_real(fd)
    elif (f1 is None) and (nfreq is None):
        n = 1 + 2*(fd.shape[0]-1)
        td = np.fft.irfft(fd/2, n=n, axis=0, norm='forward')
    else:
        raise ValueError(
            "Provide either both `f1` and `nfreq` or neither.")
    return td


def td_to_fd(td: np.ndarray) -> np.ndarray:
    td= atleast_2d(td)
    n = td.shape[0]
    return np.fft.rfft(td*2, n=n, axis=0, norm='forward')


def wave_elevation(waves):
    """
    example time domain:
    >> elevation_fd = wave_elevation(waves)
    >> nfd = 2 * len(waves['omega']) + 1
    >> elevation_td = fd_to_td(elevation_fd, nfd)
    """
    if not waves.omega[0]==0.0:
        raise ValueError("first frequency must be 0.0")
    if not np.allclose(np.diff(waves.omega), np.diff(waves.omega)[0]):
        raise ValueError("Wave frequencies must be evenly spaced.")
    dw = waves.omega[1]
    fd = np.sqrt((2*waves['S'] / (2*np.pi)) * dw) * np.exp(1j*waves['phase'])
    return fd.values


def wave_excitation(exc_coeff, waves):
    """
    wave frequencies same as exc_coeff. Directions can be a subset.
    frequencies must be evenly spaced and start at 0.
    """
    omega_w = waves['omega'].values
    omega_e = exc_coeff['omega'].values
    dir_w = waves['wave_direction'].values
    dir_e = exc_coeff['wave_direction'].values
    exc_coeff = exc_coeff.transpose(
        'omega', 'wave_direction', 'influenced_dof').values
    wave_elev_fd = np.expand_dims(wave_elevation(waves), -1)

    if not np.allclose(omega_w, omega_e):
        raise ValueError("Wave and excitation frequencies do not match.")

    subset, sub_ind = subset_close(dir_w, dir_e)

    if not subset:
        raise ValueError(
            "Some wave directions are not in excitation coefficients " +
            f"\n Wave direction(s): {(np.rad2deg(dir_w))} (deg)" +
            f"\n BEM direction(s): {np.rad2deg(dir_e)} (deg).")

    return np.sum(wave_elev_fd*exc_coeff[:, sub_ind, :], axis=1)


def read_netcdf(fpath: str | Path) -> xr.Dataset:
    """Read a NetCDF file with possibly complex entries as an xarray
    DataSet.
    """
    with xr.open_dataset(fpath) as ds:
        ds.load()
    return cpy.io.xarray.merge_complex_values(ds)


def write_netcdf(fpath: str | Path, data: xr.Dataset) -> None:
    """Save an xarray dataSet with possibly complex entries as a NetCDF
    file.
    """
    cpy.io.xarray.separate_complex_values(data).to_netcdf(fpath)


def check_linear_damping(hydro_data, tol=1e-6) -> xr.Dataset:
    hydro_data_new = hydro_data.copy(deep=True)
    radiation = hydro_data_new['radiation_damping']
    friction = hydro_data_new['friction']
    ndof = len(hydro_data_new.influenced_dof)
    assert ndof == len(hydro_data.radiating_dof)
    for idof in range(ndof):
        iradiation = radiation.isel(radiating_dof=idof, influenced_dof=idof)
        ifriction = friction.isel(radiating_dof=idof, influenced_dof=idof)
        dmin = (iradiation+ifriction).min()
        if dmin <= 0.0 + tol:
            dof = hydro_data_new.influenced_dof.values[idof]
            _log.warning(
                f'Linear damping for DOF "{dof}" has negative or close to ' +
                'zero terms. Shifting up via linear friction.')
            hydro_data_new['friction'][idof, idof] = ifriction + (tol-dmin)
    return hydro_data_new


# no unit tests yet
def add_zerofreq_to_xr(data):
    """frequency variable must be called `omega`."""
    if not np.isclose(data.coords['omega'][0].values, 0):
        tmp = data.isel(omega=0).copy(deep=True)
        tmp['omega'] = tmp['omega'] * 0
        vars = [var for var in list(data.keys()) if 'omega' in data[var].dims]
        print(vars)
        for var in vars:
            tmp[var] = tmp[var] * 0
        data = xr.concat([tmp, data], dim='omega', data_vars='minimal')
    return data


def standard_forces(hydro_data: xr.Dataset):
    """
    """
    hydro_data = hydro_data.transpose(
         "omega", "wave_direction", "radiating_dof", "influenced_dof")

    # intrinsic impedance
    w = hydro_data['omega']
    A = hydro_data['added_mass']
    B = hydro_data['radiation_damping']
    K = hydro_data['hydrostatic_stiffness']
    m = hydro_data['mass']
    Bf = hydro_data['friction']

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
    for name, value in position_transfer_functions.items():
        value = value.transpose("radiating_dof", "influenced_dof", "omega")
        linear_force_functions[name] = (
            f_from_imp(mimo_transfer_mat(value)) )

    # wave excitation
    def f_from_waves(force_coeff):
        def f(wec, x_wec, x_opt, waves):
            f_fd = complex_to_real(wave_excitation(force_coeff, waves))
            return np.dot(wec.time_mat, f_fd)
        return f

    excitation_coefficients = {
        'Froude_Krylov': hydro_data['Froude_Krylov_force'],
        'diffraction': hydro_data['diffraction_force']
    }

    for name, value in excitation_coefficients.items():
        linear_force_functions[name] = f_from_waves(value)

    return linear_force_functions


def run_bem(fb: cpy.FloatingBody, freq: Iterable[float] = [np.infty],
            wave_dirs: Iterable[float] = [0],
            rho: float = _default_parameters['rho'],
            g: float = _default_parameters['g'],
            depth: float = _default_parameters['depth'],
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
    write_info = {'hydrostatics': True, 'mesh': True, 'wavelength': True,
                  'wavenumber': True}
    wec_im = fb.copy(name=f"{fb.name}_immersed").keep_immersed_part()
    bem_data = solver.fill_dataset(test_matrix, [wec_im], **write_info)
    return change_bem_convention(bem_data)


def change_bem_convention(bem_data):
    """Change the convention from `-iωt` to `+iωt`.
    """
    bem_data['Froude_Krylov_force'] = np.conjugate(
        bem_data['Froude_Krylov_force'])
    bem_data['diffraction_force'] = np.conjugate(bem_data['diffraction_force'])
    return bem_data


def linear_hydrodynamics(bem_data, mass=None, hydrostatic_stiffness=None, friction=None):
    """Add mass, hydrostatic stiffness, and linear friction to BEM.
    Complex conjugate of Capytaine excitation coefficients.
    """
    vars = {'mass': mass, 'friction': friction,
            'hydrostatic_stiffness': hydrostatic_stiffness}

    dims = ['radiating_dof', 'influenced_dof']

    hydro_data = bem_data.copy(deep=True)

    for name, data in vars.items():
        org = name in hydro_data.variables.keys()
        new = data is not None
        if new and org:
            if not np.allclose(data, hydro_data.variables[name]):
                raise ValueError(
                    f'BEM data already has variable "{name}" ' +
                    'with diferent values')
            else :
                _log.warning(
                    f'Variable "{name}" is already in BEM data ' +
                    'with same value.')
        elif (not new) and (not org):
            if name=='friction':
                ndof = len(hydro_data["influenced_dof"])
                hydro_data[name] = (dims, np.zeros([ndof, ndof]))
            raise ValueError(
                f'Variable "{name}" is not in BEM data and was not provided.')
        elif new:
            hydro_data[name] = (dims, data)

    return hydro_data


def atleast_2d(a):
    return np.expand_dims(a, -1) if len(a.shape)==1 else a


def subset_close(subset_a: float | npt.ArrayLike,
                set_b: float | npt.ArrayLike,
                rtol: float = 1.e-5, atol:float = 1.e-8,
                equal_nan: bool = False) -> tuple[bool, list]:
    """
    Compare if two arrays are subset equal within a tolerance.
    Parameters
    ----------
    subset_a: float | npt.ArrayLike
        First array which is tested for being subset.
    set_b: float | npt.ArrayLike
        Second array which is tested for containing subset_a.
    rtol: float
        The relative tolerance parameter.
    atol: float
        The absolute tolerance parameter.
    equal_nan: bool
        Whether to compare NaNs as equal.
    Returns
    -------
    subset: bool
        Whether the first array is a subset of second array
    ind: list
        List with integer indices where the first array's elements
        are located inside the second array.
    """
    assert len(np.unique(subset_a.round(decimals = 6))) == len(subset_a), "Elements in subset_a not unique"
    assert len(np.unique(set_b.round(decimals = 6))) == len(set_b), "Elements in set_b not unique"

    ind = []
    tmp_result = [False for i in range(len(subset_a))]
    for subset_element in subset_a:
        for set_element in set_b:
            if np.isclose(subset_element, set_element, rtol, atol, equal_nan):
                tmp_set_ind = np.where(
                    np.isclose(set_element, set_b , rtol, atol, equal_nan))
                tmp_subset_ind = np.where(
                    np.isclose(subset_element, subset_a , rtol, atol,
                               equal_nan))
                ind.append( int(tmp_set_ind[0]) )
                tmp_result[ int(tmp_subset_ind[0]) ] = True
    subset = all(tmp_result)
    return subset, ind