"""Provide core functionality for solving the pseudo-spectral problem
for wave energy converters (WEC).
"""


from __future__ import annotations

__all__ = [
    'WEC', 'TWEC', 'TStateFunction',
    'frequency', 'time', 'time_mat', 'derivative_mat', 'degrees_to_radians',
    'ncomponents', 'standard_forces', 'mimo_transfer_mat', 'wave_excitation',
    'fd_to_td', 'td_to_fd', 'read_netcdf', 'write_netcdf',
    'vec_to_dofmat', 'dofmat_to_vec', 'run_bem',
    'add_zerofreq_to_xr', 'complex_to_real', 'real_to_complex', 'wave_elevation',
    'linear_hydrodynamics', 'check_linear_damping', 'inertia',
]   # TODO: clean exports

# TODO: clean imports
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
from scipy.optimize import minimize, OptimizeResult, Bounds
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

# TODO: Docstrings
# TODO: Type hints
class WEC:
    """
    """
    def __init__(self, f1, nfreq, forces, constraints=None,
                 mass: Optional[np.ndarray] = None,
                 ndof: Optional[int] = None,
                 inertia_in_forces: bool = False) -> TWEC:
        """
        """
        self._freq = frequency(f1, nfreq)
        self._time = time(f1, nfreq)
        self._time_mat = time_mat(f1, nfreq)
        self._derivative_mat = derivative_mat(f1, nfreq)
        self.forces = forces
        self.inertia_in_forces = inertia_in_forces
        self.constraints = constraints if (constraints is not None) else []

        # inertia options
        def _missing(var_name, condition):
            msg = (f"`{var_name}` must be provided if `inertia_in_forces` is" +
                    f"`{condition}`.")
            return msg

        def _ignored(var_name, condition):
            msg = (f"`{var_name}` is not used when `inertia_in_forces` is " +
                    f"`{condition}` and should not be provided")
            return msg

        if inertia_in_forces:
            condition = "True"
            if mass is not None:
                mass = None
                _log.warning(_ignored("mass", condition))
            if ndof is None:
                raise ValueError(_missing("ndof", condition))
        elif not inertia_in_forces:
            condition = "False"
            if mass is None:
                raise ValueError(_missing("mass", condition))
            mass = np.atleast_2d(np.squeeze(mass))
            if ndof is not None:
                _log.warning(_ignored("ndof", condition))
                if ndof != mass.shape[0]:
                    _log.warning(
                        "Provided value of `ndof` does not match size of " +
                        f"`mass` matrix. Using `ndof={mass.shape[0]}`.")
            ndof = mass.shape[0]
        if mass.shape != (ndof, ndof):
            raise ValueError(
                "'mass' must be a square matrix of size equal to the number " +
                " of degrees of freedom.")

        self.mass = mass
        self._ndof = ndof
        self.inertia = None if inertia_in_forces else inertia(f1, nfreq, mass)

    def __repr__(self):
        f'{self.__class__.__name__}' #TODO - make this more rich

    # other initialization methods
    @staticmethod
    def from_bem(bem_data: xr.Dataset, mass: Optional[np.ndarray] = None,
                 hydrostatic_stiffness: Optional[np.ndarray] = None,
                 friction: Optional[np.ndarray] = None,
                 f_add: Optional[Mapping[str, TStateFunction]] = None,
                 constraints: Optional[list[dict]] = None,
                 min_damping: Optional[float] = 1e-6,
                 ) -> TWEC:
        """
        """
        # add mass, hydrostatic stiffness, and friction
        hydro_data = linear_hydrodynamics(
            bem_data, mass, hydrostatic_stiffness, friction)
        mass = hydro_data['mass'].values if mass is None else mass

        # add zero frequency if not included
        if not np.isclose(hydro_data.coords['omega'][0].values, 0):
            _log.warning(
                "Provided BEM data does not include the zero-frequency " +
                "components. Setting the zero-frequency components for all " +
                "coefficients (radiation and excitation) to zero.")
            hydro_data = add_zerofreq_to_xr(hydro_data)

        # frequency array
        f1 = hydro_data["omega"].values[1] / (2*np.pi)
        nfreq = len(hydro_data["omega"]) - 1
        w_check =  np.arange(0, f1*(nfreq+0.5), f1)*2*np.pi
        if not np.allclose(w_check, hydro_data["omega"].values):
            raise ValueError("Frequency array `omega` must be evenly spaced.")

        # check real part of damping diagonal > 0
        if min_damping is not None:
            hydro_data = check_linear_damping(hydro_data, min_damping)

        # forces in the dynamics equations
        linear_force_functions = standard_forces(hydro_data)
        f_add = f_add if (f_add is not None) else {}
        forces = linear_force_functions | f_add
        # constraints
        constraints = constraints if (constraints is not None) else []
        return WEC(f1, nfreq, forces, constraints, mass)

    @staticmethod
    def from_bem_file(file, mass: Optional[np.ndarray] = None,
                      hydrostatic_stiffness: Optional[np.ndarray] = None,
                      friction: Optional[np.ndarray] = None,
                      f_add: Optional[Mapping[str, TStateFunction]] = None,
                      constraints: list[dict] = None,
                      min_damping: Optional[float] = 1e-6,
                      ) -> TWEC:
        bem_data = read_netcdf(file)
        wec = WEC.from_bem(bem_data, mass, hydrostatic_stiffness, friction,
                           f_add, constraints, min_damping=min_damping)
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
                 min_damping: Optional[float] = 1e-6,
                 ) -> tuple[TWEC, xr.Dataset]:
        """Create a WEC object from a Capytaine FloatingBody
        TODO
        Parameters
        ----------
        fb
            Capytaine FloatingBody
        mass
            _description_
        hydrostatic_stiffness
            _description_
        f1
            Fundamental frequency [Hz]
        nfreq
            Number of frequencies (not including zero frequency), 
            i.e., freqs = [0, f1, 2*f1, ..., nfreq*f1]
        wave_directions
            _description_, by default np.array([0.0,])
        friction
            _description_, by default None
        f_add
            _description_, by default None
        constraints
            _description_, by default []
        rho
            _description_, by default 1025
        depth
            _description_, by default Inf
        g
            _description_, by default 9.81
        min_damping
            _description_, by default 1e-6

        Returns
        -------
        tuple[TWEC, xr.Dataset]
            _description_
        """
        
        # RUN BEM
        _log.info("This function, `WEC.from_floating_body`, returns the " +
                  "`hydro_data` xarray DataSet. If using the same " + 
                  "hydrodynamic bodies, `hydro_data` may be save to " +
                  "a .nc file using the `write_netcdf` method. " + 
                  "With this .nc file, you can avoid re-running the BEM " + 
                  "by using `WEC.from_bem_file`.")
        _log.info(f"Running Capytaine (BEM): {nfreq+1} frequencies x " +
                 f"{len(wave_directions)} wave directions.")
        freq = frequency(f1, nfreq)[1:]
        bem_data = run_bem(fb, freq, wave_directions, rho=rho, g=g, depth=depth)
        wec = WEC.from_bem(bem_data, mass, hydrostatic_stiffness, friction,
                           f_add, constraints, min_damping=min_damping)
        hydro_data = linear_hydrodynamics(
            bem_data, mass, hydrostatic_stiffness, friction)
        hydro_data = add_zerofreq_to_xr(hydro_data)
        return wec, hydro_data

    @staticmethod
    def from_impedance(f1, nfreq, impedance, exc_coeff, f_add=None,
                       constraints=None):

        # impedance matrix shape
        shape = impedance.shape
        if (impedance.ndim!=3) or (shape[0]!=shape[1]) or (shape[2]!=nfreq+1):
            raise ValueError(
                "`impedance` must have shape `ndof x ndof x (nfreq+1)`, " +
                "including the zero-frequency component.")

        # impedance force
        position_transfer = impedance / (1j*impedance.omega)
        force_impedance = force_from_position_transfer_fun(position_transfer)

        # excitation force
        force_excitation = force_from_waves(exc_coeff)

        # all forces
        f_add = {} if (f_add is None) else f_add
        forces =  force_impedance | force_excitation | f_add

        # wec
        wec = WEC(f1, nfreq, forces, constraints,
                  inertia_in_forces=True, ndof=impedance.shape[0])
        return wec

    # solve
    def solve(self, waves, obj_fun,
              nstate_opt: int,
              x_wec_0: Optional[np.ndarray] = None,
              x_opt_0: Optional[np.ndarray] = None,
              scale_x_wec: Optional[list] = None,
              scale_x_opt: npt.ArrayLike | float = 1.0,
              scale_obj: float = 1.0,
              optim_options: dict[str, Any] = {},
              use_grad: bool = True,
              maximize: bool = False,
              bounds_wec: Optional[Bounds] = None,
              bounds_opt: Optional[Bounds] = None,
              callback: Optional[Callable[[np.ndarray]]] = None,
              ):
        _log.info("Solving pseudo-spectral control problem.")

        # scale x_wec
        if scale_x_wec == None:
            scale_x_wec = [1.0] * self.ndof
        elif isinstance(scale_x_wec, float) or isinstance(scale_x_wec, int):
            scale_x_wec = [scale_x_wec] * self.ndof
        scale_x_wec = scale_dofs(scale_x_wec, self.ncomponents)

        # scale x_opt
        if isinstance(scale_x_opt, float) or isinstance(scale_x_opt, int):
            if nstate_opt is None:
                raise ValueError("If 'scale_x_opt' is a scalar, " +
                                    "'nstate_opt' must be provided")
            scale_x_opt = scale_dofs([scale_x_opt], nstate_opt)

        # scale
        scale = np.concatenate([scale_x_wec, scale_x_opt])

        # initial guess
        if x_wec_0 is None:
            x_wec_0 = np.random.randn(self.nstate_wec)
        if x_opt_0 is None:
            x_opt_0 = np.random.randn(nstate_opt)
        x0 = np.concatenate([x_wec_0, x_opt_0])*scale

        # objective function
        sign = -1.0 if maximize else 1.0

        def obj_fun_scaled(x):
            x_wec, x_opt = self.decompose_decision_var(x/scale)
            return obj_fun(self, x_wec, x_opt, waves)*scale_obj*sign

        # constraints
        constraints = self.constraints.copy()

        for i, icons in enumerate(self.constraints):
            icons_new = {"type": icons["type"]}

            def make_new_fun(icons):
                def new_fun(x):
                    x_wec, x_opt = self.decompose_decision_var(x/scale)
                    return icons["fun"](self, x_wec, x_opt, waves)
                return new_fun

            icons_new["fun"] = make_new_fun(icons)
            if use_grad:
                icons_new['jac'] = jacobian(icons_new['fun'])
            constraints[i] = icons_new

        # system dynamics through equality constraint, ma - Σf = 0
        def resid_fun(x):
            x_s = x/scale
            x_wec, x_opt = self.decompose_decision_var(x_s)
            # inertia, ma
            if not self.inertia_in_forces:
                ri = self.inertia(self, x_wec, x_opt, waves)
            else:
                ri = np.zeros([self.ncomponents, self.ndof])
            # forces, -Σf
            for f in self.forces.values():
                ri = ri - f(self, x_wec, x_opt, waves)
            return self.dofmat_to_vec(ri)

        eq_cons = {'type': 'eq', 'fun': resid_fun}
        if use_grad:
            eq_cons['jac'] = jacobian(resid_fun)
        constraints.append(eq_cons)

        # bounds
        if (bounds_wec is None) and (bounds_opt is None):
            bounds = None
        else:
            # TODO: allow for all options of Bounds.
            bounds_in = [bounds_wec, bounds_opt]
            inf_wec = np.ones(self.nstate_wec)*np.inf
            inf_opt = np.ones(nstate_opt)*np.inf
            bounds_dflt = [Bounds(lb=-inf_wec, ub=inf_wec),
                            Bounds(lb=-inf_opt, ub=inf_opt)]
            bounds_list = []
            for bi, bd in zip(bounds_in, bounds_dflt):
                if bi is not None:
                    bo = bi
                else:
                    bo = bd
                bounds_list.append(bo)
            bounds = Bounds(lb=np.hstack([le.lb for le in bounds_list])*scale,
                            ub=np.hstack([le.ub for le in bounds_list])*scale)


        # optimization problem
        optim_options['disp'] = optim_options.get('disp', True)
        problem = {'fun': obj_fun_scaled,
                    'x0': x0,
                    'method': 'SLSQP',
                    'constraints': constraints,
                    'options': optim_options,
                    'bounds': bounds,
                    'callback':callback,
                    }
        if use_grad:
            problem['jac'] = grad(obj_fun_scaled)


        # minimize
        res = minimize(**problem)

        msg = f'{res.message}    (Exit mode {res.status})'
        if res.status == 0:
            _log.info(msg)
        elif res.status == 9:
            _log.warning(msg)
        else:
            raise Exception(msg)

        # unscale
        res.x = res.x / scale
        res.fun = res.fun / scale_obj

        # post-process
        # TODO

        return res # TODO

    # properties
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
    def dt(self):
        """Time spacing."""
        return self._time[1]

    @property
    def tf(self):
        """Final time (repeat period). Not included in time vector. """
        return 1/self.f1

    @property
    def ncomponents(self):
        return ncomponents(self.nfreq)

    @property
    def nstate_wec(self):
        """Length of the  WEC dynamics state vector."""
        return self.ndof * self.ncomponents

    # other methods
    def decompose_decision_var(self, state: np.ndarray
                              ) -> tuple[np.ndarray, np.ndarray]:
        return decompose_decision_var(state, self.ndof, self.nfreq)

    def time_nsubsteps(self, nsubsteps: int):
        return time(self.f1, self.nfreq, nsubsteps)

    def time_mat_nsubsteps(self, nsubsteps: int):
        return time_mat(self.f1, self.nfreq, nsubsteps)

    def vec_to_dofmat(self, vec: np.ndarray):
        return vec_to_dofmat(vec, self.ndof)

    def dofmat_to_vec(self, mat: np.ndarray):
        return dofmat_to_vec(mat)

    def fd_to_td(self, fd: np.ndarray):
        return fd_to_td(fd, self.f1, self.nfreq)

    def td_to_fd(self, td: np.ndarray):
        return td_to_fd(td)


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


def check_linear_damping(hydro_data, min_damping=1e-6) -> xr.Dataset:
    hydro_data_new = hydro_data.copy(deep=True)
    radiation = hydro_data_new['radiation_damping']
    friction = hydro_data_new['friction']
    ndof = len(hydro_data_new.influenced_dof)
    assert ndof == len(hydro_data.radiating_dof)
    for idof in range(ndof):
        iradiation = radiation.isel(radiating_dof=idof, influenced_dof=idof)
        ifriction = friction.isel(radiating_dof=idof, influenced_dof=idof)
        dmin = (iradiation+ifriction).min()
        if dmin <= 0.0 + min_damping:
            dof = hydro_data_new.influenced_dof.values[idof]
            delta = min_damping-dmin
            _log.warning(
                f'Linear damping for DOF "{dof}" has negative or close to ' +
                'zero terms. Shifting up via linear friction of ' +
                f'{delta.values} N/(m/s).')
            hydro_data_new['friction'][idof, idof] = (ifriction + delta)
    return hydro_data_new


def force_from_position_transfer_fun(position_transfer):
    def force(wec, x_wec, x_opt, waves):
        transfer_mat = mimo_transfer_mat(position_transfer)
        force_fd = wec.vec_to_dofmat(np.dot(transfer_mat, x_wec))
        return np.dot(wec.time_mat, force_fd)
    return force


def force_from_impedance(omega, impedance):
    return force_from_position_transfer_fun(impedance/(1j*omega))


def force_from_waves(force_coeff):
    def force(wec, x_wec, x_opt, waves):
        force_fd = complex_to_real(wave_excitation(force_coeff, waves))
        return np.dot(wec.time_mat, force_fd)
    return force


def inertia(f1, nfreq, mass):
    omega = np.reshape(frequency(f1, nfreq)*2*np.pi, [1,1,-1])
    mass = np.expand_dims(mass, -1)
    position_transfer_function = -1*omega**2*mass + 0j
    inertia_fun = force_from_position_transfer_fun(position_transfer_function)
    return inertia_fun


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
    Bf = hydro_data['friction']

    position_transfer_functions = dict()
    position_transfer_functions['radiation'] = 1j*w*B + -1*w**2*A
    position_transfer_functions['hydrostatics'] = (
        (K + 0j).expand_dims({"omega": B.omega}) )
    position_transfer_functions['friction'] = 1j*w*Bf

    linear_force_functions = dict()
    for name, value in position_transfer_functions.items():
        value = value.transpose("radiating_dof", "influenced_dof", "omega")
        value = -1*value  # RHS of equation: ma = Σf
        linear_force_functions[name] = (
            force_from_position_transfer_fun(value))

    # wave excitation
    excitation_coefficients = {
        'Froude_Krylov': hydro_data['Froude_Krylov_force'],
        'diffraction': hydro_data['diffraction_force']
    }

    for name, value in excitation_coefficients.items():
        linear_force_functions[name] = force_from_waves(value)

    return linear_force_functions


def add_zerofreq_to_xr(data):
    """frequency variable must be called `omega`."""
    if not np.isclose(data.coords['omega'][0].values, 0):
        tmp = data.isel(omega=0).copy(deep=True)
        tmp['omega'] = tmp['omega'] * 0
        vars = [var for var in list(data.keys()) if 'omega' in data[var].dims]
        for var in vars:
            tmp[var] = tmp[var] * 0
        data = xr.concat([tmp, data], dim='omega', data_vars='minimal')
    return data


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
    write_info = {
                #  'hydrostatics': True, 
                #   'mesh': True, 
                #   'wavelength': True,
                #   'wavenumber': True,
                  }
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
            else:
                raise ValueError(f'Variable "{name}" is not in BEM data and ' +
                                 'was not provided.')
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


def scale_dofs(scale_list: list[float], ncomponents: int) -> np.ndarray:
    """Create a scaling vector based on a different scale for each DOF.

    Parameters
    ----------
    scale_list: list
        Scale for each DOF.
    ncomponents: int
        Number of elements in the state vector for each DOF.

    Returns
    -------
    np.ndarray: Scaling vector.
    """
    ndof = len(scale_list)
    scale = []
    for dof in range(ndof):
        scale += [scale_list[dof]] * ncomponents
    return np.array(scale)


def decompose_decision_var(state: np.ndarray, ndof, nfreq
                               ) -> tuple[np.ndarray, np.ndarray]:
        """Split the state vector into the WEC dynamics state and the
        optimization (control) state. x = [x_wec, x_opt].
        """
        nstate_wec = ndof * ncomponents(nfreq)
        return state[:nstate_wec], state[nstate_wec:]
