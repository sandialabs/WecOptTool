"""Provide core functionality for solving the pseudo-spectral problem.
"""


from __future__ import annotations
# import stat  # required for Python 3.8 & 3.9 support


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


# logger
_log = logging.getLogger(__name__)

# default values
_default_parameters = {'rho': 1025.0, 'g': 9.81, 'depth': np.infty}


class WEC:
    """Class representing a specific wave energy converter (WEC).
    An instance contains the  following information about the WEC,
    environment, and dynamic modeling:
    """
    # * Geometry
    # * Degrees of freedom
    # * Mass properties
    # * Hydrostatic properties
    # * Linear frequency domain hydrodynamic coefficients
    # * Water properties
    # * Additional dynamic forces (power take-off, mooring, nonlinear
    #   hydrodynamics, etc.)
    # * Constraints

    def __init__(self, f0, nfreq, ndof, forces, constraints, wave_directions):
        """
        f0 in Hz
        wave_directions (list/array) in degrees
        forces: f(wec, x_wec, x_opt, wave) -> array ndof x ntimes
        """
        self._freq = frequency(f0, nfreq)
        self._time = time(f0, nfreq)
        self._time_mat = time_mat(f0, nfreq)
        self._derivative_mat = derivative_mat(f0, nfreq)
        self._ndof = ndof
        self.forces = forces
        self.constraints = constraints if (constraints is not None) else []
        self._wave_directions = degrees_to_radians(wave_directions, sort=True)

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
        linear_force__functions, _ = _standard_forces(bem_data)
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
        # TODO: _log.info saying that the bem_data is returned and should be saved for quicker initialization later
        # RUN BEM
        _log.info(f"Running Capytaine (BEM): {nfreq} frequencies x " +
                 f"{len(wave_directions)} wave directions.")
        freq = frequency(f0, nfreq)
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
        transfer_mat = _mimo_transfer_mat(impedance)  # TODO

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
        return len(self.freq)-1

    @property
    def f0(self):
        return self.freq[1]

    @property
    def period(self):
        """Period :math:T=1/f in seconds."""
        return 1/self.freq

    @property
    def omega(self):
        """Angular frequency array in radians per second ω=2πf."""
        return self.freq * 2 * np.pi

    @property
    def w0(self):
        """Initial angular frequency (and spacing) in rad/s.
        See ``freq``.
        """
        return self.freq[1] * 2 * np.pi

    # properties: waves
    @property
    def wave_directions(self):
        """Wave directions in degrees."""
        return self._wave_directions * 180/np.pi

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


    def _get_state_scale(self,
                         scale_x_wec: Optional[list] = None,
                         scale_x_opt: npt.ArrayLike | float = 1.0,
                         nstate_opt: Optional[int] = None) -> np.ndarray:
        """Create a combined scaling array for the state vector. """
        # scale for x_wec
        if scale_x_wec == None:
            scale_x_wec = [1.0] * self.ndof
        elif isinstance(scale_x_wec, float) or isinstance(scale_x_wec, int):
            scale_x_wec = [scale_x_wec] * self.ndof
        scale_x_wec = scale_dofs(scale_x_wec, self.ncomponents)

        # scale for x_opt
        if isinstance(scale_x_opt, float) or isinstance(scale_x_opt, int):
            if nstate_opt is None:
                raise ValueError("If 'scale_x_opt' is a scalar, " +
                                 "'nstate_opt' must be provided")
            scale_x_opt = scale_dofs([scale_x_opt], nstate_opt)

        return np.concatenate([scale_x_wec, scale_x_opt])

    def _dynamic_residual(self, x: np.ndarray, waves) -> np.ndarray:
        """Solve WEC dynamics in residual form so that they may be
        enforced through a nonlinear constraint within an optimization
        problem.

        Parameters
        ----------
        x : np.ndarray
            Decision variable for optimization problem
        waves :
            TODO

        Returns
        -------
        np.ndarray
            Residuals at collocation points

        """
        x_wec, x_opt = self.decompose_decision_var(x)

        #TODO: remove key (just for debugging)
        force_residual = 0
        for key, force_func in self.forces.items():
            force_residual = force_residual + force_func(self, x_wec, x_opt, waves)

        #TODO - get signs right (f_i - f_exc - f_add)

        return force_residual


    def solve(self,
              waves: xr.Dataset,
              obj_fun: Callable[[WEC, np.ndarray, np.ndarray], float],
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
              unconstrained_first: Optional[bool] = False,
              callback: Callable[[np.ndarray]] = None,
              ) -> tuple[xr.Dataset, xr.Dataset, np.ndarray, np.ndarray, float,
                         OptimizeResult]:
        """Solve the WEC co-design problem.

        Parameters
        ----------
        waves: xr.Dataset
            The wave, described by two 2D DataArrays:
            elevation variance `S` (m^2*s) and phase `phase` (radians)
            with coordinates of radial frequency `omega` (radians)
            and wave direction `wave_direction` (radians).
            The frequencies and  wave directions must match those in
            the `bem_data` in `self.hydro`.
        obj_fun: function
            Objective function for the control optimization.
            Takes three inputs:
            (1) the WEC object,
            (2) the WEC dynamics state (1D np.ndarray), and
            (3) the optimization state (1D np.ndarray)
            and outputs the scalar objective function:
            tuple[WEC, np.ndarray, np.ndarray] -> float.
        nstate_opt: int
            Length of the optimization (controls) state vector.
        x_wec_0: np.ndarray
            Initial guess for the WEC dynamics state.
            If ``None`` it is randomly initiated.
        x_opt_0: np.ndarray
            Initial guess for the optimization (control) state.
            If ``None`` it is randomly initiated.
        scale_x_wec: list
            Factors to scale each DOF in ``x_wec`` by, to improve
            convergence. List length ``ndof``.
        scale_x_opt: npt.ArrayLike | float
            Factor to scale ``x_opt`` by, to improve convergence.
            A single float or an array of size ``nstate_opt``.
        scale_obj: float
            Factor to scale ``obj_fun`` by, to improve convergence.
        optim_options: dict
            Optimization options passed to the optimizer.
            See ``scipy.optimize.minimize``.
        use_grad: bool
            Whether to use gradient information in the optimization.
        maximize: bool
            Whether to maximize the objective function. The default is
            ``False`` to minimize the objective function.
        bounds_wec: Bounds
            Bounds on the WEC components of the decsision variable; see
            scipy.optimize.minimize
        bounds_opt: Bounds
            Bounds on the optimization (control) components of the decsision
            variable; see scipy.optimize.minimize
        unconstrained_first: bool
            If True, run ``solve`` without constraints to get scaling and
            initial guess. The default is False.
        callback: function
            Called after each iteration; see scipy.optimize.minimize. The
            default is reported via logging at the INFO level.

        Returns
        -------
        time_dom: xr.Dataset
            Dataset containing the time-domain results.
        freq_dom: xr.Dataset
            Dataset containing the frequency-domain results.
        x_wec: np.ndarray
            Optimal WEC state.
        x_opt: np.ndarray
            Optimal control state.
        objective: float
            optimized value of the objective function.
        res: optimize.optimize.OptimizeResult
            Raw optimization results.
        """
        _log.info("Solving pseudo-spectral control problem.")

        if x_wec_0 is None:
            x_wec_0 = np.random.randn(self.nstate_wec)
        if x_opt_0 is None:
            x_opt_0 = np.random.randn(nstate_opt)

        if unconstrained_first:
            _log.info(
                "Solving without constraints for better scaling and initial guess")
            wec1 = copy.deepcopy(self)
            wec1.constraints = []
            unconstrained_first = False
            _, _, x_wec_0, x_opt_0, obj, res = wec1.solve(waves,
                                                          obj_fun,
                                                          nstate_opt,
                                                          x_wec_0,
                                                          x_opt_0,
                                                          scale_x_wec,
                                                          scale_x_opt,
                                                          scale_obj,
                                                          optim_options,
                                                          use_grad,
                                                          maximize,
                                                          bounds_wec,
                                                          bounds_opt,
                                                          unconstrained_first,
                                                          )
            scale_x_wec = 1/np.max(np.abs(x_wec_0))
            scale_x_opt = 1/np.max(np.abs(x_opt_0))
            scale_obj = 1/np.abs(obj)
            _log.info(f"Setting x_wec_0: {x_wec_0}")
            _log.info(f"Setting x_opt_0: {x_opt_0}")
            _log.info(f"Setting scale_x_wec: {scale_x_wec}")
            _log.info(f"Setting scale_x_opt: {scale_x_opt}")
            _log.info(f"Setting scale_obj: {scale_obj}")

        # scale
        scale = self._get_state_scale(scale_x_wec, scale_x_opt, nstate_opt)

        # bounds
        bounds_in = [bounds_wec, bounds_opt]
        bounds_dflt = [Bounds(lb=-1*np.ones(self.nstate_wec)*np.inf,
                             ub=1*np.ones(self.nstate_wec)*np.inf),
                      Bounds(lb=-1*np.ones(nstate_opt)*np.inf,
                             ub=1*np.ones(nstate_opt)*np.inf)]
        bounds_list = []
        for bi, bd in zip(bounds_in, bounds_dflt):
            if bi is not None: bo = bi
            else: bo = bd
            bounds_list.append(bo)
        bounds = Bounds(lb=np.hstack([le.lb for le in bounds_list])*scale,
                        ub=np.hstack([le.ub for le in bounds_list])*scale)

        # initial guess
        x0 = np.concatenate([x_wec_0, x_opt_0])*scale

        # objective function
        sign = -1.0 if maximize else 1.0

        def obj_fun_scaled(x):
            x_wec, x_opt = self.decompose_decision_var(x/scale)
            return obj_fun(self, x_wec, x_opt)*scale_obj*sign

        # constraints
        constraints = self.constraints.copy()

        for i, icons in enumerate(self.constraints):
            icons_new = {"type": icons["type"]}

            def make_new_fun(icons):
                def new_fun(x):
                    x_wec, x_opt = self.decompose_decision_var(x/scale)
                    return icons["fun"](self, x_wec, x_opt)
                return new_fun

            icons_new["fun"] = make_new_fun(icons)
            if use_grad:
                icons_new['jac'] = jacobian(icons_new['fun'])
            constraints[i] = icons_new

        # system dynamics through equality constraint
        def resid_fun(x):
            ri = self._dynamic_residual(x=x/scale, waves=waves)
            return self.dofmat_to_vec(ri)

        eq_cons = {'type': 'eq',
                   'fun': resid_fun,
                   }
        if use_grad:
            eq_cons['jac'] = jacobian(resid_fun)
        constraints.append(eq_cons)

        optim_options['disp'] = optim_options.get('disp', True)

        if callback is None:
            def callback(x):
                x_wec, x_opt = self.decompose_decision_var(x)
                _log.info("[max(x_wec), max(x_opt), obj_fun(x)]: " \
                    + f"[{np.max(np.abs(x_wec)):.2e}, " \
                    + f"{np.max(np.abs(x_opt)):.2e}, " \
                    + f"{np.max(obj_fun_scaled(x)):.2e}]")

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
            _log.error(msg)

        # unscale
        res.x = res.x / scale
        res.fun = res.fun / scale_obj

        # post-process
        x_wec, x_opt = self.decompose_decision_var(res.x)
        fd_x, td_x = self._post_process_wec_dynamics(x_wec, x_opt)
        fd_we = fd_we.reset_coords(drop=True)
        time_dom = xr.merge([td_x, td_we])
        freq_dom = xr.merge([fd_x, fd_we])

        objective = res.fun * sign

        return time_dom, freq_dom, x_wec, x_opt, objective, res



def frequency(f0: float, nfreq: int) -> np.ndarray:
    """Construct equally spaced frequency array.
    """
    return np.arange(0, nfreq+1)*f0


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


def degrees_to_radians(degrees: float | npt.ArrayLike, sort: bool = True,
                       ) -> float | np.ndarray:
    """Convert degrees to radians in range -π to π and sort.
    """
    radians = np.asarray(np.remainder(np.deg2rad(degrees), 2*np.pi))
    radians[radians > np.pi] -= 2*np.pi
    if (radians.size == 1):
        radians = radians.item()
    elif (sort):
        radians = np.sort(radians)
    return radians


def _check_damping(bem_data, tol=1e-6) -> xr.Dataset:
    damping = bem_data['radiation_damping'] + bem_data['friction']
    dmin = np.diagonal(damping,axis1=1,axis2=2).min()
    if dmin <= 0.0 + tol:
        _log.warning(f'Linear damping has negative' +
                    ' or close to zero terms; shifting up via linear friction.')
        bem_data['friction'] = bem_data['friction'] + tol-dmin

    return bem_data


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
    nsteps = nsubsteps * (2*nfreq + 1)
    return np.linspace(0, 1/f0, nsteps, endpoint=False)


def time_mat(f0: float, nfreq: int,
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
    time = time(f0, ncomponents, nsubsteps)
    omega = frequency(f0, nfreq) * 2*np.pi
    wt = np.outer(time, omega[1:])
    time_mat = np.empty((nsubsteps*ncomponents, ncomponents))
    time_mat[:, 0] = 1.0
    time_mat[:, 1::2] = np.cos(wt)
    time_mat[:, 2::2] = np.sin(wt)
    return time_mat


def vec_to_dofmat(vec: np.ndarray, ndof: int) -> np.ndarray:
        """Convert a vector back to a matrix with one column per DOF.
        Opposite of ``dofmat_to_vec``. """
        return np.reshape(vec, (-1, ndof), order='F')


def mimo_transfer_mat(imp: np.ndarray, ndof:int) -> np.ndarray:
    """Create a block matrix of the MIMO transfer function.
    """
    elem = [[None]*ndof for _ in range(ndof)]
    def block(re, im): return np.array([[re, im], [-im, re]])
    for idof in range(ndof):
        for jdof in range(ndof):
            Zp = imp[1:, idof, jdof]
            Zp0 = imp[0, idof, jdof]
            re = np.real(Zp)
            im = np.imag(Zp)
            blocks = [block(ire, iim) for (ire, iim) in zip(re, im)]
            blocks =[Zp0] + blocks
            elem[idof][jdof] = block_diag(*blocks)
    return np.block(elem)


def fd_to_td(fd: np.ndarray, n: Optional[int] = None) -> np.ndarray:
    return np.fft.irfft(fd/2, n=n, axis=0, norm='forward')


def td_to_fd(td: np.ndarray, n: Optional[int] = None) -> np.ndarray:
    return np.fft.rfft(td*2, n=n, axis=0, norm='forward')


def wave_excitation(exc_coeff: xr.Dataset, waves: xr.Dataset
                    ) -> tuple[xr.Dataset, xr.Dataset]:
    """Compute the frequency- and time-domain wave excitation force.

    Parameters
    ----------
    exc_coeff: xarray.Dataset
        Exctiation BEM data for the WEC obtained from `capytaine`.
    waves : xarray.Dataset
        The wave, described by two 2D DataArrays:
        elevation variance `S` (m^2*s) and phase `phase` (radians)
        with coordinates of radial frequency `omega` (radians)
        and wave direction `wave_direction` (radians). The frequencies
        and  wave directions must match those in the `exc_coeff`.

    Returns
    -------
    freq_dom: xarray.Dataset
        Frequency domain wave excitation and elevation.
    time_dom: xarray.Dataset
        Time domain wave excitation and elevation.
    """
    #TODO
    # if not np.allclose(waves['omega'].values, exc_coeff['omega'].values):
    #     raise ValueError("Wave and BEM frequencies do not match")

    #TODO
    # w_dir_subset, w_indx = subsetclose(waves['wave_direction'].values,
    #             exc_coeff['wave_direction'].values)
    # if not w_dir_subset:
    #     raise ValueError(
    #         "Some wave directions are not in BEM solution " +
    #         "\n Wave direction(s):" +
    #         f"{(np.rad2deg(waves['wave_direction'].values))} (deg)" +
    #         " \n BEM directions: " +
    #         f"{np.rad2deg(exc_coeff['wave_direction'].values)} (deg).")

    # add zero frequency
    assert waves.omega[0] != 0
    tmp = waves.isel(omega=0).copy(deep=True)
    tmp['omega'] = tmp['omega'] * 0
    tmp['S'] = tmp['S'] * 0
    tmp['phase'] = tmp['phase'] * 0
    waves_p0 = xr.concat([tmp, waves], dim='omega')

    assert exc_coeff.omega[0] != 0
    tmp = exc_coeff.isel(omega=0).copy(deep=True)
    tmp['omega'] = tmp['omega'] * 0
    tmp = tmp * 0
    tmp['wavenumber'] = 0.0
    tmp['wavelength'] = np.inf
    exc_coeff_p0 = xr.concat([tmp, exc_coeff], dim='omega')

    # complex amplitude
    dw = waves_p0.omega[1] - waves_p0.omega[0]
    wave_elev_fd = (np.sqrt(2*waves_p0['S'] / (2*np.pi) * dw) *
                    np.exp(1j*waves_p0['phase']))
    wave_elev_fd.attrs['long_name'] = 'wave elevation'
    wave_elev_fd.attrs['units'] = 'm^2*s'
    wave_elev_fd = wave_elev_fd.transpose('omega', 'wave_direction')

    # excitation force
    f_exc_fd = xr.dot(exc_coeff_p0, wave_elev_fd, dims=["wave_direction"])
    f_exc_fd.attrs['long_name'] = 'wave excitation force'
    f_exc_fd.attrs['units'] = 'N^2*s or N^2*m^2*s'
    f_exc_fd = f_exc_fd.transpose('omega', 'influenced_dof')

    freq_dom = xr.Dataset(
        {'wave_elevation': wave_elev_fd, 'excitation_force': f_exc_fd},)
    freq_dom['omega'].attrs['long_name'] = 'frequency'
    freq_dom['omega'].attrs['units'] = '(radians)'

    # time domain
    nfd = 2 * len(waves['omega']) + 1
    f0 = waves['omega'][0] / (2*np.pi)
    time = np.linspace(0, 1/f0, nfd, endpoint=False)
    dims_td = ['time', ]
    coords_td = [(dims_td[0], time, {'units': 's'}), ]

    f_exc_td = fd_to_td(f_exc_fd, nfd)
    dims = dims_td + ['influenced_dof']
    coords = coords_td + [(dims[1], f_exc_fd.coords[dims[1]].data,)]
    f_exc_td = xr.DataArray(
        f_exc_td, dims=dims, coords=coords, attrs=f_exc_fd.attrs)
    f_exc_td.attrs['units'] = 'N or N*m'
    time_dom = xr.Dataset({'excitation_force': f_exc_td},)

    eta_all = fd_to_td(wave_elev_fd, nfd)
    wave_elev_td = np.sum(eta_all, axis=1)
    wave_elev_td = xr.DataArray(
        wave_elev_td, dims=dims_td, coords=coords_td, attrs=wave_elev_fd.attrs)
    wave_elev_td.attrs['units'] = 'm'
    time_dom['wave_elevation'] = wave_elev_td

    return freq_dom, time_dom


def standard_forces(bem_data: xr.Dataset):

    # add zero freq. components
    tmp = bem_data.isel(omega=0).copy(deep=True)
    tmp['omega'] = tmp['omega'] * 0
    tmp['hydrostatic_stiffness'] = bem_data['hydrostatic_stiffness'] * 1
    bem_data = xr.concat([tmp, bem_data], dim='omega')
    bem_data = bem_data.transpose("omega",
                                  "radiating_dof",
                                  "influenced_dof",
                                  "wave_direction")

    w = bem_data['omega']
    A = bem_data['added_mass']
    B = bem_data['radiation_damping']
    K = bem_data['hydrostatic_stiffness']
    m = bem_data['mass']
    Bf = bem_data['friction']

    ndof = len(bem_data.influenced_dof)

    #TODO: operate on position

    impedance_components = dict()
    impedance_components['inertia'] = 1j*w*m
    impedance_components['radiation'] = -(B + 1j*w*A)
    impedance_components['hydrostatics'] = -1j/w*K
    impedance_components['friction'] = -Bf + B*0  #TODO: this is my way of getting the shape right - kind of sloppy?

    def f_from_imp(transfer_mat):
            def f(wec, x_wec, x_opt, waves):
                f_fd = vec_to_dofmat(np.dot(transfer_mat, x_wec), ndof)
                return np.dot(wec.time_mat, f_fd)
            return f

    linear_force_mimo_matrices = dict()
    linear_force_functions = dict()
    for k, v in impedance_components.items():
        linear_force_mimo_matrices[k] = _mimo_transfer_mat(v,ndof)
        linear_force_functions[k] = f_from_imp(linear_force_mimo_matrices[k])

    def f_exc(TF, waves):
        _, td = _wave_excitation(exc_coeff=TF, waves=waves)
        return td['excitation_force']

    def f_exc_fk(wec,x_wec,x_opt,waves):
        return f_exc(TF=bem_data['Froude_Krylov_force'], waves=waves)

    def f_exc_diff(wec,x_wec,x_opt,waves):
        return f_exc(TF=bem_data['diffraction'], waves=waves)

    linear_force_functions['Froude_Krylov'] = f_exc_fk
    linear_force_functions['diffraction'] = f_exc_diff

    return linear_force_functions, linear_force_mimo_matrices


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


def subsetclose(subset_a: float | npt.ArrayLike,
                set_b: float | npt.ArrayLike,
                rtol: float = 1.e-5, atol: float = 1.e-8,
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
    result: bool
        Boolean if the entire first array is a subset of second array
    ind: list
        List with integer indices where the first array's elements
        are located inside the second array.
    """
    # TODO is there a way to check this using the given tolerances?
    assert len(np.unique(subset_a.round(decimals=6))) == len(
        subset_a), "Elements in subset_a not unique"
    assert len(np.unique(set_b.round(decimals=6))) == len(
        set_b), "Elements in set_b not unique"

    ind = []
    tmp_result = [False for i in range(len(subset_a))]
    for subset_element in subset_a:
        for set_element in set_b:
            if np.isclose(subset_element, set_element, rtol, atol, equal_nan):
                tmp_set_ind = np.where(
                    np.isclose(set_element, set_b, rtol, atol, equal_nan))
                tmp_subset_ind = np.where(
                    np.isclose(subset_element, subset_a, rtol, atol,
                               equal_nan))
                ind.append(int(tmp_set_ind[0]))
                tmp_result[int(tmp_subset_ind[0])] = True
    result = all(tmp_result)
    return(result, ind)


def derivative_mat(f0: float, nfreq: int) -> np.ndarray:
    def block(n): return np.array([[0, 1], [-1, 0]]) * n * f0 * 2*np.pi
    blocks = [block(n+1) for n in range(nfreq)]
    blocks = [0.0] + blocks
    return block_diag(*blocks)
