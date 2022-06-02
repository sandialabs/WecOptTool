"""Provide core functionality for solving the pseudo-spectral problem.
"""


from __future__ import annotations
import string
# import stat  # required for Python 3.8 & 3.9 support


# TODO
__all__ = ['WEC', 'read_netcdf', '_add_zerofreq_to_xr']
#     'real_to_complex_amplitudes', 'freq_array', '_degrees_to_radians',
#     ]


import logging
import copy
from typing import Iterable, Callable, Any, Optional, Mapping, TypeVar
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

# type aliases
TWEC = TypeVar("TWEC", bound="WEC")
TStateFunction = Callable[
    [TWEC, np.ndarray, np.ndarray, xr.Dataset], np.ndarray]


class WEC:

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
    def wave_direction(self):
        """Wave directions in degrees."""
        return self._wave_direction * 180/np.pi

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
