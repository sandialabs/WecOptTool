"""Provide core functionality for solving the pseudo-spectral problem.
"""


from __future__ import annotations  # TODO: delete after python 3.10


__all__ = ['WEC', 'freq_array', 'real_to_complex_amplitudes', 'fd_to_td',
           'td_to_fd', 'scale_dofs', 'complex_xarray_from_netcdf',
           'complex_xarray_to_netcdf', 'wave_excitation', 'run_bem',
           'power_limit', 'natural_frequency', 'plot_impedance',
           'post_process_continuous_time']


import logging
from typing import Iterable, Callable, Any
from pathlib import Path

import numpy.typing as npt
import autograd.numpy as np
from autograd.builtins import isinstance, tuple, list, dict
from autograd import grad, jacobian
import xarray as xr
import capytaine as cpy
from scipy.optimize import minimize
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
import matplotlib as mpl


log = logging.getLogger(__name__)

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

    def __init__(self, fb: cpy.FloatingBody, mass: np.ndarray,
                 hydrostatic_stiffness: np.ndarray, f0: float, nfreq: int,
                 dissipation: np.ndarray | None = None,
                 stiffness: np.ndarray | None = None,
                 f_add: Callable[[WEC, np.ndarray, np.ndarray], np.ndarray] |
                 None = None, constraints: list[dict] = [],
                 rho: float = _default_parameters['rho'],
                 depth: float = _default_parameters['depth'],
                 g: float = _default_parameters['g']) -> None:
        """
        Parameters
        ----------
        fb: capytaine.FloatingBody
            The WEC as a capytaine floating body (mesh + DOFs).
        mass: np.ndarray
            Mass matrix shape of (``ndof`` x ``ndof``).
        hydrostatic_stiffness: np.ndarray
            Hydrstatic stiffness matrix matrix of shape
            (``ndof`` x ``ndof``).
        f0: float
            Initial frequency (in Hz) for frequency array.
            Frequency array given as [``f0``, 2 ``f0``, ..., ``nfreq f0``].
        nfreq: int
            Number of frequencies in frequency array. See ``f0``.
        dissipation: np.ndarray
            Additional dissipiation for the impedance calculation in
            ``capytaine.post_pro.impedance``. Shape:
            (``ndof`` x ``ndof`` x ``1``) or (``ndof`` x ``ndof`` x ``nfreq``).
        stiffness: np.ndarray
            Additional stiffness for the impedance calculation in
            ``capytaine.post_pro.impedance``. Shape:
            (``ndof`` x ``ndof`` x ``1``) or (``ndof`` x ``ndof`` x ``nfreq``).
        f_add: function
            Additional forcing terms (e.g. PTO, mooring, etc.) for the
            WEC dynamics in the time-domain. Takes three inputs:
            (1) the WEC object,
            (2) the WEC dynamics state (1D np.ndarray), and
            (3) the optimization state (1D np.ndarray)
            and outputs the force time-series (1D np.ndarray).
        constraints: list[dict]
            Constraints for the constrained optimization.
            See ``scipy.optimize.minimize``.
        rho: float, optional
            Water density in :math:`kg/m^3`. Default is 1025.
        depth: float, optional
            Water depth in :math:`m`. Default is np.inf
        g: float, optional
            Gravitational acceleration in :math:`m/s^2`.
            Default is 9.81.
        """
        # water properties
        super().__setattr__('rho', rho)
        super().__setattr__('depth', depth)
        super().__setattr__('g', g)

        # WEC
        super().__setattr__('fb', fb)
        super().__setattr__('mass', mass)
        super().__setattr__('hydrostatic_stiffness', hydrostatic_stiffness)

        # frequency
        super().__setattr__('freq', (f0, nfreq))

        # additional WEC dynamics forces
        super().__setattr__('f_add', f_add)
        if stiffness is None:
            stiffness = 0.0
        super().__setattr__('stiffness', stiffness)
        if dissipation is None:
            dissipation = 0.0
        super().__setattr__('dissipation', dissipation)

        # constraints
        super().__setattr__('constraints', constraints)

        # log
        log.info(f"New WEC: {self.fb.name} with {self.fb.nb_dofs} DOF.")

        # BEM
        super().__setattr__('hydro', None)

    def __setattr__(self, name, value):
        """Delete dependent attributes  when user manually modifies
        an attribute.
        """
        _attrs_delete_mass = ['fb']
        _attrs_delete_stiffness = ['fb', 'rho', 'g']
        _attrs_delete_bem = ['fb', 'rho', 'g', 'depth', 'freq']
        _attrs_delete_impedance_not_bem = [
            'dissipation', 'stiffness', 'mass', 'hydrostatic_stiffness']
        log.info(f"Changing value of '{name}'. " +
                 "This might cause some attributes to be reset.")
        if name in _attrs_delete_mass:
            super().__setattr__('mass', None)
            if self.hydro is not None:
                self.hydro['mass'] = 'None'
            log.info("Mass matrix deleted. " +
                     "Assign new values to 'self.mass'")
        if name in _attrs_delete_stiffness:
            super().__setattr__('hydrostatic_stiffness', None)
            if self.hydro is not None:
                self.hydro['hydrostatic_stiffness'] = 'None'
            log.info("Hydrostatic stiffness deleted. " +
                     "Assign new values to 'self.hydrostatic_stiffness'")
        if name in _attrs_delete_bem:
            super().__setattr__('hydro', None)
            super().__setattr__('_transfer_mat', None)
            log.info("BEM data deleted. To run BEM use self.run_bem(...) " +
                     "followed by 'self.bem_calc_impedance()' to calculate " +
                     "impedance.")
        if name in _attrs_delete_impedance_not_bem:
            if self.hydro is not None:
                if 'Gi' in self.hydro:
                    self.hydro['Gi'] = 'None'
                if 'Zi' in self.hydro:
                    self.hydro['Zi'] = 'None'
            super().__setattr__('_transfer_mat', None)
            log.info("Impedance matrix deleted. To calculate " +
                     "impedance call 'self.bem_calc_impedance()'")

        # set attribute
        super().__setattr__(name, value)

        if name in _attrs_delete_impedance_not_bem:
            # keep BEM coefficients but update all other values of 'hydro'
            if self.hydro is not None:
                self._bem_add_hydrostatics()
                self._bem_add_linear_forces()

    def __repr__(self):
        str_info = (f'{self.__class__.__name__} ' +
                    f'"{self.fb.name}" with {self.fb.nb_dofs} DOF.')
        return str_info

    # PROPERTIES
    # properties: frequency
    @property
    def freq(self):
        """Frequency array f=[f0, 2*f0, ..., nfreq*f0] in Hz."""
        return self._freq

    @freq.setter
    def freq(self, freq):
        if len(freq) != 2:
            msg = "To set the frequency provide a tuple (f0, nfreq)."
            raise TypeError(msg)
        f0, nfreq = freq
        super().__setattr__('_freq', freq_array(f0, nfreq))
        # update time vector
        super().__setattr__('_time', self.make_time_vec(1))
        # update transformation matrixes
        super().__setattr__('_time_mat', self.make_time_mat(1, True))
        super().__setattr__('_derivative_mat', self._make_derivative_mat(True))

    @property
    def f0(self):
        """Initial frequency (and spacing) in Hz. See ``freq``."""
        return self._freq[0]

    @property
    def nfreq(self):
        """Number of frequencies in frequency array. See ``freq``."""
        return len(self._freq)

    @property
    def period(self):
        """Period :math:T=1/f in seconds."""
        return 1/self._freq

    @property
    def omega(self):
        """Frequency array in radians per second ω=2πf."""
        return self._freq * 2 * np.pi

    @property
    def w0(self):
        """Initial frequency (and spacing) in rad/s. See ``freq``."""
        return self._freq[0] * 2 * np.pi

    # properties: problem size
    @property
    def ndof(self):
        """Number of degrees of freedom of the WEC."""
        return self.fb.nb_dofs

    @property
    def ncomponents(self):
        """Number of state values for each DOF in the WEC dynamics."""
        return 2 * self.nfreq + 1

    @property
    def nstate_wec(self):
        """Length of the  WEC dynamics state vector."""
        return self.ndof * self.ncomponents

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

    ## METHODS
    # methods: class I/O
    def to_file(self, fpath: str | Path) -> None:
        """Save the WEC to a file. """
        # TODO
        raise NotImplementedError()

    @staticmethod
    def from_file(fpath: str | Path) -> WEC:
        """Create a WEC instance from a file saved using `WEC.to_file`.
        """
        # TODO
        raise NotImplementedError()

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

    # methods: transformation matrices
    def make_time_vec(self, nsubsteps: int = 1) -> np.ndarray:
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
        nsteps = nsubsteps * self.ncomponents
        return np.linspace(0, 1/self.f0, nsteps, endpoint=False)

    def make_time_mat(self, nsubsteps: int = 1, include_mean: bool = True
                      ) -> np.ndarray:
        """Assemble the time matrix that converts the state to
        time-series.

        Parameters
        ---------
        nsubsteps: int
            Number of subdivisions between the default (implied) time
            steps.
        include_mean: bool
            Whether the state vector includes a mean component.

        Returns
        -------
        time_mat: np.ndarray
        """
        time = self.make_time_vec(nsubsteps)
        wt = np.outer(time, self.omega)
        time_mat = np.empty((nsubsteps*self.ncomponents, self.ncomponents))
        time_mat[:, 0] = 1.0
        time_mat[:, 1::2] = np.cos(wt)
        time_mat[:, 2::2] = np.sin(wt)
        if not include_mean:
            time_mat = time_mat[:, 1:]
        return time_mat

    def _make_derivative_mat(self, include_mean: bool = True) -> np.ndarray:
        def block(n): return np.array([[0, 1], [-1, 0]]) * n * self.w0
        blocks = [block(n+1) for n in range(self.nfreq)]
        if include_mean:
            blocks = [0.0] + blocks
        return block_diag(*blocks)

    # methods: fft
    def fd_to_td(self, fd: np.ndarray) -> np.ndarray:
        """Convert from frequency domain to time domain using the FFT.
        """
        return fd_to_td(fd, self.ncomponents)

    def td_to_fd(self, td: np.ndarray) -> np.ndarray:
        """Convert from frequency domain to time domain using the iFFT.
        """
        return td_to_fd(td, self.ncomponents)

    # methods: bem & impedance
    def run_bem(self, wave_dirs: npt.ArrayLike = [0], tol: float = 1e-6
                ) -> None:
        """Run the BEM for the specified wave directions.

        See ``wot.run_bem``.

        Parameters
        ----------
        wave_dirs: list[float]
            List of wave directions to evaluate BEM at.
        tol: float
            Minimum value for the diagonal terms of
            (radiation damping + dissipation).
        """
        log.info(f"Running Capytaine (BEM): {self.nfreq} frequencies x " +
                 f"{len(wave_dirs)} wave directions.")
        write_info = ['hydrostatics', 'mesh', 'wavelength', 'wavenumber']
        data = run_bem(self.fb, self.freq, wave_dirs,
                       rho=self.rho, g=self.g, depth=self.depth,
                       write_info=write_info)
        super().__setattr__('hydro', data)
        # calculate impedance, ensure positive, create matrix
        self.bem_calc_impedance(tol)

    def read_bem(self, fpath: str | Path, tol: float = 1e-6) -> None:
        """Read a BEM solution from a NetCDF file.

        Parameters
        ----------
        fpath: str
            Name of file to read BEM data from.
        tol: float
            Minimum value for the diagonal terms of
            (radiation damping + dissipation).
        """
        log.info(f"Reading BEM data from {fpath}.")
        data = complex_xarray_from_netcdf(fpath)
        super().__setattr__('hydro', data)

        def diff(v1, v2, var):
            if not np.allclose(v1, v2):
                msg = f"Current and saved values of '{var}' are different."
                msg += "Using current value."
                log.warning(msg)

        # check: mass and stiffness
        bmass = 'mass' in self.hydro
        bstiffness = 'hydrostatic_stiffness' in self.hydro
        if bmass:
            diff(self.hydro['mass'].values, self.mass, 'mass')
        if bstiffness:
            diff(self.hydro['hydrostatic_stiffness'].values,
                 self.hydrostatic_stiffness, 'hydrostatic_stiffness')

        # check: additional linear stiffness and dissipation
        bstiffness = 'stiffness' in self.hydro
        bdissipation = 'dissipation' in self.hydro
        if bstiffness:
            diff(self.hydro['stiffness'].values, self.stiffness, 'stiffness')
        if bdissipation:
            diff(self.hydro['dissipation'].values, self.dissipation,
                 'dissipation')

        # add impedance
        self.bem_calc_impedance(tol)

    def write_bem(self, fpath: str | Path) -> None:
        """Write the BEM solution to a NetCDF file.

        Parameters
        ----------
        fpath: str
            Name of file to write BEM data to.
        """
        log.info(f"Writting BEM data to {fpath}.")
        complex_xarray_to_netcdf(fpath, self.hydro)

    def bem_calc_impedance(self, tol=1e-6):
        """Calculate the impedance, ensure positive real diagonal, and
        create impedance MIMO matrix. """
        self._bem_add_hydrostatics()
        self._bem_add_linear_forces()
        self._bem_calc_transfer_func()
        # post-process impedance: no negative or too small damping diagonal
        self._post_process_impedance(tol=tol)
        # create impedance MIMO matrix
        self._make_mimo_transfer_mat()

    def _bem_calc_transfer_func(self) -> None:
        """Calculate the transfer function matrix using Capytaine.
        """
        log.info("Calculating impedance matrix.")
        self.hydro['Gi'] = cpy.post_pro.impedance(
            self.hydro, self.dissipation, self.stiffness)
        self._bem_calc_impedance()

    def _bem_calc_impedance(self) -> None:
        """Calculate the impedance matrix."""
        self.hydro['Zi'] = self.hydro['Gi'] / (1j*self.hydro.omega)

    def _bem_add_hydrostatics(self) -> None:
        """Add hydrostatic data to self.hydro. """
        dims = ['radiating_dof', 'influenced_dof']
        self.hydro['mass'] = (dims, self.mass)
        self.hydro['hydrostatic_stiffness'] = (
            dims, self.hydrostatic_stiffness)
        self._del_impedance()

    def _bem_add_linear_forces(self) -> None:
        hydro = self.hydro.assign_coords({'dissipation': self.dissipation})
        hydro = hydro.assign_coords({'stiffness': self.stiffness})
        super().__setattr__('hydro', hydro)
        self._del_impedance()

    def _del_impedance(self) -> None:
        log.info("Impedance matrix deleted. To calculate " +
                 "impedance call 'self.bem_calc_impedance()'")
        self.hydro['Gi'] = 'None'
        self.hydro['Zi'] = 'None'
        super().__setattr__('_transfer_mat', None)

    def _post_process_impedance(self, tol=1e-6) -> None:
        """Enforce damping diagonal >= 0 + tol. """
        # ensure non-negative linear damping diagonal
        for idof in range(self.ndof):
            Gi_imag = np.imag(self.hydro['Gi'].isel(
                radiating_dof=idof, influenced_dof=idof))
            damping = Gi_imag / self.omega
            dmin = damping.min().values
            if dmin <= 0.0 + tol:
                dof = self.hydro['Gi'].influenced_dof.values[idof]
                log.warning(f'Linear damping for DOF "{dof}" has negative' +
                            ' or close to zero terms. Shifting up.')
                damping[:] = damping[:] + self.omega*(tol-dmin)

    def _make_mimo_transfer_mat(self) -> np.ndarray:
        """Create a block matrix of the MIMO transfer function +
        position.
        """
        elem = [[None]*self.ndof for _ in range(self.ndof)]
        def block(re, im): return np.array([[re, im], [-im, re]])
        for idof in range(self.ndof):
            for jdof in range(self.ndof):
                K = np.array([self.hydrostatic_stiffness[idof, jdof]])
                Zp = self.hydro['Gi'].values[:, idof, jdof]
                re = np.real(Zp)
                im = np.imag(Zp)
                blocks = [block(ire, iim) for (ire, iim) in zip(re, im)]
                blocks = [K] + blocks
                elem[idof][jdof] = block_diag(*blocks)
        super().__setattr__('_transfer_mat', np.block(elem))

    def bem_calc_inf_added_mass(self) -> None:
        """Run the BEM to obtain the infinite added mass. """
        log.info("Running Capytaine for infinite frequency.")
        inf_data = run_bem(
            self.fb, [np.infty], wave_dirs=None,
            rho=self.rho, g=self.g, depth=self.depth)
        self.hydro['Ainf'] = inf_data.added_mass[0, :, :]

    def bem_calc_rao(self) -> None:
        """Calculate BEM RAOs using capytaine. """
        self.hydro['rao'] = cpy.post_pro.rao(self.hydro)

    def plot_impedance(self, style: str = 'Bode', option: str = 'symmetric',
                       show: bool = True):
        """Plot impedance.

        See `wot.plot_impedance()`.
        """
        fig, axs = plot_impedance(
            impedance=self.hydro.Zi.values, freq=self.freq, style=style,
            option=option, dof_names=self.hydro.influenced_dof.values.tolist(),
            show=show)
        return fig, axs

    def natural_frequency(self):
        """Return natural frequency or frequencies.

        See `wot.natural_frequency()`.
        """
        return natural_frequency(self.hydro.Zi.values, freq=self.freq)

    # methods: solve
    def _get_state_scale(self,
                         scale_x_wec: list | None = None,
                         scale_x_opt: npt.ArrayLike | float = 1.0,
                         nstate_opt: int | None = None):
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

    def solve(self,
              waves: xr.Dataset,
              obj_fun: Callable[[WEC, np.ndarray, np.ndarray], float],
              nstate_opt: int,
              x_wec_0: np.ndarray | None = None,
              x_opt_0: np.ndarray | None = None,
              scale_x_wec: list | None = None,
              scale_x_opt: npt.ArrayLike | float = 1.0,
              scale_obj: float = 1.0,
              optim_options: dict[str, Any] = {},
              use_grad: bool = True,
              maximize: bool = False,
              scale_logging: bool = False,
              ) -> tuple[xr.Dataset, xr.Dataset, np.ndarray, np.ndarray, float,
                         optimize.optimize.OptimizeResult]:
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
        scale_logging: bool
            If true, print the value of the decision variable (decomposed into
            x_wec and x_opt) and objective function at each solver iteration.
            Useful for setting ``scale_x_wec``, scale_x_opt``, and
            ``scale_obj``. The default is `False``.

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
        log.info("Solving pseudo-spectral control problem.")

        # initial state
        if x_wec_0 is None:
            x_wec_0 = np.random.randn(self.nstate_wec)
        if x_opt_0 is None:
            x_opt_0 = np.random.randn(nstate_opt)
        x0 = np.concatenate([x_wec_0, x_opt_0])

        # wave excitation force
        fd_we, td_we = wave_excitation(self.hydro, waves)
        f_exc = td_we['excitation_force']

        # scale
        scale = self._get_state_scale(scale_x_wec, scale_x_opt, nstate_opt)

        # objective function
        sign = -1.0 if maximize else 1.0

        def obj_fun_scaled(x):
            x_wec, x_opt = self.decompose_decision_var(x/scale)
            return obj_fun(self, x_wec, x_opt)*scale_obj*sign

        # constraints
        constraints = self.constraints.copy()

        # constraints & Jacobians
        for i, icons in enumerate(constraints):
            icons_new = icons.copy()

            def tmp_func(x):
                x_wec, x_opt = self.decompose_decision_var(x/scale)
                return icons['fun'](self, x_wec, x_opt)

            icons_new['fun'] = tmp_func
            if use_grad:
                icons_new['jac'] = jacobian(icons_new['fun'])
            constraints[i] = icons_new

        # system dynamics through equality constraint

        def resid_fun(x):
            ri = self._dynamic_residual(x/scale, f_exc.values)
            return self.dofmat_to_vec(ri)

        eq_cons = {'type': 'eq',
                   'fun': resid_fun,
                   }
        if use_grad:
            eq_cons['jac'] = jacobian(resid_fun)
        constraints.append(eq_cons)

        # minimize
        optim_options['disp'] = optim_options.get('disp', True)

        problem = {'fun': obj_fun_scaled,
                   'x0': x0,
                   'method': 'SLSQP',
                   'constraints': constraints,
                   'options': optim_options,
                   }

        def callback(x):
            x_wec, x_opt = self.decompose_decision_var(x)
            log.debug("[mean(x_wec), mean(x_opt), obj_fun(x)]: " \
                + f"[{np.abs(np.mean(x_wec)):.2e}" \
                + f"{np.abs(np.mean(x_opt)):.2e}" \
                + f"{np.abs(obj_fun_scaled(x)):.2e}]")
        
        if scale_logging:
            problem['callback'] = callback

        if use_grad:
            problem['jac'] = grad(obj_fun_scaled)

        res = minimize(**problem)

        msg = f'{res.message}    (Exit mode {res.status})'
        if res.status == 0:
            log.info(msg)
        elif res.status == 9:
            log.warning(msg)
        else:
            log.error(msg)

        # unscale
        res.x = res.x / scale
        res.fun = res.fun / scale_obj

        # post-process
        x_wec, x_opt = self.decompose_decision_var(res.x)
        fd_x, td_x = self._post_process_wec_dynamics(x_wec)
        fd_we = fd_we.reset_coords(drop=True)
        time_dom = xr.merge([td_x, td_we])
        freq_dom = xr.merge([fd_x, fd_we])

        objective = res.fun * sign

        return time_dom, freq_dom, x_wec, x_opt, objective, res

    def _dynamic_residual(self, x: np.ndarray, f_exc: np.ndarray
                          ) -> np.ndarray:
        """Solve WEC dynamics in residual form so that they may be
        enforced through a nonlinear constraint within an optimization
        problem.

        Parameters
        ----------
        x : np.ndarray
            Decision variable for optimization problem
        f_exc : np.ndarray
            Time history of excitation forcing at collocation points in
            body coordinate system

        Returns
        -------
        np.ndarray
            Residuals at collocation points

        """
        x_wec, x_opt = self.decompose_decision_var(x)

        # linear hydrodynamic forces
        f_i = self.vec_to_dofmat(np.dot(self._transfer_mat, x_wec))
        f_i = np.dot(self.time_mat, f_i)

        # additional forces
        if self.f_add is not None:
            f_add = self.f_add(self, x_wec, x_opt)
        else:
            f_add = 0.0
        return f_i - f_exc - f_add

    def _post_process_wec_dynamics(self, x_wec: np.ndarray
                                   ) -> tuple[xr.DataArray, xr.DataArray]:
        """Transform the results from optimization solution to a form
        that the user can work with directly.
        """
        # position
        pos = self.vec_to_dofmat(x_wec)
        pos_fd = real_to_complex_amplitudes(pos)
        pos_td = self.time_mat @ pos

        # velocity
        vel = self.derivative_mat @ pos
        vel_fd = real_to_complex_amplitudes(vel)
        vel_td = self.time_mat @ vel

        # acceleration
        acc = self.derivative_mat @ vel
        acc_fd = real_to_complex_amplitudes(acc)
        acc_td = self.time_mat @ acc

        # xarray - time domain
        dims_td = ('time', 'influenced_dof')
        coords_td = [
            (dims_td[0], self.time, {'units': 's'}),
            (dims_td[1], self.hydro.influenced_dof.values)]
        attrs_pos = {'long_name': 'WEC position', 'units': 'm or (rad)'}
        attrs_vel = {'long_name': 'WEC velocity', 'units': 'm/s or (rad)/s'}
        attrs_acc = {'long_name': 'WEC acceleration',
                     'units': 'm/s^2 or (rad)/s^2'}
        pos_td = xr.DataArray(
            pos_td, dims=dims_td, coords=coords_td, attrs=attrs_pos)
        vel_td = xr.DataArray(
            vel_td, dims=dims_td, coords=coords_td, attrs=attrs_vel)
        acc_td = xr.DataArray(
            acc_td, dims=dims_td, coords=coords_td, attrs=attrs_acc)
        time_dom = xr.Dataset({'pos': pos_td, 'vel': vel_td, 'acc': acc_td},)

        # xarray - frequency domain
        omega = np.concatenate([np.array([0.0]), self.omega])
        dims_fd = ('omega', 'influenced_dof')
        coords_fd = [
            (dims_fd[0], omega, {'units': '(rad)'}),
            (dims_fd[1], self.hydro.influenced_dof.values)]
        attrs_pos['units'] = 'm^2*s or (rad)^2*s'
        attrs_vel['units'] = 'm^2/s or (rad)^2/s'
        attrs_acc['units'] = 'm^2/s^3 or (rad)^2/s^3'
        pos_fd = xr.DataArray(
            pos_fd, dims=dims_fd, coords=coords_fd, attrs=attrs_pos)
        vel_fd = xr.DataArray(
            vel_fd, dims=dims_fd, coords=coords_fd, attrs=attrs_vel)
        acc_fd = xr.DataArray(
            acc_fd, dims=dims_fd, coords=coords_fd, attrs=attrs_acc)
        freq_dom = xr.Dataset({'pos': pos_fd, 'vel': vel_fd, 'acc': acc_fd},)

        return freq_dom, time_dom


def freq_array(f0: float, nfreq: int) -> np.ndarray:
    """Construct equally spaced frequency array.
    """
    return np.arange(1, nfreq+1)*f0


def real_to_complex_amplitudes(fd: np.ndarray, first_row_is_mean: bool = True
                               ) -> np.ndarray:
    """Convert from two real amplitudes to one complex amplitude per
    frequency. """
    if first_row_is_mean:
        mean = fd[0:1, :]
        fd = fd[1:, :]
    else:
        ndof = fd.shape[1]
        mean = np.zeros([1, ndof])
    return np.concatenate((mean, fd[0::2, :] - 1j*fd[1::2, :]), axis=0)


def fd_to_td(fd: np.ndarray, n: int | None = None) -> np.ndarray:
    return np.fft.irfft(fd/2, n=n, axis=0, norm='forward')


def td_to_fd(td: np.ndarray, n: int | None = None) -> np.ndarray:
    return np.fft.rfft(td*2, n=n, axis=0, norm='forward')


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


def complex_xarray_from_netcdf(fpath: str | Path) -> xr.Dataset:
    """Read a NetCDF file with commplex entries as an xarray dataSet.
    """
    with xr.open_dataset(fpath) as ds:
        ds.load()
    return cpy.io.xarray.merge_complex_values(ds)


def complex_xarray_to_netcdf(fpath: str | Path, bem_data: xr.Dataset) -> None:
    """Save an xarray dataSet with complex entries as a NetCDF file.
    """
    cpy.io.xarray.separate_complex_values(bem_data).to_netcdf(fpath)


def wave_excitation(bem_data: xr.Dataset, waves: xr.Dataset
                    ) -> tuple[xr.Dataset, xr.Dataset]:
    """Compute the frequency- and time-domain wave excitation force.

    Parameters
    ----------
    bem_data: xarray.Dataset
        BEM data for the WEC obtained from `capytaine`.
    waves : xarray.Dataset
        The wave, described by two 2D DataArrays:
        elevation variance `S` (m^2*s) and phase `phase` (radians)
        with coordinates of radial frequency `omega` (radians)
        and wave direction `wave_direction` (radians). The frequencies
        and  wave directions must match those in the `bem_data`.

    Returns
    -------
    freq_dom: xarray.Dataset
        Frequency domain wave excitation and elevation.
    time_dom: xarray.Dataset
        Time domain wave excitation and elevation.
    """
    assert np.allclose(waves['omega'].values, bem_data['omega'].values)
    assert np.allclose(waves['wave_direction'].values,
                       bem_data['wave_direction'].values)

    # excitation BEM
    exc_coeff = bem_data['Froude_Krylov_force'] + \
        bem_data['diffraction_force']

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
        List of wave directions to evaluate BEM at.
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


def power_limit(excitation: npt.ArrayLike, impedance: npt.ArrayLike
                ) -> np.ndarray:
    """Find upper limit for power.

    Parameters
    ----------
    exctiation: np.ndarray
        Complex exctitation spectrum. Shape: ``nfreq`` x ``ndof``
    impedance: np.ndarray
        Complex impedance matrix. Shape: ``nfreq`` x ``ndof`` x ``ndof``

    Returns
    -------
    power_limit
        Upper limit for power absorption.
    """

    power_limit = -1*np.sum(np.abs(excitation)**2 / (8*np.real(impedance)))

    return power_limit


def natural_frequency(impedance: npt.ArrayLike, freq: npt.ArrayLike
                      ) -> tuple[npt.ArrayLike, int]:
    """Find the natural frequency based on the lowest magnitude impedance.

    Parameters
    ----------
    impedance: np.ndarray
        Complex impedance matrix. Shape: nfreq x ndof x ndof
    freq: list[float]
        Frequencies.

    Returns
    -------
    f_n: float
        Natural frequency.
    ind: int
        Index of natural frequency.
    """

    ind = np.argmin(np.abs(impedance), axis=0)
    f_n = freq[ind]

    return f_n, ind


def plot_impedance(impedance: npt.ArrayLike, freq: npt.ArrayLike,
                   style: str = 'Bode',
                   option: str = 'diagonal', show: bool = False,
                   dof_names: list[str] | None = None
                   ) -> tuple[mpl.figure.Figure, np.ndarray]:
    """Plot the impedance matrix.

    Parameters
    ----------
    impedance: np.ndarray
        Complex impedance matrix. Shape: ``nfreq`` x ``ndof`` x ``ndof``
    freq: list[float]
        Frequencies in Hz.
    style: {'Bode','complex'}
        Whether to plot magnitude and angle (``Bode``) or real and
        imaginary (``complex``) parts.
    option: {'diagonal', 'symmetric', 'all'}
        Which terms of the matrix to plot:
        'diagonal' to plot only the diagonal terms,
        'symmetric' to plot only the lower triangular terms, and
        'all' to plot all terms.
    show: bool
        Whether to show the figure.
    dof_names: list[str]

    Returns
    -------
    fig: matplotlib.figure.Figure
    axs: np.ndarray[matplotlib.axes._subplots.AxesSubplot]
    """
    figh = 3.5
    figw = 2 * figh
    ndof = impedance.shape[-1]
    fig, axs = plt.subplots(
        ndof*2, ndof, figsize=(ndof*figw, ndof*figh),
        sharex='all', sharey='row', squeeze=False)

    if dof_names is None:
        dof_names = [f"DOF {i}" for i in range(ndof)]

    colors = (plt.rcParams['axes.prop_cycle'].by_key()['color']*10)[:ndof]

    def get_ylim(xmin, xmax, pad_factor=0.05):
        pad = pad_factor * (xmax - xmin)
        return (xmin-pad, xmax+pad)

    phase_ylim = get_ylim(-180, 180)
    mag_ylim = get_ylim(0.0, np.max(20*np.log10(np.abs(impedance))))
    real_ylim = get_ylim(np.min([0.0, np.min(np.real(impedance))]),
                         np.max(np.real(impedance)))
    imag_ylim = get_ylim(-np.max(np.abs(np.imag(impedance))),
                         +np.max(np.abs(np.imag(impedance))))

    def delaxes(axs, idof, jdof, ndof):
        for i, ax in enumerate([axs[idof*2, jdof], axs[idof*2+1, jdof]]):
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            if idof != ndof-1 or (idof == ndof-1 and i == 0):
                ax.tick_params(axis='x', which='both', bottom=False)
                ax.spines['bottom'].set_visible(False)
            if jdof != 0:
                ax.tick_params(axis='y', which='both', left=False)
                ax.spines['left'].set_visible(False)

    for idof in range(ndof):
        color = colors[idof]
        for jdof in range(ndof):
            # labels, ticks, etc
            if jdof == 0:
                if style == 'Bode':
                    l1 = 'Magnitude (dB)'
                    l2 = 'Phase (deg)'

                elif style == 'complex':
                    l1 = 'Real'
                    l2 = 'Imaginary'

                axs[idof*2, jdof].set_ylabel(l1)
                axs[idof*2+1, jdof].set_ylabel(l2)

            if idof == ndof-1:
                axs[idof*2+1, jdof].set_xlabel('Frequency (Hz)')

            if idof == 0:
                axs[idof*2, jdof].set_xlabel(dof_names[jdof])
                axs[idof*2, jdof].xaxis.set_label_position("top")

            if jdof == ndof-1:
                ax_ylabel = axs[idof*2, jdof].twinx()
                ax_ylabel.set_ylabel(dof_names[idof], rotation=-90,
                                     labelpad=12)
                ax_ylabel.yaxis.set_label_position("right")
                ax_ylabel.tick_params(axis='y', which='both', left=False,
                                      right=False, labelright=False)
                ax_ylabel.tick_params(axis='x', which='both', bottom=False)
                ax_ylabel.tick_params(axis='y', which='both', left=False)
                ax_ylabel.spines[:].set_visible(False)

            # plot
            all = (option == 'all')
            sym = (option == 'symmetric' and jdof <= idof)
            diag = (option == 'diagonal' and jdof == idof)
            plot = True if (all or sym or diag) else False
            if plot:
                iZi = impedance[:, idof, jdof]
                if style == 'Bode':
                    p1 = np.squeeze(20*np.log10(np.abs(iZi)))
                    p2 = np.squeeze(np.rad2deg(np.angle(iZi)))
                    yl1, yh1 = mag_ylim
                    yl2, yh2 = phase_ylim
                elif style == 'complex':
                    p1 = np.squeeze(np.real(iZi))
                    p2 = np.squeeze(np.imag(iZi))
                    yl1, yh1 = real_ylim
                    yl2, yh2 = imag_ylim
                axs[idof*2, jdof].semilogx(freq, p1, '-o',
                                           color=color,
                                           markersize=4,
                                           )
                axs[idof*2+1, jdof].semilogx(freq, p2, '-o',
                                             color=color,
                                             markersize=4,
                                             )
                axs[idof*2, jdof].grid(True, which='both')
                axs[idof*2+1, jdof].grid(True, which='both')
                axs[idof*2, jdof].set_ylim(yl1, yh1)
                axs[idof*2+1, jdof].set_ylim(yl2, yh2)
            else:
                delaxes(axs, idof, jdof, ndof)

    fig.align_ylabels(axs[:, 0])
    fig.align_ylabels(axs[:, -1])
    fig.align_xlabels(axs[-1, :])
    fig.align_xlabels(axs[0, :])
    fig.tight_layout()

    if show:
        plt.show()

    return fig, axs


def post_process_continuous_time(results: xr.DataArray
                                 ) -> Callable[float, float]:
    """Create a continuous function from the results in an xarray
    DataArray.

    The DataArray must be indexed by "omega": frequency in rad/s.
    There should be no other indices.

    Parameters
    ----------
    results: xr.DataArray
        DataArray containing the pseudo-spectral results.

    Returns
    -------
    func: Callable
        Continuous-time function.
    """
    def func(t):
        t = np.array(t)
        f = np.zeros(t.shape)
        for freq, mag in zip(results.omega.values, results.values):
            f += np.real(mag)*np.cos(freq*t) - np.imag(mag)*np.sin(freq*t)
        return f

    return func
