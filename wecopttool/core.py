"""Core functionality for solving the pseudo-spectral problem for WEC.

Contains:

* The *WEC* class
* Functions for basic functionality

.. note:: All contents of this module are imported into *WecOpTool* and
          can be called directly as :python:`wecopttool.<function>`
          instead of :python:`wecopttool.core.<function>`.
"""


from __future__ import annotations


import logging
from typing import Iterable, Callable, Any, Optional, Mapping, TypeVar, Union
from pathlib import Path
import warnings
from datetime import datetime

from numpy.typing import ArrayLike
import autograd.numpy as np
from autograd.numpy import ndarray
from autograd.builtins import isinstance, tuple, list, dict
from autograd import grad, jacobian
import xarray as xr
from xarray import DataArray, Dataset
import capytaine as cpy
from scipy.optimize import minimize, OptimizeResult, Bounds
from scipy.linalg import block_diag, dft


# logger
_log = logging.getLogger(__name__)

# autograd warnings
filter_msg = "Casting complex values to real discards the imaginary part"
warnings.filterwarnings("ignore", message=filter_msg)

# default values
_default_parameters = {'rho': 1025.0, 'g': 9.81, 'depth': np.infty}
_default_min_damping = 1e-6

# type aliases
TWEC = TypeVar("TWEC", bound="WEC")
TStateFunction = Callable[
    [TWEC, ndarray, ndarray, Dataset], ndarray]
TForceDict = dict[str, TStateFunction]
TIForceDict = Mapping[str, TStateFunction]
FloatOrArray = Union[float, ArrayLike]


class WEC:
    """A wave energy converter (WEC) object for performing simulations
    using the pseudo-spectral solution method.

    To create the WEC use one of the initialization methods:

    * :meth:`wecopttool.core.WEC.__init__`
    * :meth:`wecopttool.core.WEC.from_bem`
    * :meth:`wecopttool.core.WEC.from_floating_body`
    * :meth:`wecopttool.core.WEC.from_impedance`.

    .. note:: Direct initialization of a :py:class:`wecopttool.core.WEC`
        object as :python:`WEC(f1, nfrew, forces, ...)` using
        :meth:`wecopttool.core.WEC.__init__` is discouraged. Instead
        use one of the other initialization methods listed in the
        *See Also* section.

    To solve the pseudo-spectral problem use
    :meth:`wecopttool.core.WEC.solve`.
    """
    def __init__(
        self,
        f1:float,
        nfreq:int,
        forces: TIForceDict,
        constraints: Optional[Iterable[Mapping]] = None,
        inertia_matrix: Optional[ndarray] = None,
        ndof: Optional[int] = None,
        inertia_in_forces: Optional[bool] = False,
        dof_names: Optional[Iterable[str]] = None,
        ) -> None:
        """Create a WEC object directly from its inertia matrix and
        list of forces.

        The :py:class:`wecopttool.core.WEC` class describes a WEC's
        equation of motion as :math:`ma=Σf` where the
        :python:`inertia_matrix` matrix specifies the inertia :math:`m`,
        and the :python:`forces` dictionary specifies the different
        forces to be summed. The forces can be linear or nonlinear.
        If :python:`inertia_in_forces is True` the equation of motion is
        :math:`Σf=0`, which is included to allow for initialization
        using an intrinsic impedance through the
        :python:`WEC.from_impedance` initialization function.

        .. note:: Direct initialization of a
            :py:class:`wecopttool.core.WEC` object as
            :python:`WEC(f1, nfrew, forces, ...)` is discouraged.
            Instead use one of the other initialization methods listed
            in the *See Also* section.

        Parameters
        ----------
        f1
            Fundamental frequency :python:`f1` [Hz].
        nfreq
            Number of frequencies (not including zero frequency),
            i.e., :python:`freqs = [0, f1, 2*f1, ..., nfreq*f1]`.
        forces
            Dictionary with entries :python:`{'force_name': fun}`,
            where :python:`fun` has a  signature
            :python:`def fun(wec, x_wec, x_opt, waves):`, and returns
            forces in the time-domain of size
            :python:`2*nfreq + 1 x ndof`.
        constraints
            List of constraints, see documentation for
            :py:func:`scipy.optimize.minimize` for description and
            options of constraints dictionaries.
            If :python:`None`: empty list :python:`[]`.
        inertia_matrix
           Inertia matrix of size :python:`ndof x ndof`.
           Not used if :python:`inertia_in_forces` is :python:`True`.
        ndof
            Number of degrees of freedom.
            Must be specified if :python:`inertia_in_forces is True`,
            else not used.
        inertia_in_forces
            Set to True if inertial "forces" are included in the
            :python:`forces` argument.
            This scenario is rare.
            If using an intrinsic impedance, consider initializing with
            :python:`from_impedance` instead.
        dof_names
            Names of the different degrees of freedom (e.g.
            :python:`'Heave'`).
            If :python:`None` the names
            :python:`['DOF_0', ..., 'DOF_N']` are used.

        Raises
        ------
        ValueError
            If :python:`inertia_in_forces is True` but :python:`ndof` is
            not specified.
        ValueError
            If :python:`inertia_in_forces is False` but
            :python:`inertia_matrix` is not specified.
        ValueError
            If :python:`inertia_matrix` does not have the correct size
            (:python:`ndof x ndof`).
        ValueError
            If :python:`dof_names` does not have the correct size
            (:python:`ndof`).

        See Also
        --------
        from_bem:
            Initialize a :py:class:`wecopttool.core.WEC` object from BEM
            results.
        from_floating_body:
            Initialize a :py:class:`wecopttool.core.WEC` object from a
            :python:`capitaine.FloatingBody` object.
        from_impedance:
            Initialize a :py:class:`wecopttool.core.WEC` object from an
            intrinsic impedance array and excitation coefficients.
        """
        self._freq = frequency(f1, nfreq)
        self._time = time(f1, nfreq)
        self._time_mat = time_mat(f1, nfreq)
        self._derivative_mat = derivative_mat(f1, nfreq)
        self._forces = forces
        constraints = list(constraints) if (constraints is not None) else []
        self._constraints = constraints

        # inertia options
        self._inertia_in_forces = inertia_in_forces

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
            if inertia_matrix is not None:
                _log.warning(_ignored("inertia_matrix", condition))
                inertia_matrix = None
            if ndof is None:
                raise ValueError(_missing("ndof", condition))
        elif not inertia_in_forces:
            condition = "False"
            if inertia_matrix is None:
                raise ValueError(_missing("inertia_matrix", condition))
            inertia_matrix = np.atleast_2d(np.squeeze(inertia_matrix))
            if ndof is not None:
                _log.warning(_ignored("ndof", condition))
                if ndof != inertia_matrix.shape[0]:
                    _log.warning(
                        "Provided value of `ndof` does not match size of " +
                        "`inertia_matrix`. Setting " +
                        f"`ndof={inertia_matrix.shape[0]}`.")
            ndof = inertia_matrix.shape[0]

            if inertia_matrix.shape != (ndof, ndof):
                raise ValueError(
                    "'inertia_matrix' must be a square matrix of size equal " +
                    "to the number of degrees of freedom.")
        self._inertia_matrix = inertia_matrix
        self._ndof = ndof
        if inertia_in_forces:
            _inertia = None
        else:
            _inertia = inertia(f1, nfreq, inertia_matrix)
        self._inertia = _inertia

        # names
        if dof_names is None:
            dof_names = [f'DOF_{i}' for i in range(ndof)]
        elif len(dof_names) != ndof:
            raise ValueError("`dof_names` must have length `ndof`.")
        self._dof_names = list(dof_names)

    def __str__(self) -> str:
        str = (f'{self.__class__.__name__}: ' +
               f'DOFs ({self.ndof})={self.dof_names}, ' +
               f'f=[0, {self.f1}, ..., {self.nfreq}({self.f1})] Hz.')
        return str

    def __repr__(self) -> str:
        type_ = type(self)
        module = type_.__module__
        qualname = type_.__qualname__
        repr_org = f"<{module}.{qualname} object at {hex(id(self))}>"
        return repr_org + " :: " + self.__str__()

    # other initialization methods
    @staticmethod
    def from_bem(
        bem_data: Union[Dataset, Union[str, Path]],
        inertia_matrix: Optional[ndarray] = None,
        hydrostatic_stiffness: Optional[ndarray] = None,
        friction: Optional[ndarray] = None,
        f_add: Optional[TIForceDict] = None,
        constraints: Optional[Iterable[Mapping]] = None,
        min_damping: Optional[float] = _default_min_damping,
        ) -> TWEC:
        """Create a WEC object from linear hydrodynamic coefficients
        obtained using the boundary element method (BEM) code Capytaine.

        The :python:`bem_data` can be a dataset or the name of a
        *NetCDF* file containing the dataset.

        The returned :py:class:`wecopttool.core.WEC` object contains the
        inertia and the default linear forces: radiation, diffraction,
        and Froude-Krylov. Additional forces can be specified through
        :python:`f_add`.

        Note that because Capytaine uses a different sign convention,
        the direct results from capytaine must be modified using
        :py:func:`wecopttool.core.change_bem_convention` before calling
        this initialization function.
        Instead, the recommended approach is to use
        :py:func:`wecopttool.core.run_bem`,
        rather than running Capytaine directly, which outputs the
        results in the correct convention. The results can be saved
        using :py:func:`wecopttool.core.write_netcdf`.

        In addition to the Capytaine results, if the dataset contains
        the :python:`inertia_matrix`, :python:`hydrostatic_stiffness`,
        or :python:`friction` these do not need to be provided
        separately.

        Parameters
        ----------
        bem_data
            Linear hydrodynamic coefficients obtained using the boundary
            element method (BEM) code Capytaine, with sign convention
            corrected.
        inertia_matrix
           Inertia matrix of size :python:`ndof x ndof`.
           :python:`None` if included in :python:`bem_data`.
        hydrostatic_stiffness
            Linear hydrostatic restoring coefficient of size
            :python:`nodf x ndof`.
            :python:`None` if included in :python:`bem_data`.
        friction
            Linear friction, in addition to radiation damping, of size
            :python:`nodf x ndof`.
            :python:`None` if included in :python:`bem_data` or to set
            to zero.
        f_add
            Dictionary with entries :python:`{'force_name': fun}`, where
            :python:`fun` has a  signature
            :python:`def fun(wec, x_wec, x_opt, waves):`, and returns
            forces in the time-domain of size
            :python:`2*nfreq + 1 x ndof`.
        constraints
            List of constraints, see documentation for
            :py:func:`scipy.optimize.minimize` for description and
            options of constraints dictionaries.
            If :python:`None`: empty list :python:`[]`.
        min_damping
            Minimum damping level to ensure a stable system.
            See `check_linear_damping` for more details.

        Raises
        ------
        ValueError
            If either :python:`inertia_matrix` or
            :python:`hydrostatic_stiffness` are :python:`None` and is
            not included in :python:`bem_data`.
            See :python:`linear_hydrodynamics`.
        ValueError
            If any of :python:`inertia_matrix`,
            :python:`hydrostatic_stiffness`, or :python:`stiffness` are
            both provided and included in :python:`bem_data` but have
            different values.
            See :python:`linear_hydrodynamics`.

        See Also
        --------
        run_bem, linear_hydrodynamics, change_bem_convention,
        write_netcdf, check_linear_damping
        """
        if isinstance(bem_data, (str, Path)):
            bem_data = read_netcdf(bem_data)
        # add inertia_matrix, hydrostatic stiffness, and friction
        hydro_data = linear_hydrodynamics(
            bem_data, inertia_matrix, hydrostatic_stiffness, friction)
        if inertia_matrix is None:
            inertia_matrix = hydro_data['inertia_matrix'].values

        # frequency array
        f1, nfreq = frequency_parameters(
            hydro_data.omega.values/(2*np.pi), False)

        # check real part of damping diagonal > 0
        if min_damping is not None:
            hydro_data = check_linear_damping(hydro_data, min_damping)

        # forces in the dynamics equations
        linear_force_functions = standard_forces(hydro_data)
        f_add = f_add if (f_add is not None) else {}
        forces = linear_force_functions | f_add
        # constraints
        constraints = constraints if (constraints is not None) else []
        return WEC(f1, nfreq, forces, constraints, inertia_matrix)

    @staticmethod
    def from_floating_body(
        fb: cpy.FloatingBody,
        f1: float,
        nfreq: int,
        inertia_matrix: ndarray,
        hydrostatic_stiffness: ndarray,
        friction: Optional[ndarray] = None,
        f_add: Optional[TIForceDict] = None,
        constraints: Optional[Iterable[Mapping]] = None,
        min_damping: Optional[float] = _default_min_damping,
        wave_directions: Optional[ArrayLike] = np.array([0.0,]),
        rho: Optional[float] = _default_parameters['rho'],
        g: Optional[float] = _default_parameters['g'],
        depth: Optional[float] = _default_parameters['depth'],
    ) -> TWEC:
        """Create a WEC object from a Capytaine :python:`FloatingBody`.

        Capytaine :python:`FloatingBody` objects contain information on
        the mesh and degrees of freedom.

        This initialization method calls :python:`run_bem` followed by
        :python:`from_bem`.

        This will run Capytaine to obtain the linear hydrodynamic
        coefficients, which can take from a few minutes to several
        hours.
        Instead, if the hydrodynamic coefficients can be reused, it is
        recommended to run Capytaine first and save the results using
        :python:`run_bem` and :py:func:`wecopttool.core.write_netcdf`,
        and then initialize the :py:class:`wecopttool.core.WEC` object
        using :python:`from_bem`. This initialization method should be
        reserved for the cases where the hydrodynamic coefficients
        constantly change and are not reused, as for example for
        geometry optimization.

        Parameters
        ----------
        fb
            Capytaine FloatingBody.
        f1
            Fundamental frequency :python:`f1` [Hz].
        nfreq
            Number of frequencies (not including zero frequency),
            i.e., :python:`freqs = [0, f1, 2*f1, ..., nfreq*f1]`.
        inertia_matrix
           Inertia matrix of size :python:`ndof x ndof`.
        hydrostatic_stiffness
            Linear hydrostatic restoring coefficient of size
            :python:`nodf x ndof`.
        friction
            Linear friction, in addition to radiation damping, of size
            :python:`nodf x ndof`.
            :python:`None` to set to zero.
        f_add
            Dictionary with entries :python:`{'force_name': fun}`, where
            :python:`fun` has a  signature
            :python:`def fun(wec, x_wec, x_opt, waves):`, and returns
            forces in the time-domain of size
            :python:`2*nfreq + 1 x ndof`.
        constraints
            List of constraints, see documentation for
            :py:func:`scipy.optimize.minimize` for description and
            options of constraints dictionaries.
            If :python:`None`: empty list :python:`[]`.
        min_damping
            Minimum damping level to ensure a stable system.
            See `check_linear_damping` for more details.
        wave_directions
            List of wave directions [degrees] to evaluate BEM at.
        rho
            Water density in :math:`kg/m^3`.
        g
            Gravitational acceleration in :math:`m/s^2`.
        depth
            Water depth in :math:`m`.

        Returns
        -------
        WEC
            An instance of the :py:class:`wecopttool.core.WEC` class.

        See Also
        --------
        run_bem, write_netcdf, WEC.from_bem
        """

        # RUN BEM
        _log.info(f"Running Capytaine (BEM): {nfreq+1} frequencies x " +
                 f"{len(wave_directions)} wave directions.")
        freq = frequency(f1, nfreq)[1:]
        bem_data = run_bem(
            fb, freq, wave_directions, rho=rho, g=g, depth=depth)
        wec = WEC.from_bem(
            bem_data, inertia_matrix, hydrostatic_stiffness, friction, f_add,
            constraints, min_damping=min_damping)
        return wec

    @staticmethod
    def from_impedance(
        freqs: ArrayLike,
        impedance: ArrayLike,
        exc_coeff: ArrayLike,
        hydrostatic_stiffness: ndarray,
        f_add: Optional[TIForceDict] = None,
        constraints: Optional[Iterable[Mapping]] = None,
    ) -> TWEC:
        """Create a WEC object from the intrinsic impedance and
        excitation coefficients.

        The intrinsic (mechanical) impedance :math:`Z(ω)` linearly
        relates excitation forces :math:`F(ω)` to WEC velocity
        :math:`U(ω)` as :math:`ZU=F`.
        Using linear hydrodynamic coefficients, e.g. from a BEM code
        like Capytaine, the impedance is given as
        :math:`Z(ω) = (m+A(ω))*iω + B(ω) + B_f + K/(iω)`.
        The impedance can also be obtained experimentally.
        Note that the impedance is not defined at :math:`ω=0`.


        Parameters
        ----------
        freqs
            Frequency vector [Hz] not including the zero frequency,
            :python:`freqs = [f1, 2*f1, ..., nfreq*f1]`.
        impedance
            Complex impedance of size :python:`ndof x ndof x nfreq`.
        exc_coeff
            Complex excitation transfer function of size
            :python:`ndof x nfreq`.
        hydrostatic_stiffness
            Linear hydrostatic restoring coefficient of size
            :python:`nodf x ndof`.
        f_add
            Dictionary with entries :python:`{'force_name': fun}`, where
            :python:`fun` has a  signature
            :python:`def fun(wec, x_wec, x_opt, waves):`, and returns
            forces in the time-domain of size
            :python:`2*nfreq + 1 x ndof`.
        constraints
            List of constraints, see documentation for
            :py:func:`scipy.optimize.minimize` for description and
            options of constraints dictionaries.
            If :python:`None`: empty list :python:`[]`.

        Raises
        ------
        ValueError
            If :python:`impedance` does not have the correct size:
            :python:`ndof x ndof x nfreq`.
        """
        f1, nfreq = frequency_parameters(freqs, False)

        # impedance matrix shape
        shape = impedance.shape
        ndim = impedance.ndim
        if (ndim!=3) or (shape[0]!=shape[1]) or (shape[2]!=nfreq):
            raise ValueError(
                "`impedance` must have shape `ndof x ndof x (nfreq)`.")

        # impedance force
        omega = freqs * 2*np.pi
        transfer_func = impedance * (1j*omega)
        transfer_func0 = np.expand_dims(hydrostatic_stiffness, 2)
        transfer_func = np.concatenate([transfer_func0, transfer_func], 2)
        transfer_func = -1 * transfer_func  # RHS of equation: ma = Σf
        force_impedance = force_from_rao_transfer_function(transfer_func)

        # excitation force
        force_excitation = force_from_waves(exc_coeff)

        # all forces
        f_add = {} if (f_add is None) else f_add
        forces =  {
            'intrinsic_impedance': force_impedance,
            'excitation': force_excitation
        }
        forces = forces | f_add

        # wec
        wec = WEC(f1, nfreq, forces, constraints,
                  inertia_in_forces=True, ndof=shape[0])
        return wec

    # solve
    def solve(self,
        waves: Dataset,
        obj_fun: TStateFunction,
        nstate_opt: int,
        x_wec_0: Optional[ndarray] = None,
        x_opt_0: Optional[ndarray] = None,
        scale_x_wec: Optional[list] = None,
        scale_x_opt: Optional[FloatOrArray] = 1.0,
        scale_obj: Optional[float] = 1.0,
        optim_options: Optional[Mapping[str, Any]] = {},
        use_grad: Optional[bool] = True,
        maximize: Optional[bool] = False,
        bounds_wec: Optional[Bounds] = None,
        bounds_opt: Optional[Bounds] = None,
        callback: Optional[Callable[[ndarray]]] = None,
        ) -> tuple[Dataset, Dataset, OptimizeResult]:
        """Simulate WEC dynamics using a pseudo-spectral solution
        method.

        Parameters
        ----------
        waves
            :py:class:`xarray.Dataset` with the structure and elements
            shown by :py:mod:`wecopttool.waves`.
        obj_fun
            Objective function to minimize for pseudo-spectral solution,
            must have signature :python:`fun(wec, x_wec, x_opt, waves)`
            and return a scalar.
        nstate_opt
            Length of the optimization (controls) state vector.
        x_wec_0
            Initial guess for the WEC dynamics state.
            If :python:`None` it is randomly initiated.
        x_opt_0
            Initial guess for the optimization (control) state.
            If :python:`None` it is randomly initiated.
        scale_x_wec
            Factor(s) to scale each DOF in :python:`x_wec` by, to
            improve convergence.
            A single float or an array of size :python:`ndof`.
        scale_x_opt
            Factor(s) to scale :python:`x_opt` by, to improve
            convergence.
            A single float or an array of size :python:`nstate_opt`.
        scale_obj
            Factor to scale :python:`obj_fun` by, to improve
            convergence.
        optim_options
            Optimization options passed to the optimizer.
            See :py:func:`scipy.optimize.minimize`.
        use_grad
             If :python:`True`, optimization with utilize
             :python:`autograd` for gradients.
        maximize
            Whether to maximize the objective function.
            The default is to minimize the objective function.
        bounds_wec
            Bounds on the WEC components of the decision variable.
            See :py:func:`scipy.optimize.minimize`.
        bounds_opt
            Bounds on the optimization (control) components of the
            decision variable.
            See :py:func:`scipy.optimize.minimize`.
        callback
            Function called after each iteration.
            See :py:func:`scipy.optimize.minimize`.
            The default is reported via logging at the INFO level.

        Returns
        -------
        res_fd
            Dynamic responses in the frequency-domain.
        res_td
            Dynamic responses in the time-domain.
        res
            Results produced by :py:func:`scipy.optimize.minimize`.

        Raises
        ------
        ValueError
            If :python:`scale_x_opt` is a scalar and
            :python:`nstate_opt` is not provided.
        Exception
            If the optimizer fails for any reason other than maximum
            number of states, i.e. for exit modes other than 0 or 9.
            See :py:mod:`scipy.optimize` for exit mode details.

        See Also
        --------
        wecopttool.waves,
        """

        _log.info("Solving pseudo-spectral control problem.")

        # x_wec scaling vector
        if scale_x_wec == None:
            scale_x_wec = [1.0] * self.ndof
        elif isinstance(scale_x_wec, float) or isinstance(scale_x_wec, int):
            scale_x_wec = [scale_x_wec] * self.ndof
        scale_x_wec = scale_dofs(scale_x_wec, self.ncomponents)

        # x_opt scaling vector
        if isinstance(scale_x_opt, float) or isinstance(scale_x_opt, int):
            if nstate_opt is None:
                raise ValueError("If 'scale_x_opt' is a scalar, " +
                                    "'nstate_opt' must be provided")
            scale_x_opt = scale_dofs([scale_x_opt], nstate_opt)

        # composite scaling vector
        scale = np.concatenate([scale_x_wec, scale_x_opt])

        # decision variable initial guess
        if x_wec_0 is None:
            x_wec_0 = np.random.randn(self.nstate_wec)
        if x_opt_0 is None:
            x_opt_0 = np.random.randn(nstate_opt)
        x0 = np.concatenate([x_wec_0, x_opt_0])*scale

        # objective function
        sign = -1.0 if maximize else 1.0

        def obj_fun_scaled(x):
            x_wec, x_opt = self.decompose_state(x/scale)
            return obj_fun(self, x_wec, x_opt, waves)*scale_obj*sign

        # constraints
        constraints = self.constraints.copy()

        for i, icons in enumerate(self.constraints):
            icons_new = {"type": icons["type"]}

            def make_new_fun(icons):
                def new_fun(x):
                    x_wec, x_opt = self.decompose_state(x/scale)
                    return icons["fun"](self, x_wec, x_opt, waves)
                return new_fun

            icons_new["fun"] = make_new_fun(icons)
            if use_grad:
                icons_new['jac'] = jacobian(icons_new['fun'])
            constraints[i] = icons_new

        # system dynamics through equality constraint, ma - Σf = 0
        def resid_fun(x):
            x_s = x/scale
            x_wec, x_opt = self.decompose_state(x_s)
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

        # callback
        if callback is None:
            def callback(x):
                x_wec, x_opt = self.decompose_state(x)
                _log.info("[max(x_wec), max(x_opt), obj_fun(x)]: "
                          + f"[{np.max(np.abs(x_wec)):.2e}, "
                          + f"{np.max(np.abs(x_opt)):.2e}, "
                          + f"{np.max(obj_fun_scaled(x)):.2e}]")

        # optimization problem
        optim_options['disp'] = optim_options.get('disp', True)
        problem = {'fun': obj_fun_scaled,
                    'x0': x0,
                    'method': 'SLSQP',
                    'constraints': constraints,
                    'options': optim_options,
                    'bounds': bounds,
                    'callback':callback,  # TODO: allow callback functions to take (wec, x_wec, x_opt, waves) as arguments not x
                    }
        if use_grad:
            problem['jac'] = grad(obj_fun_scaled)

        # minimize
        optim_res = minimize(**problem)

        msg = f'{optim_res.message}    (Exit mode {optim_res.status})'
        if optim_res.status == 0:
            _log.info(msg)
        elif optim_res.status == 9:
            _log.warning(msg)
        else:
            raise Exception(msg)

        # unscale
        optim_res.x = optim_res.x / scale
        optim_res.fun = optim_res.fun / scale_obj
        # TODO: unscale all the other fields in the results, e.g. 'jac'

        return optim_res


    def post_process(self,
        res: OptimizeResult,
        waves: Dataset,
        nsubsteps: Optional[int] = 1,
    ) -> tuple[Dataset, Dataset]:
        """Post-process the results from :python:`WEC.solve`.

        Parameters
        ----------
        waves
            :py:class:`xarray.Dataset` with the structure and elements
            shown by :py:mod:`wecopttool.waves`.
        res
            Results produced by :py:func:`scipy.optimize.minimize`.
        nsubsteps
            Number of steps between the default (implied) time steps.
            A value of :python:`1` corresponds to the default step
            length.

        Returns
        -------
        results_fd
            Dynamic responses in the frequency-domain.
        results_td
            Dynamic responses in the time-domain.
        """
        create_time = f"{datetime.utcnow()}"

        pos_attr = {'long_name': 'Position', 'units': 'm or rad'}
        vel_attr = {'long_name': 'Velocity', 'units': 'm/s or rad/s'}
        acc_attr = {'long_name': 'Acceleration', 'units': 'm/s^2 or rad/s^2'}
        omega_attr = {'long_name': 'Radial frequency', 'units': 'rad/s'}
        freq_attr = {'long_name': 'Frequency', 'units': 'Hz'}
        period_attr = {'long_name': 'Period', 'units': 's'}
        time_attr = {'long_name': 'Time', 'units': 's'}
        dof_attr = {'long_name': 'Degree of freedom'}
        force_attr = {'long_name': 'Force or moment', 'units': 'N or Nm'}
        wave_elev_attr = {'long_name': 'Wave elevation', 'units': 'm'}
        x_wec, x_opt = self.decompose_state(res.x)
        
        omega_coord = ("omega", self.omega, omega_attr)
        freq_coord = ("omega", self.frequency, freq_attr)
        period_coord = ("omega", self.period, period_attr)
        dof_coord = ("influenced_dof", self.dof_names, dof_attr)
        
        # frequency domain
        force_da_list = []
        for name, force in self.forces.items():
            force_td_tmp = force(self, x_wec, x_opt, waves)
            force_fd = self.td_to_fd(force_td_tmp, fft=True)
            force_da = DataArray(data=force_fd,
                                 dims=["omega", "influenced_dof"],
                                 coords={
                                     'omega': omega_coord,
                                     'freq': freq_coord, 
                                     'period': period_coord,
                                     'influenced_dof': dof_coord},
                                 attrs=force_attr
                                 ).expand_dims({'type': [name]})
            force_da_list.append(force_da)

        fd_forces = xr.concat(force_da_list, dim='type')
        fd_forces.type.attrs['long_name'] = 'Type'
        fd_forces.name = 'force'
        fd_forces.attrs['long_name'] = 'Force'

        pos = self.vec_to_dofmat(x_wec)
        pos_fd = real_to_complex(pos)

        vel = self.derivative_mat @ pos
        vel_fd = real_to_complex(vel)

        acc = self.derivative_mat @ vel
        acc_fd = real_to_complex(acc)

        fd_state = Dataset(
            data_vars={
                'pos': (['omega', 'influenced_dof'], pos_fd, pos_attr),
                'vel': (['omega', 'influenced_dof'], vel_fd, vel_attr),
                'acc': (['omega', 'influenced_dof'], acc_fd, acc_attr)},
            coords={
                'omega': omega_coord,
                'freq': freq_coord,
                'period': period_coord,
                'influenced_dof': dof_coord},
            attrs={"time_created_utc": create_time}
        )

        results_fd = xr.merge([fd_state, fd_forces, waves])
        results_fd = results_fd.transpose('omega', 'influenced_dof', 'type',
                                          'wave_direction')
        results_fd = results_fd.fillna(0)

        # time domain
        t_dat = self.time_nsubsteps(nsubsteps)
        time = DataArray(
            data=t_dat, name='time', dims='time', coords=[t_dat])
        results_td = results_fd.map(lambda x: time_results(x, time))

        results_td['pos'].attrs = pos_attr
        results_td['vel'].attrs = vel_attr
        results_td['acc'].attrs = acc_attr
        results_td['wave_elev'].attrs = wave_elev_attr
        results_td['force'].attrs = force_attr
        results_td['time'].attrs = time_attr
        results_td.attrs['time_created_utc'] = create_time

        return results_fd, results_td

    # properties
    @property
    def forces(self) -> TForceDict:
        """Dictionary of forces."""
        return self._forces

    @forces.setter
    def forces(self, val):
        self._forces = dict(val)

    @property
    def constraints(self) -> list[dict]:
        """List of constraints."""
        return self._constraints

    @constraints.setter
    def constraints(self, val):
        self._constraints = list(val)

    @property
    def inertia_in_forces(self) -> bool:
        """Whether inertial "forces" are included in the
        :python:`forces` dictionary.
        """
        return self._inertia_in_forces

    @property
    def inertia_matrix(self) -> ndarray:
        """Inertia (mass) matrix.
        :python:`None` if  :python:`inertia_in_forces is True`.
        """
        return self._inertia_matrix

    @property
    def inertia(self) -> TStateFunction:
        """Function representing the inertial term :math:`ma` in the
        WEC's dynamics equation.
        """
        return self._inertia

    @property
    def dof_names(self) -> list[str]:
        """Names of the different degrees of freedom."""
        return self._dof_names

    @property
    def ndof(self) -> int:
        """Number of degrees of freedom."""
        return self._ndof

    @property
    def frequency(self) -> ndarray:
        """Frequency vector [Hz]."""
        return self._freq

    @property
    def f1(self) -> float:
        """Fundamental frequency :python:`f1` [Hz]."""
        return self._freq[1]

    @property
    def nfreq(self) -> int:
        """Number of frequencies, not including the zero-frequency."""
        return len(self._freq)-1

    @property
    def omega(self) -> ndarray:
        """Radial frequency vector [rad/s]."""
        return self._freq * (2*np.pi)
    
    @property
    def period(self) -> ndarray:
        """Period vector [s]."""
        return np.concatenate([[np.Infinity], 1/self._freq[1:]])

    @property
    def w1(self) -> float:
        """Fundamental radial frequency [rad/s]."""
        return self.omega[1]

    @property
    def time(self) -> ndarray:
        """Time vector [s], size `2*nfreq+1 x ndof`, not containing the
        end time `tf`."""
        return self._time

    @property
    def time_mat(self) -> ndarray:
        """Matrix to create time-series from Fourier coefficients.

        For some array of Fourier coefficients :python:`x`, size
        :python:`2*nfreq+1 x ndof`, the time series, also size
        :python:`2*nfreq+1 x ndof`, is obtained as
        :python:`time_mat @ x`.
        """
        return self._time_mat

    @property
    def derivative_mat(self) -> ndarray:
        """Matrix to create Fourier coefficients of the derivative of
        some quantity.

        For some array of Fourier coefficients :python:`x`, size
        :python:`2*nfreq+1 x ndof`, the Fourier coefficients of the
        derivative of :python:`x` are obtained as
        :python:`derivative_mat @ x`.
        """
        return self._derivative_mat

    @property
    def dt(self) -> float:
        """Time spacing [s]."""
        return self._time[1]

    @property
    def tf(self) -> float:
        """Final time (repeat period) [s]. Not included in
        :python:`time` vector.
        """
        return 1/self.f1

    @property
    def nt(self) -> int:
        """Number of timesteps."""
        return self.ncomponents

    @property
    def ncomponents(self) -> int:
        """Number of Fourier components (:python:`2*nfreq + 1`) for each
        degree of freedom.
        """
        return ncomponents(self.nfreq)

    @property
    def nstate_wec(self) -> int:
        """Length of the WEC dynamics state vector consisting of the
        Fourier coefficient of the position of each degree of freedom.
        """
        return self.ndof * self.ncomponents

    # other methods
    def decompose_state(self,
        state: ndarray
    ) -> tuple[ndarray, ndarray]:
        """Split the state vector into the WEC dynamics state and the
        optimization (control) state.

        Calls :py:meth:`wecopttool.core.decompose_state` with the
        appropriate inputs for the WEC object.

        Examples
        --------
        >>> x_wec, x_opt = wec.decompose_state(x)

        Parameters
        ----------
        state
            Combined WEC and optimization states.

        Returns
        -------
        state_wec
            WEC state vector.
        state_opt
            Optimization (control) state.

        See Also
        --------
        decompose_state
        """
        return decompose_state(state, self.ndof, self.nfreq)

    def time_nsubsteps(self, nsubsteps: int) -> ndarray:
        """Create a time vector with finer discretization.

        Calls :py:func:`wecopttool.core.time` with the appropriate
        inputs for the WEC object.

        Parameters
        ----------
        nsubsteps
            Number of substeps between implied/default time steps.

        See Also
        --------
        time, WEC.time
        """
        return time(self.f1, self.nfreq, nsubsteps)

    def time_mat_nsubsteps(self, nsubsteps: int) -> ndarray:
        """Create a time matrix similar to :python:`WEC.time_mat` but
        with finer time-domain discretization.

        Calls :py:func:`wecopttool.core.time_mat` with the appropriate
        inputs for the WEC object.

        Parameters
        ----------
        nsubsteps
            Number of substeps between implied/default time steps.

        See Also
        --------
        time_mat, WEC.time_mat, WEC.time_nsubsteps
        """
        return time_mat(self.f1, self.nfreq, nsubsteps)

    def vec_to_dofmat(self, vec: ndarray) -> ndarray:
        """Convert a vector to a matrix with one column per degree of
        freedom.

        Opposite of :py:meth:`wecopttool.core.WEC.dofmat_to_vec`.

        Calls :py:func:`wecopttool.core.vec_to_dofmat` with the
        appropriate inputs for the WEC object.

        Examples
        --------
        >>> x_wec, x_opt = wec.decompose_state(x)
        >>> x_wec_mat = wec.vec_to_dofmat(x_wec)

        Parameters
        ----------
        vec
            One-dimensional vector.

        See Also
        --------
        vec_to_dofmat, WEC.dofmat_to_vec
        """
        return vec_to_dofmat(vec, self.ndof)

    def dofmat_to_vec(self, mat: ndarray) -> ndarray:
        """Flatten a matrix to a vector.

        Opposite of :py:meth:`wecopttool.core.WEC.vec_to_dofmat`.

        Calls :py:func:`wecopttool.core.dofmat_to_vec` with the
        appropriate inputs for the WEC object.

        Parameters
        ----------
        mat
            Matrix with one column per degree of freedom.

        See Also
        --------
        dofmat_to_vec, WEC.vec_to_dofmat
        """
        return dofmat_to_vec(mat)

    def fd_to_td(self, fd: ndarray) -> ndarray:
        """Convert a frequency-domain array to time-domain.

        Opposite of :meth:`wecopttool.core.WEC.td_to_fd`.

        Calls :python:`wecopttool.fd_to_td` with the appropriate inputs
        for the WEC object.

        Parameters
        ----------
        fd
            Frequency-domain complex array with shape `WEC.nfreq+1 x N`
            for any `N`.

        See Also
        --------
        fd_to_td, WEC.td_to_fd
        """
        return fd_to_td(fd, self.f1, self.nfreq, True)

    def td_to_fd(
        self,
        td: ndarray,
        fft: Optional[bool] = True,
        ) -> ndarray:
        """Convert a time-domain array to frequency-domain.

        Opposite of :meth:`wecopttool.core.WEC.fd_to_td`.

        Calls :python:`wecopttool.fd_to_td` with the appropriate inputs
        for the WEC object.

        Parameters
        ----------
        td
            Time-domain real array with shape
            :python:`2*WEC.nfreq+1 x N` for any :python:`N`.
        fft
            Whether to use the real FFT.

        See Also
        --------
        td_to_fd, WEC.fd_to_td
        """
        return td_to_fd(td, fft, True)


def ncomponents(
    nfreq : int,
    zero_freq: Optional[bool] = True,
) -> int:
    """Number of Fourier components (:python:`2*nfreq + 1`) for each
    DOF.

    Parameters
    ----------
    nfreq
        Number of frequencies.
    zero_freq
        Whether to include the zero-frequency.
    """
    ncomp = 2*nfreq
    if zero_freq:
        ncomp = ncomp + 1
    return ncomp


def frequency(
    f1: float,
    nfreq: int,
    zero_freq: Optional[bool] = True,
) -> ndarray:
    """Construct equally spaced frequency array.

    The array includes :python:`0` and has length of :python:`nfreq+1`.
    :python:`f1` is fundamental frequency (1st harmonic).

    Returns the frequency array, e.g.,
    :python:`freqs = [0, f1, 2*f1, ..., nfreq*f1]`.

    Parameters
    ----------
    f1
        Fundamental frequency :python:`f1` [Hz].
    nfreq
        Number of frequencies.
    zero_freq
        Whether to include the zero-frequency.
    """
    freq = np.arange(0, nfreq+1)*f1
    freq = freq[1:] if not zero_freq else freq
    return freq


def time(
    f1: float,
    nfreq: int,
    nsubsteps: Optional[int] = 1,
) -> ndarray:
    """Assemble the time vector with :python:`nsubsteps` subdivisions.

    Returns the 1D time vector, in seconds, starting at time
    :python:`0`, and not containing the end time :python:`tf=1/f1`.
    The time vector has length :python:`(2*nfreq+1)*nsubsteps`.
    The timestep length is :python:`dt = dt_default * 1/nsubsteps`,
    where :python:`dt_default=tf/(2*nfreq+1)`.

    Parameters
    ----------
    f1
        Fundamental frequency :python:`f1` [Hz].
    nfreq
        Number of frequencies.
    nsubsteps
        Number of steps between the default (implied) time steps.
        A value of :python:`1` corresponds to the default step length.
    """
    if nsubsteps < 1:
        raise ValueError("`nsubsteps` must be 1 or greater")
    nsteps = nsubsteps * ncomponents(nfreq)
    return np.linspace(0, 1/f1, nsteps, endpoint=False)


def time_mat(
    f1: float,
    nfreq: int,
    nsubsteps: Optional[int] = 1,
    zero_freq: Optional[bool] = True,
) -> ndarray:
    """Assemble the time matrix that converts the state to a
    time-series.

    For a state :python:`x` consisting of the mean (DC) component
    followed by the real and imaginary components of the Fourier
    coefficients as
    :python:`x=[X0, Re(X1), Im(X1), ..., Re(Xn), Im(Xn)]`,
    the response vector in the time-domain is given as
    :python:`x(t)=Mx`, where :python:`M` is the time matrix.

    The time matrix has size :python:`(nfreq*2+1) x (nfreq*2+1)`.

    Parameters
    ---------
    f1
        Fundamental frequency :python:`f1` [Hz].
    nfreq
        Number of frequencies.
    nsubsteps
        Number of steps between the default (implied) time steps.
        A value of :python:`1` corresponds to the default step length.
    zero_freq
        Whether the first frequency should be zero.
    """
    t = time(f1, nfreq, nsubsteps)
    omega = frequency(f1, nfreq) * 2*np.pi
    wt = np.outer(t, omega[1:])
    ncomp = ncomponents(nfreq)
    time_mat = np.empty((nsubsteps*ncomp, ncomp))
    time_mat[:, 0] = 1.0
    time_mat[:, 1::2] = np.cos(wt)
    time_mat[:, 2::2] = -np.sin(wt)
    if not zero_freq:
        time_mat = time_mat[:, 1:]
    return time_mat


def derivative_mat(
    f1: float,
    nfreq: int,
    zero_freq: Optional[bool] = True,
) -> ndarray:
    """Assemble the derivative matrix that converts the state vector of
    a response to the state vector of its derivative.

    For a state :python:`x` consisting of the mean (DC) component
    followed by the real and imaginary components of the Fourier
    coefficients as
    :python:`x=[X0, Re(X1), Im(X1), ..., Re(Xn), Im(Xn)]`,
    the state of its derivative is given as :python:`x(t)=Dx`, where
    :python:`D` is the derivative matrix.

    The derivative matrix has size :python:`(nfreq*2+1) x (nfreq*2+1)`.

    Parameters
    ---------
    f1
        Fundamental frequency :python:`f1` [Hz].
    nfreq
        Number of frequencies.
    zero_freq
        Whether the first frequency should be zero.
    """
    def block(n): return np.array([[0, -1], [1, 0]]) * n*f1 * 2*np.pi
    blocks = [block(n+1) for n in range(nfreq)]
    if zero_freq:
        blocks = [0.0] + blocks
    return block_diag(*blocks)


def degrees_to_radians(
    degrees: FloatOrArray,
    sort: Optional[bool] = True,
) -> Union[float, ndarray]:
    """Convert a 1D array of angles in degrees to radians in the range
    :math:`(-π, π]` and optionally sort them.

    Parameters
    ----------
    degrees
        1D array of angles in degrees.
    sort
        Whether to sort the angles from smallest to largest in
        :math:`(-π, π]`.
    """
    radians = np.asarray(np.remainder(np.deg2rad(degrees), 2*np.pi))
    radians[radians > np.pi] -= 2*np.pi
    if radians.size == 1:
        radians = radians.item()
    elif sort:
        radians = np.sort(radians)
    return radians


def vec_to_dofmat(vec: ArrayLike, ndof: int) -> ndarray:
    """Convert a vector back to a matrix with one column per DOF.

    Returns a matrix with :python:`ndof` columns.
    The number of rows is inferred from the size of the input vector.

    Opposite of :py:func:`wecopttool.core.dofmat_to_vec`.

    Parameters
    ----------
    vec
        1D array consisting of concatenated arrays of several DOFs, as
        :python:`vec = [vec_1, vec_2, ..., vec_ndof]`.
    ndof
        Number of degrees of freedom.

    See Also
    --------
    dofmat_to_vec,
    """
    return np.reshape(vec, (-1, ndof), order='F')


def dofmat_to_vec(mat: ArrayLike) -> ndarray:
    """Flatten a matrix that has one column per DOF.

    Returns a 1D vector.

    Opposite of :py:func:`wecopttool.core.vec_to_dofmat`.

    Parameters
    ----------
    mat
        Matrix to be flattened.

    See Also
    --------
    vec_to_dofmat,
    """
    return np.reshape(mat, -1, order='F')


def mimo_transfer_mat(
    transfer_mat: ArrayLike,
    zero_freq: Optional[bool] = True,
) -> ndarray:
    """Create a block matrix of the MIMO transfer function.

    The input is a complex transfer matrix that relates the complex
    Fourier representation of two variables.
    For example, it can be an impedance matrix or an RAO transfer
    matrix.
    The input complex impedance matrix has shape
    :python`ndof x ndof (nfreq)`.

    Returns the 2D real matrix that transform the state representation
    of the input variable variable to the state representation of the
    output variable.
    Here, a state representation :python:`x` consists of the mean (DC)
    component followed by the real and imaginary components of the
    Fourier coefficients as
    :python:`x=[X0, Re(X1), Im(X1), ..., Re(Xn), Im(Xn)]`.

    Parameters
    ----------
    transfer_mat
        Complex transfer matrix.
    zero_freq
        Whether the first frequency should be zero.
    """
    ndof = transfer_mat.shape[0]
    assert transfer_mat.shape[1] == ndof
    elem = [[None]*ndof for _ in range(ndof)]
    def block(re, im): return np.array([[re, -im], [im, re]])
    for idof in range(ndof):
        for jdof in range(ndof):
            if zero_freq:
                Zp0 = transfer_mat[idof, jdof, 0]
                assert np.all(np.isreal(Zp0))
                Zp0 = np.real(Zp0)
                Zp = transfer_mat[idof, jdof, 1:]
            else:
                Zp0 = [0.0]
                Zp = transfer_mat[idof, jdof, :]
            re = np.real(Zp)
            im = np.imag(Zp)
            blocks = [block(ire, iim) for (ire, iim) in zip(re, im)]
            blocks =[Zp0] + blocks
            elem[idof][jdof] = block_diag(*blocks)
    return np.block(elem)


def real_to_complex(
    fd: ArrayLike,
    zero_freq: Optional[bool] = True,
) -> ndarray:
    """Convert from two real amplitudes to one complex amplitude per
    frequency.

    The input is a real 2D array with each column containing the real
    and imaginary components of the Fourier coefficients for some
    response.
    The column length is :python:`2*nfreq+1`.
    The entries of a column representing a response :python:`x` are
    :python:`x=[X0, Re(X1), Im(X1), ..., Re(Xn), Im(Xn)]`.

    Returns a complex 2D array with each column containing the complex
    Fourier coefficients.
    Columns are length :python:`nfreq+1`, and the first row corresponds
    to the real-valued zero-frequency (mean, DC) components.
    The entries of a column representing a response :python:`x` are
    :python:`x=[X0, X1, ..., Xn]`.

    Parameters
    ----------
    fd
        Array containing the real and imaginary components of the
        Fourier coefficients.
    zero_freq
        Whether the mean (DC) component is included.

    See Also
    --------
    complex_to_real,
    """
    fd= atleast_2d(fd)
    if zero_freq:
        assert fd.shape[0]%2==1
        mean = fd[0:1, :]
        fd = fd[1:, :]
    fdc = fd[0::2, :] + 1j*fd[1::2, :]
    if zero_freq:
        fdc = np.concatenate((mean, fdc), axis=0)
    return fdc


def complex_to_real(
    fd: ArrayLike,
    zero_freq: Optional[bool] = True,
) -> ndarray:
    """Convert from one complex amplitude to two real amplitudes per
    frequency.

    The input is a complex 2D array with each column containing the
    Fourier coefficients for some response.
    Columns are length :python:`nfreq+1`, and the first row corresponds
    to the real-valued zero-frequency (mean, DC) components.
    The entries of a column representing a response :python:`x` are
    :python:`x=[X0, X1, ..., Xn]`.

    Returns a real 2D array with each column containing the real and
    imaginary components of the Fourier coefficients.
    The column length is :python:`2*nfreq+1`.
    The entries of a column representing a response :python:`x` are
    :python:`x=[X0, Re(X1), Im(X1), ..., Re(Xn), Im(Xn)]`.

    Parameters
    ----------
    fd
        Array containing the complex Fourier coefficients.
    zero_freq
        Whether the mean (DC) component is included.

    See Also
    --------
    real_to_complex,
    """
    fd = atleast_2d(fd)
    nfreq = fd.shape[0] - 1 if zero_freq else fd.shape[0]
    ndof = fd.shape[1]
    if zero_freq:
        assert np.all(np.isreal(fd[0, :]))
        a = np.real(fd[0:1, :])
        b = np.real(fd[1:, :])
        c = np.imag(fd[1:, :])
    else:
        b = np.real(fd)
        c = np.imag(fd)
    out = np.concatenate([np.transpose(b), np.transpose(c)])
    out = np.reshape(np.reshape(out, [-1], order='F'), [-1, ndof])
    if zero_freq:
        out = np.concatenate([a, out])
        assert out.shape == (2*nfreq+1, ndof)
    else:
        assert out.shape == (2*nfreq, ndof)
    return out


def fd_to_td(
    fd: ArrayLike,
    f1: Optional[float] = None,
    nfreq: Optional[int] = None,
    zero_freq: Optional[bool] = True,
) -> ndarray:
    """Convert a complex array of Fourier coefficients to a real array
    of time-domain responses.

    The input is a complex 2D array with each column containing the
    Fourier coefficients for some response.
    Columns are length :python:`nfreq+1`, and the first row corresponds
    to the real-valued zero-frequency (mean, DC) components.
    The entries of a column representing a response :python:`x` are
    :python:`x=[X0, X1, ..., Xn]`.

    Returns a real array with same number of columns and
    :python:`2*nfreq+1` rows, containing the time-domain response at
    times :python:`wecopttool.time(f1, nfreq, nsubsteps=1)`.

    If both :python:`f1` and :python:`nfreq` are provided, it uses the
    time matrix :python:`wecopttool.time_mat(f1, nfreq, nsubsteps=1)`,
    else it uses the inverse real FFT (:python:`numpy.fft.irfft`).

    Opposite of :meth:`wecopttool.core.td_to_fd`.

    Parameters
    ----------
    fd
        Array containing the complex Fourier coefficients.
    f1
        Fundamental frequency :python:`f1` [Hz].
    nfreq
        Number of frequencies.
    zero_freq
        Whether the mean (DC) component is included.

    Raises
    ------
    ValueError
        If only one of :python:`f1` or :python:`nfreq` is provided.
        Must provide both or neither.

    See Also
    --------
    td_to_fd, time, time_mat
    """
    fd = atleast_2d(fd)
    if (f1 is not None) and (nfreq is not None):
        tmat = time_mat(f1, nfreq)
        if not zero_freq:
            tmat = tmat[:, 1:]
        td = tmat @ complex_to_real(fd, zero_freq)
    elif (f1 is None) and (nfreq is None):
        n = 1 + 2*(fd.shape[0]-1)
        td = np.fft.irfft(fd/2, n=n, axis=0, norm='forward')
    else:
        raise ValueError(
            "Provide either both `f1` and `nfreq` or neither.")
    return td


def td_to_fd(
    td: ArrayLike,
    fft: Optional[bool] = True,
    zero_freq: Optional[bool] = True,
) -> ndarray:
    """Convert a real array of time-domain responses to a complex array
    of Fourier coefficients.

    Opposite of :meth:`wecopttool.core.fd_to_td`

    Parameters
    ----------
    td
        Real array of time-domains responses.
    fft
        Whether to use the real FFT.
    zero_freq
        Whether the mean (DC) component is returned.

    See Also
    --------
    fd_to_td
    """
    td= atleast_2d(td)
    n = td.shape[0]
    if fft:
        fd = np.fft.rfft(td*2, n=n, axis=0, norm='forward')
    else:
        fd = np.dot(dft(n, 'n')[:n//2+1, :], td*2)
    if not zero_freq:
        fd = fd[1:, :]
    return fd


def wave_excitation(exc_coeff: Dataset, waves: Dataset) -> ndarray:
    """Calculate the complex, frequency-domain, excitation force due to
    waves.

    The resulting force is indexed only by frequency and not direction
    angle.
    The input :python:`waves` frequencies must be same as
    :python:`exc_coeff`, but the directions can be a subset.

    Parameters
    ----------
    exc_coeff
        Complex excitation coefficients indexed by frequency and
        direction angle.
    waves
        Complex frequency-domain wave elevation.

    Raises
    ------
    ValueError
        If the frequency vectors of :python:`exc_coeff` and
        :python:`waves` are different.
    ValueError
        If any of the directions in :python:`waves` is not in
        :python:`exc_coeff`.
    """
    omega_w = waves['omega'].values
    omega_e = exc_coeff['omega'].values
    dir_w = waves['wave_direction'].values
    dir_e = exc_coeff['wave_direction'].values
    exc_coeff = exc_coeff.transpose(
        'omega', 'wave_direction', 'influenced_dof').values

    wave_elev_fd = np.expand_dims(waves.values, -1)

    if not np.allclose(omega_w, omega_e):
        raise ValueError(f"Wave and excitation frequencies do not match. WW: {omega_w}, EE: {omega_e}")

    subset, sub_ind = subset_close(dir_w, dir_e)

    if not subset:
        raise ValueError(
            "Some wave directions are not in excitation coefficients " +
            f"\n Wave direction(s): {(np.rad2deg(dir_w))} (deg)" +
            f"\n BEM direction(s): {np.rad2deg(dir_e)} (deg).")

    return np.sum(wave_elev_fd*exc_coeff[:, sub_ind, :], axis=1)


def read_netcdf(fpath: Union[str, Path]) -> Dataset:
    """Read a *NetCDF* file with possibly complex entries as a
    :py:class:`xarray.Dataset`.

    Can handle complex entries in the *NetCDF* by using
    :python:`capytaine.io.xarray` utilities.

    Parameters
    ----------
    fpath
        Path to the *NetCDF* file.

    See Also
    --------
    write_netcdf,
    """
    with xr.open_dataset(fpath) as ds:
        ds.load()
    return cpy.io.xarray.merge_complex_values(ds)


def write_netcdf(fpath: Union[str, Path], data: Dataset) -> None:
    """Save an :py:class:`xarray.Dataset` with possibly complex entries as a
    *NetCDF* file.

    Can handle complex entries in the *NetCDF* by using
    :python:`capytaine.io.xarray` utilities.

    Parameters
    ----------
    fpath
        Name of file to save.
    data
        Dataset to save.

    See Also
    --------
    read_netcdf,
    """
    cpy.io.xarray.separate_complex_values(data).to_netcdf(fpath)


def check_linear_damping(
    hydro_data: Dataset,
    min_damping: Optional[float] = 1e-6,
) -> Dataset:
    """Ensure that the linear hydrodynamics (friction + radiation
    damping) have positive damping.

    Shifts the :python:`friction` up if necessary.
    Returns the (possibly) updated Dataset with
    :python:`damping >= min_damping`.

    Parameters
    ----------
    hydro_data
        Linear hydrodynamic data.
    min_damping
        Minimum threshold for damping. Default is 1e-6.
    """
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


def force_from_rao_transfer_function(
    rao_transfer_mat: ArrayLike,
    zero_freq: Optional[bool] = True,
) -> TStateFunction:
    """Create a force function from its position transfer matrix.

    This is the position equivalent to the velocity-based
    :py:func:`wecopttool.core.force_from_impedance`.

    Parameters
    ----------
    rao_transfer_mat
        Complex position transfer matrix.
    zero_freq
        Whether the first frequency should be zero. Default is
        :python:`True`.

    See Also
    --------
    force_from_impedance,
    """
    def force(wec, x_wec, x_opt, waves):
        transfer_mat = mimo_transfer_mat(rao_transfer_mat, zero_freq)
        force_fd = wec.vec_to_dofmat(np.dot(transfer_mat, x_wec))
        return np.dot(wec.time_mat, force_fd)
    return force


def force_from_impedance(
    omega: ArrayLike,
    impedance: ArrayLike,
) -> TStateFunction:
    """Create a force function its impedance.

    Parameters
    ----------
    omega
        Radial frequency vector.
    impedance
        Complex impedance matrix.

    See Also
    --------
    force_from_rao_transfer_function,
    """
    return force_from_rao_transfer_function(impedance/(1j*omega), False)


def force_from_waves(force_coeff: ArrayLike) -> TStateFunction:
    """Create a force function from waves excitation coefficients.

    Parameters
    ----------
    force_coeff
        Complex excitation coefficients indexed by frequency and
        direction angle.
    """
    def force(wec, x_wec, x_opt, waves):
        force_fd = complex_to_real(wave_excitation(force_coeff, waves), False)
        return np.dot(wec.time_mat[:, 1:], force_fd)
    return force


def inertia(
    f1: float,
    nfreq: int,
    inertia_matrix: ArrayLike
) -> TStateFunction:
    """Create the inertia "force" from the inertia matrix.

    Parameters
    ----------
    f1
        Fundamental frequency :python:`f1` [Hz].
    nfreq
        Number of frequencies.
    inertia_matrix
        Inertia matrix.
    """
    omega = np.reshape(frequency(f1, nfreq, False)*2*np.pi, [1,1,-1])
    inertia_matrix = np.expand_dims(inertia_matrix, -1)
    rao_transfer_function = -1*omega**2*inertia_matrix + 0j
    inertia_fun = force_from_rao_transfer_function(
        rao_transfer_function, False)
    return inertia_fun


def standard_forces(hydro_data: Dataset) -> TForceDict:
    """Create functions for linear hydrodynamic forces.

    Returns a dictionary with the standard linear forces:
    radiation, hydrostatic, friction, Froude—Krylov, and diffraction.
    The functions are type :python:'StateFunction` (see Type Aliases in
    API Documentation).

    Parameters
    ----------
    hydro_data
        Linear hydrodynamic data.
    """
    hydro_data = hydro_data.transpose(
         "omega", "wave_direction", "radiating_dof", "influenced_dof")

    # intrinsic impedance
    w = hydro_data['omega']
    A = hydro_data['added_mass']
    B = hydro_data['radiation_damping']
    K = hydro_data['hydrostatic_stiffness']
    Bf = hydro_data['friction']

    rao_transfer_functions = dict()
    rao_transfer_functions['radiation'] = (1j*w*B + -1*w**2*A, False)
    rao_transfer_functions['friction'] = (1j*w*Bf, False)

    # include zero_freq in hydrostatics
    hs = ((K + 0j).expand_dims({"omega": B.omega}))
    tmp = hs.isel(omega=0).copy(deep=True)
    tmp['omega'] = tmp['omega'] * 0
    hs = xr.concat([tmp, hs], dim='omega') #, data_vars='minimal')
    rao_transfer_functions['hydrostatics'] = (hs, True)

    linear_force_functions = dict()
    for name, (value, zero_freq) in rao_transfer_functions.items():
        value = value.transpose("radiating_dof", "influenced_dof", "omega")
        value = -1*value  # RHS of equation: ma = Σf
        linear_force_functions[name] = (
            force_from_rao_transfer_function(value, zero_freq))

    # wave excitation
    excitation_coefficients = {
        'Froude_Krylov': hydro_data['Froude_Krylov_force'],
        'diffraction': hydro_data['diffraction_force']
    }

    for name, value in excitation_coefficients.items():
        linear_force_functions[name] = force_from_waves(value)

    return linear_force_functions


def run_bem(
    fb: cpy.FloatingBody,
    freq: Iterable[float] = [np.infty],
    wave_dirs: Iterable[float] = [0],
    rho: float = _default_parameters['rho'],
    g: float = _default_parameters['g'],
    depth: float = _default_parameters['depth'],
    write_info: Optional[Mapping[str, bool]] = None,
    njobs: int = 1,
) -> Dataset:
    """Run Capytaine for a range of frequencies and wave directions.

    This simplifies running *Capytaine* and ensures the output are in
    the correct convention (see
    :py:func:`wecopttool.core.change_bem_convention`).

    It creates the *test matrix*,
    calls :python:`capytaine.FloatingBody.keep_immersed_part`,
    calls :python:`capytaine.BEMSolver()fill_dataset`,
    and changes the sign convention using
    :py:func:`wecopttool.core.change_bem_convention`.

    Parameters
    ----------
    fb
        The WEC as a Capytaine floating body (mesh + DOFs).
    freq
        List of frequencies [Hz] to evaluate BEM at.
    wave_dirs
        List of wave directions [degrees] to evaluate BEM at.
    rho
        Water density in :math:`kg/m^3`.
    g
        Gravitational acceleration in :math:`m/s^2`.
    depth
        Water depth in :math:`m`.
    write_info
        Which additional information to write.
        Options are:
        :python:`['hydrostatics', 'mesh', 'wavelength', 'wavenumber']`.
        See :python:`capytiane.io.xarray.assemble_dataset` for more
        details.
    njobs
        Number of jobs to run in parallel.
        See :python:`capytaine.bem.solver.fill_dataset`

    See Also
    --------
    change_bem_convention,
    """
    if wave_dirs is not None:
        wave_dirs = np.atleast_1d(degrees_to_radians(wave_dirs))
    solver = cpy.BEMSolver()
    test_matrix = Dataset(coords={
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
    if write_info is None:
        write_info = {'hydrostatics': False,
                      'mesh': False,
                      'wavelength': False,
                      'wavenumber': False,
                     }
    wec_im = fb.copy(name=f"{fb.name}_immersed").keep_immersed_part()
    bem_data = solver.fill_dataset(
        test_matrix, wec_im, n_jobs=njobs, **write_info)
    return change_bem_convention(bem_data)


def change_bem_convention(bem_data: Dataset) -> Dataset:
    """Change the convention from `-iωt` to `+iωt`.

    Change the linear hydrodynamic coefficients from the Capytaine
    convention (:math:`x(t)=Xe^{-iωt}`), where :math:`X` is the
    frequency-domain response, to the more standard convention
    used in WecOptTool (:math:`x(t)=Xe^{+iωt}`).

    NOTE: This might change in Capytaine in the future.

    Parameters
    ----------
    bem_data
        Linear hydrodynamic coefficients for the WEC.
    """
    bem_data['Froude_Krylov_force'] = np.conjugate(
        bem_data['Froude_Krylov_force'])
    bem_data['diffraction_force'] = np.conjugate(bem_data['diffraction_force'])
    return bem_data


def linear_hydrodynamics(
    bem_data: Dataset,
    inertia_matrix: Optional[ArrayLike] = None,
    hydrostatic_stiffness: Optional[ArrayLike] = None,
    friction: Optional[ArrayLike] = None
) -> Dataset:
    """Add rigid body inertia_matrix, hydrostatic stiffness, and linear
    friction to BEM data.

    Returns the Dataset with the additional information added.

    Parameters
    ----------
    bem_data
        Linear hydrodynamic coefficients obtained using the boundary
        element method (BEM) code Capytaine, with sign convention
        corrected.
    inertia_matrix
        Inertia matrix of size `ndof x ndof`.
        `None` if included in `bem_data`.
    hydrostatic_stiffness
        Linear hydrostatic restoring coefficient of size `nodf x ndof`.
        `None` if included in `bem_data`.
    friction
        Linear friction, in addition to radiation damping, of size
        `nodf x ndof`.
        `None` if included in `bem_data` or to set to zero.

    Raises
    ------
    ValueError
        If either :python:`inertia_matrix` or
        :python:`hydrostatic_stiffness` are :python:`None` and is not
        included in :python:`bem_data`.
    ValueError
        If any of :python:`inertia_matrix`,
        :python:`hydrostatic_stiffness`, or :python:`friction` are both
        provided and included in :python:`bem_data` but have different
        values.
    """
    vars = {'inertia_matrix': inertia_matrix, 'friction': friction,
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
                raise ValueError(
                    f'Variable "{name}" is not in BEM data and ' +
                    'was not provided.')
        elif new:
            data = atleast_2d(data)
            hydro_data[name] = (dims, data)

    return hydro_data


def atleast_2d(array: ArrayLike) -> ndarray:
    """Ensure an array is at least 2D, otherwise add trailing dimensions
    to make it 2D.

    This differs from :python:`numpy.atleast_2d` in that the additional
    dimensions are appended at the end rather than at the begining.
    This might be an option in :python:`numpy.atleast_2d` in the future,
    see `NumPy #12336 <https://github.com/numpy/numpy/issues/12336>`_.

    Parameters
    ----------
    array
        Input array.
    """
    array = np.atleast_1d(array)
    return np.expand_dims(array, -1) if len(array.shape)==1 else array


def subset_close(
    set_a: FloatOrArray,
    set_b: FloatOrArray,
    rtol: float = 1.e-5,
    atol: float = 1.e-8,
    equal_nan: bool = False,
) -> tuple[bool, list]:
    """Check if the first set :python:`set_a` is contained, to some
    tolerance, in the second set :python:`set_b`.

    Parameters
    ----------
    set_a
        First array which is tested for being subset.
    set_b
        Second array which is tested for containing :python:`set_a`.
    rtol
        The relative tolerance parameter. Passed to
        :python:`numpy.isclose`.
    atol
        The absolute tolerance parameter. Passed to
        :python:`numpy.isclose`.
    equal_nan
        Whether to compare NaNs as equal. Passed to
        :python:`numpy.isclose`.

    Returns
    -------
    subset
        Whether the first array is a subset of the second array.
    ind
        List with integer indices where the first array's elements are
        located inside the second array.
        Only contains values if :python:`subset==True`.

    Raises
    ------
    ValueError
        If either of the two arrays contains repeated elements.
    """
    if len(np.unique(set_a.round(decimals = 6))) != len(set_a):
        raise ValueError("Elements in set_a not unique")
    if len(np.unique(set_b.round(decimals = 6))) != len(set_b):
        raise ValueError("Elements in set_b not unique")

    ind = []
    tmp_result = [False for _ in range(len(set_a))]
    for subset_element in set_a:
        for set_element in set_b:
            if np.isclose(subset_element, set_element, rtol, atol, equal_nan):
                tmp_set_ind = np.where(
                    np.isclose(set_element, set_b , rtol, atol, equal_nan))
                tmp_subset_ind = np.where(
                    np.isclose(subset_element, set_a , rtol, atol,
                               equal_nan))
                ind.append( int(tmp_set_ind[0]) )
                tmp_result[ int(tmp_subset_ind[0]) ] = True
    subset = all(tmp_result)
    ind = ind if subset else []
    return subset, ind


def scale_dofs(scale_list: Iterable[float], ncomponents: int) -> ndarray:
    """Create a scaling vector based on a different scale for each DOF.

    Returns a 1D array of length :python:`NDOF x ncomponents` where the
    number of DOFs (:python:`NDOF`) is the length of
    :python:`scale_list`.
    The first :python:`ncomponents` entries have the value of the first
    scale :python:`scale_list[0]`, the next :python:`ncomponents`
    entries have the value of the second scale :python:`scale_list[1]`,
    and so on.


    Parameters
    ----------
    scale_list
        Scale for each DOF.
    ncomponents
        Number of elements in the state vector for each DOF.
    """
    ndof = len(scale_list)
    scale = []
    for dof in range(ndof):
        scale += [scale_list[dof]] * ncomponents
    return np.array(scale)


def decompose_state(
    state: ndarray,
    ndof: int,
    nfreq: int,
) -> tuple[ndarray, ndarray]:
    """Split the state vector into the WEC dynamics state and the
    optimization (control) state.

    The WEC dynamics state consists of the Fourier coefficients of
    the position of each degree of freedom.
    The optimization state depends on the chosen control states for
    the problem.

    Parameters
    ----------
    state
        Combined WEC and optimization states.
    ndof
        Number of degrees of freedom for the WEC dynamics.
    nfreq
        Number of frequencies.

    Returns
    -------
    state_wec
        WEC state vector.
    state_opt
        Optimization (control) state.
    """
    nstate_wec = ndof * ncomponents(nfreq)
    return state[:nstate_wec], state[nstate_wec:]


def frequency_parameters(
    freqs: ArrayLike,
    zero_freq: bool = True,
) -> tuple[float, int]:
    """Return the fundamental frequency and the number of frequencies
    in a frequency array.

    This function can be used as a check for inputs to other functions
    since it raises an error if the frequency vector does not have
    the correct format :python:`freqs = [0, f1, 2*f1, ..., nfreq*f1]`.

    Parameters
    ----------
    freqs
        The frequency array, starting at zero and having equal spacing.
    zero_freq
        Whether the first frequency should be zero.

    Returns
    -------
    f1
        Fundamental frequency :python:`f1` [Hz]
    nfreq
        Number of frequencies (not including zero frequency),
        i.e., :python:`freqs = [0, f1, 2*f1, ..., nfreq*f1]`.

    Raises
    ------
    ValueError
        If the frequency vector is not evenly spaced.
    ValueError
        If the zero-frequency was expected but not included or not
        expected but included.
    """
    if np.isclose(freqs[0], 0.0):
        if zero_freq:
            freqs0 = freqs[:]
        else:
            raise ValueError('Zero frequency was included.')
    else:
        if zero_freq:
            raise ValueError(
                'Frequency array must start with the zero frequency.')
        else:
            freqs0 = np.concatenate([[0.0,], freqs])

    f1 = freqs0[1]
    nfreq = len(freqs0) - 1
    f_check = np.arange(0, f1*(nfreq+0.5), f1)
    if not np.allclose(f_check, freqs0):
        raise ValueError("Frequency array `omega` must be evenly spaced by" +
                         "the fundamental frequency " +
                         "(i.e.,`omega = [0, f1, 2*f1, ..., nfreq*f1])")
    return f1, nfreq


def time_results(fd: DataArray, time: DataArray) -> ndarray:
    """Create a :py:class:`xarray.DataArray` of time-domain results from
    :py:class:`xarray.DataArray` of frequency-domain results.

    Parameters
    ----------
    fd
        Frequency domain response.
    time
        Time array.
    """
    out = np.zeros((*fd.isel(omega=0).shape, len(time)))
    for w, mag in zip(fd.omega, fd):
        out = out + \
            np.real(mag)*np.cos(w*time) - np.imag(mag)*np.sin(w*time)
    return out


def add_zerofreq_to_xr(data):
    """Add a zero-frequency component to an :python:`xarray.Dataset`.

    Frequency variable must be called :python:`omega`.
    """
    if not np.isclose(data.coords['omega'][0].values, 0):
        tmp = data.isel(omega=0).copy(deep=True)
        tmp['omega'] = tmp['omega'] * 0
        vars = [var for var in list(data.keys()) if 'omega' in data[var].dims]
        for var in vars:
            tmp[var] = tmp[var] * 0
        data = xr.concat([tmp, data], dim='omega', data_vars='minimal')
    return data