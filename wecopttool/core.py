
from __future__ import annotations  # TODO: delete after python 3.10
import logging
from typing import Iterable, Callable, Any
from pathlib import Path

import numpy.typing as npt
import autograd.numpy as np
from autograd.builtins import isinstance, tuple, list, dict
from autograd import jacobian
import xarray as xr
import capytaine as cpy
from scipy import optimize
from scipy import sparse
import matplotlib.pyplot as plt


log = logging.getLogger(__name__)

# Default values
_default_parameters = {'rho': 1000.0, 'g': 9.81, 'depth': np.infty}


class WEC:
    """
    """

    def __init__(self, fb: cpy.FloatingBody, mass_matrix: np.ndarray,
                 hydrostatic_stiffness: np.ndarray, f0: float, num_freq: int,
                 dissipation: np.ndarray | None = None,
                 stiffness: np.ndarray | None = None,
                 f_add: Callable[[WEC, np.ndarray, np.ndarray], np.ndarray] |
                 None = None,
                 rho: float = _default_parameters['rho'],
                 depth: float = _default_parameters['depth'],
                 g: float = _default_parameters['g']) -> None:
        """
        Parameters
        ----------
        fb: capytaine.FloatingBody
            The WEC as a capytaine floating body (mesh + DOFs).
        mass_matrix: np.ndarray
            Mass matrix shape of (``ndof`` x ``ndof``).
        hydrostatic_stiffness: np.ndarray
            Hydrstatic stiffness matrix matrix of shape
            (``ndof`` x ``ndof``).
        f0: float
            Initial frequency (in Hz) for frequency array.
            Frequency array given as [f0, 2*f0, ..., num_freq*f0].
        num_freq: int
            Number of frequencies in frequency array. See ``f0``.
        dissipation: np.ndarray
            Additional dissipiation for the impedance calculation in
            ``capytaine.post_pro.impedance``. Shape:
            (``ndof``x``ndof``x1) or (``ndof``x``ndof``x``num_freq``).
        stiffness: np.ndarray
            Additional stiffness for the impedance calculation in
            ``capytaine.post_pro.impedance``. Shape:
            (``ndof``x``ndof``x1) or (``ndof``x``ndof``x``num_freq``).
        f_add: function
            Additional forcing terms (e.g. PTO, mooring, etc.) for the
            WEC dynamics in the time-domain. Takes three inputs:
            (1) the WEC object,
            (2) the WEC dynamics state (1D np.ndarray), and
            (3) the optimization state (1D np.ndarray)
            and outputs the force time-series (1D np.ndarray).
        rho: float, optional
            Water density in :math:`kg/m^3`.
        depth: float, optional
            Water depth in :math:`m`.
        g: float, optional
            Gravitational acceleration in :math:`m/s^2`.
        """
        # water properties
        super().__setattr__('rho', rho)
        super().__setattr__('depth', depth)
        super().__setattr__('g', g)
        if g != _default_parameters['g']:
            # TODO: Capytaine: fixed after next release of capytaine
            raise NotImplementedError('Currently only g=9.81 can be used.')

        # WEC
        super().__setattr__('fb', fb)
        super().__setattr__('mass_matrix', mass_matrix)
        super().__setattr__('hydrostatic_stiffness', hydrostatic_stiffness)

        # frequency
        super().__setattr__('freq', (f0, num_freq))

        # additional WEC dynamics forces
        super().__setattr__('f_add', f_add)
        super().__setattr__('dissipation', dissipation)
        super().__setattr__('stiffness', stiffness)

        # log
        super().__setattr__(
            '_dof_str', 'DOF' if self.fb.nb_dofs == 1 else 'DOFs')
        log.info(
            f"New WEC: {self.fb.name} with {self.fb.nb_dofs} {self._dof_str}")

    def __setattr__(self, name, value):
        """ Delete dependent attributes  when user manually modifies
        an attribute.
        """
        _attrs_delete_mass = ['fb']
        _attrs_delete_stiffness = ['fb', 'rho', 'g']
        _attrs_delete_impedance_not_bem = ['dissipation', 'stiffness']
        _attrs_delete_bem = ['fb', 'rho', 'g', 'depth', 'freq']
        log.info(f"Changing value of '{name}'. " +
                 "This might cause some attributes to be reset.")
        if name in _attrs_delete_mass:
            super().__setattr__('name', value)
            log.info("  Mass matrix deleted. " +
                     "Assign new values to 'self.mass_matrix'")
        if name in _attrs_delete_stiffness:
            super().__setattr__('hydrostatic_stiffness', None)
            log.info("  Hydrostatic stiffness deleted. " +
                     "Assign new values to 'self.hydrostatic_stiffness'")
        if name in _attrs_delete_impedance_not_bem:
            if 'Zi' in self.hydro:
                self.hydro['Zi'] = None
                log.info("  Impedance matrix deleted. To calculate " +
                         "impedance call 'self._bem_calc_impedance()'")
        if name in _attrs_delete_bem:
            super().__setattr__('hydro', xr.DataArray())
            super().__setattr__('_gi_block_scaled', None)
            super().__setattr__('_gi_scale', None)
            log.info("  BEM data deleted. To run BEM use self.run_bem(...) ")
        super().__setattr__(name, value)

    def __repr__(self):
        str_info = (f'{self.__class__.__name__} "{self.fb.name}" ' +
                    f'with {self.fb.nb_dofs} {self._dof_str}')
        return str_info

    # frequency properties
    @property
    def freq(self):
        """ Freequency array f=[f0, 2*f0, ..., num_freq*f0] in Hz. """
        return self._freq

    @freq.setter
    def freq(self, freq):
        if len(freq) != 2:
            msg = "To set the frequency provide a tuple (f0, num_freq)"
            raise TypeError(msg)
        f0, num_freq = freq
        super().__setattr__('_freq', freq_array(f0, num_freq))
        # update phi and dphi
        super().__setattr__('_phi', self._make_phi())
        super().__setattr__('_phi_for_fi', self._make_phi_for_fi())
        super().__setattr__('_dphi', self._make_dphi())

    @property
    def f0(self):
        """ Initial frequency (and spacing) in Hz. See ``freq``. """
        return self._freq[0]

    @property
    def num_freq(self):
        """ Number of frequencies in frequency array. See ``freq``. """
        return len(self._freq)

    @property
    def period(self):
        """ Period :math:T=1/f in seconds. """
        return 1/self._freq

    @property
    def omega(self):
        """ Frequency array in radians per second ω=2πf. """
        return self._freq * 2 * np.pi

    @property
    def omegap0(self):
        """ Like ``omega`` but also includes zero frequency: [0, ω]. """
        return np.concatenate([[0.0], self.omega])

    @property
    def phi(self):
        """ Matrix to convert from frequency domain to time domain.
        Set when frequency is set. """
        return self._phi

    @property
    def dphi(self):
        """ Derivative matrix to convert from position to velocity in
        the frequency domain. Set when frequency is set. """
        return self._dphi

    # problem size
    @property
    def ndof(self):
        """ Number of degrees of freedom of the WEC. """
        return self.fb.nb_dofs

    @property
    def nfd(self):
        """ Number of frequencies in the two sided spectrum. """
        return 2 * self.num_freq + 1

    @property
    def nfd_nomean(self):
        """ Like ``nfd`` but for the no mean (no f=0) cases. """
        return 2 * self.num_freq

    @property
    def num_x_wec(self):
        """ Length of the  WEC dynamics state vector. """
        return self.ndof * self.nfd

    # time
    @property
    def time(self) -> np.ndarray:
        """ Time array. """
        return np.linspace(0, 1/self.f0, self.nfd)

    # save/load class object
    def to_file(self, fpath: str | Path) -> None:
        # TODO: implement
        raise NotImplementedError()

    @staticmethod
    def from_file(fpath: str | Path) -> WEC:
        # TODO: implement
        raise NotImplementedError()

    # state vector
    def decompose_decision_var(self, x: np.ndarray
                               ) -> tuple[np.ndarray, np.ndarray]:
        """ Split the state vector into the WEC dynamics state and the
        optimization state. x = [x_wec, x_opt].

        Parameters
        ----------
        x: np.ndarray

        Returns
        -------
        x_wec: np.ndarray
        x_opt: np.ndarray
        """
        return x[:self.num_x_wec], x[self.num_x_wec:]

    def vec_to_dofmat(self, vec: np.ndarray) -> np.ndarray:
        """ Convert a flat vector back to a matrix with one row per DOF.
        Opposite of ``dofmat_to_vec``. """
        return np.reshape(vec, (self.ndof, -1))

    def dofmat_to_vec(self, mat: np.ndarray) -> np.ndarray:
        """ Flatten a matrix that has one row per DOF.
        Opposite of ``vec_to_dofmat``. """
        return np.reshape(mat, -1)

    # Transformation matrices
    def _make_phi(self) -> np.ndarray:
        t = np.linspace(0, 1/self.f0, self.nfd, endpoint=False).reshape(1, -1)
        w = self.omega.reshape(1, -1)
        mm = w.T @ t
        phi = np.ones((self.nfd, self.nfd))
        phi[1::2, ::] = np.cos(mm)
        phi[2::2, ::] = np.sin(mm)
        return phi

    def _make_phi_for_fi(self) -> np.ndarray:
        """ Shuffle phi rather than f_i, since autograd doesn't like it.
        """
        tmp_1 = self.phi[0]
        tmp_2 = self.phi[1::2]
        tmp_3 = self.phi[2::2]
        return np.vstack([tmp_1, tmp_2, tmp_3])

    def _make_dphi(self) -> np.ndarray:
        omega = np.array(self.omega)
        dphi = np.diag((np.array([[1], [0]]) * omega).flatten('F')[:-1], 1)
        dphi = dphi - dphi.transpose()
        dphi = np.concatenate((np.zeros((2*self.num_freq, 1)), dphi), axis=1)
        return dphi.T

    # ifft
    def fd_to_td(self, fd: np.ndarray) -> np.ndarray:
        """ Convert from frequency domain to time ddomain using IFFT."""
        return np.fft.irfft(fd/(2/self.nfd), n=self.nfd)

    # bem & impedance
    def run_bem(self, wave_dirs: npt.ArrayLike = [0]) -> None:
        """ Run the BEM for the specified wave directions.

        See ``wot.run_bem``.

        Parameters
        ----------
        wave_dirs: list[float]
            List of wave directions to evaluate BEM at.
        """
        log.info(f"Running Capytaine (BEM): {self.num_freq} frequencies x " +
                 f"{len(wave_dirs)} wave directions.")
        write_info = ['hydrostatics', 'mesh', 'wavelength', 'wavenumber']
        data = run_bem(self.fb, self.freq, wave_dirs,
                       rho=self.rho, g=self.g, depth=self.depth,
                       write_info=write_info)
        super().__setattr__('hydro', data)
        # add mass and stiffness
        self._bem_add_hydrostatics()
        # calculate impedance
        self._bem_calc_impedance()
        # post-processing needed for solving dynamics
        self._post_process_impedance()

    def read_bem(self, fpath: str | Path) -> None:
        """ Read a BEM solution from a NetCDF file.

        Parameters
        ----------
        fpath: str
            Name of file to read BEM data from.
        """
        log.info(f"Reading BEM data from {fpath}.")
        data = from_netcdf(fpath)
        super().__setattr__('hydro', data)
        # add mass and stiffness
        bmass = 'mass' in self.hydro
        bstiffness = 'hydrostatic_stiffness' in self.hydro
        if bmass:
            assert np.allclose(self.hydro['mass'].values, self.mass_matrix)
        if bstiffness:
            assert np.allclose(self.hydro['hydrostatic_stiffness'].values,
                               self.hydrostatic_stiffness)
        if not (bmass and bstiffness):
            self._bem_add_hydrostatics()
        # add impedance
        if not 'Zi' in self.hydro:
            self._bem_calc_impedance()
        # post-processing needed for solving dynamics
        self._post_process_impedance()

    def write_bem(self, fpath: str | Path) -> None:
        """
        Write the BEM solution to a NetCDF file.

        Parameters
        ----------
        fpath: str
            Name of file to write BEM data to.
        """
        log.info(f"Writting BEM data to {fpath}.")
        to_netcdf(fpath, self.hydro)

    def _bem_add_hydrostatics(self) -> None:
        """ Add hydrostatic data to self.hydro. """
        dims = ['radiating_dof', 'influenced_dof']
        self.hydro['mass'] = (dims, self.mass_matrix)
        self.hydro['hydrostatic_stiffness'] = (
            dims, self.hydrostatic_stiffness)

    def _bem_calc_impedance(self) -> None:
        """ Calculate the impedance matrix.
        """
        # TODO: Move this to outside WEC.
        #       wrapper on Capytaine impedance that makes symmetric, etc.
        log.info("Calculating impedance matrix.")
        # TODO: Capytaine: use capytaine after next release
        # impedance = cpy.post_pro.impedance(
        #     self.hydro, self.dissipation, self.stiffness)
        impedance = _cpy_impedance(
            self.hydro, self.dissipation, self.stiffness)
        impedance = impedance * -1j/impedance.omega

        # TODO: Should not be zero either (maybe % of max?)
        # ensure non-negative diagonal
        for iw in impedance['omega'].values:
            B = np.real(impedance.sel(omega=iw).values).diagonal()
            B_min = B.min()
            if B_min < 0.0:
                log.warning("Impedance matrix has negative diagonal terms." +
                            " Setting to zero.")
                for j, jB in enumerate(B):
                    impedance.loc[{"omega": iw}][j, j] = (
                        np.max([0.0, jB]) +
                        1j * np.imag(impedance.loc[{"omega": iw}][j, j]))

        # make symmetric
        impedance = impedance.transpose(
            'omega', 'radiating_dof', 'influenced_dof')
        impedance_transp = impedance.transpose(
            'omega', 'influenced_dof', 'radiating_dof')
        impedance = (impedance + impedance_transp) / 2

        # store
        self.hydro['Zi'] = impedance

    def _post_process_impedance(self) -> None:
        """ Calculate the Gi block. """
        _gi_block = self._make_gi_block()
        _gi_scale = 1/np.linalg.norm(_gi_block.toarray())
        _gi_block_scaled = _gi_scale * _gi_block.toarray()
        super().__setattr__('_gi_scale', _gi_scale)
        super().__setattr__('_gi_block_scaled', _gi_block_scaled)

    def _make_gi_block(self) -> sparse.dia.dia_matrix:
        """
        Create a block matrix of the MIMO impedance + position.
        """
        impedance = self.hydro['Zi'].values
        elem = [[None]*self.ndof for _ in range(self.ndof)]

        for idof in range(self.ndof):
            for jdof in range(self.ndof):
                K = self.hydrostatic_stiffness[idof, jdof]
                w_impedance = self.omega*impedance[:, idof, jdof]
                elem[idof][jdof] = np.diag(
                    np.concatenate(([K], 1j * w_impedance)))

        return sparse.dia_matrix(np.block(elem))

    def bem_calc_inf_added_mass(self) -> None:
        """ Run the BEM to obtain infinite added mass. """
        log.info("Running Capytaine for infinite frequency.")
        inf_data = run_bem(
            self.fb, [np.infty], wave_dirs=None,
            rho=self.rho, g=self.g, depth=self.depth)
        self.hydro['Ainf'] = inf_data.added_mass[0, :, :]

    def bem_calc_rao(self) -> None:
        """ Calculate BEM RAOs using capytaine. """
        self.hydro['rao'] = cpy.post_pro.rao(self.hydro)

    def plot_impedance(self, option: str = 'symmetric', show: bool = True):
        """ Plot impedance.

        See `wot.plot_impedance()`.
        """
        fig, axs = plot_impedance(
            Zi=self.hydro.Zi.values, freq=self.freq, option=option,
            dof_names=self.hydro.influenced_dof.values.tolist(), show=show)
        return fig, axs

    # solve
    def solve(self, waves: xr.Dataset,
              obj_fun: Callable[[WEC, np.ndarray, np, ndarray], float],
              num_x_opt: int, constraints: list[dict] = [],
              x_wec_0: np.ndarray | None = None,
              x_opt_0: np.ndarray | None = None,
              scale_x_wec: float = 1.0, scale_x_opt: float = 1.0,
              scale_obj: float = 1.0, optim_options: dict[str, Any] = {}
              ) -> tuple[xr.Dataset, xr.Dataset, np.ndarray,
                         optimize.optimize.OptimizeResult]:
        """ Solve the WEC control co-design problem.

        Parameters
        ----------
        waves: xr.Dataset
            The wave, described by two 2D DataArrays:
            elevation variance `S` (m^2*s) and phase `phase` (radians)
            with coordinates of radial frequency `omega` (radians)
            and wave direction `wave_direction` (radians).
            The frequencies and  wave directions must match those in
            the `bem_data`.
        obj_fun: function
            Objective function for the control optimization.
            Takes three inputs:
            (1) the WEC object,
            (2) the WEC dynamics state (1D np.ndarray), and
            (3) the optimization state (1D np.ndarray)
            and outputs the scalar objective function:
            tuple[WEC, np.ndarray, np.ndarray] -> float.
        num_x_opt: int
            Length of the optimization (controls) state vector.
        constraints: list[dict]
            Constraints for the constrained optimization.
            See ``scipy.optimize.minimize``.
        x_wec_0: np.ndarray
            Initial guess for the WEC dynamics state.
            If ``None`` it is randomly initiated.
        x_opt_0: np.ndarray
            Initial guess for the optimization (controls) state.
            If ``None`` it is randomly initiated.
        scale_x_wec: float
            Factor to scale ``x_wec`` by, to improve convergence.
        scale_x_opt: float
            Factor to scale ``x_opt`` by, to improve convergence.
        scale_obj: float
            Factor to scale ``obj_fun`` by, to improve convergence.
        optim_options: dict
            Optimization options passed to the optimizer.
            See ``scipy.optimize.minimize``.

        Returns
        -------
        freq_dom: xr.Dataset
            Dataset containing the frequency-domain results.
        time_dom: xr.Dataset
            Dataset containing the time-domain results.
        x_opt: np.ndarray
            Optimal control state.
        res: optimize.optimize.OptimizeResult
            Raw optimization results.
        """
        log.info("Solving pseudo-spectral control problem.")
        # initial state
        if x_wec_0 is None:
            x_wec_0 = np.random.randn(self.num_x_wec)
        if x_opt_0 is None:
            x_opt_0 = np.random.randn(num_x_opt)
        x0 = np.concatenate([x_wec_0, x_opt_0])

        # wave excitation force
        fd_we, td_we = wave_excitation(self.hydro, waves)
        f_exc = td_we['excitation_force']

        # scale
        scale = np.concatenate([
            scale_x_wec * np.ones(self.num_x_wec),
            scale_x_opt * np.ones(num_x_opt)])

        # objective function
        def obj_fun_scaled(x):
            x_wec, x_opt = self.decompose_decision_var(x/scale)
            cost = obj_fun(self, x_wec, x_opt)*scale_obj
            return cost

        # system dynamics through equality constraint
        def resid_fun(x):
            ri = self._dynamic_residual(x/scale, f_exc.values)
            return self.dofmat_to_vec(ri)

        eq_cons = {'type': 'eq',
                   'fun': resid_fun,
                   'jac': jacobian(resid_fun),
                   }
        constraints.append(eq_cons)

        # minimize
        options = {'ftol': 1e-6,
                   'eps': 1.4901161193847656e-08,
                   'disp': False,
                   'iprint': 1,
                   'maxiter': 100,
                   'finite_diff_rel_step': None,
                   }

        for key, value in optim_options.items():
            options[key] = value

        res = optimize.minimize(fun=obj_fun_scaled,
                                x0=x0,
                                jac=jacobian(obj_fun_scaled),
                                method='SLSQP',
                                constraints=constraints,
                                options=options,
                                )

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
        freq_dom_x, time_dom_x = self._post_process_x_wec(x_wec)
        fd_we = fd_we.reset_coords(drop=True)
        freq_dom = xr.merge([freq_dom_x, fd_we])
        time_dom = xr.merge([time_dom_x, td_we])
        return freq_dom, time_dom, x_opt, res

    def _dynamic_residual(self, x: np.ndarray, f_exc: np.ndarray
                          ) -> np.ndarray:
        """
        Solve WEC dynamics in residual form so that they may be
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
        # WEC position
        x_wec, x_opt = self.decompose_decision_var(x)
        x_fd = self.vec_to_dofmat(x_wec)
        x_fd_hat = fd_folded(x_fd)
        x_fd_hat_vec = self.dofmat_to_vec(x_fd_hat)

        fi_fd = self.vec_to_dofmat(np.dot(self._gi_block_scaled, x_fd_hat_vec))
        fi_fd_tmp_0 = np.real(fi_fd[:, 0:1])
        tmp_1_0 = np.real(fi_fd[:, 1::])
        tmp_1_1 = -np.imag(fi_fd[:, 1::])
        fi_fd = np.hstack((fi_fd_tmp_0, tmp_1_0, tmp_1_1))
        f_i = np.dot(fi_fd, self._phi_for_fi)

        if self.f_add is not None:
            f_add = self.f_add(self, x_wec, x_opt)
        else:
            f_add = 0.0

        return f_exc + f_add - f_i

    def _post_process_x_wec(self, x_wec: np.ndarray
                            ) -> tuple[xr.DataArray, xr.DataArray]:
        """
        Transform the results from optimization solution to a form that
        the user can work with directly.
        """
        # scale
        x_wec = x_wec * self._gi_scale

        # position
        x_fd = self.vec_to_dofmat(x_wec)
        x_fd_hat = fd_folded(x_fd)
        x_td = x_fd @ self.phi

        # velocity
        vel_fd = x_fd @ self.dphi
        vel_fd_hat = fd_folded_nomean(vel_fd)
        vel_td = vel_fd @ self.phi[1:, :]

        # xarray
        dims_fd = ('influenced_dof', 'omega')
        coords_fd = [
            (dims_fd[0], self.hydro.influenced_dof.values),
            (dims_fd[1], self.omegap0, {'units': '(rad)'})]
        dims_td = ('influenced_dof', 'time')
        coords_td = [
            (dims_td[0], self.hydro.influenced_dof.values),
            (dims_td[1], self.time, {'units': 's'})]
        attrs_x = {'long_name': 'WEC position', 'units': 'm or (rad)'}
        attrs_vel = {'long_name': 'WEC velocity', 'units': 'm/s or (rad)/s'}
        x_td = xr.DataArray(
            x_td, dims=dims_td, coords=coords_td, attrs=attrs_x)
        vel_td = xr.DataArray(
            vel_td, dims=dims_td, coords=coords_td, attrs=attrs_vel)
        attrs_x['units'] = 'm^2*s or (rad)^2*s'
        attrs_vel['units'] = 'm^2/s or (rad)^2/s'
        x_fd = xr.DataArray(
            x_fd_hat, dims=dims_fd, coords=coords_fd, attrs=attrs_x)
        vel_fd = xr.DataArray(
            vel_fd_hat, dims=dims_fd, coords=coords_fd, attrs=attrs_vel)

        freq_dom = xr.Dataset({'pos': x_fd, 'vel': vel_fd},)
        time_dom = xr.Dataset({'pos': x_td, 'vel': vel_td},)

        return freq_dom, time_dom


def freq_array(f0: float, num_freq: int) -> np.ndarray:
    """ Cunstruct equally spaced frequency array.
    """
    return np.arange(1, num_freq+1)*f0


def fd_folded(fd: np.ndarray) -> np.ndarray:
    """ Convert a two-sided spectrum to one sided. """
    mean = fd[:, 0:1]
    return np.concatenate((mean, fd[:, 1::2] - fd[:, 2::2]*1j), axis=1)


def fd_folded_nomean(fd: np.ndarray) -> np.ndarray:
    """ Like ``fd_folded`` but for spectrum with no mean (no f=0).
    """
    ndof = fd.shape[0]
    mean = np.zeros([ndof, 1])
    return np.concatenate((mean, fd[:, 0::2] - fd[:, 1::2]*1j), axis=1)


def wave_excitation(bem_data: xr.Dataset, waves: xr.Dataset
                    ) -> tuple[xr.Dataset, xr.Dataset]:
    """ Compute the frequency- and time-domain wave excitation force.

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
    wavesp0 = xr.concat([tmp, waves], dim='omega')

    assert exc_coeff.omega[0] != 0
    tmp = exc_coeff.isel(omega=0).copy(deep=True)
    tmp['omega'] = tmp['omega'] * 0
    tmp = tmp * 0
    tmp['wavenumber'] = 0.0
    tmp['wavelength'] = np.inf
    exc_coeff_p0 = xr.concat([tmp, exc_coeff], dim='omega')

    # complex amplitude
    dw = wavesp0.omega[1] - wavesp0.omega[0]
    wave_elev_fd = (np.sqrt(2*wavesp0['S'] / (2*np.pi) * dw) *
                    np.exp(1j*wavesp0['phase']))
    wave_elev_fd.attrs['long_name'] = 'wave elevation'
    wave_elev_fd.attrs['units'] = 'm^2*s'

    # excitation force
    f_exc_fd = xr.dot(exc_coeff_p0, wave_elev_fd, dims=["wave_direction"])
    f_exc_fd.attrs['long_name'] = 'wave excitation force'
    f_exc_fd.attrs['units'] = 'N^2*s or N^2*m^2*s'
    f_exc_fd = f_exc_fd.transpose('influenced_dof', 'omega')

    freq_dom = xr.Dataset(
        {'wave_elevation': wave_elev_fd, 'excitation_force': f_exc_fd},)
    freq_dom['omega'].attrs['long_name'] = 'frequency'
    freq_dom['omega'].attrs['units'] = '(radians)'

    # time domain
    nfd = 2 * len(waves['omega']) + 1
    f0 = waves['omega'][0] / (2*np.pi)
    time = np.linspace(0, 1/f0, nfd)
    dims_td = ['time', ]
    coords_td = [(dims_td[0], time, {'units': 's'}), ]

    def fd_to_td(freq_dom):
        return np.fft.irfft(freq_dom/(2/nfd), n=nfd)

    f_exc_td = fd_to_td(f_exc_fd)
    dims = ['influenced_dof'] + dims_td
    coords = [(dims[0], f_exc_fd.coords[dims[0]].data,)] + coords_td
    f_exc_td = xr.DataArray(
        f_exc_td, dims=dims, coords=coords, attrs=f_exc_fd.attrs)
    f_exc_td.attrs['units'] = 'N or N*m'
    time_dom = xr.Dataset({'excitation_force': f_exc_td},)

    eta_all = fd_to_td(wave_elev_fd)
    wave_elev_td = np.sum(eta_all, axis=0)
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
    """ Run Capytaine for a range of frequencies and wave directions.

    Parameters
    ----------
    fb: capytaine.FloatingBody
        The WEC as a capytaine floating body (mesh + DOFs).
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
        List of informmation to keep, passed to `capytaine` solver.
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
    wec_im = fb.copy(name=f"{fb.name}_immersed").keep_immersed_part()
    write_info = {key: True for key in write_info}
    return solver.fill_dataset(test_matrix, [wec_im], **write_info)


def plot_impedance(Zi: npt.ArrayLike, freq: npt.ArrayLike,
                   option: str = 'diagonal', show: bool = False,
                   dof_names: list[str] | None = None):
    """ Plot the impedance matrix.

    Parameters
    ----------
    Zi: np.ndarray
        Complex impedance matrix. Shape: nfreq x ndof x ndof
    freq: list[float]
        Frequencies in Hz.
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
    figh = 2.5
    figw = 2 * figh
    ndof = Zi.shape[-1]
    fig, axs = plt.subplots(
        ndof*2, ndof, figsize=(ndof*figw, ndof*figh),
        sharex='all', sharey='row', squeeze=False)

    if dof_names is None:
        dof_names = [f"DOF {i}" for i in range(ndof)]

    colors = (plt.rcParams['axes.prop_cycle'].by_key()['color']*10)[:ndof]
    phase_pad = 18
    mag_max = np.max(20*np.log10(np.abs(Zi)))
    mag_pad = 0.05 * mag_max

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
                axs[idof*2, jdof].set_ylabel('Magnitude (dB)')
                axs[idof*2+1, jdof].set_ylabel('Phase (deg)')

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
                iZi = Zi[:, idof, jdof]
                mag = np.squeeze(20*np.log10(np.abs(iZi)))
                ang = np.squeeze(np.rad2deg(np.angle(iZi)))
                axs[idof*2, jdof].semilogx(freq, mag, '-o', color=color)
                axs[idof*2+1, jdof].semilogx(freq, ang, '-o', color=color)

                axs[idof*2, jdof].grid(True, which='both')
                axs[idof*2+1, jdof].grid(True, which='both')

                axs[idof*2, jdof].set_ylim(0-mag_pad, mag_max+mag_pad)
                axs[idof*2+1, jdof].set_ylim(-180-phase_pad, 180+phase_pad)
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


def from_netcdf(fpath: str | Path) -> xr.Dataset:
    """ Read a NetCDF file with commplex entries as an xarray dataSet.
    """
    return cpy.io.xarray.merge_complex_values(xr.open_dataset(fpath))


def to_netcdf(fpath: str | Path, bem_data: xr.Dataset) -> None:
    """ Save an xarray dataSet with complex entries as a NetCDF file.
    """
    cpy.io.xarray.separate_complex_values(bem_data).to_netcdf(fpath)


def _cpy_impedance(bem_data, dissipation=None, stiffness=None):
    # TODO: Capytaine: this is in Capytaine but not in release yet
    omega = bem_data.coords['omega']
    impedance = (-omega**2*(bem_data['mass'] + bem_data['added_mass']) +
                 1j*omega*bem_data['radiation_damping'] +
                 bem_data['hydrostatic_stiffness'])
    if dissipation is not None:
        impedance = impedance + 1j*omega*dissipation
    if stiffness is not None:
        impedance = impedance + stiffness
    return impedance
