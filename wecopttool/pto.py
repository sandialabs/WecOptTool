"""Provide power take-off (PTO) forces and produced energy functions
for common PTO control approaches.

The PTO produced energy can be used as the objective function for the
control optimization.
The PTO force can be included as an additional force in the WEC
dynamics.

Contains:

* The *PTO* class
* Controller functions

"""


from __future__ import annotations


from typing import Optional, TypeVar, Callable, Union

import autograd.numpy as np
from autograd.builtins import isinstance, tuple, list, dict
from autograd.numpy import ndarray
from scipy.linalg import block_diag
from scipy.optimize import OptimizeResult
from xarray import DataArray, Dataset
from datetime import datetime
from scipy.optimize import OptimizeResult

from wecopttool.core import complex_to_real, td_to_fd
from wecopttool.core import dofmat_to_vec, vec_to_dofmat
from wecopttool.core import TWEC, TStateFunction, FloatOrArray


# type aliases
TPTO = TypeVar("TPTO", bound="PTO")
TEFF = Callable[[FloatOrArray, FloatOrArray], FloatOrArray]


class PTO:
    """A power take-off (PTO) object to be used in conjunction with a
    :py:class:`wecopttool.core.WEC` object.
    """

    def __init__(self,
        ndof: int,
        kinematics: Union[TStateFunction, ndarray],
        controller: Optional[TStateFunction] = None,
        impedance: Optional[ndarray] = None,
        loss: Optional[TEFF] = None,
        names: Optional[list[str]] = None,
    ) -> None:
        """Create a PTO object.

        The :py:class:`wecopttool.pto.PTO` class describes the
        kinematics, control logic, impedance and/or non-=linear loss map
        of a power take-off system. The forces/moments applied by a
        :py:class:`wecopttool.pto.PTO` object can be applied to a
        :py:class:`wecopttool.core.WEC` object through the
        :python:`WEC.f_add` property. The power produced by a
        :py:class:`wecopttool.pto.PTO` object can be used for the
        :python:`obj_fun` of pseudo-spectral optimization problem when
        calling :python:`WEC.solve`.

        Parameters
        ----------
        ndof
            Number of degrees of freedom.
        kinematics
            Transforms state from WEC to PTO frame. May be a matrix
            (for linear kinematics) or function (for nonlinear
            kinematics).
        controller
            Function with signature
            :python:`def fun(wec, x_wec, x_opt, waves, nsubsteps):`
            or matrix with shape (PTO DOFs, WEC DOFs) that converts
            from the WEC DOFs to the PTO DOFs.
        impedance
            Matrix representing the PTO impedance.
        loss
            Function that maps flow and effort variables to a
            non-linear loss. Outputs are between 0-1.
        names
            PTO names.
        """
        self._ndof = ndof
        # names
        if names is None:
            names = [f'PTO_{i}' for i in range(ndof)]
        elif ndof == 1 and isinstance(names, str):
            names = [names]
        self._names = names
        # kinematics
        if callable(kinematics):
            def kinematics_fun(wec, x_wec, x_opt, waves, nsubsteps=1):
                pos_wec = wec.vec_to_dofmat(x_wec)
                tmat = self._tmat(wec, nsubsteps)
                pos_wec_td = np.dot(tmat, pos_wec)
                return kinematics(pos_wec_td)
        else:
            def kinematics_fun(wec, x_wec, x_opt, waves, nsubsteps=1):
                n = wec.nt*nsubsteps
                return np.repeat(kinematics[:, :, np.newaxis], n, axis=-1)
        self._kinematics = kinematics_fun
        # controller
        if controller is None:
            controller = controller_unstructured

        def force(wec, x_wec, x_opt, waves, nsubsteps=1):
            return controller(self, wec, x_wec, x_opt, waves, nsubsteps)

        self._force = force

        # power
        self._impedance = impedance
        self._loss = loss  # TODO: change to 'loss'
        if impedance is not None:
            impedance_abcd = _make_abcd(impedance, ndof)
            self._transfer_mat = _make_mimo_transfer_mat(impedance_abcd, ndof)
        else:
            self._transfer_mat = None

    @property
    def ndof(self) -> int:
        """Number of degrees of freedom."""
        return self._ndof

    @property
    def names(self) -> ndarray:
        """DOF Names."""
        return self._names

    @property
    def kinematics(self) -> TStateFunction:
        """Kinematics function.
        """
        return self._kinematics

    @property
    def force(self) -> TStateFunction:
        """PTO force in PTO coordinates."""
        return self._force

    @property
    def impedance(self) -> ndarray:
        """Impedance matrix."""
        return self._impedance

    @property
    def loss(self) -> TEFF:
        """Nonlinear loss function."""
        return self._loss

    @property
    def transfer_mat(self) -> ndarray:
        """Transfer matrix."""
        return self._transfer_mat

    def _tmat(self, wec: TWEC, nsubsteps: Optional[int] = 1):
        if nsubsteps==1:
            tmat = wec.time_mat
        else:
            tmat = wec.time_mat_nsubsteps(nsubsteps)
        return tmat

    def _fkinematics(self,
        f_wec,
        wec: TWEC,
        x_wec: ndarray,
        x_opt: Optional[ndarray] = None,
        waves: Optional[Dataset] = None,
        nsubsteps: Optional[int] = 1,
    ) -> ndarray:
        """Return time-domain values in the PTO frame.

        Parameters
        ----------
        f_wec
            Fourier coefficients of some quantity "f" in the WEC frame.
        wec
            :py:class:`wecopttool.core.WEC` object.
        x_wec
            WEC dynamic state.
        x_opt
            Optimization (control) state.
        waves
            :py:class:`xarray.Dataset` with the structure and elements
            shown by :py:mod:`wecopttool.waves`.
        nsubsteps
            Number of steps between the default (implied) time steps.
            A value of :python:`1` corresponds to the default step
            length.
        """
        time_mat = self._tmat(wec, nsubsteps)
        f_wec_td = np.dot(time_mat, f_wec)
        assert f_wec_td.shape == (wec.nt*nsubsteps, wec.ndof)
        f_wec_td = np.expand_dims(np.transpose(f_wec_td), axis=0)
        kinematics_mat = self.kinematics(wec, x_wec, x_opt, waves, nsubsteps)
        return np.transpose(np.sum(kinematics_mat*f_wec_td, axis=1))

    def position(self,
        wec: TWEC,
        x_wec: ndarray,
        x_opt: ndarray,
        waves: Optional[Dataset] = None,
        nsubsteps: Optional[int] = 1,
    ) -> ndarray:
        """Calculate the PTO position time-series.

        Parameters
        ----------
        wec
            :py:class:`wecopttool.core.WEC` object.
        x_wec
            WEC dynamic state.
        x_opt
            Optimization (control) state.
        waves
            :py:class:`xarray.Dataset` with the structure and elements
            shown by :py:mod:`wecopttool.waves`.
        nsubsteps
            Number of steps between the default (implied) time steps.
            A value of :python:`1` corresponds to the default step
            length.
        """
        pos_wec = wec.vec_to_dofmat(x_wec)
        return self._fkinematics(pos_wec, wec, x_wec, x_opt, waves, nsubsteps)

    def velocity(self,
        wec: TWEC,
        x_wec: ndarray,
        x_opt: ndarray,
        waves: Optional[Dataset] = None,
        nsubsteps: Optional[int] = 1,
    ) -> ndarray:
        """Calculate the PTO velocity time-series.

        Parameters
        ----------
        wec
            :py:class:`wecopttool.core.WEC` object.
        x_wec
            WEC dynamic state.
        x_opt
            Optimization (control) state.
        waves
            :py:class:`xarray.Dataset` with the structure and elements
            shown by :py:mod:`wecopttool.waves`.
        nsubsteps
            Number of steps between the default (implied) time steps.
            A value of :python:`1` corresponds to the default step
            length.
        """
        pos_wec = wec.vec_to_dofmat(x_wec)
        vel_wec = np.dot(wec.derivative_mat, pos_wec)
        return self._fkinematics(vel_wec, wec, x_wec, x_opt, waves, nsubsteps)

    def acceleration(self,
        wec: TWEC,
        x_wec: ndarray,
        x_opt: ndarray,
        waves: Optional[Dataset] = None,
        nsubsteps: Optional[int] = 1,
    ) -> np.ndarray:
        """Calculate the PTO acceleration time-series.

        Parameters
        ----------
        wec
            :py:class:`wecopttool.core.WEC` object.
        x_wec
            WEC dynamic state.
        x_opt
            Optimization (control) state.
        waves
            :py:class:`xarray.Dataset` with the structure and elements
            shown by :py:mod:`wecopttool.waves`.
        nsubsteps
            Number of steps between the default (implied) time steps.
            A value of :python:`1` corresponds to the default step
            length.
        """
        pos_wec = wec.vec_to_dofmat(x_wec)
        vel_wec = np.dot(wec.derivative_mat, pos_wec)
        acc_wec = np.dot(wec.derivative_mat, vel_wec)
        return self._fkinematics(acc_wec, wec, x_wec, x_opt, waves, nsubsteps)

    def force_on_wec(self,
        wec: TWEC,
        x_wec: ndarray,
        x_opt: ndarray,
        waves: Optional[Dataset] = None,
        nsubsteps: Optional[int] = 1,
    ) -> ndarray:
        """Calculate the PTO force on WEC.

        Parameters
        ----------
        wec
            :py:class:`wecopttool.core.WEC` object.
        x_wec
            WEC dynamic state.
        x_opt
            Optimization (control) state.
        waves
            :py:class:`xarray.Dataset` with the structure and elements
            shown by :py:mod:`wecopttool.waves`.
        nsubsteps
            Number of steps between the default (implied) time steps.
            A value of :python:`1` corresponds to the default step
            length.
        """
        force_td = self.force(wec, x_wec, x_opt, waves, nsubsteps)
        assert force_td.shape == (wec.nt*nsubsteps, self.ndof)
        force_td = np.expand_dims(np.transpose(force_td), axis=0)
        assert force_td.shape == (1, wec.ndof, wec.nt*nsubsteps)
        kinematics_mat = self.kinematics(wec, x_wec, x_opt, waves, nsubsteps)
        kinematics_mat = np.transpose(kinematics_mat, (1,0,2))
        return np.transpose(np.sum(kinematics_mat*force_td, axis=1))

    def mechanical_power(self,
        wec: TWEC,
        x_wec: ndarray,
        x_opt: ndarray,
        waves: Optional[Dataset] = None,
        nsubsteps: Optional[int] = 1,
    ) -> np.ndarray:
        """Calculate the mechanical power time-series in each PTO DOF
        for a given system state.

        Parameters
        ----------
        wec
            :py:class:`wecopttool.core.WEC` object.
        x_wec
            WEC dynamic state.
        x_opt
            Optimization (control) state.
        waves
            :py:class:`xarray.Dataset` with the structure and elements
            shown by :py:mod:`wecopttool.waves`.
        nsubsteps
            Number of steps between the default (implied) time steps.
            A value of :python:`1` corresponds to the default step
            length.
        """
        force_td = self.force(wec, x_wec, x_opt, waves, nsubsteps)
        vel_td = self.velocity(wec, x_wec, x_opt, waves, nsubsteps)
        return vel_td * force_td

    def mechanical_energy(self,
        wec: TWEC,
        x_wec: ndarray,
        x_opt: ndarray,
        waves: Optional[Dataset] = None,
        nsubsteps: Optional[int] = 1,
    ) -> float:
        """Calculate the mechanical energy in each PTO DOF for a given
        system state.

        Parameters
        ----------
        wec
            :py:class:`wecopttool.core.WEC` object.
        x_wec
            WEC dynamic state.
        x_opt
            Optimization (control) state.
        waves
            :py:class:`xarray.Dataset` with the structure and elements
            shown by :py:mod:`wecopttool.waves`.
        nsubsteps
            Number of steps between the default (implied) time steps.
            A value of :python:`1` corresponds to the default step
            length.
        """
        power_td = self.mechanical_power(wec, x_wec, x_opt, waves, nsubsteps)
        return np.sum(power_td) * wec.dt/nsubsteps

    def mechanical_average_power(self,
        wec: TWEC,
        x_wec: ndarray,
        x_opt: ndarray,
        waves: Optional[Dataset] = None,
        nsubsteps: Optional[int] = 1,
    ) -> float:
        """Calculate average mechanical power in each PTO DOF for a
        given system state.

        Parameters
        ----------
        wec
            :py:class:`wecopttool.core.WEC` object.
        x_wec
            WEC dynamic state.
        x_opt
            Optimization (control) state.
        waves
            :py:class:`xarray.Dataset` with the structure and elements
            shown by :py:mod:`wecopttool.waves`.
        nsubsteps
            Number of steps between the default (implied) time steps.
            A value of :python:`1` corresponds to the default step
            length.
        """
        energy = self.mechanical_energy(wec, x_wec, x_opt, waves, nsubsteps)
        return energy / wec.tf

    def power(self,
        wec: TWEC,
        x_wec: ndarray,
        x_opt: ndarray,
        waves: Optional[Dataset] = None,
        nsubsteps: Optional[int] = 1,
    ) -> ndarray:
        """Calculate the power time-series in each PTO DOF for a given
        system state.

        Parameters
        ----------
        wec
            :py:class:`wecopttool.core.WEC` object.
        x_wec
            WEC dynamic state.
        x_opt
            Optimization (control) state.
        waves
            :py:class:`xarray.Dataset` with the structure and elements
            shown by :py:mod:`wecopttool.waves`.
        nsubsteps
            Number of steps between the default (implied) time steps.
            A value of :python:`1` corresponds to the default step
            length.
        """
        # convert e1 (PTO force), q1 (PTO velocity) to e2,q2
        if self.impedance is not None:
            e1_td = self.force(wec, x_wec, x_opt, waves)
            q1_td = self.velocity(wec, x_wec, x_opt, waves)
            q1 = complex_to_real(td_to_fd(q1_td, False))
            e1 = complex_to_real(td_to_fd(e1_td, False))
            vars_1 = np.hstack([q1, e1])
            vars_1_flat = dofmat_to_vec(vars_1)
            vars_2_flat = np.dot(self.transfer_mat, vars_1_flat)
            vars_2 = vec_to_dofmat(vars_2_flat, 2*self.ndof)
            e2 = vars_2[:, self.ndof:]
            q2 = vars_2[:, :self.ndof]
            time_mat = self._tmat(wec, nsubsteps)
            e2_td = np.dot(time_mat, e2)
            q2_td = np.dot(time_mat, q2)
        else:
            # e1_td = self.force(wec, x_wec, x_opt, waves)
            # q1_td = self.velocity(wec, x_wec, x_opt, waves)
            e2_td = self.force(wec, x_wec, x_opt, waves, nsubsteps)
            q2_td = self.velocity(wec, x_wec, x_opt, waves, nsubsteps)
        # power
        power_out = e2_td * q2_td
        if self.loss is not None:
            power_out = power_out * (1-self.loss(e2_td, q2_td))
        return power_out

    def energy(self,
        wec: TWEC,
        x_wec: ndarray,
        x_opt: ndarray,
        waves: Optional[Dataset] = None,
        nsubsteps: Optional[int] = 1,
    ) -> float:
        """Calculate the energy in each PTO DOF for a given system
        state.

        Parameters
        ----------
        wec
            :py:class:`wecopttool.core.WEC` object.
        x_wec
            WEC dynamic state.
        x_opt
            Optimization (control) state.
        waves
            :py:class:`xarray.Dataset` with the structure and elements
            shown by :py:mod:`wecopttool.waves`.
        nsubsteps
            Number of steps between the default (implied) time steps.
            A value of :python:`1` corresponds to the default step
            length.
        """
        power_td = self.power(wec, x_wec, x_opt, waves, nsubsteps)
        return np.sum(power_td) * wec.dt/nsubsteps

    def average_power(self,
        wec: TWEC,
        x_wec: ndarray,
        x_opt: ndarray,
        waves: Optional[Dataset] = None,
        nsubsteps: Optional[int] = 1,
    ) -> float:
        """Calculate the average power in each PTO DOF for a given
        system state.

        Parameters
        ----------
        wec
            :py:class:`wecopttool.core.WEC` object.
        x_wec
            WEC dynamic state.
        x_opt
            Optimization (control) state.
        waves
            :py:class:`xarray.Dataset` with the structure and elements
            shown by :py:mod:`wecopttool.waves`.
        nsubsteps
            Number of steps between the default (implied) time steps.
            A value of :python:`1` corresponds to the default step
            length.
        """
        energy = self.energy(wec, x_wec, x_opt, waves, nsubsteps)
        return energy / wec.tf

    def post_process(self,
        wec: TWEC,
        res: OptimizeResult,
        waves: Optional[DataArray] = None,
        nsubsteps: Optional[int] = 1,
    ) -> tuple[Dataset, Dataset]:
        """Transform the results from optimization solution to a form
        that the user can work with directly.

        Examples
        --------
        The :meth:`wecopttool.core.WEC.solve` method only returns the
        post-processed results for the :py:class:`wecopttool.core.WEC`
        object.

        >>> res_wec_fd, res_wec_td, res_opt = wec.solve(waves=wave,
                                              obj_fun=pto.average_power,
                                              nstate_opt=2*nfreq+1)

        To get the post-processed results for the
        :py:class:`wecopttool.pto.PTO`, you may call

        >>> res_pto_fd, res_pto_td = pto.post_process(wec,res_opt)

        For smoother plots, you can set :python:`nsubsteps` to a value
        greater than 1.

        >>> res_pto_fd, res_pto_td = pto.post_process(wec,res_opt,
                                                      nsubsteps=4)
        >>> res_pto_td.power.plot()

        Parameters
        ----------
        wec
            :py:class:`wecopttool.core.WEC` object.
        res
            Results produced by :py:func:`scipy.optimize.minimize`.
        waves
            :py:class:`xarray.Dataset` with the structure and elements
            shown by :py:mod:`wecopttool.waves`.
        nsubsteps
            Number of steps between the default (implied) time steps.
            A value of :python:`1` corresponds to the default step
            length.

        Returns
        -------
        results_fd
            :py:class:`xarray.Dataset` with frequency domain results.
        results_td
            :py:class:`xarray.Dataset` with time domain results.
        """
        create_time = f"{datetime.utcnow()}"
        
        x_wec, x_opt = wec.decompose_state(res.x)

        # position
        pos_td = self.position(wec, x_wec, x_opt, waves, nsubsteps)
        pos_fd = wec.td_to_fd(pos_td[::nsubsteps])

        # velocity
        vel_td = self.velocity(wec, x_wec, x_opt, waves, nsubsteps)
        vel_fd = wec.td_to_fd(vel_td[::nsubsteps])

        # acceleration
        acc_td = self.acceleration(wec, x_wec, x_opt, waves, nsubsteps)
        acc_fd = wec.td_to_fd(acc_td[::nsubsteps])

        # force
        force_td = self.force(wec, x_wec, x_opt, waves, nsubsteps)
        force_fd = wec.td_to_fd(force_td[::nsubsteps])

        # power
        power_td = self.power(wec, x_wec, x_opt, waves, nsubsteps)
        power_fd = wec.td_to_fd(power_td[::nsubsteps])

        # mechanical power
        mech_power_td = self.mechanical_power(wec, x_wec, x_opt, waves,
                                              nsubsteps)
        mech_power_fd = wec.td_to_fd(mech_power_td[::nsubsteps])

        pos_attr = {'long_name': 'Position', 'units': 'm or rad'}
        vel_attr = {'long_name': 'Velocity', 'units': 'm/s or rad/s'}
        acc_attr = {'long_name': 'Acceleration', 'units': 'm/s^2 or rad/s^2'}
        force_attr = {'long_name': 'Force or moment on WEC',
                      'units': 'N or Nm'}
        power_attr = {'long_name': 'Power', 'units': 'W'}
        mech_power_attr = {'long_name': 'Mechanical power', 'units': 'W'}
        omega_attr = {'long_name': 'Radial frequency', 'units': 'rad/s'}
        freq_attr = {'long_name': 'Frequency', 'units': 'Hz'}
        period_attr = {'long_name': 'Period', 'units': 's'}
        dof_attr = {'long_name': 'PTO degree of freedom'}
        time_attr = {'long_name': 'Time', 'units': 's'}

        t_dat = wec.time_nsubsteps(nsubsteps)

        results_fd = Dataset(
            data_vars={
                'pos': (['omega','dof'], pos_fd, pos_attr),
                'vel': (['omega','dof'], vel_fd, vel_attr),
                'acc': (['omega','dof'], acc_fd, acc_attr),
                'force': (['omega','dof'], force_fd, force_attr),
                'power': (['omega','dof'], power_fd, power_attr),
                'mech_power': (['omega','dof'], mech_power_fd, mech_power_attr)
            },
            coords={
                'omega':('omega', wec.omega, omega_attr),
                'freq':('omega', wec.frequency, freq_attr),
                'period':('omega', wec.period, period_attr),
                'dof':('dof', self.names, dof_attr)},
            attrs={"time_created_utc": create_time}
            )

        results_td = Dataset(
            data_vars={
                'pos': (['time','dof'], pos_td, pos_attr),
                'vel': (['time','dof'], vel_td, vel_attr),
                'acc': (['time','dof'], acc_td, acc_attr),
                'force': (['time','dof'], force_td, force_attr),
                'power': (['time','dof'], power_td, power_attr),
                'mech_power': (['time','dof'], mech_power_td, mech_power_attr)
            },
            coords={
                'time':('time', t_dat, time_attr),
                'dof':('dof', self.names, dof_attr)},
            attrs={"time_created_utc": create_time}
            )

        return results_fd, results_td


# power conversion chain
def _make_abcd(impedance: ndarray, ndof: int) -> ndarray:
    """Transform the impedance matrix into ABCD form from a MIMO
    transfer function.

    Parameters
    ----------
    impedance
        Matrix representing the PTO impedance.
        Size 2*n_dof.
    ndof
        Number of degrees of freedom.
        Must be specified if :python:`inertia_in_forces is True`, else
        not used.
    """
    z_11 = impedance[:ndof, :ndof, :]  # Fu
    z_12 = impedance[:ndof, ndof:, :]  # Fi
    z_21 = impedance[ndof:, :ndof, :]  # Vu
    z_22 = impedance[ndof:, ndof:, :]  # Vi
    z_12_inv = np.linalg.inv(z_12.T).T

    mmult = lambda a,b: np.einsum('mnr,mnr->mnr', a, b)
    abcd_11 = -1 * mmult(z_12_inv, z_11)
    abcd_12 = z_12_inv
    abcd_21 = z_21 - mmult(z_22, mmult(z_12_inv, z_11))
    abcd_22 = mmult(z_22, z_12_inv)
    return np.block([[[abcd_11], [abcd_12]], [[abcd_21], [abcd_22]]])


def _make_mimo_transfer_mat(
    impedance_abcd: ndarray,
    ndof: int,
) -> np.ndarray:
    """Create a block matrix of a MIMO transfer function.

    Parameters
    ----------
    impedance
        PTO impedance in ABCD form.
    ndof
        Number of degrees of freedom.
        Must be specified if :python:`inertia_in_forces is True`, else
        not used.
    """
    elem = [[None]*2*ndof for _ in range(2*ndof)]
    def block(re, im): return np.array([[re, -im], [im, re]])
    for idof in range(2*ndof):
        for jdof in range(2*ndof):
            Zp = impedance_abcd[idof, jdof, :]
            re = np.real(Zp)
            im = np.imag(Zp)
            blocks = [block(ire, iim) for (ire, iim) in zip(re, im)]
            blocks = [0.0] + blocks
            elem[idof][jdof] = block_diag(*blocks)
    return np.block(elem)


# controllers
def controller_unstructured(
    pto: TPTO,
    wec: TWEC,
    x_wec: ndarray,
    x_opt: ndarray,
    waves: Optional[Dataset] = None,
    nsubsteps: Optional[int] = 1,
) -> ndarray:
    """Unstructured numerical optimal controller that returns a time
    history of PTO forces.

    Parameters
    ----------
    pto
        :py:class:`wecopttool.pto.PTO` object.
    wec
        :py:class:`wecopttool.core.WEC` object.
    x_wec
        WEC dynamic state.
    x_opt
        Optimization (control) state.
    waves
        :py:class:`xarray.Dataset` with the structure and elements
        shown by :py:mod:`wecopttool.waves`.
    nsubsteps
        Number of steps between the default (implied) time steps.
        A value of :python:`1` corresponds to the default step
        length.
    """
    x_opt = np.reshape(x_opt, (-1, pto.ndof), order='F')
    tmat = pto._tmat(wec, nsubsteps)
    return np.dot(tmat, x_opt)


def controller_pid(
    pto: TPTO,
    wec: TWEC,
    x_wec: ndarray,
    x_opt: ndarray,
    waves: Optional[Dataset] = None,
    nsubsteps: Optional[int] = 1,
    proportional: Optional[bool] = True,
    integral: Optional[bool] = True,
    derivative: Optional[bool] = True,
) -> ndarray:
    """Proportional-integral-derivative (PID) controller that returns
    a time history of PTO forces.

    Parameters
    ----------
    pto
        :py:class:`wecopttool.pto.PTO` object.
    wec
        :py:class:`wecopttool.core.WEC` object.
    x_wec
        WEC dynamic state.
    x_opt
        Optimization (control) state.
    waves
        :py:class:`xarray.Dataset` with the structure and elements shown
        by :py:mod:`wecopttool.waves`.
    nsubsteps
            Number of steps between the default (implied) time steps.
            A value of :python:`1` corresponds to the default step
            length.
    proportional
        True to include proportional gain
    integral
        True to include integral gain
    derivative
        True to include derivative gain
    """
    ndof = pto.ndof
    force_td = np.zeros([wec.nt, ndof])
    idx = 0

    def update_force_td(response):
        nonlocal idx, force_td
        gain = np.reshape(x_opt[idx*ndof:(idx+1)*ndof], [1, ndof])
        force_td = force_td + gain*response
        idx = idx + 1

    if proportional:
        vel_td = pto.velocity(wec, x_wec, x_opt, waves, nsubsteps)
        update_force_td(vel_td)
    if integral:
        pos_td = pto.position(wec, x_wec, x_opt, waves, nsubsteps)
        update_force_td(pos_td)
    if derivative:
        acc_td = pto.acceleration(wec, x_wec, x_opt, waves, nsubsteps)
        update_force_td(acc_td)
    return force_td


def controller_pi(
    pto: TPTO,
    wec: TWEC,
    x_wec: ndarray,
    x_opt: ndarray,
    waves: Optional[Dataset] = None,
    nsubsteps: Optional[int] = 1,
) -> ndarray:
    """Proportional-integral (PI) controller that returns a time
    history of PTO forces.

    Parameters
    ----------
    pto
        :py:class:`wecopttool.pto.PTO` object.
    wec
        :py:class:`wecopttool.core.WEC` object.
    x_wec
        WEC dynamic state.
    x_opt
        Optimization (control) state.
    waves
        :py:class:`xarray.Dataset` with the structure and elements shown
        by :py:mod:`wecopttool.waves`.
    nsubsteps
            Number of steps between the default (implied) time steps.
            A value of :python:`1` corresponds to the default step
            length.
    """
    force_td = controller_pid(pto, wec, x_wec, x_opt, waves, nsubsteps,
                              True, True, False)
    return force_td


def controller_p(
    pto: TPTO,
    wec: TWEC,
    x_wec: ndarray,
    x_opt: ndarray,
    waves: Optional[Dataset] = None,
    nsubsteps: Optional[int] = 1,
) -> ndarray:
    """Proportional (P) controller that returns a time history of
    PTO forces.

    Parameters
    ----------
    pto
        :py:class:`wecopttool.pto.PTO` object.
    wec
        :py:class:`wecopttool.core.WEC` object.
    x_wec
        WEC dynamic state.
    x_opt
        Optimization (control) state.
    waves
        :py:class:`xarray.Dataset` with the structure and elements shown
        by :py:mod:`wecopttool.waves`.
    nsubsteps
        Number of steps between the default (implied) time steps.
        A value of :python:`1` corresponds to the default step length.
    """
    force_td = controller_pid(pto, wec, x_wec, x_opt, waves, nsubsteps,
                               True, False, False)
    return force_td