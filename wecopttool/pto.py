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


__all__ = [
    "PTO",
    "controller_unstructured",
    "controller_pid",
    "controller_pi",
    "controller_p",
]


from typing import Optional, TypeVar, Callable, Union

import autograd.numpy as np
from autograd.builtins import isinstance, tuple, list, dict
from autograd.numpy import ndarray
from scipy.linalg import block_diag
from scipy.optimize import OptimizeResult
from xarray import DataArray, Dataset
import datetime
from scipy.optimize import OptimizeResult

from wecopttool.core import complex_to_real, td_to_fd
from wecopttool.core import dofmat_to_vec, vec_to_dofmat
from wecopttool.core import TWEC, TStateFunction, FloatOrArray


# type aliases
TPTO = TypeVar("TPTO", bound="PTO")
TLOSS = Callable[[FloatOrArray, FloatOrArray], FloatOrArray]


class PTO:
    """A power take-off (PTO) object to be used in conjunction with a
    :py:class:`wecopttool.WEC` object.
    """

    def __init__(self,
        ndof: int,
        kinematics: Union[TStateFunction, ndarray],
        controller: Optional[TStateFunction] = None,
        impedance: Optional[ndarray] = None,
        loss: Optional[TLOSS] = None,
        names: Optional[list[str]] = None,
    ) -> None:
        """Create a PTO object.

        The :py:class:`wecopttool.pto.PTO` class describes the
        kinematics, control logic, impedance and/or non-linear power
        loss of a power take-off system.
        The forces/moments applied by a
        :py:class:`wecopttool.pto.PTO` object can be applied to a
        :py:class:`wecopttool.WEC` object through the
        :py:attr:`wecopttool.WEC.f_add` property.
        The power produced by a :py:class:`wecopttool.pto.PTO` object
        can be used for the :python:`obj_fun` of pseudo-spectral
        optimization problem when calling
        :py:meth:`wecopttool.WEC.solve`.

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
            :python:`def fun(pto, wec, x_wec, x_opt, waves, nsubsteps):`
            or matrix with shape (PTO DOFs, WEC DOFs) that converts
            from the WEC DOFs to the PTO DOFs.
        impedance
            Matrix representing the PTO impedance.
        loss
            Function that maps flow and effort variables to a
            non-linear power loss.
            The output is the dissipated power (loss) in Watts.
            This should be a positive value.
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
        if impedance is not None:
            check_1 = impedance.shape[0] == impedance.shape[1] == 2*self.ndof
            check_2 = len(impedance.shape) == 3
            if not (check_1 and check_2):
                raise TypeError(
                    "Impedance should have size [2*ndof, 2*ndof, nfreq]"
                )
            for i in range(impedance.shape[2]-1):
                check_3 = (
                    np.allclose(np.real(impedance[:, :, i+1]), np.real(impedance[:, :, 0]))
                )
                if not check_3:
                    raise ValueError(
                        "Real component of impedance must be constant for " +
                        "all frequencies."
                    )
            impedance_abcd = _make_abcd(impedance, ndof)
            self._transfer_mat = _make_mimo_transfer_mat(impedance_abcd, ndof)
        else:
            self._transfer_mat = None
        self._impedance = impedance
        self._loss = loss

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
    def loss(self) -> TLOSS:
        """Nonlinear power loss function with outputs in Watts."""
        return self._loss

    @property
    def transfer_mat(self) -> ndarray:
        """Transfer matrix."""
        return self._transfer_mat

    def _tmat(self, wec, nsubsteps: Optional[int] = 1):
        if nsubsteps==1:
            tmat = wec.time_mat
        else:
            tmat = wec.time_mat_nsubsteps(nsubsteps)
        return tmat

    def _fkinematics(self,
        f_wec: ndarray,
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
            :py:class:`wecopttool.WEC` object.
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
            :py:class:`wecopttool.WEC` object.
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
            :py:class:`wecopttool.WEC` object.
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
            :py:class:`wecopttool.WEC` object.
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
        acc_wec = np.dot(wec.derivative2_mat, pos_wec)
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
            :py:class:`wecopttool.WEC` object.
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
        assert force_td.shape == (1, self.ndof, wec.nt*nsubsteps)
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
            :py:class:`wecopttool.WEC` object.
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
            :py:class:`wecopttool.WEC` object.
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
            :py:class:`wecopttool.WEC` object.
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

    def power_variables(self,
        wec: TWEC,
        x_wec: ndarray,
        x_opt: ndarray,
        waves: Optional[Dataset] = None,
        nsubsteps: Optional[int] = 1,
    ) -> tuple[ndarray, ndarray]:
        """Calculate the power variables (flow q and effort e) time-series
        in each PTO DOF for a given system state.

        Parameters
        ----------
        wec
            :py:class:`wecopttool.WEC` object.
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
        # convert q1 (PTO velocity), e1 (PTO force)
        # to q2 (flow variable), e2 (effort variable)
        if self.impedance is not None:
            q1_td = self.velocity(wec, x_wec, x_opt, waves)
            e1_td = self.force(wec, x_wec, x_opt, waves)
            q1 = complex_to_real(td_to_fd(q1_td, False))
            e1 = complex_to_real(td_to_fd(e1_td, False))
            vars_1 = np.hstack([q1, e1])
            vars_1_flat = dofmat_to_vec(vars_1)
            vars_2_flat = np.dot(self.transfer_mat, vars_1_flat)
            vars_2 = vec_to_dofmat(vars_2_flat, 2*self.ndof)
            q2 = vars_2[:, :self.ndof]
            e2 = vars_2[:, self.ndof:]
            time_mat = self._tmat(wec, nsubsteps)
            q2_td = np.dot(time_mat, q2)
            e2_td = np.dot(time_mat, e2)
        else:
            q2_td = self.velocity(wec, x_wec, x_opt, waves, nsubsteps)
            e2_td = self.force(wec, x_wec, x_opt, waves, nsubsteps)
        return q2_td, e2_td

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
            :py:class:`wecopttool.WEC` object.
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
        q2_td, e2_td = self.power_variables(wec, x_wec,
                                            x_opt, waves, nsubsteps)
        # power
        power_out = q2_td * e2_td
        if self.loss is not None:
            power_out = power_out + self.loss(q2_td, e2_td)
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
            :py:class:`wecopttool.WEC` object.
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
            :py:class:`wecopttool.WEC` object.
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

    def transduced_flow(self,
        wec: TWEC,
        x_wec: ndarray,
        x_opt: ndarray,
        waves: Optional[Dataset] = None,
        nsubsteps: Optional[int] = 1,
    ) -> float:
        """Calculate the transduced flow variable time-series in each PTO DOF
        for a given system state. Equals the PTO velocity if no impedance
        is defined.

        Examples for PTO impedance and corresponding flow variables:

        - OWC: (pneumatic admittance)^-1 : flow = volumetric air flow

        - Drive-train: rotational impedance : flow = rotational velocity

        - Generator: winding impedance: flow = electric current

        - Drive-train and Generator combined: flow = electric current

        Parameters
        ----------
        wec
            :py:class:`wecopttool.WEC` object.
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
        q2_td, _ = self.power_variables(wec, x_wec,
                                        x_opt, waves, nsubsteps)
        return q2_td

    def transduced_effort(self,
        wec: TWEC,
        x_wec: ndarray,
        x_opt: ndarray,
        waves: Optional[Dataset] = None,
        nsubsteps: Optional[int] = 1,
    ) -> float:
        """Calculate the transduced flow variable time-series in each PTO DOF
        for a given system state. Equals the PTO force if no impedance
        is defined.

        Examples for PTO impedance and corresponding effort variables:

        - OWC: (pneumatic admittance)^-1 : effort =  air pressure

        - Drive-train: rotational impedance : effort = torque

        - Generator: winding impedance: effort = voltage

        - Drive-train and Generator combined: effort = voltage

        Parameters
        ----------
        wec
            :py:class:`wecopttool.WEC` object.
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
        _, e2_td = self.power_variables(wec, x_wec, x_opt, waves, nsubsteps)
        return e2_td

    def post_process(self,
        wec: TWEC,
        res: Union[OptimizeResult, list],
        waves: Optional[DataArray] = None,
        nsubsteps: Optional[int] = 1,
    ) -> tuple[list[Dataset], list[Dataset]]:
        """Transform the results from optimization solution to a form
        that the user can work with directly.

        Examples
        --------
        The :py:meth:`wecopttool.WEC.solve` method only returns the
        raw results dictionary produced by :py:func:`scipy.optimize.minimize`.

        >>> res_opt = wec.solve(waves=wave,
                                obj_fun=pto.average_power,
                                nstate_opt=2*nfreq+1)

        To get the post-processed results for the
        :py:class:`wecopttool.pto.PTO`, you may call

        >>> res_pto_fd, res_pto_td = pto.post_process(wec,res_opt[0],wave)

        For smoother plots, you can set :python:`nsubsteps` to a value
        greater than 1.

        >>> res_pto_fd, res_pto_td = pto.post_process(wec,res_opt,
                                                      nsubsteps=4)
        >>> res_pto_td[0].power.plot()

        Parameters
        ----------
        wec
            :py:class:`wecopttool.WEC` object.
        res
            Results produced by :py:meth:`wecopttool.WEC.solve`.
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
            list of :py:class:`xarray.Dataset` with frequency domain results.
        results_td
            list of :py:class:`xarray.Dataset` with time domain results.
        """
        def _postproc(wec, res, waves, nsubsteps):

            create_time = f"{datetime.datetime.now(datetime.UTC)}"

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
            elec_power_td = self.power(wec, x_wec, x_opt, waves, nsubsteps)
            elec_power_fd = wec.td_to_fd(elec_power_td[::nsubsteps])

            # mechanical power
            mech_power_td = self.mechanical_power(wec, x_wec, x_opt, waves,
                                                nsubsteps)
            mech_power_fd = wec.td_to_fd(mech_power_td[::nsubsteps])

            # stack mechanical and electrical power
            power_names = ['mech','elec']
            power_fd = np.stack((mech_power_fd,elec_power_fd))
            power_td = np.stack((mech_power_td,elec_power_td))

            pos_attr = {'long_name': 'Position', 'units': 'm or rad'}
            vel_attr = {'long_name': 'Velocity', 'units': 'm/s or rad/s'}
            acc_attr = {'long_name': 'Acceleration',
                        'units': 'm/s^2 or rad/s^2'}
            force_attr = {'long_name': 'PTO force or torque',
                        'units': 'N or Nm'}
            power_attr = {'long_name': 'Power', 'units': 'W'}
            mech_power_attr = {'long_name': 'Mechanical power', 'units': 'W'}
            omega_attr = {'long_name': 'Radial frequency', 'units': 'rad/s'}
            freq_attr = {'long_name': 'Frequency', 'units': 'Hz'}
            period_attr = {'long_name': 'Period', 'units': 's'}
            dof_attr = {'long_name': 'PTO degree of freedom'}
            time_attr = {'long_name': 'Time', 'units': 's'}
            type_attr = {'long_name': 'Power type'}

            t_dat = wec.time_nsubsteps(nsubsteps)

            results_fd = Dataset(
                data_vars={
                    'pos': (['omega','dof'], pos_fd, pos_attr),
                    'vel': (['omega','dof'], vel_fd, vel_attr),
                    'acc': (['omega','dof'], acc_fd, acc_attr),
                    'force': (['omega','dof'], force_fd, force_attr),
                    'power': (['type','omega','dof'], power_fd, power_attr),
                },
                coords={
                    'omega':('omega', wec.omega, omega_attr),
                    'freq':('omega', wec.frequency, freq_attr),
                    'period':('omega', wec.period, period_attr),
                    'dof':('dof', self.names, dof_attr),
                    'type':('type', power_names, power_attr)},
                attrs={"time_created_utc": create_time}
                )

            results_td = Dataset(
                data_vars={
                    'pos': (['time','dof'], pos_td, pos_attr),
                    'vel': (['time','dof'], vel_td, vel_attr),
                    'acc': (['time','dof'], acc_td, acc_attr),
                    'force': (['time','dof'], force_td, force_attr),
                    'power': (['type','time','dof'], power_td, power_attr),
                },
                coords={
                    'time':('time', t_dat, time_attr),
                    'dof':('dof', self.names, dof_attr),
                    'type':('type', power_names, power_attr)},
                attrs={"time_created_utc": create_time}
                )

            if self.impedance is not None:
            #transduced flow and effort variables
                q2_td, e2_td = self.power_variables(wec, x_wec, x_opt,
                                                    waves, nsubsteps)
                q2_fd = wec.td_to_fd(q2_td[::nsubsteps])
                e2_fd = wec.td_to_fd(e2_td[::nsubsteps])

                q2_attr = {'long_name': 'Transduced Flow',
                        'units': 'A or m^3/s or rad/s or m/s'}
                e2_attr = {'long_name': 'Transduced Effort',
                        'units': 'V or N/m^2 or Nm or Ns'}

                results_td = results_td.assign({
                            'trans_flo': (['time','dof'], q2_td, q2_attr),
                            'trans_eff': (['time','dof'], e2_td, e2_attr),
                        })
                results_fd = results_fd.assign({
                            'trans_flo': (['omega','dof'], q2_fd, q2_attr),
                            'trans_eff': (['omega','dof'], e2_fd, e2_attr),
                        })


            return results_fd, results_td

        results_fd = []
        results_td = []
        for idx, ires in enumerate(res):
            ifd, itd = _postproc(wec, ires, waves.sel(realization=idx), nsubsteps)
            results_fd.append(ifd)
            results_td.append(itd)
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

    row_1 = np.hstack([abcd_11, abcd_12])
    row_2 = np.hstack([abcd_21, abcd_22])
    return np.vstack([row_1, row_2])


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
    """
    def block(re, im): return np.array([[re, -im], [im, re]])
    for idof in range(2*ndof):
        for jdof in range(2*ndof):
            Zp = impedance_abcd[idof, jdof, :]
            re = np.real(Zp)
            im = np.imag(Zp)
            # Exclude the sine component of the 2-point wave
            blocks = [block(ire, iim) for (ire, iim) in zip(re[:-1], im[:-1])]
            # re[0] added for the zero frequency power loss (DC), could be re[n]
            blocks = [re[0]] + blocks + [re[-1]]
            if jdof==0:
                row = block_diag(*blocks)
            else:
                row = np.hstack([row, block_diag(*blocks)])
        if idof==0:
            mat = row
        else:
            mat = np.vstack([mat, row])
    return mat


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
        :py:class:`wecopttool.WEC` object.
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
    tmat = pto._tmat(wec, nsubsteps)
    x_opt = np.reshape(x_opt[:len(tmat[0])*pto.ndof], (-1, pto.ndof), order='F')
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
    saturation: Optional[FloatOrArray] = None,
) -> ndarray:
    """Proportional-integral-derivative (PID) controller that returns
    a time history of PTO forces.

    Parameters
    ----------
    pto
        :py:class:`wecopttool.pto.PTO` object.
    wec
        :py:class:`wecopttool.WEC` object.
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
    proportional
        True to include proportional gain
    integral
        True to include integral gain
    derivative
        True to include derivative gain
    saturation
        Maximum and minimum control value.
        Can be symmetric ([ndof]) or asymmetric ([ndof, 2]).
    """
    ndof = pto.ndof
    force_td_tmp = np.zeros([wec.nt*nsubsteps, ndof])

    # PID force
    idx = 0

    def update_force_td(response):
        nonlocal idx, force_td_tmp
        gain = np.diag(x_opt[idx*ndof:(idx+1)*ndof])
        force_td_tmp = force_td_tmp + np.dot(response, gain.T)
        idx = idx + 1
        return

    if proportional:
        vel_td = pto.velocity(wec, x_wec, x_opt, waves, nsubsteps)
        update_force_td(vel_td)
    if integral:
        pos_td = pto.position(wec, x_wec, x_opt, waves, nsubsteps)
        update_force_td(pos_td)
    if derivative:
        acc_td = pto.acceleration(wec, x_wec, x_opt, waves, nsubsteps)
        update_force_td(acc_td)

    # Saturation
    if saturation is not None:
        saturation = np.atleast_2d(np.squeeze(saturation))
        assert len(saturation)==ndof
        if len(saturation.shape) > 2:
            raise ValueError("`saturation` must have <= 2 dimensions.")
        if saturation.shape[1] == 1:
            f_min, f_max = -1*saturation, saturation
        elif saturation.shape[1] == 2:
            f_min, f_max = saturation[:,0], saturation[:,1]
        else:
            raise ValueError("`saturation` must have 1 or 2 columns.")

        force_td_list = []
        for i in range(ndof):
            tmp = np.clip(force_td_tmp[:,i], f_min[i], f_max[i])
            force_td_list.append(tmp)
        force_td = np.array(force_td_list).T
    else:
        force_td = force_td_tmp

    return force_td


def controller_pi(
    pto: TPTO,
    wec: TWEC,
    x_wec: ndarray,
    x_opt: ndarray,
    waves: Optional[Dataset] = None,
    nsubsteps: Optional[int] = 1,
    saturation: Optional[FloatOrArray] = None,
) -> ndarray:
    """Proportional-integral (PI) controller that returns a time
    history of PTO forces.

    Parameters
    ----------
    pto
        :py:class:`wecopttool.pto.PTO` object.
    wec
        :py:class:`wecopttool.WEC` object.
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
    saturation
        Maximum and minimum control value.
        Can be symmetric ([ndof]) or asymmetric ([ndof, 2]).
    """
    force_td = controller_pid(
        pto, wec, x_wec, x_opt, waves, nsubsteps,
        True, True, False, saturation,
    )
    return force_td


def controller_p(
    pto: TPTO,
    wec: TWEC,
    x_wec: ndarray,
    x_opt: ndarray,
    waves: Optional[Dataset] = None,
    nsubsteps: Optional[int] = 1,
    saturation: Optional[FloatOrArray] = None,
) -> ndarray:
    """Proportional (P) controller that returns a time history of
    PTO forces.

    Parameters
    ----------
    pto
        :py:class:`wecopttool.pto.PTO` object.
    wec
        :py:class:`wecopttool.WEC` object.
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
    saturation
        Maximum and minimum control value. Can be symmetric ([ndof]) or
        asymmetric ([ndof, 2]).
    """
    force_td = controller_pid(
        pto, wec, x_wec, x_opt, waves, nsubsteps,
        True, False, False, saturation,
    )
    return force_td


# utilities
def nstate_unstructured(nfreq: int, ndof: int) -> int:
    """
    Number of states needed to represent an unstructured controller.

    Parameters
    ----------
    nfreq
        Number of frequencies.
    ndof
        Number of degrees of freedom.
    """
    return 2*nfreq*ndof


def nstate_pid(
        nterm: int,
        ndof: int,
) -> int:
    """
    Number of states needed to represent an unstructured controller.

    Parameters
    ----------
    nterm
        Number of terms (e.g. 1 for P, 2 for PI, 3 for PID).
    ndof
        Number of degrees of freedom.
    """
    return int(nterm*ndof)
