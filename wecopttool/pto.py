
"""Provide power take-off (PTO) forces and produced energy functions
for common PTO control approaches.

The PTO produced energy can be used as the objective function for the
control optimization.
The PTO force can be included as an additional force in the WEC
dynamics.
"""


from __future__ import annotations  # TODO: delete after python 3.10
from typing import Callable

import numpy.typing as npt
import autograd.numpy as np
from autograd.builtins import isinstance, tuple, list, dict
import xarray as xr

from wecopttool.core import WEC, real_to_complex_amplitudes


class _PTO:
    """Base PTO class for subclassing."""

    def __init__(self, kinematics: np.ndarray, names: list[str] | None = None
                 ) -> None:
        """
        Parameters
        ----------
        kinematics: np.ndarray
            Matrix that converts from the WEC DOFs to the PTO DOFs.
            Shape: (PTO DOFs, WEC DOFs).
        names: list, optional
            Names of the PTOs in each DOF. List of strings of length PTO
            DOFs.
        """
        self.kinematics = kinematics
        # names
        if names is None:
            names = [f'pto_{i+1}' for i in range(self.ndof)]
        self.names = names

    @property
    def ndof(self):
        """Number of PTO degrees of freedom."""
        return self.kinematics.shape[0]

    @property
    def ndof_wec(self):
        """Number of WEC degrees of freedom."""
        return self.kinematics.shape[1]

    @property
    def nstate(self):
        """Total number of PTO states."""
        return self.nstate_per_dof * self.ndof

    @property
    def _kinematics_t(self):
        return np.transpose(self.kinematics)

    def _pto_to_wec_dofs(self, pto):
        return np.dot(pto, self.kinematics)

    def _wec_to_pto_dofs(self, pto):
        return np.dot(pto, self._kinematics_t)

    def _dofmat_to_vec(self, mat: np.ndarray) -> np.ndarray:
        """Flatten a matrix that has one column per DOF.
        Opposite of ``vec_to_dofmat``. """
        return np.reshape(mat, -1, order='F')

    def _vec_to_dofmat(self, vec: np.ndarray) -> np.ndarray:
        """Convert a vector back to a matrix with one column per DOF.
        Opposite of ``dofmat_to_vec``.
        """
        return np.reshape(vec, (self.nstate_per_dof, self.ndof), order='F')

    def _pseudo_spectral(self, wec: WEC, x_opt: npt.ArrayLike,
                         nsubsteps: int = 1) -> np.ndarray:
        x = self._vec_to_dofmat(x_opt)
        time_mat = wec.make_time_mat(nsubsteps, include_mean=False)
        return np.dot(time_mat, x)

    def position(self, wec: WEC, x_wec: npt.ArrayLike, x_opt: npt.ArrayLike,
                 nsubsteps: int = 1) -> np.ndarray:
        """Calculate the PTO position time-series."""
        wec_pos = wec.vec_to_dofmat(x_wec)
        pos = self._wec_to_pto_dofs(wec_pos)
        time_mat = wec.make_time_mat(nsubsteps)
        return np.dot(time_mat, pos)

    def velocity(self, wec: WEC, x_wec: npt.ArrayLike, x_opt: npt.ArrayLike,
                 nsubsteps: int = 1) -> np.ndarray:
        """Calculate the PTO velocity time-series."""
        wec_pos = wec.vec_to_dofmat(x_wec)
        wec_vel = np.dot(wec.derivative_mat, wec_pos)
        vel = self._wec_to_pto_dofs(wec_vel)
        time_mat = wec.make_time_mat(nsubsteps)
        return np.dot(time_mat, vel)

    def acceleration(self, wec: WEC, x_wec: npt.ArrayLike,
                     x_opt: npt.ArrayLike, nsubsteps: int = 1
                     ) -> np.ndarray:
        """Calculate the PTO acceleration time-series."""
        wec_pos = wec.vec_to_dofmat(x_wec)
        wec_vel = np.dot(wec.derivative_mat, wec_pos)
        wec_acc = np.dot(wec.derivative_mat, wec_vel)
        acc = self._wec_to_pto_dofs(wec_acc)
        time_mat = wec.make_time_mat(nsubsteps)
        return np.dot(time_mat, acc)

    def power(self, wec: WEC, x_wec: npt.ArrayLike, x_opt: npt.ArrayLike,
              nsubsteps: int = 1) -> np.ndarray:
        """Calculate the PTO power time-series in each PTO DOF
        for a given system state.
        """
        force_td = self.force(wec, x_wec, x_opt, nsubsteps)
        vel_td = self.velocity(wec, x_wec, x_opt, nsubsteps)
        return vel_td * force_td

    def force_on_wec(self, wec: WEC, x_wec: npt.ArrayLike, x_opt: npt.ArrayLike,
                     nsubsteps: int = 1) -> np.ndarray:
        """Calculate the PTO forces time-series in the WEC DOFs for a
        given system state.

        See ``force``.
        """
        fpto_td = self.force(wec, x_wec, x_opt, nsubsteps)
        return self._pto_to_wec_dofs(fpto_td)

    def energy(self, wec: WEC, x_wec: npt.ArrayLike, x_opt: npt.ArrayLike,
               nsubsteps: int = 1) -> float:
        """Calculate the energy (in Joules) by the PTO during
        the period t=0-T=1/f0.

        Parameters
        ----------
        wec: WEC
            The WEC.
        x_wec: np.ndarray
            WEC dynamics state.
        x_opt: np.ndarray
            Optimization (PTO controller) state.
        nsubsteps: int
            Number of subdivisions between the default (implied) time
            steps.

        Returns
        -------
        float
            Energy (in Joules) over the period t=0-T.
        """
        power_td = self.power(wec, x_wec, x_opt, nsubsteps)
        return np.sum(power_td) * wec.dt/nsubsteps

    def average_power(self, wec: WEC, x_wec: npt.ArrayLike,
                      x_opt: npt.ArrayLike, nsubsteps: int = 1) -> float:
        """Calculate the average power of the PTO for the given wave
        spectrum.

        Parameters
        ----------
        wec: WEC
            The WEC.
        x_wec: np.ndarray
            WEC dynamics state.
        x_opt: np.ndarray
            Optimization (PTO controller) state.
        nsubsteps: int
            Number of subdivisions between the default (implied) time
            steps.

        Returns
        -------
        float
            Average power (in Watts).
        """
        return self.energy(wec, x_wec, x_opt, nsubsteps) / wec.tf

    def post_process(self, wec: WEC, x_wec: npt.ArrayLike, x_opt: npt.ArrayLike
                     ) -> tuple[xr.Dataset, xr.Dataset]:
        """Transform the results from optimization solution to a form
        that the user can work with directly.
        """
        # position
        wec_pos = wec.vec_to_dofmat(x_wec)
        pos = wec_pos @ self._kinematics_t
        pos_fd = real_to_complex_amplitudes(pos)
        pos_td = wec.time_mat @ pos

        # velocity
        vel = wec.derivative_mat @ pos
        vel_fd = real_to_complex_amplitudes(vel)
        vel_td = wec.time_mat @ vel

        # acceleration
        acc = wec.derivative_mat @ vel
        acc_fd = real_to_complex_amplitudes(acc)
        acc_td = wec.time_mat @ acc

        # force
        force_td = self.force(wec, x_wec, x_opt)
        force_fd = wec.td_to_fd(force_td)

        # power
        power_td = self.power(wec, x_wec, x_opt)
        power_fd = wec.td_to_fd(power_td)

        # assemble time-domain xarray
        dims = ['time', 'dof_pto']
        coords = [(dims[0], wec.time, {'units': 's'}), (dims[1], self.names)]
        attrs_f = {'long_name': 'PTO force', 'units': 'N or N*m'}
        attrs_p = {'long_name': 'PTO power', 'units': 'W'}
        attrs_pos = {'long_name': 'PTO position', 'units': 'm or (rad)'}
        attrs_vel = {'long_name': 'PTO velocity', 'units': 'm/s or (rad)/s'}
        attrs_acc = {'long_name': 'PTO acceleration',
                    'units': 'm/s^2 or (rad)/s^2'}
        force_td = xr.DataArray(
            force_td, dims=dims, coords=coords, attrs=attrs_f)
        power_td = xr.DataArray(
            power_td, dims=dims, coords=coords, attrs=attrs_p)
        pos_td = xr.DataArray(
            pos_td, dims=dims, coords=coords, attrs=attrs_pos)
        vel_td = xr.DataArray(
            vel_td, dims=dims, coords=coords, attrs=attrs_vel)
        acc_td = xr.DataArray(
            acc_td, dims=dims, coords=coords, attrs=attrs_acc)
        time_dom = xr.Dataset({'pos': pos_td, 'vel': vel_td, 'acc': acc_td,
                            'force': force_td, 'power': power_td},)

        # assemble frequency-domain xarray
        omega = np.concatenate([np.array([0.0]), wec.omega])
        dims[0] = 'omega'
        coords[0] = (dims[0], omega, {'units': '(rad)'})
        attrs_f['units'] = 'N^2*s'
        attrs_p['units'] = 'W^2*s'
        attrs_pos['units'] = 'm^2*s or (rad)^2*s'
        attrs_vel['units'] = 'm^2/s or (rad)^2/s'
        attrs_acc['units'] = 'm^2/s^3 or (rad)^2/s^3'
        force_fd = xr.DataArray(force_fd, dims=dims, coords=coords, attrs=attrs_f)
        power_fd = xr.DataArray(power_fd, dims=dims, coords=coords, attrs=attrs_p)
        pos_fd = xr.DataArray(pos_fd, dims=dims, coords=coords, attrs=attrs_pos)
        vel_fd = xr.DataArray(vel_fd, dims=dims, coords=coords, attrs=attrs_vel)
        acc_fd = xr.DataArray(acc_fd, dims=dims, coords=coords, attrs=attrs_acc)
        freq_dom = xr.Dataset({'pos': pos_fd, 'vel': vel_fd, 'acc': acc_fd,
                            'force': force_fd, 'power': power_fd},)

        return time_dom, freq_dom

    # define in subclass:
    _errmsg = "This should be implemented in the subclasses."

    @property
    def nstate_per_dof(self):
        """Number of PTO states per PTO DOF."""
        raise NotImplementedError(self._errmsg)

    def force(self, wec: WEC, x_wec: npt.ArrayLike, x_opt: npt.ArrayLike,
              nsubsteps: int = 1) -> np.ndarray:
        """Calculate the PTO forces time-series in each PTO DOF for a
        given system state.

        Parameters
        ----------
        wec: wecopttool.WEC
            The WEC.
        x_wec: np.ndarray
            WEC dynamics state.
        x_opt: np.ndarray
            Optimization (PTO control) state.
        nsubsteps: int
            Number of subdivisions between the default (implied) time
            steps.

        Returns
        -------
        np.ndarray
            PTO force in the time domain in the PTO's DOFs.
        """
        raise NotImplementedError(self._errmsg)


class PseudoSpectralPTO(_PTO):
    """Pseudo-spectral PTO control.

    Unstructured numerical optimal time-dependent PTO-force.
    Equivalent to conjugate gradient (CC) if no additional constraints.
    See ``_PTO``.
    """

    def __init__(self, nfreq: int, kinematics: np.ndarray,
                 names: list[str] | None = None) -> None:
        """
        Parameters
        ----------
        nfreq: int
            Number of frequencies in pseudo-spectral problem. Should
            match the BEM and wave frequencies.
        kinematics: np.ndarray
            Matrix that converts from the WEC DOFs to the PTO DOFs.
            Shape: (PTO DOFs, WEC DOFs).
        """
        super().__init__(kinematics, names)
        self.nfreq = nfreq

    @property
    def nstate_per_dof(self):
        return 2 * self.nfreq

    def force(self, wec: WEC, x_wec: npt.ArrayLike, x_opt: npt.ArrayLike,
              nsubsteps: int = 1) -> np.ndarray:
        return self._pseudo_spectral(wec, x_opt, nsubsteps)

    def energy(self, wec: WEC, x_wec: npt.ArrayLike, x_opt: npt.ArrayLike,
               nsubsteps: int = 1) -> float:
        if nsubsteps == 1:
            wec_pos = wec.vec_to_dofmat(x_wec)
            wec_vel = np.dot(wec.derivative_mat, wec_pos)
            vel = self._wec_to_pto_dofs(wec_vel)
            vel_vec = self._dofmat_to_vec(vel[1:, :])
            energy_produced = 1/(2*wec.f0) * np.dot(vel_vec, x_opt)
        else:
            energy_produced = super().energy(wec, x_wec, x_opt, nsubsteps)
        return energy_produced


class ProportionalPTO(_PTO):
    """Proportional (P) PTO controller (a.k.a., "proportional damping").

    PTO force is a constant times the velocity.
    See ``_PTO``.
    """

    def __init__(self, kinematics: np.ndarray, names: list[str] | None = None
                 ) -> None:
        super().__init__(kinematics, names)

    @property
    def nstate_per_dof(self):
        return 1

    def force(self, wec: WEC, x_wec: npt.ArrayLike, x_opt: npt.ArrayLike,
              nsubsteps: int = 1) -> np.ndarray:
        vel_td = self.velocity(wec, x_wec, x_opt, nsubsteps)
        force_td = np.reshape(x_opt, [-1, 1]) * vel_td
        return force_td


class ProportionalIntegralPTO(_PTO):
    """Proportional integral (PI) PTO controller.

    PTO force is a constant times the velocity plus a constant times
    position. See ``_PTO``.
    """

    def __init__(self, kinematics: np.ndarray, names: list[str] | None = None
                 ) -> None:
        super().__init__(kinematics, names)

    @property
    def nstate_per_dof(self):
        return 2

    def force(self, wec: WEC, x_wec: npt.ArrayLike, x_opt: npt.ArrayLike,
              nsubsteps: int = 1) -> np.ndarray:
        vel_td = self.velocity(wec, x_wec, x_opt, nsubsteps)
        pos_td = self.position(wec, x_wec, x_opt, nsubsteps)
        u = np.reshape(x_opt, [-1, 1])
        B = np.hstack([vel_td, pos_td])
        force_td = np.dot(B,u)
        return force_td


class PseudoSpectralLinearPTO(_PTO):
    """Linear structured PTO.

    Models the PTO dynamics and provides output electrical current,
    voltage, and power.
    In addition to the kinematics matrix (WEC motions to PTO motions),
    requires specifying a linear PTO impedance matrix that converts from
    [PTO velocity, output current]^T to [PTO force, output voltage]^T.
    """

    def __init__(self, nfreq: int, kinematics: np.ndarray,
                 impedance: np.ndarray, names: list[str] | None = None,
                 ) -> None:
        """
        Parameters
        ----------
        nfreq: int
            Number of frequencies in pseudo-spectral problem. Should
            match the BEM and wave frequencies.
        impedance: np.ndarray
            Matrix representing the PTO impedance.
        """
        super().__init__(kinematics, names)
        self.impedance= impedance
        self.nfreq = nfreq

    @property
    def nstate_per_dof(self):
        return 2 * self.nfreq

    @property
    def _impedance_t(self):
        return np.swapaxes(self.impedance, 0, 1)

    def _impedance_to_abc(self, impedance):
        """ [i,V]^T = ABC [v, F]^T """
        # TODO
        # # for 1 DOF:
        # Z_11, Z_12 = impedance[0, :]
        # Z_21, Z_22 = impedance[1, :]
        # return np.array([[-Z_11, 1],[(Z_21*Z_12-Z_11*Z_22), Z_22]])/Z_12
        raise NotImplementedError()

    def _calc_flow_vars(self, wec: WEC, x_wec: npt.ArrayLike,
                        x_opt: npt.ArrayLike) -> np.ndarray:
        """Create vector of PTO velocity and current. """
        wec_pos = wec.vec_to_dofmat(x_wec)
        position = self._wec_to_pto_dofs(wec_pos)
        velocity = np.dot(wec.derivative_mat, position)[1:, :]
        current = self._vec_to_dofmat(x_opt)
        return np.hstack([velocity, current])

    def _calc_effort_vars(self, flow_vars: np.array) -> np.array:
        if len(self.impedance.shape)==2:
            effort_vars = np.dot(flow_vars, self._impedance_t)
        else:
            e_i = []
            for i in range(self.nfreq):
                e_i.append(np.dot(flow_vars[i*2:i*2+2, :],
                                  self._impedance_t[:, :, i]))
            effort_vars = np.vstack(e_i)
        return effort_vars

    def _split_effort_vars(self, e):
        return e[:, :self.ndof], e[:, self.ndof:]

    def force(self, wec: WEC, x_wec: npt.ArrayLike, x_opt: npt.ArrayLike,
              nsubsteps: int = 1) -> np.ndarray:
        flow_vars = self._calc_flow_vars(wec, x_wec, x_opt)
        force, _ = self._split_effort_vars(self._calc_effort_vars(flow_vars))
        return self._pseudo_spectral(wec, force, nsubsteps)

    def energy(self, wec: WEC, x_wec: npt.ArrayLike, x_opt: npt.ArrayLike,
               nsubsteps: int = 1) -> float:
        if nsubsteps == 1:
            # velocity PS
            wec_pos = wec.vec_to_dofmat(x_wec)
            position = self._wec_to_pto_dofs(wec_pos[1:, :])
            velocity = np.dot(wec.derivative_mat, position)
            vel_vec = self._dofmat_to_vec(velocity)
            # force PS
            flow_vars = self._calc_flow_vars(wec, x_wec, x_opt)
            force, _ = self._split_effort_vars(
                self._calc_effort_vars(flow_vars))
            force_vec = self._dofmat_to_vec(force)
            # energy
            energy_produced = 1/(2*wec.f0) * np.dot(vel_vec, force_vec)
        else:
            energy_produced = super().energy(wec, x_wec, x_opt, nsubsteps)
        return energy_produced

    def electric_current(self, wec: WEC, x_wec: npt.ArrayLike,
                        x_opt: npt.ArrayLike, nsubsteps: int = 1
                        ) -> np.ndarray:
        """Calculate electric current time-series for each PTO DOF. """
        return self._pseudo_spectral(wec, x_opt, nsubsteps)

    def electric_voltage(self, wec: WEC, x_wec: npt.ArrayLike,
                         x_opt: npt.ArrayLike, nsubsteps: int = 1
                         ) -> np.ndarray:
        """Calculate electric voltage time-series for each PTO DOF. """
        flow_vars = self._calc_flow_vars(wec, x_wec, x_opt)
        _, volt = self._split_effort_vars(self._calc_effort_vars(flow_vars))
        return self._pseudo_spectral(wec, volt, nsubsteps)

    def electric_power(self, wec: WEC, x_wec: npt.ArrayLike,
                       x_opt: npt.ArrayLike, nsubsteps: int = 1
                       ) -> np.ndarray:
        """Calculate electric power time-series for each PTO DOF. """
        current = self.electric_current(wec, x_wec, x_opt, nsubsteps)
        voltage = self.electric_voltage(wec, x_wec, x_opt, nsubsteps)
        return current*voltage

    def electric_average_power(self, wec: WEC, x_wec: npt.ArrayLike,
                               x_opt: npt.ArrayLike, nsubsteps: int = 1
                               ) -> np.ndarray:
        """Calculate the average power of the PTO for the given wave
        spectrum.
        """
        return self.electric_energy(wec, x_wec, x_opt, nsubsteps) / wec.tf

    def electric_energy(self, wec: WEC, x_wec: npt.ArrayLike,
                        x_opt: npt.ArrayLike, nsubsteps: int = 1
                        ) -> np.ndarray:
        """Calculate the electric energy (in Joules) by the PTO during
        the period t = 0-T = 1/f0."""
        if nsubsteps == 1:
            flow_vars = self._calc_flow_vars(wec, x_wec, x_opt)
            _, volt = self._split_effort_vars(
                self._calc_effort_vars(flow_vars))
            volt_vec = self._dofmat_to_vec(volt)
            energy_produced = 1/(2*wec.f0) * np.dot(x_opt, volt_vec)
        else:
            power_td = self.electric_power(wec, x_wec, x_opt, nsubsteps)
            energy_produced = np.sum(power_td) * wec.dt/nsubsteps
        return energy_produced

    def post_process(self, wec: WEC, x_wec: npt.ArrayLike, x_opt: npt.ArrayLike
                     ) -> tuple[xr.Dataset, xr.Dataset]:
        time_dom, freq_dom = super().post_process(wec, x_wec, x_opt)
        # TODO: add electrical quantities
        return time_dom, freq_dom
