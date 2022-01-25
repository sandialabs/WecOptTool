
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
            Number of frequencies in pseudo-spectral problem. Should match
            the BEM and wave frequencies.
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
        x_pto = np.reshape(x_opt, (self.nstate_per_dof, self.ndof), order='F')
        time_mat = wec.make_time_mat(nsubsteps, include_mean=False)
        return np.dot(time_mat, x_pto)

    def energy(self, wec: WEC, x_wec: npt.ArrayLike, x_opt: npt.ArrayLike,
               nsubsteps: int = 1) -> float:
        if nsubsteps == 1:
            wec_pos = wec.vec_to_dofmat(x_wec)
            wec_vel = np.dot(wec.derivative_mat, wec_pos)
            vel = self._wec_to_pto_dofs(wec_vel)
            vel_vec = wec.dofmat_to_vec(vel[1:, :])
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
