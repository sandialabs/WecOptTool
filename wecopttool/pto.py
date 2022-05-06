
"""Provide power take-off (PTO) forces and produced energy functions
for common PTO control approaches.

The PTO produced energy can be used as the objective function for the
control optimization.
The PTO force can be included as an additional force in the WEC
dynamics.
"""


from __future__ import annotations  # TODO: delete after python 3.10

import numpy.typing as npt
import autograd.numpy as np
from autograd.builtins import isinstance, tuple, list, dict
import xarray as xr
from scipy.linalg import block_diag

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

    def _pto_force_to_wec_force(self, f_pto):
        return np.dot(f_pto, self.kinematics)

    def _wec_pos_to_pto_pos(self, pos_wec):
        return np.dot(pos_wec, self._kinematics_t)

    def _dofmat_to_vec(self, mat: np.ndarray) -> np.ndarray:
        """Flatten a matrix that has one column per DOF.
        Opposite of ``vec_to_dofmat``. """
        return np.reshape(mat, -1, order='F')

    def _vec_to_dofmat(self, vec: np.ndarray) -> np.ndarray:
        """Convert a vector back to a matrix with one column per DOF.
        Opposite of ``dofmat_to_vec``.
        """
        return np.reshape(vec, (self.nstate_per_dof, -1), order='F')

    def _pseudo_spectral(self, wec: WEC, x_opt: npt.ArrayLike,
                         nsubsteps: int = 1) -> np.ndarray:
        x = self._vec_to_dofmat(x_opt)
        time_mat = wec.make_time_mat(nsubsteps, include_mean=False)
        return np.dot(time_mat, x)

    def position(self, wec: WEC, x_wec: npt.ArrayLike, x_opt: npt.ArrayLike,
                 nsubsteps: int = 1) -> np.ndarray:
        """Calculate the PTO position time-series."""
        wec_pos = wec.vec_to_dofmat(x_wec)
        pos = self._wec_pos_to_pto_pos(wec_pos)
        time_mat = wec.make_time_mat(nsubsteps)
        return np.dot(time_mat, pos)

    def velocity(self, wec: WEC, x_wec: npt.ArrayLike, x_opt: npt.ArrayLike,
                 nsubsteps: int = 1) -> np.ndarray:
        """Calculate the PTO velocity time-series."""
        wec_pos = wec.vec_to_dofmat(x_wec)
        pos = self._wec_pos_to_pto_pos(wec_pos)
        vel = np.dot(wec.derivative_mat, pos)
        time_mat = wec.make_time_mat(nsubsteps)
        return np.dot(time_mat, vel)

    def acceleration(self, wec: WEC, x_wec: npt.ArrayLike,
                     x_opt: npt.ArrayLike, nsubsteps: int = 1
                     ) -> np.ndarray:
        """Calculate the PTO acceleration time-series."""
        wec_pos = wec.vec_to_dofmat(x_wec)
        pos = self._wec_pos_to_pto_pos(wec_pos)
        vel = np.dot(wec.derivative_mat, pos)
        acc = np.dot(wec.derivative_mat, vel)
        time_mat = wec.make_time_mat(nsubsteps)
        return np.dot(time_mat, acc)

    def power(self, wec: WEC, x_wec: npt.ArrayLike, x_opt: npt.ArrayLike,
              nsubsteps: int = 1) -> np.ndarray:
        """Calculate the PTO power time-series in each PTO DOF
        for a given system state.
        """
        force_td = self.force(wec, x_wec, x_opt, nsubsteps)
        vel_td = self.velocity(wec, x_wec, x_opt, nsubsteps)
        return -1*vel_td * force_td

    def force_on_wec(self, wec: WEC, x_wec: npt.ArrayLike, x_opt: npt.ArrayLike,
                     nsubsteps: int = 1) -> np.ndarray:
        """Calculate the PTO forces time-series in the WEC DOFs for a
        given system state.

        See ``force``.
        """
        fpto_td = self.force(wec, x_wec, x_opt, nsubsteps)
        return self._pto_force_to_wec_force(fpto_td)

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
        coords = [(dims[0], wec.time, {'units': 's',
                                       'long_name':'Time'}), (dims[1], self.names)]
        attrs_f = {'long_name': 'PTO force', 'units': 'N or N*m'}
        attrs_p = {'long_name': 'PTO power', 'units': 'W'}
        attrs_pos = {'long_name': 'PTO position', 'units': 'm or rad'}
        attrs_vel = {'long_name': 'PTO velocity', 'units': 'm/s or rad/s'}
        attrs_acc = {'long_name': 'PTO acceleration',
                    'units': 'm/s^2 or rad/s^2'}
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
        coords[0] = (dims[0], omega, {'units': 'rad/s',
                                      'long_name': 'Frequency'})
        attrs_f['units'] = 'N^2*s'
        attrs_p['units'] = 'W^2*s'
        attrs_pos['units'] = 'm^2*s or rad^2*s'
        attrs_vel['units'] = 'm^2/s or rad^2/s'
        attrs_acc['units'] = 'm^2/s^3 or rad^2/s^3'
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
            pos = self._wec_pos_to_pto_pos(wec_pos)
            vel = np.dot(wec.derivative_mat, pos)
            vel_vec = self._dofmat_to_vec(vel[1:, :])
            energy_produced = -1 * 1/(2*wec.f0) * np.dot(vel_vec, x_opt)
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
        force_td = np.reshape(x_opt, [1,-1]) * vel_td
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
        u = np.reshape(x_opt, [1,-1])
        B = np.hstack([vel_td, pos_td])
        tmp1 = u * B
        force_td = tmp1[:,0:self.ndof] + tmp1[:,self.ndof:]
        return force_td


class _LinearPTO(_PTO):
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
        _PTO.__init__(self, kinematics, names)
        self.nfreq = nfreq
        if len(impedance.shape) == 2:
            impedance = np.tile(np.expand_dims(impedance, 2), self.nfreq)
        self.impedance = impedance
        self._make_abcd()
        self._make_mimo_transfer_mat()

    @property
    def ndof2(self):
        return 2 * self.ndof

    def _vec_to_dofmat(self, vec: np.ndarray) -> np.ndarray:
        return np.reshape(vec, (2 * self.nfreq, -1), order='F')

    def _make_abcd(self):
        z_11 = self.impedance[:self.ndof, :self.ndof, :]  # Fv
        z_12 = self.impedance[:self.ndof, self.ndof:, :]  # Fi
        z_21 = self.impedance[self.ndof:, :self.ndof, :]  # Vv
        z_22 = self.impedance[self.ndof:, self.ndof:, :]  # Vi
        z_12_inv = np.linalg.inv(z_12.T).T

        mmult = lambda a,b: np.einsum('mnr,mnr->mnr', a, b)
        abcd_11 = -1 * mmult(z_12_inv, z_11)
        abcd_12 = z_12_inv
        abcd_21 = z_21 - mmult(z_22, mmult(z_12_inv, z_11))
        abcd_22 = mmult(z_22, z_12_inv)

        abcd = np.zeros(self.impedance.shape)*1j
        abcd[:self.ndof, :self.ndof, :] = abcd_11
        abcd[:self.ndof, self.ndof:, :] = abcd_12
        abcd[self.ndof:, :self.ndof, :] = abcd_21
        abcd[self.ndof:, self.ndof:, :] = abcd_22
        self._impedance_abcd = abcd

    def _make_mimo_transfer_mat(self) -> np.ndarray:
        """Create a block matrix of the MIMO transfer function.
        """
        elem = [[None]*self.ndof2 for _ in range(self.ndof2)]
        def block(re, im): return np.array([[re, im], [-im, re]])
        for idof in range(self.ndof2):
            for jdof in range(self.ndof2):
                Zp = self._impedance_abcd[idof, jdof, :]
                re = np.real(Zp)
                im = np.imag(Zp)
                blocks = [block(ire, iim) for (ire, iim) in zip(re, im)]
                elem[idof][jdof] = block_diag(*blocks)
        self._transfer_mat = np.block(elem)

    def _calc_mech_vars(self, wec: WEC, x_wec: npt.ArrayLike,
                        x_opt: npt.ArrayLike) -> np.ndarray:
        """Create vector of PTO velocity and force. """
        wec_pos = wec.vec_to_dofmat(x_wec)
        position = self._wec_pos_to_pto_pos(wec_pos)
        velocity = np.dot(wec.derivative_mat, position)[1:, :]
        force = self._force_fd_mat(wec, x_wec, x_opt)
        return np.hstack([velocity, force])

    def _calc_elec_vars(self, mech_vars: np.array) -> np.array:
        mech_flat = self._dofmat_to_vec(mech_vars)
        elec_flat = np.dot(self._transfer_mat, mech_flat)
        elec_vars = self._vec_to_dofmat(elec_flat)
        return elec_vars

    def _split_vars(self, vars):
        return vars[:, :self.ndof], vars[:, self.ndof:]

    def force(self, wec: WEC, x_wec: npt.ArrayLike, x_opt: npt.ArrayLike,
              nsubsteps: int = 1) -> np.ndarray:
        force_fd = self._force_fd_mat(wec, x_wec, x_opt)
        return self._pseudo_spectral(wec, force_fd, nsubsteps)

    def energy(self, wec: WEC, x_wec: npt.ArrayLike, x_opt: npt.ArrayLike,
               nsubsteps: int = 1) -> float:
        if nsubsteps == 1:
            # velocity PS
            wec_pos = wec.vec_to_dofmat(x_wec)
            position = self._wec_pos_to_pto_pos(wec_pos)
            velocity = np.dot(wec.derivative_mat, position)
            vel_vec = self._dofmat_to_vec(velocity[1:, :])
            # force PS
            force = self._force_fd_mat(wec, x_wec, x_opt)
            force_vec = self._dofmat_to_vec(force)
            # energy
            energy_produced = 1/(2*wec.f0) * np.dot(vel_vec, force_vec)
        else:
            energy_produced = _PTO.energy(self, wec, x_wec, x_opt, nsubsteps)
        return energy_produced

    def electric_current(self, wec: WEC, x_wec: npt.ArrayLike,
                        x_opt: npt.ArrayLike, nsubsteps: int = 1
                        ) -> np.ndarray:
        """Calculate electric current time-series for each PTO DOF. """
        mech_vars = self._calc_mech_vars(wec, x_wec, x_opt)
        current, _ = self._split_vars(self._calc_elec_vars(mech_vars))
        return self._pseudo_spectral(wec, current, nsubsteps)

    def electric_voltage(self, wec: WEC, x_wec: npt.ArrayLike,
                         x_opt: npt.ArrayLike, nsubsteps: int = 1
                         ) -> np.ndarray:
        """Calculate electric voltage time-series for each PTO DOF. """
        mech_vars = self._calc_mech_vars(wec, x_wec, x_opt)
        _, volt = self._split_vars(self._calc_elec_vars(mech_vars))
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
            mech_vars = self._calc_mech_vars(wec, x_wec, x_opt)
            current, volt = self._split_vars(
                self._calc_elec_vars(mech_vars))
            volt_vec = self._dofmat_to_vec(volt)
            current_vec = self._dofmat_to_vec(current)
            energy_produced = 1/(2*wec.f0) * np.dot(current_vec, volt_vec)
        else:
            power_td = self.electric_power(wec, x_wec, x_opt, nsubsteps)
            energy_produced = np.sum(power_td) * wec.dt/nsubsteps
        return energy_produced

    def post_process(self, wec: WEC, x_wec: npt.ArrayLike, x_opt: npt.ArrayLike
                     ) -> tuple[xr.Dataset, xr.Dataset]:
        time_dom, freq_dom = _PTO.post_process(self, wec, x_wec, x_opt)

        # current
        current_td = self.electric_current(wec, x_wec, x_opt)
        current_fd = wec.td_to_fd(current_td)

        # voltage
        voltage_td = self.electric_voltage(wec, x_wec, x_opt)
        voltage_fd = wec.td_to_fd(voltage_td)

        # power
        epower_td = self.electric_power(wec, x_wec, x_opt)
        epower_fd = wec.td_to_fd(epower_td)

        # time-domain
        dims = time_dom.dims
        coords = time_dom.coords
        attrs_current = {'long_name': 'PTO current', 'units': 'A'}
        attrs_voltage = {'long_name': 'PTO voltage', 'units': 'V'}
        attrs_epower = {'long_name': 'PTO electric power', 'units': 'W'}
        current_td = xr.DataArray(
            current_td, dims=dims, coords=coords, attrs=attrs_current)
        voltage_td = xr.DataArray(
            voltage_td, dims=dims, coords=coords, attrs=attrs_voltage)
        epower_td = xr.DataArray(
            epower_td, dims=dims, coords=coords, attrs=attrs_epower)
        time_dom["current"] = current_td
        time_dom["voltage"] = voltage_td
        time_dom["epower"] = epower_td

        # frequency-domain
        dims = freq_dom.dims
        coords = freq_dom.coords
        attrs_current['units'] = 'A^2*s'
        attrs_voltage['units'] = 'V^2*s'
        attrs_epower['units'] = 'W^2*s'
        current_fd = xr.DataArray(
            current_fd, dims=dims, coords=coords, attrs=attrs_current)
        voltage_fd = xr.DataArray(
            voltage_fd, dims=dims, coords=coords, attrs=attrs_voltage)
        epower_fd = xr.DataArray(
            epower_fd, dims=dims, coords=coords, attrs=attrs_epower)
        freq_dom["current"] = current_fd
        freq_dom["voltage"] = voltage_fd
        freq_dom["epower"] = epower_fd

        return time_dom, freq_dom

    # define in subclass:
    def _force_fd_mat(self, wec: WEC, x_wec: npt.ArrayLike,
                      x_opt: npt.ArrayLike) -> np.ndarray:
        raise NotImplementedError(_PTO._errmsg)


class PseudoSpectralLinearPTO(_LinearPTO):

    @property
    def nstate_per_dof(self):
        return 2 * self.nfreq

    def _force_fd_mat(self, wec: WEC, x_wec: npt.ArrayLike,
                      x_opt: npt.ArrayLike) -> np.ndarray:
        return self._vec_to_dofmat(x_opt)


class ProportionalLinearPTO(_LinearPTO):
    """Proportional (P) PTO controller with linear PTO.

    PTO force is a constant times the velocity.
    """

    @property
    def nstate_per_dof(self):
        return 1

    def _force_fd_mat(self, wec: WEC, x_wec: npt.ArrayLike,
                      x_opt: npt.ArrayLike) -> np.ndarray:
        wec_pos = wec.vec_to_dofmat(x_wec)
        position = self._wec_pos_to_pto_pos(wec_pos)
        velocity = np.dot(wec.derivative_mat, position)
        u = np.reshape(x_opt, [1, -1])
        B = velocity[1:, :]
        force_fd = u * B
        return force_fd


class ProportionalIntegralLinearPTO(_LinearPTO):
    """Proportional (P) PTO controller with linear PTO.

    PTO force is a constant times the velocity.
    """

    @property
    def nstate_per_dof(self):
        return 2

    def _force_fd_mat(self, wec: WEC, x_wec: npt.ArrayLike,
                      x_opt: npt.ArrayLike) -> np.ndarray:
        wec_pos = wec.vec_to_dofmat(x_wec)
        position = self._wec_pos_to_pto_pos(wec_pos)
        velocity = np.dot(wec.derivative_mat, position)
        u = np.reshape(x_opt, [1, -1])
        B = np.hstack([position[1:, :], velocity[1:, :]])
        tmp1 = u * B
        force_fd = tmp1[:, 0:self.ndof] + tmp1[:, self.ndof:]
        return force_fd
