"""Provide power take-off (PTO) forces and produced energy functions
for common PTO control approaches.

The PTO produced energy can be used as the objective function for the
control optimization.
The PTO force can be included as an additional force in the WEC
dynamics.
"""


from __future__ import annotations


from typing import Optional, TypeVar, Callable, Iterable, Union

import autograd.numpy as np
from autograd.builtins import isinstance, tuple, list, dict
from autograd.numpy import ndarray
import numpy.typing as npt
from scipy.linalg import block_diag
from xarray import DataArray, Dataset
import xarray as xr
from tests.test_waves import ndbc_omnidirectional
from datetime import datetime
from scipy.optimize import OptimizeResult

from wecopttool.core import WEC, real_to_complex, complex_to_real
from wecopttool.core import td_to_fd, dofmat_to_vec, vec_to_dofmat
from wecopttool.core import TWEC, TStateFunction, FArrayLike


# type aliases
TPTO = TypeVar("TPTO", bound="PTO")


class PTO:
    """A power take-off (PTO) object to be used in conjunction with a 
    :python:`WEC` object.
    TODO
    """

    def __init__(self, 
                 ndof: int, 
                 kinematics: Union[Callable, npt.ArrayLike], 
                 controller: Optional[Callable] = None, 
                 impedance=None, #TODO - type?
                 efficiency=None, #TODO - type?
                 names: Optional[list[str]] = None):
        """
        TODO
        """
        self._ndof = ndof
        # names
        if names is None:
            self._names = [f'PTO_{i}' for i in range(ndof)]
        else:
            self._names = names
            # TODO: if 1 dof and single string, convert to list.
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
        self._efficiency = efficiency
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
        """Kinemtaics function."""
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
    def efficiency(self) -> Callable[[FArrayLike, FArrayLike], FArrayLike]:
        """Efficiency function."""
        return self._efficiency

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
                     x_wec, 
                     x_opt: Optional[ndarray] = None, 
                     waves: Optional[Dataset] = None, 
                     nsubsteps: Optional[int] = 1,
                     ):
        """ Return time-domain values in the PTO frame.
        `f_wec`: Fourier coefficients of some quantity "f" in the WEC frame.
        TODO
        """
        time_mat = self._tmat(wec, nsubsteps)
        f_wec_td = np.dot(time_mat, f_wec)
        # assert f_wec_td.shape == (wec.nt*nsubsteps, wec.ndof)
        f_wec_td = np.expand_dims(np.transpose(f_wec_td), axis=0)
        # assert f_wec_td.shape == (1, wec.ndof, wec.nt*nsubsteps)
        kinematics_mat = self.kinematics(wec, x_wec, x_opt, waves, nsubsteps)
        return np.transpose(np.sum(kinematics_mat*f_wec_td, axis=1))

    def position(self, 
                 wec: TWEC, 
                 x_wec: npt.ArrayLike,
                 x_opt: Optional[npt.ArrayLike],
                 waves: Optional[Dataset] = None,
                 nsubsteps: Optional[int] = 1,):
        """Calculate the PTO position time-series.
        TODO
        """
        pos_wec = wec.vec_to_dofmat(x_wec)
        return self._fkinematics(pos_wec, wec, x_wec, x_opt, waves, nsubsteps)

    def velocity(self, 
                 wec: TWEC, 
                 x_wec: npt.ArrayLike,
                 x_opt: Optional[npt.ArrayLike],
                 waves: Optional[Dataset] = None,
                 nsubsteps: Optional[int] = 1):
        """Calculate the PTO velocity time-series."""
        pos_wec = wec.vec_to_dofmat(x_wec)
        vel_wec = np.dot(wec.derivative_mat, pos_wec)
        return self._fkinematics(vel_wec, wec, x_wec, x_opt, waves, nsubsteps)

    def acceleration(self, 
                     wec: TWEC, 
                     x_wec: npt.ArrayLike,
                     x_opt: Optional[npt.ArrayLike],
                     waves: Optional[Dataset] = None,
                     nsubsteps: Optional[int] = 1,
                     ) -> np.ndarray:
        """Calculate the PTO acceleration time-series.
        TODO
        """
        pos_wec = wec.vec_to_dofmat(x_wec)
        vel_wec = np.dot(wec.derivative_mat, pos_wec)
        acc_wec = np.dot(wec.derivative_mat, vel_wec)
        return self._fkinematics(acc_wec, wec, x_wec, x_opt, waves, nsubsteps)

    def force_on_wec(self, 
                     wec: TWEC, 
                     x_wec: npt.ArrayLike,
                     x_opt: Optional[npt.ArrayLike],
                     waves: Optional[Dataset] = None, 
                     nsubsteps: Optional[int] = 1):
        force_td = self.force(wec, x_wec, x_opt, waves, nsubsteps)
        assert force_td.shape == (wec.nt, self.ndof)
        force_td = np.expand_dims(np.transpose(force_td), axis=0)
        assert force_td.shape == (1, wec.ndof, wec.nt)
        kinematics_mat = self.kinematics(wec, x_wec, x_opt, waves)
        kinematics_mat = np.transpose(kinematics_mat, (1,0,2))
        return np.transpose(np.sum(kinematics_mat*force_td, axis=1))

    def mechanical_power(self, 
                         wec: TWEC, 
                         x_wec: npt.ArrayLike,
                         x_opt: Optional[npt.ArrayLike],
                         waves: Optional[Dataset] = None, 
                         nsubsteps: Optional[int] = 1,
                         ) -> np.ndarray:
        """Calculate the PTO power time-series in each PTO DOF
        for a given system state.
        TODO
        """
        force_td = self.force(wec, x_wec, x_opt, waves, nsubsteps)
        vel_td = self.velocity(wec, x_wec, x_opt, waves, nsubsteps)
        return vel_td * force_td

    def mechanical_energy(self, 
                          wec: TWEC, 
                          x_wec: npt.ArrayLike,
                          x_opt: Optional[npt.ArrayLike],
                          waves: Optional[Dataset] = None, 
                          nsubsteps: Optional[int] = 1,
                          ) -> float:
        """
        TODO
        """
        power_td = self.mechanical_power(wec, x_wec, x_opt, waves, nsubsteps)
        return np.sum(power_td) * wec.dt/nsubsteps

    def mechanical_average_power(self, 
                                 wec: TWEC, 
                                 x_wec: npt.ArrayLike,
                                 x_opt: Optional[npt.ArrayLike],
                                 waves: Optional[Dataset] = None, 
                                 nsubsteps: Optional[int] = 1,
                                 ) -> float:
        """TODO
        """
        energy = self.mechanical_energy(wec, x_wec, x_opt, waves, nsubsteps)
        return energy / wec.tf

    def power(self, 
              wec: TWEC, 
              x_wec: npt.ArrayLike,
              x_opt: Optional[npt.ArrayLike], 
              waves: Optional[Dataset] = None, 
              nsubsteps: Optional[int] = 1,
              ) -> ndarray:
        """TODO
        """
        e1_td = self.force(wec, x_wec, x_opt, waves, nsubsteps)
        q1_td = self.velocity(wec, x_wec, x_opt, waves, nsubsteps)
        # convert e1 (PTO force), q1 (PTO velocity) to e2,q2
        if self.impedance is not None:
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
            e2_td = e1_td
            q2_td = q1_td
        # power
        power_out = e2_td * q2_td
        if self.efficiency is not None:
            power_out = power_out * self.efficiency(e2_td, q2_td)
        return power_out

    def energy(self, 
               wec: TWEC, 
               x_wec: npt.ArrayLike,
               x_opt: Optional[npt.ArrayLike],
               waves: Optional[Dataset] = None, 
               nsubsteps: Optional[int] = 1,
               ) -> float:
        """TODO"""
        power_td = self.power(wec, x_wec, x_opt, waves, nsubsteps)
        return np.sum(power_td) * wec.dt/nsubsteps

    def average_power(self, 
                      wec: TWEC, 
                      x_wec: npt.ArrayLike,
                      x_opt: Optional[npt.ArrayLike],
                      waves: Optional[Dataset] = None, 
                      nsubsteps: Optional[int] = 1,
                      ) -> float:
        energy = self.energy(wec, x_wec, x_opt, waves, nsubsteps)
        return energy / wec.tf
    
    def post_process(self, 
                     wec: TWEC, 
                     res: OptimizeResult,
                     waves: xr.DataArray = None,
                     nsubsteps: Optional[int] = 1,
                     ) -> tuple[xr.Dataset, xr.Dataset]:
        """TODO
        """
        
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
        omega_attr = {'long_name': 'Frequency', 'units': 'rad/s'}
        dof_attr = {'long_name': 'PTO degree of freedom'}
        time_attr = {'long_name': 'Time', 'units': 's'}
        
        t_dat = wec.time_nsubsteps(nsubsteps)
        
        results_fd = xr.Dataset(
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
                'dof':('dof', self.names, dof_attr)},
            attrs={"time_created_utc": f"{datetime.utcnow()}"}
            )
        
        results_td = xr.Dataset(
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
            attrs={"time_created_utc": f"{datetime.utcnow()}"}
            )
        
        return results_fd, results_td



# power conversion chain
def _make_abcd(impedance: npt.ArrayLike, ndof: int) -> ndarray:
    """TODO"""
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


def _make_mimo_transfer_mat(impedance_abcd: npt.ArrayLike, 
                            ndof: int,
                            ) -> np.ndarray:
    """Create a block matrix of the MIMO transfer function.
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
def controller_unstructured(pto, 
                            wec: TWEC, 
                            x_wec: npt.ArrayLike, 
                            x_opt: npt.ArrayLike, 
                            waves: Optional[Dataset] = None, 
                            nsubsteps=1) -> ndarray:
    """TODO"""
    x_opt = np.reshape(x_opt, (-1, pto.ndof), order='F')
    tmat = pto._tmat(wec, nsubsteps)
    return np.dot(tmat, x_opt)


def controller_pid(pto, 
                   wec: TWEC, 
                   x_wec: npt.ArrayLike, 
                   x_opt: npt.ArrayLike,
                   waves: Optional[Dataset] = None, 
                   nsubsteps: Optional[int] = 1,
                   proportional: Optional[bool] = True, 
                   integral: Optional[bool] = True, 
                   derivative: Optional[bool] = True,
                   ) -> ndarray:
    """TODO"""
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


def controller_pi(pto, 
                  wec: TWEC,
                  x_wec: npt.ArrayLike, 
                  x_opt: npt.ArrayLike, 
                  waves: Optional[Dataset] = None, 
                  nsubsteps: Optional[int] = 1,
                  ) -> ndarray:
    "TODO"
    force_td = controller_pid(pto, wec, x_wec, x_opt, waves, nsubsteps,
                               True, True, False)
    return force_td


def controller_p(pto, 
                 wec: TWEC, 
                 x_wec: npt.ArrayLike, 
                 x_opt: npt.ArrayLike, 
                 waves: Optional[Dataset] = None, 
                 nsubsteps: Optional[int] = 1,
                 ) -> ndarray:
    """TODO"""
    force_td = controller_pid(pto, wec, x_wec, x_opt, waves, nsubsteps,
                               True, False, False)
    return force_td
