"""Provide power take-off (PTO) forces and produced energy functions
for common PTO control approaches.

The PTO produced energy can be used as the objective function for the
control optimization.
The PTO force can be included as an additional force in the WEC
dynamics.
"""


from __future__ import annotations


from typing import Optional

import autograd.numpy as np
from autograd.builtins import isinstance, tuple, list, dict
import numpy.typing as npt
from scipy.linalg import block_diag

from wecopttool.core import WEC, complex_to_real, td_to_fd, dofmat_to_vec, vec_to_dofmat


class PTO:

    def __init__(self, ndof, kinematics, controller,
                 power, names: Optional[list[str]] = None):
        """
        ndof: int
        kinematics: (...) -> kinematics matrix
        controller: (...) -> pto force in td in pto frame
        power: (...) -> power time-series
        (...) = (pto, wec, x_wec, x_opt, waves, nsubsteps)
        """
        self.ndof = ndof
        self.force = controller
        self.kinematics = kinematics
        self.power = power
        names = [f'PTO_{i+1}' for i in range(ndof)] if names is None else names
        self.names = names

    def _tmat(self, wec, nsubsteps: int = 1):
        if nsubsteps==1:
            tmat = wec.time_mat
        else:
            tmat = wec.time_mat_nsubsteps(nsubsteps)
        return tmat

    def _kinematics(self, f_wec, wec, x_wec, x_opt=None, waves=None, nsubsteps: int = 1):
        """ Return time-domain values in the PTO frame.
        `f_wec`: Fourier coefficients of some quantity "f" in the WEC frame.
        """
        time_mat = self._tmat(wec, nsubsteps)
        f_wec_td = np.dot(time_mat, f_wec)
        assert f_wec_td.shape == (wec.nt, wec.ndof)
        f_wec_td = np.expand_dims(np.transpose(f_wec_td), axis=0)
        assert f_wec_td.shape == (1, wec.ndof, wec.nt)
        kinematics_mat = self.kinematics(self, wec, x_wec, x_opt, waves)
        return np.transpose(np.sum(kinematics_mat*f_wec_td, axis=1))

    def position(self, wec: WEC, x_wec: npt.ArrayLike,
                 x_opt: Optional[npt.ArrayLike],
                 waves=None, nsubsteps: int = 1):
        """Calculate the PTO position time-series."""
        pos_wec = wec.vec_to_dofmat(x_wec)
        return self._kinematics(pos_wec, wec, x_wec, x_opt, waves, nsubsteps)

    def velocity(self, wec: WEC, x_wec: npt.ArrayLike,
                 x_opt: Optional[npt.ArrayLike],
                 waves=None, nsubsteps: int = 1):
        """Calculate the PTO velocity time-series."""
        pos_wec = wec.vec_to_dofmat(x_wec)
        vel_wec = np.dot(wec.derivative_mat, pos_wec)
        return self._kinematics(vel_wec, wec, x_wec, x_opt, waves, nsubsteps)

    def acceleration(self, wec: WEC, x_wec: npt.ArrayLike,
                     x_opt: Optional[npt.ArrayLike],
                     waves=None, nsubsteps: int = 1) -> np.ndarray:
        """Calculate the PTO acceleration time-series."""
        pos_wec = wec.vec_to_dofmat(x_wec)
        vel_wec = np.dot(wec.derivative_mat, pos_wec)
        acc_wec = np.dot(wec.derivative_mat, vel_wec)
        return self._kinematics(acc_wec, wec, x_wec, x_opt, waves, nsubsteps)

    def force_on_wec(self, wec: WEC, x_wec: npt.ArrayLike,
                     x_opt: Optional[npt.ArrayLike],
                     waves=None, nsubsteps: int = 1):
        force_td = self.force(self, wec, x_wec, x_opt, waves, nsubsteps)
        assert force_td.shape == (wec.nt, self.ndof)
        force_td = np.expand_dims(np.transpose(force_td), axis=0)
        assert force_td.shape == (1, wec.ndof, wec.nt)
        kinematics_mat = self.kinematics(self, wec, x_wec, x_opt, waves)
        kinematics_mat = np.transpose(kinematics_mat, (1,0,2))
        return np.transpose(np.sum(kinematics_mat*force_td, axis=1))

    def mechanical_power(self, wec: WEC, x_wec: npt.ArrayLike,
                         x_opt: Optional[npt.ArrayLike],
                         waves=None, nsubsteps: int = 1) -> np.ndarray:
        """Calculate the PTO power time-series in each PTO DOF
        for a given system state.
        """
        force_td = self.force(self, wec, x_wec, x_opt, waves, nsubsteps)
        vel_td = self.velocity(wec, x_wec, x_opt, waves, nsubsteps)
        return vel_td * force_td

    def mechanical_energy(self, wec: WEC, x_wec: npt.ArrayLike,
                          x_opt: Optional[npt.ArrayLike],
              waves=None, nsubsteps: int = 1):
        power_td = self.mechanical_power(wec, x_wec, x_opt, waves, nsubsteps)
        return np.sum(power_td) * wec.dt/nsubsteps

    def mechanical_average_power(self, wec: WEC, x_wec: npt.ArrayLike,
                                 x_opt: Optional[npt.ArrayLike],
                                 waves=None, nsubsteps: int = 1):
        energy = self.mechanical_energy(wec, x_wec, x_opt, waves, nsubsteps)
        return energy / wec.tf

    def energy(self, wec: WEC, x_wec: npt.ArrayLike,
               x_opt: Optional[npt.ArrayLike],
               waves=None, nsubsteps: int = 1):
        power_td = self.power(self, wec, x_wec, x_opt, waves, nsubsteps)
        return np.sum(power_td) * wec.dt/nsubsteps

    def average_power(self, wec: WEC, x_wec: npt.ArrayLike,
                      x_opt: Optional[npt.ArrayLike],
                      waves=None, nsubsteps: int = 1):
        energy = self.energy(wec, x_wec, x_opt, waves, nsubsteps)
        return energy / wec.tf


# kinematics
def kinematics_linear(kinematics_mat):

    def fun(pto, wec, x_wec, x_opt, waves, nsubsteps=1):
        n = (wec.nt-1)*nsubsteps + 1
        return np.repeat(kinematics_mat[:, :, np.newaxis], n, axis=-1)

    return fun


def kinematics_nonlinear(kinematics_fun):
    """
    kinematics_fun(pos_wec_td: [ndof_wec x npos_wec]) -> [ndof_pto x ndof_wec x npos_wec]
    The given function should work on multiple input positions.
    """

    def fun(pto, wec, x_wec, x_opt, waves):
        pos_wec = wec.vec_to_dofmat(x_wec)
        pos_wec_td = np.dot(wec.time_mat, pos_wec)
        return kinematics_fun(pos_wec_td)

    return fun


# controllers
def controller_unstructured():

    def fun(pto, wec, x_wec, x_opt, waves, nsubsteps=1):
        x_opt = np.reshape(x_opt, (-1, pto.ndof), order='F')
        tmat = pto._tmat(wec, nsubsteps)
        return np.dot(tmat, x_opt)

    return fun


def controller_pid(proportional=True, integral=True, derivative=True):

    def fun(pto, wec, x_wec, x_opt, waves, nsubsteps=1):
        ndof = pto.ndof
        force_td = np.zeros([wec.nt, ndof])
        idx = 0

        def update_force_td(B):
            nonlocal idx, force_td
            u = np.reshape(x_opt[idx*ndof:(idx+1)*ndof], [1, ndof])
            force_td = force_td + u*B
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

    return fun


# power conversion chain
def power_linear(impedance):
    ndof = impedance.shape[0] // 2
    impedance_abcd = _make_abcd(impedance, ndof)
    transfer_mat = _make_mimo_transfer_mat(impedance_abcd, ndof)

    def fun(pto, wec, x_wec, x_opt, waves, nsubsteps=1):
        velocity_td = pto.velocity(wec, x_wec, x_opt, waves)
        velocity = complex_to_real(td_to_fd(velocity_td, False))
        force_td = pto.force(pto, wec, x_wec, x_opt, waves)
        force = complex_to_real(td_to_fd(force_td, False))
        mech_vars = np.hstack([velocity[1:, :], force[1:, :]])
        mech_flat = dofmat_to_vec(mech_vars)
        elec_flat = np.dot(transfer_mat, mech_flat)
        elec_vars = vec_to_dofmat(elec_flat, 2*ndof)
        current = elec_vars[:, :ndof]
        voltage = elec_vars[:, ndof:]
        time_mat = pto._tmat(wec, nsubsteps)[:, 1:]
        current_td = np.dot(time_mat, current)
        voltage_td = np.dot(time_mat, voltage)
        return current_td*voltage_td

    return fun


def power_efficiency_map(efficiency, impedance=None):
    """`impedance` maps PTO velocity and force to some other
    flow (`q`) & effort (`e`) variables, e.g. generator rotational
    speed and torque. These new variables are the inputs to the
    efficiency map.
    The `efficiency` map gives the ratio `P/P0` of actual power `P` to
    maximum power `P0=e*q` for given values of `e` and `q`.
    """
    raise NotImplementedError()
    # if impedance is None, create np.eye(2*ndof) + repeat n times
    # impedance_abcd = _make_abcd(impedance, pto.ndof)

    def fun(pto, wec, x_wec, x_opt, waves, nsubsteps=1):
        # get new flow and effort variable (inputs to efficiency map) by using the MIMO impedance matrix
        # get mechanical power at efficiency map input stage by multiplying the two variables
        # get efficiencies from map, multiply by mech power, integrate?
        # return power time-series, [ntime_steps x ndof_pto]
        return power

    return fun


def _make_abcd(impedance, ndof):
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


def _make_mimo_transfer_mat(impedance_abcd, ndof) -> np.ndarray:
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
                elem[idof][jdof] = block_diag(*blocks)
        return np.block(elem)
