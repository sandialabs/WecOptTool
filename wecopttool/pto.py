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

from wecopttool.core import WEC


class PTO:

    def __init__(self, ndof, ncomponents, kinematics, controller, names: list[str] | None = None):
        """
        ndof: int
        kinematics: wec, wec_position -> kinematics matrix
        controller: wec, x_wec, x_opt, waves -> pto force in td in pto frame
        """
        self.ndof = ndof
        self.ncomponents = ncomponents
        self.controller = controller  # a function of (wec, pto, x_wec, x_opt, waves)
        # kinematics
        if callable(kinematics):
            kinematics_fun = kinematics
        else:
            kinematics = np.atleast_2d(kinematics)
            if kinematics.shape[0]!=ndof or kinematics.ndim!=2:
                raise ValueError(
                    "`kinematics` matrix must have shape equal to " +
                    "number of PTO DOF x number of WEC DOF.")

            def kinematics_fun(wec, pos_wec_td):
                n = pos_wec_td.shape[0]
                return np.repeat(kinematics[:, :, np.newaxis], n, axis=-1)

        self.kinematics = kinematics_fun
        # names
        if names is None:
            names = [f'pto_{i+1}' for i in range(ndof)]
        self.names = names

    @property
    def nstate(self):
        return self.ndof*self.ncomponents

    def _position(self, wec, x_wec):
        return wec.vec_to_dofmat(x_wec)

    def _velocity(self, wec, x_wec):
        pos_wec = self._position(wec, x_wec)
        return np.dot(wec.derivative_mat, pos_wec)

    def _acceleration(self, wec, x_wec):
        vel_wec = self._velocity(wec, x_wec)
        return np.dot(wec.derivative_mat, vel_wec)

    def _kinematics(self, wec, x_wec, f_wec):
        pos_wec = wec.vec_to_dofmat(x_wec)
        pos_wec_td = np.dot(wec.time_mat, pos_wec)
        f_wec_td = np.dot(wec.time_mat, f_wec)
        assert f_wec_td.shape == (wec.nt, wec.ndof)
        f_wec_td = np.expand_dims(np.transpose(f_wec_td), axis=0)
        assert f_wec_td.shape == (1, wec.ndof, wec.nt)
        kinematics_mat = self.kinematics(wec, pos_wec_td)
        return np.transpose(np.sum(kinematics_mat*f_wec_td, axis=1))

    def position(self, wec: WEC, x_wec: npt.ArrayLike, x_opt: Optional[npt.ArrayLike],
                 waves=None, nsubsteps: int = 1) -> np.ndarray:
        """Calculate the PTO position time-series."""
        return self._kinematics(wec, x_wec, self._position(wec, x_wec))

    def velocity(self, wec: WEC, x_wec: npt.ArrayLike, x_opt: Optional[npt.ArrayLike],
                 waves=None, nsubsteps: int = 1) -> np.ndarray:
        """Calculate the PTO position time-series."""
        return self._kinematics(wec, x_wec, self._velocity(wec, x_wec))

    def acceleration(self, wec: WEC, x_wec: npt.ArrayLike, x_opt: Optional[npt.ArrayLike],
                 waves=None, nsubsteps: int = 1) -> np.ndarray:
        """Calculate the PTO position time-series."""
        return self._kinematics(wec, x_wec, self._acceleration(wec, x_wec))

    def force_on_wec(self, wec: WEC, x_wec: npt.ArrayLike, x_opt: Optional[npt.ArrayLike],
                     waves=None, nsubsteps: int = 1):
        pos_wec = wec.vec_to_dofmat(x_wec)
        pos_wec_td = np.dot(wec.time_mat, pos_wec)
        force_td = self.controller(wec, self, x_wec, x_opt, waves, nsubsteps)
        assert force_td.shape == (wec.nt, self.ndof)
        force_td = np.expand_dims(np.transpose(force_td), axis=0)
        assert force_td.shape == (1, wec.ndof, wec.nt)
        kinematics_mat = self.kinematics(wec, pos_wec_td)
        return np.transpose(np.sum(kinematics_mat*force_td, axis=1))

    def mechanical_power(self, wec: WEC, x_wec: npt.ArrayLike, x_opt: Optional[npt.ArrayLike],
              waves=None, nsubsteps: int = 1) -> np.ndarray:
        """Calculate the PTO power time-series in each PTO DOF
        for a given system state.
        """
        force_td = self.controller(wec, self, x_wec, x_opt, waves, nsubsteps)
        vel_td = self.velocity(wec, x_wec, x_opt, waves, nsubsteps)
        return vel_td * force_td

    def mechanical_energy(self, wec: WEC, x_wec: npt.ArrayLike, x_opt: Optional[npt.ArrayLike],
              waves=None, nsubsteps: int = 1):
        power_td = self.mechanical_power(wec, x_wec, x_opt, waves, nsubsteps)
        return np.sum(power_td) * wec.dt/nsubsteps

    def mechanical_average_power(self, wec: WEC, x_wec: npt.ArrayLike, x_opt: Optional[npt.ArrayLike],
              waves=None, nsubsteps: int = 1):
        energy = self.mechanical_energy(wec, x_wec, x_opt, waves, nsubsteps)
        return energy / wec.tf


# controllers
def controller_unstructured(wec: WEC, pto:PTO, x_wec: npt.ArrayLike,
                            x_opt: Optional[npt.ArrayLike], waves=None,
                            nsubsteps=1):
    x_opt = np.reshape(x_opt, (pto.ncomponents, pto.ndof), order='F')
    tmat = wec.time_mat if nsubsteps==1 else wec.time_mat_nsubsteps(nsubsteps)
    return np.dot(tmat, x_opt)


def _controller_pid(wec: WEC, pto:PTO, x_wec: npt.ArrayLike,
                   x_opt: Optional[npt.ArrayLike], waves=None,
                   nsubsteps=1,
                   proportional=True, integral=True, derivative=True):
    force_td = np.zeros([wec.nt, pto.ndof])
    idx = 0

    def update_force_td(B):
        nonlocal idx, force_td
        u = np.reshape(x_opt[idx*pto.ndof:(idx+1)*pto.ndof], [1, pto.ndof])
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


def controller_pid(wec: WEC, pto:PTO, x_wec: npt.ArrayLike,
                   x_opt: Optional[npt.ArrayLike], waves=None,
                   nsubsteps=1):
    force = _controller_pid(wec, pto, x_wec, x_opt, waves, nsubsteps,
                            proportional=True, integral=True, derivative=True)
    return force


def controller_pi(wec: WEC, pto:PTO, x_wec: npt.ArrayLike,
                  x_opt: Optional[npt.ArrayLike], waves=None,
                  nsubsteps=1):
    force = _controller_pid(wec, pto, x_wec, x_opt, waves, nsubsteps,
                            proportional=True, integral=True, derivative=False)
    return force


def controller_p(wec: WEC, pto:PTO, x_wec: npt.ArrayLike,
                 x_opt: Optional[npt.ArrayLike], waves=None,
                 nsubsteps=1):
    force = _controller_pid(wec, pto, x_wec, x_opt, waves, nsubsteps,
                            proportional=True, integral=False, derivative=False)
    return force