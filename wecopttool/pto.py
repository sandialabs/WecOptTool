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

from wecopttool.core import WEC, real_to_complex_amplitudes


class PTO:

    def __init__(self, ndof, kinematics, controller, names: list[str] | None = None):
        self.ndof = ndof
        self.kinematics = kinematics  # a function of (wec, x_wec). x_opt, waves for consistency?
        self.controller = controller  # a function of (wec, x_wec, x_opt, waves)

        if names is None:
            names = [f'pto_{i+1}' for i in range(self.ndof)]
        self.names = names

    def _position(self, wec, x_wec):
        return wec.vec_to_dofmat(x_wec)

    def _velocity(self, wec, x_wec):
        pos_wec = self._position(self, wec, x_wec)
        return np.dot(wec.derivative_mat, pos_wec)

    def _acceleration(self, wec, x_wec):
        vel_wec = self._velocity(self, wec, x_wec)
        return np.dot(wec.derivative_mat, vel_wec)

    def _kinematics(self, wec, x_wec, f_wec):
        pos_wec = wec.vec_to_dofmat(x_wec)
        pos_wec_td = np.dot(wec.time_mat, pos_wec)
        f_wec_td = np.dot(wec.time_mat, f_wec)
        assert f_wec_td.shape == (wec.nt, wec.ndof)
        f_pto_td = np.empty((wec.nt, self.ndof))  # TODO: more efficient way to do this, no loop.
        for i in range(wec.nt):
            ipos_wec_td = pos_wec_td[i, :]
            if_wec_td = f_wec_td[i, :]
            kinematics_mat = self.kinematics(wec, ipos_wec_td)
            f_pto_td[i, :] = np.dot(if_wec_td, np.transpose(kinematics_mat))
        return f_pto_td

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
                     waves=None, nsubsteps: int = 1)
        pos_wec = wec.vec_to_dofmat(x_wec)
        pos_wec_td = np.dot(wec.time_mat, pos_wec)
        force_td = self.controller(wec, x_wec, x_opt, waves, nsubsteps)
        assert force_td.shape == (wec.nt, self.ndof)
        force_wec_td = np.empty((wec.nt, wec.ndof))  # TODO: more efficient way to do this, no loop.
        for i in range(wec.nt):
            ipos_wec_td = pos_wec_td[i, :]
            iforce_td = force_td[i, :]
            kinematics_mat = self.kinematics(wec, ipos_wec_td)
            force_wec_td[i, :] = np.dot(iforce_td, kinematics_mat)
        return force_wec_td

    def mechanical_power(self, wec: WEC, x_wec: npt.ArrayLike, x_opt: Optional[npt.ArrayLike],
              waves=None, nsubsteps: int = 1) -> np.ndarray:
        """Calculate the PTO power time-series in each PTO DOF
        for a given system state.
        """
        force_td = self.controller(wec, x_wec, x_opt, waves, nsubsteps)
        vel_td = self.velocity(wec, x_wec, x_opt, waves, nsubsteps)
        return vel_td * force_td

    def mechanical_energy(self, wec: WEC, x_wec: npt.ArrayLike, x_opt: Optional[npt.ArrayLike],
              waves=None, nsubsteps: int = 1):
        power_td = self.mechanical_power(wec, x_wec, x_opt, waves, nsubsteps)
        return np.sum(power_td) * wec.dt/nsubsteps

    def mechanical_average_power(self, wec: WEC, x_wec: npt.ArrayLike, x_opt: Optional[npt.ArrayLike],
              waves=None, nsubsteps: int = 1):
        self.mechanical_energy(wec, x_wec, x_opt, waves, nsubsteps) / wec.tf


# controllers
def controller_unstructured():
    return

# controller_pid()

# kinematics
def kinematics_linear():
    return
