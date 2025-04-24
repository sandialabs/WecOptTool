"""Provide simple PID controllers

Contains:

* pid_controller class

"""


from __future__ import annotations


__all__ = [
    "unstructured_controller",
    "pid_controller",
]

import logging
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

# logger
_log = logging.getLogger(__name__)

TPTO = TypeVar("TPTO", bound="PTO")
Tcontroller = TypeVar("Tcontroller", bound='_controller_')

class _controller_:
    pass


class pid_controller(_controller_):
    """PID controller object to be used in conjunction with a
    :py:class:`wecopttool.pto` object.
    """
    def __init__(self,
                 ndof_pto: int,
                 proportional: Optional[bool] = True,
                 integral: Optional[bool] = False,
                 derivative: Optional[bool] = False,
                 saturation: Optional[FloatOrArray] = None,
                 ):

        self._proportional = proportional
        self._integral = integral
        self._derivative = derivative
        self._saturation = saturation

        self._ndof = ndof_pto
        
    @property
    def proportional(self) -> bool:
        '''True if proportional control element used.'''
        return self._proportional
    
    @property
    def integral(self) -> bool:
        '''True if integral control element used.'''
        return self._integral
    
    @property
    def derivative(self) -> bool:
        '''True if derivative control element used.'''
        return self._derivative
    
    @property
    def saturation(self) -> FloatOrArray:
        '''Saturation values for control force. `None` if no saturation.'''
        return self._saturation
    
    @property
    def ndof(self) -> int:
        '''Number of degrees of freedom'''
        return self._ndof
    
    @property
    def ngains(self) -> int:
        '''Number of controller gains per dof'''
        return self.proportional + self.integral + self.derivative
        
    @property
    def nstate(self) -> int:
        '''Total number of controller gains across all DOFs'''
        return self.ndof * self.ngains
        
    def _gains(self, x_opt):
        idx = 0
        ndof = self.ndof

        if self.proportional:
            gain_p = np.diag(x_opt[idx*ndof:(idx+1)*ndof])
            idx = idx + 1
        else:
            gain_p = np.zeros([ndof, ndof])

        if self.integral:
            gain_i = np.diag(x_opt[idx*ndof:(idx+1)*ndof])
            idx = idx + 1
        else:
            gain_i = np.zeros([ndof, ndof])

        if self.derivative:
            gain_d = np.diag(x_opt[idx*ndof:(idx+1)*ndof])
        else:
            gain_d = np.zeros([ndof, ndof])

        return gain_p, gain_i, gain_d
        
    def force(self,
              pto: TPTO,
              wec: TWEC,
              x_wec: ndarray,
              x_opt: ndarray,
              waves: Optional[Dataset] = None,
              nsubsteps: Optional[int] = 1):
        '''Time history of PTO force'''

        gain_p, gain_i, gain_d = self._gains(x_opt)

        vel_td = pto.velocity(wec, x_wec, x_opt, waves, nsubsteps)
        pos_td = pto.position(wec, x_wec, x_opt, waves, nsubsteps)
        acc_td = pto.acceleration(wec, x_wec, x_opt, waves, nsubsteps)

        force_td = (
            np.dot(vel_td, gain_p.T) +
            np.dot(pos_td, gain_i.T) +
            np.dot(acc_td, gain_d.T)
        )

        if self.saturation is not None:
            force_td = self._apply_saturation(force_td)

        return force_td
    
    def _apply_saturation(self, force_td):
        if self.saturation is not None:
            saturation = np.atleast_2d(np.squeeze(self.saturation))
            assert len(saturation)==self.ndof
            if len(saturation.shape) > 2:
                raise ValueError("`saturation` must have <= 2 dimensions.")

            if saturation.shape[1] == 1:
                f_min, f_max = -1*saturation, saturation
            elif saturation.shape[1] == 2:
                f_min, f_max = saturation[:,0], saturation[:,1]
            else:
                raise ValueError("`saturation` must have 1 or 2 columns.")

            force_td_list = []
            for i in range(self.ndof):
                tmp = np.clip(force_td[:,i], f_min[i], f_max[i])
                force_td_list.append(tmp)
            force_td = np.array(force_td_list).T
        return force_td


class unstructured_controller(_controller_):
    def __init__(self):
        pass

    def force(self,
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