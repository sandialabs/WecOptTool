""" This module provides PTO force and power functions for commonn PTO
control approaches. The PTO power can be used as the objective function
for the control optimization.
"""
from typing import Union, Callable

import autograd.numpy as np
from autograd.builtins import isinstance, tuple, list, dict
import xarray as xr
import numpy.typing as npt

import wecopttool as wot


def pseudospectral_pto(num_freq: int, kinematics: np.ndarray,
                       pto_names: Union[list[str], None] = None
                       ) -> tuple[int, Callable, Callable, Callable]:
    """ Create the relevant parameters and functions for a
    peudo-spectral PTO control.

    Pseudo-spectral (PS) control: Optimal time-dependent PTO-force.
    Equivalent to conjugate gradient (CC) if no additional constraints.

    Parameters
    ----------
    num_freq: int
        Number of frequencies in pseudo-spectral problem. Should match
        the BEM and wave frequencies.
    kinematics: np.ndarray
        Matrix that converts from the WEC DOFs to the PTO DOFs.
        Shape: (PTO DOFs, WEC DOFs).
    pto_names: list, optional
        Names of the PTOs in each DOF. List of strings of length PTO
        DOFs.

    Returns
    -------
    num_x: int
        Number of states describing the PTO controller.
    f_pto: function
        Function that outputs the PTO force for a given state.
    power: function
        Function that outputs the PTO power for a given state.
    post_process: function
        Function for postprocessing the optimization results.
    """
    ndof_pto = kinematics.shape[0]
    num_x_perdof = 2 * num_freq
    num_x = num_x_perdof * ndof_pto

    def f_pto(wec: wot.WEC, x_wec: npt.ArrayLike, x_opt: npt.ArrayLike
              ) -> np.ndarray:
        """
        Calculate the PTO force for a given system state.

        Parameters
        ----------
        wec: wecopttool.WEC
            The WEC.
        x_wec: np.ndarray
            WEC dynamics state.
        x_opt: np.ndarray
            Optimization (PTO controller) state.

        Returns
        -------
        np.ndarray
            PTO force in the time domain.
        """
        x_pto = np.reshape(x_opt, (ndof_pto, num_x_perdof))
        fpto_td = kinematics.transpose() @ x_pto @ wec.phi[1::, ::]
        return fpto_td

    def power(wec: wot.WEC, x_wec: npt.ArrayLike, x_opt: npt.ArrayLike
              ) -> np.ndarray:
        """
        Calculate the PTO power for a given system state.

        Parameters
        ----------
        wec: wot.WEC
            The WEC.
        x_wec: np.ndarray
            WEC dynamics state.
        x_opt: np.ndarray
            Optimization (PTO controller) state.

        Returns
        -------
        float
            PTO power produced over the simulation time.
        """
        x_wec = wec.vec_to_dofmat(x_wec)
        wec_vel_fd = x_wec @ wec.dphi
        pto_vel_fd = kinematics @ wec_vel_fd
        # return float(pto_vel_fd.flatten() @ x_opt / (2*wec.f0))
        return pto_vel_fd.flatten() @ x_opt / (2*wec.f0)

    def post_process(wec: wot.WEC, time_dom: xr.Dataset, freq_dom: xr.Dataset,
                     x_opt: npt.ArrayLike) -> tuple[xr.Dataset, xr.Dataset]:
        """ Calculate the time and frequency domain PTO force and
        power, and add these to the results datasets.

        Parameters
        ----------
        wec: capytaine.FloatingBody
            The WEC.
        time_dom: xr.Dataset
            Time domain results output from ``wecopttool.WEC.solve``.
        freq_dom: xr.Dataset
            Frequency domain results output from ``wecopttool.WEC.solve``.
        x_opt: np.ndarray
            Final (optimized) PTO controller state.

        Returns
        -------
        time_dom: xr.Dataset
            Time domain results with PTO force and power added.
        freq_dom: xr.Dataset
            Frequency domain results with PTO force and power added.
        """
        x_pto = x_opt.reshape(ndof_pto, num_x_perdof)

        # time domain
        f_pto_td = (x_pto @ wec.phi[1:, :])
        power_td = kinematics @ time_dom['vel'].values * f_pto_td

        # frequency domain
        f_pto_fd = wot.fd_folded_nomean(x_pto)
        power_fd = kinematics @ freq_dom['vel'].values * f_pto_fd / (2*wec.f0)

        time_dom, freq_dom = _add_pto_info(
            time_dom, freq_dom, f_pto_td, power_td, f_pto_fd, power_fd,
            pto_names)
        return time_dom, freq_dom

    return num_x, f_pto, power, post_process


def proportional_pto(kinematics: np.ndarray,
                     pto_names: Union[list[str], None] = None
                     ) -> tuple[int, Callable, Callable, Callable]:
    """ Create the relevant parameters and functions for a
    proportional PTO control.

    Proportional (P) control: PTO force proportional to velocity.

    Parameters
    ----------
    kinematics: np.ndarray
        Matrix that converts from the WEC DOFs to the PTO DOFs.
        Shape: (PTO DOFs, WEC DOFs).
    pto_names: list, optional
        Names of the PTOs in each DOF. List of strings of length PTO
        DOFs.

    Returns
    -------
    num_x: int
        Number of states describing the PTO controller.
    f_pto: function
        Function that outputs the PTO force for a given state.
    power: function
        Function that outputs the PTO power for a given state.
    post_process: function
        Function for postprocessing the optimization results.
    """
    ndof_pto = kinematics.shape[0]
    num_x_perdof = 1
    num_x = num_x_perdof * ndof_pto

    def f_pto(wec: wot.WEC, x_wec: npt.ArrayLike, x_opt: npt.ArrayLike
              ) -> np.ndarray:
        """
        Calculate the PTO force for a given system state.

        Parameters
        ----------
        wec: wot.WEC
            The WEC as a capytaine floating body (mesh + DOFs).
        x_wec: np.ndarray
            WEC dynamics state.
        x_opt: np.ndarray
            Optimization (PTO controller) state.

        Returns
        -------
        np.ndarray
            PTO force in the time domain.
        """
        x_pto = x_opt.reshape([ndof_pto, 1])
        x_wec = wec.vec_to_dofmat(x_wec)
        wec_vel_fd = x_wec @ wec.dphi
        wec_vel_td = wec_vel_fd @ wec.phi[1:, :]
        pto_vel_td = kinematics @ wec_vel_td
        f_pto_td = -1.0 * x_pto * pto_vel_td
        return kinematics.transpose() @ f_pto_td

    def power(wec: wot.WEC, x_wec: npt.ArrayLike, x_opt: npt.ArrayLike
              ) -> np.ndarray:
        """
        Calculate the PTO power for a given system state.

        Parameters
        ----------
        wec: wot.WEC
            The WEC.
        x_wec: np.ndarray
            WEC dynamics state.
        x_opt: np.ndarray
            Optimization (PTO controller) state.

        Returns
        -------
        float
            PTO power produced over the simulation time.
        """
        x_pto = x_opt.reshape([ndof_pto, 1])
        x_wec = wec.vec_to_dofmat(x_wec)
        wec_vel_fd = x_wec @ wec.dphi
        pto_vel_fd = kinematics @ wec_vel_fd
        f_pto_fd = -1.0 * x_pto * pto_vel_fd
        return np.sum(pto_vel_fd.flatten() * f_pto_fd.flatten()) / (2*wec.f0)

    def post_process(wec: wot.WEC, time_dom: xr.Dataset, freq_dom: xr.Dataset,
                     x_opt: npt.ArrayLike) -> [xr.Dataset, xr.Dataset]:
        """ Calculate the time and frequency domain PTO force and
        power, and add these to the results datasets.

        Parameters
        ----------
        wec: wot.WEC
            The WEC.
        time_dom: xr.Dataset
            Time domain results output from ``wecopttool.WEC.solve``.
        freq_dom: xr.Dataset
            Frequency domain results output from ``wecopttool.WEC.solve``.
        x_opt: np.ndarray
            Final (optimized) PTO controller state.

        Returns
        -------
        time_dom: xr.Dataset
            Time domain results with PTO force and power added.
        freq_dom: xr.Dataset
            Frequency domain results with PTO force and power added.
        """
        # time domain
        x_pto = x_opt.reshape([ndof_pto, 1])
        pto_vel_td = kinematics @ time_dom['vel'].values
        f_pto_td = -1.0 * x_pto * pto_vel_td
        power_td = pto_vel_td * f_pto_td

        # frequency domain
        pto_vel_fd = kinematics @ freq_dom['vel'].values
        f_pto_fd = x_pto * pto_vel_fd
        power_fd = pto_vel_fd * f_pto_fd / (2*wec.f0)

        time_dom, freq_dom = _add_pto_info(
            time_dom, freq_dom, f_pto_td, power_td, f_pto_fd, power_fd,
            pto_names)
        return time_dom, freq_dom

    return num_x, f_pto, power, post_process


def _add_pto_info(time_dom, freq_dom, f_pto_td, power_td, f_pto_fd, power_fd,
                  pto_names=None):
    """ Add the PTO force and power to the time and frequency domain
    datasets.
    """
    ndof_pto = f_pto_td.shape[0]
    if pto_names is None:
        pto_names = [f'pto_{i+1}' for i in range(ndof_pto)]
    else:
        assert len(pto_names) == ndof_pto
    dims = ['dof_pto', 'time']
    coords = [(dims[0], pto_names), time_dom.time]
    attrs_f = {'long_name': 'PTO force', 'units': 'N or N*m'}
    attrs_p = {'long_name': 'PTO power', 'units': 'W'}
    f_pto_td = xr.DataArray(f_pto_td, dims=dims, coords=coords, attrs=attrs_f)
    power_td = xr.DataArray(power_td, dims=dims, coords=coords, attrs=attrs_p)
    time_dom['pto_force'] = f_pto_td
    time_dom['power'] = power_td

    dims[1] = 'omega'
    coords[1] = freq_dom.omega
    attrs_f['units'] = 'N^2*s'
    attrs_p['units'] = 'W^2*s'
    f_pto_fd = xr.DataArray(f_pto_fd, dims=dims, coords=coords, attrs=attrs_f)
    power_fd = xr.DataArray(power_fd, dims=dims, coords=coords, attrs=attrs_p)
    freq_dom['pto_force'] = f_pto_fd
    freq_dom['power'] = power_fd

    return time_dom, freq_dom
