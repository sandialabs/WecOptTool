""" This module provides PTO force and power functions for commonn PTO
control approaches. The PTO power can be used as the objective function
for the control optimization.
"""

from __future__ import annotations  # TODO: delete after python 3.10
from typing import Callable

import numpy.typing as npt
import autograd.numpy as np
from autograd.builtins import isinstance, tuple, list, dict
import xarray as xr

import wecopttool as wot


_finput = [wot.WEC, npt.ArrayLike,  npt.ArrayLike]


# power upper bound
def power_ub(wec: wot.WEC, freq_dom: xr.Dataset, kinematics: np.ndarray
             ) -> tuple[np.ndarray, np.ndarray]:
    """ Calculate the upper bound for PTO power.

    Parameters
    ----------
    wec: wecopttool.WEC
        The WEC.
    kinematics: np.ndarray
        Matrix that converts from the WEC DOFs to the PTO DOFs.
        Shape: (PTO DOFs, WEC DOFs).
    freq_dom: xr.Dataset
        Frequency domain results output from ``wecopttool.WEC.solve``.

    Returns
    -------
    pub_fd: np.ndarray
        Power upper bound in the frequency domain.
    pub_td: np.ndarray
        Power upper bound in the time domain.
    """
    ndof_pto = kinematics.shape[0]

    # frequency domain
    f_exc = np.abs(freq_dom.excitation_force.values[:, 1:])
    f_exc = np.dot(kinematics, f_exc)
    zi = wec.hydro.Zi
    zi = np.array([np.real(zi.values[:, i, i]) for i in range(wec.ndof)])
    zi = np.dot(kinematics, zi)
    pub_fd = 1/8 * f_exc**2 / zi
    pub_fd = np.concatenate([np.zeros((ndof_pto, 1)), pub_fd], axis=1)

    # time_domain
    pub_td = wec.fd_to_td(pub_fd)

    return pub_fd, pub_td


def pseudospectral_pto(
        num_freq: int, kinematics: np.ndarray,
        pto_names: list[str] | None = None) -> tuple[
        int, Callable[_finput, np.ndarray], Callable[_finput, float],
        Callable[_finput, tuple(xr.Dataset, xr.Dataset)]]:
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
        fpto_td = np.dot(np.dot(np.transpose(kinematics), x_pto),
                         wec.phi[1::, ::])
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
        wec_vel_fd = np.dot(x_wec, wec.dphi)
        pto_vel_fd = np.dot(kinematics, wec_vel_fd)
        return np.dot(np.reshape(pto_vel_fd, -1), x_opt) / (2*wec.f0)

    def post_process(wec: wot.WEC, time_dom: xr.Dataset, freq_dom: xr.Dataset,
                     x_opt: npt.ArrayLike) -> tuple[xr.Dataset, xr.Dataset]:
        """ Calculate the time and frequency domain PTO position,
        velocity, force and power, and add these to the results
        datasets.

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

        # frequency domain
        pos_fd = np.dot(kinematics, freq_dom['pos'].values)
        vel_fd = np.dot(kinematics, freq_dom['vel'].values)
        f_pto_fd = wot.fd_folded_nomean(x_pto)
        power_fd = vel_fd * f_pto_fd

        # time domain
        f_pto_td = (x_pto @ wec.phi[1:, :])
        power_td = kinematics @ time_dom['vel'].values * f_pto_td
        pos_td = np.dot(kinematics, time_dom['pos'].values)
        vel_td = np.dot(kinematics, time_dom['vel'].values)

        # power upper bound
        pub_fd, pub_td = power_ub(wec, freq_dom, kinematics)

        time_dom, freq_dom = _add_pto_info(
            time_dom, freq_dom, f_pto_fd, power_fd, pos_fd, vel_fd, pub_fd,
            f_pto_td, power_td, pos_td, vel_td, pub_td, pto_names)

        return time_dom, freq_dom

    return num_x, f_pto, power, post_process


def proportional_pto(
        kinematics: np.ndarray, pto_names: list[str] | None = None) -> tuple[
        int, Callable[_finput, np.ndarray], Callable[_finput, float],
        Callable[_finput, tuple(xr.Dataset, xr.Dataset)]]:
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
        wec_vel_fd = np.dot(x_wec, wec.dphi)
        wec_vel_td = np.dot(wec_vel_fd, wec.phi[1:, :])
        pto_vel_td = np.dot(kinematics, wec_vel_td)
        f_pto_td = -1.0 * x_pto * pto_vel_td
        return np.dot(np.transpose(kinematics), f_pto_td)

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
        x_pto = np.reshape(x_opt, (ndof_pto, 1))
        x_wec = wec.vec_to_dofmat(x_wec)
        wec_vel_fd = np.dot(x_wec, wec.dphi)
        pto_vel_fd = np.dot(kinematics, wec_vel_fd)
        f_pto_fd = -1.0 * x_pto * pto_vel_fd
        power = np.sum(np.reshape(pto_vel_fd, -1) *
                       np.reshape(f_pto_fd, -1)) / (2*wec.f0)
        return power

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
        x_pto = x_opt.reshape([ndof_pto, 1])

        # frequency domain
        pto_vel_fd = kinematics @ freq_dom['vel'].values
        f_pto_fd = x_pto * pto_vel_fd
        power_fd = pto_vel_fd * f_pto_fd / (2*wec.f0)
        pos_fd = np.dot(kinematics, freq_dom['pos'].values)
        vel_fd = np.dot(kinematics, freq_dom['vel'].values)

        # time domain
        pto_vel_td = kinematics @ time_dom['vel'].values
        f_pto_td = -1.0 * x_pto * pto_vel_td
        power_td = pto_vel_td * f_pto_td
        pos_td = np.dot(kinematics, time_dom['pos'].values)
        vel_td = np.dot(kinematics, time_dom['vel'].values)

        # power upper bound
        pub_fd, pub_td = power_ub(wec, freq_dom, kinematics)

        time_dom, freq_dom = _add_pto_info(
            time_dom, freq_dom, f_pto_fd, power_fd, pos_fd, vel_fd, pub_fd,
            f_pto_td, power_td, pos_td, vel_td, pub_td, pto_names)

        return time_dom, freq_dom

    return num_x, f_pto, power, post_process


def _add_pto_info(time_dom: xr.Dataset, freq_dom: xr.Dataset,
                  f_pto_fd: np.ndarray, power_fd: np.ndarray,
                  pos_fd: np.ndarray, vel_fd: np.ndarray, pub_fd: np.ndarray,
                  f_pto_td: np.ndarray, power_td: np.ndarray,
                  pos_td: np.ndarray, vel_td: np.ndarray, pub_td: np.ndarray,
                  pto_names: list[str] | None = None):
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
    attrs_pos = {'long_name': 'PTO position', 'units': 'm or (rad)'}
    attrs_vel = {'long_name': 'PTO velocity', 'units': 'm/s or (rad)/s'}
    attrs_pub = {'long_name': 'PTO power upper bound', 'units': 'W'}
    f_pto_td = xr.DataArray(f_pto_td, dims=dims, coords=coords, attrs=attrs_f)
    power_td = xr.DataArray(power_td, dims=dims, coords=coords, attrs=attrs_p)
    pos_td = xr.DataArray(pos_td, dims=dims, coords=coords, attrs=attrs_pos)
    vel_td = xr.DataArray(vel_td, dims=dims, coords=coords, attrs=attrs_vel)
    pub_td = xr.DataArray(pub_td, dims=dims, coords=coords, attrs=attrs_pub)

    time_dom['pto_force'] = f_pto_td
    time_dom['power'] = power_td
    time_dom['pto_pos'] = pos_td
    time_dom['pto_vel'] = vel_td
    time_dom['power_ub'] = pub_td

    dims[1] = 'omega'
    coords[1] = freq_dom.omega
    attrs_f['units'] = 'N^2*s'
    attrs_p['units'] = 'W^2*s'
    attrs_pos['units'] = 'm^2*s or (rad)^2*s'
    attrs_vel['units'] = 'm^2/s or (rad)^2/s'
    attrs_pub['units'] = 'W^2*s'
    f_pto_fd = xr.DataArray(f_pto_fd, dims=dims, coords=coords, attrs=attrs_f)
    power_fd = xr.DataArray(power_fd, dims=dims, coords=coords, attrs=attrs_p)
    pos_fd = xr.DataArray(pos_fd, dims=dims, coords=coords, attrs=attrs_pos)
    vel_fd = xr.DataArray(vel_fd, dims=dims, coords=coords, attrs=attrs_vel)
    pub_fd = xr.DataArray(pub_fd, dims=dims, coords=coords, attrs=attrs_pub)
    freq_dom['pto_force'] = f_pto_fd
    freq_dom['power'] = power_fd
    freq_dom['pto_pos'] = pos_fd
    freq_dom['pto_vel'] = vel_fd
    freq_dom['power_ub'] = pub_fd

    return time_dom, freq_dom
