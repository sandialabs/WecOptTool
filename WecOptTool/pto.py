
import autograd.numpy as np
import xarray as xr


def pseudospectral_pto(num_freq, kinematics, pto_names=None):
    """ Pseudo-spectral (PS) control: Optimal time-dependent PTO-force.
    Equivalent to conjugate gradient (CC) if no additional constraints.
    """
    ndof_pto = kinematics.shape[0]
    num_x_perdof = 2 * num_freq
    num_x = num_x_perdof * ndof_pto

    def f_pto(wec, x_wec, x_opt):
        x_pto = np.reshape(x_opt, (ndof_pto, num_x_perdof))
        fpto_td = kinematics.transpose() @ x_pto @ wec.phi[1::, ::]
        return fpto_td

    def power(wec, x_wec, x_opt):
        x_wec = wec.vec_to_dofmat(x_wec)
        wec_vel_fd = x_wec @ wec.dphi
        pto_vel_fd = kinematics @ wec_vel_fd
        return pto_vel_fd.flatten() @ x_opt / (2*wec.f0)

    def post_process(wec, time_dom, freq_dom, x_opt):
        x_pto = x_opt.reshape(ndof_pto, num_x_perdof)

        # time domain
        f_pto_td = (x_pto @ wec.phi[1:, :])
        power_td = kinematics @ time_dom['vel'].values * f_pto_td

        # frequency domain
        f_pto_fd = wec.fd_folded_nomean(x_pto, ndof_pto)
        power_fd = kinematics @ freq_dom['vel'].values * f_pto_fd / (2*wec.f0)

        time_dom, freq_dom = _add_pto_info(
            time_dom, freq_dom, f_pto_td, power_td, f_pto_fd, power_fd,
            pto_names)
        return time_dom, freq_dom

    return num_x, f_pto, power, post_process


def proportional_pto(kinematics, pto_names=None):
    """ Proportional (P) control: PTO force proportional to velocity.
    """
    ndof_pto = kinematics.shape[0]
    num_x_perdof = 1
    num_x = num_x_perdof * ndof_pto

    def f_pto(wec, x_wec, x_opt):
        x_pto = x_opt.reshape([ndof_pto, 1])
        x_wec = wec.vec_to_dofmat(x_wec)
        wec_vel_fd = x_wec @ wec.dphi
        wec_vel_td = wec_vel_fd @ wec.phi[1:, :]
        pto_vel_td = kinematics @ wec_vel_td
        f_pto_td = -1.0 * x_pto * pto_vel_td
        return kinematics.transpose() @ f_pto_td

    def power(wec, x_wec, x_opt):
        x_pto = x_opt.reshape([ndof_pto, 1])
        x_wec = wec.vec_to_dofmat(x_wec)
        wec_vel_fd = x_wec @ wec.dphi
        pto_vel_fd = kinematics @ wec_vel_fd
        f_pto_fd = -1.0 * x_pto * pto_vel_fd
        return np.sum(pto_vel_fd.flatten() * f_pto_fd.flatten()) / (2*wec.f0)

    def post_process(wec, time_dom, freq_dom, x_opt):
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
