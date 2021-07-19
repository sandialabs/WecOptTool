import autograd.numpy as np
import xarray as xr

""" Pseudo-spectral (PS): Optimal time-dependent PTO-force
    * Equivalent to conjugate gradient (CC) if no additional constraints
"""
def pseudospectral_pto(num_freq, kinematics):
    ndof_pto = kinematics.shape[0]
    num_x_perdof = 2 * num_freq
    num_x = num_x_perdof * ndof_pto

    def f_pto(wec, x_wec, x_opt):
        x_pto = np.reshape(x_opt, (ndof_pto, num_x_perdof))
        fpto = kinematics.transpose() @ x_pto @ wec.Phi[1::, ::]
        return fpto

    def power(wec, x_wec, x_opt):
        wec_vel = wec.vec_to_dofmat(wec.Dphi @ x_wec)
        pto_vel = kinematics @ wec_vel
        return pto_vel.flatten() @ x_opt / (2*wec.f0)

    def post_process(wec, TD, FD, x_opt):
        x_pto = x_opt.reshape(ndof_pto, -1)

        # time domain
        f_pto = (x_pto @ wec.Phi[1:, :])
        power = TD['vel'].values * f_pto

        # frequency domain
        F_PTO = wec.fd_folded_nomean(x_pto)
        POWER = FD['vel'] * F_PTO / (2*wec.f0)

        TD, FD = _add_pto_info(TD, FD, f_pto, power, F_PTO, POWER)
        return TD, FD

    return num_x, f_pto, power, post_process


def _add_pto_info(TD, FD, f_pto, power, F_PTO, POWER):
    dims = ['dof_pto', 'time']
    coords = [(dims[0], ['pto_1']), TD.time]
    attrs_f = {'long_name': 'PTO force', 'units': 'N'}
    attrs_p = {'long_name': 'PTO power', 'units': 'W'}
    f_pto = xr.DataArray(f_pto, dims=dims, coords=coords, attrs=attrs_f)
    power = xr.DataArray(power, dims=dims, coords=coords, attrs=attrs_p)
    TD['pto_force'] = f_pto
    TD['power'] = power

    dims[1] = 'omega'
    coords[1] = FD.omega
    attrs_f['units'] = 'N^2*s'
    attrs_p['units'] = 'W^2*s'
    F_PTO = xr.DataArray(F_PTO, dims=dims, coords=coords, attrs=attrs_f)
    POWER = xr.DataArray(POWER, dims=dims, coords=coords, attrs=attrs_p)
    FD['pto_force'] = F_PTO
    FD['power'] = POWER

    return TD, FD

""" Proportional (P): PTO force proportional to velocity.
"""
def proportional_pto(kinematics):
    ndof_pto = kinematics.shape[0]
    num_x_perdof = 1
    num_x = num_x_perdof * ndof_pto

    def f_pto(wec, x_wec, x_opt):
        wec_vel = wec.vec_to_dofmat(wec.Dphi @ x_wec)
        wec_vel_td = wec_vel @ wec.Phi[1:, :]
        pto_vel_td = kinematics @ wec_vel_td
        f_pto_td = -1.0 * x_opt * pto_vel_td
        return kinematics.transpose() @ f_pto_td

    def power(wec, x_wec, x_opt):
        wec_vel = wec.vec_to_dofmat(wec.Dphi @ x_wec)
        pto_vel = kinematics @ wec_vel
        return np.sum(pto_vel.flatten() * x_opt) / (2*wec.f0)

    def post_process(wec, TD, FD, x_opt):
        # time domain
        f_pto = TD['vel'].values * x_opt
        power = TD['vel'].values * f_pto

        # frequency domain
        F_PTO = FD['vel'] * x_opt
        POWER = FD['vel'] * F_PTO / (2*wec.f0)

        TD, FD = _add_pto_info(TD, FD, f_pto, power, F_PTO, POWER)
        return TD, FD

    return num_x, f_pto, power, post_process

