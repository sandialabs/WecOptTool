
import logging

import autograd.numpy as np
from autograd.builtins import isinstance, tuple, list, dict
from autograd import jacobian
import xarray as xr
import capytaine as cpy
from scipy import optimize
from scipy import sparse


log = logging.getLogger(__name__)

# Default values
_default_parameters = {'rho': 1000.0, 'g': 9.81, 'depth': np.infty}

# TODO: write/create WEC instance to/from file
# TODO: implement: plot_impedance() and get_power_ub()
# TODO: several ongoing changes to Capytaine (see 'TODO: Capytaine'),
#       modify here once those changes are in a Capytaine release
# TODO: figure out why infinite frequency BEM run is not working
# TODO: Docstrings & type hints
#       https://numpydoc.readthedocs.io/en/latest/format.html
#       https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html


class WEC:
    """
    """

    def __init__(self, fb, mass_matrix, hydrostatic_stiffness, f0, num_freq,
                 dissipation=None, stiffness=None, f_add=None,
                 rho=_default_parameters['rho'],
                 depth=_default_parameters['depth'],
                 g=_default_parameters['g']):
        """
        """
        log.info("New WEC.")
        # water properties
        super().__setattr__('rho', rho)
        super().__setattr__('depth', depth)
        super().__setattr__('g', g)
        if g != _default_parameters['g']:
            # TODO: Capytaine: modify Capytaine solver.fill_dataset()
            raise NotImplementedError('Currently only g=9.81 can be used.')

        # WEC
        super().__setattr__('fb', fb)
        super().__setattr__('mass_matrix', mass_matrix)
        super().__setattr__('hydrostatic_stiffness', hydrostatic_stiffness)

        # frequency
        # self.freq = (f0, num_freq)
        super().__setattr__('freq', (f0, num_freq))

        # additional WEC dynamics forces
        def f_zero(self, x_wec, x_opt): return 0.0
        f_add = f_add if (f_add is not None) else f_zero
        super().__setattr__('f_add', f_add)
        super().__setattr__('dissipation', dissipation)
        super().__setattr__('stiffness', stiffness)

    def __setattr__(self, name, value):
        _attrs_delete_mass = ['fb']
        _attrs_delete_stiffness = ['fb', 'rho', 'g']
        _attrs_delete_impedance_not_bem = ['dissipation', 'stiffness']
        _attrs_delete_bem = ['fb', 'rho', 'g', 'depth', 'freq']
        log.info(f"Changing value of '{name}'. " +
                 "This might cause some attributes to be reset.")
        if name in _attrs_delete_mass:
            super().__setattr__('name', value)
            log.info("  Mass matrix deleted. " +
                     "Assign new values to 'self.mass_matrix'")
        if name in _attrs_delete_stiffness:
            super().__setattr__('hydrostatic_stiffness', None)
            log.info("  Hydrostatic stiffness deleted. " +
                     "Assign new values to 'self.hydrostatic_stiffness'")
        if name in _attrs_delete_impedance_not_bem:
            if 'Zi' in self.hydro:
                super().__setattr__('Zi', None)
                log.info("  Impedance matrix deleted. To calculate " +
                         "impedance call 'self._bem_calc_impedance()'")
        if name in _attrs_delete_bem:
            super().__setattr__('hydro', xr.DataArray())
            super().__setattr__('_gi_block_scaled', None)
            super().__setattr__('_gi_scale', None)
            log.info("  BEM data deleted. To run BEM use self.run_bem(...) ")
        super().__setattr__(name, value)

    # frequency properties
    @property
    def freq(self):
        return self._freq

    @freq.setter
    def freq(self, freq):
        if len(freq) != 2:
            msg = "To set the frequency provide a tuple (f0, num_freq)"
            raise TypeError(msg)
        f0, num_freq = freq
        super().__setattr__('_freq', freq_array(f0, num_freq))
        # update phi and dphi
        super().__setattr__('_phi', self._make_phi())
        super().__setattr__('_phi_fs', self._make_phi_fs())
        super().__setattr__('_dphi', self._make_dphi())

    @property
    def f0(self):
        return self._freq[0]

    @property
    def num_freq(self):
        return len(self._freq)

    @property
    def period(self):
        return 1/self._freq

    @property
    def omega(self):
        return self._freq * 2 * np.pi

    @property
    def omegap0(self):
        return np.concatenate([[0.0], self.omega])

    @property
    def phi(self):
        """ Set when frequency is set. """
        return self._phi

    @property
    def phi_fs(self):
        """ Set when frequency is set. """
        return self._phi_fs

    @property
    def phi_vel(self):
        return self.phi[1:, :]

    @property
    def dphi(self):
        """ Set when frequency is set. """
        return self._dphi

    # problem size
    @property
    def ndof(self):
        return self.fb.nb_dofs

    @property
    def nfd(self):
        return 2 * self.num_freq + 1

    @property
    def nfd_nomean(self):
        # return self.nfd - 1
        return 2 * self.num_freq

    @property
    def num_x_wec(self):
        return self.ndof * self.nfd

    # time
    @property
    def time(self):
        return np.linspace(0, 1/self.f0, self.nfd)

    @staticmethod
    def from_file(fpath):
        # TODO: implement
        raise NotImplementedError()

    def to_file(self, fpath):
        # TODO: implement
        raise NotImplementedError()

    # state vector
    def decompose_decision_var(self, x):
        return x[:self.num_x_wec], x[self.num_x_wec:]

    def vec_to_dofmat(self, vec):
        return np.reshape(vec, (self.ndof, -1))

    def dofmat_to_vec(self, mat):
        return mat.flatten()

    def _make_phi(self):
        """
        phi is a matrix for the transformation from Fourier series
        coefficients to time series
        """
        t = np.linspace(0, 1/self.f0, self.nfd, endpoint=False).reshape(1, -1)
        w = self.omega.reshape(1, -1)
        mm = w.T @ t
        phi = np.ones((self.nfd, self.nfd))
        phi[1::2, ::] = np.cos(mm)
        phi[2::2, ::] = np.sin(mm)
        return phi

    def _make_phi_fs(self):
        """ shuffle phi rather than f_i, since autograd doesn't like it.
        """
        tmp_1 = self.phi[0]
        tmp_2 = self.phi[1::2]
        tmp_3 = self.phi[2::2]
        return np.vstack([tmp_1, tmp_2, tmp_3])

    def _make_dphi(self):
        """
        dphi is a matrix used to transform between position and velocity
        """
        omega = np.array(self.omega)
        dphi = np.diag((np.array([[1], [0]]) * omega).flatten('F')[:-1], 1)
        dphi = dphi - dphi.transpose()
        dphi = np.concatenate((np.zeros((2*self.num_freq, 1)), dphi), axis=1)
        return dphi.T

    # frequency domain
    def fd_to_td(self, fd):
        return np.fft.irfft(fd/(2/self.nfd), n=self.nfd)

    def fd_folded(self, fd):
        mean = fd[:, 0:1]
        return np.concatenate((mean, fd[:, 1::2] - fd[:, 2::2]*1j), axis=1)

    def fd_folded_nomean(self, fd, ndof=None):
        if ndof is None:
            ndof = self.ndof
        mean = np.zeros([ndof, 1])
        return np.concatenate((mean, fd[:, 0::2] - fd[:, 1::2]*1j), axis=1)

    # bem
    def run_bem(self, wave_dirs=[0]):
        log.info(f"Running Capytaine (BEM): {self.num_freq} frequencies x " +
                 f"{len(wave_dirs)} wave directions.")
        write_info = ['hydrostatics', 'mesh', 'wavelength', 'wavenumber']
        data = run_bem(self.fb, self.freq, wave_dirs,
                       rho=self.rho, g=self.g, depth=self.depth,
                       write_info=write_info)
        super().__setattr__('hydro', data)
        # add mass and stiffness
        self._bem_add_hydrostatics()
        # calculate impedance
        self._bem_calc_impedance()

    def read_bem(self, fpath):
        log.info(f"Reading BEM data from {fpath}.")
        data = from_netcdf(fpath)
        super().__setattr__('hydro', data)
        # add mass and stiffness
        bmass = 'mass' in self.hydro
        bstiffness = 'hydrostatic_stiffness' in self.hydro
        if bmass:
            assert np.allclose(self.hydro['mass'].values, self.mass_matrix)
        if bstiffness:
            assert np.allclose(self.hydro['hydrostatic_stiffness'].values,
                               self.hydrostatic_stiffness)
        if not (bmass and bstiffness):
            self._bem_add_hydrostatics()
        # add impedance
        if not 'Zi' in self.hydro:
            self._bem_calc_impedance()
        # post-processing needed for solving dynamics
        self._post_process_impedance()

    def write_bem(self, fpath=None):
        """
        Write the BEM solution to netCDF file

        Parameters
        ----------
        fpath : str
        """
        log.info(f"Writting BEM data to {fpath}.")
        to_netcdf(fpath, self.hydro)

    def _bem_add_hydrostatics(self,):
        dims = ['radiating_dof', 'influenced_dof']
        self.hydro['mass'] = (dims, self.mass_matrix)
        self.hydro['hydrostatic_stiffness'] = (
            dims, self.hydrostatic_stiffness)

    def bem_calc_inf_added_mass(self):
        # TODO: not working
        log.info("Running Capytaine for infinite frequency.")
        inf_data = run_bem(
            self.fb, [np.infty], wave_dirs=None,
            rho=self.rho, g=self.g, depth=self.depth)
        self.hydro['Ainf'] = inf_data.added_mass[0, :, :]

    def bem_calc_rao(self):
        self.hydro['rao'] = cpy.post_pro.rao(self.hydro)

    def _bem_calc_impedance(self):
        """
        """
        log.info("Calculating impedance matrix.")
        # TODO: Capytaine: use capytaine after next release
        # impedance = cpy.post_pro.impedance(
        #     self.hydro, self.dissipation, self.stiffness)
        impedance = cpy_impedance(
            self.hydro, self.dissipation, self.stiffness)
        impedance = impedance * -1j/impedance.omega

        # ensure non-negative diagonal
        for iw in impedance['omega'].values:
            B = np.real(impedance.sel(omega=iw).values).diagonal()
            B_min = B.min()
            if B_min < 0.0:
                log.warning("Impedance matrix has negative diagonal terms." +
                            " Setting to zero.")
                for j, jB in enumerate(B):
                    impedance.loc[{"omega": iw}][j, j] = (
                        np.max([0.0, jB]) +
                        1j * np.imag(impedance.loc[{"omega": iw}][j, j]))

        # make symmetric
        impedance = impedance.transpose(
            'omega', 'radiating_dof', 'influenced_dof')
        impedance_transp = impedance.transpose(
            'omega', 'influenced_dof', 'radiating_dof')
        impedance = (impedance + impedance_transp) / 2

        # store
        self.hydro['Zi'] = impedance

        # post-processing needed for solving dynamics
        self._post_process_impedance()

    def _post_process_impedance(self,):
        _gi_block = self._make_gi_block()
        _gi_scale = 1/np.linalg.norm(_gi_block.toarray())
        _gi_block_scaled = _gi_scale * _gi_block.toarray()
        super().__setattr__('_gi_scale', _gi_scale)
        super().__setattr__('_gi_block_scaled', _gi_block_scaled)

    def _make_gi_block(self):
        """
        Makes a block matrix of the MIMO impedance + position

        Returns
        -------
        gi_block : np.ndarray.
        """
        impedance = self.hydro['Zi'].values
        # elem = [[0] * self.ndof] * self.ndof
        elem = [[None]*self.ndof for _ in range(self.ndof)]

        for idof in range(self.ndof):
            for jdof in range(self.ndof):
                K = self.hydrostatic_stiffness[idof, jdof]
                w_impedance = self.omega*impedance[:, idof, jdof]
                elem[idof][jdof] = np.diag(
                    np.concatenate(([K], 1j * w_impedance)))

        return sparse.dia_matrix(np.block(elem))

    # solve
    def solve(self, waves, obj_fun, num_x_opt, constraints=[],
              x_wec_0=None, x_opt_0=None, scale_x_wec=1.0,
              scale_x_opt=1.0, scale_obj=1.0, optim_options={}):
        """
        """
        log.info("Solving pseudo-spectral control problem.")
        # initial state
        if x_wec_0 is None:
            x_wec_0 = np.random.randn(self.num_x_wec)
        if x_opt_0 is None:
            x_opt_0 = np.random.randn(num_x_opt)
        x0 = np.concatenate([x_wec_0, x_opt_0])

        # wave excitation force
        fd_we, td_we = wave_excitation(self.hydro, waves)
        f_exc = td_we['excitation_force']

        # scale
        scale = np.concatenate([
            scale_x_wec * np.ones(self.num_x_wec),
            scale_x_opt * np.ones(num_x_opt)])

        # objective function
        def obj_fun_scaled(x):
            x_wec, x_opt = self.decompose_decision_var(x/scale)
            cost = obj_fun(self, x_wec, x_opt)*scale_obj
            return cost

        # system dynamics through equality constraint
        def resid_fun(x):
            ri = self._dynamic_residual(x/scale, f_exc.values)
            return self.dofmat_to_vec(ri)

        eq_cons = {'type': 'eq',
                   'fun': resid_fun,
                   'jac': jacobian(resid_fun),
                   }
        constraints.append(eq_cons)

        # minimize
        options = {'ftol': 1e-6,
                   'eps': 1.4901161193847656e-08,
                   'disp': False,
                   'iprint': 1,
                   'maxiter': 100,
                   'finite_diff_rel_step': None,
                   }

        for key, value in optim_options.items():
            options[key] = value

        res = optimize.minimize(fun=obj_fun_scaled,
                                x0=x0,
                                jac=jacobian(obj_fun_scaled),
                                method='SLSQP',
                                constraints=constraints,
                                options=options,
                                )

        msg = f'{res.message}    (Exit mode {res.status})'
        if res.status == 0:
            log.info(msg)
        elif res.status == 9:
            log.warning(msg)
        else:
            log.error(msg)

        # unscale
        res.x /= scale
        res.fun /= scale_obj

        # post-process
        x_wec, x_opt = self.decompose_decision_var(res.x)
        freq_dom_x, time_dom_x = self._post_process_x_wec(x_wec)
        fd_we = fd_we.reset_coords(drop=True)
        freq_dom = xr.merge([freq_dom_x, fd_we])
        time_dom = xr.merge([time_dom_x, td_we])
        return freq_dom, time_dom, x_opt, res

    def _dynamic_residual(self, x, f_exc):
        """
        Solves WEC dynamics in residual form so that they may be
        enforced through a nonlinear constraint within an optimization
        problem

        Parameters
        ----------
        x : np.ndarray
            Decision variable for optimization problem
        f_exc : np.ndarray
            Time history of excitation forcing at collocation points in
            body coordinate system

        Returns
        -------
        np.ndarray
            Residuals at collocation points

        """
        # WEC position
        x_wec, x_opt = self.decompose_decision_var(x)
        x_fd = self.vec_to_dofmat(x_wec)
        x_fd_hat = self.fd_folded(x_fd)
        x_fd_hat_vec = self.dofmat_to_vec(x_fd_hat)

        fi_fd = self.vec_to_dofmat(self._gi_block_scaled @ x_fd_hat_vec)
        fi_fd_tmp_0 = np.real(fi_fd[:, 0:1])
        tmp_1_0 = np.real(fi_fd[:, 1::])
        tmp_1_1 = -np.imag(fi_fd[:, 1::])
        fi_fd = np.hstack((fi_fd_tmp_0, tmp_1_0, tmp_1_1))
        f_i = fi_fd @ self.phi_fs

        f_add = self.f_add(self, x_wec, x_opt)

        return f_exc + f_add - f_i

    def _post_process_x_wec(self, x_wec):
        """
        Transform the results from optimization solution to form that
        user can work with directly
        """
        # scale
        x_wec *= self._gi_scale

        # position
        x_fd = self.vec_to_dofmat(x_wec)
        x_fd_hat = self.fd_folded(x_fd)
        x_td = x_fd @ self.phi

        # velocity
        vel_fd = x_fd @ self.dphi
        vel_fd_hat = self.fd_folded_nomean(vel_fd)
        vel_td = vel_fd @ self.phi[1:, :]

        # xarray
        dims_fd = ('influenced_dof', 'omega')
        coords_fd = [
            (dims_fd[0], self.hydro.influenced_dof.values),
            (dims_fd[1], self.omegap0, {'units': '(radians)'})]
        dims_td = ('influenced_dof', 'time')
        coords_td = [
            (dims_td[0], self.hydro.influenced_dof.values),
            (dims_td[1], self.time, {'units': 's'})]
        attrs_x = {'units': 'm or rad', 'long_name': 'WEC position'}
        attrs_vel = {'units': 'm/s or rad/s', 'long_name': 'WEC velocity'}
        x_fd = xr.DataArray(
            x_fd_hat, dims=dims_fd, coords=coords_fd, attrs=attrs_x)
        x_td = xr.DataArray(
            x_td, dims=dims_td, coords=coords_td, attrs=attrs_x)
        vel_fd = xr.DataArray(
            vel_fd_hat, dims=dims_fd, coords=coords_fd, attrs=attrs_vel)
        vel_td = xr.DataArray(
            vel_td, dims=dims_td, coords=coords_td, attrs=attrs_vel)

        freq_dom = xr.Dataset({'pos': x_fd, 'vel': vel_fd},)
        time_dom = xr.Dataset({'pos': x_td, 'vel': vel_td},)

        return freq_dom, time_dom

    def plot_impedance(self, ):
        """
        """
        # TODO: implement, see original code
        raise NotImplementedError()

    def get_pow_ub(self, ):
        """
        Find the upper theoretical limit of power
        """
        # TODO: implement (here or  pto.py?).
        #       See original code
        #       Use PTO DOFs instead of WEC DOFS
        raise NotImplementedError()


def freq_array(f0, num_freq):
    return np.arange(1, num_freq+1)*f0


def wave_excitation(bem_data, waves):
    """
    """
    assert np.allclose(waves['omega'].values, bem_data['omega'].values)
    assert np.allclose(waves['wave_direction'].values,
                       bem_data['wave_direction'].values)

    # excitation BEM
    exc_coeff = bem_data['Froude_Krylov_force'] + \
        bem_data['diffraction_force']

    # add zero frequency
    assert waves.omega[0] != 0
    tmp = waves.isel(omega=0).copy(deep=True)
    tmp['omega'] *= 0
    tmp['S'] *= 0
    tmp['phase'] *= 0
    wavesp0 = xr.concat([tmp, waves], dim='omega')

    assert exc_coeff.omega[0] != 0
    tmp = exc_coeff.isel(omega=0).copy(deep=True)
    tmp['omega'] *= 0
    tmp *= 0
    tmp['wavenumber'] = 0.0
    tmp['wavelength'] = np.inf
    exc_coeff_p0 = xr.concat([tmp, exc_coeff], dim='omega')

    # complex amplitude
    dw = wavesp0.omega[1] - wavesp0.omega[0]
    wave_elev_fd = (np.sqrt(2*wavesp0['S'] / (2*np.pi) * dw) *
                    np.exp(1j*wavesp0['phase']))
    wave_elev_fd.attrs['long_name'] = 'wave elevation'
    wave_elev_fd.attrs['units'] = 'm^2*s'

    # excitation force
    f_exc_fd = xr.dot(exc_coeff_p0, wave_elev_fd, dims=["wave_direction"])
    f_exc_fd.attrs['long_name'] = 'wave excitation force'
    f_exc_fd.attrs['units'] = 'N^2*s or N^2*m^2*s'
    f_exc_fd = f_exc_fd.transpose('influenced_dof', 'omega')

    freq_dom = xr.Dataset(
        {'wave_elevation': wave_elev_fd, 'excitation_force': f_exc_fd},)
    freq_dom['omega'].attrs['long_name'] = 'frequency'
    freq_dom['omega'].attrs['units'] = '(radians)'

    # time domain
    nfd = 2 * len(waves['omega']) + 1
    f0 = waves['omega'][0] / (2*np.pi)
    time = np.linspace(0, 1/f0, nfd)
    dims_td = ['time', ]
    coords_td = [(dims_td[0], time, {'units': 's'}), ]

    def fd_to_td(freq_dom):
        return np.fft.irfft(freq_dom/(2/nfd), n=nfd)

    f_exc_td = fd_to_td(f_exc_fd)
    dims = ['influenced_dof'] + dims_td
    coords = [(dims[0], f_exc_fd.coords[dims[0]],)] + coords_td
    f_exc_td = xr.DataArray(
        f_exc_td, dims=dims, coords=coords, attrs=f_exc_fd.attrs)
    f_exc_td.attrs['units'] = 'N or N*m'
    time_dom = xr.Dataset({'excitation_force': f_exc_td},)

    eta_all = fd_to_td(wave_elev_fd)
    wave_elev_td = np.sum(eta_all, axis=0)
    wave_elev_td = xr.DataArray(
        wave_elev_td, dims=dims_td, coords=coords_td, attrs=wave_elev_fd.attrs)
    wave_elev_td.attrs['units'] = 'm'
    time_dom['wave_elevation'] = wave_elev_td

    return freq_dom, time_dom


def run_bem(fb, freq=[np.infty], wave_dirs=[0],
            rho=_default_parameters['rho'],
            g=_default_parameters['g'],
            depth=_default_parameters['depth'],
            write_info=[]):
    """ Run Capytaine for a range of frequencies and wave directions.
    """
    solver = cpy.BEMSolver()
    test_matrix = xr.Dataset(coords={
        'rho': [rho],
        'water_depth': [depth],
        'omega': [ifreq*2*np.pi for ifreq in freq],
        'wave_direction': wave_dirs,
        'radiating_dof': list(fb.dofs.keys()),
        # TODO: Capytaine: modify solver.fill_dataset() to not hardcode g=9.81.
        'g': [g],
    })
    if wave_dirs is None:
        # radiation only problem, no diffraction or excitation
        test_matrix.drop('wave_direction')
    # TODO: Capytaine: run keep_immersed_part() automatically (#62)
    wec_im = fb.copy(name=f"{fb.name}_immersed").keep_immersed_part()
    write_info = {key: True for key in write_info}
    return solver.fill_dataset(test_matrix, [wec_im], **write_info)


def from_netcdf(fpath):
    return cpy.io.xarray.merge_complex_values(xr.open_dataset(fpath))


def to_netcdf(fpath, bem_data):
    cpy.io.xarray.separate_complex_values(bem_data).to_netcdf(fpath)


def cpy_impedance(bem_data, dissipation=None, stiffness=None):
    # TODO: Capytaine: this is in Capytaine but not in release yet
    omega = bem_data.coords['omega']
    impedance = (-omega**2*(bem_data['mass'] + bem_data['added_mass']) +
                 1j*omega*bem_data['radiation_damping'] +
                 bem_data['hydrostatic_stiffness'])
    if dissipation is not None:
        impedance = impedance + 1j*omega*dissipation
    if stiffness is not None:
        impedance = impedance + stiffness
    return impedance
