
import autograd.numpy as np
from autograd.builtins import isinstance, tuple, list, dict
from autograd import jacobian
import xarray as xr
import capytaine as cpy
from scipy import optimize
from scipy import sparse

# Default values
_default_parameters = {'rho': 1000.0, 'g': 9.81, 'depth': np.infty}

# TODO: write/create WEC instance to/from file
# TODO: logging
# TODO: docstrings, type hinting
# TODO: style (pep8): line length, naming convention, etc.
# TODO: reshape & flatten

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
        # BEM placeholder
        self.hydro = xr.DataArray()

        # water properties
        self.rho = rho
        self.depth = depth
        if g != _default_parameters['g']:
            # TODO: requires modifying Capytaine solver.fill_dataset()
            raise NotImplementedError('Currently only g=9.81 can be used.')
        self.g = g

        # WEC
        self.fb = fb
        self.mass_matrix = mass_matrix
        self.hydrostatic_stiffness = hydrostatic_stiffness

        # frequency
        self.freq = (f0, num_freq)

        # additional WEC dynamics forces
        def f_zero(self, x_wec, x_opt): return 0.0
        self.f_add = f_add if (f_add is not None) else f_zero
        self.dissipation = dissipation
        self.stiffness = stiffness

    def __setattr__(self, name, value):
        # TODO: logging
        _attrs_delete_mass = ['fb']
        _attrs_delete_stiffness = ['fb', 'rho', 'g']
        _attrs_delete_impedance_not_bem = ['dissipation', 'stiffness']
        _attrs_delete_bem = ['fb', 'rho', 'g', 'depth', 'freq']
        if name in _attrs_delete_mass:
            self.mass_matrix = None
        if name in _attrs_delete_stiffness:
            self.hydrostatic_stiffness = None
        if name in _attrs_delete_impedance_not_bem:
            if 'Zi' in self.hydro:
                self.hydro['Zi'] = None
        if name in _attrs_delete_bem:
            self.hydro = xr.DataArray()
            self._Gi_block_scaled = None
            self._Gi_scale = None
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
        self._freq = freq_array(f0, num_freq)
        # update Phi and DPhi
        self._Phi = self._make_Phi()
        self._Dphi = self._make_Dphi()

    @property
    def f0(self):
        return self._freq[0]

    @property
    def num_freq(self):
        return len(self._freq)

    @property
    def T(self):
        return 1/self._freq

    @property
    def omega(self):
        return self._freq * 2 * np.pi

    @property
    def omegap0(self):
        return np.concatenate([[0.0], self.omega])

    @property
    def Phi(self):
        """ Set when frequency is set. """
        return self._Phi

    @property
    def Phi_vel(self):
        return self.Phi[1:, :]

    @property
    def Dphi(self):
        """ Set when frequency is set. """
        return self._Dphi

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

    # state vector
    def decompose_decision_var(self, x):
        return x[:self.num_x_wec], x[self.num_x_wec:]

    def vec_to_dofmat(self, vec):
        return np.reshape(vec, (self.ndof, -1))

    def dofmat_to_vec(self, mat):
        return mat.flatten()

    def _make_Phi(self) -> np.array:
        """
        Phi is a matrix for the transformation from Fourier series
        coefficients to time series
        """
        T = np.linspace(0, 1/self.f0, self.nfd, endpoint=False).reshape(1, -1)
        w = self.omega.reshape(1, -1)
        MM = w.T @ T
        Phi = np.ones((self.nfd, self.nfd))
        Phi[1::2, ::] = np.cos(MM)
        Phi[2::2, ::] = np.sin(MM)
        return Phi

    def _make_Dphi(self) -> np.array:
        """
        Dphi is a matrix used to transform between position and velocity
        """
        omega = np.array(self.omega)
        Dphi = np.diag((np.array([[1], [0]]) * omega).flatten('F')[:-1], 1)
        Dphi = Dphi - Dphi.transpose()
        Dphi = np.concatenate((np.zeros((2*self.num_freq, 1)), Dphi), axis=1)
        return Dphi

    # frequency domain
    def fd_to_td(self, FD):
        return np.fft.irfft(FD/(2/self.nfd), n=self.nfd)

    def fd_folded(self, FD):
        mean = FD[:, 0:1]
        return np.concatenate((mean, FD[:, 1::2] - FD[:, 2::2]*1j), axis=1)

    def fd_folded_nomean(self, FD):
        mean = np.zeros([self.ndof, 1])
        return np.concatenate((mean, FD[:, 0::2] - FD[:, 1::2]*1j), axis=1)

    # bem
    def run_bem(self, wave_dirs=[0]):
        write_info = ['hydrostatics', 'mesh', 'wavelength', 'wavenumber']
        data = run_bem(self.fb, self.freq, wave_dirs,
                       rho=self.rho, g=self.g, depth=self.depth,
                       write_info=write_info)
        self.hydro = data
        # add mass and stiffness
        self._bem_add_hydrostatics()
        # calculate impedance
        self._bem_calc_impedance()

    def read_bem(self, fpath) -> xr.Dataset:
        self.hydro = from_netcdf(fpath)
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

    def write_bem(self, fpath: str = None):
        """
        Write the BEM solution to netCDF file

        Parameters
        ----------
        fpath : str
        """
        to_netcdf(fpath, self.hydro)

    def _bem_add_hydrostatics(self,):
        dims = ['radiating_dof', 'influenced_dof']
        self.hydro['mass'] = (dims, self.mass_matrix)
        self.hydro['hydrostatic_stiffness'] = (
            dims, self.hydrostatic_stiffness)

    def bem_calc_Ainf(self):
        inf_data = run_bem(
            self.fb, [np.infty], wave_dirs=None,
            rho=self.rho, g=self.g, depth=self.depth)
        self.hydro['Ainf'] = inf_data.added_mass[0, :, :]

    def bem_calc_rao(self):
        self.hydro['rao'] = cpy.post_pro.rao(self.hydro)

    def _bem_calc_impedance(self):
        """
        """
        # TODO: use capytaine after next release
        # A = cpy.post_pro.impedance(
            # self.hydro, self.dissipation, self.stiffness)
        A = cpy_impedance(
            self.hydro, self.dissipation, self.stiffness)
        Zi = A * -1j/A.omega

        # ensure non-negative diagonal
        for iw in Zi['omega'].values:
            B = np.real(Zi.sel(omega=iw).values).diagonal()
            B_min = B.min()
            if B_min < 0.0:
                msg = "WARNING: impedance matrix has negative diagonal " + \
                    "terms. Setting to zero."
                print(msg)  # TODO: use logging
                for j, jB in enumerate(B):
                    Zi.loc[{"omega": iw}][j, j] = (
                        np.max([0.0, jB]) +
                        1j * np.imag(Zi.loc[{"omega": iw}][j, j]))

        # make symmetric
        Zi = Zi.transpose('omega', 'radiating_dof', 'influenced_dof')
        ZiT = Zi.transpose('omega', 'influenced_dof', 'radiating_dof')
        Zi = (Zi + ZiT) / 2

        # store
        self.hydro['Zi'] = Zi

        # post-processing needed for solving dynamics
        self._post_process_impedance()

    def _post_process_impedance(self,):
        _Gi_block = self._make_Gi_block()
        self._Gi_scale = 1/np.linalg.norm(_Gi_block.toarray())
        self._Gi_block_scaled = self._Gi_scale * _Gi_block.toarray()

    def _make_Gi_block(self) -> np.ndarray:
        """
        Makes a block matrix of the MIMO impedance + position

        Returns
        -------
        Gi_block : np.ndarray.
        """
        Zi = self.hydro.Zi.values
        elem = [[0] * self.ndof] * self.ndof

        for i in range(self.ndof):
            for j in range(self.ndof):
                K = self.hydrostatic_stiffness[i, j]
                wZi = self.omega*Zi[:, i, j]
                elem[i][j] = np.diag(np.concatenate(([K], 1j * wZi)))

        Gi_block = sparse.dia_matrix(np.block(elem))

        return Gi_block

    # solve
    def solve(self, waves, obj_fun, num_x_opt, x_wec_0=None, x_opt_0=None,
              scale_x_wec=1.0, scale_x_opt=1.0, scale_obj=1.0,
              constraints=[], optim_options={}):
        """
        """
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
                'disp': True,
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

        # unscale
        res.x /= scale
        res.fun /= scale_obj

        # post-process
        x_wec, x_opt = self.decompose_decision_var(res.x)
        FD_X, TD_X = self._post_process_x_wec(x_wec)
        fd_we = fd_we.reset_coords(drop=True)
        FD = xr.merge([FD_X, fd_we])
        TD = xr.merge([TD_X, td_we])
        return FD, TD, x_opt, res

    def _dynamic_residual(self, x, f_exc):
        """
        Solves WEC dynamics in residual form so that they may be enforced through
        a nonlinear constraint within an optimization problem

        Parameters
        ----------
        x : np.ndarray
            Decision variable for optimization problem
        f_exc : np.ndarray
            Time history of excitation forcing at collocation points in body
            coordinate system

        Returns
        -------
        np.ndarray
            Residuals at collocation points

        """
        # WEC position
        x_wec, x_opt = self.decompose_decision_var(x)
        X = self.vec_to_dofmat(x_wec)
        X_hat = self.fd_folded(X)
        X_hat_vec = self.dofmat_to_vec(X_hat)

        # TODO: not working for multiple DOF
        Fi = self.vec_to_dofmat(self._Gi_block_scaled @ X_hat_vec)
        Fi_fs_tmp_0 = np.real(Fi[:, 0:1])
        tmp_1_0 = np.real(Fi[:, 1::])
        tmp_1_1 = -np.imag(Fi[:, 1::])
        # Fi_fs_tmp_1 = np.dstack([tmp_1_0, tmp_1_1]).reshape([self.ndof, -1]) # TODO: does not work, even though same array! autograd?
        Fi_fs_tmp_1 = np.vstack([tmp_1_0, tmp_1_1]).flatten('F').reshape(self.ndof, -1) # TODO: does not work with multiple DOF
        Fi_fs = np.hstack((Fi_fs_tmp_0, Fi_fs_tmp_1))
        fi = Fi_fs @ self.Phi

        f_add = self.f_add(self, x_wec, x_opt)

        residual = f_exc + f_add - fi
        return residual

    def _post_process_x_wec(self, x_wec):
        """
        Transform the results from optimization solution to form that user can work
        with directly
        """
        # scale
        x_wec *= self._Gi_scale

        # position
        X = self.vec_to_dofmat(x_wec)
        X_hat = self.fd_folded(X)
        x = X @ self.Phi

        # velocity
        VEL = X @ self.Dphi.T
        VEL_hat = self.fd_folded_nomean(VEL)
        vel = VEL @ self.Phi[1:, :]

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
        X = xr.DataArray(X_hat, dims=dims_fd, coords=coords_fd, attrs=attrs_x)
        x = xr.DataArray(x, dims=dims_td, coords=coords_td, attrs=attrs_x)
        VEL = xr.DataArray(VEL_hat, dims=dims_fd,
                           coords=coords_fd, attrs=attrs_vel)
        vel = xr.DataArray(vel, dims=dims_td,
                           coords=coords_td, attrs=attrs_vel)

        FD = xr.Dataset({'pos': X, 'vel': VEL},)
        TD = xr.Dataset({'pos': x, 'vel': vel},)

        return FD,TD

    def plot_impedance(self, diag_only:bool=True, axs=None):
        """
        Parameters
        ----------
        diag_only : bool, optional
            Only plot diagonal elements. The default is True.
        axs : matplotlib.axes._subplots.AxesSubplot, optional

        Raises
        ------
        NotImplementedError
            DESCRIPTION.

        Returns
        -------
        None.

        """
        raise NotImplementedError()  # TODO
        # if axs is None:
        #     fig, axs = plt.subplots()

        # if diag_only:
        #     for idx, dof in enumerate(self.hydro.radiating_dof.data.tolist()):
        #         Zi_frd = control.frd(self.hydro.Zi[:,idx,idx], self.hydro.omega)
        #         control.bode_plot(Zi_frd, self.hydro.omega,
        #                           dB=True,
        #                           Hz=True,
        #                           marker='.',
        #                           label=dof)
        #     plt.legend()
        # else:
        #     raise NotImplementedError()

    def get_pow_ub(self, S, dof: int =2) -> xr.DataArray:
        """
        Find the upper theoretical limit of power

        Parameters
        ----------
        S : pandas.core.frame.DataFrame
            Wave spectrum created by MHKiT.
        dof : int, optional
            degree-of-freedom. The default is 2 (heave).

        Returns
        -------
        P_ub : xr.DataArray
            Spectrum of power upper bound (sum for total).

        """
        # TODO: implement (here or  pto.py?)
        # TODO: use PTO DOFs instead of WEC DOFS
        # assert isinstance(dof, int)

        # Fexc = self._get_waveExcitation(S)['F_EXC'].isel(
        #     influenced_dof=dof).squeeze()
        # # print(self.hydro)
        # Zi = self.hydro['Zi'].isel(dict(influenced_dof=dof,
        #                                 radiating_dof=dof)).squeeze()

        # P_ub = 1/8 * np.abs(Fexc)**2 / np.real(Zi)

        # return P_ub



def freq_array(f0, num_freq):
    return np.arange(1, num_freq+1)*f0

# WEC dynamics
def wave_excitation(bem_data, waves):
    """
    """
    assert np.allclose(waves['omega'].values, bem_data['omega'].values)

    # excitation BEM
    H = bem_data['Froude_Krylov_force'] + bem_data['diffraction_force']

    # add zero frequency
    assert waves.omega[0] != 0
    tmp = waves.isel(omega=0).copy(deep=True)
    tmp['omega'] *= 0
    tmp['S'] *= 0
    tmp['phase'] *= 0
    wavesp0 = xr.concat([tmp, waves], dim='omega')

    assert H.omega[0] != 0
    tmp = H.isel(omega=0).copy(deep=True)
    tmp['omega'] *= 0
    tmp *= 0
    tmp['wavenumber'] = 0.0
    tmp['wavelength'] = np.inf
    Hp0 = xr.concat([tmp, H], dim='omega')

    # complex amplitude
    dw = wavesp0.omega[1] - wavesp0.omega[0]
    ETA = np.sqrt(2*wavesp0['S']/(2*np.pi)*dw)*np.exp(1j*wavesp0['phase'])
    ETA.attrs['long_name'] = 'wave amplitude'
    ETA.attrs['units'] = 'm^2*s'

    # excitation force
    F_EXC = xr.dot(Hp0, ETA, dims=["wave_direction"])
    F_EXC.attrs['long_name'] = 'wave excitation force'
    F_EXC.attrs['units'] = 'N^2*s or N^2*m^2*s'
    F_EXC = F_EXC.transpose('influenced_dof', 'omega')

    FD = xr.Dataset({'wave_elevation': ETA, 'excitation_force': F_EXC},)
    FD['omega'].attrs['long_name'] = 'frequency'
    FD['omega'].attrs['units'] = '(radians)'

    # time domain
    nfd = 2 * len(waves['omega']) + 1
    f0 = waves['omega'][0] / (2*np.pi)
    time = np.linspace(0, 1/f0, nfd)
    dims_td = ['time', ]
    coords_td = [(dims_td[0], time, {'units': 's'}), ]

    def fd_to_td(FD):
        return np.fft.irfft(FD/(2/nfd), n=nfd)

    f_exc = fd_to_td(F_EXC)
    dims = ['influenced_dof'] + dims_td
    coords = [(dims[0], F_EXC.coords[dims[0]],),] + coords_td
    f_exc = xr.DataArray(f_exc, dims=dims, coords=coords, attrs=F_EXC.attrs)
    f_exc.attrs['units'] = 'N or N*m'
    TD = xr.Dataset({'excitation_force': f_exc},)

    if len(ETA.wave_direction) == 1:
        eta = fd_to_td(np.squeeze(ETA))
        eta = xr.DataArray(eta, dims=dims_td, coords=coords_td, attrs=ETA.attrs)
        eta.attrs['units'] = 'm'
        TD['wave_elevation'] = eta
    # TODO: time domain elevation for multidirectional wave?

    return FD, TD


# BEM Capytaine
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
        # TODO: modify capytaine solver.fill_dataset() to not hardcode g=9.81.
        'g': [g],
    })
    if wave_dirs is None:
        # radiation only problem, no diffraction or excitation
        test_matrix.drop('wave_direction')
    # TODO: Capytaine run keep_immersed_part() automatically (#62)
    wec_im = fb.copy(name=f"{fb.name}_immersed").keep_immersed_part()
    write_info = {key: True for key in write_info}
    return solver.fill_dataset(test_matrix, [wec_im], **write_info)


def from_netcdf(fpath):
    return cpy.io.xarray.merge_complex_values(xr.open_dataset(fpath))


def to_netcdf(fpath, bem_data):
    cpy.io.xarray.separate_complex_values(bem_data).to_netcdf(fpath)


def cpy_impedance(dataset, dissipation=None, stiffness=None):
    # TODO: this is in Capytaine but not in release yet
    omega = dataset.coords['omega']  # Range of frequencies in the dataset
    A = (-omega**2*(dataset['mass'] + dataset['added_mass'])
         + 1j*omega*dataset['radiation_damping']
         + dataset['hydrostatic_stiffness'])
    if dissipation is not None:
        A = A + 1j*omega*dissipation
    if stiffness is not None:
        A = A + stiffness
    return A
