import capytaine as cpy
import autograd.numpy as np
import numpy as anp
from autograd.builtins import isinstance, tuple
# from autograd import grad
# from autograd import hessian
# from autograd import jacobian
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import sys
import os
from meshmagick import mesh as mm
from meshmagick import hydrostatics as hs
import meshio
import tempfile
import logging
import datetime
from scipy.linalg import block_diag
from scipy.optimize import basinhopping
from capytaine.io.xarray import separate_complex_values
from capytaine.io.xarray import merge_complex_values
import glob
import json
import control
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy import optimize
from  mhkit import wave
from scipy import fft
from scipy import sparse
import types
import warnings


np.set_printoptions(precision=3)
np.set_printoptions(linewidth=160)
np.set_printoptions(threshold=sys.maxsize)

DataSet_type = xr.core.dataset.Dataset


def calc_impedance(hydro:DataSet_type, damp_frac:float=0.05, make_sym:bool=True):
    """
    Calculate intrinsic impedance (see, e.g., Falnes).
    
    @book{falnes2002ocean,
          title={Ocean Waves and Oscillating Systems: Linear Interactions Including Wave-Energy Extraction},
          author={Falnes, J.},
          isbn={9781139431934},
          url={https://books.google.com/books?id=bl1FyQjCklgC},
          year={2002},
          publisher={Cambridge University Press}
    }

    Parameters
    ----------
    hydro : xr.core.dataset.Dataset
        Hydro structure returned from Capytaine with mass matrix and hydrostatics
    damp_frac : float, optional
        Frictional damping. The default is 0.05.
    make_sym : bool, optional
        Make symmetric. The default it True.

    Returns
    -------
    Zi : xr.core.dataset.DataArray
        Intrinsic impedance.

    """
    
    assert isinstance(hydro, DataSet_type), 'hydro must be xr.DataSet, received {:}'.format(type(hydro))
    assert isinstance(damp_frac, float), 'damp_frac must be float, received {:}'.format(type(damp_frac))
    
    friction_damping = np.eye(hydro.radiation_damping[0,:,:].shape[0])*damp_frac
        
    Zi = hydro.radiation_damping + friction_damping + \
            1j * (hydro.omega * (hydro.mass + hydro.added_mass) \
                  - hydro.hydrostatic_stiffness / hydro.omega )
                
    if make_sym:
        Zi.values = (Zi.values + Zi.values.transpose(0,2,1))/2

    return Zi



class WEC:
    
    def __init__(self, mesh, f0, num_freq, modes, run_bem=True, 
                 cog=None, kinematics=None,
                 name=None, wrk_dir=None, verbose=True):
        
        self.params = dict()      
        
        if not isinstance(f0, (float,int)):
            raise TypeError('f0 must be of type float or int')

        self.params['f0'] = f0


        if not isinstance(num_freq, int):
            raise TypeError('num_freq must be of type int')
        self.params['num_freq'] = num_freq

        freq = np.arange(1, num_freq+1)*f0
        
        assert len(modes) == 6                      # TODO times number of bodies
        assert isinstance(modes, (np.ndarray, list))
        modes = np.asarray(modes)
        # assert np.all(np.isin(modes,[0,1]))       # TODO
        self.params['modes'] = modes.tolist()

        if cog is not None:
            raise NotImplementedError
        else:
            cog = [0,0,0]

        self.kinematics = kinematics

        if name is None:
            date_str = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
            name = 'wec_{:}'.format(date_str)

        if not isinstance(name, str):
            raise TypeError('Name must be a string')
        self.params['name'] = name
        
        if wrk_dir is None:
            wrk_dir = os.path.join('.',self.params['name'])
        elif wrk_dir is False:
            wrk_dir = tempfile.mkdtemp()

        if not isinstance(wrk_dir, str):
            raise TypeError('wrk_dir must be of type str')
            
        self.params['wrk_dir'] = wrk_dir
        os.makedirs(self.params['wrk_dir'], exist_ok=True)
        
        if verbose is not True:
            raise NotImplementedError
        else:
            log_file = os.path.join(self.params['wrk_dir'], 'log.txt')
            logging.basicConfig(filename=log_file)
        

        self.files = dict()
        self.files['geo'] = os.path.join(self.params['wrk_dir'],
                                         self.params['name'] + '.geo')
        self.files['stl'] = os.path.join(self.params['wrk_dir'],
                                         self.params['name'] + '.stl')

        if isinstance(mesh,str):
            warnings.warn("Using existing STL file")
        elif isinstance(mesh,meshio._mesh.Mesh):
            mesh.write(self.files['stl'])
        else:
            raise TypeError('Must be an STL file path or meshio mesh object')
        

        self.fb = cpy.FloatingBody.from_file(self.files['stl'],
                                             name=self.params['name'])

        self.fb.keep_immersed_part()
        self.fb.add_all_rigid_body_dofs()
        tmp_mesh = mm.Mesh(self.fb.mesh.vertices, 
                           self.fb.mesh.faces)
        
        self.hsa = hs.Hydrostatics(tmp_mesh,
                                   cog=cog,
                                   mass=None,       # TODO
                                   rho_water=1e3,   # TODO
                                   verbose=True)
        
        hsd = self.hsa.hs_data
        m = hsd['disp_mass']
        I = np.array([[hsd['Ixx'], -1*hsd['Ixy'], -1*hsd['Ixz']],
                      [-1*hsd['Ixy'], hsd['Iyy'], -1*hsd['Iyz']],
                      [-1*hsd['Ixz'], -1*hsd['Iyz'], hsd['Izz']]])
        M = block_diag(m, m, m, I) # TODO - scale inertias if mass is specified by user
        
        self.fb.mass = self.fb.add_dofs_labels_to_matrix(M)
        kHS = block_diag(0,0,hsd['stiffness_matrix'],0)
        self.fb.hydrostatic_stiffness = self.fb.add_dofs_labels_to_matrix(kHS)
        
        
        assert isinstance(run_bem, (bool,str))
        if run_bem == True:
            self.run_bem(freq)
            self.write_bem()
        elif isinstance(run_bem, str):
            self.hydro = self.__read_bem__(run_bem)
        
        self.files['json'] = os.path.join(self.params['wrk_dir'], 
                                          self.params['name'] + '.json')
        with open(self.files['json'], "w") as outfile:  
            json.dump(self.params, outfile, sort_keys=True, indent=4)               
        
        
        
    def run_bem(self, freq=None, wave_dirs=None, post_proc=True):
        
        if freq is None:
            freq = np.arange(1, self.params['num_freq']+1)*self.params['f0']
        assert isinstance(freq, np.ndarray)
        
        if wave_dirs is None:
            wave_dirs = [0]
        else:
            raise NotImplementedError

        assert isinstance(wave_dirs, list)      # TODO check list contains floats
        
        
        solver = cpy.BEMSolver()                # TODO: enable setting this
        test_matrix = xr.Dataset(coords={
            'rho': 1e3,                         # TODO: enable setting this
            'water_depth': [np.infty],          # TODO: enable setting this
            'omega': freq*2*np.pi,
            'wave_direction': wave_dirs,
            'radiating_dof': list(self.fb.dofs.keys()),
            })
        
        data = solver.fill_dataset(test_matrix, [self.fb],
                                   hydrostatics=True,
                                   mesh=True,
                                   wavelength=True,
                                   wavenumber=True)

        data['freq'] = data.omega / (2 * np.pi)
        data['freq'].attrs['units'] = 'Hz'
        data = data.set_coords('freq')

        data['T'] = 1 / data.freq
        data['T'].attrs['units'] = 's'
        data = data.set_coords('T')
        
        self.hydro = data
        self.hydro['displaced_volume'] = self.hsa.hs_data['disp_volume'] # TODO - redundant probably remove
        
        if True:                                                            # TODO
            # Infinite frequency added mass
            inf_test_matrix = xr.Dataset(coords={
                                        'rho': 1e3,                         # TODO
                                        'water_depth': [np.infty],
                                        'omega': [np.infty],
                                        'radiating_dof': list(self.fb.dofs.keys()),
                                        })
            inf_data = solver.fill_dataset(inf_test_matrix, [self.fb])
            self.inf_data = inf_data
            self.hydro['Ainf'] = inf_data.added_mass[0,:,:]
            
        if post_proc:
            self.__post_proc_bem__()
            
    def __post_proc_bem__(self):
           
        self.hydro['Zi'] = calc_impedance(self.hydro)
        self.hydro['rao'] = cpy.post_pro.rao(self.hydro)
        self.Zi_block = self.__make_Zi_block__()
        self.Gi_block = self.__make_Gi_block__()
        self.Dphi = self.__make_Dphi__()
        self.Phi = self.__make_Phi__()
        self.num_scale = self.__get_num_scale__()

    def write_bem(self, fpath:str=None):
        """
        Write the BEM solution to netCDF file

        Parameters
        ----------
        fpath : str, optional
            DESCRIPTION. The default is the WEC's wrk_dir.
        """
        if fpath is None:
            fpath = os.path.join(self.params['wrk_dir'], self.params['name'] + '.nc')
        assert isinstance(fpath, str), 'fpath must be type(str), received {:}'.format(type(fpath))
        
        separate_complex_values(self.hydro).to_netcdf(fpath)
        
        
    @staticmethod
    def from_file(name: str, fpath: str=None) -> 'WEC':
        """
        Generate a WEC object directly from a directory of previous results

        Parameters
        ----------
        name : str
        fpath : str, optional
            The default is '.'

        Returns
        -------
        my_wec : WecOptTool.core.Wec

        """
        
        if fpath is None:
            fpath = '.'
        assert isinstance(fpath, str)
        
        assert isinstance(name, str)

        param_file = glob.glob(os.path.join(fpath,name+'*.json'))[0]

        with open(param_file) as f:
            params = json.load(f)

        bem_file = glob.glob(os.path.join(fpath,name+'*.nc'))
        if not bem_file:
            pass
        else:
            params['run_bem'] = bem_file[0]   

        params['wrk_dir'] = fpath
        params['mesh'] = glob.glob(os.path.join(fpath,name+'*.stl'))[0]
        my_wec = WEC(**params)
        return my_wec
        
        
    def __read_bem__(self, fpath=None) -> xr.Dataset:
        if fpath is None: #TODO - should be able to do this as "from_dir" and build entire object
            fpath = os.path.join(self.params['wrk_dir'], self.params['name'] + '.nc')
        assert isinstance(fpath, str), 'fpath must be type(str), received {:}'.format(type(fpath))
        assert os.path.exists(fpath)
        hydro = merge_complex_values(xr.open_dataset(fpath)) # TODO - this could result in a mismatch between mesh and hydro
        self.hydro = hydro
        self.__post_proc_bem__()
        return hydro
        
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
        
        if axs is None:
            fig, axs = plt.subplots()
        
        if diag_only:
            for idx, dof in enumerate(self.hydro.radiating_dof.data.tolist()):
                Zi_frd = control.frd(self.hydro.Zi[:,idx,idx], self.hydro.omega)
                control.bode_plot(Zi_frd, self.hydro.omega,
                                  dB=True, 
                                  Hz=True,
                                  marker='.', 
                                  label=dof)
            plt.legend()
        else:
            raise NotImplementedError() # TODO
            
            
    def get_pow_ub(self, S, dof:int=2) -> xr.DataArray:
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
        
        assert isinstance(dof, int)
        
        
        Fexc = self.get_waveExcitation(S)['F_EXC'].isel(influenced_dof=dof).squeeze()
        # print(self.hydro)
        Zi = self.hydro['Zi'].isel(dict(influenced_dof=dof,
                                        radiating_dof=dof)).squeeze()
        
        P_ub = 1/8 * np.abs(Fexc)**2 / np.real(Zi)
        
        return P_ub
    
    def get_pow(self, S, controller='P', dof=2):
        
        Fexc = self.get_waveExcitation(S)['Fexc'].isel(influenced_dof=dof).squeeze()
        Zi = self.hydro['Zi'].isel(dict(influenced_dof=dof,
                                        radiating_dof=dof)).squeeze()
        
        if controller == 'P':
            def dampingPower(x):
                P = -0.5 * x[0] * np.sum(np.abs(Fexc / (Zi + x[0]))**2)
                fval = P.values[()]
                return fval
            
            x0 = [1e-10]
            bounds = Bounds(lb=0, ub=np.inf)
            
            res = minimize(dampingPower, 
                           x0, 
                           method='L-BFGS-B',
                           bounds=bounds,
                           options={
                               'disp': False,
                               # 'maxiter': 10,
                               })
       
        return res.fun
            
    def get_waveExcitation(self, S, seed:int=1) -> xr.Dataset:
        """
        Get wave excitation forcing

        Parameters
        ----------
        S : pd.core.frame.DataFrame, xr.Dataset, xr.DataArray
            Wave spectrum.
        seed : int, optional
            Random number generator seed. The default is 1.

        Returns
        -------
        DS : xr.Dataset
            Excitation spectrum.

        """
        
        assert isinstance(S, (pd.core.frame.DataFrame, xr.Dataset, xr.DataArray)), \
            'S must be a pandas DataFrame or Xarray DataArray'
        
        if isinstance(S, pd.core.frame.DataFrame):
            varn = list(S.keys())[0]
            S = S.to_xarray().rename({'index':'freq', varn:'S'})['S']
            S['freq'] = S.freq*(2*np.pi)
            S = S/(2*np.pi)
            S = S.rename({'freq':'omega'})
            DS = S.to_dataset()
            DS.S.attrs['units'] = 'm^2s/rad'
            DS.attrs['name'] = varn 
        elif isinstance(S, xr.DataArray):
            DS = S.to_dataset()
        else:
            DS = S
            
        if DS.omega[0] != 0:
            tmp1 = DS.isel(omega=0)
            tmp1['omega'] = 0
            tmp1['S'] = 0
            DS = xr.concat([tmp1, DS], dim='omega')
            
        np.random.seed(seed)
        DS = DS.assign(phi = 2*np.pi*np.random.rand(len(DS['S']))*DS.S**0)
        DS['dw'] = np.gradient(DS.omega)*DS.S**0
        
        DS['ETA'] = np.sqrt(2*DS['S']*DS['dw'])*np.exp(1j*DS.phi)
        DS['ETA'].attrs['long_name'] = 'Complex wave amplitude'
        
        H = self.hydro.excitation_force
        DS['F_EXC'] = H * DS['ETA']
        DS['F_EXC'][0] = 0+1j*0
        DS['freq'][0] = 0
        DS['F_EXC'].attrs['long_name'] = 'Complex excitation'
        
        w0 = self.hydro.omega[0]
        f0 = w0/2/np.pi
        N = 2*(len(DS.omega)-1)+1
        T = 1/f0
        t = np.linspace(0,T,N)
        dt = t[1] - t[0]
        
        def fd_to_td(FD):
            return anp.fft.irfft(FD/(2/N), n=N)
            
        da_eta = xr.apply_ufunc(fd_to_td,
                       DS.ETA,
                       vectorize=True,
                       input_core_dims=[["omega"]],
                       output_core_dims=[["time"]],
                       exclude_dims=set(("omega",)),
                       )
        da_eta = da_eta.assign_coords(dict(time=t))
        da_eta.name = 'eta'
        da_eta.attrs['long_name'] = 'Free surface elevation'
        da_eta.attrs['units'] = 'm'
        
        da_f_exc = xr.apply_ufunc(fd_to_td,
                       DS.F_EXC,
                       vectorize=True,
                       input_core_dims=[["omega"]],
                       output_core_dims=[["time"]],
                       exclude_dims=set(("omega",)),
                       )
        da_f_exc = da_f_exc.assign_coords(dict(time=t))
        
        da_f_exc.name = 'f_exc'
        da_f_exc.attrs['long_name'] = 'Excitation force'
        # da_f_exc.attrs['units'] = 'm' # TODO

        # assert np.array_equal(H.omega.values, DS.omega.values), \
        #     'S.omega does not match BEM omega'
        
        out_ds = xr.merge([DS, da_eta, da_f_exc])
        
        out_ds['time'].attrs['long_name'] = 'Time'
        out_ds['time'].attrs['units'] = 's'
        
        out_ds['omega'].attrs['long_name'] = 'Frequency'
        out_ds['omega'].attrs['units'] = 'rad/s'

        return out_ds
    
    
    def __make_Zi_block__(self) -> np.ndarray:
        """
        Makes a block matrix of the MIMO impdeance

        Returns
        -------
        Zi_block : np.ndarray.
        """
    
        omega = self.hydro.omega.values
        num_freq = len(omega)
        
        Zi = self.hydro.Zi.values
        num_modes = Zi.shape[2]
        elem = [[0] * num_modes for i in range(num_modes)]

        for idx_mode in range(num_modes):
            for jdx_mode in range(num_modes):
                elem[idx_mode][jdx_mode] = np.diag(Zi[:,idx_mode,jdx_mode])
                
        Zi_block = sparse.dia_matrix(np.block(elem))

        return Zi_block
    
    def __make_Gi_block__(self) -> np.ndarray:
        """
        Makes a block matrix of the MIMO impdeance + position

        Returns
        -------
        Gi_block : np.ndarray.
        """
    
        omega = self.hydro.omega.values
        num_freq = len(omega)
        
        Zi = self.hydro.Zi.values
        num_modes = Zi.shape[2]
        elem = [[0] * num_modes for i in range(num_modes)]

        for idx_mode in range(num_modes):
            for jdx_mode in range(num_modes):
                elem[idx_mode][jdx_mode] = np.diag(np.concatenate(([self.hydro.hydrostatic_stiffness.values[idx_mode,jdx_mode]], 1j * omega*Zi[:,idx_mode,jdx_mode])))
                
        Gi_block = sparse.dia_matrix(np.block(elem))

        return Gi_block
    
    def gen_initial_guess(self, x_extra=None) -> np.ndarray:
        
        if isinstance(x_extra,int):
            x_extra = np.zeros(x_extra)
        assert isinstance(x_extra, np.ndarray)
        assert x_extra.ndim == 1
        
        num_modes = self.hydro.radiating_dof.size
        num_freq = self.hydro.omega.size
        
        x_pos = np.zeros(num_modes*(2*num_freq+1))
        x = np.concatenate([x_pos, x_extra])
        
        return x
    
    def __make_Phi__(self) -> np.array:
        """
        Phi is a matrix for the transformation from Fourier series
        coefficients to time series
        """
        w = self.hydro.omega.values
        dw = np.mean(np.diff(w))
        nf = self.hydro.omega.size
        T = np.linspace(0, 2*np.pi/dw, 2*nf+1, endpoint=False)
        MM = np.reshape(w,(-1,1)) @ np.reshape(T,(1,-1))
        Phi = np.ones((2*nf+1, 2*nf+1))
        Phi[1::2,::] = np.cos(MM)
        Phi[2::2,::] = np.sin(MM)
        return Phi
    
    def __make_Dphi__(self) -> np.array:
        """
        Dphi is a matrix used to transform between position and velocity
        """
        num_modes = len(self.hydro.radiating_dof)
        omega = np.array(self.hydro.omega)
        Dphi = np.diag((np.array([[1],[0]]) * omega).flatten('F')[:-1],1)   # TODO - numpy is row wise
        Dphi = Dphi - Dphi.transpose()
        Dphi = np.concatenate((np.zeros((2*len(omega),1)),Dphi ), axis=1)
        # Dphi = np.concatenate((np.zeros((2*len(omega),1)),np.eye(2*len(omega)) ), axis=1)
        # Dphi = block_diag(np.zeros((1,1)), Dphi)                          # TODO - not sure about this
        Dphi = block_diag(*([Dphi]*num_modes))
        
        return Dphi


    def solve(self, S, obj_fun, F_pto_fun, F_ext_fun, x0) -> optimize.optimize.OptimizeResult:
        """
        Solves the numerical optimization problem, typically to maximize power
        as defined by obj_fun

        Parameters
        ----------
        S : TYPE
            Wave spectrum generated in MHKit.
        obj_fun : function
            Objective function to minimize (e.g., power, where negative power is absorbed by WEC).
        F_pto_fun : function
            Function returning PTO force at collocation points in body frame based on decision variable.
        F_ext_fun : function
            Function returning other sum of forces at collocation points in body frame based on decision variable.
        x0 : np.ndarray
            Initial guess for decision variable.

        Returns
        -------
        x_hat : scipy.optimize.optimize.OptimizeResult
            Solution.
        """
        num_freq = self.hydro.omega.size
        num_modes = self.hydro.influenced_dof.size
        
        Dphi = self.Dphi
        Zi_block = self.Zi_block
        # Z_block = sparse.dia_matrix( np.identity(num_freq*num_modes))
        # f_exc = np.asarray(0)                                               # TODO
        F_exc = np.zeros((num_modes, 2*num_freq))
        F_exc_hat = F_exc[:,0::2] - 1j*F_exc[:,1::2]
        F_exc_hat_with_mean = np.concatenate((np.zeros((num_modes,1)), F_exc_hat), axis=1)
        f_exc = np.fft.irfft(F_exc_hat_with_mean, 2*num_freq) * num_freq
        
        # power maximization
        my_obj_fun = lambda x : obj_fun(x, self)
        
        # system dynamics through equality constraint
        resid_fun = lambda x : dynamic_residual(x,              # TODO - can set this up on instantiation once
                                                _block,
                                                self,
                                                f_exc,
                                                F_pto_fun,
                                                F_ext_fun,
                                                Dphi)
        
        # resid_cnstrnt = optimize.NonlinearConstraint(fun=resid_fun, 
        #                                              lb=0, 
        #                                              ub=0,
        #                                              jac=jacobian(resid_fun),
        #                                              )
        
        # eq_constr = {"fun": lambda x : dynamic_residual(x, Z_block, self, f_exc, F_pto_fun, F_ext_fun, Dphi),
        #              "type": "eq"}
        
        
        # x_hat = optimize.minimize(fun=my_obj_fun,
        #                           x0 = x0,
        #                            jac=jacobian(my_obj_fun),
        #                            # hess=hessian(my_obj_fun),
        #                           method = 'trust-constr',
        #                           constraints = resid_cnstrnt,
        #                           options = {
        #                               'verbose':3
        #                                      # 'ftol': 1e-10, 
        #                                      # 'eps': 1e-16, 
        #                                      # 'maxiter': 1e4, 
        #                                      # 'disp': True, 
        #                                      # 'iprint':2,
        #                                      # 'finite_diff_rel_step': 0.5,
        #                                      },
        #                           )
        
        eq_cons = {'type': 'eq',
           'fun': resid_fun,
           # 'jac': jacobian(resid_fun),
           }
        
        x_hat = optimize.minimize(fun=my_obj_fun,
                        x0=x0,
                        # jac=grad(my_obj_fun),
                        method='SLSQP',
                        constraints = [eq_cons], 
                        options = {'ftol': 1e-8, 
                                   'iprint':2,
                                   'disp': True},
                        )
        
        return x_hat
    
    
    def __get_num_scale__(self):
        return 1/np.linalg.norm(self.Gi_block.toarray())
    
    def dynamic_residual(self,
                         x: np.ndarray,
                         f_exc: np.ndarray,                     # TODO: make this a property of WEC
                         f_pto_fun: types.FunctionType,
                         f_ext_fun: types.FunctionType) -> np.ndarray:
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
        f_pto_fun : types.FunctionType
            Function that acceps decision variable and WEC, returns PTO forcing at 
            collocation points in body coordinate system
        f_ext_fun : types.FunctionType
            Function that acceps decision variable and WEC, returns other forcing at 
            collocation points in body coordinate system

        Returns
        -------
        np.ndarray
            Residuals at collocation points
    
        """
        
        assert isinstance(x, np.ndarray)
        assert isinstance(f_exc, np.ndarray)
        assert isinstance(f_pto_fun, types.FunctionType)
        assert isinstance(f_ext_fun, types.FunctionType)
             
    
        # WEC position
        x_wec, _, nf, nm = self.decompose_decision_var(x)
        
        # WEC velocity (each row is a mode, each column is a Fourier component)
        X = np.reshape(x_wec, (nm, -1))
        
        # complex velocity with position at beginning
        X_hat = np.concatenate((np.reshape(X[:,0],(-1,1)), X[:,1::2] - X[:,2::2]*1j ), axis=1)
        
        Gi_block_scaled = self.num_scale * self.Gi_block.toarray() # TODO: do this only once
    
        Fi = np.squeeze(np.reshape(Gi_block_scaled @ X_hat.flatten(), (nm, -1)))
        Fi_fs_tmp_0 = np.real(Fi[0])
        Fi_fs_tmp_1 = np.vstack([np.real(Fi[1::]), -np.imag(Fi[1::])]).ravel('F')
        Fi_fs = np.hstack((np.array(Fi_fs_tmp_0), Fi_fs_tmp_1))
        fi = Fi_fs @ self.Phi
        
        residual = f_exc + f_pto_fun(x) + f_ext_fun(x) - fi

        return residual.flatten()
        
    
    def decompose_decision_var(self, x: np.array) -> np.array:
        """
        Take slices of decision variable to decompose into:
        
            1: elements pertaining to WEC dynamics
            2: elements pertaining to user's states (e.g., PTO, controller, etc.)
    
    
        Parameters
        ----------
        x : np.array
            DESCRIPTION.
    
        Returns
        -------
        x_wec : np.array
            Position state of WEC.
        x_user : np.array
            User states.
        num_freq : int
            Number of frequency components.
        num_modes : int
            Number of body modes.
    
        """
        
        num_modes = self.hydro.influenced_dof.size
        num_freq = self.hydro.omega.size
    
        # assert x.ndim == 1
        assert len(x) >= num_modes*(2*num_freq+1)
        
        x_wec = x[:num_modes*(2*num_freq+1)]
        x_user = x[num_modes*(2*num_freq+1):]
        
        return x_wec, x_user, num_freq, num_modes
    

    def post_process(self, x):
        """
        Transform the results from optimization solution to form that user can work 
        with directly
        """
        x_wec_scaled, _, nf, nm = self.decompose_decision_var(x)
        x_wec = x_wec_scaled * self.num_scale
        
        X = np.reshape(x_wec, (nm, -1))
        
        # complex position
        X_hat = X_hat = np.concatenate((np.reshape(X[:,0],(-1,1)), X[:,1::2] - X[:,2::2]*1j), 
                                       axis=1)
        
        # time-domain position
        x = X @ self.Phi
        
        return X_hat, x
    

