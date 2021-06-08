import WecOptTool as wot
import WecOptTool.WaveBot as wb
import matplotlib.pyplot as plt
import sys
import pygmsh
from scipy.linalg import block_diag
import os
import autograd.numpy as np
from autograd import grad
from autograd import jacobian
from autograd import hessian
from cyipopt import minimize_ipopt

# easier to read numpy output
np.set_printoptions(precision=3)
np.set_printoptions(linewidth=160)
np.set_printoptions(threshold=sys.maxsize)

# set working dir to location of this file
os.chdir(os.path.split(os.path.realpath(__file__))[0])

#%% This block is in the example script

name = 'example_WaveBot_nf2'

# frequency vector is f0*np.arange(1,num_freq+1)
f0 = 0.05
num_freq = 2

# kinematics is dict containing whatever you like (your functions will use it)
kinematics = dict()
kinematics['Jac'] = np.zeros((6,3), float)
kinematics['Jac'][0,0] = 1
kinematics['Jac'][2,1] = 1
kinematics['Jac'][4,2] = 1

wec = wot.WEC(geom=wb.hull(), 
              f0=f0,
              num_freq=num_freq,
              modes=np.ones(6),
              kinematics=kinematics,
              run_bem=False,
              cog=None,
              meshArgs=None,
              name=name,
              wrk_dir=name,
              verbose=True,
              )

# will run BEM first time, then load results from saved file
try:
    wec.hydro = wec.__read_bem__()
except Exception:
    wec.run_bem()


#%%

num_freq = wec.hydro.omega.size
num_modes = wec.hydro.influenced_dof.size
omega = wec.hydro.omega.values

tmp = np.zeros(omega.size*2-1)
tmp[::2] = omega

# b = np.concatenate((np.zeros((tmp.size+1,1)), np.diag(tmp,1) - np.diag(tmp,-1)), axis=1)
b = np.concatenate((np.zeros((tmp.size+1,1)), np.eye(2*num_freq) ), axis=1)
Dphi = block_diag(*[b]*num_modes)


Zi_block = wec.Z_block
Gi_block = wec.G_block

F_exc = np.zeros((num_modes, 2*num_freq))
F_exc_hat = F_exc[:,0::2] - 1j*F_exc[:,1::2]
F_exc_hat_with_mean = np.concatenate((np.zeros((num_modes,1)), F_exc_hat), axis=1)
f_exc = np.fft.irfft(F_exc_hat_with_mean, 2*num_freq) * num_freq

def obj_fun(x):
    
    x_wec, x_pto, nf, _ = wec.decompose_decision_var(x)
    
    # transform to body coordinate system
    Jac = wec.kinematics['Jac']
    x_vel = Jac.transpose() @ np.reshape(Dphi @ x_wec, (-1, 2*nf))
    
    # calculate total power (negative is absorbed)
    P = x_vel.flatten() @ x_pto / (2*wec.params['f0'])
    
    return P


def F_pto_fun(x):
    """
    returns time vector of forces/moments at collocation points proper 
    coordinate system
    """
    x_wec, x_pto, nf, nm = wec.decompose_decision_var(x)
    nc=2*nf+2
    
    # PTO force, each row is a mode, each column is a Fourier coefficient
    X_pto = np.reshape(x_pto, (-1,2*nf))
    
    # convert to complex amplitude (each row is a mode, each column is a Fourier coefficient)
    X_pto_hat = X_pto[:,::2] - X_pto[:,1::2]*1j
    
    # time domain force for each PTO, each row is a mode, each column is a collocation point in time
    fp3 = np.fft.irfft(np.concatenate((np.zeros((3,1)), X_pto_hat), axis=1), nc) * nc/2;
    
    # transform to body coordinate system
    Jac = wec.kinematics['Jac']
    f_pto = Jac @ fp3
    return f_pto


def F_ext_fun(x):
    f_ext = 0
    return f_ext


def resid_fun(x):
    # WEC position
    x_wec, _, nf, nm = wec.decompose_decision_var(x)
    
    # WEC velocity (each row is a mode, each column is a Fourier component)
    X = np.reshape(x_wec, (nm, -1))
    
    # complex velocity
    # X_hat = X[:,::2] - X[:,1::2]*1j
    X_hat = np.concatenate((np.reshape(X[:,0],(-1,1)), X[:,1::2] - X[:,2::2]*1j ), axis=1)
    # time domain force for each PTO, each row is a mode, each column is a collocation point in time
    tmp = Gi_block.toarray() # TODO
    # tmp = np.eye(Gi_block.shape[0])
    Fi = np.reshape(tmp @ X_hat.flatten(), (nm, -1))
    fi = np.fft.irfft(Fi, 2*nf+2) * nf
    
    residual = F_pto_fun(x) + F_ext_fun(x) - fi
    return residual.flatten()


#%%

x0 = np.random.rand(num_modes*(2*num_freq+1) + 3*(2*num_freq))

eq_cons = {'type': 'eq',
            'fun': resid_fun,
            'jac': jacobian(resid_fun),
            }
           

res = minimize_ipopt(fun=obj_fun,
                x0=x0,
                jac=grad(obj_fun),
                constraints=(eq_cons),
                options={'print_level': 5},
                tol=1e-14
                )

# plt.close('all')
fig, ax = plt.subplots()
ax.plot(x0, label='$x_0$', marker='.')
ax.plot(res.x, label='$x$', marker='o')
ax.legend()
