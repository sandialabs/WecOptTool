from mhkit.wave.resource import jonswap_spectrum
from autograd import grad, jacobian, hessian
from scipy.linalg import block_diag
from scipy.optimize import minimize
from cyipopt import minimize_ipopt
import WecOptTool.WaveBot as wb
import matplotlib.pyplot as plt
import WecOptTool.core as wot
import autograd.numpy as np
import pygmsh
import sys
import os

# easier to read numpy output
np.set_printoptions(precision=3)
np.set_printoptions(linewidth=160)
np.set_printoptions(threshold=sys.maxsize)

# set working dir to location of this file
os.chdir(os.path.split(os.path.realpath(__file__))[0])

#%% This block is in the example script

name = 'example_WaveBot_oneDof_nf18'

# frequency vector is f0*np.arange(1,num_freq+1)
f0 = 0.05
num_freq = 18

# kinematics is dict containing whatever you like (your functions will use it)
kinematics = dict()
kinematics['Jac'] = np.zeros((1,1), float)
kinematics['Jac'][0,0] = 1
# kinematics['Jac'][2,1] = 1
# kinematics['Jac'][4,2] = 1

wec = wot.WEC(mesh=wb.hull(), 
              f0=f0,
              num_freq=num_freq,
              modes=np.ones(6),             # TODO: not sure this is doing anything
              kinematics=kinematics,
              run_bem=False,
              cog=None,
              name=name,
              wrk_dir=name,
              verbose=True,
              )

wec.fb.keep_only_dofs(['Heave'])

wec.fb.mass = wec.fb.mass.sel(influenced_dof=['Heave'],radiating_dof=['Heave']) # TODO

#%%

# will run BEM first time, then load results from saved file
try:
    wec.hydro = wec.__read_bem__()
except Exception:
    wec.run_bem(post_proc=False)
    # wec.hydro = wec.hydro.sel(influenced_dof='Heave',radiating_dof='Heave')
    wec.__post_proc_bem__()
    wec.write_bem()
    
# wec.hydro = wec.hydro.sel(influenced_dof='Heave',radiating_dof='Heave')

#%%

num_freq = wec.hydro.omega.size
num_modes = wec.hydro.influenced_dof.size
omega = wec.hydro.omega.values

def obj_fun(x):
    
    x_wec, x_pto, nf, _ = wec.decompose_decision_var(x*(1/scale))
    
    # transform to body coordinate system
    Jac = wec.kinematics['Jac']
    x_vel = Jac.transpose() @ np.reshape(wec.Dphi @ x_wec, (-1, 2*nf))
    
    # calculate total power (negative is absorbed)
    P = scale_cost * x_vel.flatten() @ x_pto / (2*wec.params['f0'])
    # print(type(P))
    return P


def F_pto_fun(x):
    """
    returns time vector of forces/moments at collocation points proper 
    coordinate system
    """
    
    x_wec, x_pto, nf, nm = wec.decompose_decision_var(x)
    nc=2*nf+1
    
    # PTO force, each row is a mode, each column is a Fourier coefficient
    # X_pto = np.reshape(x_pto, (-1,2*nf))
    
    # # convert to complex amplitude (each row is a mode, each column is a Fourier coefficient)
    # X_pto_hat = X_pto[:,::2] - X_pto[:,1::2]*1j
    
    # # time domain force for each PTO, each row is a mode, each column is a collocation point in time
    # fp3 = np.fft.irfft(np.concatenate((np.zeros((1,1)), X_pto_hat), axis=1), nc) * nc/2;
    fp3 = x_pto @ wec.Phi[1::,::]
    # transform to body coordinate system
    Jac = wec.kinematics['Jac']
    f_pto = np.squeeze(Jac) * fp3
    return f_pto


def F_ext_fun(x):
    """
    External forcing due, e.g., to mooring.

    Parameters
    ----------
    x : np.ndarray
        DESCRIPTION.

    Returns
    -------
    f_ext : np.ndarray
        DESCRIPTION.

    """
    f_ext = 0
    return f_ext

#%% Generate wave spectrum and excitation

S = jonswap_spectrum(omega/2/np.pi, 
                      Tp=1/(wec.hydro.omega/2/np.pi).values[11], 
                      Hs=0.2, 
                      gamma=7)

# convert spectrum to a regular wave
df = wec.hydro.omega[1] - wec.hydro.omega[0]
idx = 3
a = 0.25
sa = a**2 / (2*df)*2*np.pi
S = S*0
S.iloc[idx] = sa

ds_exc = wec.get_waveExcitation(S)
print(ds_exc)
ds_exc.eta.plot()

#%% Solve for dynamics

f_exc = ds_exc['f_exc'].values

my_resid_fun = lambda x: wec.dynamic_residual(x*(1/scale), f_exc, F_pto_fun, F_ext_fun)

eq_cons = {'type': 'eq',
            'fun': my_resid_fun,
            'jac':jacobian(my_resid_fun)
            }

# scale
scale_wec = 0.000003
scale_pto = 0.00001
scale_cost = 1e-10
scale_wec = scale_wec * np.ones(num_modes*(2*num_freq+1))
scale_pto = scale_pto * np.ones(1*(2*num_freq))
scale = np.concatenate([scale_wec, scale_pto])

# minimize
x0 = np.random.rand(num_modes*(2*num_freq+1) + 1*(2*num_freq))

res = minimize(fun=obj_fun,
               x0=x0,
               jac=jacobian(obj_fun),
               constraints=(eq_cons),
               options={'disp':True,
                        'iprint':5
                        },
               tol=1e-5, # 1e-14,
               method='SLSQP'
               )

# unscale
res.x *= 1/scale
res.fun *= 1/scale_cost

# print
print(np.max(np.abs(res.x[:2*num_freq+1])))
print(np.max(np.abs(res.x[2*num_freq+1:])))
print(res.fun)

# plot to confirm optimization solution
fig, ax = plt.subplots()
ax.plot(x0, label='$x_0$', marker='.')
ax.plot(res.x, label='$x$', marker='o')
ax.legend()

print(np.max(np.abs(res.x)))
print(res.fun)


X_hat, x = wec.post_process(res.x)

#%% Spectral plot

fig, ax = plt.subplots(nrows=2,
                        sharex=True)

ax[0].stem(ds_exc.F_EXC.omega, 
           np.abs(ds_exc.F_EXC.squeeze())/np.max(np.abs(ds_exc.F_EXC.squeeze())),
           markerfmt='s',
           label='Excitation')
# markerline.set_markerfacecolor('none')
ax[1].stem(ds_exc.F_EXC.omega, 
           np.angle(ds_exc.F_EXC.squeeze()),
           markerfmt='s',
           label='Excitation')


ax[0].stem(np.hstack((0,wec.hydro.omega)), 
          np.abs(X_hat.flatten())/np.max(np.abs(X_hat.flatten())),
         markerfmt='o',
        label='Response')
# markerline.set_markerfacecolor('none')
ax[1].stem(np.hstack((0,wec.hydro.omega)), 
           np.angle(X_hat.flatten()),
         markerfmt='o',
        label='Response')

ax[0].legend()

ax[0].set_ylabel('Normalized magnitude [ ]')
ax[1].set_xlabel('Frequency [rad/s]')
ax[1].set_ylabel('Angle [rad]')

#%% Time history plot

fig, ax = plt.subplots(nrows=3,
                        sharex=True)
ds_exc.f_exc.plot(ax=ax[0],marker='.')
ax[0].set_ylabel('Excitation force [N]')

ds_exc.eta.plot(ax=ax[1],
                marker='.',
                label=ds_exc.eta.attrs['long_name'])
ax[1].set_ylabel('$\eta$ [m]')
# ax[1].legend()

ax[2].plot(ds_exc.time,
            x.flatten(),
            marker='.',
            )
ax[2].set_ylabel('Position [m]')

ax[-1].set_xlabel('Time [s]')

for axi in ax:
    axi.set_title('')
    axi.label_outer()

def test_example():
    pass

