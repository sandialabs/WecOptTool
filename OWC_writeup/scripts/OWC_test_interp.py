from datetime import datetime

import autograd.numpy as np
import capytaine as cpy
import matplotlib.pyplot as plt
from scipy.optimize import brute
import pickle

import wecopttool as wot

import autograd.numpy as np

savedtorque_power, savedV_wec_power, savedpower = pickle.load(open('power_coarse.p', 'rb'))
savedtorque_force, savedV_wec_force, savedforce = pickle.load(open('force_coarse.p', 'rb'))

def h_poly_helper(tt):
  A = np.array([
      [1, 0, -3, 2],
      [0, 1, -2, 1],
      [0, 0, 3, -2],
      [0, 0, -1, 1]
      ], dtype=tt[-1].dtype)
  return [
    sum( A[i, j]*tt[j] for j in range(4) )
    for i in range(4) ]

def h_poly(t):
  tt = [ None for _ in range(4) ]
  tt[0] = 1
  for i in range(1, 4):
    tt[i] = tt[i-1]*t
  return h_poly_helper(tt)

def H_poly(t):
  tt = [ None for _ in range(4) ]
  tt[0] = t
  for i in range(1, 4):
    tt[i] = tt[i-1]*t*i/(i+1)
  return h_poly_helper(tt)

def interp_func(x, y):
  "Returns integral of interpolating function"
  if len(y)>1:
    m = (y[1:] - y[:-1])/(x[1:] - x[:-1])
    m = np.concatenate([m[[0]], (m[1:] + m[:-1])/2, m[[-1]]])
  def f(xs):
    if len(y)==1: # in the case of 1 point, treat as constant function
      return y[0] + np.zeros_like(xs)
    I = np.searchsorted(x[1:], xs)
    if max(x)<max(xs):
        print(max(xs))
        print(max(x))
        
    dx = (x[I+1]-x[I])
    hh = h_poly((xs-x[I])/dx)
    return hh[0]*y[I] + hh[1]*m[I]*dx + hh[2]*y[I+1] + hh[3]*m[I+1]*dx
  return f

def myinterp(x, y, xs):
    if xs.size==1:
        return interp_func(x,y)(xs)
    else:
        return [interp_func(x,y)(xs[ii]) for ii in range(xs.size)]

def interp2d(interp1d, xdata, ydata, fdata, xpt, ypt):

    yinterp = np.zeros((ydata.size,xpt.size))
    output = np.zeros((xpt.size,ypt.size)) #Array{R}(undef, nxpt, nypt)

    for i in range(ydata.size):
        yinterp[i, :] = interp1d(xdata, fdata[:, i], xpt)
    
    # yinterp = [np.array(interp1d(xdata,fdata[:, ii],xpt)) for ii in range(ydata.size)]

    for i in range(xpt.size):
        output[i, :] = interp1d(ydata, yinterp[:, i], ypt)
    
    # output = [interp1d(ydata,yinterp[:, ii],ypt) for ii in range(xpt.size)]

    return output


# def myforce_on_wec(wec,x_wec,x_opt,
#     waves = None,
#     nsubsteps = 1,
# )
    
#     if nsubsteps==1:
#         tmat = wec.time_mat
#     else:
#         tmat = wec.time_mat_nsubsteps(nsubsteps)

#     # Unstructured Controller Force
#     ndof = 1
#     x_opt = np.reshape(x_opt, (-1, ndof), order='F')
#     force_td = np.dot(tmat, x_opt)
#     assert force_td.shape == (wec.nt*nsubsteps, ndof)
#     force_td = np.expand_dims(np.transpose(force_td), axis=0)
#     assert force_td.shape == (1, ndof, wec.nt*nsubsteps)
#     kinematics = np.eye(ndof)
#     n = wec.nt*nsubsteps
#     kinematics_mat = np.repeat(kinematics[:, :, np.newaxis], n, axis=-1)    # kinematics_mat = self.kinematics(wec, x_wec, x_opt, waves, nsubsteps)
#     kinematics_mat = np.transpose(kinematics_mat, (1,0,2))
    
#     return np.transpose(np.sum(kinematics_mat*force_td, axis=1))

def myfancyfunction(wec,#: TWEC,
        x_wec,#: ndarray,
        x_opt,#: ndarray,
        waves = None,
        nsubsteps = 1):
    """Calculate average mechanical power in each PTO DOF for a
    given system state.

    Parameters
    ----------
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

    if nsubsteps==1:
        tmat = wec.time_mat
    else:
        tmat = wec.time_mat_nsubsteps(nsubsteps)

    # Unstructured Controller Force
    ndof = 1
    x_opt = np.reshape(x_opt, (-1, ndof), order='F')
    torque_td = np.dot(tmat, x_opt)

    # WEC Velocty
    pos_wec = wec.vec_to_dofmat(x_wec)
    vel_wec = np.dot(wec.derivative_mat, pos_wec)
    vel_wec_td = np.dot(tmat, vel_wec)
    assert vel_wec_td.shape == (wec.nt*nsubsteps, wec.ndof)
    vel_wec_td = np.expand_dims(np.transpose(vel_wec_td), axis=0)


    # if callable(kinematics):
    #     def kinematics_fun(wec, x_wec, x_opt, waves, nsubsteps=1):
    #         pos_wec = wec.vec_to_dofmat(x_wec)
    #         # tmat = self._tmat(wec, nsubsteps)
    #         pos_wec_td = np.dot(tmat, pos_wec)
    #         return kinematics(pos_wec_td)
    # else:
    #     def kinematics_fun(wec, x_wec, x_opt, waves, nsubsteps=1):
    #         n = wec.nt*nsubsteps
    #         return np.repeat(kinematics[:, :, np.newaxis], n, axis=-1)
    kinematics = np.eye(ndof)
    n = wec.nt*nsubsteps
    kinematics_mat = np.repeat(kinematics[:, :, np.newaxis], n, axis=-1)#kinematics_fun(wec, x_wec, x_opt, waves, nsubsteps)
    vel_td = np.transpose(np.sum(kinematics_mat*vel_wec_td, axis=1))

    # Turbine Power Considering Wec Avalaible Power from control and veloctiy
    # power_td = self_rectifying_turbine_power_internalsolve(torque_td,vel_td)
    power_td = interp2d(myinterp, savedtorque_power, savedV_wec_power, savedpower, torque_td, vel_td)
    force_td = interp2d(myinterp, savedtorque_force, savedV_wec_force, savedforce, torque_td, vel_td)
    energy = np.sum(power_td) * wec.dt/nsubsteps
    return energy / wec.tf, force_td

def mymechanical_average_power(wec,#: TWEC,
        x_wec,#: ndarray,
        x_opt,#: ndarray,
        waves = None,
        nsubsteps = 1,
    ) -> float:

    power, force_td = myfancyfunction(wec,x_wec,x_opt,waves = None,nsubsteps = 1)

    return power

def myforce_on_wec(wec,#: TWEC,
        x_wec,#: ndarray,
        x_opt,#: ndarray,
        waves = None,
        nsubsteps = 1,
    ):

    power, force_td = myfancyfunction(wec,x_wec,x_opt,waves = None,nsubsteps = 1)

    return force_td

wb = wot.geom.WaveBot()  # use standard dimensions
mesh_size_factor = 0.5 # 1.0 for default, smaller to refine mesh
mesh = wb.mesh(mesh_size_factor)
fb = cpy.FloatingBody.from_meshio(mesh, name="WaveBot")
fb.add_translation_dof(name="Heave")
ndof = fb.nb_dofs

stiffness = wot.hydrostatics.stiffness_matrix(fb).values
mass = wot.hydrostatics.inertia_matrix(fb).values

# fb.show_matplotlib()
# _ = wb.plot_cross_section(show=True)  # specific to WaveBot

f1 = 0.05
nfreq = 50
freq = wot.frequency(f1, nfreq, False) # False -> no zero frequency

bem_data = wot.run_bem(fb, freq)

name = ["PTO_Heave",]
# kinematics = np.eye(ndof)
# controller = None
# loss = None
# pto_impedance = None
# pto = wot.pto.PTO(ndof, kinematics, controller, pto_impedance, loss, name)

# PTO dynamics forcing function
f_add = {'PTO': myforce_on_wec}

# Constraint
f_max = 2000.0
nsubsteps = 4

def const_f_pto(wec, x_wec, x_opt, waves): # Format for scipy.optimize.minimize
    f = myforce_on_wec(wec, x_wec, x_opt, waves, nsubsteps)
    return f_max - np.abs(f.flatten())

ineq_cons = {'type': 'ineq',
             'fun': const_f_pto,
             }
constraints = [ineq_cons]

wec = wot.WEC.from_bem(
    bem_data,
    inertia_matrix=mass,
    hydrostatic_stiffness=stiffness,
    constraints=constraints,
    friction=None,
    f_add=f_add,
)

amplitude = 0.0625  
wavefreq = 0.3
phase = 30
wavedir = 0
waves = wot.waves.regular_wave(f1, nfreq, wavefreq, amplitude, phase, wavedir)

# obj_fun = pto.mechanical_average_power
obj_fun = mymechanical_average_power
nstate_opt = 2*nfreq+1

options = {'maxiter': 400, 'ftol': 1e-6}#, 'gtol': 1e-8}
scale_x_wec = 1e1
scale_x_opt = 1e-3
scale_obj = 1e-2
x_opt_0 = np.zeros(nstate_opt)+100.0
results = wec.solve(
    waves, 
    obj_fun, 
    nstate_opt,
    optim_options=options,
    x_opt_0=x_opt_0, 
    scale_x_wec=scale_x_wec,
    scale_x_opt=scale_x_opt,
    scale_obj=scale_obj,
    )

opt_mechanical_average_power = results.fun
print(f'Optimal average mechanical power: {opt_mechanical_average_power} W')