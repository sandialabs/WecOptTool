from datetime import datetime

import autograd.numpy as np
import capytaine as cpy
import matplotlib.pyplot as plt
from scipy.optimize import brute

import wecopttool as wot

import autograd.numpy as np

A_wec = 0.1
A_turbine = 0.01
P_ambient = 101325.0 #pa (1 atm)
rho = 1.225 #kg/m3 air
R_turbine = 0.298/2 #m
H_blade = (1.0-0.7)*R_turbine #m
chord = 0.054 #m
N_blade = 30.0
lambda_air = 1.401

csv = np.genfromtxt('Ca_performance_table.csv', delimiter=",",skip_header=1)
inv_TSRdata_Ca = csv[:,0]
Ca_data = csv[:,1]

csv = np.genfromtxt('Ct_performance_table.csv', delimiter=",",skip_header=1)
inv_TSRdata_Ct = csv[:,0]
Ct_data = csv[:,1]

csv = np.genfromtxt('Eta_performance_table.csv', delimiter=",",skip_header=1)
inv_TSRdata_eta = csv[:,0]
eta_data = csv[:,1]

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
    dx = (x[I+1]-x[I])
    hh = h_poly((xs-x[I])/dx)
    return hh[0]*y[I] + hh[1]*m[I]*dx + hh[2]*y[I+1] + hh[3]*m[I+1]*dx
  return f

def myinterp(x, y, xs):
  return interp_func(x,y)(xs)

def self_rectifying_turbine_omega_incompressible(omega_turbine,V_wec):
  inv_TSR = (V_wec * A_wec) / (A_turbine * omega_turbine * R_turbine)
  Ca = myinterp(inv_TSRdata_Ca,Ca_data,inv_TSR)
  eta = myinterp(inv_TSRdata_eta,eta_data,inv_TSR)
  delta_P_turbine = Ca * 0.5 * rho * ((V_wec*A_wec/A_turbine)**2 + (omega_turbine*R_turbine)**2) * H_blade * chord * N_blade / A_wec

  F_turbine = delta_P_turbine/A_turbine
  F_wec = F_turbine*A_wec/A_turbine

  power = delta_P_turbine * V_wec * A_wec * eta
  print("inv_TSR")
  print(inv_TSR[10])
  print("Delta P")
  print(delta_P_turbine[10])
  print("eta")
  print(eta[10])
  return F_wec, power

def self_rectifying_turbine_omega_compressible(omega_turbine,V_wec,delta_P_turbine_guess):
  inv_TSR = (V_wec * A_wec) / (A_turbine * (delta_P_turbine_guess/P_ambient + 1)^(1/lambda_air) * omega_turbine * R_turbine)
  Ca = myinterp(inv_TSRdata_Ca,Ca_data,inv_TSR)
  eta = myinterp(inv_TSRdata_eta,eta_data,inv_TSR)
  
  delta_P_turbine = Ca * 0.5 * rho * ((V_wec*A_wec/(A_turbine * (delta_P_turbine_guess/P_ambient + 1)^(1/lambda_air)))**2 + (omega_turbine*R_turbine)**2) * H_blade * chord * N_blade / A_wec

  residual = delta_P_turbine_guess - delta_P_turbine

  F_turbine = delta_P_turbine/A_turbine
  F_wec = F_turbine*A_wec/A_turbine

  power = delta_P_turbine * V_wec * A_wec * eta

  return F_wec, power, residual

def self_rectifying_turbine_torque_incompressible(torque_turbine,V_wec,omega_turbine_guess):
  inv_TSR = (V_wec * A_wec) / (A_turbine * omega_turbine_guess * R_turbine)
  Ct = myinterp(inv_TSRdata_Ct,Ct_data,inv_TSR)
  Ca = myinterp(inv_TSRdata_Ca,Ca_data,inv_TSR)
  eta = myinterp(inv_TSRdata_eta,eta_data,inv_TSR)

  factorA = torque_turbine/(Ct*0.5*rho*H_blade*chord*N_blade*R_turbine)
  factorB = (V_wec*A_wec / (A_turbine))^2
  omega_turbine = 1/R_turbine * np.sqrt(factorA - factorB)

  delta_P_turbine = Ca * 0.5 * rho * ((V_wec*A_wec/(A_turbine))**2 + (omega_turbine*R_turbine)**2) * H_blade * chord * N_blade / A_wec

  residual = omega_turbine_guess - omega_turbine

  F_turbine = delta_P_turbine/A_turbine
  F_wec = F_turbine*A_wec/A_turbine

  power = delta_P_turbine * V_wec * A_wec * eta

  return F_wec, power, residual

def self_rectifying_turbine_torque_compressible(torque_turbine,V_wec,delta_P_turbine_guess,omega_turbine_guess):
  inv_TSR = (V_wec * A_wec) / (A_turbine * (delta_P_turbine_guess/P_ambient + 1)^(1/lambda_air) * omega_turbine_guess * R_turbine)
  Ct = myinterp(inv_TSRdata_Ct,Ct_data,inv_TSR)
  Ca = myinterp(inv_TSRdata_Ca,Ca_data,inv_TSR)
  eta = myinterp(inv_TSRdata_eta,eta_data,inv_TSR)

  factorA = torque_turbine/(Ct*0.5*rho*H_blade*chord*N_blade*R_turbine)
  factorB = (V_wec*A_wec / (A_turbine * (delta_P_turbine_guess/P_ambient + 1)^(1/lambda_air)))^2
  omega_turbine = 1/R_turbine * np.sqrt(factorA - factorB)

  delta_P_turbine = Ca * 0.5 * rho * ((V_wec*A_wec/(A_turbine * (delta_P_turbine_guess/P_ambient + 1)^(1/lambda_air)))**2 + (omega_turbine*R_turbine)**2) * H_blade * chord * N_blade / A_wec

  residual1 = delta_P_turbine_guess - delta_P_turbine
  residual2 = omega_turbine_guess - omega_turbine

  F_turbine = delta_P_turbine/A_turbine
  F_wec = F_turbine*A_wec/A_turbine

  power = delta_P_turbine * V_wec * A_wec * eta

  return F_wec, power, residual1, residual2

def myfancyfunction(wec,#: TWEC,
        x_wec,#: ndarray,
        x_opt,#: ndarray,
        waves = None,
        nsubsteps = 1,
    ):

    if nsubsteps==1:
        tmat = wec.time_mat
    else:
        tmat = wec.time_mat_nsubsteps(nsubsteps)

    # Unstructured Controller Force
    ndof = 1
    x_opt = np.reshape(x_opt, (-1, ndof), order='F')
    omega_turbine = np.dot(tmat, x_opt)

    # WEC Velocty
    pos_wec = wec.vec_to_dofmat(x_wec)
    V_wec = np.dot(wec.derivative_mat, pos_wec)
    V_wec_td = np.dot(tmat, V_wec)
    assert V_wec_td.shape == (wec.nt*nsubsteps, wec.ndof)
    V_wec_td = np.expand_dims(np.transpose(V_wec_td), axis=0)


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
    V_wec = np.transpose(np.sum(kinematics_mat*V_wec_td, axis=1))

    # Turbine Power Considering Wec Avalaible Power from control and velocity
    return self_rectifying_turbine_omega_incompressible(omega_turbine,V_wec)

def mymechanical_average_power(wec,#: TWEC,
        x_wec,#: ndarray,
        x_opt,#: ndarray,
        waves = None,
        nsubsteps = 1,
    ) -> float:

    F_wec, power_td = myfancyfunction(wec,x_wec,x_opt,waves = waves,nsubsteps = nsubsteps)

    energy = np.sum(power_td) * wec.dt/nsubsteps
    return energy / wec.tf

def myforce_on_wec(wec,x_wec,x_opt,
    waves = None,
    nsubsteps = 1,
    ):

    F_wec, power_td = myfancyfunction(wec,x_wec,x_opt,waves = waves,nsubsteps = nsubsteps)

    return F_wec

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

# make the pto just to get the post processing plotting
name = ["PTO_Heave",]
kinematics = np.eye(ndof)
controller = None
loss = None
pto_impedance = None
pto = wot.pto.PTO(ndof, kinematics, controller, pto_impedance, loss, name)

# obj_fun = pto.mechanical_average_power
obj_fun = mymechanical_average_power
nstate_opt = 2*nfreq+1

options = {'maxiter': 400, 'ftol': 1e-6}#, 'gtol': 1e-8}
scale_x_wec = 1e1
scale_x_opt = 1e-3
scale_obj = 1e-2
x_opt_0 = np.zeros(nstate_opt)+500.0
x_opt_0 = np.array([-1.35820313e+00,-1.35820313e+00,1.27395575e+00,3.40663821e-01,-1.60176907e-01,6.72369527e-01,5.85006393e-01,2.05316692e-01,7.67355542e-01,-4.02968817e-01,-2.56042235e-02,1.18714270e+03,-2.05955794e+03,2.10003941e-01,1.02685095e+00,1.95416862e-01,6.46744053e-01,3.67696994e-01,-2.31206444e-01,-2.66093903e-01,6.37978846e-01,-3.52209582e-01,1.54584573e-01,4.01602961e+00,-7.68086363e+01,5.20604122e-02,4.08862136e-02,-1.69685620e-01,-7.12413859e-02,3.96954721e-01,-9.04069030e-02,-3.47134718e-01,5.07550884e-01,-9.45214013e-01,-7.51856252e-01,4.47332655e+02,-9.51562665e+00,-1.08701100e-01,3.49702444e-01,-2.46220202e+00,-4.12677098e-02,-5.36790310e-01,-1.95200653e+00,-3.29747032e-01,-1.19942508e+00,-4.12288613e+00,1.22432656e+00,3.73045502e+01,3.15940280e+01,1.67030742e+00,2.86704131e+00,4.75081926e-02,-3.52248358e-01,9.77102632e-01,-7.97998069e-01,-7.95980662e-01,-1.40340607e-01,-3.86320537e-01,-4.74585983e-01,4.34203964e+01,5.31633277e+01,9.17683498e-01,5.60296892e-01,-8.41864196e-01,-3.61902981e+00,1.55309417e+00,-2.63564981e+00,1.57028794e+00,-2.29978747e-01,-4.17581485e+00,-2.75686009e+00,-3.71676537e+01,2.60487459e+01,-1.14666111e+00,3.23316149e+00,-1.13941058e+00,-5.48810103e-02,1.07021103e+00,2.10510855e-01,2.46316068e-01,-8.57633907e-01,-3.87906675e-01,-1.51653609e-01,5.08780704e+00,1.03028143e+00,4.07266194e-01,8.01792959e-01,1.59517656e+00,-8.56679154e-01,1.39065632e+00,-2.10597448e-01,1.36302416e-01,5.12913991e-01,8.07886103e-02,-2.10917559e+00,-1.42136131e+01,-1.34012039e+01,-1.11057334e+00,4.40781648e-01,-2.32283498e-01,-8.21216680e-01])
x_opt_0 = (x_opt_0+10.0)*50.0
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



nsubsteps = 5
pto_fdom, pto_tdom = pto.post_process(wec, results, waves, nsubsteps=nsubsteps)
wec_fdom, wec_tdom = wec.post_process(results, waves, nsubsteps=nsubsteps)

plt.figure()
pto_tdom['mech_power'].plot()
plt.show()

plt.figure()
wec_tdom['pos'].plot()
plt.show()

plt.figure()
pto_tdom['force'].plot()
plt.show()