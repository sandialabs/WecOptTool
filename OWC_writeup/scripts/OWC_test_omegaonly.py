from datetime import datetime

import autograd.numpy as np
import capytaine as cpy
import matplotlib.pyplot as plt
from scipy.optimize import brute
import pickle

import wecopttool as wot

import autograd.numpy as np

# savedtorque_power, savedV_wec_power, savedpower = pickle.load(open('power_coarse.p', 'rb'))
# savedtorque_force, savedV_wec_force, savedforce = pickle.load(open('force_coarse.p', 'rb'))

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
    # if max(x)<max(xs):
    #     print(max(xs))
    #     print(max(x))
        
    dx = (x[I+1]-x[I])
    hh = h_poly((xs-x[I])/dx)
    return hh[0]*y[I] + hh[1]*m[I]*dx + hh[2]*y[I+1] + hh[3]*m[I+1]*dx
  return f

def myinterp(x, y, xs):
    # if xs.size==1:
    return interp_func(x,y)(xs)
    # else:
    #     return [interp_func(x,y)(xs[ii]) for ii in range(xs.size)]

# def interp2d(interp1d, xdata, ydata, fdata, xpt, ypt):

#     yinterp = np.zeros((ydata.size,xpt.size))
#     output = np.zeros((xpt.size,ypt.size)) #Array{R}(undef, nxpt, nypt)

#     for i in range(ydata.size):
#         yinterp[i, :] = interp1d(xdata, fdata[:, i], xpt)
    
#     # yinterp = [np.array(interp1d(xdata,fdata[:, ii],xpt)) for ii in range(ydata.size)]

#     for i in range(xpt.size):
#         output[i, :] = interp1d(ydata, yinterp[:, i], ypt)
    
#     # output = [interp1d(ydata,yinterp[:, ii],ypt) for ii in range(xpt.size)]

#     return output

def self_rectifying_turbine_power_internalsolve(omega_turbine,V_wec): #m/s omega_turbine_guess: ndarray, #rad/s
    # ) -> np.ndarray:#tuple[np.ndarray, np.ndarray]:
    """Calculate the turbine power.  Omega is assumed always positive (the whole point of a self rectifying turbine),
    but power can be put into the system, which for now is assumed symmetric in the performance curves.

    Parameters
    ----------
    torque
        Torque on the turbine
    V_wec
        Velocity of the wec
    """
    # len xwec is set (ndof * ncomponents (2*nfreq+1))
    # len xopt is any length and be anything, but pto module has rules about shape etc.  
    # TODO: put in self
    rho = 1.225
    blade_chord = 0.054 #m
    N_blades = 30.0
    blade_radius = 0.298/2 #m
    blade_height = (1.0-0.7)*blade_radius #m
    A_wec = 2.0 #m2
    A_turb = np.pi*blade_radius**2 #m2
    P_ambient = 101325.0 #pa (1 atm)

    # power = np.zeros_like(torque)

    V_turb = np.abs(A_wec*V_wec/A_turb) #turbine always sees velocity as positive
    # power1 = np.abs(torque_ii*omega_turbine_guess)
    V_turbcompressed = V_turb#/((P_ambient+power1/(A_turb*V_turb))/P_ambient)**(1/1.401) #compressibility correction pv = nrt -> assume adiabiatic p2/p1 = (V1/V2)^specific_heat_ratio (1.401 for air for the operating temp we're at)
    
    inv_TSR = V_turbcompressed/(omega_turbine*blade_radius)
    inv_TSRdata_Ca = np.array([-1000000.0,
        -100.0,
        -50.0,
        -20.0,
        -10.0,
        0.0,
        0.012668019,
        0.03426608,
        0.045142045,
        0.057553153,
        0.071173399,
        0.085301268,
        0.097366303,
        0.118841776,
        0.129754941,
        0.139778827,
        0.158789719,
        0.169510357,
        0.179040696,
        0.19001597,
        0.20320735,
        0.217075432,
        0.235373075,
        0.243281766,
        0.261535771,
        0.276437612,
        0.28738014,
        0.309746514,
        0.314496222,
        0.330525042,
        0.346010286,
        0.35791277,
        0.39363278,
        0.426007626,
        0.432751135,
        0.448256149,
        0.472991905,
        0.49238604,
        0.506538082,
        0.522258542,
        0.541141927,
        0.559711436,
        0.587280356,
        0.603243099,
        0.623148907,
        0.643251111,
        0.66566147,
        0.680382139,
        0.702029778,
        0.72676211,
        0.75149701,
        0.778443456,
        0.802495913,
        0.830306138,
        0.859654739,
        0.890547726,
        0.921439344,
        0.953872962,
        0.987842604,
        1.021805398,
        1.055763211,
        1.089717911,
        1.123670121,
        1.157624821,
        1.191565203,
        1.225508697,
        1.259449701,
        1.293384479,
        1.327313966,
        1.361237849,
        1.395167335,
        1.429090907,
        1.463007631,
        1.496926223,
        1.530844503,
        1.564760293,
        1.598674838,
        1.632587515,
        1.666499881,
        1.700410691,
        1.734322745,
        1.768229819,
        1.802136271,
        1.836043345,
        1.869948551,
        1.903858116,
        1.937757408,
        1.971661992,
        2.005578093,
        2.039467113,
        2.073365782,
        2.107270366,
        2.14117246,
        2.175075487,
        2.208974156,
        2.242872826,
        2.276771495,
        2.310670165,
        2.344568834,
        2.378467504,
        2.412366173,
        2.446264843,
        2.480163512,
        2.503649469,
        2.509371154,
        2.65,
        2.75,
        3.0,
        4.0,
        5.0,
        7.0,
        10.0,
        20.0,
        50.0,
        100.0,
        1000000.0])

    Ca_data = np.array([0.092282521,
        0.092282521,
        0.092282521,
        0.092282521,
        0.092282521,
        0.092282521,
        0.092282521,
        0.108898362,
        0.166429569,
        0.220299948,
        0.281008945,
        0.337880945,
        0.389410575,
        0.459040664,
        0.518464217,
        0.578526192,
        0.629890894,
        0.672544117,
        0.723541827,
        0.778887736,
        0.830091701,
        0.892335842,
        0.945980007,
        0.994584051,
        1.05696206,
        1.106830028,
        1.161446822,
        1.228899216,
        1.290898399,
        1.359077526,
        1.408090836,
        1.465159246,
        1.562243494,
        1.655088735,
        1.697592835,
        1.759255085,
        1.811679549,
        1.859073611,
        1.914558511,
        1.982617847,
        2.036904363,
        2.087541557,
        2.145242742,
        2.194618208,
        2.255209854,
        2.300621402,
        2.344023427,
        2.396784052,
        2.445120045,
        2.495353764,
        2.547230542,
        2.59005665,
        2.643459039,
        2.691239423,
        2.737428015,
        2.785870789,
        2.833437265,
        2.88173104,
        2.92692827,
        2.967744011,
        3.005373215,
        3.041010834,
        3.075055183,
        3.110692802,
        3.137169126,
        3.165637035,
        3.192511676,
        3.215403146,
        3.23490892,
        3.250829839,
        3.270335613,
        3.286057374,
        3.297397646,
        3.30993287,
        3.322268935,
        3.333011731,
        3.342957894,
        3.351709104,
        3.360261157,
        3.367817416,
        3.37617031,
        3.381336666,
        3.386104705,
        3.391271062,
        3.395242467,
        3.402002092,
        3.402189484,
        3.405762572,
        3.4066704527,
        3.410319686,
        3.410108761,
        3.413681849,
        3.415661668,
        3.418238963,
        3.418028038,
        3.417817113,
        3.417606188,
        3.417395263,
        3.417184338,
        3.416973413,
        3.416762488,
        3.416551563,
        3.416340638,
        3.408526823,
        3.408526823,
        3.408526823,
        3.408526823,
        3.408526823,
        3.408526823,
        3.408526823,
        3.408526823,
        3.408526823,
        3.408526823,
        3.408526823,
        3.408526823,
        3.408526823])
    
    # if inv_TSR > 1000000.0:
    #     inv_TSR = np.array(1000000.0)

    Ca_used = myinterp(inv_TSRdata_Ca,Ca_data,inv_TSR)
    # Ca_used = 0.4093*inv_TSR**3 - 2.5095*inv_TSR**2 + 5.1228*inv_TSR - 0.0864
    flowrate_turb = V_turbcompressed*A_turb
    flowrate_wec = A_wec*V_wec

    deltaP_turb = Ca_used/flowrate_turb*0.5*rho*blade_height*blade_chord*N_blades*V_turbcompressed*(V_turbcompressed**2+(omega_turbine*blade_radius)**2)
    deltaP_wec = deltaP_turb*flowrate_turb/flowrate_wec
    force_wec = deltaP_wec*A_wec

    power2 = deltaP_turb*flowrate_turb#                    Ca_used*0.5*rho*blade_height*blade_chord*N_blades*V_turbcompressed*(V_turbcompressed**2+(omega_turbine*blade_radius)**2)

    return power2, force_wec


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
    omega_td = np.dot(tmat, x_opt)

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
    power_td,force_wec = self_rectifying_turbine_power_internalsolve(omega_td,vel_td)
    print('omega')
    print(omega_td)
    print('vel_td')
    print(vel_td)
    energy = np.sum(power_td) * wec.dt/nsubsteps
    return energy / wec.tf,force_wec

def mymechanical_average_power(wec,#: TWEC,
        x_wec,#: ndarray,
        x_opt,#: ndarray,
        waves = None,
        nsubsteps = 1,
    ):

    power, force_td = myfancyfunction(wec,x_wec,x_opt,waves = None,nsubsteps = 1)
    print('power')
    print(power)
    return power

def myforce_on_wec(wec,#: TWEC,
        x_wec,#: ndarray,
        x_opt,#: ndarray,
        waves = None,
        nsubsteps = 1,
    ):

    power, force_td = myfancyfunction(wec,x_wec,x_opt,waves = None,nsubsteps = 1)
    print('force_td')
    print(force_td)
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
# bnds = ((0, None), (0, None))
mylb = 0.1
myub = 3000.0
bounds_opt = ((mylb, myub), (mylb, myub), (mylb, myub), (mylb, myub), (mylb, myub), (mylb, myub),(mylb, myub), (mylb, myub), (mylb, myub), (mylb, myub), (mylb, myub),(mylb, myub), (mylb, myub), (mylb, myub), (mylb, myub), (mylb, myub),(mylb, myub), (mylb, myub), (mylb, myub), (mylb, myub), (mylb, myub),(mylb, myub), (mylb, myub), (mylb, myub), (mylb, myub), (mylb, myub),(mylb, myub), (mylb, myub), (mylb, myub), (mylb, myub), (mylb, myub),(mylb, myub), (mylb, myub), (mylb, myub), (mylb, myub), (mylb, myub),(mylb, myub), (mylb, myub), (mylb, myub), (mylb, myub), (mylb, myub),(mylb, myub), (mylb, myub), (mylb, myub), (mylb, myub), (mylb, myub),(mylb, myub), (mylb, myub), (mylb, myub), (mylb, myub), (mylb, myub),(mylb, myub), (mylb, myub), (mylb, myub), (mylb, myub), (mylb, myub),(mylb, myub), (mylb, myub), (mylb, myub), (mylb, myub), (mylb, myub),(mylb, myub), (mylb, myub), (mylb, myub), (mylb, myub), (mylb, myub),(mylb, myub), (mylb, myub), (mylb, myub), (mylb, myub), (mylb, myub),(mylb, myub), (mylb, myub), (mylb, myub), (mylb, myub), (mylb, myub),(mylb, myub), (mylb, myub), (mylb, myub), (mylb, myub), (mylb, myub),(mylb, myub), (mylb, myub), (mylb, myub), (mylb, myub), (mylb, myub),(mylb, myub), (mylb, myub), (mylb, myub), (mylb, myub), (mylb, myub),(mylb, myub), (mylb, myub), (mylb, myub), (mylb, myub), (mylb, myub),(mylb, myub), (mylb, myub), (mylb, myub), (mylb, myub), (mylb, myub))
results = wec.solve(
    waves, 
    obj_fun, 
    nstate_opt,
    optim_options=options,
    x_opt_0=x_opt_0, 
    bounds_opt=bounds_opt,
    scale_x_wec=scale_x_wec,
    scale_x_opt=scale_x_opt,
    scale_obj=scale_obj,
    )

opt_mechanical_average_power = results.fun
print(f'Optimal average mechanical power: {opt_mechanical_average_power} W')