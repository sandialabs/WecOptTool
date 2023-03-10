from datetime import datetime

import autograd.numpy as np
import capytaine as cpy
import matplotlib.pyplot as plt
from scipy.optimize import brute

import wecopttool as wot

import autograd.numpy as np

def myforce_on_wec(wec,x_wec,x_opt,
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
    force_td = np.dot(tmat, x_opt)
    assert force_td.shape == (wec.nt*nsubsteps, ndof)
    force_td = np.expand_dims(np.transpose(force_td), axis=0)
    assert force_td.shape == (1, ndof, wec.nt*nsubsteps)
    kinematics = np.eye(ndof)
    n = wec.nt*nsubsteps
    kinematics_mat = np.repeat(kinematics[:, :, np.newaxis], n, axis=-1)    # kinematics_mat = self.kinematics(wec, x_wec, x_opt, waves, nsubsteps)
    kinematics_mat = np.transpose(kinematics_mat, (1,0,2))
    
    return np.transpose(np.sum(kinematics_mat*force_td, axis=1))

def mymechanical_average_power(wec,#: TWEC,
        x_wec,#: ndarray,
        x_opt,#: ndarray,
        waves = None,
        nsubsteps = 1,
    ) -> float:
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
    force_td = np.dot(tmat, x_opt)

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
    power_td = vel_td * force_td
    energy = np.sum(power_td) * wec.dt/nsubsteps
    return energy / wec.tf

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