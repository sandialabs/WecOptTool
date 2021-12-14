#!/usr/bin/env python3

import logging
import os

import autograd.numpy as np
from autograd.builtins import isinstance, tuple, list, dict
import capytaine as cpy
import matplotlib.pyplot as plt
import pygmsh
import gmsh
import os
from scipy.optimize import brute

import wecopttool as wot
from wecopttool.example_devices import WaveBot

os.chdir(os.path.split(os.path.realpath(__file__))[0])

logging.basicConfig(level=logging.INFO)

#%%

r2_vals = np.arange(0.05, 0.95, 0.1)

r1 = 0.88
r2_0 = 0.35
h2_0 = 0.37
V0 = 1/3*np.pi*h2_0*(r1**2+r2_0**2+(r1*r2_0))

def h2_from_r2(r2,V=V0,r1=r1):
    h2 = V/(1/3*np.pi*(r1**2+r2**2+(r1*r2)))
    return h2

mapres = map(h2_from_r2, r2_vals)
h2_vals = list(mapres)

#%%

fig, ax = plt.subplots()

def design_obj_fun(x):
    
    r2 = x[0]
    
    h2 = h2_from_r2(r2)
    
    
    # generate a surface mesh for the WaveBot with which to perform BEM calcs
    wb = WaveBot(r2=r2,h2=h2)
    mesh = wb.mesh(mesh_size_factor=0.5)
    
    wb.plot_cross_section(ax,
                          label=f"r2={r2:.2f}, h2={h2:.2f}")


    # capytaine floating body (mesh + DOFs)
    fb = cpy.FloatingBody.from_meshio(mesh, name="WaveBot")
    fb.add_translation_dof(name="HEAVE")

    # mass & hydrostatic stiffness
    rho = 1000.0  # fresh water

    hs_data = wot.hydrostatics.hydrostatics(fb, rho=rho)
    mass_33 = wot.hydrostatics.mass_matrix_constant_density(hs_data)[2, 2]
    mass = np.atleast_2d(mass_33)
    stiffness_33 = wot.hydrostatics.stiffness_matrix(hs_data)[2, 2]
    stiffness = np.atleast_2d(stiffness_33)

    # frequencies
    f0 = 0.05
    nfreq = 50

    # PTOs (additional force + constraint + objective function)
    kinematics = np.eye(fb.nb_dofs)
    pto = wot.pto.PseudoSpectralPTO(nfreq, kinematics)

    # constraints at 4x points
    nsubsteps = 4
    f_max = 2000.0


    def const_f_pto(wec, x_wec, x_opt):
        f = pto.force_on_wec(wec, x_wec, x_opt, nsubsteps)
        return f_max - np.abs(f.flatten())


    ineq_cons = {'type': 'ineq',
                 'fun': const_f_pto,
                 }
    constraints = [ineq_cons]

    # additional friction
    dissipation = 160.3

    # create WECs
    f_added = pto.force_on_wec

    my_wec = wot.WEC(fb, mass, stiffness, f0, nfreq, rho=rho, f_add=f_added,
                     # constraints=constraints, 
                     dissipation=dissipation,
                     )

    # create save directory
    results_super_dir = 'tutorial_3_design_optimization_results'
    if not os.path.exists(results_super_dir):
      os.makedirs(results_super_dir)
      
    results_dir = os.path.join(results_super_dir, f"{x[0]:.2f}")
    if not os.path.exists(results_dir):
      os.makedirs(results_dir)

    # read BEM if available, else run & save
    fname = os.path.join(results_dir, 'bem.nc')
    if os.path.exists(fname):
        my_wec.read_bem(fname)
    else:
        my_wec.run_bem()
        my_wec.write_bem(fname)
        mesh.write(os.path.join(results_dir,f"{x[0]:.2f}.stl"))
        
        
    wfreq = 0.8
    amplitude = 0.0625
    phase = -40
    waves = wot.waves.regular_wave(f0, nfreq, wfreq, amplitude, phase)

    options = {'maxiter': 1000, 'ftol': 1e-8}
    obj_fun = pto.average_power
    nstate_opt = pto.nstate
    maximize = True

    wec_tdom, wec_fdom, x_wec, x_opt, obj, _ = my_wec.solve(waves, 
                                                            obj_fun, 
                                                            nstate_opt, 
                                                            optim_options=options, 
                                                            scale_x_opt=1e3)
    
    avg_power = obj
    
    return avg_power





res = brute(func=design_obj_fun,
            ranges=(slice(r2_vals[0], r2_vals[-1], np.diff(r2_vals)[0]),), # range over which to search
            full_output=True, # return eval grid and obj fun values
            finish=None, # no "polishing" at end (e.g., via gradient method)
            )


ax.legend()
ax.set_title('WaveBot hull cross sections')

#%%

fig, ax2 = plt.subplots()

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color'][:len(res[2])]
ax2.plot(res[2],res[3],'k', zorder=0)
ax2.scatter(res[2],res[3],c=colors,zorder=1)
ax2.set_xlabel('h1 [m]')
ax2.set_ylabel('Obj. fun')
