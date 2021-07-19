#!/usr/bin/env python3

import os
import logging

import autograd.numpy as np
import capytaine as cpy
import matplotlib.pyplot as plt

import WecOptTool as wot
from WecOptTool.examples import WaveBot


logging.basicConfig(level=logging.INFO)

# water properties
rho = 1e3

# frequencies
f0 = 0.05
num_freq = 30

# Capytaine floating body
# TODO: Capytaine fb from meshio/pygmsh mesh (Issues #13, #62)
mesh_size_factor = 0.25
mesh_file = 'mesh.stl'
mesh = WaveBot.hull(mesh_size_factor=mesh_size_factor)
mesh.write(mesh_file)
fb = cpy.FloatingBody.from_file(mesh_file, name='WaveBot')
os.remove(mesh_file)
fb.add_translation_dof(name="Heave")

# mass and hydrostatic stiffness
hs_data = wot.hydrostatics.hydrostatics(fb, rho=rho)
M33 = wot.hydrostatics.mass_matrix_constant_density(hs_data)[2, 2]
M = np.atleast_2d(M33)
K33 = wot.hydrostatics.stiffness_matrix(hs_data)[2, 2]
K = np.atleast_2d(K33)

# PTO: state, force, power (objective function)
kinematics = np.eye(fb.nb_dofs)
num_x_pto, f_pto, power_pto, pto_postproc = \
    wot.pto.pseudospectral_pto(num_freq, kinematics)

# create WEC
wec = wot.WEC(fb, M, K, f0, num_freq, f_add=f_pto, rho=rho)

# run BEM
wec.run_bem()

# wave
freq = 0.3
amplitude = 0.0625
phase = -40
waves = wot.waves.regular_wave(f0, num_freq, freq, amplitude, phase)

# Scale # TODO (makes it worst in some cases ... why?)
scale_wec = 1.0  # 1e-4
scale_opt = 1.0  # 1e-3
scale_obj = 1.0  # 1e-8

# Constraints
constraints = []

# Solve dynamics & opt control
options = {'maxiter': 1000, }

FD, TD, x_opt, res = wec.solve(
    waves, power_pto, num_x_pto,
    constraints=constraints, optim_options=options,
    scale_x_wec=scale_wec, scale_x_opt=scale_opt, scale_obj=scale_obj)

# post-process: PTO
TD['vel'] = TD['vel']
TD, FD = pto_postproc(wec, TD, FD, x_opt)

# save
TD.to_netcdf('TD.nc')
wot.to_netcdf('FD.nc', FD)

# example plots
plt.figure()
TD['pos'].sel(influenced_dof='Heave').plot()

plt.figure()
TD['excitation_force'].sel(influenced_dof='Heave').plot()

plt.figure()
TD['pto_force'].sel(dof_pto='pto_1').plot()

plt.figure()
np.abs(FD['wave_elevation'].sel(wave_direction=0.0)).plot()

plt.figure()
np.abs(FD['pto_force'].sel(dof_pto='pto_1')).plot(marker='o', linestyle='', fillstyle='none')

plt.show()
