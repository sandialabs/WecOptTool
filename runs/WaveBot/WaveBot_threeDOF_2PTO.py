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
num_freq = 18

# Capytaine floating body
mesh = WaveBot.mesh()
# TODO: Capytaine fb from meshio/pygmsh mesh (Issues #13, #62)
mesh_file = 'tmp_mesh.stl'
mesh.write(mesh_file)
fb = cpy.FloatingBody.from_file(mesh_file, name='WaveBot')
os.remove(mesh_file)
fb.add_translation_dof(name="HEAVE")
fb.add_translation_dof(name="SURGE")
fb.rotation_center = np.array([0, 0, 0])
fb.add_rotation_dof(name="PITCH")

# mass and hydrostatic stiffness
hs_data = wot.hydrostatics.hydrostatics(fb, rho=rho)
idx = np.array([[0, 2, 4]])
M = wot.hydrostatics.mass_matrix_constant_density(hs_data)[idx, idx.T]
K = wot.hydrostatics.stiffness_matrix(hs_data)[idx, idx.T]

# PTO: state, force, power (objective function)
pto_names = ['HEAVE PTO', 'PITCH PTO']
kinematics = np.array([[0, 1, 0], [0, 0, 1]])
num_x_pto, f_pto, power_pto, pto_postproc = \
    wot.pto.pseudospectral_pto(num_freq, kinematics, pto_names)

# create WEC
wec = wot.WEC(fb, M, K, f0, num_freq, f_add=f_pto, rho=rho)

# wave
freq = 0.2
amplitude = 0.25
phase = 0.0
waves = wot.waves.regular_wave(f0, num_freq, freq, amplitude, phase)

# run BEM
wec.run_bem()

# Solve
FD, TD, x_opt, res = wec.solve(waves, power_pto, num_x_pto)

# post-process: PTO
TD, FD = pto_postproc(wec, TD, FD, x_opt)

# save
TD.to_netcdf('TD.nc')
wot.to_netcdf('FD.nc', FD)

# example time domain plots
plt.figure()
TD['wave_elevation'].plot()

plt.figure()
TD['pos'].sel(influenced_dof='HEAVE').plot()

plt.figure()
TD['pos'].sel(influenced_dof='PITCH').plot()

plt.figure()
TD['pos'].sel(influenced_dof='SURGE').plot()

plt.figure()
TD['pto_force'].sel(dof_pto='HEAVE PTO').plot()

plt.figure()
TD['pto_force'].sel(dof_pto='PITCH PTO').plot()

# example frequency domain plots
fd_lines = {'marker': 'o', 'linestyle': '', 'fillstyle': 'none'}

plt.figure()
np.abs(FD['excitation_force'].sel(influenced_dof='HEAVE')).plot(**fd_lines)

plt.figure()
np.abs(FD['pto_force'].sel(dof_pto='HEAVE PTO')).plot(**fd_lines)

plt.figure()
np.abs(FD['pto_force'].sel(dof_pto='PITCH PTO')).plot(**fd_lines)

plt.show()
