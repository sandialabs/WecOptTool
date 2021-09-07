#!/usr/bin/env python3

import os
import logging

import autograd.numpy as np
import capytaine as cpy
import matplotlib.pyplot as plt

import wecopttool as wot
from wecopttool.examples import wavebot


logging.basicConfig(level=logging.INFO)

# water properties
rho = 1e3

# frequencies
f0 = 0.05
num_freq = 25
mesh_size_factor = 0.5

# Capytaine floating body
mesh = wavebot.mesh(mesh_size_factor=mesh_size_factor)
# TODO: Capytaine fb from meshio/pygmsh mesh (Issues #13, #62)
mesh_file = 'tmp_mesh.stl'
mesh.write(mesh_file)
fb = cpy.FloatingBody.from_file(mesh_file, name='WaveBot')
os.remove(mesh_file)
fb.add_translation_dof(name="HEAVE")

# mass and hydrostatic stiffness
hs_data = wot.hydrostatics.hydrostatics(fb, rho=rho)
mass_33 = wot.hydrostatics.mass_matrix_constant_density(hs_data)[2, 2]
mass = np.atleast_2d(mass_33)
stiffness_33 = wot.hydrostatics.stiffness_matrix(hs_data)[2, 2]
stiffness = np.atleast_2d(stiffness_33)

# PTO: state, force, power (objective function)
kinematics = np.eye(fb.nb_dofs)
num_x_pto, f_pto, power_pto, pto_postproc = \
    wot.pto.pseudospectral_pto(num_freq, kinematics)

# create WEC
wec = wot.WEC(fb, mass, stiffness, f0, num_freq, f_add=f_pto, rho=rho)

# wave
freq = 0.2
amplitude = 0.25
phase = 0.0
waves = wot.waves.regular_wave(f0, num_freq, freq, amplitude, phase)

# run BEM
wec.run_bem()

# Scale
scale_wec = [1.0]
scale_opt = 1000.0
scale_obj = 1.0

# Constraints
constraints = []

# Solve dynamics & opt control
options = {'maxiter': 10000, 'ftol': 1e-8}

fdom, tdom, x_opt, res = wec.solve(
    waves, power_pto, num_x_pto,
    constraints=constraints, optim_options=options,
    scale_x_wec=scale_wec, scale_x_opt=scale_opt, scale_obj=scale_obj)

# post-process: PTO
tdom, fdom = pto_postproc(wec, tdom, fdom, x_opt)

# save
tdom.to_netcdf('tdom.nc')
wot.to_netcdf('fdom.nc', fdom)

# example time domain plots
plt.figure()
tdom['wave_elevation'].plot()

plt.figure()
tdom['pos'].plot()

plt.figure()
tdom['pto_force'].plot()

# example frequency domain plots
fd_lines = {'marker': 'o', 'linestyle': '', 'fillstyle': 'none'}

plt.figure()
np.abs(fdom['excitation_force']).plot(**fd_lines)

plt.figure()
np.abs(fdom['pto_force']).plot(**fd_lines)

plt.show()
