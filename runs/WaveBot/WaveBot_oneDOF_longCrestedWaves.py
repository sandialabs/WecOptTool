#!/usr/bin/env python3

import os
import logging

import autograd.numpy as np
import capytaine as cpy
import matplotlib.pyplot as plt
from mhkit.wave.resource import jonswap_spectrum

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

# wave
Tp = 6.0
Hs = 0.5
gamma = 3.3
wave_direction = 0.0

spectrum_name = 'JONSWAP, Hs = 0.5 m, Tp = 6.0 s, gamma=3.3'

def spectrum(freq):
    return jonswap_spectrum(freq, Tp, Hs, gamma).values

waves = wot.waves.long_crested_wave(
    f0, num_freq, spectrum, wave_direction, spectrum_name)

# run BEM
wec.run_bem(wave_dirs=waves['wave_direction'].values)

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
TD['excitation_force'].plot()

plt.figure()
TD['pos'].plot()

plt.figure()
TD['pto_force'].plot()

# example frequency domain plots
fd_lines = {'marker': 'o', 'linestyle': '', 'fillstyle': 'none'}

plt.figure()
np.abs(FD['excitation_force']).plot(**fd_lines)

plt.figure()
np.abs(FD['pto_force']).plot(**fd_lines)

plt.show()
