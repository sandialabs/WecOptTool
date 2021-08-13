#!/usr/bin/env python3

import os
import logging

import autograd.numpy as np
import capytaine as cpy
import matplotlib.pyplot as plt
from mhkit.wave.resource import jonswap_spectrum

import wecopttool as wot
from wecopttool.examples import WaveBot


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
period_peak = 6.0
height_sig = 0.5
gamma = 3.3
wave_direction = 0.0

spectrum_name = \
    f'JONSWAP, Hs = {height_sig} m, Tp = {period_peak} s, gamma= {gamma}'


def spectrum(freq):
    return jonswap_spectrum(freq, period_peak, height_sig, gamma).values


waves = wot.waves.long_crested_wave(
    f0, num_freq, spectrum, wave_direction, spectrum_name)

# run BEM
wec.run_bem(wave_dirs=waves['wave_direction'].values)

# Solve
f_dom, t_dom, x_opt, res = wec.solve(waves, power_pto, num_x_pto)

# post-process: PTO
t_dom, f_dom = pto_postproc(wec, t_dom, f_dom, x_opt)

# save
t_dom.to_netcdf('t_dom.nc')
wot.to_netcdf('f_dom.nc', f_dom)

# example time domain plots
plt.figure()
t_dom['wave_elevation'].plot()

plt.figure()
t_dom['excitation_force'].plot()

plt.figure()
t_dom['pos'].plot()

plt.figure()
t_dom['pto_force'].plot()

# example frequency domain plots
fd_lines = {'marker': 'o', 'linestyle': '', 'fillstyle': 'none'}

plt.figure()
np.abs(f_dom['excitation_force']).plot(**fd_lines)

plt.figure()
np.abs(f_dom['pto_force']).plot(**fd_lines)

plt.show()
