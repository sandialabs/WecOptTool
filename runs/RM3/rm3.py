#!/usr/bin/env python3

import os
import logging

import autograd.numpy as np
import capytaine as cpy
import matplotlib.pyplot as plt
from scipy.linalg import block_diag

import wecopttool as wot
from wecopttool.examples import rm3


logging.basicConfig(level=logging.INFO)


def parallel_ax(inertia, mass, distance_vec):
    r = np.reshape(distance_vec, (3, 1))
    r2 = (r.T @ r) * np.eye(3) - r @ r.T
    return inertia + mass * r2


def idx(body, dof):
    return 6 * body + dof


bodies = {'float': 0, 'spar': 1, 'rm3': 2}
dofs = {'SURGE': 0, 'SWAY': 1, 'HEAVE': 2, 'ROLL': 3, 'PITCH': 4, 'YAW': 5}

# case selection
ndof = 4
if ndof == 2:
    rm3_translation_dofs = []
    rm3_rotation_dofs = []
elif ndof == 4:
    rm3_translation_dofs = ['SURGE']
    rm3_rotation_dofs = ['PITCH']
elif ndof == 7:
    rm3_translation_dofs = ['SURGE', 'SWAY']
    rm3_rotation_dofs = ['ROLL', 'PITCH', 'YAW']
else:
    raise ValueError('`ndof` must be 2, 4 or 7.')
rm3_dofs = rm3_translation_dofs + rm3_rotation_dofs
rm3_ndofs = len(rm3_dofs)

# water properties
rho = 1000

# frequencies
f0 = 0.0032
num_freq = 100  # 260

# float - heave
float_fb = cpy.FloatingBody.from_file(
    rm3.float_mesh_file, name='float').translate(rm3.float_cog)
float_fb.add_translation_dof(name="HEAVE")

# float mass & stiffness
float_hs_data = wot.hydrostatics.hydrostatics(float_fb, rho=rho)
float_stiffness = wot.hydrostatics.stiffness_matrix(float_hs_data)
float_mass = wot.hydrostatics.mass_matrix_constant_density(float_hs_data)
float_lmass = float_mass[0, 0]
float_mass[3:, 3:] = rm3.float_moi

# spar - heave
spar_fb = cpy.FloatingBody.from_file(
    rm3.spar_mesh_file, name='spar').translate(rm3.spar_cog)
spar_fb.add_translation_dof(name="HEAVE")

# spar - mass & stiffness
spar_hs_data = wot.hydrostatics.hydrostatics(spar_fb, rho=rho)
spar_stiffness = wot.hydrostatics.stiffness_matrix(spar_hs_data)
spar_mass = wot.hydrostatics.mass_matrix_constant_density(spar_hs_data)
spar_lmass = spar_mass[0, 0]
spar_mass[3:, 3:] = rm3.spar_moi

# RM3
rm3_fb = float_fb + spar_fb

# RM3 - translation DOFs
for dof in rm3_translation_dofs:
    rm3_fb.add_translation_dof(name=dof)

# RM3 - mass & stiffness
rm3_lmass = float_lmass + spar_lmass
rm3_cog = (float_lmass*rm3.float_cog + spar_lmass*rm3.spar_cog) / rm3_lmass
moi_1 = parallel_ax(rm3.float_moi, float_lmass, rm3.float_cog - rm3_cog)
moi_2 = parallel_ax(rm3.spar_moi, spar_lmass, rm3.spar_cog - rm3_cog)
rm3_moi = moi_1 + moi_2
rm3_mass = block_diag(rm3_lmass, rm3_lmass, rm3_lmass, rm3_moi)
rm3_hs_data = wot.hydrostatics.hydrostatics(rm3_fb, rho=rho)
rm3_stiffness = wot.hydrostatics.stiffness_matrix(rm3_hs_data)

# RM3 - rotation DOFs
rm3_fb.rotation_center = rm3_cog
for dof in rm3_rotation_dofs:
    rm3_fb.add_rotation_dof(name=dof)

# mass & hydrostatic stiffness
mass_all = block_diag(float_mass, spar_mass, rm3_mass)
stiffness_all = block_diag(float_stiffness, spar_stiffness, rm3_stiffness)
idxs = np.array([
    [idx(bodies['float'], dofs['HEAVE']), idx(bodies['spar'], dofs['HEAVE'])] +
    [idx(bodies['rm3'], dofs[dof]) for dof in rm3_dofs]])
mass = mass_all[idxs.T, idxs]
stiffness = stiffness_all[idxs.T, idxs]

# PTO: state, force, power (objective function)
kinematics = np.array([[1.0, -1.0] + [0.0]*rm3_ndofs])
num_x_pto, f_pto, power_pto, pto_postproc = \
    wot.pto.pseudospectral_pto(num_freq, kinematics)

# create WEC
wec = wot.WEC(rm3_fb, mass, stiffness, f0, num_freq, f_add=f_pto, rho=rho)

# wave
freq = 0.125
amplitude = 1.25
phase = None
waves = wot.waves.regular_wave(f0, num_freq, freq, amplitude, phase)

# run BEM
bem_dir = 'hydroData'
bem_file = os.path.join(bem_dir, f'bem_{ndof}d.nc')
if not os.path.exists(bem_dir):
    os.makedirs(bem_dir)
if os.path.isfile(bem_file):
    wec.read_bem(bem_file)
else:
    wec.run_bem()
    wec.write_bem(bem_file)

# solve
f_dom, t_dom, x_opt, res = wec.solve(waves, power_pto, num_x_pto)

# post-process: PTO
t_dom, f_dom = pto_postproc(wec, t_dom, f_dom, x_opt)

# save
save_dir = 'results'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
t_dom.to_netcdf(os.path.join(save_dir, f't_dom_{ndof}d.nc'))
wot.to_netcdf(os.path.join(save_dir, f'f_dom_{ndof}d.nc'), f_dom)

# plots
figw = 10
figh = 2.5

# time domain plots - wave
plt.figure(figsize=(figw, figh*1))
t_dom['wave_elevation'].plot()
plt.tight_layout()

# time domain plots - WEC
fig, axs = plt.subplots(ndof, 1, figsize=(figw, figh*ndof), sharex=True)
t_dom['pos'].sel(influenced_dof='float__HEAVE').plot(ax=axs[0])
t_dom['pos'].sel(influenced_dof='spar__HEAVE').plot(ax=axs[1])
for i, dof in enumerate(rm3_dofs):
    t_dom['pos'].sel(influenced_dof=dof).plot(ax=axs[i+2])
plt.tight_layout()
fig.align_ylabels(axs)

# time domain plots - PTO
keys = ['pto_pos', 'pto_vel', 'pto_force', 'power']
nkeys = len(keys)
fig, axs = plt.subplots(nkeys, 1, figsize=(figw, figh*nkeys), sharex=True)
for i, key in enumerate(keys):
    t_dom[key].plot(ax=axs[i])
    axs[i].set_title('')
plt.tight_layout()
fig.align_ylabels(axs)

plt.show()
