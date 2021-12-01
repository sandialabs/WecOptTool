#!/usr/bin/env python3

## LOGGING - RECOMMENDED
#  Control ammount of information printed with the logging level
import logging
logging.basicConfig(level=logging.INFO)

## DERIVATIVES - REQUIRED
import autograd.numpy as np
from autograd.builtins import isinstance, tuple, list, dict

## IMPORT OTHER PACKAGES
import os

import capytaine as cpy
import matplotlib.pyplot as plt
import pygmsh
import gmsh

import wecopttool as wot


## CREATE THE MESH
#  mesh
h1 = 0.17  # Height of cylindrical section in meters
h2 = 0.37  # Height of conical section in meters
r1 = 0.88  # Top radius in meters
r2 = 0.35  # Bottom radius in meters
freeboard = 0.01  # Height above water of cylindrical section in meters
                  #   Draft of cylindrical section = h1 - freeboard
mesh_size_factor = 0.1  # Control for mesh discretization (number of cells).
                        #   =1 for baseline, <1 to refine.

with pygmsh.occ.Geometry() as geom:
    gmsh.option.setNumber('Mesh.MeshSizeFactor', mesh_size_factor)
    cyl = geom.add_cylinder([0, 0, 0], [0, 0, -h1], r1)
    cone = geom.add_cone([0, 0, -h1], [0, 0, -h2], r1, r2)
    geom.translate(cyl, [0, 0, freeboard])
    geom.translate(cone, [0, 0, freeboard])
    geom.boolean_union([cyl, cone])
    mesh = geom.generate_mesh()

## CREATE A CAPYTAINE FLOATING BODY (MESH + DOFS)
fb = cpy.FloatingBody.from_meshio(mesh, name="WaveBot")
fb.add_translation_dof(name="HEAVE")

## MASS AND HYDROSTATIC STIFFNESS MATRICES
rho = 1000.0  # fresh water

hs_data = wot.hydrostatics.hydrostatics(fb, rho=rho)
mass_33 = wot.hydrostatics.mass_matrix_constant_density(hs_data)[2, 2]
mass = np.atleast_2d(mass_33)
stiffness_33 = wot.hydrostatics.stiffness_matrix(hs_data)[2, 2]
stiffness = np.atleast_2d(stiffness_33)

## DEFINE PSEUDOSPECTRAL PROBLEM FREQUENCIES
f0 = 0.05
nfreq = 50

## PTO (additional force + constraint + objective function)
kinematics = np.eye(fb.nb_dofs)
pto = wot.pto.PseudoSpectralPTO(nfreq, kinematics)

## CONSTRAINTS at 4x points
nsubsteps = 4
f_max = 2000.0


def const_f_pto(wec, x_wec, x_opt):
    f = pto.force_on_wec(wec, x_wec, x_opt, nsubsteps)
    return f_max - np.abs(f.flatten())


ineq_cons = {'type': 'ineq',
             'fun': const_f_pto,
             }
constraints = [ineq_cons]

## CREATE WEC
f_add = pto.force_on_wec # PTO force

wec = wot.WEC(fb, mass, stiffness, f0, nfreq, rho=rho,
              f_add=f_add, constraints=constraints)

## RUN/READ BEM
# create save directory
results_dir = 'results_tutorial_1'
if not os.path.exists(results_dir):
  os.makedirs(results_dir)

# read BEM if available, else run & save
fname = os.path.join(results_dir, 'bem.nc')
if os.path.exists(fname):
    wec.read_bem(fname)
else:
    wec.run_bem()
    wec.write_bem(fname)

## CREATE REGULAR WAVE
wfreq = 0.3
amplitude = 0.0625
phase = -40
waves = wot.waves.regular_wave(f0, nfreq, wfreq, amplitude, phase)

## SOLVE
options = {'maxiter': 1000, 'ftol': 1e-8}
obj_fun = pto.energy
nstate_opt = pto.nstate
maximize = True
wec_tdom, wec_fdom, x_wec, x_opt, obj, res = wec.solve(
    waves, obj_fun, nstate_opt, optim_options=options, maximize=maximize)

## POST-PROCESS PTO
pto_tdom, pto_fdom = pto.post_process(wec, x_wec, x_opt)


########################################################################
## SAVE, VIEW, & PLOT RESULTS
########################################################################
# print
print(f"\nEnergy produced from 0-{wec.tf}s: {obj} J")
print(f"Average power: {obj/wec.tf} W\n")

# save
fname = os.path.join(results_dir, 'wec_tdom.nc')
wec_tdom.to_netcdf(fname)

fname = os.path.join(results_dir, 'wec_fdom.nc')
wot.complex_xarray_to_netcdf(fname, wec_fdom)

fname = os.path.join(results_dir, 'pto_tdom.nc')
pto_tdom.to_netcdf(fname)

fname = os.path.join(results_dir, 'pto_fdom.nc')
wot.complex_xarray_to_netcdf(fname, pto_fdom)

# view mesh
wec.fb.show()
wec.fb.show_matplotlib()

# plot impedance
wec.plot_impedance() # see function doc for options.

# example time domain plots
plt.figure()
wec_tdom['wave_elevation'].plot()

plt.figure()
wec_tdom['pos'].plot()

plt.figure()
pto_tdom['force'].plot()

plt.figure()
pto_tdom['power'].plot()

# example frequency domain plots
fd_lines = {'marker': 'o', 'linestyle': '', 'fillstyle': 'none'}

plt.figure()
np.abs(wec_fdom['excitation_force']).plot(**fd_lines)

plt.figure()
np.abs(pto_fdom['force']).plot(**fd_lines)

plt.show()


########################################################################
## COMPARE TO CONJUGATE GRADIENT CONTROL (UNCONSTRAINED)
########################################################################
idof = 0
Fe = wec_fdom['excitation_force'][1:, idof]
Zi = wec.hydro.Zi[:, idof, idof]

cc_vel_fd = Fe / (2*Zi.real)
cc_pos_fd = cc_vel_fd / (1j*cc_vel_fd.omega)
cc_force_fd = -1.0 * Zi.conj() * cc_vel_fd

cc_pos_fd = np.concatenate([[0.0], cc_pos_fd])
cc_vel_fd = np.concatenate([[0.0], cc_vel_fd])
cc_force_fd = np.concatenate([[0.0], cc_force_fd])

cc_pos_td = wec.fd_to_td(cc_pos_fd)
cc_vel_td = wec.fd_to_td(cc_vel_fd)
cc_force_td = wec.fd_to_td(cc_force_fd)

cc_power_td = -cc_vel_td * cc_force_td
cc_power_fd = wec.td_to_fd(cc_power_td)

Fe = wec_fdom['excitation_force'][:, idof]

# plot
ncases = 2

label_cc = 'CC'
label_ps = 'PS'
color_cc = 'tab:blue'
color_ps = 'tab:orange'
color_w = '0.0'
color_s = '0.25'
lw = 2

## Plot time domain
fig, axs = plt.subplots(nrows=6, sharex=True)

# plot PS force limits
xlim = 10.0
axs[4].plot([0.0, xlim], [f_max, f_max], '--', c=color_s, lw=0.5)
axs[4].plot([0.0, xlim], [-f_max, -f_max], '--', c=color_s, lw=0.5)

wec_tdom['wave_elevation'].plot(ax=axs[0], color=color_w, lw=lw)
wec_tdom['excitation_force'].sel(influenced_dof='HEAVE').plot(
    ax=axs[1], color=color_w, lw=lw)
axs[2].plot(wec.time, cc_pos_td, color=color_cc, label=label_cc, lw=lw)
wec_tdom['pos'].sel(influenced_dof='HEAVE').plot(
    ax=axs[2], color=color_ps, label=label_ps, lw=lw)
axs[3].plot(wec.time, cc_vel_td, color=color_cc, label=label_cc, lw=lw)
wec_tdom['vel'].sel(influenced_dof='HEAVE').plot(
    ax=axs[3], color=color_ps, label=label_ps, lw=lw)
axs[4].plot(wec.time, cc_force_td, color=color_cc, label=label_cc, lw=lw)
pto_tdom['force'].sel(dof_pto='pto_1').plot(
    ax=axs[4], color=color_ps, label=label_ps, lw=lw)  # marker='.'
axs[5].plot(wec.time, cc_power_td, color=color_cc, label=label_cc, lw=lw)
pto_tdom['power'].sel(dof_pto='pto_1').plot(
    ax=axs[5], color=color_ps, label=label_ps, lw=lw)

# format subplots
ylims = [0.05, 1000.0, 0.2, 0.5, 5000.0, 500.0]
names = ['η [m]', 'Fₑ [N]', 'z [m]', 'u [m/s]',
         'Fᵤ [N]', 'P [W]']
for ax, ylim, name in zip(axs, ylims, names):
    ax.set_title('')
    if ax is not axs[-1]:
        ax.set_xlabel('')
    ax.set_ylabel(name)
    ax.label_outer()
    ax.set_xticks([i for i in range(int(xlim)+1)], minor=False)
    ax.grid(color='0.75', linestyle='-', linewidth=0.5, which='major')
    ax.tick_params(direction='in')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlim([0.0, xlim])
    ax.set_yticks([-2*ylim, -ylim, 0, ylim], minor=False)
axs[2].legend(ncol=ncases, loc='upper right')

fig.align_ylabels(axs)
fig.tight_layout()


## Plot frequency domain
omega = np.concatenate([[0.0], wec.omega])
cases = [label_cc, label_ps]

fig, axs = plt.subplots(2, ncases, sharex='col', sharey='row')


def plot_fd(axs, omega, fdom, marker, label, rmface=False):

    def _plot_fd(ax, omega, fdom, marker, label, rmface=False):
        markers, stems, base = ax.stem(
            omega,
            fdom,
            basefmt='k-',
            linefmt='b-',
            markerfmt='b'+marker,
            label=label)
        width = 1.0
        stems.set_linewidth(width)
        base.set_linewidth(width)
        markers.set_markeredgewidth(width)
        markers.set_markersize(10)
        if rmface:
            markers.set_markerfacecolor('none')

    omega = omega / (wfreq*2*np.pi)
    mag = np.squeeze(20*np.log10(np.abs(fdom),
                     out=np.ones(fdom.shape)*np.nan, where=fdom != 0))
    ang = np.squeeze(np.angle(fdom))

    _plot_fd(axs[0], omega, mag, marker, label, rmface)
    _plot_fd(axs[1], omega, ang, marker, label, rmface)


iaxs = axs[:, 0]
plot_fd(iaxs, omega, Fe, 'o', 'Fₑ', True)
plot_fd(iaxs, omega, cc_vel_fd, '.', 'u')
plot_fd(iaxs, omega, cc_force_fd, '_', 'Fᵤ')

iaxs = axs[:, 1]
plot_fd(iaxs, omega, Fe, 'o', 'Fₑ', True)
plot_fd(iaxs, omega, wec_fdom['vel'], '.', 'u')
plot_fd(iaxs, omega, pto_fdom['force'], '_', 'Fᵤ')

# format subplots
locs = [1, 3, 5, 7]
ylims = [100.0, np.pi]
xlims = [0, omega[-1]/(wfreq*2*np.pi)]
for i in range(ncases):
    iaxs = axs if ncases == 1 else axs[:, i]
    for j in range(2):
        iaxs[j].set_xticks([0]+locs, minor=False)
        iaxs[j].set_yticks([-ylims[j], 0, ylims[j]], minor=False)
        iaxs[j].label_outer()
        iaxs[j].grid(color='0.75', linestyle='-',
                     linewidth=0.5, which='major')
        iaxs[j].tick_params(direction='in')
        iaxs[j].spines['right'].set_visible(False)
        iaxs[j].spines['top'].set_visible(False)
        iaxs[j].set_xlim(xlims)
        iaxs[j].set_xticklabels(['0']+[f'{k} ω₀' for k in locs])
    iaxs[1].set_yticklabels(["-π", 0, "π"], minor=False)
    iaxs[0].set_title(cases[i])
    iaxs[1].set_xlabel('Frequency [rad/s]')
    iaxs[1].set_ylim([-np.pi, np.pi])
    if i == 0:
        iaxs[0].legend(ncol=1, loc='upper right')
        iaxs[0].set_ylabel('Magnitude [dB]')
        iaxs[1].set_ylabel('Angle [rad]')
        fig.align_ylabels(iaxs)

fig.tight_layout()


plt.show()
