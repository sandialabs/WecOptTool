#!/usr/bin/env python3

import logging
import os

import autograd.numpy as np
from autograd.builtins import isinstance, tuple, list, dict
import capytaine as cpy
import matplotlib.pyplot as plt
import pygmsh
import gmsh

import wecopttool as wot


logging.basicConfig(level=logging.INFO)

#  mesh
h1 = 0.17
h2 = 0.37
r1 = 0.88
r2 = 0.35
freeboard = 0.01
mesh_size_factor = 0.1

with pygmsh.occ.Geometry() as geom:
    gmsh.option.setNumber('Mesh.MeshSizeFactor', mesh_size_factor)
    cyl = geom.add_cylinder([0, 0, 0], [0, 0, -h1], r1)
    cone = geom.add_cone([0, 0, -h1], [0, 0, -h2], r1, r2)
    geom.translate(cyl, [0, 0, freeboard])
    geom.translate(cone, [0, 0, freeboard])
    geom.boolean_union([cyl, cone])
    mesh = geom.generate_mesh()

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
pto_ps = wot.pto.PseudoSpectralPTO(nfreq, kinematics)
pto_p = wot.pto.ProportionalPTO(kinematics)

# constraints at 4x points
nsubsteps = 4
f_max = 2000.0


def const_f_pto(wec, x_wec, x_opt):
    f = pto_ps.force_on_wec(wec, x_wec, x_opt, nsubsteps)
    return f_max - np.abs(f.flatten())


ineq_cons = {'type': 'ineq',
             'fun': const_f_pto,
             }
constraints = [ineq_cons]

# additional friction
dissipation = 160.3

# create WECs
f_add_ps = pto_ps.force_on_wec
f_add_p = pto_p.force_on_wec

wec_ps = wot.WEC(fb, mass, stiffness, f0, nfreq, rho=rho, f_add=f_add_ps,
                 constraints=constraints, dissipation=dissipation)

wec_cc = wot.WEC(fb, mass, stiffness, f0, nfreq, rho=rho,
                 f_add=f_add_ps, dissipation=dissipation)

wec_p = wot.WEC(fb, mass, stiffness, f0, nfreq, rho=rho,
                f_add=f_add_p, dissipation=dissipation)

# create save directory
results_dir = 'results_validation_a'
if not os.path.exists(results_dir):
  os.makedirs(results_dir)

# read BEM if available, else run & save
fname = os.path.join(results_dir, 'bem.nc')
if os.path.exists(fname):
    wec_ps.read_bem(fname)
else:
    wec_ps.run_bem()
    wec_ps.write_bem(fname)
wec_cc.read_bem(fname)
wec_p.read_bem(fname)

# wave
wfreq = 0.3
amplitude = 0.0625
phase = -40
waves = wot.waves.regular_wave(f0, nfreq, wfreq, amplitude, phase)

# solve
options = {'maxiter': 1000, 'ftol': 1e-8}
obj_fun_ps = pto_ps.average_power
nstate_opt_ps = pto_ps.nstate
obj_fun_p = pto_p.average_power
nstate_opt_p = pto_p.nstate

wec_tdom_cc, wec_fdom_cc, x_wec_cc, x_opt_cc, obj_cc, _ = wec_cc.solve(
    waves, obj_fun_ps, nstate_opt_ps, optim_options=options)
print("\nCC-controller:")
print(f"    Average power: {obj_cc} W")
print("    MATLAB: -121 W\n")

wec_tdom_ps, wec_fdom_ps, x_wec_ps, x_opt_ps, obj_ps, _ = wec_ps.solve(
    waves, obj_fun_ps, nstate_opt_ps, optim_options=options)
print("\nPS-controller:")
print(f"    Average power: {obj_ps} W")
print("    MATLAB: -97 W\n")

wec_tdom_p, wec_fdom_p, x_wec_p, x_opt_p, obj_p, _ = wec_p.solve(
    waves, obj_fun_p, nstate_opt_p, optim_options=options)
print("\nP-controller:")
print(f"    Average power: {obj_p} W")
print("    MATLAB: -28 W\n")

# post-process PTO
pto_tdom_ps, pto_fdom_ps = pto_ps.post_process(wec_ps, x_wec_ps, x_opt_ps)
pto_tdom_cc, pto_fdom_cc = pto_ps.post_process(wec_cc, x_wec_cc, x_opt_cc)
pto_tdom_p, pto_fdom_p = pto_p.post_process(wec_p, x_wec_p, x_opt_p)

# # theoretical conjugate gradient
idof = 0
Fe = wec_fdom_cc['excitation_force'][1:, idof]
Zi = wec_cc.hydro.Zi[:, idof, idof]

cct_vel_fd = Fe / (2*Zi.real)
cct_pos_fd = cct_vel_fd / (1j*cct_vel_fd.omega)
cct_force_fd = -Zi.conj() * cct_vel_fd

cct_pos_fd = np.concatenate([[0.0], cct_pos_fd])
cct_vel_fd = np.concatenate([[0.0], cct_vel_fd])
cct_force_fd = np.concatenate([[0.0], cct_force_fd])

cct_pos_td = wec_cc.fd_to_td(cct_pos_fd)
cct_vel_td = wec_cc.fd_to_td(cct_vel_fd)
cct_force_td = wec_cc.fd_to_td(cct_force_fd)

cct_power_td = cct_vel_td * cct_force_td
cct_power_fd = wec_cc.td_to_fd(cct_power_td)

Fe = wec_fdom_cc['excitation_force'][:, idof]

# plot
plt.rcParams.update({"text.usetex": True, })

td_wecs = [wec_tdom_cc, wec_tdom_ps, wec_tdom_p]
td_ptos = [pto_tdom_cc, pto_tdom_ps, pto_tdom_p]
colors = ['tab:blue', 'tab:orange', 'tab:red']
labels = ['CC', 'PS', 'P']

label_cct = 'CC(th)'
color_cct = '0.0'

color_s = '0.25'
lw = 2
ncol_leg = 2

## Plot time domain
fig, axs = plt.subplots(nrows=6, sharex=True)

# plot PS force limits
xlim = 10.0
axs[4].plot([0.0, xlim], [f_max, f_max], '--', c=color_s, lw=0.5)
axs[4].plot([0.0, xlim], [-f_max, -f_max], '--', c=color_s, lw=0.5)

# plot wave elevation and excitation
wec_tdom_ps['wave_elevation'].plot(ax=axs[0], color=color_cct, lw=lw)
wec_tdom_ps['excitation_force'].sel(influenced_dof='HEAVE').plot(
    ax=axs[1], color=color_cct, lw=lw)

# plot cc theoretical
axs[2].plot(wec_ps.time, cct_pos_td, color=color_cct,
            label=label_cct, lw=lw*1.5)
axs[3].plot(wec_ps.time, cct_vel_td, color=color_cct,
            label=label_cct, lw=lw*1.5)
axs[4].plot(wec_ps.time, cct_force_td, color=color_cct,
            label=label_cct, lw=lw*1.5)
axs[5].plot(wec_ps.time, cct_power_td, color=color_cct,
            label=label_cct, lw=lw*1.5)

# plot WOT cases
for tdom_wec, tdom_pto, color, label in zip(td_wecs, td_ptos, colors, labels):
    tdom_wec['pos'].sel(influenced_dof='HEAVE').plot(
        ax=axs[2], color=color, label=label, lw=lw)
    tdom_wec['vel'].sel(influenced_dof='HEAVE').plot(
        ax=axs[3], color=color, label=label, lw=lw)
    tdom_pto['force'].sel(dof_pto='pto_1').plot(
        ax=axs[4], color=color, label=label, lw=lw)
    tdom_pto['power'].sel(dof_pto='pto_1').plot(
        ax=axs[5], color=color, label=label, lw=lw)

# format subplots
ylims = [0.05, 1000.0, 0.2, 0.5, 5000.0, 500.0]
names = ['$\eta$ [$m$]', '$F_e$ [$N$]', '$z$ [$m$]', '$u$ [$m/s$]',
         '$F_u$ [$N$]', '$P$ [$W$]']
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
axs[2].legend(ncol=ncol_leg, loc='upper right')

fig.align_ylabels(axs)
fig.tight_layout()

## Plot frequency domain
ncases = len(labels)
omega = np.concatenate([[0.0], wec_ps.omega])

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
        marker_size = 10.0
        stems.set_linewidth(width)
        base.set_linewidth(width)
        markers.set_markeredgewidth(width)
        markers.set_markersize(marker_size)
        if rmface:
            markers.set_markerfacecolor('none')

    omega = omega / (wfreq*2*np.pi)
    mag = np.squeeze(20*np.log10(np.abs(fdom),
                     out=np.ones(fdom.shape)*np.nan, where=fdom!=0))
    ang = np.squeeze(np.angle(fdom))

    _plot_fd(axs[0], omega, mag, marker, label, rmface)
    _plot_fd(axs[1], omega, ang, marker, label, rmface)


fd_wecs = [wec_fdom_cc, wec_fdom_ps, wec_fdom_p]
fd_ptos = [pto_fdom_cc, pto_fdom_ps, pto_fdom_p]


for i in range(ncases):
    iaxs = axs[:, i]
    plot_fd(iaxs, omega, Fe, 'o', '$F_e$', True)
    plot_fd(iaxs, omega, fd_wecs[i]['vel'], '.', '$u$')
    plot_fd(iaxs, omega, fd_ptos[i]['force'], '_', '$F_u$')

# theoretical CC
iaxs = axs[:, 0]
plot_fd(iaxs, omega, cct_vel_fd, 'D', '$u_{(th)}$', True)
plot_fd(iaxs, omega, cct_force_fd, 'x', '$F_{u,(th)}$')

# format subplots
locs = [1, 3, 5, 7]
ylims = [100.0, np.pi]
xlims = [0, omega[-1]/(wfreq*2*np.pi)]
for i in range(ncases):
    iaxs = axs[:, i]
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
        iaxs[j].set_xticklabels(['0']+[f'${k} \omega_0$' for k in locs])
    iaxs[1].set_yticklabels(["$-\pi$", 0, "$\pi$"], minor=False)
    iaxs[0].set_title(labels[i])
    iaxs[1].set_xlabel('Frequency [rad/s]')
    iaxs[1].set_ylim([-np.pi, np.pi])
    if i == 0:
        iaxs[0].legend(ncol=1, loc='upper right')
        iaxs[0].set_ylabel('Magnitude [dB]')
        iaxs[1].set_ylabel('Angle [rad]')
        fig.align_ylabels(iaxs)

fig.tight_layout()

plt.show()
