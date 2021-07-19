#!/usr/bin/env python3

import os

import matplotlib.pyplot as plt
import xarray as xr
import numpy as np

import WecOptTool as wot


plt.rcParams.update({"text.usetex": True,})

# cases
cases = ['CC', 'P', 'PS']
colors = ['tab:blue', 'tab:red', 'tab:orange']
ncases = len(cases)

# load data
results_dir = 'results'
TD_all = []
FD_all = []
for case in cases:
    td_file = os.path.join(results_dir, f'TD_{case}.nc')
    TD_all.append(xr.load_dataset(td_file))
    fd_file = os.path.join(results_dir, f'FD_{case}.nc')
    FD_all.append(wot.from_netcdf(fd_file))

# Plot time domain
fig, axs = plt.subplots(nrows=6, sharex=True)

for TD, color, case in zip(TD_all, colors, cases):
    TD['wave_elevation'].plot(ax=axs[0], color=color, label=case)
    TD['excitation_force'].sel(influenced_dof='Heave').plot(
        ax=axs[1], color=color, label=case)
    TD['pos'].sel(influenced_dof='Heave').plot(
        ax=axs[2], color=color, label=case)
    TD['vel'].sel(influenced_dof='Heave').plot(
        ax=axs[3], color=color, label=case)
    TD['pto_force'].sel(dof_pto='pto_1').plot(
        ax=axs[4], color=color, label=case)
    TD['power'].sel(dof_pto='pto_1').plot(ax=axs[5], color=color, label=case)

# format subplots
xlim = 10.0
ylims = [0.05, 1000.0, 0.2, 0.5, 5000.0, 500.0]
names = ['$\eta$ [$m$]', '$F_e$ [$N$]', '$z$ [$m$]', '$u$ [$m/s$]',
         '$F_u$ [$N$]', '$P$ [$W$]']
for ax, ylim, name in zip(axs, ylims, names):
    ax.set_title('')
    if ax is not axs[-1]:
        ax.set_xlabel('')
    ax.set_ylabel(name)
    ax.label_outer()
    ax.set_xticks([i for i in range(11)], minor=False)
    ax.grid(color='0.75', linestyle='-', linewidth=0.5, which='major')
    ax.tick_params(direction='in')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlim([0.0, xlim])
    ax.set_yticks([-2*ylim, -ylim, 0, ylim], minor=False)
axs[0].legend(ncol=ncases, loc='upper right')

fig.align_ylabels(ax)
fig.tight_layout()

# Plot frequency domain
fig, axs = plt.subplots(2, ncases, sharex='col', sharey='row')

def plot_FD(axs, FD, marker, label, rmface=False):

    def _plot_FD(ax, omega, FD, marker, label, rmface=False):
        markers, stems, base = ax.stem(
            omega,
            FD,
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

    omega = FD.omega
    mag = np.squeeze(20*np.log10(np.abs(FD)))
    ang = np.squeeze(np.angle(FD))

    _plot_FD(axs[0], omega, mag, marker, label, rmface)
    _plot_FD(axs[1], omega, ang, marker, label, rmface)

for i, FD in enumerate(FD_all):
    iaxs = axs if ncases == 1 else axs[:, i]
    plot_FD(iaxs, FD['excitation_force'], 'o', '$F_e$', True)
    plot_FD(iaxs, FD['vel'], '.', '$u$')
    plot_FD(iaxs, FD['pto_force'], '_', '$F_u$')

# format subplots
locs = [1, 3, 5, 7]
ylims = [100.0, 2.0]
xlims = [0, 8]
for i in range(ncases):
    iaxs = axs if ncases == 1 else axs[:, i]
    for j in range(2):
        iaxs[j].set_xticks(locs, minor=False)
        iaxs[j].set_yticks([-ylims[j], 0, ylims[j]], minor=False)
        iaxs[j].label_outer()
        iaxs[j].grid(color='0.75', linestyle='-',
                       linewidth=0.5, which='major')
        iaxs[j].tick_params(direction='in')
        iaxs[j].spines['right'].set_visible(False)
        iaxs[j].spines['top'].set_visible(False)
        iaxs[j].set_xlim(xlims)
        iaxs[j].set_xticklabels([f'${k} \omega_0$' for k in locs])
    iaxs[0].set_title(cases[i])
    iaxs[1].set_xlabel('Frequency [rad/s]')
    iaxs[1].set_ylim([-np.pi, np.pi])
    if i == 0:
        iaxs[0].legend(ncol=1, loc='upper right')
        iaxs[0].set_ylabel('Magnitude [dB]')
        iaxs[1].set_ylabel('Angle [rad]')
        fig.align_ylabels(iaxs)

fig.tight_layout()

# show
plt.show()
