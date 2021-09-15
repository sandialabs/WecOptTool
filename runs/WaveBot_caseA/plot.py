#!/usr/bin/env python3

import os

import matplotlib.pyplot as plt
import xarray as xr
import numpy as np

import wecopttool as wot
from preprocess import freq


plt.rcParams.update({"text.usetex": True, })

# cases
cases = ['cc', 'ps']  # ['cc', 'ps', 'p']
colors = ['tab:blue', 'tab:orange']  # ['tab:blue', 'tab:orange', 'tab:red']
ncases = len(cases)

# load data
results_dir = 'results'
tdom_all = []
fdom_all = []
for case in cases:
    td_file = os.path.join(results_dir, f'tdom_{case}.nc')
    tdom_all.append(xr.load_dataset(td_file))
    fd_file = os.path.join(results_dir, f'fdom_{case}.nc')
    fdom_all.append(wot.from_netcdf(fd_file))

# Plot time domain
fig, axs = plt.subplots(nrows=6, sharex=True)
lw = 2

# plot PS force limits
xlim = 10.0  # tdom.time[-1]
axs[4].plot([0.0, 10.0], [2000, 2000], '--', c='0.25', lw=0.5)
axs[4].plot([0.0, 10.0], [-2000, -2000], '--', c='0.25', lw=0.5)

for tdom, color, case in zip(tdom_all, colors, cases):
    tdom['wave_elevation'].plot(ax=axs[0], color=color, label=case, lw=lw)
    tdom['excitation_force'].sel(influenced_dof='Heave').plot(
        ax=axs[1], color=color, label=case, lw=lw)
    tdom['pos'].sel(influenced_dof='Heave').plot(
        ax=axs[2], color=color, label=case, lw=lw)
    tdom['vel'].sel(influenced_dof='Heave').plot(
        ax=axs[3], color=color, label=case, lw=lw)
    tdom['pto_force'].sel(dof_pto='pto_1').plot(
        ax=axs[4], color=color, label=case, lw=lw) #, marker='o')
    tdom['power'].sel(dof_pto='pto_1').plot(
        ax=axs[5], color=color, label=case, lw=lw)

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
axs[0].legend(ncol=ncases, loc='upper right')

fig.align_ylabels(ax)
fig.tight_layout()


# Plot frequency domain
fig, axs = plt.subplots(2, ncases, sharex='col', sharey='row')

def plot_fd(axs, fdom, marker, label, rmface=False):

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

    omega = fdom.omega / (freq*2*np.pi)
    mag = np.squeeze(20*np.log10(np.abs(fdom)))
    ang = np.squeeze(np.angle(fdom))

    _plot_fd(axs[0], omega, mag, marker, label, rmface)
    _plot_fd(axs[1], omega, ang, marker, label, rmface)


for i, fdom in enumerate(fdom_all):
    iaxs = axs if ncases == 1 else axs[:, i]
    plot_fd(iaxs, fdom['excitation_force'], 'o', '$F_e$', True)
    plot_fd(iaxs, fdom['vel'], '.', '$u$')
    plot_fd(iaxs, fdom['pto_force'], '_', '$F_u$')

# format subplots
locs = [1, 3, 5, 7]
ylims = [100.0, 2.0]
xlims = [0, fdom.omega.values[-1]/(freq*2*np.pi)]
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
        iaxs[j].set_xticklabels(['0']+[f'${k} \omega_0$' for k in locs])
    iaxs[0].set_title(cases[i])
    iaxs[1].set_xlabel('Frequency [rad/s]')
    iaxs[1].set_ylim([-np.pi, np.pi])
    if i == 0:
        iaxs[0].legend(ncol=1, loc='upper right')
        iaxs[0].set_ylabel('Magnitude [dB]')
        iaxs[1].set_ylabel('Angle [rad]')
        fig.align_ylabels(iaxs)

fig.tight_layout()

# # Plot time domain (INTERPOLATED)
# time = np.arange(0, 10, 0.01)
# omega = fdom_all[0].omega.values

# data_all = []
# for tdom, fdom in zip(tdom_all, fdom_all):
#     interp = {}
#     we = np.zeros(time.shape)
#     ef = np.zeros(time.shape)
#     x = np.zeros(time.shape)
#     v = np.zeros(time.shape)
#     f = np.zeros(time.shape)
#     p = np.zeros(time.shape)
#     for i in range(len(omega)):
#         we += np.real(np.squeeze(fdom['wave_elevation'].values)[i] *
#                       np.exp(1j*omega[i]*time))
#         ef += np.real(np.squeeze(fdom['excitation_force'].values)[i] *
#                       np.exp(1j*omega[i]*time))
#         x += np.real(np.squeeze(fdom['pos'].values)[i] *
#                      np.exp(1j*omega[i]*time))
#         v += np.real(np.squeeze(fdom['vel'].values)[i] *
#                      np.exp(1j*omega[i]*time))
#         f += np.real(np.squeeze(fdom['pto_force'].values)[i] *
#                      np.exp(1j*omega[i]*time))
#         p += np.real(
#             np.squeeze(fdom['power'].values * (omega[1]/(2*np.pi)))[i] * np.exp(1j*omega[i]*time))
#     interp['we'] = we
#     interp['ef'] = ef
#     interp['x'] = x
#     interp['v'] = v
#     interp['f'] = f
#     interp['p'] = p
#     data_all.append(interp)

# fig, axs = plt.subplots(nrows=6, sharex=True)

# # plot PS force limits
# xlim = 10.0  # tdom.time[-1]
# axs[4].plot([0.0, 10.0], [2000, 2000], '--', c='0.25', lw=0.5)
# axs[4].plot([0.0, 10.0], [-2000, -2000], '--', c='0.25', lw=0.5)

# for data, color, case in zip(data_all, colors, cases):
#     axs[0].plot(time, data['we'], color=color, label=case, lw=lw)
#     axs[1].plot(time, data['ef'], color=color, label=case, lw=lw)
#     axs[2].plot(time, data['x'], color=color, label=case, lw=lw)
#     axs[3].plot(time, data['v'], color=color, label=case, lw=lw)
#     axs[4].plot(time, data['f'], color=color, label=case, lw=lw)
#     axs[5].plot(time, data['p'], color=color, label=case, lw=lw)

# # time-domain markers
# plot_markers = False
# if plot_markers:
#     for tdom, color, case in zip(tdom_all, colors, cases):
#         tdom['wave_elevation'].plot(
#             ax=axs[0], color=color, label=case, linestyle="None", marker='o')
#         tdom['excitation_force'].sel(influenced_dof='Heave').plot(
#             ax=axs[1], color=color, label=case, linestyle="None", marker='o')
#         tdom['pos'].sel(influenced_dof='Heave').plot(
#             ax=axs[2], color=color, label=case, linestyle="None", marker='o')
#         tdom['vel'].sel(influenced_dof='Heave').plot(
#             ax=axs[3], color=color, label=case, linestyle="None", marker='o')
#         tdom['pto_force'].sel(dof_pto='pto_1').plot(
#             ax=axs[4], color=color, label=case, linestyle="None", marker='o')
#         tdom['power'].sel(dof_pto='pto_1').plot(
#             ax=axs[5], color=color, label=case, linestyle="None", marker='o')

# # format subplots
# xlim = 10.0  # tdom.time[-1]
# ylims = [0.05, 1000.0, 0.2, 0.5, 5000.0, 500.0]
# names = ['$\eta$ [$m$]', '$F_e$ [$N$]', '$z$ [$m$]', '$u$ [$m/s$]',
#          '$F_u$ [$N$]', '$P$ [$W$]']
# for ax, ylim, name in zip(axs, ylims, names):
#     ax.set_title('')
#     if ax is not axs[-1]:
#         ax.set_xlabel('')
#     ax.set_ylabel(name)
#     ax.label_outer()
#     ax.set_xticks([i for i in range(int(xlim)+1)], minor=False)
#     ax.grid(color='0.75', linestyle='-', linewidth=0.5, which='major')
#     ax.tick_params(direction='in')
#     ax.spines['right'].set_visible(False)
#     ax.spines['top'].set_visible(False)
#     ax.set_xlim([0.0, xlim])
#     ax.set_yticks([-2*ylim, -ylim, 0, ylim], minor=False)
# axs[0].legend(ncol=ncases, loc='upper right')

# fig.align_ylabels(ax)
# fig.tight_layout()

# show
# plt.show(block=False); _=input("Press [enter] to close figures.")
plt.show()
