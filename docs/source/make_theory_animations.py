# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


import imageio.v3 as iio
from pygifsicle import optimize
import os
import tempfile

# get relative paths
path_to_current_file = os.path.realpath(__file__)
current_directory = os.path.dirname(path_to_current_file)
odir = os.path.join(current_directory, "..", "_build", "html", "_static")

if not os.path.exists(odir):
    os.makedirs(odir)

gif_name = 'theory_animation'

N = 30  # number of frames
M = 10  # number of times to repeat final frame
# will produce M+N total frames

with tempfile.TemporaryDirectory() as tmpdirname:

    fig, ax = plt.subplots()
    ax.set_xlim(0, 2*np.pi)
    ax.set_ylim(-1, 1)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.xticks(color='w')
    plt.yticks(color='w')

    xx = np.linspace(0, 2*np.pi, N)
    yy = 0.5*np.sin(xx)
    ax.plot(xx, yy, 'k--',
            label='correct solution')

    xdata, ydata = [], []
    ln, = ax.plot([], [], 'ro',
                  markersize=5,
                  markerfacecolor='none',
                  label='time-stepping solution',
                  )

    ax.legend()

    td_fnames = []
    fig.tight_layout()

    for ind, (xd, yd) in enumerate(zip(xx, yy)):
        xdata.append(xd)
        ydata.append(yd)
        ln.set_data(xdata, ydata)
        fname = os.path.join(tmpdirname, f"{gif_name}_td_{ind}.png")
        fig.savefig(fname)
        td_fnames.append(fname)

    plt.close(fig)

    ps_fnames = []

    for ind in range(N):

        fig, ax = plt.subplots()
        ax.set_xlim(0, 2*np.pi)
        ax.set_ylim(-1, 1)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.xticks(color='w')
        plt.yticks(color='w')

        ax.plot(xx, yy, 'k--',
                label='correct solution')

        ax.plot(xx, yy*(ind+1)/N, 'bs',
                markersize=5,
                markerfacecolor='none',
                label='pseudo-spectral solution',
                )

        ax.legend()

        fname = os.path.join(tmpdirname, f"{gif_name}_ps_{ind}.png")
        fig.tight_layout()
        fig.savefig(fname)
        ps_fnames.append(fname)
        plt.close(fig)

    fnames = {
        'td': td_fnames,
        'ps': ps_fnames,
    }

    for key in fnames:
        # repeat last frame M times
        fnames[key] = fnames[key] + [fnames[key][-1]]*M

        frames = np.stack([iio.imread(filename) for filename in fnames[key]],
                          axis=0)
        gif_path = os.path.join(odir, f"{gif_name}_{key}.gif")
        iio.imwrite(gif_path, frames)
        optimize(gif_path)
