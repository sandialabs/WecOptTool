#!/usr/bin/env python3

import os
import logging

import autograd.numpy as np
from autograd.builtins import isinstance, tuple, list, dict
import capytaine as cpy
import xarray as xr

import wecopttool as wot
from preprocess import rho, f0, num_freq


logging.basicConfig(level=logging.WARNING)

# I/O
data_dir = 'data'
results_dir = 'results'
case = 'ps'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Capytaine floating body
mesh_file = os.path.join(data_dir, 'mesh.stl')
fb = cpy.FloatingBody.from_file(mesh_file, name='WaveBot')
fb.add_translation_dof(name="Heave")

# mass and hydrostatic stiffness
mass = np.atleast_2d(np.loadtxt(os.path.join(data_dir, 'mass_matrix')))
stiffness = np.atleast_2d(
    np.loadtxt(os.path.join(data_dir, 'hydrostatic_stiffness')))

# PTO: state, force, power (objective function)
kinematics = np.eye(fb.nb_dofs)
num_x_pto, f_pto, power_pto, pto_postproc = \
    wot.pto.pseudospectral_pto(num_freq, kinematics)

# create WEC
wec = wot.WEC(fb, mass, stiffness, f0, num_freq, f_add=f_pto, rho=rho)

# read BEM
bem_file = os.path.join(data_dir, 'bem.nc')
wec.read_bem(bem_file)

# wave
waves = xr.open_dataset(os.path.join(data_dir, 'waves.nc'))

# Scale
scale_wec = [1.0]
scale_opt = 1e6
scale_obj = 1.0

# Constraints
f_max = 2000.0

scale = wec.scale(scale_wec, scale_opt, num_x_pto)


def const_f_pto(x):
    xs = x*(1/scale)
    x_wec, x_pto = wec.decompose_decision_var(xs)
    f = np.abs(f_pto(wec, x_wec, x_pto)[0, :])
    return f_max - f


ineq_cons = {'type': 'ineq',
             'fun': const_f_pto,
             }
constraints = [ineq_cons]

# Solve dynamics & opt control
# options = {'maxiter': 10000, 'ftol': 1e-9}  # scipy
options = {}  # ipopt

fdom, tdom, x_opt, res = wec.solve(
    waves, power_pto, num_x_pto,
    constraints=constraints, optim_options=options,
    scale_x_wec=scale_wec, scale_x_opt=scale_opt, scale_obj=scale_obj)
# post-process: PTO
tdom, fdom = pto_postproc(wec, tdom, fdom, x_opt)

# save
td_file = os.path.join(results_dir, f'tdom_{case}.nc')
tdom.to_netcdf(td_file)

fd_file = os.path.join(results_dir, f'fdom_{case}.nc')
wot.to_netcdf(fd_file, fdom)
