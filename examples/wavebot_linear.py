# WaveBot Linear Model

import autograd.numpy as np
import capytaine as cpy
from capytaine.io.meshio import load_from_meshio
import matplotlib.pyplot as plt
plt.style.use('tableau-colorblind10')

import wecopttool as wot

###############################################################################
## 1. Setup
###############################################################################
## Frequency
wavefreq = 0.3 # Hz
f1 = wavefreq
nfreq = 10
freq = wot.frequency(f1, nfreq, False) # False -> no zero frequency

## Waves
amplitude = 0.0625 # m
phase = 30 # degrees
wavedir = 0 # degrees

waves = wot.waves.regular_wave(f1, nfreq, wavefreq, amplitude, phase, wavedir)

## BEM
# mesh
wb = wot.geom.WaveBot()  # use standard dimensions
mesh_size_factor = 0.2 # 1.0 for default, smaller to refine mesh
mesh = wb.mesh(mesh_size_factor)
mesh_obj = load_from_meshio(mesh, 'WaveBot')
lid_mesh = mesh_obj.generate_lid(-2e-2)
fb = cpy.FloatingBody(mesh=mesh_obj, lid_mesh=lid_mesh, name="WaveBot")

# DOFs
fb.add_translation_dof(name="Heave")
ndof = fb.nb_dofs

# run BEM
bem_data = wot.run_bem(fb, freq)
wot.write_netcdf('bem_data.nc', bem_data)

# fix BEM
bem_data = wot.add_linear_friction(bem_data, friction = None)
bem_data = wot.check_radiation_damping(bem_data)

## PTO
# PTO impedance
omega = bem_data.omega.values
gear_ratio = 12.0
torque_constant = 6.7
winding_resistance = 0.5#*1e-5 #TODO
winding_inductance = 0.0
drivetrain_inertia = 2.0#*1e-5 #TODO
drivetrain_friction = 1.0#*1e-5 #TODO
drivetrain_stiffness = 0.0

drivetrain_impedance = (1j*omega*drivetrain_inertia +
                        drivetrain_friction +
                        1/(1j*omega)*drivetrain_stiffness)

winding_impedance = winding_resistance + 1j*omega*winding_inductance


pto_impedance_11 = -1* gear_ratio**2 * drivetrain_impedance
off_diag = np.sqrt(3.0/2.0) * torque_constant * gear_ratio
pto_impedance_12 = -1*(off_diag+0j) * np.ones(omega.shape)
pto_impedance_21 = -1*(off_diag+0j) * np.ones(omega.shape)
pto_impedance_22 = winding_impedance
pto_impedance = np.array([
    [pto_impedance_11, pto_impedance_12],
    [pto_impedance_21, pto_impedance_22],
])

# kinematic matrix
kinematics = np.eye(ndof)

###############################################################################
## 2. Frequency Domain Solver
###############################################################################
wave_realization = waves.isel(realization=0)
p_opt_average, tdom, fdom, thevenin = wot.utilities.linear_solve(bem_data, pto_impedance, wave_realization, kinematics, nsubsteps=5)
print("Average power: ", p_opt_average, "W")

###############################################################################
## 3. Time Domain:WecOptTool
###############################################################################
# PTO
pto_names = ['PTO_Heave',]
# kinematics = np.eye(ndof)
controller = None
pto_loss = None

pto = wot.pto.PTO(ndof, kinematics, controller, pto_impedance, pto_loss, pto_names)

# additional forces
friction = None
f_add = {'PTO': pto.force_on_wec}

# constraints
constraints = None

# WEC object
wec = wot.WEC.from_bem(
    bem_data,
    constraints=constraints,
    friction=friction,
    f_add=f_add
)

# objective function
obj_fun = pto.average_power
nstate_opt = 2*nfreq

# solve
scale_x_wec = 1e1
scale_x_opt = 1e-3
scale_obj = 1e-2

results = wec.solve(
    waves,
    obj_fun,
    nstate_opt,
    scale_x_wec=scale_x_wec,
    scale_x_opt=scale_x_opt,
    scale_obj=scale_obj,
)
opt_average_power = results[0].fun
print(f'Optimal average electrical power: {opt_average_power} W')

# post-process
nsubsteps = 5
wec_fdom, wec_tdom = wec.post_process(wec, results, waves, nsubsteps)
pto_fdom, pto_tdom = pto.post_process(wec, results, waves, nsubsteps)

###############################################################################
## 4. Comparison Plots
###############################################################################
fig, ax = plt.subplots()
ax.plot(tdom['time'], tdom['power'])
pto_tdom[0]['power'].sel(type='elec').plot(ax=ax, ls="--")
plt.title("Power")

fig, ax = plt.subplots()
ax.plot(tdom['time'], tdom['trans_eff'])
pto_tdom[0]['trans_eff'].plot(ax=ax, ls="--")
plt.title("Voltage")

fig, ax = plt.subplots()
ax.plot(tdom['time'], tdom['trans_flo'])
pto_tdom[0]['trans_flo'].plot(ax=ax, ls="--")
plt.title("Current")
