#!/usr/bin/env python3

# water properties
rho = 1e3

# frequency array
f0 = 0.05
num_freq = 50

# wec
mesh_size_factor = 0.1

# wave
freq = 0.3
amplitude = 0.0625
phase = -40


if __name__ == "__main__":
    import os
    import logging

    import numpy as np
    import capytaine as cpy

    import wecopttool as wot
    from wecopttool.examples import wavebot

    logging.basicConfig(level=logging.INFO)

    # create results directory
    data_dir = 'data'
    # TODO: if not exist (Windows)
    os.makedirs(data_dir)

    # mesh
    # TODO: Capytaine fb from meshio/pygmsh mesh (Issues #13, #62)
    mesh_file = os.path.join(data_dir, 'mesh.stl')
    mesh = wavebot.mesh(mesh_size_factor=mesh_size_factor)
    mesh.write(mesh_file)

    # create Capytaine floating body (mesh + DOFs)
    fb = cpy.FloatingBody.from_file(mesh_file, name='WaveBot')
    fb.add_translation_dof(name="Heave")

    # mass and hydrostatic stiffness
    hs_data = wot.hydrostatics.hydrostatics(fb, rho=rho)
    mass_33 = wot.hydrostatics.mass_matrix_constant_density(hs_data)[2, 2]
    mass = np.atleast_2d(mass_33)
    stiffness_33 = wot.hydrostatics.stiffness_matrix(hs_data)[2, 2]
    stiffness = np.atleast_2d(stiffness_33)
    np.savetxt(os.path.join(data_dir, 'mass_matrix'), mass)
    np.savetxt(os.path.join(data_dir, 'hydrostatic_stiffness'), stiffness)

    # create WEC
    wec = wot.WEC(fb, mass, stiffness, f0, num_freq, rho=rho)

    # run BEM
    wec.run_bem()
    wec.write_bem(os.path.join(data_dir, 'bem.nc'))

    # plot BEM
    fig1, axs1 = wec.plot_impedance(style='Bode', show=False)
    fig2, axs2 = wec.plot_impedance(style='complex', show=False)
    fig1.savefig(os.path.join(data_dir, 'impedance_bode.pdf'))
    fig2.savefig(os.path.join(data_dir, 'impedance_complex.pdf'))

    # wave
    waves = wot.waves.regular_wave(f0, num_freq, freq, amplitude, phase)
    waves.to_netcdf(os.path.join(data_dir, 'waves.nc'))
