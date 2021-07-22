#!/usr/bin/env python3

# water properties
rho = 1e3

# frequency array
f0 = 0.05
num_freq = 30  # 48 # TODO

# wec
mesh_size_factor = 0.25

# wave
freq = 0.3
amplitude = 0.0625
phase = -40


if __name__ == "__main__":
    import os
    import logging

    import numpy as np
    import capytaine as cpy

    import WecOptTool as wot
    from WecOptTool.examples import WaveBot


    logging.basicConfig(level=logging.INFO)

    # create results directory
    data_dir = 'data'
    os.makedirs(data_dir)

    # mesh
    # TODO: Capytaine fb from meshio/pygmsh mesh (Issues #13, #62)
    mesh_file = os.path.join(data_dir, 'mesh.stl')
    mesh = WaveBot.mesh(mesh_size_factor=mesh_size_factor)
    mesh.write(mesh_file)

    # create Capytaine floating body (mesh + DOFs)
    fb = cpy.FloatingBody.from_file(mesh_file, name='WaveBot')
    fb.add_translation_dof(name="Heave")

    # mass and hydrostatic stiffness
    hs_data = wot.hydrostatics.hydrostatics(fb, rho=rho)
    M33 = wot.hydrostatics.mass_matrix_constant_density(hs_data)[2, 2]
    M = np.atleast_2d(M33)
    K33 = wot.hydrostatics.stiffness_matrix(hs_data)[2, 2]
    K = np.atleast_2d(K33)
    np.savetxt(os.path.join(data_dir, 'mass_matrix'), M)
    np.savetxt(os.path.join(data_dir, 'hydrostatic_stiffness'), K)

    # create WEC
    wec = wot.WEC(fb, M, K, f0, num_freq, rho=rho)

    # run BEM
    wec.run_bem()
    wec.write_bem(os.path.join(data_dir, 'BEM.nc'))

    # wave
    waves = wot.waves.regular_wave(f0, num_freq, freq, amplitude, phase)
    waves.to_netcdf(os.path.join(data_dir, 'waves.nc'))
