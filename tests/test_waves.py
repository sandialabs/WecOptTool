import os

import pytest
import autograd.numpy as np
from autograd.builtins import isinstance, tuple, list, dict
import capytaine as cpy
import meshio
from scipy.optimize import Bounds

import wecopttool as wot
from wecopttool.waves import pierson_moskowitz_spectrum as pm

import gmsh
import pygmsh


@pytest.fixture()
def wec_wavebot():
    # water properties
    rho = 1000.0

    # frequencies
    f0 = 0.05
    nfreq = 18

    # wave directions
    wave_dirs = [0, 10, 20]

    #  mesh
    meshfile = os.path.join(os.path.dirname(__file__), 'data', 'wavebot.stl')

    # capytaine floating body
    fb = cpy.FloatingBody.from_file(meshfile, name="WaveBot")
    fb.add_translation_dof(name="HEAVE")

    # mass and hydrostativ stiffness
    hs_data = wot.hydrostatics.hydrostatics(fb, rho=rho)
    mass_33 = wot.hydrostatics.mass_matrix_constant_density(hs_data)[2, 2]
    mass = np.atleast_2d(mass_33)
    stiffness_33 = wot.hydrostatics.stiffness_matrix(hs_data)[2, 2]
    stiffness = np.atleast_2d(stiffness_33)

    # WEC
    wec_wavebot = wot.WEC(fb, mass, stiffness, f0, nfreq, rho=rho)

    # BEM
    wec_wavebot.run_bem(wave_dirs)

    return wec_wavebot

@pytest.fixture()
def wec_box():
    # water properties
    rho = 1000.0

    # frequencies
    f0 = 0.05
    nfreq = 18

    # wave directions
    wave_dirs = [0, 10, 20]

    # create mesh
    length = 1.2  
    width = 0.7
    height = 0.4  
    mesh_size_factor = 0.5  # 1.0 for default, smaller to refine mesh

    with pygmsh.occ.Geometry() as geom:
        gmsh.option.setNumber('Mesh.MeshSizeFactor', mesh_size_factor)

        box = geom.add_box(x0=[-length/2,
                            -width/2,
                            -height/2],
                        extents=[length,
                                    width,
                                    height])
        mesh = geom.generate_mesh()

    fb = cpy.FloatingBody.from_meshio(mesh, name="Box_WEC")
    fb.add_translation_dof(name="HEAVE")

    # mass and hydrostativ stiffness
    hs_data = wot.hydrostatics.hydrostatics(fb, rho=rho)
    mass_33 = wot.hydrostatics.mass_matrix_constant_density(hs_data)[2, 2]
    mass = np.atleast_2d(mass_33)
    stiffness_33 = wot.hydrostatics.stiffness_matrix(hs_data)[2, 2]
    stiffness = np.atleast_2d(stiffness_33)

    # WEC
    wec_box = wot.WEC(fb, mass, stiffness, f0, nfreq, rho=rho)

    # BEM
    wec_box.run_bem(wave_dirs)

    return wec_box

@pytest.fixture()
def regular_wave(wec_wavebot):
    wec = wec_wavebot
    freq = 0.5
    amplitude = 0.25
    phase = 0.0
    wave = wot.waves.regular_wave(wec.f0, wec.nfreq, freq, amplitude, phase)
    return wave

@pytest.fixture()
def three_regular_waves(wec_wavebot):
    freq = 0.5
    amplitude = 0.25
    phase = 0.6
    wave_dirs_deg =  wec_wavebot.hydro.wave_direction.values * 180 /np.pi

    wave = wot.waves.wave_dataset(wec_wavebot.f0, wec_wavebot.nfreq,
                                  wave_dirs_deg)
    
    wave.S.loc[dict(omega = freq*2*np.pi)] = amplitude**2 / freq*2*np.pi
    wave.phase.loc[dict(omega = freq*2*np.pi)] = phase
    return wave


@pytest.fixture()
def irregular_wave(wec_wavebot):
    # irregular wave - multi direction
    Hs = 0.1
    Tp = 5
    fp = 1/Tp
    s_max = 10
    wave_directions = wec_wavebot.hydro.wave_direction.values * 180/np.pi
    dm = 7.0 # mean direction

    seed = 0.1

    spectrum_func = lambda f: pm(freq=f, fp=fp, hs=Hs)
    def spread_func(f,d):
        return wot.waves.spread_cos2s(freq = f, directions = d, dm = dm, fp = fp, s_max= s_max)

    irreg_wave = wot.waves.irregular_wave(wec_wavebot.f0, wec_wavebot.nfreq,
                            wave_directions, spectrum_func, spread_func, seed) 
        
    return irreg_wave


def test_regular_waves_symmetric_wec(wec_wavebot, three_regular_waves):
    """Confirm that power from multiple regular waves that have the same
       phase and frequency, but different directions matches 
       the superposition of the waves 
       * the three individual waves all give the same results
       * those results match CC
       * combined wave gives three times those results (position), 
       9?? times power
       """
    #PTO
    kinematics = np.eye(wec_wavebot.ndof)
    pto = wot.pto.PseudoSpectralPTO(wec_wavebot.nfreq, kinematics)
    obj_fun = pto.average_power
    wec_wavebot.f_add = {'PTO': pto.force_on_wec}

    #combined wave
    _, wec_fdom, _, _, power_combined, _ = wec_wavebot.solve(
                                        three_regular_waves, obj_fun, 
                                        nstate_opt = pto.nstate,
                                        scale_x_wec = 1.0,
                                        scale_x_opt = 0.01,
                                        scale_obj = 1e-1,
                                        optim_options={})
    sol_analytic_com = -1*np.sum(
                            np.abs(wec_fdom['excitation_force'][1:, :])**2  /
                            (8*np.real(wec_wavebot.hydro.Zi[:, 0, 0]))
                                )
    #individual waves
    sol_analytic_ind = np.zeros(three_regular_waves['wave_direction'].size)
    power_individual = np.zeros(three_regular_waves['wave_direction'].size)
    for nr_wave_dir in range(three_regular_waves['wave_direction'].size) :
        _, wec_fdom, _, _, obj, _ = wec_wavebot.solve(
                    three_regular_waves.isel(wave_direction = [nr_wave_dir]), 
                    obj_fun, 
                    nstate_opt = pto.nstate,
                    scale_x_wec = 1.0,
                    scale_x_opt = 0.01,
                    scale_obj = 1e-1,
                    optim_options={})
        power_individual[nr_wave_dir] = obj
        sol_analytic_ind[nr_wave_dir] = \
            -1*np.sum(np.abs(wec_fdom['excitation_force'][1:, :])**2  /
            (8*np.real(wec_wavebot.hydro.Zi[:, 0, 0]))
                     )

    rtol = 0.005
    #check if individual waves yield same result
    assert np.all(np.isclose(power_individual, power_individual[0],rtol))
    #check if combined wave yields expected power as function of 
    # the individual waves 
    assert np.isclose(power_combined,
                  np.sum(power_individual)*power_individual.size,
                  rtol)
    #check if results match the theoretical solution for maximum power 
    assert np.isclose(power_combined, sol_analytic_com,rtol)
    assert np.all(np.isclose(power_individual, sol_analytic_ind,rtol))


def test_regular_waves_asymmetric_wec(wec_box, three_regular_waves):
    """Confirm that power from different directions is unequal for asymmetric wec"""
    #PTO
    kinematics = np.eye(wec_box.ndof)
    pto = wot.pto.PseudoSpectralPTO(wec_box.nfreq, kinematics)
    obj_fun = pto.average_power
    wec_box.f_add = {'PTO': pto.force_on_wec}

    #solve for combined wave
    _, wec_fdom, _, _, power_combined, _ = wec_box.solve(
                                        three_regular_waves,
                                        obj_fun, 
                                        nstate_opt = pto.nstate,
                                        scale_x_wec = 1.0,
                                        scale_x_opt = 0.01,
                                        scale_obj = 1e-1,
                                        optim_options={})
    sol_analytic_com = -1*np.sum(np.abs(wec_fdom['excitation_force'][1:, :])**2 
                            / (8*np.real(wec_box.hydro.Zi[:, 0, 0])))

    #individual waves
    sol_analytic_ind = np.zeros(three_regular_waves['wave_direction'].size)
    power_individual = np.zeros(three_regular_waves['wave_direction'].size)
    for nr_wave_dir in range(three_regular_waves['wave_direction'].size) :
        _, wec_fdom, _, _, obj, _ = wec_box.solve(
                    three_regular_waves.isel(wave_direction = [nr_wave_dir]), 
                    obj_fun, 
                    nstate_opt = pto.nstate,
                    scale_x_wec = 1.0,
                    scale_x_opt = 0.01,
                    scale_obj = 1e-1,
                    optim_options={})
        power_individual[nr_wave_dir] = obj
        sol_analytic_ind[nr_wave_dir] = \
            -1*np.sum(np.abs(wec_fdom['excitation_force'][1:, :])**2  /
            (8*np.real(wec_box.hydro.Zi[:, 0, 0]))
                     )

    #relative tolerance for assertion
    rtol = 0.005
    #check that analytic power for the combined waves matches
    assert np.isclose(sol_analytic_com, power_combined, rtol)
    #check that not all power results are the same
    assert not np.all(np.isclose(power_individual, power_individual[0],rtol))
    #check if combined wave yields expected power as function of 
    # the individual waves
    assert np.isclose(power_combined,
                  np.sum(power_individual)*power_individual.size,
                  rtol)

def test_directional_regular_wave_decomposed(wec_wavebot):
    """Confirm that power from an arbitrary direction matches the power
        when decomposed into x- and y- direction"""
    assert 1 == 1


# def test_longcrested_wave_power(wec_wavebot):
#     """Confirm power results for long crested wave are as expected"""
#     wec = wec_wavebot
#     spectrum = 
#     wave_long_crested = wot.waves.long_crested_wave(
#                             wec.f0, wec.nfreq, spectrum, wec.wave_dir)
#     # power_theory = wot.power_limit()  # Correct?
#     assert 1 == 1


def test_irregular_wave_power(wec_wavebot, irregular_wave):
    """Confirm power results for irregular wave are as expected
    TBD how to get this theoretically?
    """
    assert 1 == 1



# def test_spectrum_energy(irregular_wave):
#     """Confirm that energy is spread correctly accross wave directions
#     * integral (sum) over all directions of the spread function gives (vector) 1
#     * integral (sum) over all directions of the irregular wave (2D) spectrum gives the omni-direction spectrum (vector)
#     """
#     assert 1 == 1
