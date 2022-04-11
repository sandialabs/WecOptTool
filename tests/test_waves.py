import os

import pytest
import autograd.numpy as np
from autograd.builtins import isinstance, tuple, list, dict
import capytaine as cpy
import meshio
from scipy.optimize import Bounds

import wecopttool as wot
from wecopttool.waves import pierson_moskowitz_spectrum as pm





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

# @pytest.fixture()
# def wec_box():
#     return wec_box

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
    
    wave.S.loc[dict(omega = freq*2*np.pi)] = amplitude**2
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

def test_dummy():
    assert 1 == 1

def test_regular_waves_symmetric_wec(wec_wavebot, three_regular_waves):
    """Confirm that power from multiple regular waves matches superposition
       Waves have the same phase and frequency, but different directions
       * the three individual waves all give the same results
       * those results match CC
       * combined wave gives three times those results
       """
    assert 1 == 1


# def test_regular_waves_asymmetric_wec(wec_box, three_regular_waves):
#     """Confirm that power from different directions is unequal for asymmetric wec"""
#     assert 1 == 1



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
