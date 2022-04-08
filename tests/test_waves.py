import os

import pytest
import autograd.numpy as np
from autograd.builtins import isinstance, tuple, list, dict
import capytaine as cpy
import meshio
from scipy.optimize import Bounds

import wecopttool as wot


@pytest.fixture()
def wec_wavebot():
    return wec_wavebot

@pytest.fixture()
def wec_box():
    return wec_box

@pytest.fixture()
def regular_wave():
    return wave

@pytest.fixture()
def three_regular_waves(wec_wavebot):
    wave = wot.waves.wave_dataset(f0, nfreq, directions)
    # wave.spectrum[specific freq, all directions] = amplitude**2
    # wave.phase[specific freq, all directions] = same phase for all directions
    return wave_data


@pytest.fixture()
def irregular_wave():
    return irreg_wave



def test_regular_waves_symmetric_wec(wec_wavebot, three_regular_waves):
    """Confirm that power from multiple regular waves matches superposition
       Waves have the same phase and frequency, but different directions
       * the three individual waves all give the same results
       * those results match CC
       * combined wave gives three times those results
       """


def test_regular_waves_asymmetric_wec(wec_box, three_regular_waves):
    """Confirm that power from different directions is unequal for asymmetric wec"""


def test_longcrested_wave_power(wec_wavebot):
    """Confirm power results for long crested wave are as expected"""
    wave_long_crested = long_crested_wave(f0, nfreq, spectrum, wave_dir)
    power_theory = wot.power_limit()  # Correct?


def test_irregular_wave_power(wec_wavebot, irregular_wave):
    """Confirm power results for irregular wave are as expected
    TBD how to get this theoretically?
    """



def test_spectrum_energy(irregular_wave):
    """Confirm that energy is spread correctly accross wave directions
    * integral (sum) over all directions of the spread function gives (vector) 1
    * integral (sum) over all directions of the irregular wave (2D) spectrum gives the omni-direction spectrum (vector)
    """
