""" Unit tests for functions in the `core.py` module.
Does not include the WEC class.
"""
import os
import random

import pytest
from pytest import approx
import numpy as np
import xarray as xr

import wecopttool as wot


@pytest.fixture()
def f0(): return 0.12


@pytest.fixture()
def nfreq(): return 5


@pytest.fixture(scope="function")
def nsubsteps(): return random.randint(2, 10)


@pytest.fixture()
def ncomponents(nfreq): return wot.ncomponents(nfreq)


@pytest.fixture(scope="module")
def bem_data():  # TODO: run BEM + add mass, hydro, friction
    cwd = os.path.dirname(os.path.abspath(__file__))
    bem_file = os.path.join(cwd, 'data', 'bem.nc')
    return wot.read_netcdf(bem_file)


def test_ncomponents(ncomponents, nfreq):
    assert ncomponents==2*nfreq+1


def test_frequency(f0, nfreq):
    freqs = wot.frequency(f0, nfreq)
    assert ((freqs.ndim==1) and (len(freqs)==nfreq+1))  # shape
    assert ((freqs[0]==0.0) and (freqs[-1]==approx(f0*nfreq))) # first & last
    assert np.diff(freqs)==approx(np.diff(freqs)[0])  # evenly spaced


def test_time(f0, nfreq, nsubsteps, ncomponents):
    time = wot.time(f0, nfreq)
    time_1 = wot.time(f0, nfreq, 1)
    time_sub = wot.time(f0, nfreq, nsubsteps)
    assert time==approx(time_1)  # default behavior
    for t,n in [(time, 1), (time_sub, nsubsteps)]:
        assert ((t.ndim==1) and (len(t)==ncomponents*n))  # shape
        assert np.diff(t)==approx(np.diff(t)[0])  # evenly spaced
        t_last = 1.0/f0-np.diff(t)[0]
        assert ((t[0]==0.0) and (t[-1]==approx(t_last)))  # first & last


def test_time_mat(f0, nfreq, nsubsteps, ncomponents):
    time_mat = wot.time_mat(f0, nfreq)
    time_mat_1 = wot.time_mat(f0, nfreq, 1)
    time_mat_sub = wot.time_mat(f0, nfreq, nsubsteps)
    # synthetic case
    f0_s = 0.5
    nfreq_s = 2
    time_mat_synthetic = wot.time_mat(f0_s, nfreq_s)
    f = np.array([0, 1, 2])*f0_s
    w = 2*np.pi * f
    t = 1/(2*nfreq_s+1) * 1/f0_s * np.arange(0, 2*nfreq_s+1)
    cos, sin = np.cos, np.sin
    expected_synthetic = np.array([
        [1, 1, 0, 1, 0],
        [1, cos(w[1]*t[1]), -sin(w[1]*t[1]), cos(w[2]*t[1]), -sin(w[2]*t[1])],
        [1, cos(w[1]*t[2]), -sin(w[1]*t[2]), cos(w[2]*t[2]), -sin(w[2]*t[2])],
        [1, cos(w[1]*t[3]), -sin(w[1]*t[3]), cos(w[2]*t[3]), -sin(w[2]*t[3])],
        [1, cos(w[1]*t[4]), -sin(w[1]*t[4]), cos(w[2]*t[4]), -sin(w[2]*t[4])]
    ])
    # assert
    assert time_mat==approx(time_mat_1)  # default behavior
    for tm,n in [(time_mat, 1), (time_mat_sub, nsubsteps)]:
        assert tm.shape==(n*ncomponents, ncomponents)  # shape
        assert all(tm[:, 0]==1.0) # zero-frequency components
        assert all(tm[0, 1:]==np.array([1, 0]*nfreq))  # time zero
    assert time_mat_synthetic==approx(expected_synthetic) # full matrix


def derivative_mat(f0, nfreq):
    der = wot.derivative_mat(f0, nfreq)
    # synthetic case
    f0_s = 0.5
    nfreq_s = 2
    der_synthetic = wot.derivative_mat(f0_s, nfreq_s)
    w0 = 2*np.pi*f0_s
    expected_synthetic = np.array([
        [0,  0,   0,    0,     0],
        [0,  0, -w0,    0,     0],
        [0, w0,   0,    0,     0],
        [0,  0,   0,    0, -2*w0],
        [0,  0,   0, 2*w0,     0]
    ])
    assert der.shape==(2*nfreq+1, 2*nfreq+1)  # shape
    assert ((all(der[:, 0]==0.0)) and (all(der[0, :]==0.0)))  # first row & col
    assert der_synthetic==approx(expected_synthetic)  # full matrix


def test_degrees_to_radians():
    degree = 44.26
    degrees = [0.0, -1000, 2.53, 9.99*np.pi]
    rads = wot.degrees_to_radians(degrees)
    rads_st = wot.degrees_to_radians(degrees, sort=True)
    rads_us = wot.degrees_to_radians(degrees, sort=False)
    rads_np = wot.degrees_to_radians(np.array(degrees))
    rad = wot.degrees_to_radians(degree)
    rad_cyc = wot.degrees_to_radians(degree+random.randint(-10,10)*360)
    assert rads==approx(rads_st)  # default behavior
    assert rads==approx(rads_np)  # numpy arrays
    assert (all(rads>=-np.pi) and all(rads<np.pi))  # range (-π, π]
    assert (all(rads_us>=-np.pi) and all(rads_us<np.pi))  # range (-π, π]
    assert np.sort(rads)==approx(np.sort(rads_us))  # sorting
    assert ((rad>=-np.pi) and (rad<np.pi))  # scalar
    assert rad==approx(rad_cyc)  # cyclic
    for r,d in [(0, 0), (np.pi, 180), (0, 360), (np.pi, -180)]:
        assert r==approx(wot.degrees_to_radians(d))  # special cases


def test_standard_forces(bem_data):
    forces = wot.standard_forces(bem_data)
    # for now, make up an x, etc. plot time domain. make sure it makes sense. make tests out of it.