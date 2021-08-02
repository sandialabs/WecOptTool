#!/usr/bin/env python
#  -*- coding: utf-8 -*-
from scipy.optimize import minimize
from WecOptTool.examples import WaveBot
from autograd import jacobian
import autograd.numpy as np
from numpy import testing
import WecOptTool as wot
import capytaine as cpy
from mhkit import wave
import xarray as xr
import pytest
import pygmsh
import os


path_to_current_file = os.path.realpath(__file__)
current_directory = os.path.dirname(path_to_current_file)
dat_dir = os.path.join(current_directory, os.path.join('data'))


# TODO: Need to move files from run.WaveBot to WecOptTool.examples
# and setup the call to the example files
# def test_example_waveBot1DOF(script_runner):
    # ret = script_runner.run('example_WaveBot_oneDof')
    # assert ret.success


# # -----------------------------------------------------------------------------

@pytest.mark.dependency()
def test_WaveBot_geom():
    # water properties
    rho = 1e3

    # frequencies
    f0 = 0.05    
    num_freq = 18
    mesh = WaveBot.mesh()
    mesh_file = 'tmp_mesh.stl'
    mesh.write(mesh_file)
    fb = cpy.FloatingBody.from_file(mesh_file, name='WaveBot')
    os.remove(mesh_file)
    fb.add_translation_dof(name="HEAVE")

    pytest.f0 = f0
    pytest.num_freq = num_freq
    pytest.fb = fb
    
    # mass and hydrostatic stiffness
    hs_data = wot.hydrostatics.hydrostatics(fb, rho=rho)
    M33 = wot.hydrostatics.mass_matrix_constant_density(hs_data)[2, 2]
    M = np.atleast_2d(M33)
    K33 = wot.hydrostatics.stiffness_matrix(hs_data)[2, 2]
    K = np.atleast_2d(K33)

    # PTO: state, force, power (objective function)
    kinematics = np.eye(fb.nb_dofs)
    num_x_pto, f_pto, power_pto, pto_postproc = \
    wot.pto.pseudospectral_pto(num_freq, kinematics)

    # create WEC
    pytest.wec = wot.WEC(fb, M, K, f0, num_freq, f_add=f_pto, rho=rho)
    # pytest.wec = wot.WEC(mesh=WaveBot.mesh(), 
                         # modes=np.ones(6),
                         # f0=0.05,
                         # num_freq=1,
                         # run_bem=False,
                         # name=None, 
                         # wrk_dir=False)

@pytest.mark.dependency(depends=['test_WaveBot_geom'])
def test_run_bem():
    #freq = pytest.wec.params['f0']*np.arange(1, pytest.wec.params['num_freq']+1)
    #pytest.wec.run_bem()
    # wave
    freq = 0.2
    amplitude = 0.25
    phase = 0.0
    waves = wot.waves.regular_wave(pytest.f0, pytest.num_freq, freq, amplitude, phase)

    # run BEM
    pytest.wec.run_bem()

@pytest.mark.dependency(depends=['test_run_bem'])    
def test_write_bem():
    pytest.wec.write_bem()

# -----------------------------------------------------------------------------

def test_bad_mesh_type():
    with pytest.raises(TypeError):
        bad_wec = wot.WEC(mesh=None, 
                         modes=np.ones(6),
                         f0=0.05,
                         num_freq=1)

def test_bad_f0_type():
    with pytest.raises(TypeError):
        bad_wec = wot.WEC(mesh=pytest.fb,
                         modes=np.ones(6),
                         f0=None,
                         num_freq=1)

def test_bad_num_freq_type():
    with pytest.raises(TypeError):
        bad_wec = wot.WEC(mesh=pytest.fb,
                         modes=np.ones(6),
                         f0=0.05,
                         num_freq=float(1.234))

def test_bad_name_type():
    with pytest.raises(TypeError):
        bad_wec = wot.WEC(mesh=pytest.fb,
                         modes=np.ones(6),
                         f0=0.05,
                         num_freq=float(1.234),
                         name=int(1))

# -----------------------------------------------------------------------------

# @pytest.fixture
# def wec_from_file():
    # wec = wot.WEC.from_file(name='test_WaveBot',
                            # fpath=os.path.join(dat_dir,'test_WaveBot'))
    # return wec

# def test_from_file(wec_from_file):
    # assert True
     
# def test_get_pow_ub(wec_from_file):
    # Spd = wave.resource.jonswap_spectrum(f=wec_from_file.hydro.omega/2/np.pi, 
                                         # Tp=1/0.6, 
                                         # Hs=1)

    # pub = [wec_from_file.get_pow_ub(Spd, dof=dofi) for dofi in range(6)]
    # da1 = xr.concat(pub, dim='radiating_dof')
    # da2 = xr.open_dataarray(os.path.join(dat_dir,'test_WaveBot','pub.nc'))
    # xr.testing.assert_allclose(a=da1, b=da2)
    
# def test_calc_impedance_symmetric(wec_from_file):
    # Zi = wot.calc_impedance(wec_from_file.hydro)
    # testing.assert_almost_equal(Zi.values, Zi.values.transpose((0,2,1)))

# def test_plot_impedance(wec_from_file):
    # wec_from_file.plot_impedance()
    
# @pytest.fixture
# def my_Z_block(wec_from_file):
    # return wec_from_file.__make_Zi_block__().toarray()

# def test_make_Z_block_symmetric(my_Z_block):
    # testing.assert_almost_equal(my_Z_block, my_Z_block.transpose())

# def test_Z_block_shape(my_Z_block,wec_from_file):
    # m = wec_from_file.hydro.radiating_dof.size
    # n = wec_from_file.hydro.omega.size
    # exp_shape = np.array([m*n, m*n])
    # shape = np.array(my_Z_block.shape)

    # testing.assert_array_equal(exp_shape, shape)


# @pytest.mark.parametrize("x_extra", [
    # (np.zeros(np.random.randint(0, 100, size=None))),
    # (np.random.randint(0, 100, size=None))
# ])
# def test_gen_initial_guess(x_extra, wec_from_file):
    # m = wec_from_file.hydro.radiating_dof.size
    # n = wec_from_file.hydro.omega.size
    # x0 = wec_from_file.gen_initial_guess(x_extra=x_extra)

    # if isinstance(x_extra, int):
        # exp_size = m*(2*n + 1) + x_extra
    # else:
        # exp_size = m*(2*n + 1) + x_extra.size

    # assert x0.ndim == 1
    # assert x0.size == exp_size


# def test_Dphi_shape(wec_from_file):
    # m = wec_from_file.hydro.radiating_dof.size
    # n = wec_from_file.hydro.omega.size
    # exp_shape = np.array([2*m*n,m*(2*n+1)])

    # shape = np.array(wec_from_file.Dphi.shape)

    # testing.assert_array_equal(exp_shape, shape)


# @pytest.mark.repeat(4) # run multiple time for different x0's
# def test_WaveBot_oneDof_no_wave():
    # wec = wot.WEC.from_file(name='example_WaveBot_oneDof_nf18',
                            # fpath=os.path.join(dat_dir,
                                # 'example_WaveBot_oneDof_nf18'))

    # wec.kinematics = dict()
    # wec.kinematics['Jac'] = np.zeros((1,1), float)
    # wec.kinematics['Jac'][0,0] = 1

    # num_freq = wec.hydro.omega.size
    # num_modes = wec.hydro.influenced_dof.size

    # def obj_fun(x):
        # x_wec, x_pto, nf, _ = wec.decompose_decision_var(x)
        # Jac = wec.kinematics['Jac']
        # x_vel = Jac.transpose() @ np.reshape(wec.Dphi @ x_wec, (-1, 2*nf))
        # P = x_vel.flatten() @ x_pto / (2*wec.params['f0'])
        # return P


    # def F_pto_fun(x):
        # x_wec, x_pto, nf, nm = wec.decompose_decision_var(x)
        # nc=2*nf+1
        # fp3 = x_pto @ wec.Phi[1::,::]
        # Jac = wec.kinematics['Jac']
        # f_pto = np.squeeze(Jac) * fp3
        # return f_pto


    # def F_ext_fun(x):
        # f_ext = 0
        # return f_ext


    # x0 = np.random.rand(num_modes*(2*num_freq+1) + 1*(2*num_freq))

    # f_exc = np.zeros((num_modes*2*num_freq+1))

    # my_resid_fun = lambda x: wec.dynamic_residual(x, f_exc, F_pto_fun, F_ext_fun)

    # eq_cons = {'type': 'eq',
                # 'fun': my_resid_fun,
                # 'jac':jacobian(my_resid_fun)
                # }
               
    # res = minimize(fun=obj_fun,
                    # x0=x0,
                    # jac=jacobian(obj_fun),
                    # constraints=(eq_cons),
                    # options={'disp':False,
                             # },
                    # tol=1e-12
                    # )

    # assert np.max(np.abs(res.x)) < 5e-3
    # assert np.abs(res.fun) < 5e-9
