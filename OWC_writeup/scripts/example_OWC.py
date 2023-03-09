from datetime import datetime

import autograd.numpy as np
import capytaine as cpy
import matplotlib.pyplot as plt
from scipy.optimize import brute

import wecopttool as wot

import autograd.numpy as np

A_wec = 0.1
A_turbine = 0.01
P_ambient = 101325.0 #pa (1 atm)
rho = 1.225 #kg/m3 air
R_turbine = 0.298/2 #m
H_blade = (1.0-0.7)*R_turbine #m
chord = 0.054 #m
N_blade = 30.0
lambda_air = 1.401

inv_TSRdata_Ca = np.array([-1000000.0,
                -100.0,
                -50.0,
                -20.0,
                -10.0,
                0.0,
                0.012668019,
                0.03426608,
                0.045142045,
                0.057553153,
                0.071173399,
                0.085301268,
                0.097366303,
                0.118841776,
                0.129754941,
                0.139778827,
                0.158789719,
                0.169510357,
                0.179040696,
                0.19001597,
                0.20320735,
                0.217075432,
                0.235373075,
                0.243281766,
                0.261535771,
                0.276437612,
                0.28738014,
                0.309746514,
                0.314496222,
                0.330525042,
                0.346010286,
                0.35791277,
                0.39363278,
                0.426007626,
                0.432751135,
                0.448256149,
                0.472991905,
                0.49238604,
                0.506538082,
                0.522258542,
                0.541141927,
                0.559711436,
                0.587280356,
                0.603243099,
                0.623148907,
                0.643251111,
                0.66566147,
                0.680382139,
                0.702029778,
                0.72676211,
                0.75149701,
                0.778443456,
                0.802495913,
                0.830306138,
                0.859654739,
                0.890547726,
                0.921439344,
                0.953872962,
                0.987842604,
                1.021805398,
                1.055763211,
                1.089717911,
                1.123670121,
                1.157624821,
                1.191565203,
                1.225508697,
                1.259449701,
                1.293384479,
                1.327313966,
                1.361237849,
                1.395167335,
                1.429090907,
                1.463007631,
                1.496926223,
                1.530844503,
                1.564760293,
                1.598674838,
                1.632587515,
                1.666499881,
                1.700410691,
                1.734322745,
                1.768229819,
                1.802136271,
                1.836043345,
                1.869948551,
                1.903858116,
                1.937757408,
                1.971661992,
                2.005578093,
                2.039467113,
                2.073365782,
                2.107270366,
                2.14117246,
                2.175075487,
                2.208974156,
                2.242872826,
                2.276771495,
                2.310670165,
                2.344568834,
                2.378467504,
                2.412366173,
                2.446264843,
                2.480163512,
                2.503649469,
                2.509371154,
                2.65,
                2.75,
                3.0,
                4.0,
                5.0,
                7.0,
                10.0,
                20.0,
                50.0,
                100.0,
                1000000.0])

Ca_data = np.array([0.092282521,
        0.092282521,
        0.092282521,
        0.092282521,
        0.092282521,
        0.092282521,
        0.092282521,
        0.108898362,
        0.166429569,
        0.220299948,
        0.281008945,
        0.337880945,
        0.389410575,
        0.459040664,
        0.518464217,
        0.578526192,
        0.629890894,
        0.672544117,
        0.723541827,
        0.778887736,
        0.830091701,
        0.892335842,
        0.945980007,
        0.994584051,
        1.05696206,
        1.106830028,
        1.161446822,
        1.228899216,
        1.290898399,
        1.359077526,
        1.408090836,
        1.465159246,
        1.562243494,
        1.655088735,
        1.697592835,
        1.759255085,
        1.811679549,
        1.859073611,
        1.914558511,
        1.982617847,
        2.036904363,
        2.087541557,
        2.145242742,
        2.194618208,
        2.255209854,
        2.300621402,
        2.344023427,
        2.396784052,
        2.445120045,
        2.495353764,
        2.547230542,
        2.59005665,
        2.643459039,
        2.691239423,
        2.737428015,
        2.785870789,
        2.833437265,
        2.88173104,
        2.92692827,
        2.967744011,
        3.005373215,
        3.041010834,
        3.075055183,
        3.110692802,
        3.137169126,
        3.165637035,
        3.192511676,
        3.215403146,
        3.23490892,
        3.250829839,
        3.270335613,
        3.286057374,
        3.297397646,
        3.30993287,
        3.322268935,
        3.333011731,
        3.342957894,
        3.351709104,
        3.360261157,
        3.367817416,
        3.37617031,
        3.381336666,
        3.386104705,
        3.391271062,
        3.395242467,
        3.402002092,
        3.402189484,
        3.405762572,
        3.4066704527,
        3.410319686,
        3.410108761,
        3.413681849,
        3.415661668,
        3.418238963,
        3.418028038,
        3.417817113,
        3.417606188,
        3.417395263,
        3.417184338,
        3.416973413,
        3.416762488,
        3.416551563,
        3.416340638,
        3.408526823,
        3.408526823,
        3.408526823,
        3.408526823,
        3.408526823,
        3.408526823,
        3.408526823,
        3.408526823,
        3.408526823,
        3.408526823,
        3.408526823,
        3.408526823,
        3.408526823])

inv_TSRdata_Ct = np.array([-1000000,
                -100,
                -10,
                -1,
                0,
                0.026828855,
                0.061012163,
                0.095195471,
                0.129390802,
                0.163601319,
                0.19783525,
                0.232095125,
                0.266361327,
                0.300659802,
                0.337110351,
                0.371665655,
                0.403564085,
                0.436495224,
                0.465399555,
                0.514548355,
                0.544918292,
                0.571192363,
                0.59270955,
                0.618927127,
                0.642745995,
                0.66620479,
                0.69188684,
                0.739707487,
                0.76472745,
                0.789742192,
                0.813204716,
                0.8366742,
                0.861685462,
                0.886707165,
                0.911725387,
                0.936747091,
                0.960198676,
                0.985222229,
                1.013346631,
                1.041475672,
                1.06804186,
                1.093049642,
                1.121175783,
                1.152420632,
                1.182099998,
                1.208656035,
                1.238318497,
                1.272654306,
                1.306992646,
                1.341322127,
                1.375644648,
                1.409964004,
                1.444286525,
                1.478596389,
                1.512902457,
                1.547211689,
                1.581516492,
                1.615813701,
                1.650105848,
                1.684399893,
                1.718689508,
                1.752971531,
                1.78725735,
                1.82154127,
                1.855813168,
                1.890090128,
                1.924355698,
                1.958615573,
                1.992872284,
                2.027130893,
                2.061392033,
                2.095648111,
                2.129901658,
                2.16414951,
                2.1983923,
                2.232637621,
                2.266882941,
                2.301128262,
                2.335369154,
                2.369611943,
                2.403853467,
                2.438098788,
                2.467671948,
                2.6,
                3,
                10,
                50,
                100,
                1000000])
            
Ct_data = np.array([0.375722776,
                0.375722776,
                0.375722776,
                0.375722776,
                0.375722776,
                0.375722776,
                0.375722776,
                0.374275093,
                0.376102447,
                0.382066689,
                0.394408635,
                0.413817765,
                0.434950599,
                0.464874322,
                0.494891355,
                0.522413413,
                0.557058674,
                0.595505788,
                0.637675307,
                0.733645009,
                0.775246014,
                0.808599425,
                0.858889048,
                0.898233279,
                0.942662941,
                0.98307891,
                1.0176516,
                1.109300279,
                1.151661245,
                1.192600155,
                1.234031877,
                1.277359673,
                1.317350546,
                1.36018553,
                1.402072477,
                1.44490746,
                1.483359638,
                1.526698737,
                1.568074078,
                1.610713468,
                1.650948891,
                1.689991727,
                1.731841086,
                1.776591773,
                1.818222198,
                1.855692513,
                1.892718255,
                1.932811829,
                1.973594884,
                2.011964754,
                2.04843855,
                2.084050494,
                2.120524291,
                2.15355068,
                2.185542846,
                2.218396865,
                2.250044291,
                2.279623273,
                2.307823292,
                2.336540422,
                2.364050959,
                2.389493052,
                2.415969367,
                2.441928572,
                2.464612739,
                2.488675869,
                2.509636333,
                2.529045463,
                2.547592742,
                2.566657132,
                2.586411003,
                2.604785911,
                2.622471338,
                2.638605431,
                2.653360562,
                2.668805174,
                2.684249786,
                2.699694399,
                2.713932418,
                2.728687549,
                2.743097939,
                2.758542551,
                2.771242526,
                2.771242526,
                2.771242526,
                2.771242526,
                2.771242526,
                2.771242526,
                2.771242526])

inv_TSRdata_eta = np.array([-1000000.0,
                -100.0,
                -50.0,
                -20.0,
                -10.0,
                -5.0,
                -3.0,
                -1.0,
                -0.562,
                -0.17,
                0.08,
                0.263603728,
                0.285361488,
                0.303031426,
                0.312196059,
                0.315338846,
                0.321228826,
                0.32568587,
                0.332745054,
                0.337165679,
                0.344625482,
                0.356436838,
                0.363274991,
                0.36907706,
                0.376260575,
                0.382753366,
                0.389798736,
                0.39763153,
                0.408171956,
                0.417358566,
                0.428092394,
                0.437458592,
                0.448874352,
                0.459423569,
                0.472478225,
                0.485210544,
                0.496411761,
                0.511279564,
                0.523608961,
                0.538073843,
                0.55573792,
                0.571320621,
                0.587069095,
                0.608826855,
                0.632035133,
                0.658144445,
                0.688605309,
                0.720516691,
                0.752428072,
                0.784339454,
                0.816250835,
                0.848162217,
                0.880073598,
                0.91198498,
                0.943896361,
                0.975807743,
                1.007719124,
                1.039630506,
                1.071541888,
                1.103453269,
                1.135364651,
                1.167276032,
                1.199187414,
                1.231098795,
                1.263010177,
                1.294921558,
                1.32683294,
                1.358744321,
                1.390655703,
                1.422567085,
                1.454478466,
                1.486389848,
                1.518301229,
                1.550212611,
                1.582123992,
                1.614035374,
                1.645946755,
                1.677858137,
                1.709769518,
                1.7416809,
                1.773592282,
                1.805503663,
                1.837415045,
                1.869326426,
                1.901237808,
                1.933149189,
                1.965060571,
                1.996971952,
                2.057893681,
                2.089805062,
                2.121716444,
                2.153627825,
                2.185539207,
                2.217450589,
                2.281273352,
                2.345096115,
                2.408918878,
                2.469840606,
                3.27,
                5.0,
                10.0,
                20.0,
                50.0,
                100.0,
                1000000.0])

eta_data = np.array([0.003429765,  
            0.003429765,
            0.003429765,
            0.003429765,
            0.003429765,
            0.003429765,
            0.003429765,
            0.003429765,
            0.0065,
            0.011477158,
            0.03,
            0.075190746,
            0.109436258,
            0.133695731,
            0.147823096,
            0.155766371,
            0.163040511,
            0.173411038,
            0.183500092,
            0.191855048,
            0.199981875,
            0.217788488,
            0.225356447,
            0.233872485,
            0.243477412,
            0.252925278,
            0.261402283,
            0.269837529,
            0.278543469,
            0.287291074,
            0.297463119,
            0.305032595,
            0.313937652,
            0.323279961,
            0.33256815,
            0.341011385,
            0.348811865,
            0.357759094,
            0.364826008,
            0.372254637,
            0.379637054,
            0.386450734,
            0.393406272,
            0.400963161,
            0.40850769,
            0.414986945,
            0.422387209,
            0.42837799,
            0.432588372,
            0.436196157,
            0.439858725,
            0.442261318,
            0.443842188,
            0.445368277,
            0.446593068,
            0.446996136,
            0.446577483,
            0.446323173,
            0.445082797,
            0.444143719,
            0.442136402,
            0.440539946,
            0.438258721,
            0.435867934,
            0.433093676,
            0.430319418,
            0.427736895,
            0.42468873,
            0.421996644,
            0.419058041,
            0.416420737,
            0.413536916,
            0.410680486,
            0.408180135,
            0.405679785,
            0.402987699,
            0.400213441,
            0.397658309,
            0.394938833,
            0.392246747,
            0.389664224,
            0.386753012,
            0.384225271,
            0.381752311,
            0.379251961,
            0.376724219,
            0.374196478,
            0.371696128,
            0.367054705,
            0.36477348,
            0.362245739,
            0.359854951,
            0.357546336,
            0.352901715,
            0.348065358,
            0.343338564,
            0.338748725,
            0.33688156,
            0.33688156,
            0.33688156,
            0.33688156,
            0.33688156,
            0.33688156,
            0.33688156,
            0.33688156])

def h_poly_helper(tt):
  A = np.array([
      [1, 0, -3, 2],
      [0, 1, -2, 1],
      [0, 0, 3, -2],
      [0, 0, -1, 1]
      ], dtype=tt[-1].dtype)
  return [
    sum( A[i, j]*tt[j] for j in range(4) )
    for i in range(4) ]

def h_poly(t):
  tt = [ None for _ in range(4) ]
  tt[0] = 1
  for i in range(1, 4):
    tt[i] = tt[i-1]*t
  return h_poly_helper(tt)

def H_poly(t):
  tt = [ None for _ in range(4) ]
  tt[0] = t
  for i in range(1, 4):
    tt[i] = tt[i-1]*t*i/(i+1)
  return h_poly_helper(tt)

def interp_func(x, y):
  "Returns integral of interpolating function"
  if len(y)>1:
    m = (y[1:] - y[:-1])/(x[1:] - x[:-1])
    m = np.concatenate([m[[0]], (m[1:] + m[:-1])/2, m[[-1]]])
  def f(xs):
    if len(y)==1: # in the case of 1 point, treat as constant function
      return y[0] + np.zeros_like(xs)
    I = np.searchsorted(x[1:], xs)
    dx = (x[I+1]-x[I])
    hh = h_poly((xs-x[I])/dx)
    return hh[0]*y[I] + hh[1]*m[I]*dx + hh[2]*y[I+1] + hh[3]*m[I+1]*dx
  return f

def myinterp(x, y, xs):
  return interp_func(x,y)(xs)

def self_rectifying_turbine_omega_incompressible(omega_turbine,V_wec):
    inv_TSR = (V_wec * A_wec) / (A_turbine * omega_turbine * R_turbine)
    Ca = myinterp(inv_TSRdata_Ca,Ca_data,inv_TSR)
    eta = myinterp(inv_TSRdata_eta,eta_data,inv_TSR)
    delta_P_turbine = Ca * 0.5 * rho * ((V_wec*A_wec/A_turbine)**2 + (omega_turbine*R_turbine)**2) * H_blade * chord * N_blade / A_wec

    F_turbine = delta_P_turbine/A_turbine
    F_wec = F_turbine*A_wec/A_turbine

    power = delta_P_turbine * V_wec * A_wec * eta
    print("inv_TSR")
    print(inv_TSR[10])
    print("Delta P")
    print(delta_P_turbine[10])
    print("eta")
    print(eta[10])
    return F_wec, power

def self_rectifying_turbine_omega_compressible(omega_turbine,V_wec,delta_P_turbine_guess):
    inv_TSR = (V_wec * A_wec) / (A_turbine * (delta_P_turbine_guess/P_ambient + 1)^(1/lambda_air) * omega_turbine * R_turbine)
    Ca = myinterp(inv_TSRdata_Ca,Ca_data,inv_TSR)
    eta = myinterp(inv_TSRdata_eta,eta_data,inv_TSR)
    
    delta_P_turbine = Ca * 0.5 * rho * ((V_wec*A_wec/(A_turbine * (delta_P_turbine_guess/P_ambient + 1)^(1/lambda_air)))**2 + (omega_turbine*R_turbine)**2) * H_blade * chord * N_blade / A_wec

    residual = delta_P_turbine_guess - delta_P_turbine

    F_turbine = delta_P_turbine/A_turbine
    F_wec = F_turbine*A_wec/A_turbine

    power = delta_P_turbine * V_wec * A_wec * eta

    return F_wec, power, residual

def self_rectifying_turbine_torque_incompressible(torque_turbine,V_wec,omega_turbine_guess):
    inv_TSR = (V_wec * A_wec) / (A_turbine * omega_turbine_guess * R_turbine)
    Ct = myinterp(inv_TSRdata_Ct,Ct_data,inv_TSR)
    Ca = myinterp(inv_TSRdata_Ca,Ca_data,inv_TSR)
    eta = myinterp(inv_TSRdata_eta,eta_data,inv_TSR)

    factorA = torque_turbine/(Ct*0.5*rho*H_blade*chord*N_blade*R_turbine)
    factorB = (V_wec*A_wec / (A_turbine))^2
    omega_turbine = 1/R_turbine * np.sqrt(factorA - factorB)

    delta_P_turbine = Ca * 0.5 * rho * ((V_wec*A_wec/(A_turbine))**2 + (omega_turbine*R_turbine)**2) * H_blade * chord * N_blade / A_wec

    residual = omega_turbine_guess - omega_turbine

    F_turbine = delta_P_turbine/A_turbine
    F_wec = F_turbine*A_wec/A_turbine

    power = delta_P_turbine * V_wec * A_wec * eta

return F_wec, power, residual

def self_rectifying_turbine_torque_compressible(torque_turbine,V_wec,delta_P_turbine_guess,omega_turbine_guess):
    inv_TSR = (V_wec * A_wec) / (A_turbine * (delta_P_turbine_guess/P_ambient + 1)^(1/lambda_air) * omega_turbine_guess * R_turbine)
    Ct = myinterp(inv_TSRdata_Ct,Ct_data,inv_TSR)
    Ca = myinterp(inv_TSRdata_Ca,Ca_data,inv_TSR)
    eta = myinterp(inv_TSRdata_eta,eta_data,inv_TSR)

    factorA = torque_turbine/(Ct*0.5*rho*H_blade*chord*N_blade*R_turbine)
    factorB = (V_wec*A_wec / (A_turbine * (delta_P_turbine_guess/P_ambient + 1)^(1/lambda_air)))^2
    omega_turbine = 1/R_turbine * np.sqrt(factorA - factorB)

    delta_P_turbine = Ca * 0.5 * rho * ((V_wec*A_wec/(A_turbine * (delta_P_turbine_guess/P_ambient + 1)^(1/lambda_air)))**2 + (omega_turbine*R_turbine)**2) * H_blade * chord * N_blade / A_wec

    residual1 = delta_P_turbine_guess - delta_P_turbine
    residual2 = omega_turbine_guess - omega_turbine

    F_turbine = delta_P_turbine/A_turbine
    F_wec = F_turbine*A_wec/A_turbine

    power = delta_P_turbine * V_wec * A_wec * eta

return F_wec, power, residual1, residual2

def myfancyfunction(wec,#: TWEC,
        x_wec,#: ndarray,
        x_opt,#: ndarray,
        waves = None,
        nsubsteps = 1,
    ):

    if nsubsteps==1:
        tmat = wec.time_mat
    else:
        tmat = wec.time_mat_nsubsteps(nsubsteps)

    # Unstructured Controller Force
    ndof = 1
    x_opt = np.reshape(x_opt, (-1, ndof), order='F')
    omega_turbine = np.dot(tmat, x_opt)

    # WEC Velocty
    pos_wec = wec.vec_to_dofmat(x_wec)
    V_wec = np.dot(wec.derivative_mat, pos_wec)
    V_wec_td = np.dot(tmat, V_wec)
    assert V_wec_td.shape == (wec.nt*nsubsteps, wec.ndof)
    V_wec_td = np.expand_dims(np.transpose(V_wec_td), axis=0)


    # if callable(kinematics):
    #     def kinematics_fun(wec, x_wec, x_opt, waves, nsubsteps=1):
    #         pos_wec = wec.vec_to_dofmat(x_wec)
    #         # tmat = self._tmat(wec, nsubsteps)
    #         pos_wec_td = np.dot(tmat, pos_wec)
    #         return kinematics(pos_wec_td)
    # else:
    #     def kinematics_fun(wec, x_wec, x_opt, waves, nsubsteps=1):
    #         n = wec.nt*nsubsteps
    #         return np.repeat(kinematics[:, :, np.newaxis], n, axis=-1)
    kinematics = np.eye(ndof)
    n = wec.nt*nsubsteps
    kinematics_mat = np.repeat(kinematics[:, :, np.newaxis], n, axis=-1)#kinematics_fun(wec, x_wec, x_opt, waves, nsubsteps)
    V_wec = np.transpose(np.sum(kinematics_mat*V_wec_td, axis=1))

    # Turbine Power Considering Wec Avalaible Power from control and velocity
    return self_rectifying_turbine_omega_incompressible(omega_turbine,V_wec)

def mymechanical_average_power(wec,#: TWEC,
        x_wec,#: ndarray,
        x_opt,#: ndarray,
        waves = None,
        nsubsteps = 1,
    ) -> float:

    F_wec, power_td = myfancyfunction(wec,x_wec,x_opt,waves = waves,nsubsteps = nsubsteps)

    energy = np.sum(power_td) * wec.dt/nsubsteps
    return energy / wec.tf

def myforce_on_wec(wec,x_wec,x_opt,
    waves = None,
    nsubsteps = 1,
    ):

    F_wec, power_td = myfancyfunction(wec,x_wec,x_opt,waves = waves,nsubsteps = nsubsteps)

    return F_wec

wb = wot.geom.WaveBot()  # use standard dimensions
mesh_size_factor = 0.5 # 1.0 for default, smaller to refine mesh
mesh = wb.mesh(mesh_size_factor)
fb = cpy.FloatingBody.from_meshio(mesh, name="WaveBot")
fb.add_translation_dof(name="Heave")
ndof = fb.nb_dofs

stiffness = wot.hydrostatics.stiffness_matrix(fb).values
mass = wot.hydrostatics.inertia_matrix(fb).values

# fb.show_matplotlib()
# _ = wb.plot_cross_section(show=True)  # specific to WaveBot

f1 = 0.05
nfreq = 50
freq = wot.frequency(f1, nfreq, False) # False -> no zero frequency

bem_data = wot.run_bem(fb, freq)

name = ["PTO_Heave",]
# kinematics = np.eye(ndof)
# controller = None
# loss = None
# pto_impedance = None
# pto = wot.pto.PTO(ndof, kinematics, controller, pto_impedance, loss, name)

# PTO dynamics forcing function
f_add = {'PTO': myforce_on_wec}

# Constraint
f_max = 2000.0
nsubsteps = 4

def const_f_pto(wec, x_wec, x_opt, waves): # Format for scipy.optimize.minimize
    f = myforce_on_wec(wec, x_wec, x_opt, waves, nsubsteps)
    return f_max - np.abs(f.flatten())

ineq_cons = {'type': 'ineq',
             'fun': const_f_pto,
             }
constraints = [ineq_cons]

wec = wot.WEC.from_bem(
    bem_data,
    inertia_matrix=mass,
    hydrostatic_stiffness=stiffness,
    constraints=constraints,
    friction=None,
    f_add=f_add,
)

amplitude = 0.0625  
wavefreq = 0.3
phase = 30
wavedir = 0
waves = wot.waves.regular_wave(f1, nfreq, wavefreq, amplitude, phase, wavedir)

# make the pto just to get the post processing plotting
name = ["PTO_Heave",]
kinematics = np.eye(ndof)
controller = None
loss = None
pto_impedance = None
pto = wot.pto.PTO(ndof, kinematics, controller, pto_impedance, loss, name)

# obj_fun = pto.mechanical_average_power
obj_fun = mymechanical_average_power
nstate_opt = 2*nfreq+1

options = {'maxiter': 400, 'ftol': 1e-6}#, 'gtol': 1e-8}
scale_x_wec = 1e1
scale_x_opt = 1e-3
scale_obj = 1e-2
x_opt_0 = np.zeros(nstate_opt)+500.0
x_opt_0 = np.array([-1.35820313e+00,-1.35820313e+00,1.27395575e+00,3.40663821e-01,-1.60176907e-01,6.72369527e-01,5.85006393e-01,2.05316692e-01,7.67355542e-01,-4.02968817e-01,-2.56042235e-02,1.18714270e+03,-2.05955794e+03,2.10003941e-01,1.02685095e+00,1.95416862e-01,6.46744053e-01,3.67696994e-01,-2.31206444e-01,-2.66093903e-01,6.37978846e-01,-3.52209582e-01,1.54584573e-01,4.01602961e+00,-7.68086363e+01,5.20604122e-02,4.08862136e-02,-1.69685620e-01,-7.12413859e-02,3.96954721e-01,-9.04069030e-02,-3.47134718e-01,5.07550884e-01,-9.45214013e-01,-7.51856252e-01,4.47332655e+02,-9.51562665e+00,-1.08701100e-01,3.49702444e-01,-2.46220202e+00,-4.12677098e-02,-5.36790310e-01,-1.95200653e+00,-3.29747032e-01,-1.19942508e+00,-4.12288613e+00,1.22432656e+00,3.73045502e+01,3.15940280e+01,1.67030742e+00,2.86704131e+00,4.75081926e-02,-3.52248358e-01,9.77102632e-01,-7.97998069e-01,-7.95980662e-01,-1.40340607e-01,-3.86320537e-01,-4.74585983e-01,4.34203964e+01,5.31633277e+01,9.17683498e-01,5.60296892e-01,-8.41864196e-01,-3.61902981e+00,1.55309417e+00,-2.63564981e+00,1.57028794e+00,-2.29978747e-01,-4.17581485e+00,-2.75686009e+00,-3.71676537e+01,2.60487459e+01,-1.14666111e+00,3.23316149e+00,-1.13941058e+00,-5.48810103e-02,1.07021103e+00,2.10510855e-01,2.46316068e-01,-8.57633907e-01,-3.87906675e-01,-1.51653609e-01,5.08780704e+00,1.03028143e+00,4.07266194e-01,8.01792959e-01,1.59517656e+00,-8.56679154e-01,1.39065632e+00,-2.10597448e-01,1.36302416e-01,5.12913991e-01,8.07886103e-02,-2.10917559e+00,-1.42136131e+01,-1.34012039e+01,-1.11057334e+00,4.40781648e-01,-2.32283498e-01,-8.21216680e-01])
x_opt_0 = (x_opt_0+10.0)*50.0
results = wec.solve(
    waves, 
    obj_fun, 
    nstate_opt,
    optim_options=options,
    x_opt_0=x_opt_0, 
    scale_x_wec=scale_x_wec,
    scale_x_opt=scale_x_opt,
    scale_obj=scale_obj,
    )

opt_mechanical_average_power = results.fun
print(f'Optimal average mechanical power: {opt_mechanical_average_power} W')



nsubsteps = 5
pto_fdom, pto_tdom = pto.post_process(wec, results, waves, nsubsteps=nsubsteps)
wec_fdom, wec_tdom = wec.post_process(results, waves, nsubsteps=nsubsteps)

plt.figure()
pto_tdom['mech_power'].plot()
plt.show()

plt.figure()
wec_tdom['pos'].plot()
plt.show()

plt.figure()
pto_tdom['force'].plot()
plt.show()