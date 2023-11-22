""" Integration tests spanning WecOptTool.
"""
import pytest
from pytest import approx
import wecopttool as wot
import capytaine as cpy
import autograd.numpy as np
from scipy.optimize import Bounds
import xarray as xr


kplim = -1e1


@pytest.fixture(scope="module")
def f1():
    """Fundamental frequency [Hz]"""
    return 0.05


@pytest.fixture(scope="module")
def nfreq():
    """Number of frequencies in the array"""
    return 50

@pytest.fixture(scope='module')
def pto():
    """Basic PTO: unstructured, 1 DOF, mechanical power."""
    ndof = 1
    kinematics = np.eye(ndof)
    pto = wot.pto.PTO(ndof, kinematics)
    return pto


@pytest.fixture(scope='module')
def p_controller_pto():
    """Basic PTO: proportional (P) controller, 1 DOF, mechanical power."""
    ndof = 1
    pto = wot.pto.PTO(ndof=ndof, kinematics=np.eye(ndof),
                      controller=wot.pto.controller_p,
                      names=["P controller PTO"])
    return pto


@pytest.fixture(scope='module')
def pi_controller_pto():
    """Basic PTO: proportional-integral (PI) controller, 1 DOF, mechanical
    power."""
    ndof = 1
    pto = wot.pto.PTO(ndof=ndof, kinematics=np.eye(ndof),
                      controller=wot.pto.controller_pi,
                      names=["PI controller PTO"])
    return pto


@pytest.fixture(scope="module")
def fb():
    """Capytaine FloatingBody object"""
    try:
        import wecopttool.geom as geom
    except ImportError:
        pytest.skip(
            'Skipping integration tests due to missing optional geometry ' +
            'dependencies. Run `pip install wecopttool[geometry]` to run ' +
            'these tests.'
            )
    mesh_size_factor = 0.5
    wb = geom.WaveBot()
    mesh = wb.mesh(mesh_size_factor)
    fb = cpy.FloatingBody.from_meshio(mesh, name="WaveBot")
    fb.add_translation_dof(name="Heave")
    return fb


@pytest.fixture(scope="module")
def bem(f1, nfreq, fb):
    """Boundary elemement model (Capytaine) results"""
    freq = wot.frequency(f1, nfreq, False)
    return wot.run_bem(fb, freq)


@pytest.fixture(scope='module')
def regular_wave(f1, nfreq):
    """Single frequency wave"""
    wfreq = 0.3
    wamp = 0.0625
    wphase = 0
    wdir = 0
    waves = wot.waves.regular_wave(f1, nfreq, wfreq, wamp, wphase, wdir)
    return waves


@pytest.fixture(scope='module')
def irregular_wave(f1, nfreq):
    """Idealized (Pierson-Moskowitz) spectrum wave"""
    freq = wot.frequency(f1, nfreq, False)
    fp = 0.3
    hs = 0.0625*1.9
    spec = wot.waves.pierson_moskowitz_spectrum(freq, fp, hs)
    waves = wot.waves.long_crested_wave(spec)
    return waves


@pytest.fixture(scope='module')
def wec_from_bem(f1, nfreq, bem, fb, pto):
    """Simple WEC: 1 DOF, no constraints."""
    f_add = {"PTO": pto.force_on_wec}
    wec = wot.WEC.from_bem(bem, f_add=f_add)
    return wec


@pytest.fixture(scope='module')
def wec_from_floatingbody(f1, nfreq, fb, pto):
    """Simple WEC: 1 DOF, no constraints."""
    f_add = {"PTO": pto.force_on_wec}
    wec = wot.WEC.from_floating_body(fb, f1, nfreq, f_add=f_add)
    return wec


@pytest.fixture(scope='module')
def wec_from_impedance(bem, pto, fb):
    """Simple WEC: 1 DOF, no constraints."""
    bemc = bem.copy()
    omega = bemc['omega'].values
    w = np.expand_dims(omega, [1,2])
    A = bemc['added_mass'].values
    B = bemc['radiation_damping'].values
    fb.center_of_mass = [0, 0, 0]
    fb.rotation_center = fb.center_of_mass
    fb = fb.copy(name=f"{fb.name}_immersed").keep_immersed_part()
    mass = bemc['inertia_matrix'].values
    hstiff = bemc['hydrostatic_stiffness'].values
    K = np.expand_dims(hstiff, 2)
    
    freqs = omega / (2 * np.pi)
    impedance = (A + mass)*(1j*w) + B + K/(1j*w)
    exc_coeff = bem['Froude_Krylov_force'] + bem['diffraction_force']
    f_add = {"PTO": pto.force_on_wec}

    wec = wot.WEC.from_impedance(freqs, impedance, exc_coeff, hstiff, f_add)
    return wec


@pytest.fixture(scope='module')
def resonant_wave(f1, nfreq, fb, bem):
    """Regular wave at natural frequency of the WEC"""
    hd = wot.add_linear_friction(bem)
    Zi = wot.hydrodynamic_impedance(hd)
    wn = Zi['omega'][np.abs(Zi).argmin(dim='omega')].item()
    waves = wot.waves.regular_wave(f1, nfreq, freq=wn/2/np.pi, amplitude=0.1)
    return waves


def test_solve_callback(wec_from_bem, regular_wave, pto, nfreq, capfd):
    """Check that user can set a custom callback"""

    cbstring = 'hello world!'

    def my_callback(my_wec, x_wec, x_opt, wave):
        print(cbstring)

    _ = wec_from_bem.solve(regular_wave,
                           obj_fun=pto.average_power,
                           nstate_opt=2*nfreq,
                           scale_x_wec=1.0,
                           scale_x_opt=0.01,
                           scale_obj=1e-1,
                           callback=my_callback,
                           optim_options={'maxiter': 1})

    out, err = capfd.readouterr()

    assert out.split('\n')[0] == cbstring


@pytest.mark.parametrize("bounds_opt",
                         [Bounds(lb=kplim, ub=0), ((kplim, 0),)])
def test_solve_bounds(bounds_opt, wec_from_bem, regular_wave,
                      p_controller_pto):
    """Confirm that bounds are not violated and scale correctly when
    passing bounds argument as both as Bounds object and a tuple"""

    # replace unstructured controller with proportional controller
    wec_from_bem.forces['PTO'] = p_controller_pto.force_on_wec

    res = wec_from_bem.solve(waves=regular_wave,
                             obj_fun=p_controller_pto.average_power,
                             nstate_opt=1,
                             x_opt_0=[kplim*0.1],
                             optim_options={'maxiter': 2e1,
                                            'ftol': 1e-8},
                             bounds_opt=bounds_opt,
                             )

    assert pytest.approx(kplim, 1e-5) == res['x'][-1]


def test_same_wec_init(wec_from_bem,
                       wec_from_floatingbody,
                       wec_from_impedance,
                       f1,
                       nfreq):
    """Test that different init methods for WEC class produce the same object
    """

    waves = wot.waves.regular_wave(f1, nfreq, 0.3, 0.0625)
    np.random.seed(0)
    x_wec_0 = np.random.randn(wec_from_bem.nstate_wec)
    np.random.seed(1)
    x_opt_0 = np.random.randn(wec_from_bem.nstate_wec)
    bem_res = wec_from_bem.residual(x_wec_0, x_opt_0, waves)
    fb_res = wec_from_floatingbody.residual(x_wec_0, x_opt_0, waves)
    imp_res = wec_from_impedance.residual(x_wec_0, x_opt_0, waves)
    
    assert fb_res == approx(bem_res, rel=0.01)
    assert imp_res == approx(bem_res, rel=0.01)


class TestTheoreticalPowerLimits:
    """Compare power from numerical solutions against known theoretical limits
    """

    @pytest.fixture(scope='class')
    def mass(self, fb):
        """Rigid-body mass"""
        return fb.compute_rigid_body_inertia()

    @pytest.fixture(scope='class')
    def hstiff(self, fb):
        """Hydrostatic stiffness"""
        return fb.compute_hydrostatic_stiffness()

    @pytest.fixture(scope='class')
    def hydro_impedance(self, bem):
        """Intrinsic hydrodynamic impedance"""
        hd = wot.add_linear_friction(bem)
        hd = wot.check_linear_damping(hd)
        Zi = wot.hydrodynamic_impedance(hd)
        return Zi

    def test_p_controller_resonant_wave(self,
                                        bem,
                                        resonant_wave,
                                        p_controller_pto,
                                        hydro_impedance):
        """Proportional controller should match optimum for natural resonant
        wave"""
        
        f_add = {"PTO": p_controller_pto.force_on_wec}
        wec = wot.WEC.from_bem(bem, f_add=f_add)

        res = wec.solve(waves=resonant_wave,
                        obj_fun=p_controller_pto.average_power,
                        nstate_opt=1,
                        x_wec_0=1e-1*np.ones(wec.nstate_wec),
                        x_opt_0=[-1470],
                        scale_x_wec=1e2,
                        scale_x_opt=1e-3,
                        scale_obj=1e-1,
                        optim_options={'ftol': 1e-10},
                        bounds_opt=((-1*np.infty, 0),),
                        )

        power_sol = -1*res['fun']

        res_fd, _ = wec.post_process(res, resonant_wave,nsubsteps=1)
        Fex = res_fd.force.sel(
            type=['Froude_Krylov', 'diffraction']).sum('type')
        power_optimal = (np.abs(Fex)**2/8 / np.real(hydro_impedance.squeeze())
                         ).squeeze().sum('omega').item()

        assert power_sol == approx(power_optimal, rel=0.03)

    def test_pi_controller_regular_wave(self,
                                        bem,
                                        regular_wave,
                                        pi_controller_pto,
                                        hydro_impedance):
        """PI controller matches optimal for any regular wave"""

        f_add = {"PTO": pi_controller_pto.force_on_wec}
        wec = wot.WEC.from_bem(bem, f_add=f_add)

        res = wec.solve(waves=regular_wave,
                        obj_fun=pi_controller_pto.average_power,
                        nstate_opt=2,
                        x_wec_0=1e-1*np.ones(wec.nstate_wec),
                        x_opt_0=[-1e3, 1e4],
                        scale_x_wec=1e2,
                        scale_x_opt=1e-3,
                        scale_obj=1e-2,
                        optim_options={'maxiter': 50},
                        bounds_opt=((-1e4, 0), (0, 2e4),)
                        )

        power_sol = -1*res['fun']

        res_fd, _ = wec.post_process(res, regular_wave, nsubsteps=1)
        Fex = res_fd.force.sel(
            type=['Froude_Krylov', 'diffraction']).sum('type')
        power_optimal = (np.abs(Fex)**2/8 / np.real(hydro_impedance.squeeze())
                         ).squeeze().sum('omega').item()

        assert power_sol == approx(power_optimal, rel=1e-4)
    def test_unstructured_controller_irregular_wave(self,
                                                    fb,
                                                    bem,
                                                    regular_wave,
                                                    pto,
                                                    nfreq,
                                                    hydro_impedance):
        """Unstructured (numerical optimal) controller matches optimal for any
        irregular wave when unconstrained"""

        f_add = {"PTO": pto.force_on_wec}
        wec = wot.WEC.from_bem(bem, f_add=f_add)

        res = wec.solve(waves=regular_wave,
                        obj_fun=pto.average_power,
                        nstate_opt=2*nfreq,
                        x_wec_0=1e-1*np.ones(wec.nstate_wec),
                        scale_x_wec=1e2,
                        scale_x_opt=1e-2,
                        scale_obj=1e-2,
                        )

        power_sol = -1*res['fun']

        res_fd, _ = wec.post_process(res, regular_wave, nsubsteps=1)
        Fex = res_fd.force.sel(
            type=['Froude_Krylov', 'diffraction']).sum('type')
        power_optimal = (np.abs(Fex)**2/8 / np.real(hydro_impedance.squeeze())
                         ).squeeze().sum('omega').item()

        assert power_sol == approx(power_optimal, rel=1e-3)

    def test_saturated_pi_controller(self,
                                    bem,
                                    regular_wave,
                                    pto,
                                    nfreq):
        """Saturated PI controller matches constrained unstructured controller
        for a regular wave
        """

        pto_tmp = pto
        pto = {}
        wec = {}
        nstate_opt = {}
        
        # Constraint
        f_max = 2000.0

        nstate_opt['us'] = 2*nfreq
        pto['us'] = pto_tmp
        def const_f_pto(wec, x_wec, x_opt, waves):
            f = pto['us'].force_on_wec(wec, x_wec, x_opt, waves, 
                                       nsubsteps=4)
            return f_max - np.abs(f.flatten())
        wec['us'] = wot.WEC.from_bem(bem,
                                     f_add={"PTO": pto['us'].force_on_wec},
                                     constraints=[{'type': 'ineq',
                                                   'fun': const_f_pto, }])
        
        
        ndof = 1
        nstate_opt['pi'] = 2
        def saturated_pi(pto, wec, x_wec, x_opt, waves=None, nsubsteps=1):
            return wot.pto.controller_pi(pto, wec, x_wec, x_opt, waves, 
                                         nsubsteps, 
                                         saturation=[-f_max, f_max])
        pto['pi'] = wot.pto.PTO(ndof=ndof,
                                kinematics=np.eye(ndof),
                                controller=saturated_pi,)
        wec['pi'] = wot.WEC.from_bem(bem,
                                     f_add={"PTO": pto['pi'].force_on_wec},
                                     constraints=[])
        
        x_opt_0 = {'us': np.ones(nstate_opt['us'])*0.1,
                   'pi': [-1e3, 1e4]}
        scale_x_wec = {'us': 1e1,
                       'pi': 1e1}
        scale_x_opt = {'us': 1e-3,
                       'pi': 1e-3}
        scale_obj = {'us': 1e-2,
                     'pi': 1e-2}
        bounds_opt = {'us': None,
                      'pi': ((-1e4, 0), (0, 2e4),)}
        
        res = {}
        pto_fdom = {}
        pto_tdom = {}
        for key in wec.keys():
            res[key] = wec[key].solve(waves=regular_wave,
                            obj_fun=pto[key].average_power,
                            nstate_opt=nstate_opt[key],
                            x_wec_0=1e-1*np.ones(wec[key].nstate_wec),
                            x_opt_0=x_opt_0[key],
                            scale_x_wec=scale_x_wec[key],
                            scale_x_opt=scale_x_opt[key],
                            scale_obj=scale_obj[key],
                            optim_options={'maxiter': 200},
                            bounds_opt=bounds_opt[key]
                            )
            
            nsubstep_postprocess = 4
            pto_fdom[key], pto_tdom[key] = pto[key].post_process(wec[key], 
                                                                 res[key], 
                                                                 regular_wave, 
                                                                 nsubstep_postprocess)
        
        xr.testing.assert_allclose(pto_tdom['pi'].power.squeeze().mean('time'), 
                                   pto_tdom['us'].power.squeeze().mean('time'),
                                   rtol=1e-1)
        
        xr.testing.assert_allclose(pto_tdom['us'].force.max(),
                                   pto_tdom['pi'].force.max())
