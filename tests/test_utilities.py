""" Unit tests for functions in the :python:`utilities.py` module.
"""

import pytest
import numpy as np
import xarray as xr
from matplotlib.pyplot import Figure, Axes
import wecopttool as wot
from pytest import approx
import capytaine as cpy



# test function in the utilities.py



@pytest.fixture(scope="module")
def power_flows():
    """Dictionary of power flows."""
    pflows = {'Optimal Excitation': -100,
                'Radiated': -20,
                'Actual Excitation': -70,
                'Electrical (solver)': -40,
                'Mechanical (solver)': -50,
                'Absorbed': -50,
                'Unused Potential': -30,
                'PTO Loss': -10
    }
    return pflows

@pytest.fixture(scope="module")
def f1():
    """Fundamental frequency [Hz]."""
    return 0.1


@pytest.fixture(scope="module")
def nfreq():
    """Number of frequencies in frequency vector."""
    return 5

@pytest.fixture(scope="module")
def ndof():
    """Number of degrees of freedom."""
    return 2

@pytest.fixture(scope="module")
def ndir():
    """Number of wave directions."""
    return 3

@pytest.fixture(scope='module')
def bem_data(f1, nfreq, ndof, ndir):
    """Synthetic BEM data."""
    # TODO - start using single BEM solution across entire test suite
    coords = {
        'omega': [2*np.pi*(ifreq+1)*f1 for ifreq in range(nfreq)],
        'influenced_dof': [f'DOF_{idof+1}' for idof in range(ndof)],
        'radiating_dof': [f'DOF_{idof+1}' for idof in range(ndof)],
        'wave_direction': [2*np.pi/ndir*idir for idir in range(ndir)],
    }
    radiation_dims = ['omega', 'radiating_dof', 'influenced_dof']
    excitation_dims = ['omega', 'influenced_dof', 'wave_direction']
    hydrostatics_dims = ['radiating_dof', 'influenced_dof']

    added_mass = np.ones([nfreq, ndof, ndof])
    radiation_damping = np.ones([nfreq, ndof, ndof])
    diffraction_force = np.ones([nfreq, ndof, ndir], dtype=complex) + 1j
    Froude_Krylov_force = np.ones([nfreq, ndof, ndir], dtype=complex) + 1j
    inertia_matrix = np.ones([ndof, ndof])
    hydrostatic_stiffness = np.ones([ndof, ndof])

    data_vars = {
        'added_mass': (radiation_dims, added_mass),
        'radiation_damping': (radiation_dims, radiation_damping),
        'diffraction_force': (excitation_dims, diffraction_force),
        'Froude_Krylov_force': (excitation_dims, Froude_Krylov_force),
        'inertia_matrix': (hydrostatics_dims, inertia_matrix),
        'hydrostatic_stiffness': (hydrostatics_dims, hydrostatic_stiffness)
    }
    return xr.Dataset(data_vars=data_vars, coords=coords)

@pytest.fixture(scope='module')
def intrinsic_impedance(bem_data):
    bem_data = wot.add_linear_friction(bem_data)
    intrinsic_impedance = wot.hydrodynamic_impedance(bem_data)
    return intrinsic_impedance

@pytest.fixture(scope='module')
def pi_controller_pto():
    """Basic PTO: proportional-integral (PI) controller, 1DOF, mechanical
    power."""
    ndof = 1
    pto = wot.pto.PTO(ndof=ndof, kinematics=np.eye(ndof),
                      controller=wot.controllers.pid_controller(1,True,True,False),
                      names=["PI controller PTO"])
    return pto

@pytest.fixture(scope='module')
def regular_wave(f1, nfreq):
    """Single frequency wave"""
    wfreq = 0.3
    wamp = 0.0625
    wphase = 0
    wdir = 0
    waves = wot.waves.regular_wave(f1, nfreq, wfreq, wamp, wphase, wdir)
    return waves

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
def wb_bem(f1, nfreq, fb):
    """Boundary elemement model (Capytaine) results"""
    freq = wot.frequency(f1, nfreq, False)
    return wot.run_bem(fb, freq)

@pytest.fixture(scope='class')
def wb_hydro_impedance(wb_bem):
    """Intrinsic hydrodynamic impedance"""
    hd = wot.add_linear_friction(wb_bem)
    hd = wot.check_radiation_damping(hd)
    Zi = wot.hydrodynamic_impedance(hd)
    return Zi




def test_plot_hydrodynamic_coefficients(bem_data,ndof):
    bem_figure_list = wot.utilities.plot_hydrodynamic_coefficients(bem_data)
    correct_len = ndof*(ndof+1)/2   #using only the subdiagonal elements
    #added mass
    fig_am = bem_figure_list[0][0]
    assert correct_len == len(fig_am.axes)
    assert isinstance(fig_am,Figure)
    #radiation damping
    fig_rd = bem_figure_list[1][0]
    assert correct_len == len(fig_rd.axes)
    assert isinstance(fig_rd,Figure)
    #radiation damping
    fig_ex = bem_figure_list[2][0]
    assert ndof == len(fig_ex.axes)
    assert isinstance(fig_ex,Figure)

def test_plot_bode_impedance(intrinsic_impedance, ndof):
    fig_Zi, axes_Zi = wot.utilities.plot_bode_impedance(intrinsic_impedance)

    assert 2*ndof*ndof == len(fig_Zi.axes)
    assert isinstance(fig_Zi,Figure)
    assert all([isinstance(ax, Axes) for ax in np.reshape(axes_Zi,-1)])


def test_plot_power_flow(power_flows):
    fig_sankey, ax_sankey = wot.utilities.plot_power_flow(power_flows)

    assert isinstance(fig_sankey, Figure)
    assert isinstance(ax_sankey, Axes)

def test_calculate_power_flow(wb_bem,
                              regular_wave,
                              pi_controller_pto,
                              wb_hydro_impedance):
    """PI controller matches optimal for any regular wave,
        thus we check if the radiated power is equal the absorber power
        and if the Optimal excitation is equal the actual excitation"""

    f_add = {"PTO": pi_controller_pto.force_on_wec}
    wec = wot.WEC.from_bem(wb_bem, f_add=f_add)

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

    pflows = wot.utilities.calculate_power_flows(wec,
                          pi_controller_pto,
                          res,
                          regular_wave,
                          wb_hydro_impedance)

    assert pflows['Absorbed'] == approx(pflows['Radiated'], rel=1e-4)
    assert pflows['Optimal Excitation'] == approx(pflows['Actual Excitation'], rel=1e-4)

def test_linear_solve(wb_bem, regular_wave):
    omega = wb_bem.omega.values
    gear_ratio = 12.0
    torque_constant = 6.7
    winding_resistance = 0.5
    winding_inductance = 0.0
    drivetrain_inertia = 2.0
    drivetrain_friction = 1.0
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
    pto_impedance = np.array([[pto_impedance_11, pto_impedance_12],
                                [pto_impedance_21, pto_impedance_22]])

    power, _, _, _ = wot.utilities.linear_solve(wb_bem, pto_impedance, regular_wave.isel(realization=0), np.eye(1))
    assert power == approx(-29.2, abs=0.05)
