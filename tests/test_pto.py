""" Unit tests for functions in the :python:`pto.py` module.
"""

import pytest
import numpy as np

import wecopttool as wot


# setup a simple PTO based on the WaveBot device
# TODO: replace with multi-DOF PTO
@pytest.fixture(scope="module")
def ndof():
    """Number of PTO degrees of freedom."""
    return 1


@pytest.fixture(scope="module")
def nfreq():
    """Number of frequencies (harmonics)."""
    return 2


@pytest.fixture(scope="module")
def f1():
    """Fundamental frequency [Hz]."""
    return 0.1


@pytest.fixture(scope="module")
def freq(f1, nfreq):
    """Frequency array [Hz]."""
    return np.arange(nfreq+1)*f1


@pytest.fixture(scope="module")
def omega(freq):
    """Radial frequency array [rad/s]."""
    return freq * 2*np.pi


@pytest.fixture(scope="module")
def pto_parameters():
    """Dictionary of PTO parameters."""
    params = {
        'gear_ratio': 12.0,  # [rad/m]
        'torque_constant': 6.7,  # [N*m/A]
        'winding_resistance': 0.5,  # [ohm]
        'winding_inductance': 0.0,  # [H]
        'drivetrain_inertia': 2.0,  # [kg*m^2]
        'drivetrain_friction': 1.0,  # [N*m*s/rad]
        'drivetrain_stiffness': 0.0,  # [N*m/rad]
    }
    return params


@pytest.fixture(scope="module")
def drivetrain_impedance(omega, pto_parameters):
    """Drive-train impedance."""
    imp = (
        1j*omega[1:]*pto_parameters['drivetrain_inertia'] +
        pto_parameters['drivetrain_friction'] +
        1/(1j*omega[1:])*pto_parameters['drivetrain_stiffness']
    )
    return imp


@pytest.fixture(scope="module")
def winding_impedance(omega, pto_parameters):
    """Generator winding impedance."""
    imp = (
        pto_parameters['winding_resistance'] +
        1j*omega[1:]*pto_parameters['winding_inductance']
    )
    return imp


@pytest.fixture(scope="module")
def pto_impedance(
        nfreq, pto_parameters, drivetrain_impedance, winding_impedance
    ):
    """PTO impedance matrix."""
    tc = pto_parameters['torque_constant']
    gr = pto_parameters['gear_ratio']
    # diagonals
    pto_impedance_11 = -1* gr**2 * drivetrain_impedance
    pto_impedance_22 = winding_impedance
    # off-diagonals
    off_diag = np.sqrt(3.0/2.0) * tc * gr
    pto_impedance_12 = -1*(off_diag+0j) * np.ones([nfreq])
    pto_impedance_21 = -1*(off_diag+0j) * np.ones([nfreq])
    # matrix
    imp = np.array([[pto_impedance_11, pto_impedance_12],
                    [pto_impedance_21, pto_impedance_22]])
    return imp


@pytest.fixture(scope="module")
def abcd(pto_impedance, ndof):
    return wot.pto._make_abcd(pto_impedance, ndof)


class TestSupportFunctions:
    """Test the hidden support functions in the :python:`pto` module."""

    def test_make_abcd(
            self, nfreq, pto_parameters, drivetrain_impedance,
            winding_impedance, abcd,
        ):
        """Test the function :python:`pto._make_abcd`"""
        tc = pto_parameters['torque_constant']
        gr = pto_parameters['gear_ratio']
        expected = np.zeros([2, 2, nfreq], complex)
        expected[0, 0, :] = -np.sqrt(2/3)*gr/tc*drivetrain_impedance
        expected[1, 1, :] = -np.sqrt(2/3)/gr/tc*winding_impedance
        expected[0, 1, :] = -np.sqrt(2/3)/gr/tc
        expected[1, 0, :] = (
            -np.sqrt(3/2)*gr*tc -
            np.sqrt(2/3)*gr/tc*winding_impedance*drivetrain_impedance
        )
        assert np.allclose(abcd, expected)

    def test_make_mimo_transfer_mat(self, abcd, ndof, nfreq):
        """Test the function :python:`pto._mimo_transfer_mat`"""
        mimo = wot.pto._make_mimo_transfer_mat(abcd, ndof)
        n = 2*nfreq
        expected = np.zeros([2*ndof*n, 2*ndof*n])
        # 0,0
        imp = abcd[0, 0, :]
        r0 = np.real(imp[0])
        i0 = np.imag(imp[0])
        r1 = np.real(imp[1])
        expected[:n, :n] = np.array([
            [r0, 0,    0, 0],
            [0, r0, -i0, 0],
            [0, i0,  r0, 0],
            [0, 0,    0, r1],
        ])
        # 0,1
        imp = abcd[0, 1, :]
        r0 = np.real(imp[0])
        i0 = np.imag(imp[0])
        r1 = np.real(imp[1])
        expected[:n, n:] = np.array([
            [r0, 0,    0, 0],
            [0, r0, -i0, 0],
            [0, i0,  r0, 0],
            [0, 0,    0, r1],
        ])
        # 1,0
        imp = abcd[1, 0, :]
        r0 = np.real(imp[0])
        i0 = np.imag(imp[0])
        r1 = np.real(imp[1])
        expected[n:, :n] = np.array([
            [r0, 0,    0, 0],
            [0, r0, -i0, 0],
            [0, i0,  r0, 0],
            [0, 0,    0, r1],
        ])
        # 1,1
        imp = abcd[1, 1, :]
        r0 = np.real(imp[0])
        i0 = np.imag(imp[0])
        r1 = np.real(imp[1])
        expected[n:, n:] = np.array([
            [r0, 0,    0, 0],
            [0, r0, -i0, 0],
            [0, i0,  r0, 0],
            [0, 0,    0, r1],
        ])
        # test
        assert np.allclose(mimo, expected)


class TestControllers:
    """Test the different sample controllers implemented in the
    :python:`pto` module."""

    @pytest.fixture(scope="class")
    def wec(self, f1, nfreq):
        """Empty WEC object."""
        return wot.WEC(f1, nfreq, {}, ndof=1, inertia_in_forces=True)

    @pytest.fixture(scope="class")
    def kinematics(self,):
        """PTO kinematics matrix."""
        return np.eye(1)

    @pytest.fixture(scope="class")
    def pid_p(self,):
        """PID controller proportional gains."""
        return 2.1

    @pytest.fixture(scope="class")
    def pid_i(self,):
        """PID controller integral gains."""
        return 3.4

    @pytest.fixture(scope="class")
    def pid_d(self,):
        """PID controller derivative gains."""
        return 2.1

    def test_controller_unstructured(self, wec, ndof, kinematics, omega):
        """Test the pseudo-spectral controller."""
        controller = wot.pto.controller_unstructured
        pto = wot.pto.PTO(ndof, kinematics, controller)
        amp = 1.2
        w = omega[-2]
        force = amp * np.cos(w * wec.time)
        force = force.reshape(-1, 1)
        x_opt = [0, amp, 0, 0]
        calculated = pto.force(wec, None, x_opt, None)
        assert np.allclose(force, calculated)

    def test_controller_p(self, wec, ndof, kinematics, omega, pid_p):
        """Test the proportional (P) controller."""
        controller = wot.pto.controller_p
        pto = wot.pto.PTO(ndof, kinematics, controller)
        amp = 2.3
        w = omega[-2]
        # pos = amp * np.cos(w * wec.time)
        vel = -1 * amp * w * np.sin(w * wec.time)
        force = vel*pid_p
        force = force.reshape(-1, 1)
        x_wec = [0, amp, 0, 0]
        x_opt = [pid_p,]
        calculated = pto.force(wec, x_wec, x_opt, None)
        assert np.allclose(force, calculated)

    def test_controller_pi(self, wec, ndof, kinematics, omega, pid_p, pid_i):
        """Test the PI controller."""
        controller = wot.pto.controller_pi
        pto = wot.pto.PTO(ndof, kinematics, controller)
        amp = 2.3
        w = omega[-2]
        pos = amp * np.cos(w * wec.time)
        vel = -1 * amp * w * np.sin(w * wec.time)
        force = vel*pid_p + pos*pid_i
        force = force.reshape(-1, 1)
        x_wec = [0, amp, 0, 0]
        x_opt = [pid_p, pid_i]
        calculated = pto.force(wec, x_wec, x_opt, None)
        assert np.allclose(force, calculated)

    def test_controller_pid(
            self, wec, ndof, kinematics, omega, pid_p, pid_i, pid_d,
        ):
        """Test the PID controller."""
        controller = wot.pto.controller_pid
        pto = wot.pto.PTO(ndof, kinematics, controller)
        amp = 2.3
        w = omega[-2]
        pos = amp * np.cos(w * wec.time)
        vel = -1 * amp * w * np.sin(w * wec.time)
        acc = -1 * amp * w**2 * np.cos(w * wec.time)
        force = vel*pid_p + pos*pid_i + acc*pid_d
        force = force.reshape(-1, 1)
        x_wec = [0, amp, 0, 0]
        x_opt = [pid_p, pid_i, pid_d]
        calculated = pto.force(wec, x_wec, x_opt, None)
        assert np.allclose(force, calculated)