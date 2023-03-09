


from typing import Optional, TypeVar, Callable, Union

import autograd.numpy as np
from autograd.builtins import isinstance, tuple, list, dict
from autograd.numpy import ndarray
from scipy.linalg import block_diag
from scipy.optimize import OptimizeResult
from xarray import DataArray, Dataset
from datetime import datetime
from scipy.optimize import OptimizeResult

from wecopttool.core import complex_to_real, td_to_fd
from wecopttool.core import dofmat_to_vec, vec_to_dofmat
from wecopttool.core import TWEC, TStateFunction, FloatOrArray


# type aliases
TPTO = TypeVar("TPTO", bound="PTO")
TLOSS = Callable[[FloatOrArray, FloatOrArray], FloatOrArray]


class PTO:
    """A power take-off (PTO) object to be used in conjunction with a
    :py:class:`wecopttool.WEC` object.
    """

    def __init__(self,
        ndof: int,
        kinematics: Union[TStateFunction, ndarray],
        controller: Optional[TStateFunction] = None,
        impedance: Optional[ndarray] = None,
        loss: Optional[TLOSS] = None,
        names: Optional[list[str]] = None,
    ) -> None:
        """Create a PTO object.

        The :py:class:`wecopttool.pto.PTO` class describes the
        kinematics, control logic, impedance and/or non-linear power
        loss of a power take-off system.
        The forces/moments applied by a
        :py:class:`wecopttool.pto.PTO` object can be applied to a
        :py:class:`wecopttool.WEC` object through the
        :py:attr:`wecopttool.WEC.f_add` property.
        The power produced by a :py:class:`wecopttool.pto.PTO` object
        can be used for the :python:`obj_fun` of pseudo-spectral
        optimization problem when calling
        :py:meth:`wecopttool.WEC.solve`.

        Parameters
        ----------
        ndof
            Number of degrees of freedom.
        kinematics
            Transforms state from WEC to PTO frame. May be a matrix
            (for linear kinematics) or function (for nonlinear
            kinematics).
        controller
            Function with signature
            :python:`def fun(wec, x_wec, x_opt, waves, nsubsteps):`
            or matrix with shape (PTO DOFs, WEC DOFs) that converts
            from the WEC DOFs to the PTO DOFs.
        impedance
            Matrix representing the PTO impedance.
        loss
            Function that maps flow and effort variables to a
            non-linear power loss.
            The output is the dissipated power (loss) in Watts.
            This should be a positive value.
        names
            PTO names.
        """
        self._ndof = ndof
        # names
        if names is None:
            names = [f'PTO_{i}' for i in range(ndof)]
        elif ndof == 1 and isinstance(names, str):
            names = [names]
        self._names = names
        # kinematics
        if callable(kinematics):
            def kinematics_fun(wec, x_wec, x_opt, waves, nsubsteps=1):
                pos_wec = wec.vec_to_dofmat(x_wec)
                tmat = self._tmat(wec, nsubsteps)
                pos_wec_td = np.dot(tmat, pos_wec)
                return kinematics(pos_wec_td)
        else:
            def kinematics_fun(wec, x_wec, x_opt, waves, nsubsteps=1):
                n = wec.nt*nsubsteps
                return np.repeat(kinematics[:, :, np.newaxis], n, axis=-1)
        self._kinematics = kinematics_fun
        # controller
        if controller is None:
            controller = controller_unstructured

        def force(wec, x_wec, x_opt, waves, nsubsteps=1):
            return controller(self, wec, x_wec, x_opt, waves, nsubsteps)

        self._force = force

        # power
        self._impedance = impedance
        self._loss = loss
        if impedance is not None:
            impedance_abcd = _make_abcd(impedance, ndof)
            self._transfer_mat = _make_mimo_transfer_mat(impedance_abcd, ndof)
        else:
            self._transfer_mat = None

    @property
    def ndof(self) -> int:
        """Number of degrees of freedom."""
        return self._ndof

    @property
    def names(self) -> ndarray:
        """DOF Names."""
        return self._names

    @property
    def kinematics(self) -> TStateFunction:
        """Kinematics function.
        """
        return self._kinematics

    @property
    def force(self) -> TStateFunction:
        """PTO force in PTO coordinates."""
        return self._force

    @property
    def impedance(self) -> ndarray:
        """Impedance matrix."""
        return self._impedance

    @property
    def loss(self) -> TLOSS:
        """Nonlinear power loss function with outputs in Watts."""
        return self._loss

    @property
    def transfer_mat(self) -> ndarray:
        """Transfer matrix."""
        return self._transfer_mat

    def _tmat(self, wec, nsubsteps: Optional[int] = 1):
        if nsubsteps==1:
            tmat = wec.time_mat
        else:
            tmat = wec.time_mat_nsubsteps(nsubsteps)
        return tmat

    def _fkinematics(self,
        f_wec: ndarray,
        wec: TWEC,
        x_wec: ndarray,
        x_opt: Optional[ndarray] = None,
        waves: Optional[Dataset] = None,
        nsubsteps: Optional[int] = 1,
    ) -> ndarray:
        """Return time-domain values in the PTO frame.

        Parameters
        ----------
        f_wec
            Fourier coefficients of some quantity "f" in the WEC frame.
        wec
            :py:class:`wecopttool.WEC` object.
        x_wec
            WEC dynamic state.
        x_opt
            Optimization (control) state.
        waves
            :py:class:`xarray.Dataset` with the structure and elements
            shown by :py:mod:`wecopttool.waves`.
        nsubsteps
            Number of steps between the default (implied) time steps.
            A value of :python:`1` corresponds to the default step
            length.
        """
        time_mat = self._tmat(wec, nsubsteps)
        f_wec_td = np.dot(time_mat, f_wec)
        assert f_wec_td.shape == (wec.nt*nsubsteps, wec.ndof)
        f_wec_td = np.expand_dims(np.transpose(f_wec_td), axis=0)
        kinematics_mat = self.kinematics(wec, x_wec, x_opt, waves, nsubsteps)
        return np.transpose(np.sum(kinematics_mat*f_wec_td, axis=1))

    def position(self,
        wec: TWEC,
        x_wec: ndarray,
        x_opt: ndarray,
        waves: Optional[Dataset] = None,
        nsubsteps: Optional[int] = 1,
    ) -> ndarray:
        """Calculate the PTO position time-series.

        Parameters
        ----------
        wec
            :py:class:`wecopttool.WEC` object.
        x_wec
            WEC dynamic state.
        x_opt
            Optimization (control) state.
        waves
            :py:class:`xarray.Dataset` with the structure and elements
            shown by :py:mod:`wecopttool.waves`.
        nsubsteps
            Number of steps between the default (implied) time steps.
            A value of :python:`1` corresponds to the default step
            length.
        """
        pos_wec = wec.vec_to_dofmat(x_wec)
        return self._fkinematics(pos_wec, wec, x_wec, x_opt, waves, nsubsteps)

    def velocity(self,
        wec: TWEC,
        x_wec: ndarray,
        x_opt: ndarray,
        waves: Optional[Dataset] = None,
        nsubsteps: Optional[int] = 1,
    ) -> ndarray:
        """Calculate the PTO velocity time-series.

        Parameters
        ----------
        wec
            :py:class:`wecopttool.WEC` object.
        x_wec
            WEC dynamic state.
        x_opt
            Optimization (control) state.
        waves
            :py:class:`xarray.Dataset` with the structure and elements
            shown by :py:mod:`wecopttool.waves`.
        nsubsteps
            Number of steps between the default (implied) time steps.
            A value of :python:`1` corresponds to the default step
            length.
        """
        pos_wec = wec.vec_to_dofmat(x_wec)
        vel_wec = np.dot(wec.derivative_mat, pos_wec)
        return self._fkinematics(vel_wec, wec, x_wec, x_opt, waves, nsubsteps)

    def acceleration(self,
        wec: TWEC,
        x_wec: ndarray,
        x_opt: ndarray,
        waves: Optional[Dataset] = None,
        nsubsteps: Optional[int] = 1,
    ) -> np.ndarray:
        """Calculate the PTO acceleration time-series.

        Parameters
        ----------
        wec
            :py:class:`wecopttool.WEC` object.
        x_wec
            WEC dynamic state.
        x_opt
            Optimization (control) state.
        waves
            :py:class:`xarray.Dataset` with the structure and elements
            shown by :py:mod:`wecopttool.waves`.
        nsubsteps
            Number of steps between the default (implied) time steps.
            A value of :python:`1` corresponds to the default step
            length.
        """
        pos_wec = wec.vec_to_dofmat(x_wec)
        vel_wec = np.dot(wec.derivative_mat, pos_wec)
        acc_wec = np.dot(wec.derivative_mat, vel_wec)
        return self._fkinematics(acc_wec, wec, x_wec, x_opt, waves, nsubsteps)

    def force_on_wec(self,
        wec: TWEC,
        x_wec: ndarray,
        x_opt: ndarray,
        waves: Optional[Dataset] = None,
        nsubsteps: Optional[int] = 1,
    ) -> ndarray:
        """Calculate the PTO force on WEC.

        Parameters
        ----------
        wec
            :py:class:`wecopttool.WEC` object.
        x_wec
            WEC dynamic state.
        x_opt
            Optimization (control) state.
        waves
            :py:class:`xarray.Dataset` with the structure and elements
            shown by :py:mod:`wecopttool.waves`.
        nsubsteps
            Number of steps between the default (implied) time steps.
            A value of :python:`1` corresponds to the default step
            length.
        """
        force_td = self.force(wec, x_wec, x_opt, waves, nsubsteps)
        assert force_td.shape == (wec.nt*nsubsteps, self.ndof)
        force_td = np.expand_dims(np.transpose(force_td), axis=0)
        assert force_td.shape == (1, self.ndof, wec.nt*nsubsteps)
        kinematics_mat = self.kinematics(wec, x_wec, x_opt, waves, nsubsteps)
        kinematics_mat = np.transpose(kinematics_mat, (1,0,2))
        return np.transpose(np.sum(kinematics_mat*force_td, axis=1))

    def brent(self, mya, myb, atol=2e-12, rtol=4*np.finfo(float).eps, maxiter=100): # args=()

        xpre = mya; xcur = myb
        # xblk = 0.0; fblk = 0.0; spre = 0.0; scur = 0.0
        error_num = "INPROGRESS"

        fpre = self.myf(xpre)#, args...)
        fcur = self.myf(xcur)#, args...)
        xblk = np.zeros_like(fpre)
        fblk = np.zeros_like(fpre)
        spre = np.zeros_like(fpre)
        scur = np.zeros_like(fpre)
        funcalls = 2
        iterations = 0
        
        if fpre*fcur > 0:
            error_num = "SIGNERR"
            return 0.0#, (iter=iterations, fcalls=funcalls, flag=error_num)
        
        if fpre == np.zeros_like(fpre):
            error_num = "CONVERGED"
            return xpre#, (iter=iterations, fcalls=funcalls, flag=error_num)
        
        if fcur == np.zeros_like(fcur):
            error_num = "CONVERGED"
            return xcur#, (iter=iterations, fcalls=funcalls, flag=error_num)
        

        for i in range(maxiter):
            iterations = iterations + 1
            if fpre*fcur < 0:
                xblk = xpre
                fblk = fpre
                spre = scur = xcur - xpre
            
            if np.abs(fblk) < np.abs(fcur):
                xpre = xcur
                xcur = xblk
                xblk = xpre

                fpre = fcur
                fcur = fblk
                fblk = fpre
            

            delta = (atol + rtol*np.abs(xcur))/2.0
            sbis = (xblk - xcur)/2.0
            if fcur == np.zeros_like(fcur) or np.abs(sbis) < delta:
                error_num = "CONVERGED"
                return xcur#, (iter=iterations, fcalls=funcalls, flag=error_num)
            

            if np.abs(spre) > delta and np.abs(fcur) < np.abs(fpre):
                if xpre == xblk:
                    # interpolate
                    stry = -fcur*(xcur - xpre)/(fcur - fpre)
                else:
                    # extrapolate
                    dpre = (fpre - fcur)/(xpre - xcur)
                    dblk = (fblk - fcur)/(xblk - xcur)
                    stry = -fcur*(fblk*dblk - fpre*dpre)/(dblk*dpre*(fblk - fpre))
                
                if 2*np.abs(stry) < np.minimum(np.abs(spre), 3*np.abs(sbis) - delta):
                    # good short step
                    spre = scur
                    scur = stry
                else:
                    # bisect
                    spre = sbis
                    scur = sbis
                
            else: 
                # bisect
                spre = sbis
                scur = sbis
            

            xpre = xcur; fpre = fcur
            if np.abs(scur) > delta:
                xcur = xcur + scur
            else:
                if sbis > 0:
                    xcur = xcur + delta #(sbis > 0 ? delta : -delta)
                else:
                    xcur = xcur - delta
            

            fcur = self.myf(xcur)#, args...)
            funcalls = funcalls + 1
        
        error_num = "CONVERR"
        return xcur#, (iter=iterations, fcalls=funcalls, flag=error_num)


    def self_rectifying_turbine_power_internalsolve(self,
            torque: ndarray,
            V_wec: ndarray, #m/s omega_turbine_guess: ndarray, #rad/s
        ) -> np.ndarray:#tuple[np.ndarray, np.ndarray]:
        """Calculate the turbine power, note that turbine inertia should be included in the drivetrain impedance

        Parameters
        ----------
        torque
            Torque on the turbine
        V_wec
            Velocity of the wec
        """
        # len xwec is set (ndof * ncomponents (2*nfreq+1))
        # len xopt is any length and be anything, but pto module has rules about shape etc.  
        # TODO: put in self
        rho = 1.225
        blade_chord = 0.054 #m
        N_blades = 30.0
        blade_radius = 0.298/2 #m
        blade_height = (1.0-0.7)*blade_radius #m
        A_wec = 1.0 #m2
        A_turb = 1.0 #m2
        P_ambient = 1.0 #atm

        power = np.zeros_like(torque)

        for ii in range(torque.size):
            def OWC_turbine(omega_turbine_guess):
                V_turb = A_wec*V_wec[ii]/A_turb
                power1 = torque[ii]*omega_turbine_guess
                V_turbcompressed = V_turb/((P_ambient+power1/(A_turb*V_turb))/P_ambient)**(1/1.401) #compressibility correction pv = nrt -> assume adiabiatic p2/p1 = (V1/V2)^specific_heat_ratio (1.401 for air for the operating temp we're at)

                inv_TSR = V_turbcompressed/(omega_turbine_guess*blade_radius)

                inv_TSRdata_Ct = np.array([-100,
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
                    100])

                Ct_data = np.array([0.375722776,
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
                    2.771242526])

                torque_coefficient = myinterp(inv_TSRdata_Ct,Ct_data,inv_TSR)
                # torque_coefficient = 0.2962*inv_TSR**4 - 1.7108*inv_TSR**3 + 2.9842*inv_TSR**2 - 0.4105*inv_TSR + 0.3721

                omega_turbine = np.sqrt(torque[ii]/(torque_coefficient*0.5*rho*blade_height*blade_chord*N_blades*blade_radius**3)-V_turbcompressed**2)
                
                inv_TSR = V_turbcompressed/(omega_turbine*blade_radius)

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

                power_coefficient = myinterp(inv_TSRdata_Ca,Ca_data,inv_TSR)
                # power_coefficient = 0.4093*inv_TSR**3 - 2.5095*inv_TSR**2 + 5.1228*inv_TSR - 0.0864

                power2 = power_coefficient*0.5*rho*(V_turbcompressed**2+(omega_turbine*blade_radius)**2)*blade_height*blade_chord*N_blades*V_turbcompressed

                OWCresidual = omega_turbine_guess-omega_turbine
                return OWCresidual

            def OWC_turbine2(omega_turbine_guess):
                V_turb = A_wec*V_wec[ii]/A_turb
                power1 = torque[ii]*omega_turbine_guess
                V_turbcompressed = V_turb/((P_ambient+power1/(A_turb*V_turb))/P_ambient)**(1/1.401) #compressibility correction pv = nrt -> assume adiabiatic p2/p1 = (V1/V2)^specific_heat_ratio (1.401 for air for the operating temp we're at)

                inv_TSR = V_turbcompressed/(omega_turbine_guess*blade_radius)

                inv_TSRdata_Ct = np.array([-100,
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
                    100])

                Ct_data = np.array([0.375722776,
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
                    2.771242526])

                torque_coefficient = myinterp(inv_TSRdata_Ct,Ct_data,inv_TSR)
                # torque_coefficient = 0.2962*inv_TSR**4 - 1.7108*inv_TSR**3 + 2.9842*inv_TSR**2 - 0.4105*inv_TSR + 0.3721

                omega_turbine = np.sqrt(torque[ii]/(torque_coefficient*0.5*rho*blade_height*blade_chord*N_blades*blade_radius**3)-V_turbcompressed**2)
                
                inv_TSR = V_turbcompressed/(omega_turbine*blade_radius)

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

                power_coefficient = myinterp(inv_TSRdata_Ca,Ca_data,inv_TSR)
                # power_coefficient = 0.4093*inv_TSR**3 - 2.5095*inv_TSR**2 + 5.1228*inv_TSR - 0.0864

                power2 = power_coefficient*0.5*rho*(V_turbcompressed**2+(omega_turbine*blade_radius)**2)*blade_height*blade_chord*N_blades*V_turbcompressed

                OWCresidual = omega_turbine_guess-omega_turbine
                return power2

            self.myf = OWC_turbine

            omega_star = self.brent(self,0.0,10000.0)
            power[ii] = OWC_turbine2[omega_star]
        return power

    def mechanical_power(self,
        wec: TWEC,
        x_wec: ndarray,
        x_opt: ndarray,
        waves: Optional[Dataset] = None,
        nsubsteps: Optional[int] = 1,
    ) -> np.ndarray:
        """Calculate the mechanical power time-series in each PTO DOF
        for a given system state.

        Parameters
        ----------
        wec
            :py:class:`wecopttool.WEC` object.
        x_wec
            WEC dynamic state.
        x_opt
            Optimization (control) state.
        waves
            :py:class:`xarray.Dataset` with the structure and elements
            shown by :py:mod:`wecopttool.waves`.
        nsubsteps
            Number of steps between the default (implied) time steps.
            A value of :python:`1` corresponds to the default step
            length.
        """
        force_td = self.force(wec, x_wec, x_opt, waves, nsubsteps)
        vel_td = self.velocity(wec, x_wec, x_opt, waves, nsubsteps)
        power = self.self_rectifying_turbine_power_internalsolve(force_td,vel_td)
        return power

    def mechanical_energy(self,
        wec: TWEC,
        x_wec: ndarray,
        x_opt: ndarray,
        waves: Optional[Dataset] = None,
        nsubsteps: Optional[int] = 1,
    ) -> float:
        """Calculate the mechanical energy in each PTO DOF for a given
        system state.

        Parameters
        ----------
        wec
            :py:class:`wecopttool.WEC` object.
        x_wec
            WEC dynamic state.
        x_opt
            Optimization (control) state.
        waves
            :py:class:`xarray.Dataset` with the structure and elements
            shown by :py:mod:`wecopttool.waves`.
        nsubsteps
            Number of steps between the default (implied) time steps.
            A value of :python:`1` corresponds to the default step
            length.
        """
        power_td = self.mechanical_power(wec, x_wec, x_opt, waves, nsubsteps)
        return np.sum(power_td) * wec.dt/nsubsteps

    def mechanical_average_power(self,
        wec: TWEC,
        x_wec: ndarray,
        x_opt: ndarray,
        waves: Optional[Dataset] = None,
        nsubsteps: Optional[int] = 1,
    ) -> float:
        """Calculate average mechanical power in each PTO DOF for a
        given system state.

        Parameters
        ----------
        wec
            :py:class:`wecopttool.WEC` object.
        x_wec
            WEC dynamic state.
        x_opt
            Optimization (control) state.
        waves
            :py:class:`xarray.Dataset` with the structure and elements
            shown by :py:mod:`wecopttool.waves`.
        nsubsteps
            Number of steps between the default (implied) time steps.
            A value of :python:`1` corresponds to the default step
            length.
        """
        energy = self.mechanical_energy(wec, x_wec, x_opt, waves, nsubsteps)
        return energy / wec.tf

    def power(self,
        wec: TWEC,
        x_wec: ndarray,
        x_opt: ndarray,
        waves: Optional[Dataset] = None,
        nsubsteps: Optional[int] = 1,
    ) -> ndarray:
        """Calculate the power time-series in each PTO DOF for a given
        system state.

        Parameters
        ----------
        wec
            :py:class:`wecopttool.WEC` object.
        x_wec
            WEC dynamic state.
        x_opt
            Optimization (control) state.
        waves
            :py:class:`xarray.Dataset` with the structure and elements
            shown by :py:mod:`wecopttool.waves`.
        nsubsteps
            Number of steps between the default (implied) time steps.
            A value of :python:`1` corresponds to the default step
            length.
        """
        # convert q1 (PTO velocity), e1 (PTO force) to q2, e2
        if self.impedance is not None:
            q1_td = self.velocity(wec, x_wec, x_opt, waves)
            e1_td = self.force(wec, x_wec, x_opt, waves)
            q1 = complex_to_real(td_to_fd(q1_td, False))
            e1 = complex_to_real(td_to_fd(e1_td, False))
            vars_1 = np.hstack([q1, e1])
            vars_1_flat = dofmat_to_vec(vars_1)
            vars_2_flat = np.dot(self.transfer_mat, vars_1_flat)
            vars_2 = vec_to_dofmat(vars_2_flat, 2*self.ndof)
            q2 = vars_2[:, :self.ndof]
            e2 = vars_2[:, self.ndof:]
            time_mat = self._tmat(wec, nsubsteps)
            q2_td = np.dot(time_mat, q2)
            e2_td = np.dot(time_mat, e2)
        else:
            q2_td = self.velocity(wec, x_wec, x_opt, waves, nsubsteps)
            e2_td = self.force(wec, x_wec, x_opt, waves, nsubsteps)
        # power
        power_out = q2_td * e2_td #, OWC_resid = self.self_rectifying_turbine_power(e2_td,q2_td,omega_turbine_guess)
        if self.loss is not None:
            power_out = power_out + self.loss(q2_td, e2_td)
        return power_out

    def energy(self,
        wec: TWEC,
        x_wec: ndarray,
        x_opt: ndarray,
        waves: Optional[Dataset] = None,
        nsubsteps: Optional[int] = 1,
    ) -> float:
        """Calculate the energy in each PTO DOF for a given system
        state.

        Parameters
        ----------
        wec
            :py:class:`wecopttool.WEC` object.
        x_wec
            WEC dynamic state.
        x_opt
            Optimization (control) state.
        waves
            :py:class:`xarray.Dataset` with the structure and elements
            shown by :py:mod:`wecopttool.waves`.
        nsubsteps
            Number of steps between the default (implied) time steps.
            A value of :python:`1` corresponds to the default step
            length.
        """
        power_td = self.power(wec, x_wec, x_opt, waves, nsubsteps)
        return np.sum(power_td) * wec.dt/nsubsteps

    def average_power(self,
        wec: TWEC,
        x_wec: ndarray,
        x_opt: ndarray,
        waves: Optional[Dataset] = None,
        nsubsteps: Optional[int] = 1,
    ) -> float:
        """Calculate the average power in each PTO DOF for a given
        system state.

        Parameters
        ----------
        wec
            :py:class:`wecopttool.WEC` object.
        x_wec
            WEC dynamic state.
        x_opt
            Optimization (control) state.
        waves
            :py:class:`xarray.Dataset` with the structure and elements
            shown by :py:mod:`wecopttool.waves`.
        nsubsteps
            Number of steps between the default (implied) time steps.
            A value of :python:`1` corresponds to the default step
            length.
        """
        energy = self.energy(wec, x_wec, x_opt, waves, nsubsteps)
        return energy / wec.tf

    def post_process(self,
        wec: TWEC,
        res: OptimizeResult,
        waves: Optional[DataArray] = None,
        nsubsteps: Optional[int] = 1,
    ) -> tuple[Dataset, Dataset]:
        """Transform the results from optimization solution to a form
        that the user can work with directly.

        Examples
        --------
        The :py:meth:`wecopttool.WEC.solve` method only returns the
        post-processed results for the :py:class:`wecopttool.WEC`
        object.

        >>> res_wec_fd, res_wec_td, res_opt = wec.solve(waves=wave,
                                              obj_fun=pto.average_power,
                                              nstate_opt=2*nfreq+1)

        To get the post-processed results for the
        :py:class:`wecopttool.pto.PTO`, you may call

        >>> res_pto_fd, res_pto_td = pto.post_process(wec,res_opt)

        For smoother plots, you can set :python:`nsubsteps` to a value
        greater than 1.

        >>> res_pto_fd, res_pto_td = pto.post_process(wec,res_opt,
                                                      nsubsteps=4)
        >>> res_pto_td.power.plot()

        Parameters
        ----------
        wec
            :py:class:`wecopttool.WEC` object.
        res
            Results produced by :py:func:`scipy.optimize.minimize`.
        waves
            :py:class:`xarray.Dataset` with the structure and elements
            shown by :py:mod:`wecopttool.waves`.
        nsubsteps
            Number of steps between the default (implied) time steps.
            A value of :python:`1` corresponds to the default step
            length.

        Returns
        -------
        results_fd
            :py:class:`xarray.Dataset` with frequency domain results.
        results_td
            :py:class:`xarray.Dataset` with time domain results.
        """
        create_time = f"{datetime.utcnow()}"

        x_wec, x_opt = wec.decompose_state(res.x)

        # position
        pos_td = self.position(wec, x_wec, x_opt, waves, nsubsteps)
        pos_fd = wec.td_to_fd(pos_td[::nsubsteps])

        # velocity
        vel_td = self.velocity(wec, x_wec, x_opt, waves, nsubsteps)
        vel_fd = wec.td_to_fd(vel_td[::nsubsteps])

        # acceleration
        acc_td = self.acceleration(wec, x_wec, x_opt, waves, nsubsteps)
        acc_fd = wec.td_to_fd(acc_td[::nsubsteps])

        # force
        force_td = self.force(wec, x_wec, x_opt, waves, nsubsteps)
        force_fd = wec.td_to_fd(force_td[::nsubsteps])

        # power
        power_td = self.power(wec, x_wec, x_opt, waves, nsubsteps)
        power_fd = wec.td_to_fd(power_td[::nsubsteps])

        # mechanical power
        mech_power_td = self.mechanical_power(wec, x_wec, x_opt, waves,
                                              nsubsteps)
        mech_power_fd = wec.td_to_fd(mech_power_td[::nsubsteps])

        pos_attr = {'long_name': 'Position', 'units': 'm or rad'}
        vel_attr = {'long_name': 'Velocity', 'units': 'm/s or rad/s'}
        acc_attr = {'long_name': 'Acceleration', 'units': 'm/s^2 or rad/s^2'}
        force_attr = {'long_name': 'Force or moment on WEC',
                      'units': 'N or Nm'}
        power_attr = {'long_name': 'Power', 'units': 'W'}
        mech_power_attr = {'long_name': 'Mechanical power', 'units': 'W'}
        omega_attr = {'long_name': 'Radial frequency', 'units': 'rad/s'}
        freq_attr = {'long_name': 'Frequency', 'units': 'Hz'}
        period_attr = {'long_name': 'Period', 'units': 's'}
        dof_attr = {'long_name': 'PTO degree of freedom'}
        time_attr = {'long_name': 'Time', 'units': 's'}

        t_dat = wec.time_nsubsteps(nsubsteps)

        results_fd = Dataset(
            data_vars={
                'pos': (['omega','dof'], pos_fd, pos_attr),
                'vel': (['omega','dof'], vel_fd, vel_attr),
                'acc': (['omega','dof'], acc_fd, acc_attr),
                'force': (['omega','dof'], force_fd, force_attr),
                'power': (['omega','dof'], power_fd, power_attr),
                'mech_power': (['omega','dof'], mech_power_fd, mech_power_attr)
            },
            coords={
                'omega':('omega', wec.omega, omega_attr),
                'freq':('omega', wec.frequency, freq_attr),
                'period':('omega', wec.period, period_attr),
                'dof':('dof', self.names, dof_attr)},
            attrs={"time_created_utc": create_time}
            )

        results_td = Dataset(
            data_vars={
                'pos': (['time','dof'], pos_td, pos_attr),
                'vel': (['time','dof'], vel_td, vel_attr),
                'acc': (['time','dof'], acc_td, acc_attr),
                'force': (['time','dof'], force_td, force_attr),
                'power': (['time','dof'], power_td, power_attr),
                'mech_power': (['time','dof'], mech_power_td, mech_power_attr)
            },
            coords={
                'time':('time', t_dat, time_attr),
                'dof':('dof', self.names, dof_attr)},
            attrs={"time_created_utc": create_time}
            )

        return results_fd, results_td


# power conversion chain
def _make_abcd(impedance: ndarray, ndof: int) -> ndarray:
    """Transform the impedance matrix into ABCD form from a MIMO
    transfer function.

    Parameters
    ----------
    impedance
        Matrix representing the PTO impedance.
        Size 2*n_dof.
    ndof
        Number of degrees of freedom.
    """
    z_11 = impedance[:ndof, :ndof, :]  # Fu
    z_12 = impedance[:ndof, ndof:, :]  # Fi
    z_21 = impedance[ndof:, :ndof, :]  # Vu
    z_22 = impedance[ndof:, ndof:, :]  # Vi
    z_12_inv = np.linalg.inv(z_12.T).T

    mmult = lambda a,b: np.einsum('mnr,mnr->mnr', a, b)
    abcd_11 = -1 * mmult(z_12_inv, z_11)
    abcd_12 = z_12_inv
    abcd_21 = z_21 - mmult(z_22, mmult(z_12_inv, z_11))
    abcd_22 = mmult(z_22, z_12_inv)
    return np.block([[[abcd_11], [abcd_12]], [[abcd_21], [abcd_22]]])


def _make_mimo_transfer_mat(
    impedance_abcd: ndarray,
    ndof: int,
) -> np.ndarray:
    """Create a block matrix of a MIMO transfer function.

    Parameters
    ----------
    impedance
        PTO impedance in ABCD form.
    ndof
        Number of degrees of freedom.
    """
    elem = [[None]*2*ndof for _ in range(2*ndof)]
    def block(re, im): return np.array([[re, -im], [im, re]])
    for idof in range(2*ndof):
        for jdof in range(2*ndof):
            Zp = impedance_abcd[idof, jdof, :]
            re = np.real(Zp)
            im = np.imag(Zp)
            blocks = [block(ire, iim) for (ire, iim) in zip(re, im)]
            blocks = [0.0] + blocks
            elem[idof][jdof] = block_diag(*blocks)
    return np.block(elem)


# controllers
def controller_unstructured(
    pto: TPTO,
    wec: TWEC,
    x_wec: ndarray,
    x_opt: ndarray,
    waves: Optional[Dataset] = None,
    nsubsteps: Optional[int] = 1,
) -> ndarray:
    """Unstructured numerical optimal controller that returns a time
    history of PTO forces.

    Parameters
    ----------
    pto
        :py:class:`wecopttool.pto.PTO` object.
    wec
        :py:class:`wecopttool.WEC` object.
    x_wec
        WEC dynamic state.
    x_opt
        Optimization (control) state.
    waves
        :py:class:`xarray.Dataset` with the structure and elements
        shown by :py:mod:`wecopttool.waves`.
    nsubsteps
        Number of steps between the default (implied) time steps.
        A value of :python:`1` corresponds to the default step
        length.
    """
    x_opt = np.reshape(x_opt, (-1, pto.ndof), order='F')
    tmat = pto._tmat(wec, nsubsteps)
    return np.dot(tmat, x_opt)


def controller_pid(
    pto: TPTO,
    wec: TWEC,
    x_wec: ndarray,
    x_opt: ndarray,
    waves: Optional[Dataset] = None,
    nsubsteps: Optional[int] = 1,
    proportional: Optional[bool] = True,
    integral: Optional[bool] = True,
    derivative: Optional[bool] = True,
) -> ndarray:
    """Proportional-integral-derivative (PID) controller that returns
    a time history of PTO forces.

    Parameters
    ----------
    pto
        :py:class:`wecopttool.pto.PTO` object.
    wec
        :py:class:`wecopttool.WEC` object.
    x_wec
        WEC dynamic state.
    x_opt
        Optimization (control) state.
    waves
        :py:class:`xarray.Dataset` with the structure and elements shown
        by :py:mod:`wecopttool.waves`.
    nsubsteps
            Number of steps between the default (implied) time steps.
            A value of :python:`1` corresponds to the default step
            length.
    proportional
        True to include proportional gain
    integral
        True to include integral gain
    derivative
        True to include derivative gain
    """
    ndof = pto.ndof
    force_td = np.zeros([wec.nt*nsubsteps, ndof])
    idx = 0

    def update_force_td(response):
        nonlocal idx, force_td
        gain = np.reshape(x_opt[idx*ndof:(idx+1)*ndof], [1, ndof])
        force_td = force_td + gain*response
        idx = idx + 1

    if proportional:
        vel_td = pto.velocity(wec, x_wec, x_opt, waves, nsubsteps)
        update_force_td(vel_td)
    if integral:
        pos_td = pto.position(wec, x_wec, x_opt, waves, nsubsteps)
        update_force_td(pos_td)
    if derivative:
        acc_td = pto.acceleration(wec, x_wec, x_opt, waves, nsubsteps)
        update_force_td(acc_td)
    return force_td


def controller_pi(
    pto: TPTO,
    wec: TWEC,
    x_wec: ndarray,
    x_opt: ndarray,
    waves: Optional[Dataset] = None,
    nsubsteps: Optional[int] = 1,
) -> ndarray:
    """Proportional-integral (PI) controller that returns a time
    history of PTO forces.

    Parameters
    ----------
    pto
        :py:class:`wecopttool.pto.PTO` object.
    wec
        :py:class:`wecopttool.WEC` object.
    x_wec
        WEC dynamic state.
    x_opt
        Optimization (control) state.
    waves
        :py:class:`xarray.Dataset` with the structure and elements shown
        by :py:mod:`wecopttool.waves`.
    nsubsteps
            Number of steps between the default (implied) time steps.
            A value of :python:`1` corresponds to the default step
            length.
    """
    force_td = controller_pid(pto, wec, x_wec, x_opt, waves, nsubsteps,
                              True, True, False)
    return force_td


def controller_p(
    pto: TPTO,
    wec: TWEC,
    x_wec: ndarray,
    x_opt: ndarray,
    waves: Optional[Dataset] = None,
    nsubsteps: Optional[int] = 1,
) -> ndarray:
    """Proportional (P) controller that returns a time history of
    PTO forces.

    Parameters
    ----------
    pto
        :py:class:`wecopttool.pto.PTO` object.
    wec
        :py:class:`wecopttool.WEC` object.
    x_wec
        WEC dynamic state.
    x_opt
        Optimization (control) state.
    waves
        :py:class:`xarray.Dataset` with the structure and elements shown
        by :py:mod:`wecopttool.waves`.
    nsubsteps
        Number of steps between the default (implied) time steps.
        A value of :python:`1` corresponds to the default step length.
    """
    force_td = controller_pid(pto, wec, x_wec, x_opt, waves, nsubsteps,
                               True, False, False)
    return force_td

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