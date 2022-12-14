Implementation
==============

As described in the :doc:`theory` page, WecOptTool operates by solving a constrained optimization problem, in which an equality constraint (:math:`r(x) = 0`) is used to enforce the dynamics of the system.

.. math::
    \min\limits_x J(x) \\
    \textrm{s.t.}& \\
    r(x) &= 0 \\
    c_{ineq}(x) &\leq 0 \\
    c_{eq}(x) &= 0
    :label: optim_prob_implementation

For the linear case, which is the default formulation in WecOptTool, the WEC dynamics can be expressed, in residual form, as

.. math::
    r = -(M + m_a) \ddot{x} - k x(t) - B \, \dot{x}(t) - C \int\limits_{-\infty}^t \dot{x}(t)  + f_{ext} + f_{exc} = 0.
    :label: wec_dynamics

The decision variable, :math:`x`, includes states of the body motion as well as other external states, e.g., related to controllers, PTO systems, mooring systems, etc.

.. math::
    x = [ x_{pos}, x_{ext} ]^T
    :label: decision_variable_top

Here, :math:`x_{pos} \in \mathbb{R}^{m(2n + 1)}`, where :math:`m` is the number of body modes of motion and :math:`n` is the number of frequency components.
Similarly, :math:`x_{ext} \in \mathbb{R}^{p(2n+1)}`, where :math:`p` is the number of external states.
Body modes could include multiple degrees-of-freedom for a single body (e.g., heave, surge, and pitch) and/or degrees-of-freedom from multiple bodies (e.g., heave modes from each body in a large array of bodies).
The frequency vector :math:`\omega \in \mathbb{R}^n` should span the relevant frequencies of the device operation, and also super harmonics to capture nonlinearities.

.. math::
    x_{pos} = [ x_{pos,1}, x_{pos,2}, \dots, x_{pos,m} ]^T
    :label: decision_variable_position

For each mode, :math:`x_{pos}` has :math:`2n + 1` elements (:math:`x_{pos,k} \in \mathbb{R}^{2n + 1}`).
There are :math:`2n` elements because each frequency is described by two components: one each corresponding to the cosine and sine amplitudes.
The additional element in the first position is used to capture a mean ("DC") displacement.
Thus, the portion of :math:`x_{pos}` describing the :math:`k`-th mode would be.

.. math::
    x_{pos,k} =
    [ \bar{x}_{pos,k}, a_{k,1}, b_{k,1}, \dots, a_{k,n}, b_{k,n} ]^T
    :label: decision_variable_position_detail

The external state vector (:math:`x_{ext}`) can have an analogous composition.
The mean components in :math:`x_{ext}` can be used to capture, for example, pretension in a tether.

The corresponding time vector for the chosen frequency array 

.. math::
    f = [ 0, f_1, 2 f_2, \dots, n f_1 ]^T

is given by

.. math::
    t = [ 0, \Delta t, 2 \Delta t, \dots, (2n-1) \Delta t ]^T, 

where :math:`\Delta t = t_f/(2n)` and the repeat period is :math:`t_f=2n\Delta t`.
Since the pseudo-spectral method results in continuous in time solutions, the resulting dynamic time-series can be evaluated with additional substeps in the time vector.

The code deals with the same dynamic system in different representations and there is a need to convert between them.
In general, a state :math:`x` is represented as the mean (DC) component followed by the real and imaginary components of the Fourier coefficients as 

.. math::
    x=[ X_0, \Re(X1), \Im(X1), \dots, \Re(Xn), \Im(Xn) ]^T

In the time-domain, :math:`x(t)`, this is given as :math:`Mx`, where :math:`M` is the time matrix.
Similarly, the state of its derivative is given as :math:`Dx`, where :math:`D` is the derivative matrix.
Alternatively, we sometimes need to represent the state as a complex vector :math:`x=[X_0, X_1, \dots, X_n]^T`.
Finally, sometimes we need to relate the state :math:`x` to another state :math:`y` (e.g. a force component) via an impedance matrix :math:`Z` as :math:`y = Zx`.
The matrices :math:`M`, :math:`D`, and different matrices :math:`Z`, and relevant functions are constructed internally in :py:mod:`wecopttool`.
