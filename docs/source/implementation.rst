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

For the linear case, which is the default formulation in WecOptTool, the WEC dynamics can be expressed as

.. math::
    (M + m_a) \dot{v} = - k z - B \, v - C \int\limits_{-\infty}^t v  + f_{ext} + f_{exc} .
    :label: wec_dynamics

This expression can be rewritten in residual form as

.. math::
    r = -(M + m_a) \dot{v} - k z - B \, v - C \int\limits_{-\infty}^t v  + f_{ext} + f_{exc} .
    :label: residual_dynamics

The decision variable, :math:`x`, includes states of the body motion as well as other external states, e.g., related to controllers, PTO systems, mooring systems, etc.

.. math::
    x = 
	\begin{bmatrix}
		x_{pos}\\
		x_{ext}
	\end{bmatrix}
    :label: decision_variable_top

Here, :math:`x_{pos} \in \mathbb{R}^{m(2n + 1)}`, where :math:`m` is the number of body modes of motion and :math:`n` is the number of frequency components.
Similarly, :math:`x_{ext} \in \mathbb{R}^{p(2n+1)}`, where :math:`p` is the number of external states.
Body modes could include multiple degrees-of-freedom for a single body (e.g., heave, surge, and pitch) and/or degrees-of-freedom from multiple bodies (e.g., heave modes from each body in a large array of bodies).
The frequency vector :math:`\omega \in \mathbb{R}^n` should span the relevant frequencies of the device operation, and also super harmonics to capture nonlinearities.

.. math::
    x_{pos} = 
	\begin{bmatrix}
		x_{pos,1} \\
		x_{pos,2} \\
		\vdots \\
		x_{pos,m}
	\end{bmatrix}
    :label: decision_variable_position

For each mode, :math:`x_{pos}` has :math:`2n + 1` elements (:math:`x_{pos,k} \in \mathbb{R}^{2n + 1}`).
There are :math:`2n` elements because each frequency is described by two components: one each corresponding to the cosine and sine amplitudes.
The additional element in the first position is used to capture a mean ("DC") displacement.
Thus, the portion of :math:`x_{pos}` describing the :math:`k`-th mode would be.

.. math::
	x_{pos,k} = 
	\begin{bmatrix}
		\bar{x}_{pos,k} \\
		a_{k,1} \\
		b_{k,1} \\
		\vdots \\
		a_{k,n} \\
		b_{k,n}
	\end{bmatrix}
    :label: decision_variable_position_detail

The external state vector (:math:`x_{ext}`) has an analogous composition.
The mean components in :math:`x_{ext}` can be used to capture, for example, pretension in a tether.

.. TODO: MIMO transfer function block matrices
.. TODO: dynamics evaluated in terms of position (not velocity)
.. TODO: time vector

