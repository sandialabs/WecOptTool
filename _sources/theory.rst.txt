Theory
======

.. note::
    This theory section is intended to give a very high-level understanding of key concepts for WecOptTool.
    For a more detailed explanation, please see :cite:`Bacelli2014Optimal,Bacelli2014Numerical,Coe2020Initial`.
    A journal paper will be available soon.


WecOptTool uses a pseudo-spectral method :cite:`Elnagar1995pseudospectral` to perform two tasks synchronously:

	1. Optimize the solution for maximum power generation/absorption
	2. Simulate the WEC dynamics

This can be written as a traditional optimization problem:

.. math::

   \min\limits_x p(x) \\
   \textrm{s.t.}& \\
   r(x) &= 0 \\
   c(x) &\leq 0 \\
   c_{eq}(x) &= 0

Here, :math:`x` is the decision vector, and we wish to find optimal values for :math:`x` to minimize power (:math:`p(x)`), where negative power is absorbed by the device.
This accomplishes task 1. from above.
The solution is forced to follow the dynamics of the system via the constraint :math:`r(x) = 0`, where :math:`r(x)` is a residual equation.
The remaining constraints dealing with :math:`c(x)` allow for the user to enforce additional limits, such as requiring a tether to remain in tension or limiting the maximum force exerted by a power take-off (PTO).

Solving this optimization problem, we can find the optimal power for a given WEC design subject to arbitrary constraints and including nonlinear dynamics.
The pseudo-spectral solution method is a number of key attributes that make it well-suited to this application:

	* **Explicit constraints:** Dynamic and kinematic constraints can be enforced explicitly, negating the need to run unfeasible solutions and thus reducing computational cost.
	* **Efficient simulation of nonlinear dynamics:** Frequency-domain solutions cannot include nonlinear dynamics and time-domain solutions can only do so with significant computational cost; the pseudo-spectral method can efficiently handle nonlinear systems, *as long as the motion is periodic.*
	* **Arbitrary or fixed controller structure:** As a starting point, one may desired to consider optimal control without a given structure. Later on, it may be useful to consider, e.g., a proportional feedback or latching controllers. This can be accomplished in via the pseudo-spectral method by structuring the system dynamics and the decision vector (:math:`x`).

The solution to this optimization problem can be wrapped with a *outer* design optimization problem, in which any optimization algorithm can be applied.
Some examples of problems which can be addressed within this framework include:

   * Optimization of geometry dimensions for a hull
   * Optimization of PTO components (e.g., inertia of a flywheel, a gear ratio)
   * Optimization of ballast versus pre-tension
   * Optimization of the layout for a WEC array
