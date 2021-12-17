Theory
======

.. note::
    This theory section is intended to give a very high-level understanding of key concepts for WecOptTool.
    For a more detailed explanation, please see :cite:`Bacelli2014Optimal,Bacelli2014Numerical,Coe2020Initial`.
    A journal paper will be available soon.


WecOptTool uses a pseudo-spectral method :cite:`Elnagar1995pseudospectral` to perform two tasks synchronously:

	1. Optimize the solution for the desired objective function (e.g. power generation)
	2. Simulate the wave energy converter (WEC) dynamics

This can be written as a traditional optimization problem:

.. math::

   \min\limits_x J(x) \\
   \textrm{s.t.}& \\
   r(x) &= 0 \\
   c_{ineq}(x) &\leq 0 \\
   c_{eq}(x) &= 0

Here, :math:`x` is the state vector which includes both the WEC dynamic state (:math:`x_{w}`) and the user-defined control state to be optimized (:math:`x_{u}`), as :math:`x = [x_{w}, x_{u}]`.
We wish to find optimal values for :math:`x_{u}` to minimize the objective function :math:`J(x)`.
This accomplishes *task 1* from above.
The solution is forced to follow the dynamics of the system via the constraint :math:`r(x) = 0`, where :math:`r(x)` is the WEC dynamics equation in residual form.
This accomplishes *task 2* from above.
The dynamic residual includes the linear hydrodynamic forces from a BEM solution plus any number of arbitrary nonlinear forces such as power take-off (PTO) force, mooring forces, and non-linear hydrodynamic forces.
Additionally, any number of arbitrary nonlinear constraints, such as requiring a tether to remain in tension or limiting the maximum force exerted by the PTO, can be includded in :math:`c_{ineq}(x)` and :math:`c_{eq}(x)`.

Solving this optimization problem, we can find the optimal control state :math:`x_{u}` (e.g. PTO force) that minimizes our objective function (e.g., power generation) for a given WEC design subject to arbitrary constraints and including nonlinear dynamics.
The pseudo-spectral method has a number of key attributes that make it well-suited to this application:

	* **Explicit constraints:** Dynamic and kinematic constraints can be enforced explicitly, negating the need to run unfeasible solutions and thus reducing computational cost.
	* **Efficient simulation of nonlinear dynamics:** Frequency-domain solutions cannot include nonlinear dynamics and time-domain solutions can only do so with significant computational cost; the pseudo-spectral method can efficiently handle nonlinear systems.
	* **Arbitrary or fixed controller structure:** As a starting point, one may consider optimal control without a given structure. Later on, it may be useful to consider, e.g., a proportional feedback or latching controller. This can be accomplished via the pseudo-spectral method by structuring the system dynamics and the control vector (:math:`x_{u}`).

The solution to this optimization problem can be wrapped with an *outer* design optimization problem, in which any optimization algorithm can be applied.
Some examples of problems which can be addressed within this framework include:

   * Optimization of geometry dimensions for a hull
   * Optimization of PTO components (e.g., inertia of a flywheel, a gear ratio)
   * Optimization of ballast versus pre-tension
   * Optimization of the layout for a WEC array
