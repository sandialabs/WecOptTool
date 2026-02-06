Tutorials
=========
This section presents tutorials illustrating the application of WecOptTool.
The tutorials are written as Jupyter Notebooks and are available in the `GitHub repository`_.

The first tutorial uses the `WaveBot`_ WEC, which is a single-body WEC developed at Sandia.
The tutorial has three sequential parts of increased complexity.
The first and second parts solve only the inner optimization loop for optimal mechanical and electrical power, respectively.
The third part builds on the second to solve a design optimization problem, using both the inner and outer optimization loops, and is more reflective of the optimization problems WecOptTool is designed to solve.

    - :doc:`_examples/tutorial_1_WaveBot`: Three part example of using the inner and outer optimization loops for a simple control co-design study.

The second and third tutorials use the `AquaHarmonics`_ and `LUPA`_ WECs, respectively, and provide more robust optimization examples illustrating more complex cases of the WEC body, power take-off, applied forces, and constraints.
These tutorials each have two parts, which are similar in structure to the second and third parts of Tutorial 1.

    - :doc:`_examples/tutorial_2_AquaHarmonics`: Two part example with a realistic power take-off system and co-design study. You can find an extended version 
    - :doc:`_examples/tutorial_3_LUPA`: Two part example with multiple bodies, multiple degrees of freedom, irregular waves, a mooring system, and realistic constraints.

The fourth tutorial uses the `Pioneer WEC` model, which includes a unique pitch resonator PTO system. This tutorial illustrates how to use WecOptTool to implement and optimize control strategies for less common PTO archetypes.

    - :doc:`_examples/tutorial_4_Pioneer`: Example with custom PTO physics and modeling both hydrodynamic and non-hydrodynamic degrees of freedom.

.. toctree::
    :maxdepth: 3
    :hidden:

    _examples/tutorial_1_WaveBot
    _examples/tutorial_2_AquaHarmonics
    _examples/tutorial_3_LUPA
    _examples/tutorial_4_Pioneer


Simulating WEC Dynamics without optimization
--------------------------------------------

There may be situations where it is useful to see the dynamics of the WEC subject to certain constraints without performing optimization on the device.
This can be done by setting ``nstate_opt=0`` and using an objective function that does not depend on ``x_wec``.

The easiest way to do this is to set an objective function that always equals zero. 
Constraints and additional forces can also be added, but they must be independent of ``x_opt`` since there are no constrol states.
The additional forces should also be defined at all nonzero states (i.e. returns length ``nfreq * 2``).

Example:

.. code-block:: python
    
   # define additional force
   # (must be independent of x_opt and of length nfreq * 2)
   def forcing_func(wec, x_wec, x_opt, waves):
       frc = 50
       return np.ones((nfreq*2, 1)) * frc
    f_add = {'Additional force': forcing_func}

   # define constraint
   # (must be independent of x_opt)
   f_max = 750.
   def force_constraint(wec, x_wec, x_opt, waves):
       return f_max
   ineq_cons = {'type': 'ineq',
                'fun': force_constraint,
                }
   constraints = [ineq_cons]

   # define WEC object
   wec = wot.WEC.from_bem(
        bem_data, # define beforehand as you normally would
        constraints=constraints,
        friction=None,
        f_add=f_add)

   # create dummy objective function
   obj_fun = lambda wec, x_wec, x_opt, waves : 0
   nstate_opt = 0

   # solve problem (should solve on first iteration)
   results = wec.solve(
       waves=waves, # define beforehand as you normally would
       obj_fun=obj_fun, 
       nstate_opt=nstate_opt
       )
       
   # post process
   nsubsteps = 5
   wec_fdom, wec_tdom = wec.post_process(wec, results, waves, nsubsteps=nsubsteps)
   wec_tdom[0]['pos'].plot()

.. _GitHub repository: https://github.com/sandialabs/WecOptTool/tree/main/examples
.. _WaveBot: https://doi.org/10.3390/en10040472
.. _AquaHarmonics: https://aquaharmonics.com/technology/
.. _LUPA: https://pmec-osu.github.io/LUPA/
