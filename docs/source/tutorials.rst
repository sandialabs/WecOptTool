Tutorials
=========
This section presents two tutorials illustrating the application of WecOptTool.
The tutorials are written as Jupyter Notebooks which are available in the `GitHub repository`_.
The two tutorials use the `WaveBot`_ WEC, which is a single-body WEC developed at Sandia.
The tutorials are meant to be sequential.
The first tutorial solves only the inner optimization loop, and serves as an introduction to WecOptTool.
The second tutorial builds on the first to solve a design opotimization problem, using both the inner and outer optimization loops, and is more reflective of the optimization problems WecOptTool is designed to solve.

- **Tutorial 1** - Simple example of the *inner loop* using a single-body WEC, moving in one degree of freedom, in regular waves. The example finds the optimal control strategy for a fixed WEC design.
- **Tutorial 2** - Simple example of a design optimization problem (*outer* and *inner* optimization loops). The example optimizes the WEC geometry (outer loop) while finding the optimal control strategy for each design considered (inner loop).


.. _GitHub repository: https://github.com/SNL-WaterPower/WecOptTool/tree/main/examples
.. _WaveBot: https://doi.org/10.3390/en10040472
.. _Reference Model 3: https://energy.sandia.gov/programs/renewable-energy/water-power/projects/reference-model-project-rmp/

.. toctree::
    :maxdepth: 3
    :hidden:

    _examples/tutorial_1_wavebot
    _examples/tutorial_2_wavebot_optimization