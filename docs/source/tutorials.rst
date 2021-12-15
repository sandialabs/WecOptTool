Tutorials
=========
This section presents turorials illustrating the application of WecOptTool.
The source for these tutorials is also included within the WecOptTool installation.

- **Tutorial 1** - Simple example of the *inner loop* using a one body WEC (`WaveBot`_), moving in one degree of freedom, in regular waves.
- **Tutorial 2** - Complex example of the *inner loop* using a two body WEC (`Refrence Model 3`_), with 7 degrees of freedom, in multidirectional irregular waves.
- **Tutorial 3** - Simple example of an optimization problem (*outer* and *inner* optimization loops).
                   The example optimizes the bottom radius of the `WaveBot`_ geometry for average power production.


.. _WaveBot: https://doi.org/10.3390/en10040472
.. _Reference Model 3: https://energy.sandia.gov/programs/renewable-energy/water-power/projects/reference-model-project-rmp/

.. toctree::
    :maxdepth: 1
    :hidden:

    _examples/tutorial_1_wavebot
    _examples/tutorial_2_rm3
    _examples/tutorial_3_optimization