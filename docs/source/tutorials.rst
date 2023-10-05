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

.. _GitHub repository: https://github.com/sandialabs/WecOptTool/tree/main/examples
.. _WaveBot: https://doi.org/10.3390/en10040472
.. _AquaHarmonics: https://aquaharmonics.com/technology/
.. _LUPA: https://pmec-osu.github.io/LUPA/

.. toctree::
    :maxdepth: 3
    :hidden:

    _examples/tutorial_1_WaveBot
    _examples/tutorial_2_AquaHarmonics
    _examples/tutorial_3_LUPA
    _examples/tutorial_4_Pioneer