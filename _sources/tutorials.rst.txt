Tutorials
=========
This section presents two tutorials illustrating the application of WecOptTool.
The tutorials are written as Jupyter Notebooks and are available in the `GitHub repository`_.

The first tutorial uses the `WaveBot`_ WEC, which is a single-body WEC developed at Sandia.
The tutorial has three sequential parts of increased complexity.
The first and second parts solve only the inner optimization loop for optimal mechanical and electrical power, respectively.
The third part builds on the second to solve a design optimization problem, using both the inner and outer optimization loops, and is more reflective of the optimization problems WecOptTool is designed to solve.


    - :doc:`_examples/tutorial_1_wavebot`: Three part example of using the inner and outer optimization loops for a simple control co-design study.


The second tutorial uses the `AquaHarmonics`_ WEC, and provides a more robust optimization example illustrating more complex cases of the power take-off, applied forces, and constraints.
The tutorial has two parts, which are similar in structure to the second and third parts of Tutorial 1.


    - :doc:`_examples/tutorial_2_AquaHarmonics`: Two part example with a realistic power take-off system and co-design study.


.. _GitHub repository: https://github.com/SNL-WaterPower/WecOptTool/tree/main/examples
.. _WaveBot: https://doi.org/10.3390/en10040472
.. _AquaHarmonics: https://aquaharmonics.com/technology/

.. toctree::
    :maxdepth: 3
    :hidden:

    _examples/tutorial_1_wavebot
    _examples/tutorial_2_AquaHarmonics
