Tutorials
=========
This section presents a tutorial illustrating the application of WecOptTool.
The tutorial is written as Jupyter Notebooks and is available in the `GitHub repository`_.
The tutorial uses the `WaveBot`_ WEC, which is a single-body WEC developed at Sandia.
The tutorial has three sequential parts of increased complexity.
The first and second parts solve only the inner optimization loop for optimal mechanical and electricla power, respectively.
The third part builds on the second to solve a design optimization problem, using both the inner and outer optimization loops, and is more reflective of the optimization problems WecOptTool is designed to solve.


    - :doc:`_examples/tutorial_1_wavebot`: Three part example of using the inner and outer optimization loops for a simple control co-design study.


.. _GitHub repository: https://github.com/SNL-WaterPower/WecOptTool/tree/main/examples
.. _WaveBot: https://doi.org/10.3390/en10040472


    .. toctree::
        :maxdepth: 3
        :hidden:

        _examples/tutorial_1_wavebot
