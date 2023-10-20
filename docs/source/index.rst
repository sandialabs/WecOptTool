##########
WecOptTool
##########
.. toctree::
    :maxdepth: 2
    :hidden:

    theory.rst
    implementation.rst
    tutorials.rst
    references.rst

The Wave Energy Converter Design Optimization Toolbox (WecOptTool) is an open-source software for conducting optimization studies of wave energy converters (WEC) and their control strategies.
The software uses a co-design (WEC & controls) approach where for each WEC design considered in the optimization, the optimal control strategy for that design is found.
Practically, this is implemented as two nested optimization loops.
One key feature is the use of a pseudo-spectral solution method capable of efficiently dealing with arbitrary nonlinear constraints, nonlinear dynamics, and both structured and unstructured controllers.
This allows for the optimization study (e.g., to find the WEC geometry that results in the largest power capture performance) within the WEC's constraints (e.g., maximum power take-off force, maximum PTO extension, etc.).
The code is written to support arbitrary optimization studies, control strategies, and constraints.
The code is written as a Python package and the source code can be found in the `GitHub repository`_.

Getting Started
===============
See installation instructions in the `GitHub repository`_.
The `GitHub repository`_ also has instructions for raising issues, asking questions, and contributing.
You can work through examples provided on the :ref:`tutorials` page.

Developers
==========
WecOptTool is developed by `Sandia National Laboratories`_.
The developers would like to acknowledge funding support from the US Department of Energy's Water Power Technologies Office.
The developers would also like to acknowledge benefit from past collaborations with `Data Only Greater`_ and the `Oregon State University Design Engineering Lab`_.

.. note::
    A MATLAB version of WecOptTool was previously released and, while no longer being developed, is still available on GitHub: `WecOptTool-MATLAB`_.

Sandia National Laboratories is a multi-mission laboratory managed and operated by National Technology and Engineering Solutions of Sandia, LLC., a wholly owned subsidiary of Honeywell International, Inc., for the U.S. Department of Energy's National Nuclear Security Administration under contract DE-NA0003525.

.. _GitHub repository: https://github.com/sandialabs/WecOptTool
.. _Data Only Greater: https://www.dataonlygreater.com
.. _Oregon State University Design Engineering Lab: https://design.engr.oregonstate.edu
.. _Sandia National Laboratories: https://www.sandia.gov
.. _WecOptTool-MATLAB: https://github.com/SNL-WaterPower/WecOptTool-MATLAB


Package
=======
.. autosummary::
    :toctree: api_docs
    :recursive:
    :nosignatures:
    :template: package.rst

    wecopttool