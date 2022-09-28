API Documentation
=================
The main functionality of *WecOptTool* is implemented in the *core* module.
Functions and classes in the *core* module can be accessed directly from *WecOptTool* as :python:`wecopttool.<function>` instead of :python:`wecopttool.core.<function>`.
The *core* module contains:

* The *WEC* class
* Functions for basic functionality

Other functionalities are implemented in additional modules, and can be accessed as :python:`wecoptool.<module>.<function>`.

.. note:: The :python:`.core` should be ommitted when accessing functions and classes in the *core* submodule.
          E.g., use :python:`wecopttool.WEC` instead of :python:`wecopttool.core.WEC`.
          For all other modules do include the module name, e.g., :python:`wecopttool.waves.regular_wave`.

<<<<<<< HEAD
The main way to interact with *WecOptTool* is through the :python:`WEC` class.
=======
The main way to interact with *WecOptTool* is through the WEC class:

.. autosummary::
   :toctree: api_docs
   :template: class-template-separate-files.rst
   :recursive:
   :nosignatures:

   wecopttool.core.WEC
>>>>>>> tutorials_update

Modules
-------

.. autosummary::
   :toctree: api_docs
   :recursive:
   :nosignatures:
<<<<<<< HEAD
   :template: module-template.rst

   wecopttool.core
=======
   :template: module-template-functions-only.rst

   wecopttool.core

   :template: module-template.rst

>>>>>>> tutorials_update
   wecopttool.pto
   wecopttool.waves
   wecopttool.hydrostatics
   wecopttool.geom

Type Aliases
------------

+-------------------------+----------------------------------------------------------------------------+
| Alias                   | Type                                                                       |
+=========================+============================================================================+
| :python:`StateFunction` | :python:`Callable[[TWEC, np.ndarray, np.ndarray, xr.Dataset], np.ndarray]` |
+-------------------------+----------------------------------------------------------------------------+
