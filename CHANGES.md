
# Changelog

## Version 1.1.0

* minor updates to README
* logging of decision vector and objective function
* `f_add` should be passed as a `dict`, e.g., `{'my_name': my_func}`
  * optionally treat buoyancy/gravity explicitly via user-defined functions passed to `f_add`
  * time and freq domain results calculated for entries of `f_add` after `solve` completes
* logging of decision vector and objective function
  * controlled entirely via logging package config
  * move to `info` logging level (`debug` gives too much from other packages, e.g., matplotlib)
* added tests for multiple WEC/PTO degrees of freedom.
* allow user to pass `**kwargs` via `solve` to `scipy.optimize.minimize`

**Bug fixes**

* geom.WaveBot.plot_cross_section now plots correct location of waterline
* All constraints are now being enforced when multiple constraints are passed
* Fix shape/linear algebra bugs in fixed structure PTOs with multiple PTO DOFs


## Version 1.0.2

* update Tutorial 2
    * new r2 vector
    * improve scaling
    * update discussion of results
* logging of decision vector and objective function
    * now uses absolute value of mean
    * move to `debug` logging level

**Bug fixes**

* Correct dependency name for jupyter-notebook
* Move jupyter-notebook to base install dependency

**New features**

* Create continuous time functions from the WEC dynamics and PTO pseudo-spectral results.


## Version 1.0.1

**Bug fixes**

* Fixed bug with modifying mass and other properties of the WEC object. Renamed `mass_matrix` to `mass`.
* Fix broken link in RELEASING

**New features**

* Expand Theory page in documentation to include
    * Animations and comparison between pseudo-spectral and time-stepping solutions
    * Discussion of scaling
* Add JONSWAP spectrum
* Optional logging of x and obj_fun(x) at each iteration
* Add example PI controller to PTO module
* Add core methods for
    * Theoretical power limit
    * Natural frequency
* Add tests for
    * Proportional damping (P) controller gives theoretical power limit at natural frequency
    * Proportional integral (PI) controller gives theoretical power limit for regular wave
* Check test results against previous solution where possible

## Version 1.0.0 (12/16/2021)
Initial release.
