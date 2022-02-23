
# Changelog

## Version 1.0.3

* minor updates to README

**Bug fixes**

* geom.WaveBot.plot_cross_section now plots correct location of waterline
* All constraints are now being enforced when multiple constraints are passed


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
* Move jupyter-notebook to base install dependecy

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
