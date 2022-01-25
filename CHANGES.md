
# Changelog


## Version x.x.x

**Major Changes**

*

**Bug fixes**

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
