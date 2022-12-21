# Releasing New WecOptTool Version
This section is for developers.

Before a release make sure to:

* change [version number](https://semver.org/) in `pyproject.toml`

## GitHub
In the GitHub repository, click on *Releases*, click on *Draft new release*.

* Title the release with the [version number](https://semver.org/) preceded by a `v`, e.g., `v1.0.0`. Nothing else should go in the title.
* Tag the release using the same name as the *Title*.
* Click on *Generate release notes*. This adds the PR messages and contributors. Ideally nothing more is needed, but might require minor editing/formatting.
* Select the *Create a discussion* checkmark and mark as *Announcement*.

This will trigger the PyPI, Conda, and GH-Pages build and deploy.

### Pre-releases:
For pre-releases make sure to use correct [semantic versioning](https://semver.org/) for the version number in `pyproject.toml`.
In the Github release, for the *title* and *tag* name append the pre-release version after the version name, e.g., `v1.2.0-alpha` or `v1.2.1-beta.3`, and select the *pre-release* checkmark. Do not select the *Create a discussion* checkmark.

## PyPI package
For details see the [Python packaging user guide](https://packaging.python.org/en/latest/) and in particular the [packaging tutorial](https://packaging.python.org/en/latest/tutorials/packaging-projects/).

The PyPI package is created and uploaded automatically to [TestPyPI](https://test.pypi.org/) and [PyPI](https://pypi.org/) on every GitHub release.
This is done sequentially, so that if *TestPyPi* fails the package is not pushed to `PyPi`.
See the [GitHub workflow](https://github.com/SNL-WaterPower/WecOptTool/blob/main/.github/workflows/publish-to-pypi.yml).

**NOTE:** While GitHub lets you delete a release and then create a new one with the same name, PyPI does not. You can delete releases but you cannot upload a package with the same version as a previous one (even a deleted one).

## Conda package
To do
