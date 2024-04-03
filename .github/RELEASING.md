# Releasing New WecOptTool Version
This section is for developers.

Before a release make sure to:

* change [version number](https://semver.org/) in `pyproject.toml` in the `dev` branch.
* Merge the `dev` branch into the `main` branch. **Note: the `dev` branch should only be merged into `main` when it is ready for a new release.**

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
See the [GitHub release workflow](https://github.com/sandialabs/WecOptTool/blob/main/.github/workflows/release.yml).

**NOTE:** While GitHub lets you delete a release and then create a new one with the same name, PyPI does not. You can delete releases but you cannot upload a package with the same version as a previous one (even a deleted one).

## Conda package
When a new release is available on PyPI, Conda-forge has a [bot](https://github.com/regro/autotick-bot) that will automatically find this and create a pull request in [wecopttool-feedstock](https://github.com/conda-forge/wecopttool-feedstock), the GitHub repository that houses the Conda recipe for WecOptTool. Conda-forge does not currently have full integration with `pyproject.toml` files, so we have to manually update the [`meta.yaml`](https://github.com/conda-forge/wecopttool-feedstock/blob/main/recipe/meta.yaml) file in the WecOptTool Conda recipe with any new or removed dependencies if changes were made in `pyproject.toml`. The version number, SHA256, and build number should be automatically updated by the bot, but these should also be checked just in case.

Follow the instructions on the [Conda-forge maintainer documentation](https://conda-forge.org/docs/maintainer/updating_pkgs.html#pushing-to-regro-cf-autotick-bot-branch) to push any required changes to the bot-generated pull request. Merge and close the pull request once these updates are pushed and the Conda-forge CI passes. The Conda build will now install the new release for users.
