# Releasing New WecOptTool Version
This section is for developers.

Before a release make sure to:

* update the CHANGES.md (add new version number, release date and list of changes)
* change version number in `setup.cfg`

## PyPI package
For details see the [Python packaging user guide](https://packaging.python.org/en/latest/) and in particular the [packaging tutorial](https://packaging.python.org/en/latest/tutorials/packaging-projects/).

### Initial Setup:

You will need to create a [PyPI](https://pypi.org/) (and optionally a [TestPyPI](https://test.pypi.org/)) account and be added as *owner* of the `wecopttool` project by one of the current owners.
Optionally create [API tokens](https://pypi.org/help/#apitoken) for these accounts for easier authentication.

**Package:**
Run

```bash
pip install --upgrade build
python -m build
```

This will create a `dist` directory with a `.whl` and a `.tar.gz` files.

### Upload to TestPyPI and Check:
To upload to TestPyPI run

```bash
pip install --upgrade twine
twine upload --repository testpypi dist/*
```

To check that it installs (do it in a new, clean, environment) run

```bash
pip install --index-url https://test.pypi.org/simple/ --no-deps wecopttool
```

**Note**: You will only be able to check that it installs.
You will not be able to use (import or test) this installation since dependencies are not installed.
Dependencies are not installed because they are not in the TestPyPI index.

### Upload to PyPI:
To upload to PyPI run

```bash
pip install --upgrade twine
twine upload dist/*
```

## GitHub
In the GitHub repository, under *Releases*, click on *Create new release*.

* Title the release with the [version number](https://semver.org/) preceded by a `v`, e.g., `v1.0.0`. Nothing else should go in the title.
* Tag the release using the same name as the *Title*.
* In the description, copy the corresponding section of the [changes file](https://github.com/SNL-WaterPower/WecOptTool/blob/main/CHANGES.md).
* Select the *Create a discussion* checkmark.


## Pre-releases:
We will likely not use pre-releases but if we do, make sure to use correct [semantic versioning](https://semver.org/) for the version number in `setup.cfg`.
In the Github release, for the *tag* name append the pre-release version after the version name, e.g., `v1.2.0-alpha` or `v1.2.1-beta.3`, and select the *pre-release* checkmark.