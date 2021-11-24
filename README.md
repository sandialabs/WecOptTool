[![Test-WecOptTool](https://github.com/SNL-WaterPower/WecOptTool/actions/workflows/python-package.yml/badge.svg)](https://github.com/SNL-WaterPower/WecOptTool/actions/workflows/python-package.yml)
[![Coverage Status](https://coveralls.io/repos/github/SNL-WaterPower/WecOptTool/badge.svg?branch=main)](https://coveralls.io/github/SNL-WaterPower/WecOptTool?branch=main)

# WecOptTool
The WEC Design Optimization MATLAB Toolbox (WecOptTool) allows users to perform
wave energy converter (WEC) device design optimization studies with constrained
optimal control.

*NOTE:* If you are looking for the WecOptTool code used in previous published work (MATLAB version) please see [WecOptTool-MATLAB](https://github.com/SNL-WaterPower/WecOptTool-MATLAB).

*NOTE:* This repository is under development. It will be a Python version of WecOptTool.

## Project Information
Refer to [WecOptTool documentation](https://snl-waterpower.github.io/WecOptTool/main/index.html) for more information including project overview, tutorials, theory, and API documentation.

## Citing
Please cite:
> [Main Publication or Code DOI (can make this a badge instead)]

## Getting started
### General users
WecOptTool requires Python 3.9 (waiting on vtk -> 3.10).

*Option 1* - using `pip` for [Capytiane](https://github.com/mancellin/capytaine) (requires Fortran compilers):
    ```bash
    pip install git+https://github.com/LHEEA/meshmagick.git@3.3
    pip install wecoptool
    ```

*Option 2* - using `Conda` for [capytiane](https://github.com/mancellin/capytaine) (requires the [Conda package manager](https://docs.conda.io/en/latest/)):
    ```bash
    pip install git+https://github.com/LHEEA/meshmagick.git@3.3
    conda install -c conda-forge capytaine
    pip install wecoptool
    ```

*Tutorials* - if running tutorials, to also install additional requirements do
    ```bash
    pip install wecopttool.[tutorials]
    ```
    instead.
*Note:* on a ZSH shell (Macs) do `pip install wecopttool.\[tutorials]` instead.

### Developers
Optionally create a conda environment:
    ```bash
    conda create -n wecopttool python=3.9
    conda activate wecopttool
    ```

Install MeshMagick (not packaged in PyPI or Conda)
    ```bash
    pip install git+https://github.com/LHEEA/meshmagick.git@3.3
    ```

If using conda to install Capytaine:
    ```bash
    conda install -c conda-forge capytaine
    ```

Install WecOptTool in editable mode:
    ```bash
    git clone git@github.com:SNL-WaterPower/WecOptTool.git
    cd WecOptTool
    pip install -e .[dev]
    ```
*Note:* on a ZSH shell (Mac) do `pip install -e .\[dev]` instead.

## Getting help
To report bugs use WecOptTool's [issues page](https://github.com/SNL-WaterPower/WecOptTool/issues).
For general discussion use WecOptTool's [discussion page](https://github.com/SNL-WaterPower/WecOptTool/discussions)

## Contributing
Community contributions are welcomed!

### Style guide
* Style guide: [pep8](https://www.python.org/dev/peps/pep-0008/).
* Docstrings: [pep257](https://www.python.org/dev/peps/pep-0257/) & [NumPy style])(https://numpydoc.readthedocs.io/en/latest/format.html).
* Type hints: [module documentation](https://docs.python.org/3/library/typing.html), [cheat sheet](https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html).

### Autograd
This project uses [`autograd`](https://github.com/HIPS/autograd) for automatic differentiation.
Autograd does not support all NumPy and SciPy functionalities, see [autograd documentation](https://github.com/HIPS/autograd/blob/master/docs/tutorial.md#supported-and-unsupported-parts-of-numpyscipy).
*NOTE:* using unsupported functionalities results in the gradient calculation failing silently

### Pull reuquests
  1. Create a fork of WecOptTool
  2. Create a branch for the specific issue
  3. Add desired code modifications. Put a note in the changelog. For enhancements add to documentation. Add or modify a test. Make sure all tests pass and documentation builds. Follow style guide above.
  4. Do a pull request, and give admins edit access. [Link to any open issues](https://docs.github.com/en/issues/tracking-your-work-with-issues/linking-a-pull-request-to-an-issue) and add relevant tags.

### Tests
There are a series of unit tests defined in the `tests` directory.
These can be run by calling [`pytest`](https://pytest.org) from the root directory of the repository.
    ```bash
    pytest
    ```

### Continuous integration (CI)
This project uses [GitHub Actions](https://docs.github.com/en/actions/learn-github-actions) to run tests on pull requests.

### Documentation:
See [Documentation](https://snl-waterpower.github.io/WecOptTool/main/index.html).

#### Editing the documentation
The documentation is built using [Sphinx](https://www.sphinx-doc.org/en/master/), and the [Napoleon extension](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html) for automatic code documentation.
The source code (restructured text) is in `./docs/source` and images are in `./docs/source/_static`.
The homepage source code is in `./docs/source/index.rst`.
To build the documentation
    ```bash
    cd docs
    make
    ```
The built documentation will be in `./docs/_build` and the homepage is `./docs/_build/index.html`.
To delete do `make clean`.

### Issue tracking
To report bugs use WecOptTool's [issues page](https://github.com/SNL-WaterPower/WecOptTool/issues).
For general discussion use WecOptTool's [discussion page](https://github.com/SNL-WaterPower/WecOptTool/discussions)

## Publications
[1] Pub 1
[2] Pub 2