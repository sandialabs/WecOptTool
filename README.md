[![Test-WecOptTool](https://github.com/sandialabs/WecOptTool/actions/workflows/push.yml/badge.svg)](https://github.com/sandialabs/WecOptTool/actions/workflows/push.yml)
[![Coverage Status](https://coveralls.io/repos/github/sandialabs/WecOptTool/badge.svg?branch=main)](https://coveralls.io/github/sandialabs/WecOptTool?branch=main)

# WecOptTool
The Wave Energy Converter Design Optimization Toolbox (WecOptTool) allows users to perform wave energy converter (WEC) device design optimization studies with constrained optimal control.

**NOTE:** If you are looking for the WecOptTool code used in previous published work (MATLAB version) please see [WecOptTool-MATLAB](https://github.com/SNL-WaterPower/WecOptTool-MATLAB).

## Project Information
Refer to [WecOptTool documentation](https://sandialabs.github.io/WecOptTool/) for more information, including project overview, tutorials, theory, and API documentation.

## Getting started
WecOptTool requires Python >= 3.8. Python 3.9 & 3.10 are supported.
It is strongly recommended you create a dedicated virtual environment (e.g., using `conda`, `venv`, etc.) before installing `wecopttool`.

**Option 1** - using `Conda`:

```bash
conda install -c conda-forge wecopttool
```

**Option 2** - using `pip` (requires Fortran compilers on your system):

```bash
pip install wecopttool
```

This approach is not recommended for Windows users since compiling `capytaine` on Windows requires [extra steps](https://github.com/capytaine/capytaine/issues/115).

**Geometry module and tutorials**

To use our geometry examples, including for running the tutorials, you will need to install some additional dependencies. 
For the tutorials you will also need to install `jupyter`. 

```bash
pip install wecopttool[geometry] jupyter
```

or on a Mac (`Zsh` shell)

```bash
pip install wecopttool\[geometry] jupyter
```

## Tutorials
The tutorials can be found in the `examples` directory and are written as [Jupyter Notebooks](https://jupyter.org/).
To run the tutorials, first download the notebook files and then, from the directory containing the notebooks, run `jupyter notebook`.
Using `git` to obtain the notebooks this can be done by running

```bash
git clone https://github.com/sandialabs/WecOptTool.git
cd WecOptTool/examples
jupyter notebook
```

## Getting help
To report bugs, use WecOptTool's [issues page](https://github.com/sandialabs/WecOptTool/issues).
For general discussion, use WecOptTool's [discussion page](https://github.com/sandialabs/WecOptTool/discussions)

## Contributing
If you are interested in contributing to WecOptTool, see our [contribution guidelines](https://github.com/sandialabs/WecOptTool/blob/main/.github/CONTRIBUTING.md).
