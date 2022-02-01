[![Test-WecOptTool](https://github.com/SNL-WaterPower/WecOptTool/actions/workflows/python-package.yml/badge.svg)](https://github.com/SNL-WaterPower/WecOptTool/actions/workflows/python-package.yml)
[![Coverage Status](https://coveralls.io/repos/github/SNL-WaterPower/WecOptTool/badge.svg?branch=main)](https://coveralls.io/github/SNL-WaterPower/WecOptTool?branch=main)

# WecOptTool
The Wave Energy Converter Design Optimization Toolbox (WecOptTool) allows users to perform wave energy converter (WEC) device design optimization studies with constrained optimal control.

**NOTE:** If you are looking for the WecOptTool code used in previous published work (MATLAB version) please see [WecOptTool-MATLAB](https://github.com/SNL-WaterPower/WecOptTool-MATLAB).

## Project Information
Refer to [WecOptTool documentation](https://snl-waterpower.github.io/WecOptTool/) for more information, including project overview, tutorials, theory, and API documentation.

## Getting started
WecOptTool requires Python 3.9 (waiting on vtk -> 3.10).

**Option 1** - using `pip` for [Capytiane](https://github.com/mancellin/capytaine) (requires Fortran compilers):

```bash
pip install git+https://github.com/LHEEA/meshmagick.git@3.3
pip install wecopttool
```

**Option 2** - using `Conda` for [Capytiane](https://github.com/mancellin/capytaine) (requires the [Conda package manager](https://docs.conda.io/en/latest/)):

```bash
pip install git+https://github.com/LHEEA/meshmagick.git@3.3
conda install -c conda-forge capytaine
pip install wecopttool
```

## Tutorials
The tutorials can be found in the `examples` directory and are written as [Jupyter Notebooks](https://jupyter.org/).
To run the tutorials, first download the notebook files and then, from the directory containing the notebooks, run `jupyter notebook`. 
Using `git` to obtain the notebooks this can be done by running

```bash
git clone https://github.com/SNL-WaterPower/WecOptTool.git
cd WecOptTool/examples
jupyter notebook
```

## Getting help
To report bugs, use WecOptTool's [issues page](https://github.com/SNL-WaterPower/WecOptTool/issues).
For general discussion, use WecOptTool's [discussion page](https://github.com/SNL-WaterPower/WecOptTool/discussions)

## Contributing
If your interersted in contributing to WecOptTool see our [contribution guidelines](https://github.com/SNL-WaterPower/WecOptTool/blob/main/.github/CONTRIBUTING.md).
