[![Test-WecOptTool](https://github.com/SNL-WaterPower/WecOptTool/actions/workflows/python-package.yml/badge.svg)](https://github.com/SNL-WaterPower/WecOptTool/actions/workflows/python-package.yml)

This repository is a development version of WecOptTool written in Python. If you are looking for previous work published using the MATLAB version of WecOptTool please see: [WecOptTool-MATLAB](https://github.com/SNL-WaterPower/WecOptTool-MATLAB)

## Getting started

*this is a temporary approach, will do something different for general users*

1. Install [Anaconda](https://anaconda.org) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
2. Use the [`environment.yml`](environment.yml) file in the root directory of this repository to create a dedicated [conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/environments.html) and install the packages used in this project.

    ```bash
    cd <repo root dir>
    conda env create --file environment.yml
    ```

3. Activate the new environment.

	```bash
	conda activate _wecopttool
	```

4. Install `WecOptTool` in development mode.

	```bash
	pip install -e .
	```

## Updating dependencies

If we add packages to the [`environment.yml`](environment.yml) file, you can update your conda environment with

```bash
conda env update --name _wecopttool --file environment.yml
```

## Running tests

There are a series of unit tests defined in the `test` directory.
These can be run by calling [`pytest`](https://pytest.org) from the root directory of the repository.

```bash
pytest
```

## Software License

Copyright 2020 National Technology & Engineering Solutions of Sandia, 
LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S. 
Government retains certain rights in this software.
 
WecOptTool is free software: you can redistribute it and/or modify it under the 
terms of the GNU General Public License as published by the Free Software 
Foundation, either version 3 of the License, or (at your option) any later 
version.

WecOptTool is distributed in the hope that it will be useful, but WITHOUT ANY 
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A 
PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with 
WecOptTool.  If not, see <https://www.gnu.org/licenses/>.
