This is a private repository for developing WEC design optimization studies for AquaHarmonics using [WecOptTool](https://github.com/SNL-WaterPower/WecOptTool).

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
