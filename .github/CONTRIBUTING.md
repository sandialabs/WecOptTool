# Contributing
Community contributions are welcomed! 🎊

## Installation for developers

### Optional steps
Optionally create a conda environment:

```bash
conda create -n wecopttool python=3.9
conda activate wecopttool
```

If using conda to install Capytaine:

```bash
conda install -c conda-forge capytaine
```

If you want to build the documentation locally you will also need to install [pandoc](https://pandoc.org/installing.html).
Using conda this can be done as:

```bash
conda install -c conda-forge pandoc
```

### Install
Install MeshMagick (not packaged in PyPI or Conda)

```bash
pip install git+https://github.com/LHEEA/meshmagick.git@3.3
```

Fork WecOptTool, then install WecOptTool in editable mode:

```bash
git clone git@github.com:<YOUR_USER_NAME>/WecOptTool.git
cd WecOptTool
pip install -e .[dev]
```

**Note:** on a ZSH shell (Mac) do `pip install -e .\[dev]` instead.


## Style guide
* Style guide: [pep8](https://www.python.org/dev/peps/pep-0008/).
* Docstrings: [pep257](https://www.python.org/dev/peps/pep-0257/) & [NumPy style](https://numpydoc.readthedocs.io/en/latest/format.html).
* Type hints: [module documentation](https://docs.python.org/3/library/typing.html), [cheat sheet](https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html).

## Autograd
This project uses [`autograd`](https://github.com/HIPS/autograd) for automatic differentiation.
Autograd does not support all NumPy and SciPy functionalities, see [autograd documentation](https://github.com/HIPS/autograd/blob/master/docs/tutorial.md#supported-and-unsupported-parts-of-numpyscipy).
*NOTE:* using unsupported functionalities results in the gradient calculation failing silently

## Pull Requests
  1. Create a fork of WecOptTool
  2. Create a branch for the specific issue
  3. Add desired code modifications. Put a note in the [Changelog](https://github.com/SNL-WaterPower/WecOptTool/blob/main/CHANGES.md). For enhancements add to documentation. Add or modify a test. Make sure all tests pass and documentation builds. Follow style guide above.
  4. Do a pull request, and give admins edit access. [Link to any open issues](https://docs.github.com/en/issues/tracking-your-work-with-issues/linking-a-pull-request-to-an-issue) and add relevant tags.

## Tests
There are a series of unit tests defined in the `tests` directory.
These can be run by calling [`pytest`](https://pytest.org) from the root directory of the repository.

```bash
pytest
```

## Continuous integration (CI)
This project uses [GitHub Actions](https://docs.github.com/en/actions/learn-github-actions) to run tests on pull requests.

## Documentation:
See [Documentation](https://snl-waterpower.github.io/WecOptTool/).

### Editing the documentation
The documentation is built using [Sphinx](https://www.sphinx-doc.org/en/master/), and the [Napoleon extension](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html) for automatic code documentation.
The source code (restructured text) is in `./docs/source` and images are in `./docs/source/_static`.
The homepage source code is in `./docs/source/index.rst`.

To build the documentation locally (not required but good check)

```bash
cd docs
make html
```

The built documentation will be in `./docs/_build` and the homepage is `./docs/_build/index.html`.
To delete do `make clean`.

The documentation uses the Jupyter notebook tutorials in the `examples` directory.
When building the documentation locally you will need have installed [pandoc](https://pandoc.org/installing.html).

### Editing the tutorials
The tutorials are used as part of the Documentation.
Before pushing any changes make sure that the saved version of the notebooks have the results in them, and the cells were run in order starting with `1`.
To achieve this click `clear outputs`, `restart`, `run all`, and `save` in that order.
For tutorials that have the option to read saved BEM results make sure that the saved version ran the BEM (didn't read them) by deleting the results folder before running.

## Issue tracking
To report bugs use WecOptTool's [issues page](https://github.com/SNL-WaterPower/WecOptTool/issues).
For general discussion use WecOptTool's [discussion page](https://github.com/SNL-WaterPower/WecOptTool/discussions)


## Releasing (developers only)
See [releasing instructions](https://github.com/SNL-WaterPower/WecOptTool/blob/main/.github/RELEASING.md).