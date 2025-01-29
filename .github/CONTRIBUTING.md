# Contributing
Community contributions are welcomed! 🎊

## Installation for developers

* It is recommended that you create a *virtual environment*, e.g. using `conda`, `venv`, or similar.
* If you want to build the documentation locally you will also need to install [pandoc](https://pandoc.org/installing.html) and [gifsicle](https://github.com/kohler/gifsicle). On *Windows*, we recommend installing pandoc using `conda` (i.e. `conda install -c conda-forge pandoc`)
* Building using `pip` on *MacOS* requires the manual installation of Fortran compilers, see discussion [here](https://github.com/sandialabs/WecOptTool/discussions/111). For ARM-based Macs, see [issue #324](https://github.com/sandialabs/WecOptTool/issues/324)
* On a ZSH shell (*MacOS*) do `pip install -e .\[dev]` instead of `pip install -e .[dev]` in the instructions below (i.e., escape the opening square bracket).

Using `conda` this looks like:
```bash
conda create -n wecopttool
conda activate wecopttool
conda install -c conda-forge python=3.12 capytaine wavespectra
git clone git@github.com:<YOUR_USER_NAME>/WecOptTool.git
cd WecOptTool
pip install -e .[dev]
```

And using `pip`:
```bash
git clone git@github.com:<YOUR_USER_NAME>/WecOptTool.git
cd WecOptTool
python3.12 -m venv .venv
. .venv/bin/activate
pip install -e .[dev]
```


## Style guide
* Style guide: [pep8](https://www.python.org/dev/peps/pep-0008/).
* Docstrings: [pep257](https://www.python.org/dev/peps/pep-0257/) & [NumPy style](https://numpydoc.readthedocs.io/en/latest/format.html).
* Type hints: [module documentation](https://docs.python.org/3/library/typing.html), [cheat sheet](https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html).

## Autograd
This project uses [`autograd`](https://github.com/HIPS/autograd) for automatic differentiation.
Autograd does not support all NumPy and SciPy functionalities, see [autograd documentation](https://github.com/HIPS/autograd/blob/master/docs/tutorial.md#supported-and-unsupported-parts-of-numpyscipy).
*NOTE:* using unsupported functionalities results in the gradient calculation failing silently.

## Pull Requests
  1. Create a fork of WecOptTool
  2. Create a branch for the specific issue
  3. Add desired code modifications. For enhancements add to documentation. Add or modify a test. Make sure all tests pass and documentation builds. Follow style guide above.
  4. Do a pull request to the `dev` branch, and give admins edit access. [Link to any open issues](https://docs.github.com/en/issues/tracking-your-work-with-issues/linking-a-pull-request-to-an-issue) and add relevant tags. Use a concise but descriptive PR title, as this will be part of the [release notes](https://github.com/sandialabs/WecOptTool/releases) for the next version. Start the PR title with an all caps label followed by a colon, e.g., "BUG FIX: ...", "NEW FEATURE: ...", "DOCUMENTATION: ...", etc. **Note: Pull requests should be made to the `dev` branch, not the `main` branch**. 

## Tests
There are a series of unit and integration tests defined in the `tests` directory.
These can be run by calling [`pytest`](https://pytest.org) from the root directory of the repository.

```bash
pytest
```

## Continuous integration (CI)
This project uses [GitHub Actions](https://docs.github.com/en/actions/learn-github-actions) to run tests on pull requests.

## Documentation:
See [Documentation](https://sandialabs.github.io/WecOptTool/).

### Editing the documentation
The documentation is built using [Sphinx](https://www.sphinx-doc.org/en/master/), and the [Napoleon extension](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html) for automatic code documentation.
The source code (restructured text) is in `./docs/source` and images are in `./docs/source/_static`.
The homepage source code is in `./docs/source/index.rst`.

To build the documentation locally (not required but good check), go to `./docs/versions.yaml` and change the value of `latest` to be your local branch. Then run:

```bash
python3 docs/build_docs.py
```

The built documentation will be in `./docs/pages` and the homepage is `./docs/pages/index.html`.
To delete, do `python3 docs/clean_docs.py`.

The documentation uses the Jupyter notebook tutorials in the `examples` directory.
When building the documentation locally you will need to have installed [pandoc](https://pandoc.org/installing.html) and [gifsicle](https://github.com/kohler/gifsicle).
We recommend installing pandoc using its Anaconda distribution: `conda install -c conda-forge pandoc`.

**NOTE:** it may be expedient at times to avoid running the tutorial notebooks. To do so, add [`nbsphinx_execute = 'never'`](https://nbsphinx.readthedocs.io/en/0.9.3/configuration.html#nbsphinx_execute) to `docs/source/conf.py`. Make sure not to commit these changes!

If you add or change any hyperlinks in the documentation, we recommend checking the "Build documentation" warnings in the GitHub Actions CI workflow to make sure the links will not cause an issue. The CI will not fail due to broken links, only issue a warning (see [issue #286](https://github.com/sandialabs/WecOptTool/issues/286)).

### Editing the tutorials
The tutorials are used as part of the Documentation.
Before pushing any changes make sure that the saved version of the notebooks are clear (no cells run and no results).
To achieve this click `clear outputs` and `save` in that order.
Alternatively create a pre-commit hook that strips the results.

## Issue tracking
To report bugs use WecOptTool's [issues page](https://github.com/sandialabs/WecOptTool/issues).
For general discussion use WecOptTool's [discussion page](https://github.com/sandialabs/WecOptTool/discussions)


## Releasing (developers only)
See [releasing instructions](https://github.com/sandialabs/WecOptTool/blob/main/.github/RELEASING.md).
