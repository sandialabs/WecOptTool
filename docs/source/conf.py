import os
import sys
import shutil
import yaml

from wecopttool import __version__, __version_info__


# -- Path setup --------------------------------------------------------------
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

code_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../wecopttool"))
sys.path.insert(0, code_root)

# -- Project information -----------------------------------------------------
project = 'WecOptTool'
copyright = (
    '2020 National Technology & Engineering Solutions of Sandia, ' +
    'LLC (NTESS). ' +
    'Under the terms of Contract DE-NA0003525 with NTESS, the U.S. ' +
    'Government retains certain rights in this software'
)
author = 'Sandia National Laboratories'
version = '.'.join(__version_info__[:2])
release = __version__

current_branch = os.environ.get('current_version')
if current_branch == 'latest':
    url_prefix = '.'
else:
    url_prefix = '..'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'sphinxcontrib.bibtex',
    'sphinx.ext.autosectionlabel',
    'nbsphinx',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
]

templates_path = ['_templates']

html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'navigation_depth': 5,
}
html_static_path = ['_static']
html_context = {
  'current_version' : current_branch,
  'other_versions' : [],
}

with open(os.path.join(project_root, 'docs/versions.yaml'), 'r') as v_file:
    versions = yaml.safe_load(v_file)
for name in versions.keys():
    if name == 'latest':
        html_context['other_versions'].append([name, os.path.join(url_prefix)])
    else:
        html_context['other_versions'].append([name, os.path.join(url_prefix, name)])

def setup(app):
    app.add_css_file('css/custom.css')

suppress_warnings = ['autosectionlabel.*', # nbsphinx and austosectionlabel do not play well together
                     'app.add_node', # using multiple builders in custom Sphinx objects throws a bunch of these
                     'app.add_directive',
                     'app.add_role',]

# -- References (BibTex) -----------------------------------------------------
bibtex_bibfiles = ['wecopttool_refs.bib']
bibtex_encoding = 'utf-8-sig'
bibtex_default_style = 'alpha'
bibtex_reference_style = 'label'
bibtex_foot_reference_style = 'foot'

# -- Tutorials (Jupyter) -----------------------------------------------------
print("Copy example notebooks into docs/_examples")

def all_but_ipynb(dir, contents):
    result = []
    for c in contents:
        if not c.endswith(".ipynb"):
            result += [c]
    return result

def all_but_nc(dir, contents):
    result = []
    for c in contents:
        if not (c.endswith(".nc") or c.endswith(".npz")):
            result += [c]
    return result

shutil.rmtree(os.path.join(
    project_root,  "docs/source/_examples"), ignore_errors=True)
shutil.copytree(os.path.join(project_root,  "examples"),
                os.path.join(project_root,  "docs/source/_examples"),
                ignore=all_but_ipynb)
shutil.copytree(os.path.join(project_root, "examples/data"),
                os.path.join(project_root, "docs/source/_examples/data"),
                ignore=all_but_nc)

# -- API documentation -------------------------------------------------------
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
add_module_names = False
html_show_sourcelink = False
autodoc_typehints = "description"
autodoc_type_aliases = {
    'ArrayLike': 'ArrayLike',
    'FloatOrArray': 'FloatOrArray',
    'TStateFunction': 'StateFunction',
    'TWEC': 'WEC',
    'TPTO': 'PTO',
    'TEFF': 'Callable[[ArrayLike, ArrayLike], ArrayLike]',
    'TForceDict': 'dict[str, StateFunction]',
    'TIForceDict': 'Mapping[str, StateFunction]',
    'DataArray': 'DataArray',
    'Dataset': 'Dataset',
    'Figure': 'Figure',
    'Axes': 'Axes',
    }
autodoc_class_signature = "separated"
highlight_language = 'python3'
rst_prolog = """
.. role:: python(code)
   :language: python
"""
autodoc_default_options = {
    'exclude-members': '__new__'
}
autosummary_ignore_module_all = False
autosummary_imported_members = True

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'pandas': ('https://pandas.pydata.org/docs', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference', None),
    'matplotlib': ('https://matplotlib.org/stable', None),
    'xarray': ('https://docs.xarray.dev/en/stable', None),
    'capytaine': ('https://capytaine.github.io/stable/', None),
    'wavespectra': ('https://wavespectra.readthedocs.io/en/latest', None),
}
