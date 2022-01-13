# Configuration file for the Sphinx documentation builder.
#
# For a full list of options see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------
import os
import sys
import shutil

from wecopttool import __version__, __version_info__


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)


# -- Project information -----------------------------------------------------
project = 'WecOptTool'
copyright = 'Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC(NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.'
author = 'Sandia National Laboratories'
version = '.'.join(__version_info__[:2])
release = __version__

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'sphinxcontrib.bibtex',
    'sphinx.ext.autosectionlabel',
    'nbsphinx',
]


# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'navigation_depth': 4,
}
html_static_path = ['_static']


# nbsphinx and austosectionlabel do not play well together
suppress_warnings = ["autosectionlabel.*"]


# -- Extension configuration -------------------------------------------------
# Napoleon settings (autodoc)
# See: https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# BibTeX settings
# See: https://sphinxcontrib-bibtex.readthedocs.io/en/latest/usage.html#configuration
bibtex_bibfiles = ['wecopttool_refs.bib']
bibtex_encoding = 'utf-8-sig'
bibtex_default_style = 'alpha'
bibtex_reference_style = 'label'
bibtex_foot_reference_style = 'foot'

# Jupyter Notebooks
print("Copy example notebooks into docs/_examples")

def all_but_ipynb(dir, contents):
    result = []
    for c in contents:
        # if os.path.isfile(os.path.join(dir, c)) and (not c.endswith(".ipynb")):
        if not c.endswith(".ipynb"):
            result += [c]
    return result

shutil.rmtree(os.path.join(
    project_root,  "docs/source/_examples"), ignore_errors=True)
shutil.copytree(os.path.join(project_root,  "examples"),
                os.path.join(project_root,  "docs/source/_examples"),
                ignore=all_but_ipynb)
