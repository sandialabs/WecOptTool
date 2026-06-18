import os
import sys
import shutil
import importlib
import re
import yaml

from wecopttool import __version__, __version_info__


# -- Path setup --------------------------------------------------------------
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)
source_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, source_root)

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

with open(os.path.join(project_root, 'docs/versions.yaml'), 'r') as v_file:
    versions = yaml.safe_load(v_file)


def _normalize_ref(ref: str | None) -> str | None:
    if ref is None:
        return None

    prefixes = ('refs/heads/', 'refs/tags/', 'origin/', 'heads/', 'tags/')
    for prefix in prefixes:
        if ref.startswith(prefix):
            return ref[len(prefix):]
    return ref


def _resolve_version_context(config=None, current_ref_override=None) -> tuple[str, str, list[list[str]]]:
    version_by_ref = {}
    for name, ref in versions.items():
        version_by_ref[ref] = name
        normalized = _normalize_ref(ref)
        if normalized is not None:
            version_by_ref[normalized] = name

    current_ref = (
        current_ref_override
        or getattr(config, 'smv_current_version', None)
        or os.environ.get('SPHINX_MULTIVERSION_NAME')
        or os.environ.get('current_version')
        or 'latest'
    )
    normalized_current_ref = _normalize_ref(current_ref)
    current_branch = (
        version_by_ref.get(current_ref)
        or version_by_ref.get(normalized_current_ref)
        or normalized_current_ref
        or current_ref
    )

    latest_ref = versions.get('latest', 'latest')
    normalized_latest_ref = _normalize_ref(latest_ref)

    if (
        current_branch == 'latest'
        or current_ref == latest_ref
        or current_ref == normalized_latest_ref
        or normalized_current_ref == latest_ref
        or normalized_current_ref == normalized_latest_ref
    ):
        url_prefix = '.'
    else:
        url_prefix = '..'

    other_versions = []
    for name in versions.keys():
        if name == 'latest':
            other_versions.append([name, os.path.join(url_prefix)])
        else:
            other_versions.append([name, os.path.join(url_prefix, name)])

    return current_branch, url_prefix, other_versions


skip_notebook_execution = os.environ.get('WOT_DOCS_SKIP_NOTEBOOK_EXECUTION') == '1'
skip_theory_animations = os.environ.get('WOT_DOCS_SKIP_THEORY_ANIMATIONS') == '1'

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
exclude_patterns = []

if skip_notebook_execution:
    nbsphinx_execute = 'never'

html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'navigation_depth': 5,
}
html_static_path = ['_static']
current_branch, _url_prefix, other_versions = _resolve_version_context()
html_context = {
    'current_version' : current_branch,
    'other_versions' : other_versions,
}


def _all_but_ipynb(_dir, contents):
    return [entry for entry in contents if not entry.endswith('.ipynb')]


def _all_but_nc(_dir, contents):
    return [entry for entry in contents if not (entry.endswith('.nc') or entry.endswith('.npz'))]


def _copy_examples() -> None:
    print('Copy example notebooks into docs/_examples')
    examples_dst = os.path.join(project_root, 'docs/source/_examples')
    os.makedirs(examples_dst, exist_ok=True)

    shutil.copytree(
        os.path.join(project_root, 'examples'),
        examples_dst,
        ignore=_all_but_ipynb,
        dirs_exist_ok=True,
    )
    shutil.copytree(
        os.path.join(project_root, 'examples/data'),
        os.path.join(examples_dst, 'data'),
        ignore=_all_but_nc,
        dirs_exist_ok=True,
    )


def _generate_theory_animations() -> None:
    global _theory_animations_generated
    if _theory_animations_generated:
        return

    importlib.invalidate_caches()
    module_name = 'make_theory_animations'
    if module_name in sys.modules:
        importlib.reload(sys.modules[module_name])
    else:
        importlib.import_module(module_name)
    _theory_animations_generated = True


def _cleanup_index_html(outdir: str) -> None:
    index_file = os.path.join(outdir, 'index.html')
    if not os.path.exists(index_file):
        return

    with open(index_file, 'r', encoding='utf-8') as file_handle:
        data = file_handle.read()

    updated = re.sub(r'<section id="package">.*?</section>', '', data, flags=re.DOTALL)

    with open(index_file, 'w', encoding='utf-8') as file_handle:
        file_handle.write(updated)


def _on_config_inited(_app, _config):
    outdir_name = os.path.basename(os.path.normpath(getattr(_app, 'outdir', '') or ''))
    current_ref_override = None if outdir_name in {'', 'html'} else outdir_name

    current_branch, _url_prefix, other_versions = _resolve_version_context(
        _config,
        current_ref_override=current_ref_override,
    )
    _config.html_context = dict(_config.html_context or {})
    _config.html_context['current_version'] = current_branch
    _config.html_context['other_versions'] = other_versions

    _copy_examples()

    if skip_notebook_execution:
        print('Skipping notebook execution')
    if skip_theory_animations:
        print('Skipping theory animation generation')
    else:
        _generate_theory_animations()


def _on_build_finished(app, exception):
    if exception is not None:
        return
    _cleanup_index_html(app.outdir)


def setup(app):
    app.add_css_file('css/custom.css')
    app.connect('config-inited', _on_config_inited)
    app.connect('build-finished', _on_build_finished)


_theory_animations_generated = False

suppress_warnings = ['autosectionlabel.*', # nbsphinx and austosectionlabel do not play well together
                     'app.add_node', # using multiple builders in custom Sphinx objects throws a bunch of these
                     'app.add_directive',
                     'app.add_role',
                     'ref.python',  # duplicate cross-reference targets from autosummary (e.g. wecopttool.WEC vs wecopttool.core.WEC)
                     ]

# -- References (BibTex) -----------------------------------------------------
bibtex_bibfiles = ['wecopttool_refs.bib']
bibtex_encoding = 'utf-8-sig'
bibtex_default_style = 'alpha'
bibtex_reference_style = 'label'
bibtex_foot_reference_style = 'foot'

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
