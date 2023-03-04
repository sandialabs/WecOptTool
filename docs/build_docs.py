import os
from shutil import rmtree
import source.make_theory_animations
from sphinx.application import Sphinx

docs_dir = os.path.dirname(os.path.abspath(__file__))
source_dir = os.path.join(docs_dir, 'source')
conf_dir = source_dir
build_dir = os.path.join(docs_dir, '_build')
doctree_dir = os.path.join(build_dir, '.doctrees')
example_dir = os.path.join(source_dir, '_examples')
api_dir = os.path.join(source_dir, 'api_docs')


linkcheck = Sphinx(source_dir,
                   conf_dir,
                   build_dir,
                   doctree_dir,
                   'linkcheck',
                   warningiserror=True,
                   keep_going=True)
html = Sphinx(source_dir,
              conf_dir,
              build_dir,
              doctree_dir,
              'html',
              warningiserror=True,
              keep_going=True)

def build():
    source.make_theory_animations
    linkcheck.build()
    html.build()

if __name__ == '__main__':
    build()
