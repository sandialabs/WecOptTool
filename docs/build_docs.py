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


def linkcheck():
    app = Sphinx(source_dir,
                    conf_dir,
                    build_dir,
                    doctree_dir,
                    'linkcheck',
                    warningiserror=True)
    app.build()
    app.disconnect()


def html():
    app = Sphinx(source_dir,
                conf_dir,
                build_dir,
                doctree_dir,
                'html',
                warningiserror=True)
    app.build()
    app.disconnect()


if __name__ == '__main__':
    source.make_theory_animations
    linkcheck()
    html()
