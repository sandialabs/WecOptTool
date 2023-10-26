import os
import re
import source.make_theory_animations
from sphinx.application import Sphinx

docs_dir = os.path.dirname(os.path.abspath(__file__))
source_dir = os.path.join(docs_dir, 'source')
conf_dir = source_dir
build_dir = os.path.join(docs_dir, '_build')
linkcheck_dir = os.path.join(build_dir, 'linkcheck')
html_dir = os.path.join(build_dir, 'html')
doctree_dir = os.path.join(build_dir, 'doctrees')


def linkcheck():
    app = Sphinx(source_dir,
                 conf_dir,
                 linkcheck_dir,
                 doctree_dir,
                 'linkcheck',
                 warningiserror=False)
    app.build()


def html():
    app = Sphinx(source_dir,
                 conf_dir,
                 html_dir,
                 doctree_dir,
                 'html',
                 warningiserror=True)
    
    app.build()

def cleanup():
    index_file = os.path.join(html_dir, 'index.html')
    with open(index_file, 'r', encoding='utf-8') as f:
        data = f.read()

    with open(index_file, 'w', encoding='utf-8') as f:
        data2 = re.sub(
            '\<section id="package"\>.*?\</section\>',
            '', data, flags=re.DOTALL)
        f.write(data2)

if __name__ == '__main__':
    source.make_theory_animations
    linkcheck()
    html()
    cleanup()
