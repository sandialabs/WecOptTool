import os
from shutil import rmtree

docs_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.join(docs_dir, '_build')
source_dir = os.path.join(docs_dir, 'source')
example_dir = os.path.join(source_dir, '_examples')
api_dir = os.path.join(source_dir, 'api_docs')
pages_dir = os.path.join(docs_dir, 'pages')

def clean():
    rmtree(example_dir, ignore_errors=True)
    rmtree(api_dir, ignore_errors=True)
    rmtree(build_dir, ignore_errors=True)
    rmtree(pages_dir, ignore_errors=True)

if __name__ == '__main__':
    clean()