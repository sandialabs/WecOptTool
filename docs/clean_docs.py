import os
from shutil import rmtree

docs_dir = os.path.dirname(os.path.abspath(__file__))
source_dir = os.path.join(docs_dir, 'source')
example_dir = os.path.join(source_dir, '_examples')
api_dir = os.path.join(source_dir, 'api_docs')

def clean():
    rmtree(example_dir)
    rmtree(api_dir)

if __name__ == '__main__':
    clean()