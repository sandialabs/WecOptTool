import os
from shutil import rmtree

docs_dir = os.path.dirname(os.path.abspath(__file__))
source_dir = os.path.join(docs_dir, 'source')
build_dir = os.path.join(docs_dir, '_build')
example_dir = os.path.join(source_dir, '_examples')
api_dir = os.path.join(source_dir, 'api_docs')
pages_dir = os.path.join(docs_dir, 'pages')


def _assert_within_docs(path: str) -> None:
    docs_root = os.path.realpath(docs_dir)
    candidate = os.path.realpath(path)
    if os.path.commonpath([docs_root, candidate]) != docs_root:
        raise RuntimeError(f"Refusing to delete path outside docs directory: {candidate}")


def clean() -> None:
    targets = [build_dir, example_dir, api_dir, pages_dir]
    for target in targets:
        _assert_within_docs(target)
        if os.path.exists(target):
            print(f"Removing {target}")
            rmtree(target, ignore_errors=False)
        else:
            print(f"Skipping {target} (not found)")

if __name__ == '__main__':
    clean()