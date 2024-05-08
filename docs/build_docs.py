import os
import subprocess
import shutil
import re
import yaml

from sphinx.application import Sphinx

import source.make_theory_animations


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


def build_doc(version, tag, home_branch):
    os.environ['current_version'] = version
    subprocess.run(f'git checkout {tag}', shell=True)
    subprocess.run(
        f"git checkout {home_branch} -- {os.path.join(source_dir, 'conf.py')}", shell=True)
    subprocess.run(
        f"git checkout {home_branch} -- {os.path.join(docs_dir, 'versions.yaml')}", shell=True)
    subprocess.run(
        f"git checkout {home_branch} -- {os.path.join(source_dir, '_templates/versions.html')}", shell=True)
    source.make_theory_animations
    linkcheck()
    html()
    cleanup()
    subprocess.run(
        f"git checkout {home_branch}", shell=True)


if __name__ == '__main__':
    home_name = 'latest'
    with open(os.path.join(docs_dir, 'versions.yaml'), 'r') as v_file:
        versions = yaml.safe_load(v_file)
    home_branch = versions[home_name]
    build_doc(home_name, home_branch, home_branch)
    print(f"Moving HTML pages to {os.path.join(docs_dir, 'pages')}...")
    shutil.copytree(html_dir, os.path.join(docs_dir, 'pages'))
    print('Done.')
    for name, tag in versions.items():
        if name != home_name:
            build_doc(name, tag, home_branch)
            print(f"Moving HTML pages to {os.path.join(docs_dir, 'pages', name)}...")
            shutil.copytree(html_dir, os.path.join(docs_dir, 'pages', name))
            print('Done.')
    shutil.rmtree(html_dir)
