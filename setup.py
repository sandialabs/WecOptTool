import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="WecOptTool",
    version="0.1.0",
    author="Sandia National Labs",
    author_email="",
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires=[
        'pytest',
        'numpy',
        'scipy',
        'matplotlib',
        'xarray',
        'capytaine',
        'autograd',
        'mhkit',
        'gmsh',
        'pygmsh',
    ],
    dependency_links=[
        'git+https://github.com/LHEEA/meshmagick.git@3.0'
    ],
)
