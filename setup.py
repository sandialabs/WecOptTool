import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="WecOptTool",
    version="0.0.1",
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
    python_requires='>=3.6',
    entry_points={
        'console_scripts': ['example_WaveBot_oneDof=WecOptTool.examples.example_WaveBot_oneDof:test_example']
    },
)
