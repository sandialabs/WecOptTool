#!/bin/bash
unameOut="$(uname -s)"
case "${unameOut}" in
    Linux*)     whl="https://files.pythonhosted.org/packages/97/6f/eef85213faf6565c3101bce5f825a361659ec7efce10724b432e18e300ad/gmsh-4.10.5-py2.py3-none-manylinux1_x86_64.whl";;
    CYGWIN*)    whl="https://files.pythonhosted.org/packages/6a/a5/a10fcb26dcf174f37364fd47253ed4a11794d489376e18f00dac8aaf60c7/gmsh-4.10.5-py2.py3-none-win_amd64.whl";;
    Darwin*)    whl"https://files.pythonhosted.org/packages/9c/b1/33369adbe4ec5b4dc71eaf433ff6ded08143312a6b2a0eb2e0426d6f8ab3/gmsh-4.10.5-py2.py3-none-macosx_10_15_x86_64.whl";;
esac
pip install $whl
python -m pip install .     # Python command to install the script.