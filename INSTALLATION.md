# Installation for users

## Software requirements
WecOptTool is support on Windows, MacOS, and Linux. It requires Python 3.8 or higher. [Xcode](https://developer.apple.com/xcode/) may also be required on Mac.


## Creating a virtual environment
WecOptTool depends on many other Python packages, which can be organized into a *virtual environment*. Setting up a dedicated virtual environment allows for easier and more organized management of Python packages for your projects. The instructions below will walk you through creating a dedicated virtual environment and installing WecOptTool.

Several tools exist that can both manage virtual environment and install Python pacakges. We provide instructions for two such tools:

* If you are brand new to Python, or currently use Conda and want to try a much faster alternative, [click here](#installing-using-mamba) for installation instructions using **Mamba**
* If you already have Anaconda/Miniconda installed on your computer, [click here](#installing-using-conda) for instructions using **Conda**.

### Installing using Mamba
1. Download Miniforge3 (which contains Mamba) for your operating system [here](https://github.com/conda-forge/miniforge#download)
2. Install Miniforge3 for your operating system:

   * **Windows**: Double-click on the EXE file you just downloaded and follow the prompts on the new window. At the options selection screen, check the box next to "Add Miniforge3 to my PATH environment variable". All the other default selections should work.
   * **MacOS or Linux**: Open your terminal and run the following commands to install using curl:
   ```bash
   curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
   bash Miniforge3-$(uname)-$(uname -m).sh
   ```
3. After installation completes, open a command prompt or terminal window and copy/paste the following code to confirm Mamba installed correctly and no errors are given. If installed correctly, the terminal should print both a Mamba and Conda version number (since Conda is used for some Mamba functions):
    ```bash
    mamba --version
    ```
4. Copy/paste the following code to create a new virtual environment named `wot`, activate the environment, and install WecOptTool and its dependencies in the environment:
    ```bash
    mamba create -n wot
    mamba activate wot
    mamba install wecopttool jupyter
    pip install gmsh pygmsh meshio
    ```

### Installing using Conda
1. Download Miniconda for your operating system [here](https://docs.conda.io/projects/miniconda/en/latest/index.html)
2. Install Miniconda for your operating system:

   * **Windows or Mac**: Double-click on the file you just downloaded and follow the prompts on the new window. At the options selection screen, check the box next to "Add Miniconda3 to my PATH environment variable". All the other default selections should work.
   * **Linux**: Open your terminal and run the following command:
   ```bash
   bash Miniconda3-latest-Linux-x86_64.sh
   ```
   If you downloaded the installer for another Linux distribution, replace the `.sh` filename with the name of the file you just downloaded.
3. After installation completes, open a command prompt or terminal window and copy/paste the following code to confirm Conda installed correctly and no errors are given. If installed correctly, the terminal should print a Conda version number:
    ```bash
    conda --version
    ```
4. Copy/paste the following code to create a new virtual environment named `wot`, activate the environment, and install WecOptTool and its dependencies in the environment:
    ```bash
    conda config --add channels conda-forge
    conda config --set channel_priority strict
    conda create -n wot
    conda activate wot
    conda install wecopttool jupyter
    pip install gmsh pygmsh meshio
    ```