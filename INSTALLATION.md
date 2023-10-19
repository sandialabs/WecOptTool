# Installation for users

## Software requirements
WecOptTool is supported on Windows, MacOS, and Linux. It requires Python 3.8 or higher. [Xcode](https://developer.apple.com/xcode/) may also be required on Mac.


## Creating a virtual environment
WecOptTool depends on many other Python packages, which can be organized into a *virtual environment*. Setting up a dedicated virtual environment allows for easier and more organized management of Python packages for your projects. The instructions below will walk you through creating a dedicated virtual environment and installing WecOptTool.

Several tools exist that can both manage virtual environment and install Python pacakges. We provide instructions for two such tools:

* If you are brand new to Python, or currently use Conda and want to try a much faster alternative, [click here](#installing-using-mamba) for installation instructions using **Mamba**.
* If you already have Anaconda/Miniconda installed on your computer, [click here](#installing-using-conda) for instructions using **Conda**.

### Installing using Mamba
1. Download and install Miniforge3 (which contains Mamba) for your operating system:
    
    * **Windows**: Download Miniforge3 from this [link](https://github.com/conda-forge/miniforge#download). Double-click on the file once it is downloaded, and follow the prompts on the new window to install. When the "Advanced Installation Options" prompt comes up, check the box next to "Add Miniforge3 to my PATH environment variable". All the other default selections should work.
    * **MacOC or Linux**: Open your terminal and run the following commands:
        ```bash
        curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
        bash Miniforge3-$(uname)-$(uname -m).sh
        ```
        Follow the prompts that appear in the terminal to install Miniforge3. When the `Do you wish the installer to initialize Miniforge3 by running conda init [yes|no]` prompt appears, select "yes".

2. After installation completes, open a command prompt or terminal window and copy/paste the following code to confirm Mamba installed correctly. If installed correctly, the terminal should print both a Mamba and Conda version number (since Conda is used for some Mamba functions):
    ```bash
    mamba --version
    ```
3. Copy/paste the following code to create a new virtual environment named `wot`, activate the environment, and install WecOptTool and its dependencies in the environment. Feel free to replace `wot` in the first two lines with a different environment name if you would like:
    ```bash
    mamba create -n wot
    mamba activate wot
    mamba install wecopttool jupyter
    pip install gmsh pygmsh meshio
    ```

### Installing using Conda
1. Download Miniconda3 (which contains Conda) for your operating system [here](https://docs.conda.io/projects/miniconda/en/latest/index.html). For MacOS, the `bash` option installs via the terminal and the `pkg` option installs via an interactive window, choose whichever you prefer.
2. Install Miniconda3 for your operating system:

    * **Windows or Mac `pkg`**: Double-click on the file you just downloaded and follow the prompts on the new window. When the "Advanced Installation Options" prompt comes up, check the box next to "Add Miniforge3 to my PATH environment variable". All the other default selections should work.
    * **Linux or Mac `bash`**: Open your terminal and run the following command:
    ```bash
    bash Miniconda3-latest-$(uname)-$(uname -m).sh
    ```
3. After installation completes, open a command prompt or terminal window and copy/paste the following code to confirm Conda installed correctly. If installed correctly, the terminal should print a Conda version number:
    ```bash
    conda --version
    ```
4. Copy/paste the following code to create a new virtual environment named `wot`, activate the environment, and install WecOptTool and its dependencies in the environment. Feel free to replace `wot` in the third and fourth lines with a different environment name if you would like:
    ```bash
    conda config --add channels conda-forge
    conda config --set channel_priority strict
    conda create -n wot
    conda activate wot
    conda install wecopttool jupyter
    pip install gmsh pygmsh meshio
    ```