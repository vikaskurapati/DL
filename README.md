# Implementation of Neural Networks using Python and PyTorch

## 1. Python Setup

For the following description, we assume that you are using Linux or MacOS and that you are familiar with working from a terminal. The exercises are implemented in Python 3.10, so this is what we are going to install here.

If you are using Windows, you will have to google or check out the forums for setup help from other students. There are plenty of instructions for Anaconda for Windows using a graphical user interface though.

To avoid issues with different versions of Python and Python packages we recommend that you always set up a project specific virtual environment. The most common tools for a clean management of Python environments are *pyenv*, *virtualenv* and *Anaconda*. For simplicity, we are going to focus on Anaconda.

### Anaconda setup
Download and install miniconda (minimal setup with less start up libraries) or conda (full install but larger file size) from [here](https://www.anaconda.com/products/distribution#Downloads). Create an environment using the terminal command:

`conda create --name dl python=3.10`

Next activate the environment using the command:

`conda activate dl`

Continue with installation of requirements and starting jupyter notebook using:

`pip install -r requirements.txt` 

`jupyter notebook`

Jupyter notebooks use the python version of the current active environment so make sure to always activate the `i2dl` environment before working on notebooks for this class.

## 2. Dataset Download

Datasets will generally be downloaded automatically by exercise notebooks and stored in a common datasets directory shared among all exercises. A sample directory structure for cifar10 dataset is shown below:-

    dl_exercises
        ├── datasets                   # The datasets required for all exercises will be downloaded here
            ├── cifar10                # Dataset directory
                ├── cifar10.p          # dataset files 
