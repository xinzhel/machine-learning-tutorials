If you use Anaconda, you do not need to do the following things. Common ML and data science packages are all setup.

### (for Windows Users) Install WSL or/and CUDA
* If you use a Windows PC, install WSL
* If you use a Windows PC with a Nvidia GPU, install WSL and CUDA

For details, you can refer to [my early Windows Setting Steps in 2021](https://medium.com/gitconnected/build-the-environment-for-deep-learning-in-windows-11-subsystem-of-linux-wsl-f26ffc4548b2)

> If you are Windows users, it is likely that many issues happen during installation. Unless you have a very powerful GPU, better directly use online resources, e.g., Google Colab or Kaggle.

### 2.1: Set up a Python Virtual Environment Using Conda
* Install conda: Find the bash command for your OS on [the web page](https://docs.anaconda.com/free/miniconda/#quick-command-line-install)
  
* Setup a virtual environment (Ensure that the Python version is large than Python3.3 to avoid the import error)
    ```
    > conda create -n learn_ml python=3.9
    > conda activate learn_ml
    ```
> Remove the environment if you do not use it to save your disk
    ```
    > conda remove -n learn_ml --all
    ```
<!-- Alternatively, you can use Pyenv.  But it requires more steps for configuration and managing Python versions, especially on Windows.   See [my blog post](https://gist.github.com/xinzhel/dd586583a0ff1d81b24e56f9680a4eb8) for details. -->

### 2.2: Install Python Package 
* The required packages are listed in `requirements.txt`.
* Using the `pip` package for installing other packages
```
> pip install -r requirements.txt
```

### 2.3: Install Tensorflow
For Mac with Apple Metal chips
```
> pip install tensorflow==2.16.2
> pip install tensorflow-metal==1.1.0
```

