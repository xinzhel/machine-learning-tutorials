If you use Anaconda, you do not need to do the following things. Common ML and data science packages are all setup.
### 2.1: Set up a Python Virtual Environment Using Conda
* Install conda: Find the bash command for your OS on [the web page](https://docs.anaconda.com/free/miniconda/#quick-command-line-install)
  
* Setup a virtual environment (Ensure that the Python version is large than Python3.3 to avoid the import error)
    ```
    conda create -n learn_ml python=3.7.9
    conda activate learn_ml
    ```
* Remove the environment if you do not use it to save your disk
    ```
    conda remove -n learn_ml --all
    ```

> Alternatively, you can use Pyenv.  But it requires more steps for configuration and managing Python versions, especially on Windows.  See [my blog post](https://gist.github.com/xinzhel/dd586583a0ff1d81b24e56f9680a4eb8) for details.

### 2.2: Install Python Package 
* All the required packages are listed in `requirements.txt`.
* Using the `pip` package for installing other packages
```python
pip install -r requirements.txt
```