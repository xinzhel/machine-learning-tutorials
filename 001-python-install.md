
Is your laptop equiped with NVIDIA GPUs or Apple Metal chips?
* If yes, follow all the steps 1, 2, 3, 4
* If no, use Google Colab, and only follow Step 3
## 1. OS-Specific Steps

* For Mac with Apple Metal chips, do nothing
* For Windows users and old Mac with Nvidia GPUs, install WSL and CUDA. For details, you can refer to [my early Windows Setting Steps in 2021](https://medium.com/gitconnected/build-the-environment-for-deep-learning-in-windows-11-subsystem-of-linux-wsl-f26ffc4548b2).
    > For Windows users, it is likely that many issues happen during installation. Unless you have a very powerful GPU, better directly use Google Colab
* For old Mac with Nvidia GPUs, just install CUDA

## 2. Set up a Python Virtual Environment Using Conda
* Install conda: Find the bash command for your OS on [the web page](https://docs.anaconda.com/free/miniconda/#quick-command-line-install)
  
* Setup a virtual environment (Ensure that the Python version is large than Python3.3 to avoid the import error)
    ```
    conda create -n learn_ml python=3.9
    conda activate learn_ml
    ```
> Remove the environment if you do not use it to save your disk
    ```
    conda remove -n learn_ml --all
    ```
<!-- Alternatively, you can use Pyenv.  But it requires more steps for configuration and managing Python versions, especially on Windows.   See [my blog post](https://gist.github.com/xinzhel/dd586583a0ff1d81b24e56f9680a4eb8) for details. -->


## 3. Install Some Common Python Package for Data Analysis and Machine Learning
* Run the commands below in your terminal
    ```
    pip install numpy
    pip install scikit-learn
    pip install scipy
    pip install pandas
    pip install matplotlib
    ```
> if you want to run the commands above in your notebook, add the prefix "!" 
    ```
    !pip install numpy
    !pip install scikit-learn
    !pip install scipy
    !pip install pandas
    !pip install matplotlib
    ```
* or run the command below if the working directory contains the file `requirements.txt` with the required packages
    ```
    pip install -r requirements.txt
    ```
<!-- If you use Anaconda, you may not need to do the thing 2.1, 2.2 above. Common ML and data science packages are all setup. -->

## 4. Install Python Package for Deep Learning 
##Tensorflow##
* For Mac with Apple Metal chips
    ```
    pip install tensorflow==2.16.2
    pip install tensorflow-metal==1.1.0
    ```
* For Windows users and old Mac with Nvidia GPUs, follow [the official steps](https://www.tensorflow.org/install/pip#windows-wsl2)
  
##PyTorch##
* For Mac with Apple Metal chips
    ```
    pip install pytorch==2.3.0
    ```
