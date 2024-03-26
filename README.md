
This document serves as a comprehensive course outline, providing an overview of each lecture's main topics, with direct links to specific codes, reading materials, and other resources for in-depth study. 

## Topic 1: Machine Learning Paradigms 
Go through [my blog post](https://medium.com/@sergioli/from-simple-to-complex-a-complete-overview-of-reinforcement-learning-599a8c1ea689) for details
<!-- Does self-supervised learning belong to unsupervised learning? What do you consider "unknown" in the context of unsupervised learning? 
time series forecasting is one kind of SSL, e.g., forcasting the volume of the Transaction  -->


## (Optional) Topic 2: Python Setup
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

### 2.2: Install Python Package for The Environment Using pip
All the required packages are listed in `requirements.txt`.

```python
pip install -r requirements.txt
```

## Topic 3: Use of Python
### What are Python Script, Modules and Packages? How to Use Them?
Go through [my blog post](https://medium.com/@sergioli/python-scripts-modules-andpackages-232d5f749e64)

### Python Data Types, Functions, Classes
Go through the notebook: [1-basic-python.ipynb](1-basic-python.ipynb)

## Topic 4: Basic Mathematical Disciplines
* The mathematical disciplines required for Machine Learning: Basic Linear Algebra introduced in [1-basic-linear-algebra](1-basic-linear-algebra.ipynb); Basic Statistics introduced in [2-stat](2-basic-stat.ipynb); Calculus for optimizing differentiable functions exemplified by [basic-differentiable-func](ml-tuts/basic-differentiable-func.ipynb).
  * Linear algebra provides functions mapping independent variable(s) $x$ -> dependent variable(s) $y$
  * Statistics underlies mulitple purposes of ML. Essentially, statistical estimation is used for defining objectives for learning. (Details are in [the blog post](https://medium.com/@sergioli/statistical-estimation-for-machine-learning-ad1d6135ba62))
  * For some optimization algorithms, e.g., logistic regression and neural networks, Calculus is used to calculate derivatives w.r.t. the parameters of functions. Calculus is barely discussed and required for this course.

## Topic 5: Utilizing Sample Statistics in Data Preprocessing and Wrangling
Before this topic, let's review the question: **What is the use of statistics in ML?**

Consider a dataset intended for a rating system, which spans a range from 1 to 10 but includes an outlier (100) and a missing value (NA): [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, NA]. Addressing the peculiarities of this dataset involves several critical steps:
* Understanding the Dataset Through Sample Statistics (noting the small size of this example compared to typically larger datasets)
* Handling Missing or Incomplete Values
* Identifying and Removing Outliers:
  *  **How is the Interquartile Range useful for outlier detection?** You may want to look at the from-scratch implementation of `calculate_quartiles` in [the python module](my_ml_package/stat.py) for insight.

For preprocessing multivariate data in preparation for some ML algorithms, especially algorithms based on geometric, it's crucial to:
* Normalize the data, utilizing techniques such as Min-Max Scaling: For details, see the last section of [this blog post](https://medium.com/@sergioli/data-wrangling-and-preprocessing-in-python-a-practical-guide-345aa2e55439).
  
Go through the code snippets and details in [this blog post](https://medium.com/@sergioli/data-wrangling-and-preprocessing-in-python-a-practical-guide-345aa2e55439).


## Topic 6: Statistical Concepts for Modeling
**What are other use cases of statistics in ML?** statistical inference plays an essential in machine learning, including statistical estimation of parametric models, machine learning inference/predictions. 
* You probably feel confused of these concepts. Go through the blog post for A BIG PICTURE: [Statistical Inference v.s. Statistical Estimation v.s. Machine Learning Inference](https://medium.com/@sergioli/statistical-inference-v-s-statistical-estimation-v-s-ml-inference-03f79404645a).
* Go through the very basic statistical concepts (sample, probability distribution, PDF, PMF, and so on) in [the notebook: 2-stat](2-basic-stat.ipynb). This is important to understand the questions:
  *  How to infer sampling distributions? From direct estimation of probabilities via averaging to estimation of indirect parameters (e.g., $\lambda$ for Poisson Distribution) to estimation of parameters of conditional probability distribution.
  *  Assessing the properties of data distributions to match suitable ML algorithms
*  **Why is Central Limit Theorem useful for machine learning?**
*  One Widely Used Statistical Estimation Principle: [MLE](https://medium.com/@sergioli/statistical-estimation-for-machine-learning-ad1d6135ba62)
   *  This is related to the topics of supervised learning.
  <!-- > Review it from the whole picture with one important question:  After the objective function is defined to estimate variable $p$, how it relates to update model parameters, e.g., weights of linear regression ?  Take Neural Network as an example. Firstly, we need to understand parameters in neural network are all chained via a stack of functions in each neuron of each layer. For example, if we take one step back, $p$ is commonly the output of logistic function.
    $$p = h_\theta=\frac{1}{1+e^{-\theta^{T} X}} $$
    * Secondly, since they are all chained together, we can update every parameters or, professionally saying, optimize $\theta$ in neural network by utilizing gradient from auto differentiation (This process is also called backward propagation). -->

## Topic 7: Unsupervised Learning for Clustering
* Pattern recognition
* No labels, i.e., no concrete tasks
### Topic 7.1: Clustering
  + [K-means](3-clustering-kmeans.ipynb)
  + [DB-SCAN, Hierarchical Clustering](3-clustering-others.ipynb); 
  + [Evaluation](3-clustering-eval.ipynb)

### Topic 7.2: Dimension Reduction
* When do we want to perform dimensionality reduction?
    <!-- + preserve semantics of high-dimensional data in low-dimensional subspace
    + overcome the curse of dimensionality: "When the dimensionality increases, the volume of the space increases so fast that the available data become sparse" -->
* Idea 1: Focus on directions that maximize the variance of data
<!-- * Idea 2: Removing correlation, i.e., a linear relationship between variables? (Redundant information) -->
  + Go through [the post](https://medium.com/@sergioli/principal-component-analysis-an-intuitive-mathematically-comprehensive-and-step-by-step-coding-e40f8a7f6417)
  + See [the notebook for constructing PCA](4-dim-reduct-pca.ipynb)

===========================

## Below is an unfinished outline. I will expand each of them before each workshop.

===========================
## Topic 6: Supervised Learning
  + $x$ -> Known $y$
  + Regression: See [5 - Regression](5-regression.ipynb)
  + Classification: See [6 - Classification & Regularization](6-classification-regularization.ipynb)
  + Advanced Supervised Learning Algorithms: See [Support Vector Machine](7-svm.ipynb), [Decision Tree](8-dt.ipynb) & [K Nearest Neighbours](8-knn.ipynb). 
  + Advance Supervised Learning Technique: [9 - Bagging and Boosting](9-bagging-boosting.ipynb).
  + Advanced SL concepts: [9 - Bias & Variance](9-bias_var.ipynb) This is VERY IMPORTANT to analyze your ML models!!! You have to revise these concepts Again and Again and Again...... if you do not want to become a "FAKE" ML expert. 

### Topic 6.1: Linear Regression
* From modeling perspective, Linear regression is doing affine transformation. Go through [the notebook](ml-tuts/linear-affine-transformation.ipynb) to explore linear/affine transformation.
* From objective perspective, linear regression is doing Maximum Likelihood: Specifically, if you assume that the errors $\varepsilon$ in the regression equation $y=\mathbf{w}^T \mathbf{x}+b+\varepsilon$ are normally distributed with mean 0 and some variance $\sigma^2$, then the goal of linear regression is to find the parameters ( $\mathbf{w}$ and $b$ ) that maximize the likelihood of observing the given data. This is equivalent to minimizing the sum of squared residuals, which is the ordinary least squares (OLS) objective. The connection between OLS and MLE under the assumption of normally distributed errors is a fundamental concept in statistical learning.
* Additional Reading Materials
  * [Deep Learning Chapter 5.1.4](https://github.com/janishar/mit-deep-learning-book-pdf/tree/master/complete-book-bookmarked-pdf)

## Topic 8: Neural Network
  + Neural Networks, a universal function approximator, are commonly used for supervised learning. 
  + See [10 - Deep Learning](10-deep_learning.ipynb). 
  + Beyond the scope of this couse: it can be applied for Dimension Reduction and Self-supervised Learning. 
  + Examples exist in language intelligence: [Word2vec](https://arxiv.org/abs/1301.3781), Auto-Regressive Language Models (AR-LMs) and Auto-Encoding Language Models (AE-LMs). Although they utilize techniques reminiscent of supervised learning, this classification arises because the supervisory signals are derived internally from the data itself, rather than relying on external labels. 