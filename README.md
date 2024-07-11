
This document serves as a comprehensive course outline, providing an overview of each lecture's main topics, with direct links to specific codes, reading materials, and other resources for in-depth study. 

> Disclaimer: The notebooks within the `dl_in_tensorflow_tutorials` directory were not created by me. Their legal usage is maintained by the Deakin University School of Information Technology. These materials are provided here for the convenience of students enrolled in SIT319 - Deep Learning.

## Topic 1: Machine Learning Paradigms ([Video](https://www.youtube.com/watch?v=acqzti1U3bo&list=PLJNMCL_eahmQ70zZECr2cTDLwrXJ-RpgW&index=1))
* Supervised Learning, Unsupervised Learning, and Reinforcement Learning ([Section 1 of The Article](https://medium.com/@sergioli/from-simple-to-complex-a-complete-overview-of-reinforcement-learning-599a8c1ea689))
<!-- Does self-supervised learning belong to unsupervised learning? What do you consider "unknown" in the context of unsupervised learning? 
time series forecasting is one kind of SSL, e.g., forcasting the volume of the Transaction  -->
* How Does Deep Learning Differ from Above Paradigms?

## Topic 2: Python 
* [Python Installations](001-python-install.md) 
* Features as Programming Language ; Python Data Types, Functions, Classes [001-basic-python.ipynb](001-basic-python.ipynb)
* Python Script, Modules and Packages ([Video](https://www.youtube.com/watch?v=woXs5xBjF5M&list=PLJNMCL_eahmQ70zZECr2cTDLwrXJ-RpgW&index=2) or [Article](https://medium.com/@sergioli/python-scripts-modules-andpackages-232d5f749e64))

## Topic 3: Mathematical Disciplines
Three mathematical disciplines required for Machine Learning and Deep Learning:  Linear Algebra, Statistics and Calculas

* Linear Algebra
  * provides functions mapping independent variable(s) $x$ -> dependent variable(s) $y$
  * Python Packages: [NumPy](001-linear-algebra.ipynb)

* Calculas
  * [Auto Differentiation](012-auto-diff.ipynb)
  * Optimization: calculate derivatives w.r.t. the parameters of functions
  
* Statistics
  * [Variables, Probability and Distribution](002-basic-stat.ipynb)
  * Use Case in Deep Learning: Statistical Estimation for defining objectives for learning. (Details are in [the blog post](https://medium.com/@sergioli/statistical-estimation-for-machine-learning-ad1d6135ba62))
  * Use Case in Data Preprocessing and Wrangling: Sample Statistics ([Article/Post](https://medium.com/@sergioli/data-wrangling-and-preprocessing-in-python-a-practical-guide-345aa2e55439))
    * Go through the code snippets and details in [Section 2 of the notebook](002-basic-stat.ipynb) or [this blog post](https://medium.com/@sergioli/data-wrangling-and-preprocessing-in-python-a-practical-guide-345aa2e55439).

## Topic 6: Statistical Concepts for Modeling
**What are other use cases of statistics in ML?** statistical inference plays an essential in machine learning, including statistical estimation of parametric models, machine learning inference/predictions. 
* You probably feel confused of these concepts. Go through the blog post for A BIG PICTURE: [Statistical Inference v.s. Statistical Estimation v.s. Machine Learning Inference](https://medium.com/@sergioli/statistical-inference-v-s-statistical-estimation-v-s-ml-inference-03f79404645a).
* Go through the very basic statistical concepts (sample, probability distribution, PDF, PMF, and so on) in [the notebook: 2-stat](002-basic-stat.ipynb). This is important to understand the questions:
  *  How to infer sampling distributions? From direct estimation of probabilities via averaging to estimation of indirect parameters (e.g., $\lambda$ for Poisson Distribution) to estimation of parameters of conditional probability distribution.
  *  Assessing the properties of data distributions to match suitable ML algorithms
*  **Why is Central Limit Theorem useful for machine learning?**
*  One Widely Used Statistical Estimation Principle: [MLE](https://medium.com/@sergioli/statistical-estimation-for-machine-learning-ad1d6135ba62)
   *  This is related to the topics of supervised learning.
  <!-- > Review it from the whole picture with one important question:  After the objective function is defined to estimate variable $p$, how it relates to update model parameters, e.g., weights of linear regression ?  Take Neural Network as an example. Firstly, we need to understand parameters in neural network are all chained via a stack of functions in each neuron of each layer. For example, if we take one step back, $p$ is commonly the output of logistic function.
    $$p = h_\theta=\frac{1}{1+e^{-\theta^{T} X}} $$
    * Secondly, since they are all chained together, we can update every parameters or, professionally saying, optimize $\theta$ in neural network by utilizing gradient from auto differentiation (This process is also called backward propagation). -->

## Topic 7: Unsupervised Learning
* Pattern recognition
* No labels, i.e., no concrete tasks
### Topic 7.1: Clustering 
  + [K-means](003-clustering-kmeans.ipynb) ([Video](https://www.youtube.com/watch?v=Sz6rscxUIzU&list=PLJNMCL_eahmQ70zZECr2cTDLwrXJ-RpgW))
  + [DB-SCAN, Hierarchical Clustering](003-clustering-others.ipynb); 
  + [Evaluation](003-clustering-eval.ipynb)

### Topic 7.2: Dimensionality Reduction
* When do we want to perform dimensionality reduction?
  + [Video for General Discussion](https://www.youtube.com/watch?v=Q5Fvelbnl8Q)
    <!-- + preserve semantics of high-dimensional data in low-dimensional subspace
    + overcome the curse of dimensionality: "When the dimensionality increases, the volume of the space increases so fast that the available data become sparse" -->
* Idea 1: Focus on directions that maximize the variance of data -> PCA ([Notebook](004-dim-reduct-pca.ipynb); [Video](https://www.youtube.com/watch?v=q8bCGXFqqcc))
<!-- * Idea 2: Removing correlation, i.e., a linear relationship between variables? (Redundant information) -->

## Topic 8: Supervised Learning
* Supervised Learning Models: Linear Regression, Logistic Regression, Support Vector Machine ([Notebook](007-svm.ipynb)), Decision Tree ([Notebook](008-dt.ipynb)) & K Nearest Neighbours ([Notebook](008-knn.ipynb)), Neural Network.
* Optimization Techniques
  * Closed-formed Solutions
  * Gradient Decsent
* Advance Supervised Learning Technique: 9 - Bagging and Boosting ([Notebook](009-regularization-decision-tree.ipynb)).
* Advanced learning concepts: Bias & Variance ([Notebook](009-bias-var.ipynb)) This is VERY IMPORTANT to analyze your ML models!!! You have to revise these concepts Again and Again and Again...... if you do not want to become a "FAKE" ML expert. 

* Data Splitting ([Notebook](005-dataset-spliting.ipynb)), Cross Validation 
  * Understand the underlying statistical assumption: I.I.D 

### Topic 8.1: Maximum Likelihood Estimation ([Recoding](https://www.youtube.com/watch?v=z8JgJWvYa6Y); [Post/Article](https://medium.com/@sergioli/statistical-estimation-for-machine-learning-ad1d6135ba62))
* Statistical Estimation: Formulate the objective probabilistically for "judging" which sets of parameters are good, i.e., 
  $$\max_{\theta} P(\theta \mid x, y)$$
* What is MLE?
  * Traslating the above formula as below according to Bayes Therem
  $$P(\theta \mid x, y)=\frac{P(y, x \mid \theta) \cdot P(\theta)}{P(x, y)}$$
  * Simplifying the formula without considering $P(\theta)$, i.e., maximizing the likelihood $P(y, x \mid \theta)$
  $$\max_{\theta} P(y, x \mid \theta)$$
* Go through [the blog post](https://medium.com/@sergioli/statistical-estimation-for-machine-learning-ad1d6135ba62) for detail

### Topic 8.2: Linear Regression ([Recoding](https://www.youtube.com/watch?v=6o3TGLABTHc); [Post/Article](https://medium.com/@sergioli/statistical-estimation-for-machine-learning-ad1d6135ba62))
Assuming a linear relationship between x and y

$$y_i = \beta_0 + \beta_1 x_i + \epsilon_i$$

* A linear regression model is doing linear (technically affine) transformation on the features of each example
$$\hat{y_i} = \hat{\beta_0} + \hat{\beta_1} x_i$$
  *  Go through [the notebook](ml-tuts/linear-affine-transformation.ipynb) to explore linear/affine transformation.
  
* MLE for Linear Regression 
    <!-- maximize Pr(parameters|data) $\propto$ Pr(data|parameters) * Pr(parameters) -->
  * Problem: Given labeled examples, how to calculate $\hat{\beta_0}$ and $\hat{\beta_1}$?
  * MLE Solution: Specifically, if you assume that the errors $\varepsilon$ in the regression equation $y=\hat{\beta_1}^T \mathbf{x}+\hat{\beta_0}+\varepsilon$ are normally distributed with mean 0 and some variance $\sigma^2$, then the goal of linear regression is to find the parameters ( $\hat{\beta_0}$ and $\hat{\beta_1}$ ) that maximize the likelihood of observing the given data. 
  * OLS: This is equivalent to minimizing the sum of squared residuals, which is the ordinary least squares (OLS) objective. The connection between OLS and MLE under the assumption of normally distributed errors is a fundamental concept in statistical learning.
    $$\sum_{i=1}^n\left(y_i- (\hat{\beta_0} + \hat{\beta_1} x_i)\right)^2$$
  * How to find the optimal $\hat{\beta_1}$ and $\hat{\beta_1}$?
    * Hint: Both the function w.r.t. $\hat{\beta_1}$ and the function w.r.t. $\hat{\beta_0}$ are continuous, differentiable, convex 
  <!-- If a convex function has a minimum, that minimum occurs where its derivative is zero. -->
    * Go through [recoding](https://www.youtube.com/watch?v=6o3TGLABTHc)
* Reference
    * [Deep Learning Chapter 5.1.4](https://github.com/janishar/mit-deep-learning-book-pdf/tree/master/complete-book-bookmarked-pdf)
    * [Derivative Rules](https://www.mathsisfun.com/calculus/derivatives-rules.html)
    * [Variance Derivation](https://www.kellogg.northwestern.edu/faculty/weber/decs-433/Notes_4_Random_variability.pdf)

### Topic 8.3: Logistic Regression for Classification ([Recoding](https://www.youtube.com/watch?v=trFqQP9hyJU); [Post/Article](https://medium.com/@sergioli/from-theory-to-code-maximum-likelihood-estimation-for-classification-tasks-6ecd8d075eed))
* Implementation of Logistic Regression: See [the Python module](my_ml_package/classification.py)
* Testing Logistic Regression ([Notebook](006-classification-logistic-reg.ipynb))
* Evaluation Metrics ([Notebook](006-classification-eval.ipynb))

### Topic 8.4: Support Vector Machine (Recoding will be released soon.)

### Topic 8.5: Decision Tree (Recoding will be released soon.)

## Topic 9: Bias and Variance (Recoding will be released soon.)
* Bias and Variance are the underlying causes of Underfitting and Overfitting ([Notebook](009-bias-var.ipynb))
* Ensemble Models Reduce Variance ([Notebook](009-regularization-decision-tree.ipynb))

## Topic 9: Neural Network ([Notebook](010-neural-network.ipynb))
* Math of Neural Network ([Video](https://www.youtube.com/watch?v=-2sV2GWr1fk))
* Implementing Neural Network from Scratch in minimal Python Code (NO THIRD-PARTY LIBRARY) ([Video](https://www.youtube.com/watch?v=TZfBukQO-A0&list=PLJNMCL_eahmQ70zZECr2cTDLwrXJ-RpgW&index=9))
* A simple Neural Network is just to transform data with $N$ linear functions nested with non-linear functions (e.g., sigmoid), do such transformation **again** with a different $N$, do such transformation **again** with a different $N$, ... 
  <!-- * $N$ is called the number of neurons.
  * The number of "agains" is the number of layers. -->
  <!-- + Beyond the scope of this couse: it can be applied for Dimension Reduction and Self-supervised Learning.  -->
  <!-- + Examples exist in language intelligence: [Word2vec](https://arxiv.org/abs/1301.3781), Auto-Regressive Language Models (AR-LMs) and Auto-Encoding Language Models (AE-LMs). Although they utilize techniques reminiscent of supervised learning, this classification arises because the supervisory signals are derived internally from the data itself, rather than relying on external labels.  -->
