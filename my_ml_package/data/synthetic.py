import numpy as np
import matplotlib.pyplot as plt

def generate_sample(sample_size, pdf_or_pmf=np.random.uniform):
    return [pdf_or_pmf() for _ in range(sample_size)]


def generate_correlated_X(num_features = 5, corr=0.95, sample_size = 1000):
    """ Sets up a dataset X with num_features features. 
    Each feature is normally distributed, but all features have a pairwise correlation.

    Args:
    num_features: int, the number of features
    corr: float, the correlation between the features. It will derive a correlation matrix with this value, e.g. 0.95:
        [[1.   0.95]
         [0.95 1.  ]]
    """
    corr_matrix = np.ones((num_features, num_features)) * corr + np.identity(num_features) * (1 - corr)
    # print(corr_matrix)
    X = np.random.multivariate_normal(mean=np.zeros(num_features), cov=corr_matrix, size=sample_size)
    return X

def variance_sample_for_decision_tree(sample_size = 30, num_features=5, corr=0.95):
    """ Sample easily separable by a decision tree, but with high variance. """

    X = generate_correlated_X(num_features = num_features, corr=corr, sample_size=sample_size)
    

    # The response Y was generated according to Pr(Y = 1|x1 ≤ 0.5) = 0.2, Pr(Y = 1|x1 > 0.5) = 0.8
    # what is the lowest possible error rate?
    y = np.zeros(sample_size)
    for i in range(sample_size):
        if X[i,0] <= 0.5: # {0: 80%, 1: 20%}
            y[i] = np.random.binomial(1, 0.2)
        else:             # {0: 20%, 1: 80%}
            y[i] = np.random.binomial(1, 0.8)
    return X, y.astype(int)
