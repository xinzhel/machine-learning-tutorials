from matplotlib import pyplot as plt
from scipy import stats
from sklearn import metrics
import seaborn as sns
import numpy as np

def plot_cm(y, y_pred, labels, figsize=(10,10)):
    # cm = metrics.cluster.contingency_matrix(y, y_pred)
    cm = metrics.confusion_matrix(y, y_pred)
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
    ax.set_title('Confusion Matrix'); 
    ax.xaxis.set_ticklabels(labels); ax.yaxis.set_ticklabels(labels)
    plt.show()

def plot_digit(digits, nrows=10, ncols=10):
    '''
    Args:
    digits: numpy array of shape (n, 64)
    '''
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6, 6))
    for idx, ax in enumerate(axs.ravel()):
        ax.imshow(digits[idx, :][:64].reshape((8, 8)), cmap=plt.cm.binary)
        ax.axis("off")
    _ = fig.suptitle("A selection from the 64-dimensional digits dataset", fontsize=16)
    plt.show()

def plot_histogram(data, bins=10, title='', xlabel='', ylabel=''):
    plt.hist(data, bins=bins, alpha=0.5)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def plot_pdf(values, probs):
    fig, axes = plt.subplots(1, 1)
    axes.bar(values, probs)
    axes.set_xlabel('X')
    axes.set_ylabel('P(X)')
    plt.show()

def plot_vectors(vectors, colors, operation_name):
    plt.figure()
    plt.axvline(x=0, color='grey', lw=1)
    plt.axhline(y=0, color='grey', lw=1)
    for i in range(len(vectors)):
        plt.quiver(*vectors[i][0], *vectors[i][1], angles='xy', scale_units='xy', scale=1, color=colors[i])
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.grid()
    plt.title(operation_name)
    plt.show()

def plot_cdf(values, probs):
    cumulated_probs = stats.rv_discrete(values=(values, probs)).cdf(values) # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_discrete.html

    fig, ax = plt.subplots(1, 1)
    ax.plot(values, cumulated_probs, marker='o', markerfacecolor='r', linestyle='None', markersize=10, markeredgecolor='r') # fnid more parameters here: https://matplotlib.org/stable/api/_as_gen/matplotlib.lines.Line2D.html#matplotlib.lines.Line2D.set
    ax.vlines(values, 0, cumulated_probs, colors='r', linewidth=4)

def pdf_to_cdf(values, probs):
    cumulated_probs = []
    for i in range(len(values)):
        cumulated_probs.append(sum(probs[:i+1]))

    return values, cumulated_probs

def plot_regression_line(slope, intercept, x, y):
    # Add the regression line to the plot
    plt.scatter(x, y)
    plt.plot(x, slope * x + intercept, color='red')

    # Set the title and labels for the plot
    plt.title('Regression Line Example')
    plt.xlabel('X Values')
    plt.ylabel('Y Values')

    # Show the plot
    plt.show()



def plot_data_points(X, labels):
    """
    Plot data points from X on a 2-D graph.
    
    Parameters:
    - X: numpy array of shape (n_samples, 2), data points to plot.
    
    Returns:
    - fig: matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(X[:, 0], X[:, 1], color='blue', s=100)  # s is the marker size
    
    # Annotating the points with labels
    for i, txt in enumerate(labels):
        ax.annotate(txt, (X[i, 0], X[i, 1]), textcoords="offset points", xytext=(0,10), ha='center')
    
    ax.set_title('2-D Graph of Data Points')
    ax.set_xlabel('Document 1 Frequency')
    ax.set_ylabel('Document 2 Frequency')
    ax.grid(True)
    
    return fig, ax

def plot_line(ax, X, index1, index2):
    """
    Draw a line between two data points identified by index1 and index2 on the provided figure.
    
    Parameters:
    - fig: matplotlib figure object, the figure on which to draw.
    - ax: matplotlib axes object, the axes on which to draw.
    - X: numpy array of shape (n_samples, 2), data points.
    - index1: int, index of the first data point.
    - index2: int, index of the second data point.
    """
    point1 = X[index1]
    point2 = X[index2]
    ax.plot([point1[0], point2[0]], [point1[1], point2[1]], 'r--', lw=2)

def plot_cosine_distance(ax, X, index1, index2):
    """
    Plot the cosine distance between two data points (vectors) in a 2-D space.
    """
    # Plotting the vectors
    v1 = X[index1]
    v2 = X[index2]
    ax.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1, color='r')
    ax.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1, color='b')
    
    # Setting the plot limits
    lim = max(np.linalg.norm(v1), np.linalg.norm(v2)) + 1
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    
    # Additional plot settings
    # ax.set_aspect('equal')
    # plt.grid(True)
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.title('Cosine Distance between Two Vectors')
    # plt.show()
