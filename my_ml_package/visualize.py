from matplotlib import pyplot as plt
from scipy import stats

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