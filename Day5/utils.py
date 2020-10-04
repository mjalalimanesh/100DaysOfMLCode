import numpy as np
import matplotlib.pyplot as plt
import csv


def get_order(n_samples):
    """
    return a shuffled array of integers in range(n_samples)
    """
    np.random.seed(1)
    indices = list(range(n_samples))
    np.random.shuffle(indices)
    return indices


def create_toy_data(n_samples):
    """
    return 2D toy data as nparray with two classes
    +1 and -1 with normal distributions with specified
    mean and std

    Parameters
    ----------
    n_samples : number of data points

    Returns
    -------
    features : (n_samples, 2) array 
    labels : +1 or -1
        labels for each datapoint.

    """
    n1 = int(n_samples/2) # first class
    n2 = n_samples - n1 # second class
    features = np.zeros((n_samples, 2))
    np.random.seed(10)
    # first class correspond to label 1
    xmean1 = 2
    xstd1 = 1
    ymean1 = 2
    ystd1 = 1
    np.random.seed(10)
    features[0:n1, 0] = xmean1 + np.random.randn(n1) * xstd1
    features[0:n1, 1] = ymean1 + np.random.randn(n1) * ystd1
    # second class correspond to label 2
    xmean2 = 3.5
    xstd2 = 1
    ymean2 = 3.5
    ystd2 = 1
    features[n1:n_samples, 0] = xmean2 + np.random.randn(n2) * xstd2
    features[n1:n_samples, 1] = ymean2 + np.random.randn(n2) * ystd2

    labels = np.repeat([1, -1], repeats=(n1, n2))
    return features, labels

def plot_toy_data(features, labels):
    """
    plot toy data
    """
    fig, ax = plt.subplots(1, 1)
    colors = ['r' if label == 1 else 'b' for label in labels]
    ax.scatter(features[:, 0], features[:, 1], c=colors)
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    
    return fig, ax
    
def plot_decision_boundary(thetas, ax, algo_name):
    xmin, xmax = plt.axis()[:2]
    theta, theta_0 = thetas
    x = np.linspace(xmin, xmax)
    y = -(theta[0]*x + theta_0) / (theta[1] + 1e-16)
    line, = ax.plot(x, y, 'k-', label=algo_name)
    ax.set_title("classifying toy data with {} algorithm".format(algo_name))
    ax.legend()
    return line, y
    
def load_data(path_data, extras=False):
    """
    Returns a list of dict with keys:
    * sentiment: +1 or -1 if the review was positive or negative, respectively
    * text: the text of the review

    Additionally, if the `extras` argument is True, each dict will also include the
    following information:
    * productId: a string that uniquely identifies each product
    * userId: a string that uniquely identifies each user
    * summary: the title of the review
    * helpfulY: the number of users who thought this review was helpful
    * helpfulN: the number of users who thought this review was NOT helpful
    """


    basic_fields = {'sentiment', 'text'}
    numeric_fields = {'sentiment', 'helpfulY', 'helpfulN'}

    data = []
    f_data = open(path_data, encoding="latin1")

    for datum in csv.DictReader(f_data, delimiter='\t'):
        for field in list(datum.keys()):
            if not extras and field not in basic_fields:
                del datum[field]
            elif field in numeric_fields and datum[field]:
                datum[field] = int(datum[field])

        data.append(datum)

    f_data.close()

    return data

def most_explanatory_word(theta, wordlist):
    """Returns the word associated with the bag-of-words feature having largest weight."""
    return [word for (theta_i, word) in sorted(zip(theta, wordlist))[::-1]]
