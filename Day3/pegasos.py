import numpy as np
from utils import get_order

def pegasos_single_step_update(
        feature_vector,
        label,
        L,
        eta,
        current_theta,
        current_theta_0):
    """
    Properly updates the classification parameter, theta and theta_0, on a
    single step of the Pegasos algorithm

    Args:
        feature_vector - A numpy array describing a single data point.
        label - The correct classification of the feature vector.
        L - The lamba value being used to update the parameters.
        eta - Learning rate to update parameters.
        current_theta - The current theta being used by the Pegasos
            algorithm before this update.
        current_theta_0 - The current theta_0 being used by the
            Pegasos algorithm before this update.

    Returns: A tuple where the first element is a numpy array with the value of
    theta after the current update has completed and the second element is a
    real valued number with the value of theta_0 after the current updated has
    completed.
    """
    state = np.dot(feature_vector,current_theta) + current_theta_0
    condition = (label * state) < 1
    if condition:
        theta = (1 - L*eta)*current_theta + eta*label*feature_vector
        theta_0 = current_theta_0 + eta * label
        return theta, theta_0
    return (1-L*eta)*current_theta, current_theta_0


def pegasos(feature_matrix, labels, T, L):
    """
    Runs the Pegasos algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.

    For each update, set learning rate = 1/sqrt(t),
    where t is a counter for the number of updates performed so far (between 1
    and nT inclusive).


    Args:
        feature_matrix - A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the algorithm
            should iterate through the feature matrix.
        L - The lamba value being used to update the Pegasos
            algorithm parameters.

    Returns: A tuple where the first element is a numpy array with the value of
    the theta, the linear classification parameter, found after T
    iterations through the feature matrix and the second element is a real
    number with the value of the theta_0, the offset classification
    parameter, found after T iterations through the feature matrix.
    """
    theta = np.zeros((feature_matrix.shape[1],))
    theta_0 = 0
    update_counter = 0 # number_of updates
    for t in range(T):
        for i in get_order(feature_matrix.shape[0]):
            update_counter += 1
            eta = 1.0 / np.sqrt(update_counter)
            theta, theta_0 = pegasos_single_step_update(
                                feature_matrix[i],
                                labels[i],
                                L,
                                eta,
                                theta,
                                theta_0)
    return theta, theta_0
