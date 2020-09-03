import numpy as np
from project1 import get_order

def perceptron_single_step_update(
        feature_vector,
        label,
        current_theta,
        current_theta_0):
    """
    Properly updates the classification parameter, theta and theta_0, on a
    single step of the perceptron algorithm.

    Args:
        feature_vector - A numpy array describing a single data point.
        label - The correct classification of the feature vector.
        current_theta - The current theta being used by the perceptron
            algorithm before this update.
        current_theta_0 - The current theta_0 being used by the perceptron
            algorithm before this update.

    Returns: A tuple where the first element is a numpy array with the value of
    theta after the current update has completed and the second element is a
    real valued number with the value of theta_0 after the current updated has
    completed.
    """
    condition = label*(np.dot(feature_vector, current_theta ) + current_theta_0) <= 1e-7
    if condition:
        theta = current_theta + label*feature_vector
        theta_0 = current_theta_0 + label
        return theta, theta_0
    return current_theta, current_theta_0



def average_perceptron(feature_matrix, labels, T):
    """
    Runs the average perceptron algorithm on a given set of data.
	Classifier(x) = sign(theta.x + theta_0)
    
	Args:
        feature_matrix
        labels 
        T - times the perceptron iterate through the feature matrix.

    Returns: 
        theta, theta_0 - tuple  first element nparray average
        theta second element is a real
        number with the value of the average theta_0.

    """
    nsamples, nfeatures = feature_matrix.shape
    theta = np.zeros((nfeatures,))
    theta_0 = 0
    sum_theta = np.zeros((nfeatures,))
    sum_theta_0 = 0
    for t in range(T):
        for i in get_order(nsamples):
            theta, theta_0 = perceptron_single_step_update(
                                feature_matrix[i],
                                labels[i],
                                theta,
                                theta_0)
            sum_theta = sum_theta + theta
            sum_theta_0 = sum_theta_0 + theta_0
    theta_averaged = sum_theta / (nsamples*T)
    theta_0_averaged = sum_theta_0 / (nsamples*T)
    return theta_averaged, theta_0_averaged
