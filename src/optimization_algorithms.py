"""
Created on Sat Mar 14 11:51:52 2020

@author: Simone Gayed Said
@author: Pierpasquale Colagrande
"""

import numpy as np
from src.ui import print_head, print_iteration, print_found_minimum

# Unused import but necessary
from mpl_toolkits.mplot3d import Axes3D


def sgd(f, theta, gradient, num_iterations=5000, alpha=0.001):
    algorithm = "SGD"
    print_head(algorithm, {'num_iterations': num_iterations, 'alpha': alpha})

    x_data, y_data, z_data = [theta[0]], [theta[1]], [f(theta[0], theta[1])]
    g = np.zeros(shape=2)
    t = 0
    while t < num_iterations:
        t = t + 1
        g[0] = gradient['x'](theta[0], theta[1])
        g[1] = gradient['y'](theta[0], theta[1])
        theta = theta - (alpha * g)
        x_data.append(theta[0])
        y_data.append(theta[1])
        z_data.append(f(theta[0], theta[1]))
        if t % (num_iterations / 10) == 0:
            print_iteration(theta, t)
    print_found_minimum(theta, t)
    return x_data, y_data, z_data


# Adam is an adaptive learning rate method, which means, it computes individual learning rates for different parameters.
# It uses estimations of first and second moments of gradient to adapt the learning rate.

def adam(f, theta, gradient, num_iterations=5000, alpha=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
    algorithm = "Adam"
    print_head(algorithm, {'num_iterations': num_iterations, 'alpha': alpha, 'beta_1': beta_1, 'beta_2': beta_2,
                           'epsilon': epsilon})

    x_data, y_data, z_data = [theta[0]], [theta[1]], [f(theta[0], theta[1])]
    m = 0
    v = 0
    g = np.zeros(shape=2)
    t = 0
    while t < num_iterations:
        t = t + 1
        g[0] = gradient['x'](theta[0], theta[1])
        g[1] = gradient['y'](theta[0], theta[1])
        m = beta_1 * m + (1 - beta_1) * g
        v = beta_2 * v + (1 - beta_2) * np.power(g, 2)
        m_hat = m / (1 - np.power(beta_1, t))
        v_hat = v / (1 - np.power(beta_2, t))
        theta = theta - alpha * m_hat / (np.sqrt(v_hat) + epsilon)
        x_data.append(theta[0])
        y_data.append(theta[1])
        z_data.append(f(theta[0], theta[1]))
        if t % (num_iterations / 10) == 0:
            print_iteration(theta, t)
    print_found_minimum(theta, t)
    return x_data, y_data, z_data


# The idea with Adamax is to look at the value of the second moment (v) as the L2 norm of the individual current and
# past gradients.

def adamax(f, theta, gradient, num_iterations=5000, alpha=0.001, beta_1=0.9, beta_2=0.999):
    algorithm = "AdaMax"
    print_head(algorithm, {'num_iterations': num_iterations, 'alpha': alpha, 'beta_1': beta_1, 'beta_2': beta_2})

    x_data, y_data, z_data = [theta[0]], [theta[1]], [f(theta[0], theta[1])]
    m = 0
    v = 0
    g = np.zeros(shape=2)
    t = 0
    while t < num_iterations:
        t = t + 1
        g[0] = gradient['x'](theta[0], theta[1])
        g[1] = gradient['y'](theta[0], theta[1])
        m = beta_1 * m + (1 - beta_1) * g
        m_hat = m / (1 - np.power(beta_1, t))
        v = np.maximum(beta_2 * v, np.abs(g))
        theta = theta - alpha * m_hat / v
        x_data.append(theta[0])
        y_data.append(theta[1])
        z_data.append(f(theta[0], theta[1]))
        if t % (num_iterations / 10) == 0:
            print_iteration(theta, t)
    print_found_minimum(theta, t)
    return x_data, y_data, z_data


# The idea of Nadam is to use Nesterov momentum term for the first moving averages. So, with Nesterov accelerated
# momentum we first make make a big jump in the direction of the previous accumulated gradient and then measure the
# gradient where we ended up to make a correction.

def nadam(f, theta, gradient, num_iterations=5000, alpha=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
    algorithm = "Nadam"
    print_head(algorithm, {'num_iterations': num_iterations, 'alpha': alpha, 'beta_1': beta_1, 'beta_2': beta_2,
                           'epsilon': epsilon})

    x_data, y_data, z_data = [theta[0]], [theta[1]], [f(theta[0], theta[1])]
    m = 0
    v = 0
    g = np.zeros(shape=2)
    t = 0
    while t < num_iterations:
        t = t + 1
        g[0] = gradient['x'](theta[0], theta[1])
        g[1] = gradient['y'](theta[0], theta[1])
        m = beta_1 * m + (1 - beta_1) * g
        v = beta_2 * v + (1 - beta_2) * np.power(g, 2)
        m_hat = m / (1 - np.power(beta_1, t)) + (1 - beta_1) * g / (1 - np.power(beta_1, t))
        v_hat = v / (1 - np.power(beta_2, t))
        theta = theta - alpha * m_hat / (np.sqrt(v_hat) + epsilon)
        x_data.append(theta[0])
        y_data.append(theta[1])
        z_data.append(f(theta[0], theta[1]))
        if t % (num_iterations / 10) == 0:
            print_iteration(theta, t)
    print_found_minimum(theta, t)
    return x_data, y_data, z_data


# Algorithm created to fix the convergence issues of Adam by endowing such algorithms with “long-term memory” of past
# gradients. This algorithm often also lead to improved empirical performance.

def amsgrad(f, theta, gradient, num_iterations=5000, alpha=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
    algorithm = "AMSGrad"
    print_head(algorithm, {'num_iterations': num_iterations, 'alpha': alpha, 'beta_1': beta_1, 'beta_2': beta_2,
                           'epsilon': epsilon})

    x_data, y_data, z_data = [theta[0]], [theta[1]], [f(theta[0], theta[1])]
    m = 0
    v = 0
    v_hat = 0
    g = np.zeros(shape=2)
    t = 0
    while t < num_iterations:
        t = t + 1
        g[0] = gradient['x'](theta[0], theta[1])
        g[1] = gradient['y'](theta[0], theta[1])
        m = beta_1 * m + (1 - beta_1) * g
        v = beta_2 * v + (1 - beta_2) * np.power(g, 2)
        v_hat = np.maximum(v, v_hat)
        theta = theta - alpha * m / (np.sqrt(v_hat) + epsilon)
        x_data.append(theta[0])
        y_data.append(theta[1])
        z_data.append(f(theta[0], theta[1]))
        if t % (num_iterations / 10) == 0:
            print_iteration(theta, t)
    print_found_minimum(theta, t)
    return x_data, y_data, z_data
