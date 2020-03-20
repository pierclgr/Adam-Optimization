"""
Created on Sat Mar 14 11:51:52 2020

@author: Simone Gayed Said
@author: Pierpasquale Colagrande
"""

from src.ui import print_introduction, plot_graph, plot_3d_minimization_procedure, \
    plot_2d_minimization_procedure  # ,plot_3d_surface
from src.optimization_algorithms import sgd, adam, adamax, nadam, amsgrad
from src.functions import sixhump_camel_function, easom_function

print_introduction("Six-hump Camel")

# f, X, Y, gradient, theta = easom_function()

f, X, Y, gradient, theta = sixhump_camel_function()

Z = f(X, Y)

x_data_sgd, y_data_sgd, z_data_sgd = sgd(f, theta, gradient)
x_data_adam, y_data_adam, z_data_adam = adam(f, theta, gradient)
x_data_adamax, y_data_adamax, z_data_adamax = adamax(f, theta, gradient)
x_data_nadam, y_data_nadam, z_data_nadam = nadam(f, theta, gradient)
x_data_amsgrad, y_data_amsgrad, z_data_amsgrad = amsgrad(f, theta, gradient)

x_data = {'sgd': x_data_sgd, 'adam': x_data_adam, 'adamax': x_data_adamax, 'nadam': x_data_nadam,
          'amsgrad': x_data_amsgrad}

y_data = {'sgd': y_data_sgd, 'adam': y_data_adam, 'adamax': y_data_adamax, 'nadam': y_data_nadam,
          'amsgrad': y_data_amsgrad}

z_data = {'sgd': z_data_sgd, 'adam': z_data_adam, 'adamax': z_data_adamax, 'nadam': z_data_nadam,
          'amsgrad': z_data_amsgrad}

plot_3d_minimization_procedure(X, Y, Z, x_data, y_data, z_data)

plot_2d_minimization_procedure(X, Y, Z, x_data, y_data)

plot_graph(x_data)
plot_graph(y_data, 'y')

"""
plot_3d_surface(X, Y, Z)
"""