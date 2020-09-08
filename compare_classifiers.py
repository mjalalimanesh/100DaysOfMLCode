"""
Comparison between convergence of perceptron, average perceptron,
and pegasos linear classifiers on a toy dataset
Created on Mon Sep  7 17:43:05 2020
@author: mohamad jalalimanesh
"""
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from utils import create_toy_data, plot_toy_data
from perceptron import perceptron
from averaged_perceptron import average_perceptron
from pegasos import pegasos

features, labels = create_toy_data(350)
fig, ax = plot_toy_data(features, labels)

xmin, xmax = plt.axis()[:2]
x = np.linspace(xmin, xmax)

lines = []
plotlabels = ['perceptron', 'average perceptron', 'pegasos']
plotcols = ["black", "m", "g"]

for index in range(3):
    lobj, = ax.plot(x, np.zeros(x.shape), lw=3, color=plotcols[index], \
                    label=plotlabels[index])
    lines.append(lobj)
    ax.set_title('epoch = {}'.format(str(0)))
    ax.legend(loc='upper right')

def init():
    """
    init function fot animation module
    Returns line objects
    """
    for line in lines:
        line.set_ydata(np.zeros(x.shape))
    return lines

def animate(i):
    """
    update function for animation module
    i : frame number
    """
    for lnum, line in enumerate(lines):
        if lnum == 0:
            theta, theta_0 = perceptron(features, labels, T=i+1)
            y = -(theta[0]*x + theta_0) / (theta[1] + 1e-16)
            line.set_ydata(y)  # update the data.
        elif lnum == 1:
            theta, theta_0 = average_perceptron(features, labels, T=i+1)
            y = -(theta[0]*x + theta_0) / (theta[1] + 1e-16)
            line.set_ydata(y)  # update the data.
        elif lnum == 2:
            theta, theta_0 = pegasos(features, labels, T=i+1, L=0.2)
            y = -(theta[0]*x + theta_0) / (theta[1] + 1e-16)
            line.set_ydata(y)  # update the data.

    ax.set_title('epoch = {}'.format(str(i+1)))
    return lines

ani = animation.FuncAnimation(
    fig, func=animate, init_func=init, frames=50, interval=10,  \
        repeat_delay=500, blit=False, repeat=True)

# writer1 = animation.PillowWriter(fps=10)
# ani.save("movie.gif", writer=writer1)
    