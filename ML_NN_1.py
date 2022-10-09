# This notebook shows how to:
# - implement the forward-pass (evaluation) of a deep, fully connected neural network in a few lines of python
# - do that efficiently using batches
# - illustrate the results for randomly initialized neural networks

from numpy import array, zeros, exp, random, dot, shape, reshape, meshgrid, linspace

import matplotlib.pyplot as plt # for plotting
import matplotlib
matplotlib.rcParams['figure.dpi']=300 # highres display
