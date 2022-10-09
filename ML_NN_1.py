# This notebook shows how to:
# - implement the forward-pass (evaluation) of a deep, fully connected neural network in a few lines of python
# - do that efficiently using batches
# - illustrate the results for randomly initialized neural networks

# cell 1

from numpy import array, zeros, exp, random, dot, shape, reshape, meshgrid, linspace

import matplotlib.pyplot as plt # for plotting
import matplotlib
matplotlib.rcParams['figure.dpi']=300 # highres display

# cell 2

N0=3 # input layer size
N1=2 # output layer size

# initialize random weights: array dimensions N1xN0
w=random.uniform(low=-1,high=+1,size=(N1,N0))
# initialize random biases: N1 vector
b=random.uniform(low=-1,high=+1,size=N1)
