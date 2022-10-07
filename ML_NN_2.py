
# # Nonlinear curve fitting by stochastic gradient descent
# This notebook shows how stochastic gradient descent can help fit an arbitrary function (neural networks essentially do the same, but in much higher dimensions and with many more parameters)
import numpy as np

import matplotlib.pyplot as plt # for plotting
import matplotlib
matplotlib.rcParams['figure.dpi']=300 # highres display

# Define the parametrized nonlinear function
def f(theta,x):
    """
    theta are the parameters
    x are the input values (can be an array)
    """
    return(theta[0]/((x-theta[1])**2+1.0))

# Define the gradient of the parametrized function
# with respect to its parameters
def f_grad(theta,x):
    """
    Return the gradient of f with respect to theta
    shape [n_theta,n_samples]
    where n_theta=len(theta)
    and n_samples=len(x)
    """
    return(np.array([
        1./((x-theta[1])**2+1.0)
    ,
        2*(x-theta[1])*theta[0]/((x-theta[1])**2+1.0)
    ]
    ))


# Define the actual function (the target, to be fitted)
def true_f(x):
    return( 3.0/((x-0.5)**2+1.0) )

# Get randomly sampled x values
def samples(nsamples,width=2.0):
    return(width*np.random.randn(nsamples))

# Get the average cost function on a grid of 2 parameters
def get_avg_cost(theta0s,theta1s,nsamples):
    n0=len(theta0s)
    n1=len(theta1s)
    C=np.zeros([n0,n1])
    for j0 in range(n0):
        for j1 in range(n1):
            theta=np.array([theta0s[j0],theta1s[j1]])
            x=samples(nsamples)
            C[j0,j1]=0.5*np.average((f(theta,x)-true_f(x))**2)
    return(C)
