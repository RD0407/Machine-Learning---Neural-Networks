
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

# cell 3

# take arbitrary parameters as starting point
theta=np.array([1.5,-2.3])

x=samples(100)
# illustrate the parametrized function, at sampled points,
# compare against actual function
plt.scatter(x,f(theta,x),color="orange")
plt.scatter(x,true_f(x),color="blue")
plt.show()


# cell:4

theta0s=np.linspace(-3,6,40)
theta1s=np.linspace(-2,3,40)
C=get_avg_cost(theta0s,theta1s,10000)
nlevels=20
X,Y=np.meshgrid(theta0s,theta1s,indexing='ij')
plt.contourf(X,Y,C,nlevels)
plt.contour(X,Y,C,nlevels,colors="white")
plt.xlabel("theta_0")
plt.ylabel("theta_1")
plt.show()



# cell: 6

# Now we do gradient descent and, for each step,
# we plot the (sampled) true function vs. the parametrized function
# We also plot the current location of parameters theta
# (over the average cost function)

# import functions for updating display 
# (simple animation)

from IPython.display import clear_output
from time import sleep


# take arbitrary parameters as starting point
theta=np.array([-1.0,2.0])

# do many steps of stochastic gradient descent,
# continue showing the comparison!
eta=.2 # "learning rate" (gradient descent step size)
nsamples=10 # stochastic x samples used per step
nsteps=100 # how many steps we take

x_sweep=np.linspace(-4,4,300)
