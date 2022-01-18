import numpy as np
import matplotlib.pyplot as plt
from BurgersClass import Burgers
from DataClass import Data

fom = Burgers(grid_size=101,dt=0.001,nu=0.1)
plt.plot(fom.x,fom.u)
for i in range(1001):
    fom.step()
    if(i%100==0):
        plt.plot(fom.x,fom.u)
plt.show()

'''
import arviz as az
az.style.use("arviz-darkgrid")
import matplotlib.pyplot as plt
from IPython.display import display


N = 20
n = 101
x = np.linspace(-1,1,n)
dx = 2/(n-1)
dt = 0.001
u = -1*np.sin(np.pi*x)

# variables for data collection
U = np.empty((N))
Um = np.empty((N))
Up = np.empty((N))
Y = np.empty((N))

rng = np.random.default_rng(seed=42)
for nu in [0.1]: # True parameter values
    u = -np.sin(np.pi*x)
    for j in range(N):
        for k in range(50):
            for i in range (1,len(u)-1):
                dudt = nu*(u[i+1]-2*u[i]+u[i-1])/dx**2\
                     -np.max([0,u[i]])*(u[i]-u[i-1])/dx\
                     -np.min([0,u[i]])*(u[i+1]-u[i])/dx

               # data collection
                if(i==74):
                 Um[j]=u[i]
                if(i==75):
                  Y[j]=dudt+(rng.random((1))-0.5)/10
                  U[j]=u[i]
                if(i==76):
                  Up[j]=u[76]
                u[i] = u[i]+dudt*dt


fig, axis = plt.subplots(1, 1, figsize=(5, 4))
axis.scatter(U, Y, alpha=0.6)
axis.set_ylabel("Y")
axis.set_xlabel("X");


import pymc3 as pm
print(f"Running on PyMC3 v{pm.__version__}")

Umax = np.empty_like(U)
Umin = np.empty_like(U)
for i in range(len(U)):
  Umax[i] = np.max([0,U[i]])
  Umin[i] = np.min([0,U[i]])

basic_model = pm.Model()

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    with basic_model:

        # Priors for unknown model parameters
        nu = pm.Normal("nu", mu=0, sigma=10)
        sigma = pm.HalfNormal("sigma", sigma=0.01)

        # Expected value of outcome
        mu = Umax*(Um-U)/dx + Umin*(U-Up)/dx + nu*(Up-2*U+Um)/dx**2

        # Likelihood (sampling distribution) of observations
        Y_obs = pm.Normal("Y_obs", mu=mu, sigma=sigma, observed=Y)

    map_estimate = pm.find_MAP(model=basic_model)
    map_estimate

    with basic_model:
        # draw 500 posterior samples
        trace = pm.sample(100, return_inferencedata=False, chains=2)

    trace["nu"][-5:]

    with basic_model:
        az.plot_trace(trace);
    plt.show()

    with basic_model:
        display(az.summary(trace, round_to=2))
'''
