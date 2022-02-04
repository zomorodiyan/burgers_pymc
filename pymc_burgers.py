import numpy as np
from BurgersClass import Burgers
from DataClass import Data
import random
import matplotlib.pyplot as plt
import arviz as az
az.style.use("arviz-darkgrid")
from IPython.display import display
import pymc3 as pm
import theano.tensor as tt
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

# solve & collect dat
dat = Data()
fom = Burgers(grid_size=101,dt=0.0005,nu=0.2)
for i in range(1001):
    fom.up_dudt()
    if(i%50==0):
        dat.collect(fom.u[74],fom.u[75],fom.u[76]\
                ,fom.dudt[75]*(1+random.uniform(-0.05, 0.05)))
    fom.up_u()

# define model
basic_model = pm.Model()
with basic_model:
    # Priors for unknown model parameters
    nu = pm.Normal("nu", mu=10, sigma=10)
    um = dat.um; ui = dat.ui; up = dat.up; dx = fom.dx

    # Expected value of outcome
    umax = tt.switch((ui>0).all(),ui,0)
    umin = tt.switch((ui<0).all(),ui,0)
    dudt = umax*(um-ui)/dx + umin*(ui-up)/dx + nu*(up-2*ui+um)/dx**2

    # Likelihood (sampling distribution) of observations
    pm.Normal("y", mu=dudt,observed=dat.dudt)

# trace/sample model
with basic_model:
    trace = pm.sample(500, chains=2)

# plot outputs
with basic_model:
    pm.plot_trace(trace) # pm.plot_trace(trace,['nu','sigma'])
    mngr = plt.get_current_fig_manager()
    mngr.window.setGeometry(0,-500,600,400) #(left,top,width,height)
    plt.show()

print('last five nu samples:',trace["nu"][-5:])
with basic_model:
    display(az.summary(trace, round_to=2))
