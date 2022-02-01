import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
from BurgersClass import Burgers
from DataClass import Data
import arviz as az
az.style.use("arviz-darkgrid")
import pymc3 as pm
import random

# generate and collect data
dat = Data()
fom = Burgers(grid_size=101,dt=0.0005,nu=0.1)
for i in range(1001):
    fom.up_dudt()
    if(i%50==0):
        dat.collect(fom.u[74],fom.u[75],fom.u[76]\
                ,fom.dudt[75]+random.uniform(-0.05, 0.05))
    fom.up_u()

basic_model = pm.Model()



with basic_model:
    # Priors for unknown model parameters
    nu = pm.Normal("nu", mu=5, sigma=5)
    sigma = pm.HalfNormal("sigma", sigma=0.5)

    um = dat.um; ui = dat.ui; up = dat.up; umax = dat.u_max; umin = dat.u_min
    dx = fom.dx
    # Expected value of outcome
    mu = umax*(um-ui)/dx + umin*(ui-up)/dx + nu*(up-2*ui+um)/dx**2

    # Likelihood (sampling distribution) of observations
    Y_obs = pm.Normal("Y_obs", mu=mu, sigma=sigma, observed=dat.dudt)

map_estimate = pm.find_MAP(model=basic_model)
map_estimate

with basic_model:
    # draw 500 posterior samples
    trace = pm.sample(500, return_inferencedata=False, chains=2)

trace["nu"][-5:]

with basic_model:
    az.plot_trace(trace);
mngr = plt.get_current_fig_manager()
mngr.window.setGeometry(0,-500,600,400)

plt.show()

with basic_model:
    display(az.summary(trace, round_to=2))
#problem: in func I create new variables for np.max([0,ui]) & np.min([0,ui])
