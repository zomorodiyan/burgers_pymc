import numpy as np
from BurgersClass import Burgers
from DataClass import Data
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
import arviz as az
az.style.use("arviz-darkgrid")
from IPython.display import display
import pymc3 as pm
import theano.tensor as tt
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

# solve & collect data
data = Data(101)
fom = Burgers(grid_size=101,dt=0.0005,nu=0.2)
for i in range(1001):
    fom.up_dudt()
    if(i%100==0): data.collect(fom.u,fom.dudt)
    fom.up_u()

data.plot(fom.x)

# define model
basic_model = pm.Model()
with basic_model:
    # Priors for unknown model parameters
    nu = pm.Normal("nu", mu=100, sigma=100)

    # Expected value of outcome
    um = data.u[:,74]; ui = data.u[:,75]; up = data.u[:,76]; dx = fom.dx
    umax = tt.switch((ui>0).all(),ui,0); umin = tt.switch((ui<0).all(),ui,0)
    dudt = umax*(um-ui)/dx + umin*(ui-up)/dx + nu*(up-2*ui+um)/dx**2

    # Likelihood (sampling distribution) of observations
    observed = np.zeros(len(data.dudt[:,75]))
    for i in range(len(data.dudt[:,75])):
        observed[i] = data.dudt[i,75]*(1+random.uniform(-0.5,+0.5))
    pm.Normal("y", mu=dudt,observed=observed)

# trace/sample model
with basic_model:
    trace = pm.sample(500, chains=2)

# plot outputs
with basic_model:
    pm.plot_trace(trace) # pm.plot_trace(trace,['nu','sigma'])
    mngr = plt.get_current_fig_manager()
    mngr.window.setGeometry(0,-500,600,400) #(left,top,width,height)
    plt.savefig('./fig/fig3.png')
    plt.show()

print('last five nu samples:',trace["nu"][-5:])
with basic_model:
    display(az.summary(trace, round_to=2))
