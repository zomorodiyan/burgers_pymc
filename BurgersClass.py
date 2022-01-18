from math import pi
import numpy as np

class Burgers:
    def __init__(self,grid_size=101,nu=0.1,dt=0.001):
        if (type(grid_size) not in [int]) or grid_size <= 0:
            raise TypeError("grid_numbers must be a positive integer.")
        self.grid_size = grid_size
        self.nu = nu
        self.dt = dt
        self.dx = 2/(grid_size-1) # -1 <= x <= 1
        self.x = np.linspace(-1,1,grid_size)
        self.u = -np.sin(np.pi*self.x)
        self.dudt = np.zeros((grid_size))

    def func(self,i):
        # evaluates dudt
        return self.nu * (self.u[i+1] - 2*self.u[i] + self.u[i-1]) /self.dx**2\
                -np.max([0,self.u[i]]) * (self.u[i] - self.u[i-1]) /self.dx\
                -np.min([0,self.u[i]]) * (self.u[i+1] - self.u[i]) /self.dx

    def step(self):
        # updates dudt and u
        if (type(self.nu) not in [int,float]) or self.nu < 0:
            raise TypeError("grid_numbers must be a non-negative real number.")
        for i in range(1,self.grid_size-1): #dudt at boundaries remain zero
            self.dudt[i] = self.func(i)
        self.u = self.u + self.dudt*self.dt
