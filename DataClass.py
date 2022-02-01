import numpy as np
print()

class Data:
    def __init__(self,size=0):
        self.um = np.zeros((size))
        self.ui = np.zeros((size))
        self.up = np.zeros((size))
        self.u_min = np.zeros((size))
        self.u_max = np.zeros((size))
        self.dudt = np.zeros((size))

    def collect(self,um,ui,up,dudt):
        self.um = np.append(self.um,um)
        self.ui = np.append(self.ui,ui)
        self.up = np.append(self.up,up)
        self.u_min = np.append(self.u_min,np.min([0,ui]))
        self.u_max = np.append(self.u_max,np.max([0,ui]))
        self.dudt = np.append(self.dudt,dudt)
