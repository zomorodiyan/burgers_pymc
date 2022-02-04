import numpy as np
print()

class Data:
    def __init__(self,size=0):
        self.um = np.zeros((size))
        self.ui = np.zeros((size))
        self.up = np.zeros((size))
        self.dudt = np.zeros((size))

    def collect(self,um,ui,up,dudt):
        self.um = np.append(self.um,um)
        self.ui = np.append(self.ui,ui)
        self.up = np.append(self.up,up)
        self.dudt = np.append(self.dudt,dudt)
