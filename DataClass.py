import numpy as np
print()

class Data:
    def __init__(self,size):
        self.u = np.zeros((0,size));
        self.dudt = np.zeros((0,size))

    def collect(self,u,dudt):
        self.u = np.vstack((self.u,u))
        self.dudt = np.vstack((self.dudt,dudt))
