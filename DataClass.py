import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
print()

class Data:
    def __init__(self,size):
        self.u = np.zeros((0,size));
        self.dudt = np.zeros((0,size))

    def collect(self,u,dudt):
        self.u = np.vstack((self.u,u))
        self.dudt = np.vstack((self.dudt,dudt))

    def plot(self,x):
        fig = plt.figure()
        cmap = plt.get_cmap('PRGn', 11)
        for i in range(11):
            plt.plot(x,self.u[i,:],c=cmap(i))
        for i in range(11):
            plt.plot(x[74:77],self.u[i,74:77],'.r')
        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = plt.colorbar(sm, ticks=np.linspace(0, 1, 11))
        cbar.ax.get_yaxis().labelpad = 15 # make padding for colorbar label
        cbar.set_label('time', rotation=270)
        plt.xlabel('x')
        plt.ylabel(r'${u}$', rotation=0)
        plt.savefig('./fig/fig1.png')
        plt.show()

        fig = plt.figure()
        cmap = plt.get_cmap('PRGn', 11)
        for i in range(11):
            plt.plot(x,self.dudt[i,:],c=cmap(i))
        for i in range(11):
            plt.plot(x[75],self.dudt[i,75],'.r')
        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = plt.colorbar(sm, ticks=np.linspace(0, 1, 11))
        cbar.ax.get_yaxis().labelpad = 15 # make padding for colorbar label
        cbar.set_label('time', rotation=270)
        plt.xlabel('x')
        plt.ylabel(r'$\frac{du}{dt}$', rotation=0)
        plt.savefig('./fig/fig2.png')
        plt.show()
