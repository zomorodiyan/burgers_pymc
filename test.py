import unittest
import numpy as np
from BurgersClass import Burgers
from DataClass import Data

class TestBurgers(unittest.TestCase):
    def test_init(self):
        fom = Burgers(grid_size=5)
        np.testing.assert_array_almost_equal(fom.u,np.array([0,1,0,-1,0]))

    def test_func(self):
        # tests the method for evaluating dudt
        fom = Burgers(grid_size=21) #i.e. dx=0.1
        fom.nu = 0.07
        fom.u[0] = 1; fom.u[1] = 1; fom.u[2] = 2
        self.assertAlmostEqual(fom.func(1), 7)
        fom.u[0] = -1; fom.u[1] = -2; fom.u[2] = -3
        self.assertAlmostEqual(fom.func(1), -20)

    def test_step(self):
        # tests if step() method updates dudt and u correctly
        fom = Burgers(grid_size=5,nu=0.1,dt=0.1) # dx=0.5 u=[0,1.0,0,-1.0,0]
        fom.step()
        np.testing.assert_array_almost_equal(fom.dudt
                ,np.array([0,-2.8,0,2.8,0]))
        np.testing.assert_array_almost_equal(fom.u
                ,np.array([0,0.72,0,-0.72,0]))

class TestData(unittest.TestCase):
    def test_collect(self):
        dat = Data()
        dat.collect(um=1,ui=2,up=3,dudt=4)
        self.assertAlmostEqual(dat.um, 1)
        self.assertAlmostEqual(dat.ui, 2)
        self.assertAlmostEqual(dat.up, 3)
        self.assertAlmostEqual(dat.dudt, 4)

if __name__ == '__main__':
    unittest.main()
