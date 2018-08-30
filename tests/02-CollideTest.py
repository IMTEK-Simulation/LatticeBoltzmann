#!/usr/bin/env python3
"""
Copyright 2017-2018 Lars Pastewka, Andreas Greiner

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

try:
    import unittest
    import numpy as np
    from PyLB import collide, equilibrium
    from .PyLBTest import PyLBTestCase, skip
except ImportError as err:
    import sys
    print(err)
    sys.exit(-1)

###

# "Velocities" of individual channels
c_ic = np.array([[0,  1,  0, -1,  0,  1, -1, -1,  1],    # velocities, x components
                 [0,  0,  1,  0, -1,  1,  1, -1, -1]]).T # velocities, y components

# Weight factors
w_i = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])

def d2q9_equilibrium_ref(rho_kl, u_ckl):
    """
    Return equilibrium distribution function.

    Parameters
    ----------
    rho_kl : array
        Array containing densities. Array is 2-dimensional corresponding to the
        x- and y-positions of the grid.
    u_ckl : array
        Array containing the streaming velocities. Array is 2-dimensional, with
        the first dimension being the Cartesian directions of the velocity
        vectors and the next two dimension x- and y-positions.
    """
    cu_ikl = np.dot(u_ckl.T, c_ic.T).T
    uu_kl = np.sum(u_ckl**2, axis=0)
    return (w_i*(rho_kl*(1 + 3*cu_ikl + 9/2*cu_ikl**2 - 3/2*uu_kl)).T).T


def d2q9_collide_ref(f_ikl, omega):
    """
    Collision step: Relax occupation numbers towards equilibrium occupation.

    Parameters
    ----------
    f_ikl : array
        Array containing the occupation numbers. Array is 3-dimensional, with
        the first dimension running from 0 to 8 and indicating channel. The
        next two dimensions are x- and y-position. This array is modified in
        place.
    omega : float
        Relaxation parameter.

    Returns
    -------
    rho_kl : array
        Array containing densities. Array is 2-dimensional corresponding to the
        x- and y-positions of the grid.
    u_ckl : array
        Array containing the streaming velocities. Array is 2-dimensional, with
        the first dimension being the Cartesian directions of the velocity
        vectors and the next two dimension x- and y-positions.
    """
    rho_kl = np.sum(f_ikl, axis=0)
    u_ckl = np.dot(f_ikl.T, c_ic).T/rho_kl
    f_ikl += omega*(d2q9_equilibrium_ref(rho_kl, u_ckl) - f_ikl)
    return rho_kl, u_ckl

###

class CollideTest(PyLBTestCase):
    def test_D2Q9_equilibrium(self):
        rho_kl = np.abs(np.random.random([2, 2]))
        ux_kl = np.random.random(rho_kl.shape)
        uy_kl = np.random.random(rho_kl.shape)
        e1 = np.zeros([9]+list(rho_kl.shape))
        equilibrium(rho_kl.reshape(-1), ux_kl.reshape(-1), uy_kl.reshape(-1),
                    e1.reshape(9, -1))
        e2 = d2q9_equilibrium_ref(rho_kl, np.array([ux_kl, uy_kl]))
        self.assertArrayAlmostEqual(e1, e2)

    def test_D2Q9_collide(self):
        f_ikl = np.abs(np.random.random([9, 4, 4]))
        for omega in [0.5, 1.7]:
            c1 = f_ikl.copy()
            c2 = f_ikl.copy()
            collide(c1.reshape(9, -1), omega)
            d2q9_collide_ref(c2, omega)
            self.assertArrayAlmostEqual(c1, c2)

