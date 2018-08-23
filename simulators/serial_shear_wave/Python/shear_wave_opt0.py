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

--------
| OPT0 |
--------

This is an implementation of the D2Q9 Lattice Boltzmann lattice in the simple
relaxation time approximation. The code reports the amplitude of a decaying
shear wave which can be used to measure viscosity.

The present implementation contains no optimizations but was developed to
yield a concise code. Storage order was chosen such that the indices of
locations on the 2D real-space grid are the fast indices.
"""

import numpy as np

### Parameters

# Lattice
nx = 300
ny = 300

# Number of steps
nsteps = 1000

# Relaxation parameter
omega = 0.3

# Data type
dtype = np.float32

### Auxiliary arrays

# Naming conventions for arrays: Subscript indicates type of dimension
# Example: c_ic <- 2-dimensional array
#            ^^
#            ||--- c: Cartesian index, can have value 0 or 1
#            |---- i: Channel index, can have value 0 to 8
# Example: f_ikl <- 3-dimensional array
#            ^^^
#            |||--- l: y-position, can have value 0 to ny-1
#            ||---- k: x-position, can have value 0 to nx-1
#            |----- i: Channel index, can value 0 to 8

# "Velocities" of individual channels
c_ic = np.array([[0,  1,  0, -1,  0,  1, -1, -1,  1],    # velocities, x components
                 [0,  0,  1,  0, -1,  1,  1, -1, -1]]).T # velocities, y components

# Weight factors
w_i = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36], dtype=dtype)

### Compute functions

def equilibrium(rho_kl, u_ckl):
    """
    Return the equilibrium distribution function.

    Parameters
    ----------
    rho_rl: array
        Fluid density on the 2D grid.
    u_ckl: array
        Streaming velocity on the 2D grid.

    Returns
    -------
    f_ikl: array
        Equilibrium distribution for the given fluid density *rho_kl* and
        streaming velocity *u_ckl*.
    """
    cu_ikl = np.dot(u_ckl.T, c_ic.T.astype(u_ckl.dtype)).T
    uu_kl = np.sum(u_ckl**2, axis=0)
    return (w_i*(rho_kl*(1 + 3*cu_ikl + 9/2*cu_ikl**2 - 3/2*uu_kl)).T).T

def collide(f_ikl, omega):
    """
    Carry out collision step. This relaxes the distribution of particle
    velocities towards its equilibrium.

    Parameters
    ----------
    f_ikl: array
        Distribution of particle velocity on the 2D grid. Note that this array
        is modified in place.
    omega: float
        Relaxation parameter.

    Returns
    -------
    rho_rl: array
        Current fluid density on the 2D grid.
    u_ckl: array
        Current streaming velocity on the 2D grid.
    """
    rho_kl = np.sum(f_ikl, axis=0)
    u_ckl = np.dot(f_ikl.T, c_ic).T/rho_kl
    f_ikl += omega*(equilibrium(rho_kl, u_ckl) - f_ikl)
    return rho_kl, u_ckl

def stream(f_ikl):
    """
    Carry out streaming step. This transports the particles towards the
    respective neighboring cells according to their velocities.

    Parameters
    ----------
    f_ikl: array
        Distribution of particle velocity on the 2D grid. Note that this array
        is modified in place.
    """
    for i in range(1, 9):
        f_ikl[i] = np.roll(f_ikl[i], c_ic[i], axis=(0, 1))

### Initialize probability distribution with a shear wave

x_k = np.arange(nx)
wavevector = 2*np.pi/nx
uy_k = np.sin(wavevector*x_k, dtype=dtype)
u_ck = np.array([np.zeros_like(uy_k), uy_k], dtype=dtype)

f_ikl = equilibrium(np.ones((nx, ny), dtype=dtype), u_ck.reshape((2, nx, 1)))

### Main loop

steps = np.arange(nsteps)
ampl = []

for i in steps:
    stream(f_ikl)
    rho_kl, u_ckl = collide(f_ikl, omega)
    # Fourier analysis
    ampl += [(u_ckl[1, :, ny//2]*uy_k).sum() * 2/nx]

### Write results

np.savetxt('amplitudes_opt0.out', ampl)
