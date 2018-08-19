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
| OPT1 |
--------

This is an implementation of the D2Q9 Lattice Boltzmann lattice in the simple
relaxation time approximation. The code reports the amplitude of a decaying
shear wave which can be used to measure viscosity.

The present implementation contains was optimized with respect to opt0 to
eliminate all multiplications with zero (channel velocity) in the collision
step. This requires to explicitly write out some multiplication from opt0. The
storage of the velocity field is split into the two Cartesian components ux_kl
and uy_kl now rather than having a single array u_ckl with Cartesian index as
the first dimension.
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
w_0 = 4/9
w_1234 = 1/9
w_5678 = 1/36

### Compute functions

def equilibrium(rho_kl, ux_kl, uy_kl):
    """
    Return the equilibrium distribution function.

    Parameters
    ----------
    rho_rl: array
        Fluid density on the 2D grid.
    ux_kl: array
        x-components of streaming velocity on the 2D grid.
    uy_kl: array
        y-components of streaming velocity on the 2D grid.

    Returns
    -------
    f_ikl: array
        Equilibrium distribution for the given fluid density *rho_kl* and
        streaming velocity *ux_kl*, *uy_kl*.
    """
    cu5_kl = ux_kl + uy_kl
    cu6_kl = -ux_kl + uy_kl
    cu7_kl = -ux_kl - uy_kl
    cu8_kl = ux_kl - uy_kl
    uu_kl = ux_kl**2 + uy_kl**2
    return np.array([w_0*rho_kl*(1 - 3/2*uu_kl),
                     w_1234*rho_kl*(1 + 3*ux_kl + 9/2*ux_kl**2 - 3/2*uu_kl),
                     w_1234*rho_kl*(1 + 3*uy_kl + 9/2*uy_kl**2 - 3/2*uu_kl),
                     w_1234*rho_kl*(1 - 3*ux_kl + 9/2*ux_kl**2 - 3/2*uu_kl),
                     w_1234*rho_kl*(1 - 3*uy_kl + 9/2*uy_kl**2 - 3/2*uu_kl),
                     w_5678*rho_kl*(1 + 3*cu5_kl + 9/2*cu5_kl**2 - 3/2*uu_kl),
                     w_5678*rho_kl*(1 + 3*cu6_kl + 9/2*cu6_kl**2 - 3/2*uu_kl),
                     w_5678*rho_kl*(1 + 3*cu7_kl + 9/2*cu7_kl**2 - 3/2*uu_kl),
                     w_5678*rho_kl*(1 + 3*cu8_kl + 9/2*cu8_kl**2 - 3/2*uu_kl)])

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
    ux_kl: array
        x-components of current streaming velocity on the 2D grid.
    uy_kl: array
        y-components of current streaming velocity on the 2D grid.
    """
    rho_kl = np.sum(f_ikl, axis=0)
    ux_kl = (f_ikl[1] - f_ikl[3] + f_ikl[5] - f_ikl[6] - f_ikl[7] + f_ikl[8])/rho_kl
    uy_kl = (f_ikl[2] - f_ikl[4] + f_ikl[5] + f_ikl[6] - f_ikl[7] - f_ikl[8])/rho_kl
    f_ikl += omega*(equilibrium(rho_kl, ux_kl, uy_kl) - f_ikl)
    return rho_kl, ux_kl, uy_kl

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
uy_k = np.sin(wavevector*x_k)

f_ikl = equilibrium(np.ones((nx, ny)),
                    np.zeros_like(uy_k).reshape((nx, 1)),
                    uy_k.reshape((nx, 1)))

### Main loop

steps = np.arange(nsteps)
ampl = []

for i in steps:
    stream(f_ikl)
    rho_kl, ux_kl, uy_kl = collide(f_ikl, omega)
    # Fourier analysis
    ampl += [(uy_kl[:, ny//2]*uy_k).sum() * 2/nx]

### Write results

np.savetxt('amplitudes_opt1.out', ampl)
