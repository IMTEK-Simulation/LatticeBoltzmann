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
| OPT2 |
--------

This is an implementation of the D2Q9 Lattice Boltzmann lattice in the simple
relaxation time approximation. The code reports the amplitude of a decaying
shear wave which can be used to measure viscosity.

The present implementation contains was optimized with respect to opt1 in that
the collision operation was implemented in C++. The C++ implementation can be
found in shear_wave_opt2_cext.cpp.
"""

import numpy as np

import PyLB as D2Q9

### Parameters

# Lattice
nx = 300
ny = 300

# Number of steps
nsteps = 1000

# Relaxation parameter
omega = 0.3

# Data type
dtype = np.float64

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

### Compute functions

def stream(f_ikl):
    for i in range(1, 9):
        f_ikl[i] = np.roll(f_ikl[i], c_ic[i], axis=(0, 1))

### Initialize probability distribution with a shear wave

x_k = np.arange(nx)
wavevector = 2*np.pi/nx
uy_k = np.sin(wavevector*x_k, dtype=dtype)

#f_ikl = np.zeros((9, nx, ny), dtype=float)
f_ikl = np.arange(9*nx*ny, dtype=dtype).reshape(9, nx, ny)
D2Q9.equilibrium(np.ones((nx, ny), dtype=dtype).reshape(-1),
                 np.zeros((nx, ny), dtype=dtype).reshape(-1),
                 np.resize(uy_k, (nx, ny)).T.reshape(-1),
                 f_ikl.reshape(9, -1))

### Main loop

steps = np.arange(nsteps)
ampl = []

for i in steps:
    stream(f_ikl)
    D2Q9.collide(f_ikl.reshape(9, -1), omega)
    # Fourier analysis
    ampl += [((c_ic[:, 1].dot(f_ikl[:, :, ny//2])/(f_ikl[:, :, ny//2].sum(axis=0)))*uy_k).sum() * 2/nx]

### Write results

np.savetxt('amplitudes_opt2.out', ampl)