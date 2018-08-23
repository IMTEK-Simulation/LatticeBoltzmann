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
relaxation time approximation. The writes the velocity field to a series of
files in the npy format.

The present implementation contains no optimizations but was developed to
yield a concise code. Storage order was chosen such that the indices of
locations on the 2D real-space grid are the fast indices.
"""

import sys

from enum import IntEnum

from mpi4py import MPI

import numpy as np

### Parameters

# Decomposition
ndx = int(sys.argv[1])
ndy = int(sys.argv[2])

# Lattice
nx = int(sys.argv[3])
ny = int(sys.argv[4])

# Data type
dtype = np.dtype(sys.argv[5])

# Number of steps
nsteps = 100000

# Dump velocities every this many steps
dump_freq = 10000

# Relaxation parameter
omega = np.array(1.7, dtype=dtype)

### Direction shorthands

D = IntEnum('D', 'E N W S NE NW SW SE')

### Auxiliary arrays

# Naming conventions for arrays: Subscript indicates type of dimension
# Example: c_ic <- 2-dimensional array
#            ^^
#            ||--- c: Cartesion index, can have value 0 or 1
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
    cu_ikl = np.dot(u_ckl.T, c_ic.T.astype(u_ckl.dtype)).T
    uu_kl = np.sum(u_ckl**2, axis=0)
    return (w_i*(rho_kl*(1 + 3*cu_ikl + 9/2*cu_ikl**2 - 3/2*uu_kl)).T).T

def collide(f_ikl, omega):
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
    u_ckl = np.dot(f_ikl.T, c_ic.astype(f_ikl.dtype)).T/rho_kl
    f_ikl += omega*(equilibrium(rho_kl, u_ckl) - f_ikl)
    return rho_kl, u_ckl

def stream(f_ikl):
    """
    Propagate channel occupations by one cell distance.

    Parameters
    ----------
    f_ikl : array
        Array containing the occupation numbers. Array is 3-dimensional, with
        the first dimension running from 0 to 8 and indicating channel. The
        next two dimensions are x- and y-position. This array is modified in
        place.
    """
    for i in range(1, 9):
        f_ikl[i] = np.roll(f_ikl[i], c_ic[i], axis=(0, 1))

def stream_and_bounce_back(f_ikl, u0=0.1):
    """
    Propagate channel occupations by one cell distance and apply no-slip 
    boundary condition to all four walls. The top wall can additionally have
    a velocity.

    Parameters
    ----------
    f_ikl : array
        Array containing the occupation numbers. Array is 3-dimensional, with
        the first dimension running from 0 to 8 and indicating channel. The
        next two dimensions are x- and y-position. This array is modified in
        place.
    u0 : float
        Velocity of the top wall.
    """
    fbottom_ik = f_ikl[:, :, 0].copy()
    ftop_ik = f_ikl[:, :, -1].copy()

    fleft_il = f_ikl[:, 0, :].copy()
    fright_il = f_ikl[:, -1, :].copy()

    stream(f_ikl)

    # Bottom boundary
    f_ikl[D.N, :, 0] = fbottom_ik[D.S]
    f_ikl[D.NE, :, 0] = fbottom_ik[D.SW]
    f_ikl[D.NW, :, 0] = fbottom_ik[D.SE]

    # Top boundary - sliding lid
    # We need to compute the rho *after* bounce back
    rhobottom_k = ftop_ik[D.NW] + ftop_ik[D.N] + ftop_ik[D.NE] + \
        f_ikl[D.NW, :, -1] + f_ikl[D.N, :, -1] + f_ikl[D.NE, :, -1] + \
        f_ikl[D.W, :, -1] + f_ikl[0, :, -1] + f_ikl[D.E, :, -1]
    f_ikl[D.S, :, -1] = ftop_ik[D.N]
    f_ikl[D.SE, :, -1] = ftop_ik[D.NW] + 6*w_i[D.SE]*rhobottom_k*u0
    f_ikl[D.SW, :, -1] = ftop_ik[D.NE] - 6*w_i[D.SW]*rhobottom_k*u0

    # Change to "if False" for Couette flow test
    if True:
        # Left boundary
        f_ikl[D.E, 0, :] = fleft_il[D.W]
        f_ikl[D.NE, 0, :] = fleft_il[D.SW]
        f_ikl[D.SE, 0, :] = fleft_il[D.NW]

        # Right boundary
        f_ikl[D.W, -1, :] = fright_il[D.E]
        f_ikl[D.NW, -1, :] = fright_il[D.SE]
        f_ikl[D.SW, -1, :] = fright_il[D.NE]

        # Bottom-left corner
        f_ikl[D.N, 0, 0] = fbottom_ik[D.S, 0]
        f_ikl[D.E, 0, 0] = fbottom_ik[D.W, 0]
        f_ikl[D.NE, 0, 0] = fbottom_ik[D.SW, 0]

        # Bottom-right corner
        f_ikl[D.N, -1, 0] = fbottom_ik[D.S, -1]
        f_ikl[D.W, -1, 0] = fbottom_ik[D.E, -1]
        f_ikl[D.NW, -1, 0] = fbottom_ik[D.SE, -1]

        # Top-left corner
        f_ikl[D.S, 0, -1] = ftop_ik[D.N, 0]
        f_ikl[D.E, 0, -1] = ftop_ik[D.W, 0]
        f_ikl[D.SE, 0, -1] = ftop_ik[D.NW, 0]

        # Top-right corner
        f_ikl[D.S, -1, -1] = ftop_ik[D.N, -1]
        f_ikl[D.W, -1, -1] = ftop_ik[D.E, -1]
        f_ikl[D.SW, -1, -1] = ftop_ik[D.NE, -1]

def communicate(f_ikl):
    """
    Communicate boundary regions to ghost regions.

    Parameters
    ----------
    f_ikl : array
        Array containing the occupation numbers. Array is 3-dimensional, with
        the first dimension running from 0 to 8 and indicating channel. The
        next two dimensions are x- and y-position. This array is modified in
        place.
    """
    # Send to left
    recvbuf = f_ikl[:, -1, :].copy()
    comm.Sendrecv(f_ikl[:, 1, :].copy(), left_dst,
                  recvbuf=recvbuf, source=left_src)
    f_ikl[:, -1, :] = recvbuf
    # Send to right
    recvbuf = f_ikl[:, 0, :].copy()
    comm.Sendrecv(f_ikl[:, -2, :].copy(), right_dst,
                  recvbuf=recvbuf, source=right_src)
    f_ikl[:, 0, :] = recvbuf
    # Send to bottom
    recvbuf = f_ikl[:, :, -1].copy()
    comm.Sendrecv(f_ikl[:, :, 1].copy(), bottom_dst,
                  recvbuf=recvbuf, source=bottom_src)
    f_ikl[:, :, -1] = recvbuf
    # Send to top
    recvbuf = f_ikl[:, :, 0].copy()
    comm.Sendrecv(f_ikl[:, :, -2].copy(), top_dst,
                  recvbuf=recvbuf, source=top_src)
    f_ikl[:, :, 0] = recvbuf

### I/O functions

def save_mpiio(comm, fn, g_kl):
    """
    Write a global two-dimensional array to a single file in the npy format
    using MPI I/O: https://docs.scipy.org/doc/numpy/neps/npy-format.html

    Arrays written with this function can be read with numpy.load.

    Parameters
    ----------
    comm
        MPI communicator.
    fn : str
        File name.
    g_kl : array
        Portion of the array on this MPI processes.
    """
    from numpy.lib.format import dtype_to_descr, magic
    magic_str = magic(1, 0)

    local_nx, local_ny = g_kl.shape
    nx = np.empty_like(local_nx)
    ny = np.empty_like(local_ny)

    commx = comm.Sub((True, False))
    commy = comm.Sub((False, True))
    commx.Allreduce(np.asarray(local_nx), nx)
    commy.Allreduce(np.asarray(local_ny), ny)

    arr_dict_str = str({ 'descr': dtype_to_descr(g_kl.dtype),
                         'fortran_order': False,
                         'shape': (np.asscalar(nx), np.asscalar(ny)) })
    while (len(arr_dict_str) + len(magic_str) + 2) % 16 != 15:
        arr_dict_str += ' '
    arr_dict_str += '\n'
    header_len = len(arr_dict_str) + len(magic_str) + 2

    offsetx = np.zeros_like(local_nx)
    commx.Exscan(np.asarray(ny*local_nx), offsetx)
    offsety = np.zeros_like(local_ny)
    commy.Exscan(np.asarray(local_ny), offsety)

    file = MPI.File.Open(comm, fn, MPI.MODE_CREATE | MPI.MODE_WRONLY)
    if rank == 0:
        file.Write(magic_str)
        file.Write(np.int16(len(arr_dict_str)))
        file.Write(arr_dict_str.encode('latin-1'))
    mpitype = MPI._typedict[g_kl.dtype.char]
    filetype = mpitype.Create_vector(g_kl.shape[0], g_kl.shape[1], ny)
    filetype.Commit()
    file.Set_view(header_len + (offsety+offsetx)*mpitype.Get_size(),
                  filetype=filetype)
    file.Write_all(g_kl.copy())
    filetype.Free()
    file.Close()

### Initialize MPI communicator

size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
if rank == 0:
    print('Running in parallel on {} MPI processes.'.format(size))
assert ndx*ndy == size
if rank == 0:
    print('Domain decomposition: {} x {} MPI processes.'.format(ndx, ndy))
    print('Global grid has size {}x{}.'.format(nx, ny))
    print('Using {} floating point data type.'.format(dtype))

# Create cartesian communicator and get MPI ranks of neighboring cells
comm = MPI.COMM_WORLD.Create_cart((ndx, ndy), periods=(False, False))
left_src, left_dst = comm.Shift(0, -1)
right_src, right_dst = comm.Shift(0, 1)
bottom_src, bottom_dst = comm.Shift(1, -1)
top_src, top_dst = comm.Shift(1, 1)

local_nx = nx//ndx
local_ny = ny//ndy

# We need to take care that the total number of *local* grid points sums up to
# nx. The right and topmost MPI processes are adjusted such that this is
# fulfilled even if nx, ny is not divisible by the number of MPI processes.
if right_dst < 0:
    # This is the rightmost MPI process
    local_nx = nx - local_nx*(ndx-1)
without_ghosts_x = slice(0, local_nx)
if right_dst >= 0:
    # Add ghost cell
    local_nx += 1
if left_dst >= 0:
    # Add ghost cell
    local_nx += 1
    without_ghosts_x = slice(1, local_nx+1)
if top_dst < 0:
    # This is the topmost MPI process
    local_ny = ny - local_ny*(ndy-1)
without_ghosts_y = slice(0, local_ny)
if top_dst >= 0:
    # Add ghost cell
    local_ny += 1
if bottom_dst >= 0:
    # Add ghost cell
    local_ny += 1
    without_ghosts_y = slice(1, local_ny+1)

mpix, mpiy = comm.Get_coords(rank)
print('Rank {} has domain coordinates {}x{} and a local grid of size {}x{} (including ghost cells).'.format(rank, mpix, mpiy, local_nx, local_ny))

### Initialize occupation numbers

f_ikl = equilibrium(np.ones((local_nx, local_ny), dtype=dtype),
                    np.zeros((2, local_nx, local_ny), dtype=dtype))

### Main loop

for i in range(nsteps):
    if i % 10 == 9:
        sys.stdout.write('=== Step {}/{} ===\r'.format(i+1, nsteps))
    communicate(f_ikl)
    stream_and_bounce_back(f_ikl)
    rho_kl, u_ckl = collide(f_ikl, omega)

    if i % dump_freq == 0:
        save_mpiio(comm, 'ux_{}.npy'.format(i), u_ckl[0, without_ghosts_x, without_ghosts_y])
        save_mpiio(comm, 'uy_{}.npy'.format(i), u_ckl[1, without_ghosts_x, without_ghosts_y])

### Dump final stage of the simulation to a file

save_mpiio(comm, 'ux_{}.npy'.format(i), u_ckl[0, without_ghosts_x, without_ghosts_y])
save_mpiio(comm, 'uy_{}.npy'.format(i), u_ckl[1, without_ghosts_x, without_ghosts_y])
