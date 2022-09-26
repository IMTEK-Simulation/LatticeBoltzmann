'''
Test- and Playground for the mpi implementation
    -> i have to content with 2 cores thou should just get a general handle for the
    processes involved, as an alternative i could us my own pc as a testbed (6 cores)
    -> should get a general feel for all the precesses on here
        -> most important send recive
        -> create the cartisian thingi

'''

import unittest
from mpi4py import MPI
import ipyparallel as ipp
import multiprocessing
import os
import psutil

# basic core count
# print(multiprocessing.cpu_count()) # actually returns the number of threads lol
# print(os.cpu_count())
# print(psutil.cpu_count(logical=False))
# only psutil how i want it to lol
cores = psutil.cpu_count(logical=False)

# basic example
def mpi_example():
    comm = MPI.COMM_WORLD
    return f"Hello World from rank {comm.Get_rank()}. total ranks={comm.Get_size()}"

# try out the monte carlo stuff
# basically stole out of the lecture and modified it a bit lol
def monte_carlo():
    import numpy as np
    ###
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    ###
    ntotal = 100000
    nlocal = ntotal//size
    print(nlocal)
    if rank == 0:
        # for all the cases when it can not be divided equally
        nlocal = ntotal - nlocal*(size-1)
        #
    x = np.random.random(nlocal)
    y = np.random.random(nlocal)

    nsucess = (x**2 + y**2 < 1).sum()

    reduced_nsucess = np.array(0)
    comm.Reduce(nsucess,reduced_nsucess, op= MPI.SUM, root = 0)
    return f"{4*reduced_nsucess/ntotal}"

# exchange data
def data_exchanger():
    # for some reason i have to do this
    import numpy as np
    import time
    ###
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    ###
    time.sleep(rank) # sleps 0 or 1 s
    ###
    return_value = np.arange(0,10)
    if rank == 0:
        return_value = return_value*3
        # send the value to the next rank
        comm.Send(return_value,dest=1,tag=99)
    elif rank == 1:
        values = np.arange(0,10)
        comm.Recv(values,source=0,tag=99)
        return_value = values

    return f"{return_value}"

def basic_sendrecv_snippet(f_ikl):
    # odd variables
    comm = MPI.COMM_WORLD
    ###
    left_dst = 1
    left_src = 1 # ranks of the processes in the comminicator
    ##
    # actual code
    recvbuf = f_ikl[:,-1,:].copy()
    # comm.Sendrecv(buffer send, destination
    #               buffer recev, source
    # f_ikl[:,1,:].copy() Zelle links wird gesendet
    comm.Sendrecv(f_ikl[:,1,:].copy(),left_dst,
                  recvbuf= recvbuf,source=left_src)
    f_ikl[:,-1,:] = recvbuf

def communicator_with_2cpus():
    # size just in x so its easier to actually part into 2
    import numpy as np
    nx = 10
    #
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    local_nx = nx//size
    #
    local_nx = local_nx+1
    return_array = np.zeros((local_nx,2))

    if rank == 0:
        return_array[-2,:] = 5 # send second last
        # now send the content of the last to the next
        recvbuf = return_array[-1,:]
        comm.Sendrecv(return_array[-2].copy(),1,recvbuf=recvbuf)
        # comm.Send(return_array[-2].copy(), dest=1, tag = 99)
        # comm.Recv(recvbuf,source=1)
        return_array[-1] = recvbuf # recive in the ghost zell
        #
    elif rank == 1:
        return_array[1,:] = 23
        recvbuf = return_array[0,:]
        comm.Sendrecv(return_array[1].copy(),0,recvbuf = recvbuf)
        # comm.Send(return_array[1], dest=0)
        # comm.Recv(recvbuf, source=0, tag = 99)
        return_array[0] = recvbuf
    #####
    return f"{return_array}"

def shift_stuff():
    # size stuff
    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    comm = MPI.COMM_WORLD.Create_cart((1,2), periods=(False,False))
    return_value = 0
    #
    if rank == 0:
        return_value = comm.Shift(1,1) # right/top
        # second number is the destination first is yiberish
    if rank == 1:
        return_value = comm.Shift(1,-1) # left/bottom

    ###
    return f"{return_value}"


def test_colapse_data_with_2cores():
    import simulators.simple_flows.slidingLidMPI as slidingLidMPI
    import numpy as np
    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    steps = 10000
    re = 1000
    base_lenght = 300
    rank_in_one_direction = 1  # for an MPI thingi with 9 processes -> 3x3 field
    uw = 0.1
    relaxation = (2 * re) / (6 * base_lenght * uw + re)
    comm = MPI.COMM_WORLD
    process_info = slidingLidMPI.fill_mpi_struct_fields(rank, size, 2, 1, base_lenght, relaxation, steps, uw)
    if rank == 1:
        process_info.boundaries_info.apply_bottom = True
        process_info.boundaries_info.apply_top = True
        process_info.boundaries_info.apply_right = True
        process_info.boundaries_info.apply_left = False

    # initizlize the gird
    rho = np.ones((process_info.size_x, process_info.size_y))
    ux = np.zeros((process_info.size_x, process_info.size_y))
    uy = np.zeros((process_info.size_x, process_info.size_y))
    grid = slidingLidMPI.equilibrium(rho, ux, uy)
    full_grid = np.zeros(9)
    if process_info.rank ==0:
        original_x = process_info.size_x - 2  # ie the base size of the grid on that the
        original_y = process_info.size_y - 2  # calculation ran
        temp = np.zeros((9, original_x, original_y))
        comm.Recv(temp, source=1)
        full_grid = np.concatenate((grid[:,1:-1,1:-1].copy(),temp),axis = 2)
    if process_info.rank == 1:
        comm.Send(grid[:,1:-1,1:-1].copy(),dest=0)
    return f"{full_grid.shape}"

def test_colapse_data():
    import simulators.simple_flows.slidingLidMPI as slidingLidMPI
    import numpy as np
    comm = MPI.COMM_WORLD
    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    steps = 1000
    re = 1000
    base_lenght = 40
    uw = 0.1
    relaxation = (2 * re) / (6 * base_lenght * uw + re)
    process_info = slidingLidMPI.fill_mpi_struct_fields(rank, size, 2, 2, base_lenght, relaxation, steps, uw)
    # disable communication
    process_info.boundaries_info.apply_bottom = True
    process_info.boundaries_info.apply_top = True
    process_info.boundaries_info.apply_right = True
    process_info.boundaries_info.apply_left = True
    # run sim
    slidingLidMPI.sliding_lid_mpi(process_info, comm)
    return f"{process_info}"

def test_data_creation():
    import simulators.simple_flows.slidingLidMPI as slidingLidMPI
    import numpy as np
    comm = MPI.COMM_WORLD
    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    steps = 1000
    re = 1000
    base_lenght = 40
    uw = 0.1
    relaxation = (2 * re) / (6 * base_lenght * uw + re)
    process_info = slidingLidMPI.fill_mpi_struct_fields(rank, size, 2, 2, base_lenght, relaxation, steps, uw)
    return f"{process_info}"

def test_4_simulation():
    import simulators.simple_flows.slidingLidMPI as slidingLidMPI
    import numpy as np
    comm = MPI.COMM_WORLD
    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    steps = 100000
    re = 1000
    base_lenght = 100
    uw = 0.1
    relaxation = (2 * re) / (6 * base_lenght * uw + re)
    process_info = slidingLidMPI.fill_mpi_struct_fields(rank, size, 2, 2, base_lenght, relaxation, steps, uw)
    # process_info.boundaries_info.apply_left = True
    # process_info.boundaries_info.apply_right = True
    # process_info.boundaries_info.apply_top = True
    # process_info.boundaries_info.apply_bottom = True
    slidingLidMPI.sliding_lid_mpi(process_info, comm)
    return f"{process_info}"

# Main caller
# request an MPI cluster with 2 engines
with ipp.Cluster(engines='mpi', n=cores
                 ) as rc:
    # get a broadcast_view on the cluster which is best
    # print(rc.ids)
    # suited for MPI style computation
    view = rc.broadcast_view()
    # run the mpi_example function on all engines in parallel
    r = view.apply_sync(test_4_simulation)
    # Retrieve and print the result from the engines
    print("\n".join(r))
# at this point, the cluster processes have been shutdow
