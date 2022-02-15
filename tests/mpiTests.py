'''
Test- and Playground for the mpi implementation
    -> i have to content with 2 cores thou should just get a general handle for the
    processes involved, as an alternative i could us my own pc as a testbed (6 cores)
    -> should get a general feel for all the precesses on here
        -> most important send recive
        -> create the cartisian thingi

'''
# TODO remove me !!
'''
 -> monte carlo implementation get the first feel ;<;
 -> send and recive random stuff
    -> maybe do some unittests here not sure thou ;<;
 -> try out domain decomposition i guess (sendrecv lol)
 -> test the seperation into top left usw.
 -> will have to run those tests on my own pc as i have more than 4 processors
 -> dont forget how to install!! sudo apt install mpi4py
 https://zoomadmin.com/HowToInstall/UbuntuPackage/python3-mpi4py
 use cart_comm
 use cartcomm.shift 
 just use the code from the slides lol twoards the end
'''
import unittest
from mpi4py import MPI
import ipyparallel as ipp
import multiprocessing
import os
import psutil

# basic core count
print(multiprocessing.cpu_count()) # actually returns the number of threads lol
print(os.cpu_count())
print(psutil.cpu_count(logical=False))
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
    ntotal = 1000000
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
        comm.Sendrecv(return_array[-2].copy(),1,recvbuf=recvbuf,source=1)
        # comm.Send(return_array[-2].copy(), dest=1, tag = 99)
        # comm.Recv(recvbuf,source=1)
        return_array[-1] = recvbuf # recive in the ghost zell
        #
    elif rank == 1:
        return_array[1,:] = 23
        recvbuf = return_array[0,:]
        comm.Sendrecv(return_array[1].copy(),0,recvbuf = recvbuf,source=0)
        # comm.Send(return_array[1], dest=0)
        # comm.Recv(recvbuf, source=0, tag = 99)
        return_array[0] = recvbuf
    #####
    return f"{return_array}"

def shift_stuff():
    # size stuff
    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    comm = MPI.COMM_WORLD.Create_cart((2,1), periods=(False,False))
    return_value = 0
    #
    if rank == 0:
        return_value = comm.Shift(0,1) # right
        # second number is the destination first is yiberish
    if rank == 1:
        return_value = comm.Shift(0,-1) # left

    ###
    return f"{return_value}"

# Main caller
# request an MPI cluster with 2 engines
with ipp.Cluster(engines='mpi', n=cores
                 ) as rc:
    # get a broadcast_view on the cluster which is best
    # print(rc.ids)
    # suited for MPI style computation
    view = rc.broadcast_view()
    # run the mpi_example function on all engines in parallel
    r = view.apply_sync(shift_stuff)
    # Retrieve and print the result from the engines
    print("\n".join(r))
# at this point, the cluster processes have been shutdow
