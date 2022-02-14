'''
Test- and Playground for the mpi implementation
    -> i have to content with 2 cores thou should just get a general handle for the
    processes involved, as an alternative i could us my own pc as a testbed (6 cores)
    -> should get a general feel for all the precesses on here
        -> most important send recive
        -> create the cartisian thingi

'''

import unittest
import numpy as np
from mpi4py import MPI
import ipyparallel as ipp

# basic core count
cores = 2

def mpi_example():
    comm = MPI.COMM_WORLD
    return f"Hello World from rank {comm.Get_rank()}. total ranks={comm.Get_size()}"

# request an MPI cluster with 2 engines
with ipp.Cluster(engines='mpi', n=cores
                 ) as rc:
    # get a broadcast_view on the cluster which is best
    # suited for MPI style computation
    view = rc.broadcast_view()
    # run the mpi_example function on all engines in parallel
    r = view.apply_sync(mpi_example)
    # Retrieve and print the result from the engines
    print("\n".join(r))
# at this point, the cluster processes have been shutdow
