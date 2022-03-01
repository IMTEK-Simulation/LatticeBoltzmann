# imports
import numpy as np
from dataclasses import dataclass
from mpi4py import MPI
import matplotlib.pyplot as plt
import ipyparallel as ipp
import psutil
import slidingLidMPI
# initial variables and sizess
re = 1000
base_lenght = 300
rank_in_one_direction = 0 # for an MPI thingi with 9 processes -> 3x3 field
steps = 20
uw = 0.1
size_x = base_lenght
size_y = base_lenght
relaxation = (2*re)/(6*base_lenght*uw+re)
velocity_set = np.array([[0, 1, 0, -1, 0, 1, -1, -1, 1],
                         [0,0,1,0,-1,1,1,-1,-1]]).T
cores = psutil.cpu_count(logical= False)



# call
# sliding_lid_mpi()
def indiviaual_clall():
    import slidingLidMPI
    base_lenght = 300
    steps = 1
    re = 1000
    uw = 0.1
    relaxation = (2 * re) / (6 * base_lenght * uw + re)
    rank_in_one_direction = 2  # for an MPI thingi with 9 processes -> 3x3 field max = 3
    # 4 cores 2x2 u put 2 in there
    ###
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    process_info = slidingLidMPI.fill_mpi_struct_fields(comm.Get_rank(),comm.Get_size(), rank_in_one_direction
                                                        ,rank_in_one_direction,base_lenght,
                                                       relaxation,steps,uw)
    #
    process_info.boundaries_info.apply_right = True
    process_info.boundaries_info.apply_left = True
    process_info.boundaries_info.apply_top = True
    process_info.boundaries_info.apply_bottom = True
    if rank == 0:
        process_info.boundaries_info.apply_right = False
    if rank == 1:
        process_info.boundaries_info.apply_left = False
    # slidingLidMPI.sliding_lid_mpi(process_info,comm)
    return f"{process_info}"

#multiple caller
with ipp.Cluster(engines= 'mpi', n = cores) as rc:
    view = rc.broadcast_view()
    r = view.apply_sync(indiviaual_clall)
    print("\n".join(r))