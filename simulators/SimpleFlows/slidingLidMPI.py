'''
Remider CodingRules:
Zeilenumbruch bei Spalte 120
Modulname, Klassennamen als CamelCase
Variablennamen, Methodennamen, Funktionsnamen mit unter_strichen
Bitte nicht CamelCase und Unterstriche mischen
'''

'''
I have made the decission to not include anything form the tests 
or from the original code itself.
This module should be able to work on its own, but it will be with basically no explanation in the code itself
for this look at the simpleFlowsTest.
'''

# imports
import numpy as np
from dataclasses import dataclass
from mpi4py import MPI
import matplotlib.pyplot as plt
import ipyparallel as ipp
import psutil
import time

# only vars
velocity_set = np.array([[0, 1, 0, -1, 0, 1, -1, -1, 1],
                         [0,0,1,0,-1,1,1,-1,-1]]).T
cores = psutil.cpu_count(logical= False)

# pack stuff so its together
@dataclass
class boundariesApplied:
    # left right top bottom
    apply_left : bool = False
    apply_right: bool = False
    apply_top:bool = False
    apply_bottom: bool = False

@dataclass
class cellNeighbors:
    left: int = -1
    right: int = -1
    top: int = -1
    bottom: int = -1

@dataclass
class mpiPackageStructure:
    # apply for boundaries
    boundaries_info: boundariesApplied = (False,False,False,False)
    neighbors: cellNeighbors = (-1, -1, -1, -1)
    # sizes and position in the whole grid
    size_x: int = -1
    size_y: int = -1
    pos_x : int = -1
    pos_y : int = -1
    # for MPI stuff
    rank : int = -1
    size : int = -1
    # overall
    relaxation : int = -1
    base_grid: int = -1
    steps : int = -1
    uw : int = -1

# set functions for the mpi Package Structure
def set_boundary_info(pox,poy,max_x,max_y):
    info = boundariesApplied(False,False,False,False)
    ##
    if pox == 0:
        info.apply_left = True
    if poy == 0:
        info.apply_bottom = True
    if pox == max_x:
        info.apply_right = True
    if poy == max_y:
        info.apply_top = True
    ##
    # print(info)
    return info

def get_postions_out_of_rank_size_quadratic(rank,size):
    ##
    # assume to be quadratic
    edge_lenght = int(np.sqrt(size))
    ###
    pox = rank % edge_lenght
    poy = rank//edge_lenght
    ###
    return pox,poy


def fill_mpi_struct_fields(rank,size,max_x,max_y,base_grid,relaxation,steps,uw):
    '''

    Parameters
    ----------
    rank
    size
    max_x starts from 0
    max_y starts from 0
    base_grid

    Returns
    -------

    '''
    #
    info = mpiPackageStructure()
    info.rank = rank
    info.size = size
    info.pos_x,info.pos_y = get_postions_out_of_rank_size_quadratic(rank,size)
    info.boundaries_info = set_boundary_info(info.pos_x,info.pos_y,max_x-1,max_y-1) # i should know my own code lol
    info.size_x = base_grid//(max_x) + 2
    info.size_y = base_grid //(max_y) + 2
    info.neighbors = determin_neighbors(rank,size)
    #
    info.relaxation = relaxation
    info.base_grid = base_grid
    info.steps = steps
    info.uw = uw
    return info

def determin_neighbors(rank,size):
    # determin edge lenght
    edge_lenght = int(np.sqrt(size))
    ###
    neighbor = cellNeighbors()
    neighbor.top = rank + edge_lenght
    neighbor.bottom = rank - edge_lenght
    neighbor.right = rank + 1
    neighbor.left = rank-1
    ###
    return neighbor

# main methods
def stream(grid):
    for i in range(1,9):
        grid[i] = np.roll(grid[i],velocity_set[i], axis = (0,1))

def equilibrium(rho,ux,uy):
    uxy_3plus = 3 * (ux + uy)
    uxy_3miuns = 3 * (ux - uy)
    uu = 3 * (ux * ux + uy * uy)
    ux_6 = 6 * ux
    uy_6 = 6 * uy
    uxx_9 = 9 * ux * ux
    uyy_9 = 9 * uy * uy
    uxy_9 = 9 * ux * uy
    return np.array([(2 * rho / 9) * (2 - uu),
                     (rho / 18) * (2 + ux_6 + uxx_9 - uu),
                     (rho / 18) * (2 + uy_6 + uyy_9 - uu),
                     (rho / 18) * (2 - ux_6 + uxx_9 - uu),
                     (rho / 18) * (2 - uy_6 + uyy_9 - uu),
                     (rho / 36) * (1 + uxy_3plus + uxy_9 + uu),
                     (rho / 36) * (1 - uxy_3miuns - uxy_9 + uu),
                     (rho / 36) * (1 - uxy_3plus + uxy_9 + uu),
                     (rho / 36) * (1 + uxy_3miuns - uxy_9 + uu)])


def collision(grid,rho,ux,uy,relaxation):
    grid -= relaxation * (grid - equilibrium(rho, ux, uy))


def caluculate_rho_ux_uy(grid):
    rho = np.sum(grid, axis=0)  # sums over each one individually
    ux = ((grid[1] + grid[5] + grid[8]) - (grid[3] + grid[6] + grid[7])) / rho
    uy = ((grid[2] + grid[5] + grid[6]) - (grid[4] + grid[7] + grid[8])) / rho
    return rho,ux,uy


def bounce_back(grid,uw):
    #### Left + Right
    # right so x = 0
    grid[1, 1, 1:-1] = grid[3, 0, 1:-1]
    grid[5, 1, 1:-1] = grid[7, 0, 1:-1]
    grid[8, 1, 1:-1] = grid[6, 0, 1:-1]
    # left so x = -1
    grid[3, -2, 1:-1] = grid[1, -1, 1:-1]
    grid[6, -2, 1:-1] = grid[8, -1, 1:-1]
    grid[7, -2, 1:-1] = grid[5, -1, 1:-1]

    #### TOP + Bottom
    # for bottom y = 0
    grid[2, 1:-1, 1] = grid[4, 1:-1, 0]
    grid[5, 1:-1, 1] = grid[7, 1:-1, 0]
    grid[6, 1:-1, 1] = grid[8, 1:-1, 0]
    # for top y = -1
    grid[4, 1:-1, -2] = grid[2, 1:-1, -1]
    grid[7, 1:-1, -2] = grid[5, 1:-1, -1] - 1 / 6 * uw
    grid[8, 1:-1, -2] = grid[6, 1:-1, -1] + 1 / 6 * uw


def bounce_back_choosen(grid,uw,info):
    # modification for the bounce back for the bigger grids
    # Right + Left
    if info.boundaries_info.apply_right:
        # right so x = -1
        grid[3, -2, :] = grid[1, -1, :]
        grid[6, -2, :] = grid[8, -1, :]
        grid[7, -2, :] = grid[5, -1, :]
    if info.boundaries_info.apply_left:
        # left so x = 0
        grid[1, 1, :] = grid[3, 0, :]
        grid[5, 1, :] = grid[7, 0, :]
        grid[8, 1, :] = grid[6, 0, :]

    # Bottom + Top
    if info.boundaries_info.apply_bottom:
        # for bottom y = 0
        grid[2, :, 1] = grid[4, :, 0]
        grid[5, :, 1] = grid[7, :, 0]
        grid[6, :, 1] = grid[8, :, 0]
    if info.boundaries_info.apply_top:
        # for top y = -1
        grid[4, :, -2] = grid[2, :, -1]
        grid[7, :, -2] = grid[5, :, -1] - 1 / 6 * uw
        grid[8, :, -2] = grid[6, :, -1] + 1 / 6 * uw


def comunicate(grid,info,comm):
    # if they are false we have to comunicate otherwise will have to do the boundary stuff
    # Right + Left
    # Todo: i think they can still get messed up even with tags
    if not info.boundaries_info.apply_right:
        recvbuf = grid[:, -1, :].copy()
        comm.Sendrecv(grid[:,-2, :].copy(), info.neighbors.right, recvbuf=recvbuf, sendtag = 11, recvtag = 12)
        grid[:, -1, :] = recvbuf
    if not info.boundaries_info.apply_left:
        recvbuf = grid[:, 0, :].copy()
        comm.Sendrecv(grid[:, 1, :].copy(), info.neighbors.left, recvbuf=recvbuf, sendtag = 12, recvtag = 11)
        grid[:, 0, :] = recvbuf

    # Bottom + Top
    if not info.boundaries_info.apply_bottom:
        recvbuf = grid[:,: ,0 ].copy()
        comm.Sendrecv(grid[:, :,1 ].copy(), info.neighbors.bottom, recvbuf=recvbuf, sendtag = 99, recvtag = 98)
        grid[:, :, 0] = recvbuf
    if not info.boundaries_info.apply_top:
        recvbuf = grid[:, :, -1].copy()
        comm.Sendrecv(grid[:, :, -2].copy(), info.neighbors.top, recvbuf=recvbuf, sendtag = 98, recvtag = 99)
        grid[:, :, -1] = recvbuf


def collapse_data(process_info,grid,comm):
    full_grid = np.zeros(2)
    # process 0 gets the data and does the visualization
    if process_info.rank == 0:
        full_grid = np.ones((9, process_info.base_grid, process_info.base_grid))
        original_x = process_info.size_x-2 # ie the base size of the grid on that the
        original_y = process_info.size_y-2 # calculation ran
        # write the own stuff into it first
        full_grid[:,0:original_x,0:original_y] = grid[:,1:-1,1:-1]
        temp = np.zeros((9,original_x,original_y))
        for i in range(1,process_info.size):
            comm.Recv(temp,source = i,tag = i)
            x,y = get_postions_out_of_rank_size_quadratic(i,process_info.size)
            # determin start end endpoints to copy to in the grid
            copy_start_x = 0 + original_x*x
            copy_end_x = original_x + original_x*x
            copy_start_y = 0 + original_y*y
            copy_end_y = original_y + original_y*y
            # copy
            full_grid[:,copy_start_x:copy_end_x,copy_start_y:copy_end_y] = temp
    # all the others send to p0
    else:
        comm.Send(grid[:,1:-1,1:-1].copy(),dest=0, tag = process_info.rank)

    return full_grid

def sliding_lid_mpi(process_info,comm):
    #create grid based on process Info
    rho = np.ones((process_info.size_x, process_info.size_y))
    ux = np.zeros((process_info.size_x, process_info.size_y))
    uy = np.zeros((process_info.size_x, process_info.size_y))
    grid = equilibrium(rho, ux, uy)
    # loop
    for i in range(process_info.steps):
        stream(grid)
        bounce_back_choosen(grid, process_info.uw, process_info)
        rho, ux, uy = caluculate_rho_ux_uy(grid)
        collision(grid, rho, ux, uy, process_info.relaxation)
        comunicate(grid,process_info,comm)

    # get full grid + plot
    full_grid = collapse_data(process_info, grid, comm)
    plotter(full_grid,process_info)

def plotter(full_grid,process_info):
    #plot
    if process_info.rank == 0:
        print("Making Image")
        # recalculate ux and uy
        idk,full_ux,full_uy = caluculate_rho_ux_uy(full_grid)
        # acutal plot

        x = np.arange(0, process_info.base_grid)
        y = np.arange(0, process_info.base_grid)
        X, Y = np.meshgrid(x, y)
        speed = np.sqrt(full_ux.T ** 2 + full_uy.T ** 2)
        # plot
        # plt.streamplot(X,Y,full_ux.T,full_uy.T)
        plt.streamplot(X, Y, full_ux.T, full_uy.T, color=speed, cmap=plt.cm.jet)
        ax = plt.gca()
        ax.set_xlim([0, process_info.base_grid + 1])
        ax.set_ylim([0, process_info.base_grid + 1])
        plt.title("Sliding Lid")
        plt.xlabel("x-Position")
        plt.ylabel("y-Position")
        fig = plt.colorbar()
        fig.set_label("Velocity u(x,y,t)", rotation=270, labelpad=15)
        savestring = "slidingLidmpi"+str(process_info.size)+".png"
        plt.savefig(savestring)
        plt.show()


def call():
    # vars
    steps = 10000
    re = 1000
    base_lenght = 40
    uw = 0.1
    relaxation = (2 * re) / (6 * base_lenght * uw + re)
    # calls
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank_in_one_direction = int(np.sqrt(size)) # for an MPI thingi with 9 processes -> 3x3 field
    if rank_in_one_direction*rank_in_one_direction != size:
        return RuntimeError
    process_info = fill_mpi_struct_fields(comm.Get_rank(),size,
                                          rank_in_one_direction,rank_in_one_direction,base_lenght,
                                          relaxation,steps,uw)
    print(process_info)
    sliding_lid_mpi(process_info,comm)


# call()
#comm = MPI.COMM_WORLD
#process_info = fill_mpi_struct_fields(0,4,2,2,300,0,0,0)
#fg = collapse_data(process_info,np.ones((9,152,152)),comm)
#print(fg.shape)
# g = np.zeros((9,27,27))
# k = g[:,1:-1,1:-1]
# print(k.shape)
startime = time.time()
call()
print("Took {} s".format(time.time()-startime))