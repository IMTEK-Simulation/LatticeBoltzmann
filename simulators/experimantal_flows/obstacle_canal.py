'''
Remider CodingRules:
Zeilenumbruch bei Spalte 120
Modulname, Klassennamen als CamelCase
Variablennamen, Methodennamen, Funktionsnamen mit unter_strichen
Bitte nicht CamelCase und Unterstriche mischen
'''

'''
Basic idea to make a obstacle canal/wind tunnel which houses a small object for performance 
evaluation. 
Should and can be used for all further refinements/developments. 
Will take a more classy approach this time around
Maybe augment with C++ at a later point.
'''

# General Imports
import numpy as np
from dataclasses import dataclass
from mpi4py import MPI

#Class structures for organization of an mpi call modified from simple flows
@dataclass
class boundariesApplied:
    apply_left: bool = False
    apply_right: bool = False
    apply_top: bool = False
    apply_bottom: bool = False

@dataclass
class cellNeighbors:
    left: int = -1
    right: int = -1
    top: int = -1
    bottom: int = -1

@dataclass
class mpiPackageStructure:
    boundaries_info: boundariesApplied = (False, False, False, False)
    neighbors: cellNeighbors = (-1, -1, -1, -1)
    # sizes and position in the whole grid
    size_x: int = -1
    size_y: int = -1
    pos_x: int = -1
    pos_y: int = -1
    # for MPI stuff
    rank: int = -1
    size: int = -1
    # overall
    relaxation: int = -1
    base_grid: int = -1
    steps: int = -1
    uw: int = -1

    def fill_mpi_struct_fields(self, rank, size, max_x, max_y, base_grid, relaxation, steps, uw):
        #
        self.rank = rank
        self.size = size
        self.get_postions_out_of_rank_size_quadratic(rank, size)
        self.set_boundary_info(self.pos_x, self.pos_y, max_x - 1, max_y - 1)  # i should know my own code lol
        # cheeky
        self.size_x = base_grid // (max_x) + 2
        self.size_y = base_grid // (max_y) + 2
        self.determin_neighbors(rank, size)
        #
        self.relaxation = relaxation
        self.base_grid = base_grid
        self.steps = steps
        self.uw = uw

    def get_postions_out_of_rank_size_quadratic(self, rank, size):
        ##
        # assume to be quadratic
        edge_lenght = int(np.sqrt(size))
        ###
        self.pox = rank % edge_lenght
        self.poy = rank // edge_lenght

    def determin_neighbors(self, rank, size):
        # determin edge lenght
        edge_lenght = int(np.sqrt(size))
        ###
        neighbor = cellNeighbors()
        neighbor.top = rank + edge_lenght
        neighbor.bottom = rank - edge_lenght
        neighbor.right = rank + 1
        neighbor.left = rank - 1
        ###
        self.neighbors = neighbor

    def set_boundary_info(self,pox, poy, max_x, max_y):
        info = boundariesApplied(False, False, False, False)
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
        self.boundaries_info = info


class obstacleWindTunnel:
    def __init__(self,steps,re,base_length,uw):
        self.mpi = mpiPackageStructure()
        self.velocity_set = np.array([[0, 1, 0, -1, 0, 1, -1, -1, 1],
                                      [0, 0, 1, 0, -1, 1, 1, -1, -1]]).T
        self.re = re
        relaxation = (2 * re) / (6 * base_length * uw + re)
        self.comm = MPI.COMM_WORLD
        size = self.comm.Get_size()
        # test for badness
        rank_in_one_direction = int(np.sqrt(size))  # for an MPI thingi with 9 processes -> 3x3 field
        if rank_in_one_direction * rank_in_one_direction != size:
            return RuntimeError
        self.mpi.fill_mpi_struct_fields(rank = self.comm.Get_rank(),
                                        size = size,
                                        max_x = rank_in_one_direction,
                                        max_y = rank_in_one_direction,
                                        base_grid=base_length,
                                        relaxation= relaxation,
                                        steps=steps,
                                        uw = uw)
    def run(self):
        print(self.mpi)


tun = obstacleWindTunnel(steps=1000000,re=1000,base_length=300,uw = 0.1)
tun.run()