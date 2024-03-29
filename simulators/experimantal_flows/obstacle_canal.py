'''
Remider CodingRules:
Zeilenumbruch bei Spalte 120
Modulname, Klassennamen als CamelCase
Variablennamen, Methodennamen, Funktionsnamen mit unter_strichen
Bitte nicht CamelCase und Unterstriche mischen
für statics + enum like names nehm ich ALL CAPS was glauch ich standard ist
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
import time
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import enum

# enums and statics
class boundaryStates(enum.Enum):
    NONE = enum.auto()
    BAUNCE_BACK = enum.auto()
    COMMUNICATE = enum.auto()
    PERIODIC_BOUNDARY = enum.auto()
    DEAD_BOUNDARY = enum.auto()


rho_null = 1
rho_diff = 0.01
velocity_set = np.array([[0, 1, 0, -1, 0, 1, -1, -1, 1],
                         [0, 0, 1, 0, -1, 1, 1, -1, -1]]).T
# class structures
# for basic ingidients to the packed struct with contains all of the grid info locally and globally
# (just not the girds itself)
@dataclass
class boundariesApplied:
    apply_left: boundaryStates = boundaryStates.NONE
    apply_right: boundaryStates = boundaryStates.NONE
    apply_top: boundaryStates = boundaryStates.NONE
    apply_bottom: boundaryStates = boundaryStates.NONE


@dataclass
class cellNeighbors:
    left: int = -1
    right: int = -1
    top: int = -1
    bottom: int = -1


@dataclass
class calculationCellInfo:
    # rank of the calculation cell
    rank: int = -1
    # total number of calculation cells
    size: int = -1
    # positions in x and y in the whole gird or calculation cells
    # numeration starts from 0,0
    cell_position_x: int = -1
    cell_position_y: int = -1
    # number of calculation cells in different directions
    cell_numbers_in_x: int = -1
    cell_numbers_in_y: int = -1
    # final cell positions for boundary determination
    final_cell_position_x: int = -1
    final_cell_position_y: int = -1


@dataclass
class packageStructure:
    # aka all the (static) info about the grid in its environment
    boundaries_info: boundariesApplied = (boundaryStates.NONE, boundaryStates.NONE,
                                          boundaryStates.NONE, boundaryStates.NONE)
    neighbors: cellNeighbors = (-1, -1, -1, -1)
    calculation_cell_info:calculationCellInfo = (-1,-1,-1,-1,-1,-1,-1,-1)
    # sizes and position in the whole grid
    size_x: int = -1
    size_y: int = -1
    base_grid_x: int = -1
    base_grid_y: int = -1
    local_grid_start_x: int = -1
    local_grid_start_y: int = -1
    local_grid_end_x: int = -1
    local_grid_end_y: int = -1

    def fill_calculation_cell_field(self,rank,size,number_of_cells_x,number_of_cells_y):
        calculation_cell = calculationCellInfo()
        # bind info
        calculation_cell.rank = rank
        calculation_cell.size = size # aka the number in the grid
        calculation_cell.cell_numbers_in_x = number_of_cells_x
        calculation_cell.cell_numbers_in_y = number_of_cells_y
        calculation_cell.final_cell_position_x = number_of_cells_x - 1
        calculation_cell.final_cell_position_y = number_of_cells_y - 1
        # calculate the rest
        # numbering starts with 0,0, x ->
        calculation_cell.cell_position_x = rank % number_of_cells_x
        calculation_cell.cell_position_y = rank // number_of_cells_y
        # set
        self.calculation_cell_info = calculation_cell

    def fill_packed_struct_fields(self, rank, size, number_of_cells_x, number_of_cells_y, base_grid_x, base_grid_y,
                                  boundary_state_left,boundary_state_right, boundary_state_top, boundary_state_bottom):
        # grid sizes
        self.base_grid_x = base_grid_x
        self.base_grid_y = base_grid_y
        # set all the cell related information in the context of the mpi
        self.fill_calculation_cell_field(rank = rank,size = size,
                                         number_of_cells_x= number_of_cells_x,
                                         number_of_cells_y=number_of_cells_y)
        self.set_boundary_info_local(boundary_state_left= boundary_state_left,
                                     boundary_state_right= boundary_state_right,
                                     boundary_state_top= boundary_state_top, 
                                     boundary_state_bottom = boundary_state_bottom)
        self.determin_neighbors(calculation_cell_info=self.calculation_cell_info)
        self.set_own_cell_size()

    def determin_neighbors(self, calculation_cell_info):
        # (determins the neighbour in term of its size not its grid position)
        # (we can get away with unsafe grid positions as those are set at the actual boundary of the domain)
        ###
        neighbor = cellNeighbors()
        neighbor.top = calculation_cell_info.rank + calculation_cell_info.cell_numbers_in_x
        neighbor.bottom = calculation_cell_info.rank - calculation_cell_info.cell_numbers_in_x
        neighbor.right = calculation_cell_info.rank + 1
        neighbor.left = calculation_cell_info.rank - 1
        ###
        self.neighbors = neighbor

    def set_boundary_info_local(self, boundary_state_left,boundary_state_right, boundary_state_top, boundary_state_bottom):
        # global boundary info is used for an init point will only be overwritten by communicate
        info = boundariesApplied(boundary_state_left, boundary_state_right, boundary_state_top, boundary_state_bottom)
        ##
        # if cell position in set {x;y} -> {0,max;0,max} no boundaries
        if self.calculation_cell_info.cell_position_x != 0:
            info.apply_left = boundaryStates.COMMUNICATE
        if self.calculation_cell_info.cell_position_y != 0:
            info.apply_bottom = boundaryStates.COMMUNICATE
        if self.calculation_cell_info.cell_position_x != self.calculation_cell_info.final_cell_position_x:
            info.apply_right = boundaryStates.COMMUNICATE
        if self.calculation_cell_info.cell_position_y != self.calculation_cell_info.final_cell_position_y:
            info.apply_top = boundaryStates.COMMUNICATE
        ##
        self.boundaries_info = info

    def set_own_cell_size(self):
        # case 1 basic cell no need to add further grid points
        self.size_x = self.base_grid_x // self.calculation_cell_info.cell_numbers_in_x
        self.size_y = self.base_grid_y // self.calculation_cell_info.cell_numbers_in_y
        # case 2 end cell in x -> check weather or not there are more grid points to be dealt with
        if self.calculation_cell_info.cell_position_x == self.calculation_cell_info.final_cell_position_x:
            self.size_x += self.base_grid_x % self.calculation_cell_info.cell_numbers_in_x
        # case 3 end cell in y -> check weather or not there are more grid points to be dealt with
        if self.calculation_cell_info.cell_position_y == self.calculation_cell_info.final_cell_position_y:
            self.size_y += self.base_grid_y % self.calculation_cell_info.cell_numbers_in_y
        # set local relation to global grid
        self.local_grid_start_x = (self.base_grid_x // self.calculation_cell_info.cell_numbers_in_x) * self.\
            calculation_cell_info.cell_position_x
        self.local_grid_start_y = self.base_grid_y // self.calculation_cell_info.cell_numbers_in_y * self.\
            calculation_cell_info.cell_position_y
        self.local_grid_end_x = self.local_grid_start_x + self.size_x
        self.local_grid_end_y = self.local_grid_start_y + self.size_y
        # + 2 describes boundary/com points
        self.size_x += 2
        self.size_y += 2

@dataclass
class globalObstacleInfo:
    # information related to a rectangular obstacle
    # main points that span the rectangular
    start_x: int = -1
    start_y: int = -1
    end_x: int = -1
    end_y: int = -1


@dataclass
class localObstacleInfo:
    # conceptually still a rectangular form like the global one
    boundary_state: boundariesApplied = (boundaryStates.NONE, boundaryStates.NONE,
                                         boundaryStates.NONE, boundaryStates.NONE)
    # contains more info than just the grid points that span the cell
    # boundary info (local)
    start_x: int = -1
    start_y: int = -1
    end_x: int = -1
    end_y: int = -1
    # dead cells useful for determining unused parts in the grid shape changes depending on different cases
    dead_start_x: int = -1
    dead_start_y: int = -1
    dead_end_x: int = -1
    dead_end_y: int = -1


# regular classes
def equilibrium_calculation(rho, ux, uy):
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


class obstacleWindTunnel:
    def __init__(self, steps, re,number_of_cells_x,number_of_cells_y ,base_length_x, base_length_y, uw, boundary_state_left,boundary_state_right,
                 boundary_state_top, boundary_state_bottom, title, obstacle_start, obstacle_end):
        self.time = 0
        self.packed_info = packageStructure()
        self.re = re
        self.uw = uw
        self.steps = steps
        self.title = title
        self.rho_in = rho_null + rho_diff
        self.rho_out = rho_null - rho_diff
        # principal length for calculating relaxation for the correct re number smaller should be the significant one
        principal_length = 0
        if base_length_x > base_length_y:
            principal_length = base_length_y
        else:
            principal_length = base_length_x
        self.relaxation = (2 * re) / (6 * principal_length * uw + re)
        self.relaxation = 0.5
        self.shear_viscosity = (1/self.relaxation-0.5)/3
        # set up the information for parallel use
        self.comm = MPI.COMM_WORLD # only var thats not in the packed struct
        self.packed_info.fill_packed_struct_fields(rank=self.comm.Get_rank(), size=self.comm.Get_size(),
                                                   number_of_cells_x=number_of_cells_x,
                                                   number_of_cells_y=number_of_cells_y,
                                                   base_grid_x=base_length_x, base_grid_y=base_length_y,
                                                   boundary_state_left= boundary_state_left,
                                                   boundary_state_right= boundary_state_right,
                                                   boundary_state_top = boundary_state_top,
                                                   boundary_state_bottom = boundary_state_bottom)
        # grid type variables
        self.rho = np.ones((self.packed_info.size_x, self.packed_info.size_y))
        self.ux = np.zeros((self.packed_info.size_x, self.packed_info.size_y))
        self.uy = np.zeros((self.packed_info.size_x, self.packed_info.size_y))
        self.grid = self.equilibrium()
        self.full_grid = np.ones((9, self.packed_info.base_grid_x, self.packed_info.base_grid_y))
        # obstacle info
        self.global_obstacle = globalObstacleInfo(obstacle_start[0],obstacle_start[1],obstacle_end[0],obstacle_end[1])
        self.local_obstacle = localObstacleInfo()
        self.setup_obstacle()

    def run(self):
        self.time = time.time()
        print(self.packed_info)
        # iterations
        for i in range(self.steps):
            self.stream()
            self.periodic_boundary_delta_p()
            self.bounce_back_choosen()
            self.apply_obstacle()
            self.caluculate_rho_ux_uy()
            self.collision()
            self.comunicate()
    #
        self.collapse_data()
        self.plotter_stream()
        # self.plotter_simple()

    ''' core methods '''

    def equilibrium(self):
        uxy_3plus = 3 * (self.ux + self.uy)
        uxy_3miuns = 3 * (self.ux - self.uy)
        uu = 3 * (self.ux * self.ux + self.uy * self.uy)
        ux_6 = 6 * self.ux
        uy_6 = 6 * self.uy
        uxx_9 = 9 * self.ux * self.ux
        uyy_9 = 9 * self.uy * self.uy
        uxy_9 = 9 * self.ux * self.uy
        return np.array([(2 * self.rho / 9) * (2 - uu),
                         (self.rho / 18) * (2 + ux_6 + uxx_9 - uu),
                         (self.rho / 18) * (2 + uy_6 + uyy_9 - uu),
                         (self.rho / 18) * (2 - ux_6 + uxx_9 - uu),
                         (self.rho / 18) * (2 - uy_6 + uyy_9 - uu),
                         (self.rho / 36) * (1 + uxy_3plus + uxy_9 + uu),
                         (self.rho / 36) * (1 - uxy_3miuns - uxy_9 + uu),
                         (self.rho / 36) * (1 - uxy_3plus + uxy_9 + uu),
                         (self.rho / 36) * (1 + uxy_3miuns - uxy_9 + uu)])

    def stream(self):
        for i in range(1, 9):
            self.grid[i] = np.roll(self.grid[i], velocity_set[i], axis=(0, 1))

    def bounce_back_choosen(self):
        if self.packed_info.boundaries_info.apply_right == boundaryStates.BAUNCE_BACK:
            # right so x = -1
            self.grid[3, -2, :] = self.grid[1, -1, :]
            self.grid[6, -2, :] = self.grid[8, -1, :]
            self.grid[7, -2, :] = self.grid[5, -1, :]
        if self.packed_info.boundaries_info.apply_left == boundaryStates.BAUNCE_BACK:
            # left so x = 0
            self.grid[1, 1, :] = self.grid[3, 0, :]
            self.grid[5, 1, :] = self.grid[7, 0, :]
            self.grid[8, 1, :] = self.grid[6, 0, :]

        # Bottom + Top
        if self.packed_info.boundaries_info.apply_bottom == boundaryStates.BAUNCE_BACK:
            # for bottom y = 0
            self.grid[2, :, 1] = self.grid[4, :, 0]
            self.grid[5, :, 1] = self.grid[7, :, 0]
            self.grid[6, :, 1] = self.grid[8, :, 0]
        if self.packed_info.boundaries_info.apply_top == boundaryStates.BAUNCE_BACK:
            # for top y = -1
            self.grid[4, :, -2] = self.grid[2, :, -1]
            self.grid[7, :, -2] = self.grid[5, :, -1] - 1 / 6 * self.uw
            self.grid[8, :, -2] = self.grid[6, :, -1] + 1 / 6 * self.uw

    def caluculate_rho_ux_uy(self):
        self.rho = np.sum(self.grid, axis=0)  # sums over each one individually
        self.ux = np.divide((self.grid[1] + self.grid[5] + self.grid[8]) - (self.grid[3] + self.grid[6] + self.grid[7]),
                            self.rho, where=self.rho != 0)
        self.uy = np.divide((self.grid[2] + self.grid[5] + self.grid[6]) - (self.grid[4] + self.grid[7] + self.grid[8]),
                            self.rho, where=self.rho != 0)

    def collision(self):
        self.grid -= self.relaxation * (self.grid - self.equilibrium())

    def comunicate(self):
        # Right + Left
        if self.packed_info.boundaries_info.apply_right == boundaryStates.COMMUNICATE:
            recvbuf = self.grid[:, -1, :].copy()
            self.comm.Sendrecv(self.grid[:, -2, :].copy(), self.packed_info.neighbors.right, recvbuf=recvbuf, sendtag=11, recvtag=12)
            self.grid[:, -1, :] = recvbuf
        if self.packed_info.boundaries_info.apply_left == boundaryStates.COMMUNICATE:
            recvbuf = self.grid[:, 0, :].copy()
            self.comm.Sendrecv(self.grid[:, 1, :].copy(), self.packed_info.neighbors.left, recvbuf=recvbuf, sendtag=12, recvtag=11)
            self.grid[:, 0, :] = recvbuf

        # Bottom + Top
        if self.packed_info.boundaries_info.apply_bottom == boundaryStates.COMMUNICATE:
            recvbuf = self.grid[:, :, 0].copy()
            self.comm.Sendrecv(self.grid[:, :, 1].copy(), self.packed_info.neighbors.bottom, recvbuf=recvbuf, sendtag=99, recvtag=98)
            self.grid[:, :, 0] = recvbuf
        if self.packed_info.boundaries_info.apply_top == boundaryStates.COMMUNICATE:
            recvbuf = self.grid[:, :, -1].copy()
            self.comm.Sendrecv(self.grid[:, :, -2].copy(), self.packed_info.neighbors.top, recvbuf=recvbuf, sendtag=98, recvtag=99)
            self.grid[:, :, -1] = recvbuf

    def periodic_boundary_delta_p(self):
        # movement is assumed form left to right or from top to bottom
        self.caluculate_rho_ux_uy()
        equilibrium = self.equilibrium()
        # loop through the individual sides (locally)
        # left + right
        if self.packed_info.boundaries_info.apply_left == boundaryStates.PERIODIC_BOUNDARY:
            equilibrium_in = equilibrium_calculation(self.rho_in, self.ux[-2, :],self.uy[-2, :])
            self.grid[:,0,:] = equilibrium_in + self.grid[:, -2, :] - equilibrium[:, -2, :]

        if self.packed_info.boundaries_info.apply_right == boundaryStates.PERIODIC_BOUNDARY:
            equilibrium_out = equilibrium_calculation(self.rho_out, self.ux[1, :], self.uy[1, :])
            self.grid[:, -1, :] = equilibrium_out + self.grid[:, 1, :] - equilibrium[:, 1, :]

        # Top + bottom
        if self.packed_info.boundaries_info.apply_top == boundaryStates.PERIODIC_BOUNDARY:
            equilibrium_in = equilibrium_calculation(self.rho_in, self.ux[:, -2], self.uy[:, -2])
            self.grid[:, :, 0] = equilibrium_in + self.grid[:, :, -2] - equilibrium[:, :, -2]

        if self.packed_info.boundaries_info.apply_bottom == boundaryStates.PERIODIC_BOUNDARY:
            equilibrium_out = equilibrium_calculation(self.rho_out, self.ux[:, 1], self.uy[:, 1])
            self.grid[:, -1, :] = equilibrium_out + self.grid[:, :, 1] - equilibrium[:, :, 1]

    ''' obstacle methods '''
    def setup_obstacle(self):
        print(self.global_obstacle)
        # set up the local info
        # relation between global coordinates and local coordinates
        # dont forget about that pesky communication layer
        # check weather or not inside the cell
        # Todo go from simple to harder this is the hard part! prob move to the packed struct
        # local active cells (aka boundary)
        if self.packed_info.local_grid_start_x < self.global_obstacle.start_x:
            print("T1")
            self.local_obstacle.start_x = self.global_obstacle.start_x
            self.local_obstacle.end_x = self.global_obstacle.end_x
        if self.packed_info.local_grid_start_y < self.global_obstacle.start_y:
            print("T2")
            self.local_obstacle.start_y = self.global_obstacle.start_y
            self.local_obstacle.end_y = self.global_obstacle.end_y
        # local dead cell
        self.local_obstacle.dead_start_x = self.global_obstacle.start_x + 1
        self.local_obstacle.dead_start_x = self.global_obstacle.start_y + 1
        self.local_obstacle.dead_end_x = self.global_obstacle.end_x - 1
        self.local_obstacle.dead_end_y = self.global_obstacle.end_y -1
        # Boundary states
        info = boundariesApplied(boundaryStates.BAUNCE_BACK,boundaryStates.BAUNCE_BACK,boundaryStates.BAUNCE_BACK,boundaryStates.BAUNCE_BACK)
        self.local_obstacle.boundary_state = info
        print(self.local_obstacle)


    def apply_obstacle(self):
        # set the cells to dead
        self.ux[self.local_obstacle.start_x:self.local_obstacle.end_x,self.local_obstacle.start_y:self.local_obstacle.end_y] = 0
        self.uy[self.local_obstacle.start_x:self.local_obstacle.end_x,self.local_obstacle.start_y:self.local_obstacle.end_y] = 0
        # self.grid[:,self.local_obstacle.dead_start_x:self.local_obstacle.dead_end_x, self.local_obstacle.start_y:self.local_obstacle.dead_end_y] = 0
        # apply boundaries position is relative to the pos on the rectangle
        if self.local_obstacle.boundary_state.apply_left == boundaryStates.BAUNCE_BACK:
            # 1 into 3
            self.grid[3, self.local_obstacle.start_x - 1, self.local_obstacle.start_y: self.local_obstacle.end_y] = \
                self.grid[1, self.local_obstacle.start_x, self.local_obstacle.start_y: self.local_obstacle.end_y]
            # 5 into 7
            self.grid[7,self.local_obstacle.start_x-1, self.local_obstacle.start_y: self.local_obstacle.end_y] = \
                self.grid[5, self.local_obstacle.start_x, self.local_obstacle.start_y: self.local_obstacle.end_y]
            # 8 into 6
            self.grid[6, self.local_obstacle.start_x-1, self.local_obstacle.start_y: self.local_obstacle.end_y] = \
                self.grid[8, self.local_obstacle.start_x, self.local_obstacle.start_y: self.local_obstacle.end_y]
        if self.local_obstacle.boundary_state.apply_right == boundaryStates.BAUNCE_BACK:
            # 3 into 1
            self.grid[1, self.local_obstacle.end_x+1, self.local_obstacle.start_y:self.local_obstacle.end_y] = \
                self.grid[3, self.local_obstacle.end_x, self.local_obstacle.start_y:self.local_obstacle.end_y]
            # 7 into 5
            self.grid[5, self.local_obstacle.end_x+1, self.local_obstacle.start_y:self.local_obstacle.end_y] =\
                self.grid[7, self.local_obstacle.end_x, self.local_obstacle.start_y:self.local_obstacle.end_y]
            # 8 into 6
            self.grid[6, self.local_obstacle.end_x+1, self.local_obstacle.start_y:self.local_obstacle.end_y] = \
                self.grid[8, self.local_obstacle.end_x, self.local_obstacle.start_y:self.local_obstacle.end_y]
        if self.local_obstacle.boundary_state.apply_top == boundaryStates.BAUNCE_BACK:
            # 4 into 2
            self.grid[2, self.local_obstacle.start_x:self.local_obstacle.end_x, self.local_obstacle.end_y + 1] = \
                self.grid[4, self.local_obstacle.start_x:self.local_obstacle.end_x, self.local_obstacle.end_y]
            # 7 into 5
            self.grid[5, self.local_obstacle.start_x:self.local_obstacle.end_x, self.local_obstacle.end_y + 1] = \
                self.grid[7, self.local_obstacle.start_x:self.local_obstacle.end_x, self.local_obstacle.end_y]
            # 8 into 5
            self.grid[6, self.local_obstacle.start_x:self.local_obstacle.end_x, self.local_obstacle.end_y + 1] = \
                self.grid[8, self.local_obstacle.start_x:self.local_obstacle.end_x, self.local_obstacle.end_y]
        if self.local_obstacle.boundary_state.apply_bottom == boundaryStates.BAUNCE_BACK:
            # 2 into 4
            self.grid[4, self.local_obstacle.start_x:self.local_obstacle.end_x, self.local_obstacle.start_y - 1] = \
                self.grid[2, self.local_obstacle.start_x:self.local_obstacle.end_x, self.local_obstacle.start_y]
            # 5 into 7
            self.grid[7, self.local_obstacle.start_x:self.local_obstacle.end_x, self.local_obstacle.start_y - 1] = \
                self.grid[5, self.local_obstacle.start_x:self.local_obstacle.end_x, self.local_obstacle.start_y]
            # 6 into 8
            self.grid[8, self.local_obstacle.start_x:self.local_obstacle.end_x, self.local_obstacle.start_y - 1] = \
                self.grid[6, self.local_obstacle.start_x:self.local_obstacle.end_x, self.local_obstacle.start_y]

    ''' auxiliary methods '''

    def collapse_data(self):
        # process 0 gets the data and does the visualization
        if self.packed_info.calculation_cell_info.rank == 0:
            original_x = self.packed_info.size_x - 2  # ie the base size of the grid on that the
            original_y = self.packed_info.size_y - 2  # calculation ran without extra cells for synchron +
            # write the own stuff into it first
            self.full_grid[:, 0:original_x, 0:original_y] = self.grid[:, 1:-1, 1:-1]
            temp = np.zeros((9, original_x, original_y))
            for i in range(1, self.packed_info.calculation_cell_info.size):
                self.comm.Recv(temp, source=i, tag=i)
                # determine start end endpoints to copy to in the grid
                copy_start_x = 0 + original_x * self.packed_info.base_grid_x
                copy_end_x = original_x + original_x * self.packed_info.base_grid_x
                copy_start_y = 0 + original_y * self.packed_info.base_grid_y
                copy_end_y = original_y + original_y * self.packed_info.base_grid_y
                # copy
                self.full_grid[:, copy_start_x:copy_end_x, copy_start_y:copy_end_y] = temp
            #
            self.grid = self.full_grid
        # all the others send to p0
        else:
            self.comm.Send(self.grid[:, 1:-1, 1:-1].copy(), dest=0, tag=self.packed_info.calculation_cell_info.rank)

    def plotter_stream(self):
        # plot
        if self.packed_info.calculation_cell_info.rank == 0:
            print("Making Image")
            # recalculate ux and uy
            # change the original grid to the full one
            self.caluculate_rho_ux_uy()
            # acutal plot

            x = np.arange(0, self.packed_info.base_grid_x)
            y = np.arange(0, self.packed_info.base_grid_y)
            X, Y = np.meshgrid(x, y)
            speed = np.sqrt(self.ux.T ** 2 + self.uy.T ** 2)
            # plot
            # plt.streamplot(X,Y,full_ux.T,full_uy.T)
            plt.streamplot(X, Y, self.ux.T, self.uy.T, color=speed, cmap=plt.cm.jet)
            rect = mpatches.Rectangle((self.global_obstacle.start_x,self.global_obstacle.start_y),
                                      self.global_obstacle.end_x - self.global_obstacle.start_x,
                                      self.global_obstacle.end_y - self.global_obstacle.start_y,
                                      fill = False, color="black", linewidth=2)
            ax = plt.gca()
            ax.add_patch(rect)
            ax.set_xlim([0, self.packed_info.base_grid_x + 1])
            ax.set_ylim([0, self.packed_info.base_grid_y + 1])
            plt.title(self.title)
            plt.xlabel("x-Position")
            plt.ylabel("y-Position")
            fig = plt.colorbar()
            fig.set_label("Velocity u(x,y,t)", rotation=270, labelpad=15)
            savestring = "slidingLidmpi" + str(self.packed_info.calculation_cell_info.size) + ".png"
            plt.savefig(savestring)
            plt.show()

        if self.packed_info.calculation_cell_info.rank == 0:
            savestring = "slidingLidmpi" + str(self.packed_info.calculation_cell_info.size) + ".txt"
            f = open(savestring, "w")
            totaltime = time.time() - self.time
            f.write(str(totaltime))
            f.close()

    def plotter_simple(self):
        # visualize
        delta = 2.0 * rho_diff / self.packed_info.base_grid_x / self.shear_viscosity / 2.
        y = np.linspace(0, self.packed_info.base_grid_y, self.packed_info.base_grid_y + 1) + 0.5
        u_analytical = delta * y * (self.packed_info.base_grid_y - y) / 3.
        plt.plot(u_analytical[:-1], label='Analytical')
        # plt.plot(u_analytical, label='analytical')
        number_of_cuts_in_x = 2
        for i in range(1, number_of_cuts_in_x):
            point = int(i * self.packed_info.base_grid_x / number_of_cuts_in_x)
            plt.plot(self.ux[point, 1:-1], label="Calculated")
        print(len(self.ux[25, 1:-1]))
        plt.legend()
        plt.xlabel('Position in cross section')
        plt.ylabel('Velocity')
        plt.title('Pouisuelle flow (Gridsize 100x52, $\omega = 0.5$, $\\nabla \\rho = 0.002$)')
        savestring = "PouisuelleFlow.png"
        plt.savefig(savestring)
        plt.show()


t= obstacleWindTunnel(steps=1000,re=1000,base_length_x=100,base_length_y=50,uw = 0,
                      number_of_cells_x=1,
                      number_of_cells_y=1,
                      # kinda meh the global state have to given thou 2 times
                      boundary_state_left=boundaryStates.PERIODIC_BOUNDARY,
                      boundary_state_right=boundaryStates.PERIODIC_BOUNDARY,
                      boundary_state_top=boundaryStates.BAUNCE_BACK,
                      boundary_state_bottom=boundaryStates.BAUNCE_BACK,
                      # title for the plot
                      title="Work in progress",
                      # obstacle placement in the global grid
                      obstacle_start=(50, 24),
                      obstacle_end=(53, 27))

t.run()