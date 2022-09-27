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
import enum

# enums and statics
class boundaryStates(enum.Enum):
    NONE = enum.auto()
    BAUNCE_BACK = enum.auto()
    COMMUNICATE = enum.auto()
    PERIODIC_BOUNDARY = enum.auto()


rho_null = 1
rho_diff = 0.001
# class structures
# for organization of an mpi call modified from simple flows
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
class mpiPackageStructure:
    # aka all the (static) info in on package
    boundaries_info: boundariesApplied = (boundaryStates.NONE, boundaryStates.NONE,
                                          boundaryStates.NONE, boundaryStates.NONE)
    neighbors: cellNeighbors = (-1, -1, -1, -1)
    calculation_cell_info:calculationCellInfo = (-1,-1,-1,-1,-1,-1,-1,-1)
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
    base_grid_x: int = -1
    base_grid_y: int = -1
    steps: int = -1
    uw: int = -1

    def fill_calculation_cell_field(self,rank,size,number_of_cells_x,number_of_cells_y):
        calculation_cell = calculationCellInfo()
        calculation_cell.rank = rank
        calculation_cell.size = size
        calculation_cell.cell_numbers_in_x = number_of_cells_x
        calculation_cell.cell_numbers_in_y = number_of_cells_y
        # set
        self.calculation_cell_info = calculation_cell



    def fill_mpi_struct_fields(self,
                               rank, size,
                               number_of_cells_x, number_of_cells_y,
                               base_grid_x, base_grid_y, relaxation, steps, uw,
                               global_boundary_info):
        #
        self.fill_calculation_cell_field(rank = rank,size = size,
                                         number_of_cells_x= number_of_cells_x,
                                         number_of_cells_y=number_of_cells_y)
        self.rank = rank
        self.size = size
        self.get_postions_out_of_rank_size_quadratic()
        print(self.pos_x,self.pos_y)
        self.set_boundary_info_local(calculation_cell_info = self.calculation_cell_info,
                                     global_boundary_info= global_boundary_info)  # i should know my own code lol
        # cheeky
        self.size_x = base_grid_x // number_of_cells_x + 2
        self.size_y = base_grid_y // number_of_cells_y + 2
        self.determin_neighbors(calculation_cell_info= self.calculation_cell_info)
        #
        self.relaxation = relaxation
        self.base_grid_x = base_grid_x
        self.base_grid_y = base_grid_y
        self.steps = steps
        self.uw = uw

    def get_postions_out_of_rank_size_quadratic(self):
        ##
        # assume to be quadratic
        edge_lenght = int(np.sqrt(self.size))
        ###
        self.pos_x = self.rank % edge_lenght
        self.pos_y = self.rank // edge_lenght

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

    def set_boundary_info_local(self,calculation_cell_info, global_boundary_info):
        # global boundary info is used for an init point will only be overwritten by communicate
        info = global_boundary_info
        ##
        # if cell position in set {x;y} -> {0,max;0,max} no boundaries
        if calculation_cell_info.cell_position_x != 0:
            info.apply_left = boundaryStates.COMMUNICATE
        if calculation_cell_info.cell_position_y != 0:
            info.apply_bottom = boundaryStates.COMMUNICATE
        if calculation_cell_info.cell_position_x != calculation_cell_info.final_cell_position_x:
            info.apply_right = boundaryStates.COMMUNICATE
        if calculation_cell_info.cell_position_y != calculation_cell_info.final_cell_position_y:
            info.apply_top = boundaryStates.COMMUNICATE
        ##
        self.boundaries_info = info

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
    def __init__(self, steps, re, base_length_x, base_length_y, uw, boundary_state_left,boundary_state_right,
                 boundary_state_top, boundary_state_bottom, title):
        self.time = 0
        self.mpi = mpiPackageStructure()
        self.velocity_set = np.array([[0, 1, 0, -1, 0, 1, -1, -1, 1],
                                      [0, 0, 1, 0, -1, 1, 1, -1, -1]]).T
        self.re = re
        self.title = title
        self.rho_in  = rho_null - rho_diff
        self.rho_out = rho_null + rho_diff

        self.global_boundaries = boundariesApplied(boundary_state_left,boundary_state_right,boundary_state_top,
                                                   boundary_state_bottom)
        # principal length for calculating relaxation for the correct re number smaller should be the significant one
        principal_length = 0
        if base_length_x > base_length_y:
            principal_length = base_length_y
        else:
            principal_length = base_length_x
        relaxation = (2 * re) / (6 * principal_length * uw + re)
        relaxation = 0.5 # TODO pflow fix
        self.shear_viscosity = (1/relaxation-0.5)/3
        self.comm = MPI.COMM_WORLD
        size = self.comm.Get_size()
        print(size)
        # TODO change me
        rank_in_one_direction = int(np.sqrt(size)) # for an MPI thingi with 9 processes -> 3x3 field
        self.mpi.fill_mpi_struct_fields(rank = self.comm.Get_rank(),
                                        size = size,
                                        number_of_cells_x = rank_in_one_direction,
                                        number_of_cells_y = rank_in_one_direction,
                                        base_grid_x=base_length_x,
                                        base_grid_y=base_length_y,
                                        relaxation= relaxation,
                                        steps=steps,
                                        uw = uw,
                                        global_boundary_info = self.global_boundaries)
        # grid type variables
        self.rho = np.ones((self.mpi.size_x, self.mpi.size_y))
        self.ux = np.zeros((self.mpi.size_x, self.mpi.size_y))
        self.uy = np.zeros((self.mpi.size_x, self.mpi.size_y))
        self.grid = self.equilibrium()
        self.full_grid = np.ones((9, self.mpi.base_grid_x, self.mpi.base_grid_y))

    def run(self):
        self.time = time.time()
        print(self.mpi)
        # iterations
        for i in range(self.mpi.steps):
            self.periodic_boundary_delta_p()
            self.stream()
            self.bounce_back_choosen()
            self.caluculate_rho_ux_uy()
            self.collision()
            self.comunicate()
    #
        self.collapse_data()
        # self.plotter_stream()
        self.plotter_simple()

    ''' basic functions '''

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
            self.grid[i] = np.roll(self.grid[i], self.velocity_set[i], axis=(0, 1))

    def bounce_back_choosen(self):
        if self.mpi.boundaries_info.apply_right == boundaryStates.BAUNCE_BACK:
            # right so x = -1
            self.grid[3, -2, :] = self.grid[1, -1, :]
            self.grid[6, -2, :] = self.grid[8, -1, :]
            self.grid[7, -2, :] = self.grid[5, -1, :]
        if self.mpi.boundaries_info.apply_left == boundaryStates.BAUNCE_BACK:
            # left so x = 0
            self.grid[1, 1, :] = self.grid[3, 0, :]
            self.grid[5, 1, :] = self.grid[7, 0, :]
            self.grid[8, 1, :] = self.grid[6, 0, :]

        # Bottom + Top
        if self.mpi.boundaries_info.apply_bottom == boundaryStates.BAUNCE_BACK:
            # for bottom y = 0
            self.grid[2, :, 1] = self.grid[4, :, 0]
            self.grid[5, :, 1] = self.grid[7, :, 0]
            self.grid[6, :, 1] = self.grid[8, :, 0]
        if self.mpi.boundaries_info.apply_top == boundaryStates.BAUNCE_BACK:
            # for top y = -1
            self.grid[4, :, -2] = self.grid[2, :, -1]
            self.grid[7, :, -2] = self.grid[5, :, -1] - 1 / 6 * self.mpi.uw
            self.grid[8, :, -2] = self.grid[6, :, -1] + 1 / 6 * self.mpi.uw

    def caluculate_rho_ux_uy(self):
        self.rho = np.sum(self.grid, axis=0)  # sums over each one individually
        self.ux = ((self.grid[1] + self.grid[5] + self.grid[8]) - (self.grid[3] + self.grid[6] + self.grid[7])) / self.rho
        self.uy = ((self.grid[2] + self.grid[5] + self.grid[6]) - (self.grid[4] + self.grid[7] + self.grid[8])) / self.rho

    def collision(self):
        self.grid -= self.mpi.relaxation * (self.grid - self.equilibrium())

    def comunicate(self):
        # Right + Left
        if self.mpi.boundaries_info.apply_right == boundaryStates.COMMUNICATE:
            recvbuf = self.grid[:, -1, :].copy()
            self.comm.Sendrecv(self.grid[:, -2, :].copy(), self.mpi.neighbors.right, recvbuf=recvbuf, sendtag=11, recvtag=12)
            self.grid[:, -1, :] = recvbuf
        if self.mpi.boundaries_info.apply_left == boundaryStates.COMMUNICATE:
            recvbuf = self.grid[:, 0, :].copy()
            self.comm.Sendrecv(self.grid[:, 1, :].copy(), self.mpi.neighbors.left, recvbuf=recvbuf, sendtag=12, recvtag=11)
            self.grid[:, 0, :] = recvbuf

        # Bottom + Top
        if self.mpi.boundaries_info.apply_bottom == boundaryStates.COMMUNICATE:
            recvbuf = self.grid[:, :, 0].copy()
            self.comm.Sendrecv(self.grid[:, :, 1].copy(), self.mpi.neighbors.bottom, recvbuf=recvbuf, sendtag=99, recvtag=98)
            self.grid[:, :, 0] = recvbuf
        if self.mpi.boundaries_info.apply_top == boundaryStates.COMMUNICATE:
            recvbuf = self.grid[:, :, -1].copy()
            self.comm.Sendrecv(self.grid[:, :, -2].copy(), self.mpi.neighbors.top, recvbuf=recvbuf, sendtag=98, recvtag=99)
            self.grid[:, :, -1] = recvbuf

    def periodic_boundary_delta_p(self):
        # movement is assumed form left to right or from top to bottom
        self.caluculate_rho_ux_uy()
        equilibrium = self.equilibrium()
        # loop through the individual sides (locally)
        # left + right
        if self.mpi.boundaries_info.apply_left == boundaryStates.PERIODIC_BOUNDARY:
            equilibrium_in = equilibrium_calculation(self.rho_in, self.ux[-2, :],self.uy[-2, :])
            self.grid[:,0,:] = equilibrium_in + self.grid[:, -2, :] - equilibrium[:, -2, :]

        if self.mpi.boundaries_info.apply_right == boundaryStates.PERIODIC_BOUNDARY:
            equilibrium_out = equilibrium_calculation(self.rho_out, self.ux[1, :], self.uy[1, :])
            self.grid[:, -1, :] = equilibrium_out + self.grid[:, 1, :] - equilibrium[:, 1, :]

        # Top + bottom
        if self.mpi.boundaries_info.apply_top == boundaryStates.PERIODIC_BOUNDARY:
            equilibrium_in = equilibrium_calculation(self.rho_in, self.ux[:, -2], self.uy[:, -2])
            self.grid[:, :, 0] = equilibrium_in + self.grid[:, :, -2] - equilibrium[:, :, -2]

        if self.mpi.boundaries_info.apply_bottom == boundaryStates.PERIODIC_BOUNDARY:
            equilibrium_out = equilibrium_calculation(self.rho_out, self.ux[:, 1], self.uy[:, 1])
            self.grid[:, -1, :] = equilibrium_out + self.grid[:, :, 1] - equilibrium[:, :, 1]

    def collapse_data(self):
        # process 0 gets the data and does the visualization
        if self.mpi.rank == 0:
            original_x = self.mpi.size_x - 2  # ie the base size of the grid on that the
            original_y = self.mpi.size_y - 2  # calculation ran without extra cells for synchron +
            # write the own stuff into it first
            self.full_grid[:, 0:original_x, 0:original_y] = self.grid[:, 1:-1, 1:-1]
            temp = np.zeros((9, original_x, original_y))
            for i in range(1, self.mpi.size):
                self.comm.Recv(temp, source=i, tag=i)
                # determine start end endpoints to copy to in the grid
                copy_start_x = 0 + original_x * self.mpi.base_grid_x
                copy_end_x = original_x + original_x * self.mpi.base_grid_x
                copy_start_y = 0 + original_y * self.mpi.base_grid_y
                copy_end_y = original_y + original_y * self.mpi.base_grid_y
                # copy
                self.full_grid[:, copy_start_x:copy_end_x, copy_start_y:copy_end_y] = temp
            #
            self.grid = self.full_grid
        # all the others send to p0
        else:
            self.comm.Send(self.grid[:, 1:-1, 1:-1].copy(), dest=0, tag=self.mpi.rank)

    def plotter_stream(self):
        # plot
        if self.mpi.rank == 0:
            print("Making Image")
            # recalculate ux and uy
            # change the original grid to the full one
            self.caluculate_rho_ux_uy()
            # acutal plot

            x = np.arange(0, self.mpi.base_grid_x)
            y = np.arange(0, self.mpi.base_grid_y)
            X, Y = np.meshgrid(x, y)
            speed = np.sqrt(self.ux.T ** 2 + self.uy.T ** 2)
            # plot
            # plt.streamplot(X,Y,full_ux.T,full_uy.T)
            plt.streamplot(X, Y, self.ux.T, self.uy.T, color=speed, cmap=plt.cm.jet)
            ax = plt.gca()
            ax.set_xlim([0, self.mpi.base_grid_x + 1])
            ax.set_ylim([0, self.mpi.base_grid_y + 1])
            plt.title(self.title)
            plt.xlabel("x-Position")
            plt.ylabel("y-Position")
            fig = plt.colorbar()
            fig.set_label("Velocity u(x,y,t)", rotation=270, labelpad=15)
            savestring = "slidingLidmpi" + str(self.mpi.size) + ".png"
            plt.savefig(savestring)
            plt.show()

        if self.mpi.rank == 0:
            savestring = "slidingLidmpi" + str(self.mpi.size) + ".txt"
            f = open(savestring, "w")
            totaltime = time.time() - self.time
            f.write(str(totaltime))
            f.close()

    def plotter_simple(self):
        # visualize
        delta = 2.0 * rho_diff / self.mpi.size_x / self.shear_viscosity / 2.
        y = np.linspace(0, self.mpi.size_y, self.mpi.size_y + 1) + 0.5
        u_analytical = delta * y * (self.mpi.size_y - y) / 3.
        plt.plot(u_analytical[:-1], label='Analytical')
        # plt.plot(u_analytical, label='analytical')
        number_of_cuts_in_x = 2
        for i in range(1, number_of_cuts_in_x):
            point = int(i * self.mpi.size_x / number_of_cuts_in_x)
            plt.plot(self.ux[point, 1:-1], label="Calculated")
        print(len(self.ux[25, 1:-1]))
        plt.legend()
        plt.xlabel('Position in cross section')
        plt.ylabel('Velocity')
        plt.title('Pouisuelle flow (Gridsize 100x52, $\omega = 0.5$, $\\nabla \\rho = 0.002$)')
        savestring = "PouisuelleFlow.png"
        plt.savefig(savestring)
        plt.show()


tun = obstacleWindTunnel(steps=0,re=1100,base_length_x=100,base_length_y=50,uw = 0,
                         boundary_state_left= boundaryStates.PERIODIC_BOUNDARY,
                         boundary_state_right=boundaryStates.PERIODIC_BOUNDARY,
                         boundary_state_top=boundaryStates.BAUNCE_BACK,
                         boundary_state_bottom = boundaryStates.BAUNCE_BACK,
                         title = "Work in progress")
tun.run()