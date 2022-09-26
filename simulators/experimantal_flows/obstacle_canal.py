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
import time
import matplotlib.pyplot as plt

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
        self.get_postions_out_of_rank_size_quadratic()
        print(self.pos_x,self.pos_y)
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

    def get_postions_out_of_rank_size_quadratic(self):
        ##
        # assume to be quadratic
        edge_lenght = int(np.sqrt(self.size))
        ###
        self.pos_x = self.rank % edge_lenght
        self.pos_y = self.rank // edge_lenght
        print(self.pos_x, self.pos_y)

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
        print(info)
        self.boundaries_info = info


class obstacleWindTunnel:
    def __init__(self,steps,re,base_length,uw):
        self.time = 0
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
            exit(Exception)
        self.mpi.fill_mpi_struct_fields(rank = self.comm.Get_rank(),
                                        size = size,
                                        max_x = rank_in_one_direction,
                                        max_y = rank_in_one_direction,
                                        base_grid=base_length,
                                        relaxation= relaxation,
                                        steps=steps,
                                        uw = uw)
        # grid type variables
        self.rho = np.ones((self.mpi.size_x, self.mpi.size_y))
        self.ux = np.zeros((self.mpi.size_x, self.mpi.size_y))
        self.uy = np.zeros((self.mpi.size_x, self.mpi.size_y))
        self.grid = self.equilibrium()
        self.full_gird = np.zeros(2)

    def run(self):
        self.time = time.time()
        print(self.mpi)
        # iterations
        for i in range(self.mpi.steps):
            self.stream()
            self.bounce_back_choosen()
            self.caluculate_rho_ux_uy()
            self.collision()
            self.comunicate()
    #
        self.collapse_data()
        self.plotter()

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
        if self.mpi.boundaries_info.apply_right:
            # right so x = -1
            self.grid[3, -2, :] = self.grid[1, -1, :]
            self.grid[6, -2, :] = self.grid[8, -1, :]
            self.grid[7, -2, :] = self.grid[5, -1, :]
        if self.mpi.boundaries_info.apply_left:
            # left so x = 0
            self.grid[1, 1, :] = self.grid[3, 0, :]
            self.grid[5, 1, :] = self.grid[7, 0, :]
            self.grid[8, 1, :] = self.grid[6, 0, :]

        # Bottom + Top
        if self.mpi.boundaries_info.apply_bottom:
            # for bottom y = 0
            self.grid[2, :, 1] = self.grid[4, :, 0]
            self.grid[5, :, 1] = self.grid[7, :, 0]
            self.grid[6, :, 1] = self.grid[8, :, 0]
        if self.mpi.boundaries_info.apply_top:
            # for top y = -1
            self.grid[4, :, -2] = self.grid[2, :, -1]
            self.grid[7, :, -2] = self.grid[5, :, -1] - 1 / 6 * self.mpi.uw
            self.grid[8, :, -2] = self.grid[6, :, -1] + 1 / 6 * self.mpi.uw

    def caluculate_rho_ux_uy(self):
        self.rho = np.sum(self.grid, axis=0)  # sums over each one individually
        self.ux = ((self.grid[1] + [5] + self.grid[8]) - (self.grid[3] + self.grid[6] + self.grid[7])) / self.rho
        self.uy = ((self.grid[2] + self.grid[5] + self.grid[6]) - (self.grid[4] + self.grid[7] + self.grid[8])) / self.rho

    def collision(self):
        self.grid -= self.mpi.relaxation * (self.grid - self.equilibrium())

    def comunicate(self):
        # Right + Left
        if not self.mpi.boundaries_info.apply_right:
            recvbuf = self.grid[:, -1, :].copy()
            self.comm.Sendrecv(self.grid[:, -2, :].copy(), self.mpi.neighbors.right, recvbuf=recvbuf, sendtag=11, recvtag=12)
            self.grid[:, -1, :] = recvbuf
        if not self.mpi.boundaries_info.apply_left:
            recvbuf = self.grid[:, 0, :].copy()
            self.comm.Sendrecv(self.grid[:, 1, :].copy(), self.mpi.neighbors.left, recvbuf=recvbuf, sendtag=12, recvtag=11)
            self.grid[:, 0, :] = recvbuf

        # Bottom + Top
        if not self.mpi.boundaries_info.apply_bottom:
            recvbuf = self.grid[:, :, 0].copy()
            self.comm.Sendrecv(self.grid[:, :, 1].copy(), self.mpi.neighbors.bottom, recvbuf=recvbuf, sendtag=99, recvtag=98)
            self.grid[:, :, 0] = recvbuf
        if not self.mpi.boundaries_info.apply_top:
            recvbuf = self.grid[:, :, -1].copy()
            self.comm.Sendrecv(self.grid[:, :, -2].copy(), self.mpi.neighbors.top, recvbuf=recvbuf, sendtag=98, recvtag=99)
            self.grid[:, :, -1] = recvbuf

    def collapse_data(self):
        # process 0 gets the data and does the visualization
        if self.mpi.rank == 0:
            self.full_grid = np.ones((9, self.mpi.base_grid, self.mpi.base_grid))
            original_x = self.mpi.size_x - 2  # ie the base size of the grid on that the
            original_y = self.mpi.size_y - 2  # calculation ran
            # write the own stuff into it first
            self.full_grid[:, 0:original_x, 0:original_y] = self.grid[:, 1:-1, 1:-1]
            temp = np.zeros((9, original_x, original_y))
            for i in range(1, self.mpi.size):
                self.comm.Recv(temp, source=i, tag=i)
                # TODO change me
                x, y = 300,300
                # determin start end endpoints to copy to in the grid
                copy_start_x = 0 + original_x * x
                copy_end_x = original_x + original_x * x
                copy_start_y = 0 + original_y * y
                copy_end_y = original_y + original_y * y
                # copy
                self.full_grid[:, copy_start_x:copy_end_x, copy_start_y:copy_end_y] = temp
            #
            self.grid = self.full_grid
        # all the others send to p0
        else:
            self.comm.Send(self.grid[:, 1:-1, 1:-1].copy(), dest=0, tag=self.mpi.rank)

    def plotter(self):
        # plot
        if self.mpi.rank == 0:
            print("Making Image")
            # recalculate ux and uy
            # change the original grid to the full one
            self.caluculate_rho_ux_uy()
            # acutal plot

            x = np.arange(0, self.mpi.base_grid)
            y = np.arange(0, self.mpi.base_grid)
            X, Y = np.meshgrid(x, y)
            speed = np.sqrt(self.ux.T ** 2 + self.uy.T ** 2)
            # plot
            # plt.streamplot(X,Y,full_ux.T,full_uy.T)
            plt.streamplot(X, Y, self.ux.T, self.uy.T, color=speed, cmap=plt.cm.jet)
            ax = plt.gca()
            ax.set_xlim([0, self.mpi.base_grid + 1])
            ax.set_ylim([0, self.mpi.base_grid + 1])
            plt.title("Sliding Lid")
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


tun = obstacleWindTunnel(steps=10,re=1000,base_length=300,uw = 0.1)
tun.run()