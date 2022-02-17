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

# Todo: remove me
'''
my todos:
porgramm the sliding lid but now do some domain decomposition and part the computation domain 
needs streaming, collsion, equiblrium, bounce back, mpi-decomposition
'''

# imports
import numpy as np
from dataclasses import dataclass
from mpi4py import MPI
import matplotlib.pyplot as plt

# initial variables and sizess
re = 1000
base_lenght = 300
steps = 100000
uw = 0.1
size_x = base_lenght
size_y = base_lenght
relaxation = (2*re)/(6*base_lenght*uw+re)
velocity_set = np.array([[0, 1, 0, -1, 0, 1, -1, -1, 1],
                         [0,0,1,0,-1,1,1,-1,-1]]).T

# pack stuff so its together
@dataclass
class boundariesApplied:
    # left right top bottom
    apply_left : bool = False
    apply_right: bool = False
    apply_top:bool = False
    apply_bottom: bool = False

@dataclass
class mpiPackageStructure:
    # apply for boundaries
    boundaries_info: boundariesApplied = (False,False,False,False)
    # sizes and position in the whole grid
    size_x: int = -1
    size_y: int = -1
    pos_x : int = -1
    pos_y : int = -1

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


def collision(grid,rho,ux,uy):
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


def bounce_back_choosen(grid,uw,apply_left,apply_right,apply_top,apply_bottom):
    # modification for the bounce back for the bigger grids
    #### Left + Right
    if apply_right:
        # right so x = 0
        grid[1, 1, 1:-1] = grid[3, 0, 1:-1]
        grid[5, 1, 1:-1] = grid[7, 0, 1:-1]
        grid[8, 1, 1:-1] = grid[6, 0, 1:-1]
    if apply_left:
        # left so x = -1
        grid[3, -2, 1:-1] = grid[1, -1, 1:-1]
        grid[6, -2, 1:-1] = grid[8, -1, 1:-1]
        grid[7, -2, 1:-1] = grid[5, -1, 1:-1]
    #### TOP + Bottom
    if apply_bottom:
        # for bottom y = 0
        grid[2, 1:-1, 1] = grid[4, 1:-1, 0]
        grid[5, 1:-1, 1] = grid[7, 1:-1, 0]
        grid[6, 1:-1, 1] = grid[8, 1:-1, 0]
    if apply_top:
        # for top y = -1
        grid[4, 1:-1, -2] = grid[2, 1:-1, -1]
        grid[7, 1:-1, -2] = grid[5, 1:-1, -1] - 1 / 6 * uw
        grid[8, 1:-1, -2] = grid[6, 1:-1, -1] + 1 / 6 * uw



# body
def sliding_lid_mpi():
    print("Sliding Lid")

    # initizlize the gird
    rho = np.ones((size_x+2,size_y+2))
    ux = np.zeros((size_x+2,size_y+2))
    uy = np.zeros((size_x+2,size_y+2))
    grid = equilibrium(rho,ux,uy)

    # loop
    for i in range(steps):
        stream(grid)
        bounce_back(grid,uw)
        rho, ux, uy = caluculate_rho_ux_uy(grid)
        collision(grid,rho,ux,uy)


# call
# sliding_lid_mpi()
