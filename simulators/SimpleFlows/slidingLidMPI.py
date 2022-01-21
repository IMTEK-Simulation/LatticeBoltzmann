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
import matplotlib.pyplot as plt

# initial variables and sizes
size_x = 50
size_y = 50
relaxation = 0.5
velocity_set = np.array([[0, 1, 0, -1, 0, 1, -1, -1, 1],
                         [0,0,1,0,-1,1,1,-1,-1]]).T

# main functions
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

# main body
def sliding_lid_with_MPI():
    print("Sliding Lid with MPI")

# caller
sliding_lid_with_MPI()
