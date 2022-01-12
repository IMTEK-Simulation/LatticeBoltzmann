
# Todo: remove us
'''
To get a basic feel for LB-Sim implement the  Poiseuille Flow between two plates


'''
'''
Remider CodingRules: 
Zeilenumbruch bei Spalte 120
Modulname, Klassennamen als CamelCase
Variablennamen, Methodennamen, Funktionsnamen mit unter_strichen
Bitte nicht CamelCase und Unterstriche mischen
'''
# imports
import numpy as np
import matplotlib.pyplot as plt
# Bring back stuff
relaxation = 0.5
velocity_set = np.array([[0, 1, 0, -1, 0, 1, -1, -1, 1],
                         [0,0,1,0,-1,1,1,-1,-1]]).T

def equilibrium_on_array(rho,ux,uy):
    '''
    Calculates the equilibrium function for the whole array at once
    Parameters
    ----------
    rho
    ux
    uy

    Returns
    -------

    '''
    uxy = 3 * (ux + uy)
    uu =  3 * (ux * ux + uy * uy)
    ux_6 = 6*ux
    uy_6 = 6*uy
    uxx_9 = 9 * ux*ux
    uyy_9 = 9 * uy*uy
    uxy_9 = 9 * ux*uy
    return np.array([(2 * rho / 9) * (2 - uu),
                     (rho / 18) * (2 + ux_6 + uxx_9 - uu),
                     (rho / 18) * (2 + uy_6 + uyy_9 - uu),
                     (rho / 18) * (2 - ux_6 + uxx_9 - uu),
                     (rho / 18) * (2 - uy_6 + uyy_9 - uu),
                     (rho / 36) * (1 + uxy + uxy_9 + uu),
                     (rho / 36) * (1 - uxy - uxy_9 + uu),
                     (rho / 36) * (1 - uxy + uxy_9 + uu),
                     (rho / 36) * (1 + uxy - uxy_9 + uu)])


def calculate_collision(grid):
    '''
    Performs the collision step and also calculated rho, ux, uy in the same step
    Parameters
    ----------
    grid

    Returns
    -------
    rho, ux, uy
    '''
    rho = np.sum(grid,axis = 0) # sums over each one individually
    ux = ((grid[1] + grid[5] + grid[8]) - (grid[3] + grid[6] + grid[7])) / rho
    uy = ((grid[2] + grid[5] + grid[6]) - (grid[4] + grid[7] + grid[8])) / rho
    # calculate equilibrium + apply collision
    # grid = grid - relaxation * (grid - equilbrium)
    grid -= relaxation * (grid-equilibrium_on_array(rho,ux,uy))
    return rho,ux,uy

def stream(grid):
    '''
    Performs the streaming step in place
    Parameters
    ----------
    grid

    Returns
    -------

    '''
    for i in range(1,9):
        grid[i] = np.roll(grid[i],velocity_set[i], axis = (0,1))