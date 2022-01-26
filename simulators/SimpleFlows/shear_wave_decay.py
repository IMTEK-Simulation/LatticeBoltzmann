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
implement the shear wave decay should orient myself fully to the course now
needs collsision, streaming, equilibrium
if i understood this corretly i just give the sim a random velocity in the middle that is sinlike?!
'''
# imports
import numpy as np
import matplotlib.pyplot as plt

# initial variables and sizes
steps = 2000
size_x = 200
size_y = 200
amplitude = 1
periode = 1
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
def shear_wave_decay():
    print("Shear Wave Decay")

    # initizlize the gird
    rho = np.ones((size_x, size_y))
    ux = np.zeros((size_x, size_y))
    uy = np.zeros((size_x, size_y))
    grid = equilibrium(rho, ux, uy)

    # get in the shear wave
    # np sin?
    shear_wave = amplitude * np.sin(periode*(np.linspace(-np.pi,np.pi,size_y)))
    ux[int(size_x/2),:] = shear_wave

    # loop
    for i in range(steps):
        stream(grid)
        rho,ux,uy = caluculate_rho_ux_uy(grid)
        collision(grid,rho,ux,uy)

    # visualize
    # visualize amplitude response?!
    plt.plot(ux[int(size_x/2),:])
    plt.show()



# call
shear_wave_decay()
