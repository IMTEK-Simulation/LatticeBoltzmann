
#Todo: remove us
'''
To get a basic feel here ill implement the Couette Flow between two plates
-----
some general remarks
ill do this one here in the most explixite form imaginable, in the Poiseuille Flow ill start doing optimizations
but as I am learning, ill do this first try the "dumb" way. I am aware that withhin the project there allready good
implementations of basically all the operations. But for the first try I will ignore them and will use my
implementations. (In the second try PoiseuilleFlow ill use them thou.) This should motivate my approach here. Btw in
D2Q9.

'''
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

''' Imports '''
import numpy as np
import matplotlib.pyplot as plt

''' Setup '''
print("Couetteflow")

# velocity_set
channels = 9
relaxation = 0.5
uw = 1
velocity_set = np.array([[0, 1, 0, -1, 0, 1, -1, -1, 1],
                         [0,0,1,0,-1,1,1,-1,-1]]).T
# grids
size_x = 25
size_y = 25
grid = np.ones((channels,size_x,size_y))
collision = np.zeros((channels,size_x,size_y))
rho = np.zeros((size_x,size_y))
ux = np.zeros((size_x,size_y))
uy = np.zeros((size_x,size_y))

# steps
steps = 5000
''' functions '''


def stream(grid):
    for i in range(1,9):
        grid[i] = np.roll(grid[i],velocity_set[i], axis = (0,1))


def equilibrium(rho, ux, uy):
    equilibrium = np.zeros(9)
    uxy = ux + uy
    uu = ux * ux + uy * uy
    equilibrium[0] = 2 / 9 * rho * (2 - 3 * uu)
    equilibrium[1] = rho / 18 * (2 + 6 * ux + 9 * ux * ux - 3 * uu)
    equilibrium[2] = rho / 18 * (2 + 6 * uy + 9 * uy * uy - 3 * uu)
    equilibrium[3] = rho / 18 * (2 - 6 * ux + 9 * ux * ux - 3 * uu)
    equilibrium[4] = rho / 18 * (2 - 6 * uy + 9 * uy * uy - 3 * uu)
    equilibrium[5] = rho / 36 * (1 + 3 * uxy + 9 * ux * uy + 3 * uu)
    equilibrium[6] = rho / 36 * (1 - 3 * uxy - 9 * ux * uy + 3 * uu)
    equilibrium[7] = rho / 36 * (1 - 3 * uxy + 9 * ux * uy + 3 * uu)
    equilibrium[8] = rho / 36 * (1 + 3 * uxy - 9 * ux * uy + 3 * uu)


def calculate_velocities_pressure(gridpoint):
    rho = np.sum(gridpoint)
    ux = ((gridpoint[1] + gridpoint[5] + gridpoint[8]) - (gridpoint[3] + gridpoint[6] + gridpoint[7])) / rho
    uy = ((gridpoint[2] + gridpoint[5] + gridpoint[6]) - (gridpoint[4] + gridpoint[7] + gridpoint[8])) / rho
    return rho, ux, uy

def calculate_collision():
    pass

def bounce_back(grid,uw):
    # baunce back without any velocity gain
    max_size_x = grid.shape[1]-1  # x
    max_size_y = grid.shape[2]-1  # y
    # right so x = 0
    grid[1, 1, :] = grid[3, 0, :]
    grid[5, 1, :] = grid[7, 0, :]
    grid[8, 1, :] = grid[6, 0, :]
    grid[3, 0, :] = 0
    grid[7, 0, :] = 0
    grid[6, 0, :] = 0
    # left so x = max_size_x
    grid[3, max_size_x - 1, :] = grid[1, max_size_x, :]
    grid[6, max_size_x - 1, :] = grid[8, max_size_x, :]
    grid[7, max_size_x - 1, :] = grid[5, max_size_x, :]
    grid[1, max_size_x, :] = 0
    grid[8, max_size_x, :] = 0
    grid[5, max_size_x, :] = 0
    # for bottom y = 0
    grid[2, :, 1] = grid[4, :, 0]
    grid[5, :, 1] = grid[7, :, 0]
    grid[6, :, 1] = grid[8, :, 0]
    grid[4, :, 0] = 0
    grid[7, :, 0] = 0
    grid[8, :, 0] = 0
    # for top y = max_size_y
    grid[4, :, max_size_y - 1] = grid[2, :, max_size_y]
    grid[7, :, max_size_y - 1] = grid[5, :, max_size_y] - 1 / 6 * uw
    grid[8, :, max_size_y - 1] = grid[6, :, max_size_y] + 1 / 6 * uw
    grid[2, :, max_size_y] = 0
    grid[5, :, max_size_y] = 0
    grid[6, :, max_size_y] = 0


''' body '''
for i in range(steps):
    # aquire the values for the pressure and velocities
    for k in range(size_x-1):
        for l in range(size_y-1):
            rho[k,l], ux[k,l], uy[k,l] = calculate_velocities_pressure(grid[:,k,l])
            # calculate the equilibrium-function
            collision[:,k,l] = equilibrium(rho[k,l],ux[k,l], uy[k,l])
            # calculate the collision operator
            collision[:,k,l] = (grid[:,k,l]-collision[:,k,l])
    #
    collision = collision/relaxation
    # apply collision
    grid = grid - collision
    # stream
    stream(grid)
    # baounce back
    bounce_back(grid,uw)
    # next step

''' visualization '''
#quiver?!
x = np.arange(0,size_x)
y = np.arange(0,size_y)
plt.quiver(x, y, ux, uy, scale=1)
plt.show()






