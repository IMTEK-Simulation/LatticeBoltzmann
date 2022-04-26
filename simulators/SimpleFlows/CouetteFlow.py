
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
uw = 5
velocity_set = np.array([[0, 1, 0, -1, 0, 1, -1, -1, 1],
                         [0,0,1,0,-1,1,1,-1,-1]]).T
# grids
size = 50
topbottom_boundary = 2
size_x = size                     # 50
size_y = size #+ topbottom_boundary # 52
# initilization of the grids used
grid = np.ones((channels,size_x,size_y))
rho_v = np.zeros((size_x,size_y))
ux_v = np.zeros((size_x,size_y))
uy_v = np.zeros((size_x,size_y))

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
    equilibrium[0] = (2 * rho / 9) * (2 - 3 * uu)
    equilibrium[1] = (rho / 18) * (2 + 6 * ux + 9 * ux * ux - 3 * uu)
    equilibrium[2] = (rho / 18) * (2 + 6 * uy + 9 * uy * uy - 3 * uu)
    equilibrium[3] = (rho / 18) * (2 - 6 * ux + 9 * ux * ux - 3 * uu)
    equilibrium[4] = (rho / 18) * (2 - 6 * uy + 9 * uy * uy - 3 * uu)
    equilibrium[5] = (rho / 36) * (1 + 3 * uxy + 9 * ux * uy + 3 * uu)
    equilibrium[6] = (rho / 36) * (1 - 3 * uxy - 9 * ux * uy + 3 * uu)
    equilibrium[7] = (rho / 36) * (1 - 3 * uxy + 9 * ux * uy + 3 * uu)
    equilibrium[8] = (rho / 36) * (1 + 3 * uxy - 9 * ux * uy + 3 * uu)
    return equilibrium

def equilibrium_on_array(rho,ux,uy):
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



def calculate_velocities_pressure(gridpoint):
    rho = np.sum(gridpoint)
    ux = ((gridpoint[1] + gridpoint[5] + gridpoint[8]) - (gridpoint[3] + gridpoint[6] + gridpoint[7])) / rho
    uy = ((gridpoint[2] + gridpoint[5] + gridpoint[6]) - (gridpoint[4] + gridpoint[7] + gridpoint[8])) / rho
    return rho, ux, uy

def calculate_collision(grid):
    # does everything to apply the collision step, mainly to improve performance
    # big resauces eater is the equilibrium calc
    # calculate all the grid values
    rho = np.sum(grid,axis = 0) # sums over each one individually
    ux = ((grid[1] + grid[5] + grid[8]) - (grid[3] + grid[6] + grid[7])) / rho
    uy = ((grid[2] + grid[5] + grid[6]) - (grid[4] + grid[7] + grid[8])) / rho
    # calculate equilibrium + apply collision
    # grid = grid - relaxation * (grid - equilbrium)
    grid -= relaxation * (grid-equilibrium_on_array(rho,ux,uy))
    return rho,ux,uy



def bounce_back(grid,uw):
    # baunce back without any velocity gain
    # TODO rho Wall missing
    max_size_x = grid.shape[1]-1  # x
    max_size_y = grid.shape[2]-1  # y
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
def slow_calc():
    equlibrium = np.ones((channels, size_x, size_y))
    collision  = np.ones((channels, size_x, size_y))
    for i in range(steps):
        # aquire the values for the pressure and velocities
        # basically no efficiency
        for k in range(size_x):
            for l in range(size_y):
                rho_v[k, l], ux_v[k, l], uy_v[k, l] = calculate_velocities_pressure(grid[:, k, l])
                # calculate the equilibrium-function
                eq = equilibrium(rho_v[k, l], ux_v[k, l], uy_v[k, l])
                # print(eq.shape)
                # print(equlibrium[:,k,l].shape)
                equlibrium[:, k, l] = equilibrium(rho_v[k, l], ux_v[k, l], uy_v[k, l])
                # calculate the collision operator
                collision[:, k, l] = (grid[:, k, l] - equlibrium[:, k, l])
        #
        collision = collision * relaxation
        # apply collision
        grid = grid - collision
        # stream
        stream(grid)
        # baounce back
        bounce_back(grid, uw)
        # next step
    # print(grid)

def fast_calc(rho,ux,uy):
    for i in range(steps):
        rho, ux, uy = calculate_collision(grid)
        # stream
        stream(grid)
        # baounce back
        bounce_back(grid,uw)
        # next step
    return rho, ux, uy

''' visualization '''
#quiver?!
rho_v, ux_v, uy_v = fast_calc(rho_v,ux_v,uy_v)
x = np.arange(0,size_x)
y = np.arange(0,size_y)
X,Y = np.meshgrid(x,y)
# UX, UY = np.meshgrid(ux, uy)
plt.streamplot(X,Y,ux_v,uy_v)
#print(ux)
plt.show()






