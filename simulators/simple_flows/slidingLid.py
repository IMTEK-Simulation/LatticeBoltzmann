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

# imports
import numpy as np
import matplotlib.pyplot as plt

# initial variables and sizess
re = 1000
base_lenght = 300
steps = 100000
uw = 0.1
size_x = base_lenght + 2
size_y = base_lenght + 2
relaxation = (2*re)/(6*base_lenght*uw+re)
velocity_set = np.array([[0, 1, 0, -1, 0, 1, -1, -1, 1],
                         [0,0,1,0,-1,1,1,-1,-1]]).T
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
    # TODO rho_wall
    # for bottom y = 0
    grid[2, 1:-1, 1] = grid[4, 1:-1, 0]
    grid[5, 1:-1, 1] = grid[7, 1:-1, 0]
    grid[6, 1:-1, 1] = grid[8, 1:-1, 0]
    # for top y = -1
    grid[4, 1:-1, -2] = grid[2, 1:-1, -1]
    grid[7, 1:-1, -2] = grid[5, 1:-1, -1] - 1 / 6 * uw
    grid[8, 1:-1, -2] = grid[6, 1:-1, -1] + 1 / 6 * uw

# body
def sliding_lid():
    print("Sliding Lid")

    # initizlize the gird
    rho = np.ones((size_x,size_y))
    ux = np.zeros((size_x,size_y))
    uy = np.zeros((size_x,size_y))
    grid = equilibrium(rho,ux,uy)

    # loop
    for i in range(steps):
        stream(grid)
        bounce_back(grid,uw)
        rho, ux, uy = caluculate_rho_ux_uy(grid)
        collision(grid,rho,ux,uy)

    # print(grid[2,0,:])
    # visualize
    # values
    x = np.arange(0, base_lenght)
    y = np.arange(0, base_lenght)
    X, Y = np.meshgrid(x, y)
    speed = np.sqrt(ux[1:-1,1:-1].T ** 2 + uy[1:-1,1:-1].T ** 2)
    print(speed.shape)
    # plot
    plt.streamplot(X,Y,ux[1:-1,1:-1].T,uy[1:-1,1:-1].T, color = speed, cmap= plt.cm.jet)
    ax = plt.gca()
    ax.set_xlim([0, 301])
    ax.set_ylim([0, 301])
    titleString = "Sliding Lid (Gridsize " + "{}".format(size_x) + "x" + "{}".format(size_y)
    titleString += ",  $\\omega$ = {:.2f}".format(relaxation) + ", steps = {}".format(steps) + ")"
    plt.title(titleString)
    plt.xlabel("x-Position")
    plt.ylabel("y-Position")
    fig = plt.colorbar()
    fig.set_label("Velocity u(x,y,t)", rotation=270,labelpad = 15)
    plt.savefig('temp.png')
    plt.show()
    '''
    plt.plot(ux[int(1 + size_x / 2), 1:-1], color="green")
    plt.xlabel('Position in cross section')
    plt.ylabel('velocity')
    plt.title('Constant velocity')
    plt.show()
    '''



# call
sliding_lid()
