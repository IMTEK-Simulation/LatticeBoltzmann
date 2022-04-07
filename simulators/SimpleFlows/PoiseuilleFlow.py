
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
import time

# global variables
relaxation = 0.5
size_x = 100
size_y = 50
velocity_set = np.array([[0, 1, 0, -1, 0, 1, -1, -1, 1],
                         [0,0,1,0,-1,1,1,-1,-1]]).T
rho_null = 1
diff = 0.001

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
    '''
    Performs the collision step and also calculated rho, ux, uy in the same step
    Parameters
    ----------
    grid

    Returns
    -------
    rho, ux, uy
    '''

    # calculate equilibrium + apply collision
    # grid = grid - relaxation * (grid - equilbrium)
    grid -= relaxation * (grid-equilibrium_on_array(rho,ux,uy))

def caluculate_real_values(grid):
    '''
    Calculates rho, ux, uy
    Parameters
    ----------
    grid

    Returns
    -------

    '''
    rho = np.sum(grid, axis=0)  # sums over each one individually
    ux = ((grid[1] + grid[5] + grid[8]) - (grid[3] + grid[6] + grid[7])) / rho
    uy = ((grid[2] + grid[5] + grid[6]) - (grid[4] + grid[7] + grid[8])) / rho
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

def bounce_back(grid,uw):
    '''
    Perfomrs the bounce back
    Parameters
    ----------
    grid
    uw

    Returns
    -------

    '''
    # baunce back without any velocity gain
    # for bottom y = 0
    grid[2, :, 1] = grid[4, :, 0]
    grid[5, :, 1] = np.roll(grid[7, :, 0],1)
    grid[6, :, 1] = np.roll(grid[8, :, 0],-1)
    # for top y = max_size_y
    grid[4, :, -2] = grid[2, :, -1]
    grid[7, :, -2] = np.roll(grid[5, :, -1],1) - 1 / 6 * uw
    grid[8, :, -2] = np.roll(grid[6, :, -1],-1) + 1 / 6 * uw

def periodic_boundary_with_pressure_variations(grid,rho_in,rho_out):
    '''

    Parameters
    ----------
    grid
    rho_in
    rho_out

    Returns
    -------

    '''
    # get all the values
    rho, ux, uy = caluculate_real_values(grid)
    equilibrium = equilibrium_on_array(rho, ux, uy)
    ##########
    equilibrium_in = equilibrium_on_array(rho_in, ux[-2,:], uy[-2, :])
    # inlet 1,5,8
    grid[:, 0, :] = equilibrium_in + (grid[:, -2, :] - equilibrium[:, -2, :])

    # outlet 3,6,7
    equilibrium_out = equilibrium_on_array(rho_out, ux[1, :], uy[1, :])
    # check for correct sizes
    grid[:, -1, :] = equilibrium_out + (grid[:, 1, :] - equilibrium[:, 1, :])



#########
def couette_flow():
    # main code
    print("couette Flow")
    steps = 4000
    uw = 1

    # initialize
    rho = np.ones((size_x,size_y+2))
    ux = np.zeros((size_x, size_y + 2))
    uy = np.zeros((size_x,size_y + 2))
    grid = equilibrium_on_array(rho,ux,uy)

    # loop
    for i in range(steps):
        rho, ux, uy = caluculate_real_values(grid)
        collision(grid,rho,ux,uy)
        stream(grid)
        bounce_back(grid,uw)

    # visualize
    x = np.arange(0,size_x)
    y = np.arange(0,size_y)
    X,Y = np.meshgrid(x,y)
    #plt.streamplot(X,Y,ux[:,1:51],uy[:,1:51])
    #plt.show()
    # stolen couette flowl code ;)
    plt.plot(ux[5,1:-2])
    plt.xlabel('Position in cross section')
    plt.ylabel('Velocity [m/s]')
    plt.title('Couette flow')
    savestring = "CouetteFlow.png"
    plt.savefig(savestring)
    plt.show()

def poiseuille_flow():
    # main code
    print("Poiseuille Flow")
    uw = 0.001
    steps = 4000 # crashes 4533
    rho_in = rho_null+diff
    rho_out = rho_null-diff
    # initialize
    rho = np.ones((size_x+2, size_y + 2))
    ux = np.zeros((size_x+2, size_y + 2))
    uy = np.zeros((size_x+2, size_y + 2))
    grid = equilibrium_on_array(rho, ux, uy)

    # loop
    for i in range(steps):
        periodic_boundary_with_pressure_variations(grid,rho_in,rho_out)
        stream(grid)
        bounce_back(grid, uw)
        rho, ux, uy = caluculate_real_values(grid)
        collision(grid, rho, ux, uy)

    # visualize
    x = np.arange(0, size_x+2)
    y = np.arange(0, size_y+2)
    X, Y = np.meshgrid(x, y)
    #plt.streamplot(X,Y,ux[:,1:51],uy[:,1:51])
    #plt.show()
    # stolen couette flowl code ;)
    number_of_cuts_in_x = 2
    for i in range(1,number_of_cuts_in_x):
        point = int(i*size_x/number_of_cuts_in_x)
        plt.plot(ux[point, 1:-1],label = "x = {}".format(point) )
    # plt.legend()
    plt.xlabel('Position in cross section')
    plt.ylabel('Velocity [m/s]')
    plt.title('Pouisuelle flow')
    savestring = "PouisuelleFlow.png"
    plt.savefig(savestring)
    plt.show()

def pouisuelle_flow_fancy():
    # main code
    print("Poiseuille Flow fancy")
    runs = 10
    uw = 0.000
    steps = 5000  # crashes 4533
    rho_in = rho_null + diff
    rho_out = rho_null - diff
    # initialize
    rho = np.ones((size_x + 2, size_y + 2))
    ux = np.zeros((size_x + 2, size_y + 2))
    uy = np.zeros((size_x + 2, size_y + 2))
    grid = equilibrium_on_array(rho, ux, uy)

    # plot related stuff
    x = np.arange(0, size_x)
    y = np.arange(0, size_y)
    X, Y = np.meshgrid(x, y)

    # loop
    for k in range(runs):
        uw += 0.001
        uw = round(uw,3)
        for i in range(steps):
            periodic_boundary_with_pressure_variations(grid, rho_in, rho_out)
            stream(grid)
            bounce_back(grid, uw)
            rho, ux, uy = caluculate_real_values(grid)
            collision(grid, rho, ux, uy)
            point = int(size_x/2)
            #
        plt.plot(ux[point, 1:-1], label="uw = {}".format(uw))
    ## end plot
    plt.legend()
    plt.xlabel('Position in cross section')
    plt.ylabel('Velocity [m/s]')
    plt.title('Pouisuelle flow with diffrent u-Walls')
    savestring = "PouisuelleFlowFancy.png"
    plt.savefig(savestring)
    plt.show()





def constant_velocity_in_boundary_flow():
    print("Constant thingi")
    # constants
    uw = 0
    # cant really go higher it just crashes :(
    steps = 3000

    # initilazion
    rho = np.ones((size_x + 2, size_y + 2))
    ux = np.zeros((size_x + 2, size_y + 2))
    uy = np.zeros((size_x + 2, size_y + 2))
    grid = equilibrium_on_array(rho, ux, uy)

    # propagation
    for i in range(steps):
        stream(grid)
        bounce_back(grid, uw)
        rho, ux, uy = caluculate_real_values(grid)
        ux[0, :] = 0.02
        ux[-1, :] = 0.02
        collision(grid, rho, ux, uy)

    # visiulation
    x = np.arange(0, size_x)
    y = np.arange(0, size_y)
    X, Y = np.meshgrid(x, y)
    plt.streamplot(X,Y,ux[:,1:51],uy[:,1:51])
    plt.show()
    # stolen couette flowl code ;)
    plt.plot(ux[int(1+size_x/2), 1:-2], color = "green")
    plt.xlabel('Position in cross section')
    plt.ylabel('velocity')
    plt.title('Constant velocity')
    plt.show()

####
# function
# couette_flow()
poiseuille_flow()
pouisuelle_flow_fancy()
# constant_velocity_in_boundary_flow()


