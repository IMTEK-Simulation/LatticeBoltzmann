'''
The purpose of this file is to help develop the simpleFlows
    -> mainly test to get an understanding and not hardcore unittests to show sth works
'''

import unittest
import numpy as np

'''
Tester for Streaming
'''


class testsInStreaming(unittest.TestCase):
    def test_basic_np_roll(self):
        # make a grid with a lenght and write the first element to one-> then roll the element through
        lenght = 9
        grid = np.zeros((1, 1, lenght))
        grid[:, :, 0] = 1
        # roll through the array
        for i in range(lenght - 1):
            grid = np.roll(grid, (1, 0))
        # look weather the last elemnt is 1
        self.assertEqual(grid[0, 0, lenght - 1], 1)  # add assertion here

    def test_multidimensional_roll(self):
        # now with more channels here the 4 main channels
        lenght = 9
        base = 5
        grid = np.zeros((base, 1, lenght))
        # print(grid.shape)
        grid[:, :, 0] = 1
        # roll through but ignore channel 0 !!
        for i in range(lenght):
            '''
            the np.roll rolls through the 1D-array from start to finish 
            it doesnt really matter if i write (1,0) or (0,1) as the roll still performs the same 
            '''
            grid[1, :, :, ] = np.roll(grid[1, :, :, ], (1, 0))
            grid[2, :, :, ] = np.roll(grid[2, :, :, ], (
            0, 1))  # this doesnt really change the rolling behaviour still rolls throu the array in 1D
            grid[3, :, :, ] = np.roll(grid[3, :, :, ], (-1, 0))
            grid[4, :, :, ] = np.roll(grid[4, :, :, ], (0, -1))

        # check the values
        for i in range(base - 1):  # from 1 to 4
            self.assertEqual(grid[i + 1, 0, 0], 1)  # roll through the whole array should be 1

    def test_full_multidimensional(self):
        # basic variables
        lenght = 9
        base = 9
        grid = np.zeros((base, 1, lenght))
        velocity_set = np.array([[0, 1, 0, -1, 0, 1, -1, -1, 1],
                                 [0, 0, 1, 0, -1, 1, 1, -1, -1]]).T
        grid[:, :, 0] = 1
        ###
        for i in range(lenght):
            for j in range(base - 1):
                grid[j + 1, :, :, ] = np.roll(grid[j + 1, :, :, ], velocity_set[j + 1])
                # do a follow up check in the middle
            # print(grid[5, :, :, ])
            if i == 3:
                # elements are now in the middle for the principal axis
                self.assertEqual(grid[2, 0, 4], 1)
                self.assertEqual(grid[3, 0, 5], 1)

        # check the values
        for i in range(base - 1):  # from 1 to 4
            self.assertEqual(grid[i + 1, 0, 0], 1)  # roll through the whole array should be 1

    def test_final_implementation(self):
        # basic variables
        lenght = 9
        base = 9
        grid = np.zeros((base, 1, lenght))
        velocity_set = np.array([[0, 1, 0, -1, 0, 1, -1, -1, 1],
                                 [0, 0, 1, 0, -1, 1, 1, -1, -1]]).T
        grid[:, :, 0] = 1
        ####
        '''
        basic idea is to move them constant from one point in the array to the next
        not sure about it as the original implementation behaves a bit different
        this works like my intuition -> in one step the content of the previous channel gets written into the next
        '''
        for i in range(lenght):
            for j in range(1, 5):
                # print(velocity_set[j])
                grid[j] = np.roll(grid[j], velocity_set[j])
            for j in range(5, 9):
                grid[j] = np.roll(grid[j], velocity_set[j], axis=(0, 1))
            # middle test for values
            # print(grid[5])
            if i == 3:
                # print(grid[4, :, :, ])
                # first value doesnt move
                self.assertEqual(grid[0, 0, 0], 1)
                self.assertEqual(grid[1, 0, 4], 1)
                self.assertEqual(grid[2, 0, 4], 1)
                self.assertEqual(grid[3, 0, 5], 1)
                self.assertEqual(grid[4, 0, 5], 1)
                self.assertEqual(grid[5, 0, 4], 1)
                self.assertEqual(grid[6, 0, 4], 1)
                self.assertEqual(grid[7, 0, 5], 1)
                self.assertEqual(grid[8, 0, 5], 1)

        # check the values at the end
        for i in range(base - 1):  # from 1 to 4
            self.assertEqual(grid[i + 1, 0, 0], 1)  # roll through the whole array should be 1

    def test_final_implementation_other_axis(self):
        '''
        Similar to the previous test, but this time x and y are interchanged
        might do a full 2D test, but the test cases are not that trivial i think so no idea
        '''
        # basic variables
        lenght = 9
        base = 9
        grid = np.zeros((base, lenght, 1))
        velocity_set = np.array([[0, 1, 0, -1, 0, 1, -1, -1, 1],
                                 [0, 0, 1, 0, -1, 1, 1, -1, -1]]).T
        grid[:, :, 0] = 1
        for i in range(lenght):
            for j in range(1, 5):
                # print(velocity_set[j])
                grid[j] = np.roll(grid[j], velocity_set[j])
            for j in range(5, 9):
                grid[j] = np.roll(grid[j], velocity_set[j], axis=(0, 1))
            # middle test for values
            # print(grid[5])
            if i == 3:
                # print(grid[4, :, :, ])
                # first value doesnt move
                self.assertEqual(grid[0, 0, 0], 1)
                self.assertEqual(grid[1, 4, 0], 1)
                self.assertEqual(grid[2, 4, 0], 1)
                self.assertEqual(grid[3, 5, 0], 1)
                self.assertEqual(grid[4, 5, 0], 1)
                self.assertEqual(grid[5, 4, 0], 1)
                self.assertEqual(grid[6, 4, 0], 1)
                self.assertEqual(grid[7, 5, 0], 1)
                self.assertEqual(grid[8, 5, 0], 1)

        # check the values at the end
        for i in range(base - 1):  # from 1 to 4
            self.assertEqual(grid[i + 1, 0, 0], 1)  # roll through the whole array should be 1

    def test_streaming_from_center(self):
        ###
        '''
        For whatever reason np.roll does not like to go out the left side and will move the content one line down or up
        (depending on the direction) so thats fun, will now definitly use the original implementation seems to work best lol
        '''
        lenght = 9
        base = 9
        grid = np.zeros((base, lenght, lenght))
        velocity_set = np.array([[0, 1, 0, -1, 0, 1, -1, -1, 1],
                                 [0, 0, 1, 0, -1, 1, 1, -1, -1]]).T
        # put a 1 in every channel in the middle
        grid[:, 4, 4] = 1
        for i in range(lenght):
            for j in range(1, 9):
                grid[j] = np.roll(grid[j], velocity_set[j], axis=(0, 1))

        # everything should be in the middle again
        for i in range(base):
            # print(grid[i])
            self.assertEqual(grid[i, 4, 4], 1)

    def test_withOrgiginal_implementation(self):
        '''
        The content of one 1D array gets pulled forward 2 cells at a time which is logical as they get transported over
        the boundary
        '''
        # basic variables
        lenght = 9
        base = 9
        grid = np.zeros((base, 1, lenght))
        velocity_set = np.array([[0, 1, 0, -1, 0, 1, -1, -1, 1],
                                 [0, 0, 1, 0, -1, 1, 1, -1, -1]]).T
        grid[:, :, 0] = 1
        ####
        for i in range(lenght):
            stream(grid)
            # middle test for values
            # print(grid[1])
            if i == 3:
                # print(grid[5, :, :, ])
                # first value doesnt move
                self.assertEqual(grid[0, 0, 0], 1)
                # self.assertEqual(grid[1,0,4],1)
                # self.assertEqual(grid[2, 0, 4], 1)
                # self.assertEqual(grid[3, 0, 5], 1)
                # self.assertEqual(grid[4, 0, 5], 1)
                self.assertEqual(grid[5, 0, 4], 1)
                self.assertEqual(grid[6, 0, 4], 1)
                self.assertEqual(grid[7, 0, 5], 1)
                self.assertEqual(grid[8, 0, 5], 1)

        # check the values at the end
        for i in range(base - 1):  # from 1 to 4
            self.assertEqual(grid[i + 1, 0, 0], 1)  # roll through the whole array should be 1
    def test_array_channel2(self):
        #print("What I would expect for (0,1)")
        # basic variables
        lenght = 9
        base = 9
        grid = np.zeros((base, lenght, lenght))
        velocity_set = np.array([[0, 1, 0, -1, 0, 1, -1, -1, 1],
                                 [0, 0, 1, 0, -1, 1, 1, -1, -1]]).T
        watch = 2
        grid[watch, 1, 1:8] = 1
        #print(grid[watch])
        grid[watch] = np.roll(np.flip(grid[watch].T),velocity_set[watch])
        grid[watch] = np.flip(grid[watch].T)
        #print(grid[watch])

    def test_array_channel2_question(self):
        #print("What I got")
        # basic variables
        lenght = 9
        base = 9
        grid = np.zeros((base, lenght, lenght))
        velocity_set = np.array([[0, 1, 0, -1, 0, 1, -1, -1, 1],
                                 [0, 0, 1, 0, -1, 1, 1, -1, -1]]).T
        watch = 3
        grid[watch, 1, 1:8] = 1
        #print(grid[watch])
        grid[watch] = np.roll(grid[watch],velocity_set[watch],axis = (1,0))
        #print(grid[watch])

    def test_array_allChannel_streaming(self):
        # brute force test weather or not implementation is correct
        lenght = 9
        base = 9
        grid = np.zeros((base, lenght, lenght))
        control_grid = np.zeros((base,lenght,lenght))
        velocity_set = np.array([[0, 1, 0, -1, 0, 1, -1, -1, 1],
                                 [0, 0, 1, 0, -1, 1, 1, -1, -1]]).T
        # iterate over all channels
        for watch in range(1,9):
            # put a 1 in every array and observe how they move in 1 step
            grid[watch, 1, 1:8] = 1
            control_grid[watch] = np.roll(grid[watch],velocity_set[watch],axis = (0,1))
        # check for every channel weather they move as we expect them to do
        # channel 1 in positive x-Direction
        for i in range(1,8):
            self.assertEqual([grid[1,i,1]],control_grid[1,i+1,1])
        # channel 2 in posetive y-Direction
        for i in range(1,8):
            self.assertEqual([grid[2,i,1]],control_grid[2,i,2])
        # channel 3 in negative x-Direction
        for i in range(1,8):
            self.assertEqual([grid[3,i,1]],control_grid[3,i-1,1])
        # channel 4 in negative y-Direction
        for i in range(1,8):
            self.assertEqual([grid[4,i,1]],control_grid[4,i,0])
        # channel 5 in posetive x posetive y
        for i in range(1,8):
            self.assertEqual([grid[5,i,1]],control_grid[5,i+1,2])
        # channel 6 in negative x posetive y
        for i in range(1,8):
            self.assertEqual([grid[6,i,1]],control_grid[6,i-1,2])
        # channel 7 in negative x negative y
        for i in range(1,8):
            self.assertEqual([grid[7,i,1]],control_grid[7,i-1,0])
        # channel 8 in posetive x negative y
        for i in range(1,8):
            self.assertEqual([grid[8,i,1]],control_grid[8,i+1,0])

    def test_strides_array(self):
        grid = np.array([[[1, 2, 5]], [[7, 8, 9]]])
        # print(grid)
        # print(grid[1,:,:,]) # gives the elements with the index 0


'''
tester to get an understanding about the collision operation
'''


class testsInCollision(unittest.TestCase):
    ###
    def test_equilbrium_function(self):
        '''
        this is just a function, therefore tests are kinda hard to perform
        '''
        lenght = 9
        base = 9
        relaxation = np.pi / 3
        rho = np.zeros((lenght, lenght))
        ux = np.zeros((lenght, lenght))
        uy = np.zeros((lenght, lenght))
        grid = np.ones((base, lenght, lenght))
        equlibrium = np.zeros((base, lenght, lenght))
        collision = np.zeros((base, lenght, lenght))
        ###
        # need to watch out for fields vs scalar
        # collision first
        for i in range(lenght):
            for j in range(lenght):
                # get a shorthand for a gridpoint
                gridpoint = grid[:, i, j]
                # calculate pressure and velocities at that gridpoint
                rho[i, j], ux[i, j], uy[i, j] = calculate_3pincipal_values(gridpoint)
                # calculate the equilibrium at that gridpoint
                equlibrium[:, i, j] = equlibrium_function(rho[i, j], ux[i, j], uy[i, j])
                #  collision is something with dt/tau :> 1/tau (relaxation) for all equilbrias
                collision = (grid[:, i, j] - equlibrium[:, i, j]) / relaxation
        # apply collision
        grid = grid - collision
        # do streaming
        stream(grid)
        # this should conclude 1 step


'''
Tests for the boundary
'''

class testsForBoundary(unittest.TestCase):
    # i use the original stream here couse it is equivalent to the implementation in the last test
    def test_bounce_back_1channel_resting_wall_horizontal_channels_fail(self):
        # test in channels 1 and 3 as they are the easiest to conceptulize
        # resting wall variant
        base = 9
        lenght = 9
        grid = np.zeros([base,lenght,lenght])
        # assume boundary nodes at the edges, so only a 7x7 flow thingi
        # put 1s in channel 1
        grid[1, 1:8, 1] = 1
        # print(grid)
        # instead of streaming to the other side they should come back and then we hit them against the wall again
        for i in range(15):
            streaming(grid)
            # very cumbersom to do it this very explicte way
            # right side
            grid[3, :, 7] = grid[1, :, 8]
            grid[6, :, 7] = grid[8, :, 8]
            grid[7, :, 7] = grid[5, :, 8]
            # set value to 0 to be able to bounce back again
            grid[1, :, 8] = 0
            grid[8, :, 8] = 0
            grid[5, :, 8] = 0
            # left side
            grid[1, :, 1] = grid[3, :, 0]
            grid[8, :, 1] = grid[6, :, 0]
            grid[5, :, 1] = grid[7, :, 0]
            # set values to 0
            grid[3, :, 0]
            grid[6, :, 0]
            grid[7, :, 0]
            # lol
            #print(grid[1])
            #after 7 steps the ones should be in channel 3
            if i == 6:
                for j in range(1,8):
                    self.assertEqual(grid[3,j,7],1)
            #after 14 steps it should be back in channel 1
            if i == 13:
                for j in range(1, 8):
                    self.assertEqual(grid[1, j, 1], 1)

    def test_bounce_back_1channel_resting_wall_vertical_channels_fail(self):
        # test in channels 2 and 4 as they are also easy to conceptulize
        # resting wall variant
        base = 9
        lenght = 9
        grid = np.zeros([base,lenght,lenght])
        # assume boundary nodes at the edges, so only a 7x7 flow thingi
        # put 1s in channel 2 and let it move up
        grid[2, 7, 1:8] = 1
        # print(grid[2])
        # instead of streaming to the other side they should come back and then we hit them against the wall again
        for i in range(15):
            grid[2] = np.roll(grid[2],(0,1))
            # very cumbersom to do it this very explicte way
            # right side
            grid[3, :, 7] = grid[1, :, 8]
            grid[6, :, 7] = grid[8, :, 8]
            grid[7, :, 7] = grid[5, :, 8]
            # set value to 0 to be able to bounce back again
            grid[1, :, 8] = 0
            grid[8, :, 8] = 0
            grid[5, :, 8] = 0
            # left side
            grid[1, :, 1] = grid[3, :, 0]
            grid[8, :, 1] = grid[6, :, 0]
            grid[5, :, 1] = grid[7, :, 0]
            # set values to 0
            grid[3, :, 0]
            grid[6, :, 0]
            grid[7, :, 0]
            #####
            # top side
            grid[4, 1, :] = grid[2, 0, :]
            grid[7, 1, :] = grid[5, 0, :]
            grid[8, 1, :] = grid[6, 0, :]
            # set values to 0
            grid[2, 0, :]
            grid[5, 0, :]
            grid[6, 0, :]
            # bottum side
            grid[2, 7, :] = grid[4, 8, :]
            grid[5, 7, :] = grid[7, 8, :]
            grid[6, 7, :] = grid[8, 8, :]
            # set values to 0
            grid[4, 8, :]
            grid[7, 8, :]
            grid[8, 8, :]
            # lol
            #print(grid[2])

    def test_bounce_back_vertical_channels(self):
        # resting wall variant
        base = 9
        lenght = 9
        max_size = lenght-1 # for iteration in the array
        grid = np.zeros([base, lenght, lenght])
        grid[1, 1, 1:8] = 1
        #print(grid[1])
        # back and forth
        for i in range(14):
            stream(grid)
            # check boundaries for anything
            # right so x = 0
            grid[1, 1, :] = grid[3, 0, :]
            grid[5, 1, :] = grid[7, 0, :]
            grid[8, 1, :] = grid[6, 0, :]
            grid[3, 0, :] = 0
            grid[7, 0, :] = 0
            grid[6, 0, :] = 0
            # left so x = 8
            grid[3, max_size -1, :] = grid[1, max_size, :]
            grid[6, max_size -1, :] = grid[8, max_size, :]
            grid[7, max_size -1, :] = grid[5, max_size, :]
            grid[1, max_size, :] = 0
            grid[8, max_size, :] = 0
            grid[5, max_size, :] = 0
        # check after the array bounce back 2 times weather its in the original spot again
        for i in range(1,7):
            self.assertEqual(grid[1,1,i],1)


    def test_bounce_back_horizontal_channels(self):
        base = 9
        lenght = 9
        max_size = lenght - 1  # for iteration in the array
        grid = np.zeros([base, lenght, lenght])
        grid[2, 1:8, 1] = 1
        #print(grid[2])
        # back and forth
        for i in range(14):
            stream(grid)
            # check the boundaries
            # for bottom y = 0
            grid[2, :, 1] = grid[4, :, 0]
            grid[5, :, 1] = grid[7, :, 0]
            grid[6, :, 1] = grid[8, :, 0]
            grid[4, :, 0] = 0
            grid[7, :, 0] = 0
            grid[8, :, 0] = 0
            # for top y = max_size
            grid[4, :, max_size -1 ] = grid[2, :, max_size]
            grid[7, :, max_size -1] = grid[5, :, max_size]
            grid[8, :, max_size -1] = grid[6, :, max_size]
            grid[2, :, max_size] = 0
            grid[5, :, max_size] = 0
            grid[6, :, max_size] = 0
        # check after the array bounce back 2 times weather its in the original spot again
        for i in range(1, 7):
            self.assertEqual(grid[2, i, 1], 1)

    def test_pbc_with_presure_variation(self):
        #inilzilaize stuff for tests
        channels = 9
        lenght = 10
        max_size = lenght - 1  # for iteration in the array
        rho_null = 1
        rho = rho_null* np.ones((lenght,lenght))
        ux = np.zeros((lenght,lenght))
        uy = np.zeros((lenght,lenght))
        grid = equilibrium_on_array_test(rho,ux,uy)
        # need p in p out
        # equations 5.22 and 5.23?!
        # cut off the first and last to make life easier?!
        # before streaming i guess
        # calc eq function needed
        # pressures given for the calc
        p = 1/3 * rho_null
        delta_p = 0.001
        # recalculated p into rho and put it in an array
        rho_in = (p + delta_p) *3 * np.ones((grid.shape[1]))
        rho_out = (p - delta_p) *3 * np.ones((grid.shape[1]))
        #print(rho_in.shape)

        # get all the values
        rho = np.sum(grid, axis = 0)  # sums over each one individually
        ux = ((grid[1] + grid[5] + grid[8]) - (grid[3] + grid[6] + grid[7])) / rho
        uy = ((grid[2] + grid[5] + grid[6]) - (grid[4] + grid[7] + grid[8])) / rho
        equilibrium = equilibrium_on_array_test(rho,ux,uy)

        ##########
        equilibrium_in = equilibrium_on_array_test(rho_in,ux[:,max_size], uy[:,max_size])
        #print(equilibrium_in[1].shape)
        #print(grid[1,0,:].shape)
        #print((grid[1, max_size, :] - equilibrium[1, max_size, :]))
        # inlet 1,5,8
        grid[1,0,:] = equilibrium_in[1] + (grid[1,max_size,:]- equilibrium[1,max_size,:])
        grid[5,0,:] = equilibrium_in[5] + (grid[5,max_size,:]- equilibrium[5,max_size,:])
        grid[8,0,:] = equilibrium_in[8] + (grid[8,max_size,:]- equilibrium[8,max_size,:])

        # outlet 3,6,7
        equilibrium_out = equilibrium_on_array_test(rho_out, ux[:,0], uy[:, 0])
        # check for correct sizes
        grid[3,max_size,:] = equilibrium_out[3] + (grid[3,0,:]-equilibrium[3,0,:])
        grid[6,max_size,:] = equilibrium_out[6] + (grid[6,0,:]-equilibrium[6,0,:])
        grid[7,max_size,:] = equilibrium_out[7] + (grid[7,0,:]-equilibrium[7,0,:])

        ################
        # tests
        # check for the correct sizes
        self.assertEqual(equilibrium_out[1].shape, (lenght,))
        self.assertEqual(equilibrium_in[3].shape, (lenght,))

    def test_broken_pressure(self):
        # comparision with andreas version
        channels = 9
        lenght = 10
        max_size = lenght - 1  # for iteration in the array
        rho_null = 1
        rho = rho_null * np.ones((lenght, lenght))
        ux = np.zeros((lenght, lenght))
        uy = np.zeros((lenght, lenght))
        grid1 = equilibrium_on_array_test(rho, ux, uy)
        grid2 = equilibrium_on_array_test(rho, ux, uy)
        diff = 0.0001
        rho_in = rho_null + diff
        rho_out = rho_null - diff


        #####
        # Call
        good_pressure_variation(grid1,rho_in,rho_out)
        own_periodic_boundary_with_pressure_variations(grid2,rho_in,rho_out)
        #####
        #tests
        # in flow at boundary
        self.assertEqual(grid1[1,0,1],grid2[1,0,1])
        for c in range(channels):
            for i in range(lenght):
                self.assertEqual(grid1[c, 0, i], grid2[c, 0, i]) # in side
        # out flow at boundary
        for c in range(channels):
            for i in range(lenght):
                pass
                #self.assertEqual(grid1[c,-1,i],grid2[c,-1,i]) # out side # here is the error

    def test_true_diffrences_periodic_boundary(self):
        channels = 9
        lenght = 10
        max_size = lenght - 1  # for iteration in the array
        rho_null = 1
        rho = rho_null * np.ones((lenght, lenght))
        ux = np.zeros((lenght, lenght))
        uy = np.zeros((lenght, lenght))
        grid1 = equilibrium_on_array_test(rho, ux, uy)
        grid2 = equilibrium_on_array_test(rho, ux, uy)
        diff = 0.0001
        rho_in = rho_null + diff
        rho_out = rho_null - diff

        ##### Andreas
        w = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36])  # weights
        c = np.array([[0, 1, 0, -1, 0, 1, -1, -1, 1],  # velocities, x components
                      [0, 0, 1, 0, -1, 1, 1, -1, -1]])  # velocities, y components

        rho1 = np.einsum('ij->j', grid1[:, 1, :])
        u1 = np.einsum('ai,iy->ay', c, grid1[:, 1, :]) / rho1
        cdot3u = 3 * np.einsum('ai,ay->iy', c, u1)
        usq = np.einsum('ay->y', u1 * u1)
        feqpout = rho_out * w[:, np.newaxis] * (1 + cdot3u * (1 + 0.5 * cdot3u) - 1.5 * usq[np.newaxis, :])
        wrho1 = np.einsum('i,y->iy', w, rho1)
        feq1 = wrho1 * (1 + cdot3u * (1 + 0.5 * cdot3u) - 1.5 * usq[np.newaxis, :])
        fneq1 = grid1[:, 1, :]
        fout = feqpout + (fneq1 - feq1)
        grid1[:, -1, :] = fout

        ##### MINE
        rho, ux, uy = caluculate_real_values(grid2)
        equilibrium = equilibrium_on_array_test(rho, ux, uy)
        f_equilibrium = equilibrium[:,1,:]
        f_i = grid2[:,1,:]
        equilibrium_out = equilibrium_on_array_test(rho_out, ux[:, 1], uy[:, 1])
        # check for correct sizes
        grid2[:, -1, :] = equilibrium_out + (grid2[:, 1, :] - f_equilibrium )
        # test the first element
        for c in range(channels):
            for i in range(lenght):
                 self.assertEqual(feqpout[c,i],equilibrium_out[c,i])
        # test the second element
        for c in range(channels):
            for i in range(lenght):
                self.assertEqual(feq1[c,i],f_equilibrium[c,i])
        # test the last element
        for c in range(channels):
            for i in range(lenght):
                self.assertEqual(fneq1[c,i],f_i[c,i])


        for c in range(channels):
            for i in range(lenght):
                self.assertEqual(grid1[c, -1, i], grid2[c, -1, i])  # out side





    def test_simple_stuff(self):
        size_x = 100
        size_y = 300
        x = np.arange(0,size_x)
        y = np.arange(0, size_y)
        #print(x)
        grid = np.zeros((9,5,3))
        #print(grid.shape[1])


class testsForNewCollision(unittest.TestCase):
    def test_faster_principal_calc(self):
        # init stuff
        channels = 9
        size = 50
        size_x = 50
        size_y = 50
        grid = np.ones((channels,size_x,size_y))
        rho = np.zeros((size_x, size_y))
        ux = np.zeros((size_x, size_y))
        uy = np.zeros((size_x, size_y))
        # calc stuff
        for k in range(size_x):
            for l in range(size_y):
                rho[k, l], ux[k, l], uy[k, l] =  calculate_3pincipal_values(grid[:,k,l])

        # other way of calculating
        rho2 = np.sum(grid, axis = 0)
        ux1 = ((grid[1] + grid[5] + grid[8]) - (grid[3] + grid[6] + grid[7])) / rho
        uy1 = ((grid[2] + grid[5] + grid[6]) - (grid[4] + grid[7] + grid[8])) / rho
        # test only the first value
        self.assertEqual(rho[0,0],rho2[0,0])
        self.assertEqual(ux[0, 0], ux1[0, 0])
        self.assertEqual(uy[0, 0], uy1[0, 0])

    def test_equilibrium(self):
        # init stuff
        channels = 9
        size = 50
        size_x = 50
        size_y = 50
        grid = np.ones((channels, size_x, size_y))
        rho = np.zeros((size_x, size_y))
        ux = np.zeros((size_x, size_y))
        uy = np.zeros((size_x, size_y))
        #
        rho = np.sum(grid, axis=0)  # sums over each one individually
        ux = ((grid[1] + grid[5] + grid[8]) - (grid[3] + grid[6] + grid[7])) / rho
        uy = ((grid[2] + grid[5] + grid[6]) - (grid[4] + grid[7] + grid[8])) / rho
        #
        eq2 = equilibrium_on_array_test(rho,ux,uy)
        for k in range(size_x):
            for l in range(size_y):
                eq = equlibrium_function(rho[k, l], ux[k, l], uy[k, l])
                for c in range(channels):
                    self.assertEqual(eq[c], eq2[c, k, l])

'''
functions
'''
# Copy pasta
c_ic = np.array([[0,  1,  0, -1,  0,  1, -1, -1,  1],
                 [0,  0,  1,  0, -1,  1,  1, -1, -1]]).T
def stream(f_ikl):
    for i in range(1, 9):
        f_ikl[i] = np.roll(f_ikl[i], c_ic[i], axis=(0, 1))

def equlibrium_function(rho, ux, uy):
    # TODO ask for the explicit reduction of the function 3.54 in the book especially the delta
    # still need to practice the einstein summation
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
    return equilibrium

def equilibrium_on_array_test(rho,ux,uy):
    uxy_3plus = 3 * (ux + uy)
    uxy_3miuns = 3* (ux -uy)
    uu =  3 * (ux * ux + uy * uy)
    ux_6 = 6*ux
    uy_6 = 6*uy
    uxx_9 = 9 * ux*ux
    uyy_9 = 9 * uy*uy
    uxy_9 = 9 * ux*uy
    return np.array([(2*rho/9) * (2-uu),
                    (rho / 18)* (2 + ux_6 + uxx_9-uu),
                    (rho / 18)* (2 + uy_6 + uyy_9-uu),
                    (rho / 18)* (2 - ux_6 + uxx_9-uu),
                    (rho / 18)* (2 - uy_6 + uyy_9-uu),
                    (rho / 36) * (1 + uxy_3plus + uxy_9 + uu),
                    (rho / 36) * (1 - uxy_3miuns - uxy_9 + uu),
                    (rho / 36) * (1 - uxy_3plus + uxy_9 + uu),
                    (rho / 36) * (1 + uxy_3miuns - uxy_9 + uu)])

def calculate_3pincipal_values(gridpoint):
    # just the basic equations
    rho = np.sum(gridpoint)
    ux = ((gridpoint[1] + gridpoint[5] + gridpoint[8]) - (gridpoint[3] + gridpoint[6] + gridpoint[7])) / rho
    uy = ((gridpoint[2] + gridpoint[5] + gridpoint[6]) - (gridpoint[4] + gridpoint[7] + gridpoint[8])) / rho
    return rho, ux, uy


def streaming(grid):
    # THIS IS WRONG!! but i dont want to break tests
    ####
    velocity_set = np.array([[0, 1, 0, -1, 0, 1, -1, -1, 1],
                             [0, 0, 1, 0, -1, 1, 1, -1, -1]]).T
    ####
    for j in range(1, 5):
        # print(velocity_set[j])
        grid[j] = np.roll(grid[j], velocity_set[j])
    for j in range(5, 9):
        grid[j] = np.roll(grid[j], velocity_set[j], axis=(0, 1))

def baunce_back_resting_wall(grid):
    # baunce back without any velocity gain
    max_size_x = grid.shape[1]-1 # x
    max_size_y = grid.shape[2]-1 # y
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
    grid[7, :, max_size_y - 1] = grid[5, :, max_size_y]
    grid[8, :, max_size_y - 1] = grid[6, :, max_size_y]
    grid[2, :, max_size_y] = 0
    grid[5, :, max_size_y] = 0
    grid[6, :, max_size_y] = 0

def baunce_back_top_moving(grid,uw):
    # baunce back without any velocity gain
    max_size_x = grid.shape()[1] # x
    max_size_y = grid.shape()[2] # y
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
    grid[7, :, max_size_y - 1] = grid[5, :, max_size_y] - 1/6*uw
    grid[8, :, max_size_y - 1] = grid[6, :, max_size_y] + 1/6*uw
    grid[2, :, max_size_y] = 0
    grid[5, :, max_size_y] = 0
    grid[6, :, max_size_y] = 0

def good_pressure_variation(g,pin,pout):
    w = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36])  # weights
    c = np.array([[0, 1, 0, -1, 0, 1, -1, -1, 1],  # velocities, x components
                  [0, 0, 1, 0, -1, 1, 1, -1, -1]])  # velocities, y components
    #
    rhoN = np.einsum('ij->j', g[:, -2, :])
    uN = np.einsum('ai,iy->ay', c, g[:, -2, :]) / rhoN
    cdot3u = 3 * np.einsum('ai,ay->iy', c, uN)
    usq = np.einsum('ay->y', uN * uN)
    feqpin = pin * w[:, np.newaxis] * (1 + cdot3u * (1 + 0.5 * cdot3u) - 1.5 * usq[np.newaxis, :])
    wrhoN = np.einsum('i,y->iy', w, rhoN)
    feqN = wrhoN * (1 + cdot3u * (1 + 0.5 * cdot3u) - 1.5 * usq[np.newaxis, :])
    fneqN = g[:, -2, :]
    fin = feqpin + (fneqN - feqN)
    #
    rho1 = np.einsum('ij->j', g[:, 1, :])
    u1 = np.einsum('ai,iy->ay', c, g[:, 1, :]) / rho1
    cdot3u = 3 * np.einsum('ai,ay->iy', c, u1)
    usq = np.einsum('ay->y', u1 * u1)
    feqpout = pout * w[:, np.newaxis] * (1 + cdot3u * (1 + 0.5 * cdot3u) - 1.5 * usq[np.newaxis, :])
    wrho1 = np.einsum('i,y->iy', w, rho1)
    feq1 = wrho1 * (1 + cdot3u * (1 + 0.5 * cdot3u) - 1.5 * usq[np.newaxis, :])
    fneq1 = g[:, 1, :]
    fout = feqpout + (fneq1 - feq1)
    g[:, 0, :] = fin
    g[:, -1, :] = fout

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


def own_periodic_boundary_with_pressure_variations(grid,rho_in,rho_out):

    # get all the values
    rho, ux, uy = caluculate_real_values(grid)
    equilibrium = equilibrium_on_array_test(rho, ux, uy)
    ##########
    equilibrium_in = equilibrium_on_array_test(rho_in, ux[:, -2], uy[:, -2])
    # inlet 1,5,8
    grid[:, 0, :] = equilibrium_in + (grid[:, -2, :] - equilibrium[:, -2, :])

    # TODO fehler ist in diesem Teil compare 1 to 1 to other fkt
    # outlet 3,6,7
    equilibrium_out = equilibrium_on_array_test(rho_out, ux[:, 1], uy[:, 1])
    # check for correct sizes
    grid[:, -1, :] = equilibrium_out + (grid[:, 1, :] - equilibrium[:, 1, :])

def both_perodic_boundaries(grid1,grid2,rho_in,rho_out,testsForBoundary):
    ##### Andreas
    w = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36])  # weights
    c = np.array([[0, 1, 0, -1, 0, 1, -1, -1, 1],  # velocities, x components
                  [0, 0, 1, 0, -1, 1, 1, -1, -1]])  # velocities, y components

    rho1 = np.einsum('ij->j', grid1[:, 1, :])
    u1 = np.einsum('ai,iy->ay', c, grid1[:, 1, :]) / rho1
    cdot3u = 3 * np.einsum('ai,ay->iy', c, u1)
    usq = np.einsum('ay->y', u1 * u1)
    feqpout = rho_out * w[:, np.newaxis] * (1 + cdot3u * (1 + 0.5 * cdot3u) - 1.5 * usq[np.newaxis, :])
    wrho1 = np.einsum('i,y->iy', w, rho1)
    feq1 = wrho1 * (1 + cdot3u * (1 + 0.5 * cdot3u) - 1.5 * usq[np.newaxis, :])
    fneq1 = grid1[:, 1, :]
    fout = feqpout + (fneq1 - feq1)
    grid1[:, -1, :] = fout

    ##### MINE
    rho, ux, uy = caluculate_real_values(grid2)
    equilibrium = equilibrium_on_array_test(rho, ux, uy)
    equilibrium_out = equilibrium_on_array_test(rho_out, ux[:, 1], uy[:, 1])
    # check for correct sizes
    grid2[:, -1, :] = equilibrium_out + (grid2[:, 1, :] - equilibrium[:, 1, :])




if __name__ == '__main__':
    unittest.main()
