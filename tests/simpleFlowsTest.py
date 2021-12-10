'''
The purpose of this file is to help develop the simpleFlows
    -> mainly test to get an understanding and not hardcore unittests to show sth works
'''

import unittest
import numpy as np
from PyLB import stream

'''
Tester for Streaming
'''
class testsInStreaming(unittest.TestCase):
    def test_basic_np_roll(self):
        #make a grid with a lenght and write the first element to one-> then roll the element through
        lenght = 9
        grid = np.zeros((1, 1, lenght))
        grid[:, :, 0] = 1
        #roll through the array
        for i in range(lenght-1):
            grid = np.roll(grid,(1,0))
        # look weather the last elemnt is 1
        self.assertEqual(grid[0,0,lenght-1], 1)  # add assertion here

    def test_multidimensional_roll(self):
        #now with more channels here the 4 main channels
        lenght = 9
        base = 5
        grid = np.zeros((base,1,lenght))
        #print(grid.shape)
        grid[:,:,0] = 1
        #roll through but ignore channel 0 !!
        for i in range(lenght):
            '''
            the np.roll rolls through the 1D-array from start to finish 
            it doesnt really matter if i write (1,0) or (0,1) as the roll still performs the same 
            '''
            grid[1,:,:,] =  np.roll(grid[1,:,:,], (1,0))
            grid[2,:,:,] =  np.roll(grid[2,:,:,], (0,1)) # this doesnt really change the rolling behaviour still rolls throu the array in 1D
            grid[3,:,:,]  = np.roll(grid[3,:,:,], (-1, 0))
            grid[4,:,:,]  = np.roll(grid[4,:,:,], (0, -1))

        #check the values
        for i in range(base -1): #from 1 to 4
            self.assertEqual(grid[i+1, 0, 0], 1) # roll through the whole array should be 1

    def test_full_multidimensional(self):
        #basic variables
        lenght = 9
        base = 9
        grid = np.zeros((base, 1, lenght))
        velocity_set = np.array([[0, 1, 0, -1, 0, 1, -1, -1, 1],
                                 [0, 0, 1, 0, -1, 1, 1, -1, -1]]).T
        grid[:, :, 0] = 1
        ###
        for i in range(lenght):
            for j in range(base-1):
                grid[j+1, :, :, ] = np.roll(grid[j+1,:,:,], velocity_set[j+1])
                # do a follow up check in the middle
            #print(grid[5, :, :, ])
            if(i == 3):
                #elements are now in the middle for the principal axis
                self.assertEqual(grid[2,0,4], 1)
                self.assertEqual(grid[3,0,5], 1)


        #check the values
        for i in range(base -1): #from 1 to 4
            self.assertEqual(grid[i+1, 0, 0], 1) # roll through the whole array should be 1

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
            for j in range(1 , 5):
                #print(velocity_set[j])
                grid[j] = np.roll(grid[j],velocity_set[j])
            for j in range(5, 9):
                grid[j] = np.roll(grid[j], velocity_set[j], axis=(0, 1))
            #middle test for values
            #print(grid[5])
            if i == 3:
                #print(grid[4, :, :, ])
                #first value doesnt move
                self.assertEqual(grid[0,0,0],1)
                self.assertEqual(grid[1,0,4],1)
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
        # basic variables
        lenght = 9
        base = 9
        grid = np.zeros((base, lenght, 1))
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
            for j in range(1 , 5):
                #print(velocity_set[j])
                grid[j] = np.roll(grid[j],velocity_set[j])
            for j in range(5, 9):
                grid[j] = np.roll(grid[j], velocity_set[j], axis=(0, 1))
            #middle test for values
            #print(grid[5])
            if i == 3:
                #print(grid[4, :, :, ])
                #first value doesnt move
                self.assertEqual(grid[0,0,0],1)
                self.assertEqual(grid[1,4,0],1)
                self.assertEqual(grid[2, 4, 0], 1)
                self.assertEqual(grid[3, 5,0 ], 1)
                self.assertEqual(grid[4, 5, 0], 1)
                self.assertEqual(grid[5, 4, 0], 1)
                self.assertEqual(grid[6, 4,0 ], 1)
                self.assertEqual(grid[7, 5, 0], 1)
                self.assertEqual(grid[8, 5, 0], 1)

        # check the values at the end
        for i in range(base - 1):  # from 1 to 4
            self.assertEqual(grid[i + 1, 0, 0], 1)  # roll through the whole array should be 1

    def test_withOrgiginal_implementation(self):
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
            #middle test for values
            #print(grid[1])
            if i == 3:
                #print(grid[5, :, :, ])
                #first value doesnt move
                self.assertEqual(grid[0,0,0],1)
                #self.assertEqual(grid[1,0,4],1)
                #self.assertEqual(grid[2, 0, 4], 1)
                #self.assertEqual(grid[3, 0, 5], 1)
                #self.assertEqual(grid[4, 0, 5], 1)
                self.assertEqual(grid[5, 0, 4], 1)
                self.assertEqual(grid[6, 0, 4], 1)
                self.assertEqual(grid[7, 0, 5], 1)
                self.assertEqual(grid[8, 0, 5], 1)

        # check the values at the end
        for i in range(base - 1):  # from 1 to 4
            self.assertEqual(grid[i + 1, 0, 0], 1)  # roll through the whole array should be 1

    def test_strides_array(self):
       grid = np.array([[[1,2,5]],[[7,8,9]]])
       #print(grid)
       #print(grid[1,:,:,]) # gives the elements with the index 0

'''
tester to get an understanding about the collision operation
'''
class testsInCollision(unittest.TestCase):
    ###
    def test_equilbrium_function(self):
        '''
        this is just a function, therefore tests are kinda hard to perform
        '''
        ###
        rho = 0
        ux = 0
        uy = 0
        lenght = 9
        base = 9
        grid = np.ones((base, 1, lenght))
        ###
        #need to think about fields vs scalar
        #calculate_3pincipalValues(grid)
        equlibrium_function(rho,ux,uy)

'''
functions
'''
def equlibrium_function(rho, ux, uy):
    #TODO ask for the explicit reduction of the function 3.54 in the book especially the delta
    # still need to practice the einstein summation
    equilibrium = np.zeros(9)
    uxy = ux + uy
    uu = ux*ux + uy*uy
    equilibrium[0] = 2/9*rho * (2 -3*uu)
    equilibrium[1] = rho/18 * (2+6*ux+9*ux*ux-3*uu)
    equilibrium[2] = rho/18 * (2+6*uy+9*uy*uy-3*uu)
    equilibrium[3] = rho/18 * (2-6*ux+9*ux*ux-3*uu)
    equilibrium[4] = rho/18 * (2-6*uy+9*uy*uy-3*uu)
    equilibrium[5] = rho/36 * (1+3*uxy+9*ux*uy+3*uu)
    equilibrium[6] = rho/36 * (1-3*uxy-9*ux*uy+3*uu)
    equilibrium[7] = rho/36 * (1-3*uxy+9*ux*uy+3*uu)
    equilibrium[8] = rho/36 * (1+3*uxy-9*ux*uy+3*uu)
    return equilibrium

def calculate_3pincipalValues(gridpoint):
    #just the basic equations
    rho = np.sum(gridpoint)
    ux = ((gridpoint[1]+gridpoint[5]+gridpoint[8])-(gridpoint[3]+gridpoint[6]+gridpoint[7]))/rho
    uy = ((gridpoint[2] + gridpoint[5]+gridpoint[6])-(gridpoint[4]+gridpoint[7]+gridpoint[8]))/rho

def streaming(grid):
    ####
    velocity_set = np.array([[0, 1, 0, -1, 0, 1, -1, -1, 1],
                             [0, 0, 1, 0, -1, 1, 1, -1, -1]]).T
    ####
    for j in range(1, 5):
        # print(velocity_set[j])
        grid[j] = np.roll(grid[j], velocity_set[j])
    for j in range(5, 9):
        grid[j] = np.roll(grid[j], velocity_set[j], axis=(0, 1))



if __name__ == '__main__':
    unittest.main()
