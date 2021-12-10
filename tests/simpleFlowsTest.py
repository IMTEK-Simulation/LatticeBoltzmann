'''
The purpose of this file is to help develop the simpleFlows
    -> mainly test to get an understanding and not hardcore unittests to show sth works
'''

import unittest
import numpy as np

'''
Tester
'''
class testsInStreaming(unittest.TestCase):
    def test_basic_np_roll(self):
        #make a grid with a lenght and write the first element to one-> then roll the element through
        lenght = 9
        grid = np.zeros((1, 1, lenght))
        grid[:, :, 0] = 1
        #roll through the array
        for i in range(lenght-1):
            grid = roll_through(grid)
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
            grid[1,:,:,] =  roll_through_multidimensional(grid[1,:,:,], (1,0))
            grid[2,:,:,] =  roll_through_multidimensional(grid[2,:,:,], (0,1)) # this doesnt really change the rolling behaviour still rolls throu the array in 1D
            grid[3,:,:,]  = roll_through_multidimensional(grid[3,:,:,], (-1, 0))
            grid[4,:,:,]  = roll_through_multidimensional(grid[4,:,:,], (0, -1))

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
                grid[j+1, :, :, ] = roll_through_multidimensional(grid[j+1,:,:,], velocity_set[j+1])
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
        for i in range(lenght):
            for j in range(1 , base):
                #print(velocity_set[j])
                grid[j] = stream_1D(grid[j],velocity_set[j])
            #middle test for values
            #print(grid[1])
            if i == 3:
                print(grid[1, :, :, ])
                #first value doesnt move
                self.assertEqual(grid[0,0,0],1)
                self.assertEqual(grid[1,0,4],1)
                self.assertEqual(grid[2, 0, 4], 1)
                self.assertEqual(grid[3, 0, 5], 1)
                self.assertEqual(grid[4, 0, 5], 1)
                #self.assertEqual(grid[5, 0, 4], 1)
                #self.assertEqual(grid[6, 0, 4], 1)
                #self.assertEqual(grid[7, 0, 5], 1)
                #self.assertEqual(grid[8, 0, 5], 1)



        # check the values at the end
        for i in range(base - 1):  # from 1 to 4
            self.assertEqual(grid[i + 1, 0, 0], 1)  # roll through the whole array should be 1





    def test_strides_array(self):
       grid = np.array([[[1,2,5]],[[7,8,9]]])
       #print(grid)
       #print(grid[1,:,:,]) # gives the elements with the index 0
'''
helper
'''
def roll_through(grid):
    #basic 1D-Roll throu -> just in one axis
    grid = np.roll(grid,(1,0))
    return grid

def roll_through_multidimensional(grid,velocity_set):
    #roll through in 2D with a given velocity set
    grid = np.roll(grid,velocity_set)
    return grid

def stream_1D(grid,velocity_set):
    #full roll in 2D just one jump i guess as expected
    grid = np.roll(grid,velocity_set)
    return grid

if __name__ == '__main__':
    unittest.main()
