'''
The purpose of this file is to help develop the simpleFlows
    -> mainly test to get an understanding and not hardcore unittests to show sth works
'''

import unittest
import numpy as np

'''
Tester
'''
class testsInDevelopment(unittest.TestCase):
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
            #just roll the right channel
            grid[1,:,:,] =  roll_through_multidimensional(grid[1,:,:,], (1,0))
            grid[2,:,:,] =  roll_through_multidimensional(grid[2,:,:,], (1,0))
            grid[3,:,:,]  = roll_through_multidimensional(grid[3,:,:,], (-1, 0))
            grid[4,:,:,]  = roll_through_multidimensional(grid[4,:,:,], (-1, 0))

        #check the values
        for i in range(base -1): #from 1 to 4
            self.assertEqual(grid[i+1, 0, 0], 1) # roll through the whole array should be 1

    def test_full_multidimensional(self):
        lenght = 9
        base = 9
        grid = np.zeros((base, 1, lenght))
        # print(grid.shape)
        grid[:, :, 0] = 1
        ###
        for i in range(lenght):
            for j in range(base-1):
                direction = 1
                if(j == 3 or j == 4 or j == 7 or j == 8):
                    direction = -1
                grid[j+1, :, :, ] = roll_through_multidimensional(grid[j+1,:,:,], (direction,0))
                # do a follow up check in the middle
            #print(grid[2, :, :, ])
            if(i == 3):
                #the elements have to be the same on either side
                #print(grid[3,0,4])
                self.assertEqual(grid[2,0,4], 1)
                self.assertEqual(grid[3,0,4],1)


        #check the values
        for i in range(base -1): #from 1 to 4
            self.assertEqual(grid[i+1, 0, 0], 1) # roll through the whole array should be 1



    def test_strides_array(self):
       grid = np.array([[[1,2,5]],[[7,8,9]]])
       #print(grid)
       #print(grid[1,:,:,]) # gives the elements with the index 0
'''
helper
'''
def roll_through(grid):
    grid = np.roll(grid,(1,0))
    #print(grid)
    return grid

def roll_through_multidimensional(grid,velocity_set):
    grid = np.roll(grid,velocity_set)
    return grid

if __name__ == '__main__':
    unittest.main()
