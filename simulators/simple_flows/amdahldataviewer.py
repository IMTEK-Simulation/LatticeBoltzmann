'''
Remider CodingRules:
Zeilenumbruch bei Spalte 120
Modulname, Klassennamen als CamelCase
Variablennamen, Methodennamen, Funktionsnamen mit unter_strichen
Bitte nicht CamelCase und Unterstriche mischen
'''

import numpy as np
import matplotlib.pyplot as plt
import sys
def plotter():
    #print(sys.argv)
    re = 1000
    base_lenght = 300
    steps = 1000000
    uw = 0.1
    size_x = base_lenght
    size_y = base_lenght
    relaxation = (2*re)/(6*base_lenght*uw+re)
    ###
    # Watch out only full numbers really work, when the grid can be parted equally some will have to be removed (outline anyways)
    time = np.array([17645.68878507614, #1
                     2811.8762357234955, #4
                     2074.2043981552124, #9
                     881.9688158035278, #16
                     2, #25
                     575.2479498386383, #36
                     370.4072289466858, #49 crap
                     448.95830941200256, #64 crap
                     373.6482753753662, #81 crap
                     431.27615690231323, #100
                     354.9085099697113, #121 crap
                     393.5351839065552, #144
                     385.4690911769867, #169 crap
                     293.634001493454, #196 crap
                     380.3805010318756, #225
                     ])
    time_good = np.array([
                        19907.83862733841, #1
                        4859.682349920273, #4
                        2086.8067207336426, #9
                        941.4743375778198, #16:
                        701.873462677002, #25
                        575.2479498386383, #36
                        431.27615690231323, #100
                        393.5351839065552, #144
                        380.3805010318756, #225
                        379.8842520713806, #400
                        ])
    processor_numbers = np.array([1,4,9,16,25,36,100,144,225,400])
    mlups = calculate_lattice_updates(grid_size_approximation(processor_numbers,base_lenght),time_good)

    speedup=mlups[0:]/mlups[0]
    # print(time_good)
    print("Making Image")
    # recalculate ux and uy
    titleString = "Speedup of the Sliding Lid in MLUPS \n (Gridsize " + "{}".format(size_x) + "x" +"{}".format(size_y)
    titleString += ",  $\\omega$ = {:.2f}".format(relaxation) +  ", steps = {}".format(steps) + ")"
    plt.title(titleString)
    plt.xlabel("Number of cores")
    # plt.xscale("log",base = 2)
    plt.ylabel("Speedup")
    savestring = "Amdahls" + "view" + ".png"
    plt.plot(processor_numbers,speedup)
    plt.savefig(savestring)
    plt.show()

def grid_size_approximation(cores,base_length,steps= 1000000):
    # partion of the sides
    sides = np.asarray(np.sqrt(cores),dtype=int)
    sub_length = np.asarray(base_length/sides,dtype=int) +2
    # calculate the number of grid points shared by the cores
    sub_gridpoints = sub_length**2
    total = sub_gridpoints * cores
    updates = total * steps
    return updates

def calculate_lattice_updates(updates, time):
    lups = updates/time
    mlups = lups/1000000
    return mlups

# call patterns
plotter()
