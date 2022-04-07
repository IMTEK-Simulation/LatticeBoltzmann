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

#print(sys.argv)

callnumbers = np.square(np.arange(1,16))
print (callnumbers)
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
                    17645.68878507614, #1
                    2811.8762357234955, #4
                    2074.2043981552124, #9
                    881.9688158035278, #16
                    #25
                    575.2479498386383, #36
                    431.27615690231323, #100
                    393.5351839065552, #144
                    380.3805010318756, #225
                    ])
processor_numbers = np.array([1,4,9,16,36,100,144,225])
speedup=time_good[1]/time_good[1:]
print(time_good)
print("Making Image")
# recalculate ux and uy
plt.title("Sliding Lid")
plt.xlabel("Number of cores")
plt.ylabel("Speedup")
savestring = "Amdahls" + "view" + ".png"
plt.savefig(savestring)
plt.plot(processor_numbers[1:],speedup)
plt.show()
