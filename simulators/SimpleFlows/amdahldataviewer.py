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
time = np.square(np.arange(1,16))

print("Making Image")
# recalculate ux and uy
plt.title("Sliding Lid")
plt.xlabel("Number of cores")
plt.ylabel("Time")
savestring = "Amdahls" + "view" + ".png"
plt.savefig(savestring)
plt.plot(callnumbers,time)
plt.show()
