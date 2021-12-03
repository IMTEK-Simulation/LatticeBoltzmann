
#Todo: remove us
'''
To get a basic feel here ill implement the Couette Flow between two plates
-----
some general remarks
ill do this one here in the most explixite form imaginable, in the Poiseuille Flow ill start doing optimizations
but as I am learning, ill do this first try the "dumb" way. I am aware that withhin the project there allready good
implementations of basically all the operations. But for the first try I will ignore them and will use my
implementations. (In the second try PoiseuilleFlow ill use them thou.) This should motivate my approach here. Btw in
D2Q9.

'''
'''
Remider CodingRules: 
Zeilenumbruch bei Spalte 120
Modulname, Klassennamen als CamelCase
Variablennamen, Methodennamen, Funktionsnamen mit unter_strichen
Bitte nicht CamelCase und Unterstriche mischen
'''

''' Imports '''
import numpy as np
import matplotlib.pyplot as plt


''' Setup '''
print("Couetteflow")

#weights and velocity sets
weights = np.array([4/9,1/9,1/9,1/9,1/36, 1/36, 1/36,1/36])
velocity_set = np.array([[0, 1, 0, -1, 0, 1, -1, -1, 1],
                         [0,0,1,0,-1,1,1,-1,-1]])

''' functions '''
def equilibrium():
    pass

#start with Streaming kinda easier on the head
def streaming():
    pass

def collision():
    pass

''' body '''


''' visualization '''
#quiver?!






