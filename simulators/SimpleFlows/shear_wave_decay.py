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

# Todo: remove me
'''
my todos:
implement the shear wave decay should orient myself fully to the course now
needs collsision, streaming, equilibrium
if i understood this corretly i just give the sim a random velocity in the middle that is sinlike?!
'''
# imports
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt

# initial variables and sizes
steps = 3000
size_x = 200
size_y = 200
k_y = 2*np.pi/size_x
amplitude = 0.1
periode = 1
relaxation = 0.2
velocity_set = np.array([[0, 1, 0, -1, 0, 1, -1, -1, 1],
                         [0,0,1,0,-1,1,1,-1,-1]]).T

# main functions
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


def collision_with_relaxation(grid,rho,ux,uy,relaxxation):
    grid -= relaxxation * (grid - equilibrium(rho, ux, uy))


def caluculate_rho_ux_uy(grid):
    rho = np.sum(grid, axis=0)  # sums over each one individually
    ux = ((grid[1] + grid[5] + grid[8]) - (grid[3] + grid[6] + grid[7])) / rho
    uy = ((grid[2] + grid[5] + grid[6]) - (grid[4] + grid[7] + grid[8])) / rho
    return rho,ux,uy

# fit stuff
def theo_Exp(x, v):
    return amplitude * np.exp(-v*k_y*k_y*x)

def theo_exp_with_variables(x,v,ky,amplitud):
    return amplitud * np.exp(-v * ky * ky * x)

# main body
def shear_wave_decay():
    print("Shear Wave Decay")
    # shear wave
    x_values = k_y * np.arange(0,size_x)
    shear_wave = amplitude * np.sin(periode * x_values)
    # initizlize the gird
    rho = np.ones((size_x, size_y))
    ux = np.zeros((size_x, size_y))
    ux[:, :] = shear_wave
    uy = np.zeros((size_x, size_y))
    grid = equilibrium(rho, ux, uy)

    amplitude_array = []

    # loop
    for i in range(steps):
        # standard procedure
        stream(grid)
        rho,ux,uy = caluculate_rho_ux_uy(grid)
        collision(grid,rho,ux,uy)
        ###
        # analize the amplitude
        ux_fft = np.fft.fft(ux[int(size_x/2),:])
        ampl = 2/size_y* np.abs(ux_fft)
        ampl = np.max(ampl)
        amplitude_array.append(ampl)

    # theoretical solution
    x = np.arange(0,steps)
    v = 1/3 * (1/relaxation - 1/2)
    # some sort of -e-fkt
    u_theo = amplitude * np.exp(-v*k_y*k_y*x)

    ###
    param,cv = scipy.optimize.curve_fit(theo_Exp,x,amplitude_array)
    v_s = param[0]
    print(v_s)
    print(v)
    # visualize
    fig, ax = plt.subplots()
    textstr = '\n'.join((
        r'size = %d x %d' % (size_x,size_y ),
        r'omega = %.02f' % (relaxation,),
        r'amplitude = %.02f' % (amplitude,),
        r'v_theo = %.02f' % (v,),
        r'v_sim = %.02f' % (v_s,)
    ))
    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    # place a text box in upper left in axes coords
    ax.text(0.71, 0.82, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)

    plt.plot(amplitude_array, label = "Simulated")
    plt.plot(u_theo, color = "red",label = "Theoretically")
    plt.title("Shear Wave Decay")
    plt.ylabel("Amplitude")
    plt.xlabel("# of steps")
    plt.legend()
    plt.show()

def shear_wave_decay_more(amplitud,relaxxation,ky):
    # return Params
    v_theoretical = 0
    v_simualated = 0
    amplitude_array = []
    x_values = ky * np.arange(0, size_x)
    shear_wave = amplitud * np.sin(periode * x_values)

    # initizlize the gird
    rho = np.ones((size_x, size_y))
    ux = np.zeros((size_x, size_y))
    ux[:, :] = shear_wave
    uy = np.zeros((size_x, size_y))
    grid = equilibrium(rho, ux, uy)

    # loop
    for i in range(steps):
        # standard procedure
        stream(grid)
        rho, ux, uy = caluculate_rho_ux_uy(grid)
        collision_with_relaxation(grid, rho, ux, uy,relaxxation)
        ###
        # analize the amplitude
        ux_fft = np.fft.fft(ux[int(size_x / 2), :])
        ampl = 2 / size_y * np.abs(ux_fft)
        ampl = np.max(ampl)
        amplitude_array.append(ampl)

    # v_theoretical
    x = np.arange(0, steps)
    v_theoretical = 1 / 3 * (1 / relaxxation - 1 / 2)
    # some sort of -e-fkt
    amplitude_theo = amplitud * np.exp(-v_theoretical * ky * ky * x)

    # v_simulated
    # lambda wrapper for ky and amplitud
    param, cv = scipy.optimize.curve_fit(lambda x,v : theo_exp_with_variables(x,v,ky,amplitud), x, amplitude_array)
    v_simualated = param[0]

    return v_theoretical, v_simualated,amplitude_theo, amplitude_array

def rapid_call():
    print("Mass caller")
    # put v theo and v sim in the labels
    # original amplitude
    v_theoretical_array = []
    v_siumlated_array = []
    amplitude_theo_array = []
    ampitude_array_array = []
    runs = 8
    #### Setup
    # cal patterns
    amplitud = np.array([0.1,0.1,0.1,0.1,0.3,0.3,0.3,0.3])
    relaxxation = np.array([0.2,0.2,1.5,1.5,0.2,0.2,1.5,1.5])
    nr = np.array([1, 2, 1, 2, 1, 2, 1, 2])
    ky = nr * k_y
    # running
    for i in range(runs):
        # fkt
        v_theoretical, v_simualated, amplitude_theo, amplitude_array = shear_wave_decay_more(amplitud[i],relaxxation[i] , ky[i])
        # append
        v_theoretical_array.append(v_theoretical)
        v_siumlated_array.append(v_simualated)
        amplitude_theo_array.append(amplitude_theo)
        ampitude_array_array.append(amplitude_array)

    # plotting
    x = 0
    y = 0
    fig_size = (10*2.5,8*2.5)
    axs = plt.figure(figsize = fig_size).subplots(4,2)
    for i in range(runs):
        # plotting
        axs[y, x].plot(amplitude_theo_array[i],label = "Theoretically")
        axs[y, x].plot(ampitude_array_array[i],label = "Simulated")
        axs[y,x].legend()
        title_string = ''.join((r'v_theo = %.02f, v_sim = %.02f' % (v_theoretical_array[i],v_siumlated_array[i])))
        x_lable_string = ''.join((r'Relaxation %.02f, %d * k_y, Amplitude = %.02f' % (relaxxation[i],nr[i],amplitud[i])))
        axs[y,x].set_title(title_string)
        axs[y,x].set_xlabel(x_lable_string)
        # counting
        x +=1
        if x == 2:
            x = 0
        if (i+1) % 2 == 0 and i != 0:
            y +=1

    plt.show()




# call
#shear_wave_decay()
rapid_call()

