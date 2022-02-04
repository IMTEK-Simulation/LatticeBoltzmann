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

# imports
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt

# initial variables and sizes
steps = 10000
size_x = 300
size_y = 300
k_y = 2*np.pi/size_x
amplitude_global = 0.1
periode = 1
relaxation_global = 0.2
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
    grid -= relaxation_global * (grid - equilibrium(rho, ux, uy))


def collision_with_relaxation(grid,rho,ux,uy,relaxxation):
    grid -= relaxxation * (grid - equilibrium(rho, ux, uy))


def caluculate_rho_ux_uy(grid):
    rho = np.sum(grid, axis=0)  # sums over each one individually
    ux = ((grid[1] + grid[5] + grid[8]) - (grid[3] + grid[6] + grid[7])) / rho
    uy = ((grid[2] + grid[5] + grid[6]) - (grid[4] + grid[7] + grid[8])) / rho
    return rho,ux,uy

# fit stuff
def theo_Exp(x, v):
    return amplitude_global * np.exp(-v*k_y*k_y*x)

def theo_exp_with_variables(x,v,ky,amplitud):
    return amplitud * np.exp(-v * ky * ky * x)

# main body
def shear_wave_decay():
    '''
    Original Shear Wave simulatates the function an then fits the exponential decay to it

    Returns
    -------

    '''
    print("Shear Wave Decay")
    # shear wave
    x_values = k_y * np.arange(0,size_x)
    shear_wave = amplitude_global * np.sin(periode * x_values)
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
    v = 1/3 * (1/relaxation_global - 1/2)
    # some sort of -e-fkt
    u_theo = amplitude_global * np.exp(-v*k_y*k_y*x)

    ###
    param,cv = scipy.optimize.curve_fit(theo_Exp,x,amplitude_array)
    v_s = param[0]
    #print(v_s)
    #print(v)
    # visualize
    fig, ax = plt.subplots()
    textstr = '\n'.join((
        r'size = %d x %d' % (size_x,size_y ),
        r'omega = %.02f' % (relaxation_global,),
        r'amplitude = %.02f' % (amplitude_global,),
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

def shear_wave_decay_more(amplitude,relaxation,ky):
    '''
    Calls the shear_wave_decay with
    Parameters
    ----------
    amplitude
    relaxxation
    ky

    Returns
    -------

    '''
    # return Params
    v_theoretical = 0
    v_simualated = 0
    amplitude_array = []
    x_values = ky * np.arange(0, size_x)
    shear_wave = amplitude * np.sin(periode * x_values)

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
        collision_with_relaxation(grid, rho, ux, uy,relaxation)
        ###
        # analize the amplitude
        ux_fft = np.fft.fft(ux[int(size_x / 2), :])
        ampl = 2 / size_y * np.abs(ux_fft)
        ampl = np.max(ampl)
        amplitude_array.append(ampl)

    # v_theoretical
    x = np.arange(0, steps)
    v_theoretical = 1 / 3 * (1 / relaxation - 1 / 2)
    # some sort of -e-fkt
    amplitude_theo = amplitude * np.exp(-v_theoretical * ky * ky * x)

    # v_simulated
    # lambda wrapper for ky and amplitude
    param, cv = scipy.optimize.curve_fit(lambda x,v : theo_exp_with_variables(x,v,ky,amplitude), x, amplitude_array)
    v_simualated = param[0]

    return v_theoretical, v_simualated,amplitude_theo, amplitude_array

def rapid_call():
    print("Mass caller, Generate six")
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

def shear_wave_decay_fft_analyise(amplitude,relaxation,ky_factor):
    print("Fourier Analysis of the shear wave decay")
    # stuff for the basic simulation
    ky = k_y * ky_factor
    x_values = ky * np.arange(0, size_x)
    shear_wave = amplitude * np.sin(periode * x_values)

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
        collision_with_relaxation(grid, rho, ux, uy, relaxation)

    # fft analysiation
    # should only make sense after the loop as we do not recorde the amplitude all the time
    freq_y, transform_y =  do_fft_analysis(uy[int(size_x / 2), :])
    freq_x, transform_x =  do_fft_analysis(ux[int(size_x / 2), :])
    plt.plot(freq_x,transform_x, label = "ux")
    plt.plot(freq_y, transform_y, label = "uy")
    plt.legend()
    plt.show()

def shear_wave_different_times(amplitude,relaxation,ky_factor):
    print("Shear Wave Decay Fourier Analysis at different timesteps")
    # stuff for the basic simulation
    runs = 10000
    ky = ky_factor* k_y
    x_values = ky * np.arange(0, size_x)
    shear_wave = amplitude * np.sin(periode * x_values)

    # initizlize the gird
    rho = np.ones((size_x, size_y))
    ux = np.zeros((size_x, size_y))
    ux[:, :] = shear_wave
    uy = np.zeros((size_x, size_y))
    grid = equilibrium(rho, ux, uy)

    #
    plt.figure(figsize=(12,9), dpi = 100)
    # loop
    for i in range(runs +1):
        # standard procedure
        stream(grid)
        rho, ux, uy = caluculate_rho_ux_uy(grid)
        collision_with_relaxation(grid, rho, ux, uy, relaxation)
        # every 1000 runs do an analysis
        # plot it into one diagram only analyse ux


    # label_string = ""
    fig_size = (10 * 2.5, 8 * 2.5)
    axs = plt.figure(figsize=fig_size).subplots(2, 2)
    # calcs
    freq_x, fourier_x = do_fft_analysis(ux[int(size_x / 2), :])
    freq_y, fourier_y = do_fft_analysis(uy[int(size_x / 2), :])
    fourier_x = fourier_x/np.linalg.norm(fourier_x)
    fourier_y = fourier_y / np.linalg.norm(fourier_y)
    ##
    axs[0, 0].plot(freq_x,fourier_x)
    axs[0, 0].set_xlabel("Wave number")
    axs[0, 0].set_ylabel("Amplitude vx(ky)")
    ##
    axs[1, 0].plot(freq_y,fourier_y)
    axs[1, 0].set_xlabel("Wave number")
    axs[1, 0].set_ylabel("Amplitude vy(ky)")
    ###
    freq_x, fourier_x = do_fft_analysis(ux[: ,int(size_x / 2)])
    freq_y, fourier_y = do_fft_analysis(uy[: ,int(size_x / 2)])
    fourier_x = fourier_x / np.linalg.norm(fourier_x)
    fourier_y = fourier_y / np.linalg.norm(fourier_y)
    ####
    axs[0, 1].plot(freq_x, fourier_x)
    axs[0, 1].set_xlabel("Wave number")
    axs[0, 1].set_ylabel("Amplitude vx(kx)")
    ##
    axs[1, 1].plot(freq_y, fourier_y)
    axs[1, 1].set_xlabel("Wave number")
    axs[1, 1].set_ylabel("Amplitude vy(kx)")
    title_string = "Amplitude {}".format(amplitude) \
                   + " ,relaxation {}".format(relaxation) + \
                   " , {}*ky".format(ky_factor) \
                   + ", size {}".format(size_x)
    plt.suptitle(title_string)
    plt.show()


def shear_wave_decay_return(amplitude,relaxation,ky_factor):
    # stuff for the basic simulation
    ky = k_y * ky_factor
    x_values = ky * np.arange(0, size_x)
    shear_wave = amplitude * np.sin(periode * x_values)

    # initialize the gird
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
        collision_with_relaxation(grid, rho, ux, uy, relaxation)

    return ux, uy

def analyse_different_values():
    print("Analyse diffrent k_ys")
    # call patterns
    num_of_patterns = 8
    amplitude = 0.1
    relaxation = 0.2
    amplitude_call_pattern = np.ones(num_of_patterns) * amplitude
    relaxation_call_pattern = np.ones(num_of_patterns) * relaxation
    ky_factor_call_pattern = (np.arange(num_of_patterns)+1) * 2
    # save bins
    ux_bin = []
    uy_bin = []
    freq_x_bin = []
    fourier_x_bin = []
    freq_y_bin = []
    fourier_y_bin = []
    # run all patterns
    for i in range(num_of_patterns):
        # call function
        ux, uy = shear_wave_decay_return(amplitude_call_pattern[i],relaxation_call_pattern[i],ky_factor_call_pattern[i])
        # only save the value in the middle the  rest can be discarded
        ux_bin.append(ux[int(size_x / 2), :])
        uy_bin.append(uy[int(size_x / 2), :])
    # do a fft analysis
    for i in range(num_of_patterns):
        freq_x, fourier_x = do_fft_analysis(ux_bin[i])
        freq_y, fourier_y = do_fft_analysis(uy_bin[i])
        # append
        freq_x_bin.append(freq_x)
        freq_y_bin.append(freq_y)
        fourier_x_bin.append(fourier_x)
        fourier_y_bin.append(fourier_y)
    # plotting
    x = 0
    y = 0
    fig_size = (10 * 2.5, 8 * 2.5)
    axs = plt.figure(figsize=fig_size).subplots(4, 2)
    for i in range(num_of_patterns):
        # actual plotting
        axs[y, x].plot(freq_x_bin[i],fourier_x_bin[i],label = "ux")
        axs[y, x].plot(freq_y_bin[i],fourier_y_bin[i],label = "uy")
        title_string = "Amplitude {}".format(amplitude_call_pattern[i]) \
                       + " ,relaxation {}".format(relaxation_call_pattern[i]) + \
                       " , {}*ky".format(ky_factor_call_pattern[i]) \
                       +", size {}".format(size_x)
        axs[y,x].set_title(title_string)
        axs[y, x].set_xlabel("Frequency")
        axs[y, x].set_ylabel("Amplitude")
        axs[y,x].legend()
        # counting
        x += 1
        if x == 2:
            x = 0
        if (i + 1) % 2 == 0 and i != 0:
            y += 1

    # dont forget
    plt.show()



def plotter_shear_wave():
    sample_freq = size_x
    sample_time  = 1/sample_freq
    amplitude = 0.1
    ky = k_y
    x_values = ky * np.arange(0, size_x)
    shear_wave = amplitude * np.sin(periode * x_values)
    fourier_transform = np.fft.fft(shear_wave) / len(shear_wave)
    fourier_transform = fourier_transform[range(int(len(shear_wave) / 2))]
    tp_count = len(shear_wave)
    values = np.arange(int(tp_count) / 2)
    time_period = tp_count / 100
    freq = values / time_period
    plt.plot(freq[0:10], abs(fourier_transform[0:10]))
    plt.show()

def example_fft():
    sampel_freq = 100
    sample_time = 0.01
    t = np.arange(0,10,sample_time)
    signal1_freq = 3
    signal2_freq = 9
    amplitude1 = np.sin(2*np.pi*signal1_freq*t)
    amplitude2 = np.sin(2*np.pi*signal2_freq*t)
    ampitude = amplitude1 + amplitude2
    fourier_transform = np.fft.fft(ampitude)/len(ampitude)
    fourier_transform = fourier_transform[range(int(len(ampitude)/2))]
    tp_count = len(ampitude)
    values = np.arange(int(tp_count)/2)
    time_period = tp_count/sampel_freq
    freq = values/time_period
    plt.plot(freq,abs(fourier_transform))
    plt.show()

def do_fft_analysis(signal):
    sample_freq = len(signal)
    sample_time = 1 / sample_freq
    fourier_transform = np.fft.fft(signal) / len(signal)
    # fourier_transform = fourier_transform[range(int(len(signal) / 2))]
    tp_count = len(signal)
    values = np.arange(int(tp_count) / 2)
    time_period = tp_count / sample_freq
    freq = values / time_period
    return freq, abs(fourier_transform)


# calls
# shear_wave_decay()
# rapid_call()
shear_wave_different_times(0.3,0.2,10)
# analyse_different_values()
#plotter_shear_wave()
#example_fft()

