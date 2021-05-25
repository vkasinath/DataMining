from random import seed
from random import random
import math
import matplotlib.pyplot as plt
import numpy as np

def gen_half_diag_pairs(dim):
    
    # set up first and second diagonal
    diag1 = [0 for i in range(dim)]
    diag2 = [0 for i in range(dim)]

    for i in range(dim):
        diag1[i] = (random()*2.0) - 1
        diag2[i] = (random()*2.0) - 1

    # calculate angle theta
    numer = np.dot(diag1, diag2)
    denom = np.linalg.norm(diag1)*np.linalg.norm(diag2)
    cos = numer/denom
    deg = np.degrees(np.arccos(cos))
    return deg
    

if __name__ == "__main__":

    
	# random seed
    sd = 1599558859
    seed(sd)

    # number of half-diagonal pairs
    n = 100000

    # dimensions in hyper cube
    d = [10, 100, 1000]

    # bins for different possible angles on histogram
    bin_list = [i for i in range(-1, 181, 1)]
    
    # set up 3 subplots
    str_title = "Probability Mass Function"
    fig, (ax) = plt.subplots(1,len(d))
    fig.suptitle(str_title+", N="+str(n))
    fig.canvas.set_window_title('Assignment 2, Part II, PMF') 

    plt.tight_layout()
    plt.title(str_title)
    cl = ['r','y','g']
    for i in range(len(d)):
        ax[i].grid()

    # for each hypercube of dimension d, generate n pairs and calculate angle
    for i in range(len(d)):
        deg = [gen_half_diag_pairs(d[i]) for j in range(n)] 

        # collect statistics
        dmin   = np.around(np.amin(deg),3)
        dmax   = np.around(np.amax(deg),3)
        drange = np.around(np.ptp(deg),  3)
        dmean  = np.around(np.average(deg), 3)
        dvar   = np.around(np.var(deg), 3)
        dstd   = np.around(np.std(deg), 3)

        # print out statistics
        print("n        =", n)
        print("d        =", d[i])
        print("Min      =", dmin)
        print("Max      =", dmax)
        print("Range    =", drange)
        print("Mean     =", dmean)
        print("Variance =", dvar)
        print("-----------------------\n")

        # plot PMF for d[i] hyper-cube half-diagonal pairs
        plt_str = "d = " + str(d[i])
        ax[i].set_title(plt_str)
        ax[i].set_ylabel("Probability")
        ax[i].set_xlabel("Angle")
        ax[i].hist(deg, bins=bin_list,density=True, color=cl[i%len(cl)])
        ax[i].set_xticks(np.arange(0, 181, 15.0))

        # create and plot distribution code
        x = np.copy(bin_list)
        y = 1/(dstd * np.sqrt(2 * np.pi)) * np.exp( - (bin_list - dmean)**2 / (2 * dstd**2) )
        ax[i].plot(x,y, linewidth=2, color='k')
        
    plt.show()
