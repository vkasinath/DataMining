#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import math

# configuration variables (nr= num records, deg = poly degree, const = poly constant)
nr = 10000                                                          # default records to use
deg = 1.0                                                           # default poly deg (1 = linear)
const = 0.0                                                         # default poly const
spread = 18000                                                      # default spread value
alpha = 0.95                                                        # default alpha value

def KernelMatrix(X, type, *args):
    # spread = sigma^2

    row, col = X.shape                                              # shape of matrix X
    K = np.zeros((row, row))                                        # create matrix K of zeros

    if (type == "gauss"):                                           # guassian kernel matrix
        spread = float(args[0])                                     # get spread value from optional arguments, for type = "gauss"
        for i in range(row):
            for j in range(i, row):
                if (i == j):
                    val = 1.0                                       # diagonal element [i,i], set value to 1.0
                else:
                    xt = X[i] - X[j]                                
                    val = xt.dot(np.transpose(xt))
                    val = math.exp(-1*val/(2*spread))

                K[i,j] = val
                K[j,i] = val

    elif (type == "poly"):
        deg = int(args[0])                                          # get deg and const value from optional arguments, for type = "poly"
        const = float(args[1])

        for i in range(row):
            for j in range(i, row):
                xt = np.transpose(X[i])                             # transpose row i
                val = xt.dot(X[j])                                  # dot product with row j
                val = val + const                                   # add constant, if provided
                val = np.power(val, deg)                            # power-of(deg)
                K[i,j] = val                                        # same value for both positions [I,J] and [J,I]
                K[j,i] = val
    return(K)



def CenterKernel(K):
    # pg 164
    row, col = K.shape
    I = np.zeros((row,row))
    for i in range(row):
        I[i,i] = 1.0

    I_bar = I - (1/row)
    K_bar = np.dot(I_bar, K)
    K_bar = np.dot(K_bar, I_bar)
    return(K_bar)


def ReducedDimensionality(Cr, K_bar):

    Crt = np.transpose(Cr)
    A = np.dot(Crt, K_bar)
    return (A)




if __name__ == "__main__":

    # get inputs
    filename = sys.argv[1]
    if (len(sys.argv)> 3):
        alpha = float(sys.argv[2])
        spread = float(sys.argv[3])
    elif (len(sys.argv)> 2):
        alpha = float(sys.argv[2])

    # open the file and load into a dataframe

    df = pd.read_csv(filename, sep=',')

    # remove the first column and the last column (date and rv2)
    del df['date']
    del df['rv2']
    #print(df)

    # convert pandas dataframe to numpy matrix and trim size
    ndf = df.to_numpy()
    if (ndf.shape[0] > nr):
        Trimdf = ndf[:nr, :]
    else:
        nr = ndf.shape[0]
        Trimdf = ndf

    """
    PCA by linear KPCA method
    """

    K1 = KernelMatrix(Trimdf, "poly", deg, const)                       # calculate Kernel Matrix
    #print("K = ", K1, "\n")

    K_bar1 = CenterKernel(K1)                                           # center Kernel Matrix
    #print("K_bar = ", K_bar1, "\n")

    ev1, evect1 = np.linalg.eigh(K_bar1)                                # calculate eigen values and eigen vectors
    ev1 = -np.sort(-ev1)                                                # sort eigen values descending
    evect1 = np.fliplr(evect1)                                          # since we sorted our eigen values, flip our eigen vectors L-to-R

    ev1[abs(ev1) < 0.001] = 0.0                                         # very low values of e_val -> set to zero
    nz1 = np.where(ev1 == 0.0)[0][0]                                    # capture the first occurrence of zero eigen value

    lamda1 = ev1/nr                                                     # calculate variance for each component

    ci = np.zeros((nr,nr))                                              # calculate ci
    for i in range(nr):
        if (ev1[i] > 0):                                                # only for non-zero eigen value. else leave it at zero
            ci[i] = evect1[i]/(math.sqrt(ev1[i]))
    #print("ci = ", ci, "\n")

    tot_lamda = sum(lamda1)                                             # calculate denominator - to scale ev
    frac_var = lamda1/tot_lamda                                         # fraction of total variance
    for i in range(nr):
        if (i > 0):
            frac_var[i] = frac_var[i] + frac_var[i-1]                   # calculate f(r) = cumsum of ith frac_var, for non-zero frac_vars

    #print("frac_var = ", frac_var, "\n")
    r1 = np.where(frac_var >= alpha)[0][0]                              # smallest r where frac_var[r] >= alpha, choose dimensionality
    Cr = ci[:,:r1+2]                                                    # reduced basis choose -> first 2 principal components
    #print("Cr = ", Cr, "\n")

    A1 = ReducedDimensionality(Cr, K_bar1)                              # reduced dimensionality
    #print("A = ", A1)

    """
    PCA by covariance method (repeat of prior homework)
    """

    # dfnr = df.head(nr)
    dfnr = df
    dfm = dfnr.mean(axis=0)                                             # calculate column means (means vector)

    cdm = (dfnr - dfm)                                                  # create the centered data matrix
    tcdm = cdm.transpose()                                              # transpose centered data matrix

    scm_ip = (tcdm.dot(cdm))/len(tcdm.columns)                          # calc covariance matrix scm (inner product method)

    ev2, evect2 = np.linalg.eigh(scm_ip)                                # calculate eigen values and eigen vectors
    ev2 = -np.sort(-ev2)                                                # sort eigen values descending
    evect2 = np.fliplr(evect2)                                          # since we sorted our eigen values, flip our eigen vectors L-to-R

    tot_var = sum(ev2)                                                  # calculate denominator - to scale e_val
    frac_var = ev2/tot_var                                              # scale each e_val

    n = len(ev2)
    for i in range(n):
        if (i > 0):
            frac_var[i] = frac_var[i] + frac_var[i-1]                   # calculate f(r) = cumsum of ith frac_var

    r2 = np.where(frac_var >= alpha)[0][0]                              # smallest r where frac_var[r] >= alpha, choose dimensionality

    pc_r = evect2[:,:r2+2]                                                # reduced basis
    df_t = np.transpose(cdm.to_numpy())
    A2 = ReducedDimensionality(pc_r, df_t)                              # reduced dimensionality data
    #print("A = ", A2)



    """
    PCA by Gaussian KPCA
    """

    # Gaussian kernel matrix
    deg = 1.0
    const = 0.0
    K2 = KernelMatrix(Trimdf, "gauss", spread)                          # calculate Kernel Matrix
    #print("K = ", K2, "\n")

    K_bar2 = CenterKernel(K2)                                           # center Kernel Matrix
    #print("K_bar = ", K_bar2, "\n")

    ev3, evect3 = np.linalg.eigh(K_bar2)                                # calculate eigen values and eigen vectors
    ev3 = -np.sort(-ev3)                                                # sort eigen values descending
    evect1 = np.fliplr(evect1)                                          # since we sorted our eigen values, flip our eigen vectors L-to-R

    ev3[abs(ev3) < 0.001] = 0.0                                         # very low values of e_val -> set to zero
    nz3 = np.where(ev3 == 0.0)[0][0]                                    # capture the first occurrence of zero eigen value

    lamda2 = ev3/nr                                                     # calculate variance for each component

    ci = np.zeros((nr,nr))                                              # calculate ci
    for i in range(nr):
        if (ev3[i] > 0):
            ci[i] = evect3[i]/(math.sqrt(ev3[i]))
    #print("ci = ", ci, "\n")

    tot_lamda = sum(lamda2)                                             # calculate denominator - to scale ev
    frac_var = lamda2/tot_lamda                                         # fraction of total variance
    for i in range(nr):
        if (i > 0):
            frac_var[i] = frac_var[i] + frac_var[i-1]                   # calculate f(r) = cumsum of ith frac_var, for non-zero frac_vars

    #print("frac_var = ", frac_var, "\n")
    r3 = np.where(frac_var >= alpha)[0][0]                              # smallest r where frac_var[r] >= alpha, choose dimensionality

    Cr = ci[:,:r3+2]                                                    # reduced basis choose -> first 2 principal components
    #print("Cr = ", Cr, "\n")

    A3 = ReducedDimensionality(Cr, K_bar2)                              # redcued dimensionality
    #print("A = ", A3)

    # print out final results and plot

    with np.printoptions(precision=4, suppress=True):
        print("Non-zero Linear KPCA   eigen values = ", ev1[:nz1], "\n")         # print non-zero evs, for linear kpca
        print("Co-Variance eigen values = ", ev2, "\n")                          # print non-zero evs, for co-variance method
        print("Non-zero Gaussian KPCA eigen values = ", ev3[:nz3], "\n")         # print non-zero evs, for gaussian kpca

        print("Linear KPCA Minimum # of dimensions for  ", u'\u03B1', u'\u2265', alpha, '=', r1+1, "\n")    # print out alpha bound PCA
        print("Co-Var Minimum # of dimensions for       ", u'\u03B1', u'\u2265', alpha, '=', r2+1, "\n")    # print out alpha bound PCA
        print("Gaussian KPCA Minimum # of dimensions for", u'\u03B1', u'\u2265', alpha, ", with spread of", spread, "=", r3+1, "\n")    # print out alpha bound PCA


    # scatter plot for linear KPCA, Co-Var PCA, and Gaussian KPCA
    fig, (ax) = plt.subplots(1,3)                                       # set up plot framework
    fig.suptitle("KPCA(Linear), PCA, KPCA(Gaussian) Comparision")
    fig.canvas.set_window_title('Assignment 3, Part I') 
    plt.tight_layout()

    ax[0].set_title("KPCA Linear Scatterplot - u1 vs u2")
    ax[0].scatter(A1[0,:], A1[1,:], c='r', edgecolors='k', linewidth=0.25)
    ax[0].set_ylabel("u2")
    ax[0].set_xlabel("u1")
    ax[0].grid()

    ax[1].set_title("Co-Varinace PCA Scatterplot - u1 vs u2")
    ax[1].scatter(A2[0,:], A2[1,:], c='b', edgecolors='k', linewidth=0.25)
    ax[1].set_ylabel("u2")
    ax[1].set_xlabel("u1")
    ax[1].grid()

    ax[2].set_title("Gaussian KPCA Scatterplot " + u'\u03C3\u00B2' + " = " + str(int(spread)) + " - u1 vs u2")
    ax[2].scatter(A3[0,:], A3[1,:], c='g', edgecolors='k', linewidth=0.25)
    ax[2].set_ylabel("u2")
    ax[2].set_xlabel("u1")
    ax[2].grid()

    plt.show()
