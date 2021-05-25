#!/usr/bin/env python
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys


if __name__ == "__main__":
    
    # get inputs
    filename = sys.argv[1]

    if (len(sys.argv) > 2):
        eps = sys.argv[2]
    else:
        eps = 0.001


    # read and process the file
    data_matrix = pd.read_csv(filename, sep=',')  
    #print(df)   

    # remove the first and last column
    del data_matrix['date']
    del data_matrix['rv2']
    #print(df)

    # calculate and print mean vector
    mean_vector = data_matrix.mean()
    print("Mean Vector = ", "\n", mean_vector.to_numpy(), "\n")

    # calculate and print total variance
    col_var = data_matrix.var()

    tot_var = sum(col_var)                                                     
    print("Total Var(D) = ", tot_var, "\n")  

    # create centered data matrix
    centered_dm = data_matrix - mean_vector
    tcentered_dm = centered_dm.transpose()

    # calculate and print sample covariance matrix as inner products
    scmatrix_ip = (tcentered_dm.dot(centered_dm))/len(tcentered_dm.columns)
    print("Covariance Matrix (Inner Products) = " , "\n", scmatrix_ip, "\n")

    # calculate and print sample covariance matrix as outer products
    n = len(centered_dm.index)

    for i in range(n):

        t1 = (centered_dm.iloc[i,:]).to_frame()
        t2 = t1.transpose()

        if (i == 0):
            p = t1.dot(t2)
        else:
            p = p.add(t1.dot(t2))

    scmatrix_op = p/n

    print("Covariance Matrix (Outer Product) = ", "\n", scmatrix_op, "\n")   

    # calculate and print correlation matrix
    n = len(scmatrix_ip.columns)
    cmatrix = scmatrix_ip.copy(deep=True)

    # loop through, transpose and update the correlation matrix
    for i in range(n):
        for j in range(n):
            c_ij = scmatrix_ip.iat[i,j]
            c_ii = scmatrix_ip.iat[i,i]
            c_jj = scmatrix_ip.iat[j,j]
            cmatrix.iat[i,j] = c_ij/np.sqrt(c_ii*c_jj)

    print("Correlation Matrix = ", "\n", cmatrix, "\n")  


  	# plot the most correlated, the most anti-correlated, the least correlated

    fig, (ax) = plt.subplots(1,3)
    ax[0].grid()
    ax[1].grid()
    ax[2].grid()

    # plot the 3 correlations

    # anti correlated
    a, b = cmatrix.stack().idxmin()
    print("Most Anti-Correlated = [", a, ", ", b, ",", cmatrix.loc[[a], [b]].values[0], "]\n")
    ax[0].scatter(data_matrix.loc[:,[a]], data_matrix.loc[:,[b]])
    ax[0].set_title("Most Anti-Correlated")
    ax[0].set_xlabel(a)
    ax[0].set_ylabel(b)

    x = data_matrix[a].to_numpy()
    y = data_matrix[b].to_numpy()
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    ax[0].plot(x,p(x),"k-")
    # strong negative correlation. data points are significantly clustered along negative trend line
    # may be driven by outliers on both ends, possibly skewing trend line to show negative correlation



    # least correlated
    n = len(cmatrix.columns)
    a, b = cmatrix.abs().stack().idxmin()
    print("Least Correlated = [", a, ", ", b, ",", cmatrix.loc[[a], [b]].values[0], "]\n")
    ax[1].scatter(data_matrix.loc[:,[a]], data_matrix.loc[:,[b]])
    ax[1].set_title("Least Correlated")
    ax[1].set_xlabel(a)
    ax[1].set_ylabel(b)

    x = data_matrix[a].to_numpy()
    y = data_matrix[b].to_numpy()
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    ax[1].plot(x,p(x),"k-")
    # data points show no directional clustering. no discernable pattern of data points.
    # visibility seem to be equally distributed (from 0 to 80) across all values of applications


    # max correlated
    cmatrix1 = cmatrix.copy(deep=True)
    for i in range(n):
        cmatrix1.iat[i,i] = -99

    a, b = cmatrix1.stack().idxmax()
    print("Most Correlated = [", a, ", ", b, ",", cmatrix.loc[[a], [b]].values[0], "]\n")
    ax[2].scatter(data_matrix.loc[:,[a]], data_matrix.loc[:,[b]])
    ax[2].set_title("Most Correlated")
    ax[2].set_xlabel(a)
    ax[2].set_ylabel(b)

    x = data_matrix[a].to_numpy()
    y = data_matrix[b].to_numpy()
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    ax[2].plot(x,p(x),"k-")
    # data points show very dense clustering around the trend line
    # very little evidence of outlier observations. very storng positive correlation


    #calculate and print dominant eigen vector
    itr = 0                                                            
    n = len(scmatrix_ip.columns)
    p0 = pd.DataFrame(np.ones(shape=(n,1)))
    L0 = 1
    delta = 1
    while (itr < 100 and delta > eps):
        itr = itr + 1
        p1 = pd.DataFrame(np.dot(scmatrix_ip.transpose(), p0))
        i = p1.idxmax()
        L1 = p1.iat[i[0],0]

        p1 = (1/L1)*p1
        delta = np.linalg.norm((p1-p0),ord=1)
        if abs(delta) <= eps:
            break;
        else:
            p0 = p1.copy(deep=True)
            L0 = L1

    p1 = p1*L1
    p1 = p1.div(np.linalg.norm(p1))

    print("Dominant Eigen Value  = ", L1, "\n")
    print("Dominant Eigen Vector =")
    print(p1.to_string(header=False,index=False))


    plt.show()
