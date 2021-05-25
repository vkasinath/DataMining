#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import math
import random


"""
global variables
"""

train_pct = 0.7  # rest will be test_pct

ETA = 0.005
EPS = 0.01
MAXITER = 10000


"""
function to get input parameters from argv
"""
def get_inputs():

    global ETA, EPS, MAXITER

    filename = sys.argv[1]
    if (len(sys.argv)> 4):
        ETA = float(sys.argv[2])
        EPS = float(sys.argv[3])
        MAXITER = int(float(sys.argv[4]))
    elif (len(sys.argv)> 3):
        ETA = float(sys.argv[2])
        EPS = float(sys.argv[3])
    elif (len(sys.argv)> 2):
        ETA = float(sys.argv[2])

    return (filename)


"""
process input file into a dataframe
clean up the columns
"""
def process_data(filename):

    df = pd.read_csv(filename, sep=',')

    # remove the first column and the last column (date and rv2)
    del df['date']
    del df['rv2']

    return df


"""
given Y => actual, and P => predicted
calculate accuracy
"""
def calcAccuracy(Y, P):
    
    n = Y.shape[0]
    predicted = np.empty(Y.shape[0])

    for i in range(n):
        if (Y[i] == P[i]):
            predicted[i] = 1
        else:
            predicted[i] = 0
                
    return (np.sum(predicted)/n)

"""
Predict outcome for each x, using the weights vector w
"""
def calcPredict(w, X):

    n = X.shape[0]
    Y = np.empty(n)
    
    for i in range(n):
        p = w.transpose().dot(X[i])
        if (p <= -700):
            theta = 0.0
        else:
            theta = 1.0/(1 + np.exp(-1*p))

        if (theta >= 0.5):
            Y[i] = 1
        elif (theta < 0.5):
            Y[i] = 0

    return Y


"""
Prepare training and test data
set the columns for intercept,
seperate out dependent variable (Y)
and return numpy matrix of independent variables (dat)
and dependent variable (datY)
"""
def prepData(df):

    # convert data from pandas df to numpy matrix
    dat = df.to_numpy()
    # print(train)

    # update first column (1 if value <= 50, 0 if value > 50)
    dat[dat[:,0] <= 50, 0] = 1; dat[dat[:,0] > 50,  0] = 0
    
    # store the first column from dat into datY
    datY = dat[:, 0:1]

    # yank the first column from dat, and put 1's as the first column in dat
    dat = np.delete(dat, 0, axis=1)

    # add 1 as the first column (for w0)
    dat = np.hstack((np.ones((dat.shape[0],1)), dat))

    return dat, datY


"""
given an array of actuals and its counterpart predictions
build out a pie chart for the following parts
act=pred, act=1, pred=0, act=0, pred=1
"""
def calcBuildPlot(Y, P, ax, title):

    p11 = 0
    p00 = 0
    p10 = 0
    p01 = 0
    for i in range(Y.shape[0]):
        if (Y[i] == P[i] and Y[i] == 1):
            p11 = p11 + 1
        elif (Y[i] == P[i] and Y[i] == 0):
            p00 = p00 + 1
        elif (Y[i] != P[i] and Y[i] == 1):
            p10 = p10 + 1
        elif (Y[i] != P[i] and Y[i] == 0):
            p01 = p01 + 1

    labels=['act = pred', 'act=1, pred=0','act=0, pred=1']
    vals = [p11+p00, p10, p01]
    explode = [0.1, 0, 0]
    ax.set_title(title)
    ax.pie(vals, labels=labels, explode=explode, autopct='%1.1f%%', shadow=True)

    return ax



if __name__ == "__main__":
    
    random.seed(200)
    
    filename = get_inputs()                                                 # get input filename and other parameters
    df = process_data(filename)                                             # open the file and load into a dataframe
    
    # Split data into train (train_pct), and test(1-train_pct)
    traindf =df.sample(frac=train_pct,random_state=200)                     # random 70% training data. random_state = a seed value
    testdf  =df.drop(traindf.index)                                         # remaining 30% test data

    # prepare training data for traindf
    # prepare test data for testdf
    train, trainY = prepData(traindf)
    test, testY   = prepData(testdf)

    # create weights vector with zeros
    w = np.zeros((train.shape[1],1))

    n = train.shape[0]
    # start iteration
    iter = 0
    while (iter < MAXITER):
        wn = w.copy()
        # compute gradient at i
        # generate non repetitive i from 0 to 26 for each iteration (randomized looping)
        r27 = random.sample(range(0, 27), 27)
        for i in r27:
            c1 = wn.transpose().dot(train[i])

            if (c1 <= -700):    # control overflow error with exp(...)
                theta = 0.0
            else:
                theta = 1.0/(1 + np.exp(-1*c1))
            grad = (trainY[i]-theta)*train[i]
            wn = np.add(wn.transpose(), ETA*grad).transpose()

        c_err = abs(np.linalg.norm(wn - w))
        if c_err <= EPS:
            break
    
        w = wn.copy()
        iter = iter + 1

    with np.printoptions(precision=4, suppress=True):
        print("Parameters: ", u'\u03B7', "=", ETA, ",", u'\u03B5',"=", EPS, ", MaxIter =", MAXITER)
        print("iterations taken =", iter, "final \u03B5 =", np.around(c_err,4))
        print("Weights Vector =\n", w.transpose()[0])


    # predict for train records using the weights
    trainP = calcPredict(w,train)

    # calculate accuracy for trainY
    trainAccuracy = calcAccuracy(trainY, trainP)
    print("Accuracy (train) % =", np.around(trainAccuracy*100.0, 4))

    # predict for test records using the weights
    testP = calcPredict(w,test)

    # calculate accuracy for testY
    testAccuracy = calcAccuracy(testY, testP)
    print("Accuracy (test ) % =", np.around(testAccuracy*100.0, 4))

    # plot train and test results as pie charts

    fig, (ax) = plt.subplots(1,2)                                       # set up plot framework
    fig.canvas.set_window_title('Assignment 4') 
    plt.tight_layout()


    # actual vs predicted pie chart for test and train
    ax[0] = calcBuildPlot(trainY, trainP, ax[0], "Train: Actual vs. Predicted comparison")
    ax[1] = calcBuildPlot(testY,  testP,  ax[1], "Test : Actual vs. Predicted comparison")
    plt.show()
