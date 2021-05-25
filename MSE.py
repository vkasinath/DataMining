#!/usr/bin/env python
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys


if __name__ == "__main__":
	
	# get inputs
	filename = sys.argv[1]

	if (len(sys.argv) > 2):
		alpha = sys.argv[2]
	else:
		alpha = 0.975


	# read and process the file
	data_matrix = pd.read_csv(filename, sep=',')  
	#print(df)   

	# remove the first and last column
	del data_matrix['date']
	del data_matrix['rv2']
	#print(df)

	# calculate and print mean vector
	mean_vector = data_matrix.mean()
	# print("Mean Vector = ", "\n", mean_vector.to_numpy(), "\n")

	# create centered data matrix
	centered_dm = data_matrix - mean_vector
	tcentered_dm = centered_dm.transpose()

	# calculate and print sample covariance matrix as inner products
	scmatrix_ip = (tcentered_dm.dot(centered_dm))/len(tcentered_dm.columns)
	# print("Covariance Matrix (Inner Products) = " , "\n", scmatrix_ip, "\n")

	# calculate eigen values and eigen vector
	e_val, evector = np.linalg.eigh(scmatrix_ip)

	# sort values and flip vector
	e_val = -np.sort(-e_val)

	evector = np.fliplr(evector)

	with np.printoptions(precision=4, suppress=True):
		print("Eigen values =", end='')
		print(e_val)
		print("\n")

	# compute fraction of total variance
	tot_var = sum(e_val)

	frac_var = e_val / tot_var

	mse = e_val

	for i in range(len(e_val)):
		if i > 0:
			frac_var[i] = frac_var[i] + frac_var[i-1]
			mse[i] = mse[i] + mse[i-1]

	for i in range(len(e_val)):
		if frac_var[i] >= alpha:
			break

	print("Minimum Number of Dimensions for", u'\u03B1', u'\u2265', alpha, '=', i+1, "\n")

	print("MSE for first 3 components =", np.around(mse[len(mse) -1] - mse[2]), "\n")

	pc16 = evector[:,:6]

	tpc16 = np.transpose(pc16)

	A = np.matmul(tpc16, tcentered_dm.to_numpy())

	plt.scatter(A[0,:], A[1,:], c='r', edgecolors='k', linewidth=0.5) 
	plt.xlabel('U1')
	plt.ylabel('U2')
	plt.grid()

	plt.show()