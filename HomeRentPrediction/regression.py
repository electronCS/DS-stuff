import numpy as np
import csv

fileName = 'Rent.csv'
X,Y = [],[] 
with open(fileName, newline='') as csvfile:
	csvReader = csv.reader(csvfile, delimiter=',')
	header = next(csvReader)
	print(header)
	for line in csvReader:
		X.append(line[0])
		Y.append(line[1])

X = np.asarray(X, dtype=float)
Y = np.asarray(Y, dtype=float)

x_mat = np.column_stack((np.ones(len(X)), X))
x_mat_squared = np.dot(np.transpose(x_mat), x_mat)

beta_mat = np.dot(np.linalg.inv(x_mat_squared), np.dot(np.transpose(x_mat), Y))

# we can try to estimate error using LOOCV

print(beta_mat)

