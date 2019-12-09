import os
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import pandas as pd
from scipy.optimize import minimize

from common import *


MAX_ITER   = 100
TOLERANCE  = 1e-5
NUM_CLASSES= 4

def readcsv(path):
	# genfromtxt does not work with strings
	my_data = pd.read_csv(path).values
	return my_data

def change_attributes(train_data):
	# Ignore the label
	num_attributes = np.array([4, 4, 4, 3, 3, 3])
	values_attributes = [['vhigh', 'high', 'med', 'low'], ['vhigh', 'high', 'med', 'low'], ['2', '3', '4', '5more'], ['2', '4', 'more'], ['small', 'med', 'big'], ['low', 'med', 'high']]
	label_attributes = ['unacc', 'acc', 'good', 'vgood']

	# Last one is for the label
	train_mapped = np.zeros((train_data.shape[0], np.sum(num_attributes) + len(label_attributes)))

	for i in range(num_attributes.shape[0]):
		for j in range(len(values_attributes[i])):
			key = values_attributes[i][j]
			mapping_col = int(np.sum(num_attributes[0:i]) + j)
			row_index = train_data[:,i] == key
			train_mapped[row_index, mapping_col] = 1

	# Label attributes
	for j in range(len(label_attributes)):
		key = label_attributes[j]
		mapping_col = int(np.sum(num_attributes)) + j
		row_index = train_data[:,num_attributes.shape[0]] == key
		train_mapped[row_index, mapping_col] = 1

	return train_mapped

def NLL_multiclass_gradient(w):
	# w gets passed as one dimnesional array
	w = w.reshape((-1, NUM_CLASSES))
	pred, dot_product = get_output(w, train_data, regression= "multiclass")
	gradient = get_gradient(train_data, pred, train_label, dot_product, w, regression= "multiclass")
	w = gradient.reshape((-1))
	return w

def NLL_logistic_gradient(w):
	# w gets passed as one dimnesional array
	pred, dot_product = get_output(w, train_data, regression= "logistic")
	gradient = get_gradient(train_data, pred, train_label[:,i], dot_product, w, regression= "logistic")
	w = gradient
	return w

def NLL_multiclass(w):
	# We want to minimize the minimum of log posterior
	# w gets passed as one dimnesional array
	w = w.reshape((-1, NUM_CLASSES))
	output,_ = get_output(w, train_data, regression= "multiclass")
	output = np.log(output + TOLERANCE)
	NLL = -np.sum(np.multiply(train_label, output))  + np.sum(w**2)
	return NLL

def NLL_logistic(w):
	output,_ = get_output(w, train_data, regression= "logistic")
	output = np.log(output + TOLERANCE)
	NLL = -np.sum(np.multiply(train_label[:,i], output) +  np.multiply(1-train_label[:,i], 1-output)) + np.sum(w**2)
	return NLL

#==============================================================================
# Main starts here
#==============================================================================
train = readcsv("car/train.csv")
test  = readcsv("car/test.csv")

print(train.shape)
print(test.shape)

print("Converting Categorical to Binary Features...")
train = change_attributes(train)
test = change_attributes(test)
print(train.shape)
print(test.shape)

train_label = train[:,-NUM_CLASSES:]
test_label  = test [:,-NUM_CLASSES:]
train_data  = np.hstack((train[:,0:-NUM_CLASSES], np.ones((train.shape[0],1))))
test_data   = np.hstack((test [:,0:-NUM_CLASSES], np.ones((test.shape[0], 1))))

np.random.seed(123)

print("\n================ Training single MAP Multiclass Classifier using BFGS ==================")
weight = np.zeros((train_data.shape[1], NUM_CLASSES))
#weight = np.random.randn(train_data.shape[1], NUM_CLASSES)

x0 = weight.reshape((-1))
param_multiclass = minimize(NLL_multiclass, x0, jac= NLL_multiclass_gradient, method='BFGS', tol= TOLERANCE, options={'disp': False, 'maxiter': MAX_ITER})
weight = param_multiclass.x.reshape((-1, NUM_CLASSES))
acc,_ = predict_and_test(weight, test_data, test_label, regression= "multiclass")
print("Test_Acc= {:.2f}%".format(acc))


print("\n================ Training multiple MAP Logistic Classifiers using BFGS =================")
weight = np.zeros((train_data.shape[1], NUM_CLASSES))
#weight = np.random.randn(train_data.shape[1], NUM_CLASSES)
predictions = np.zeros((test_data.shape[0], NUM_CLASSES))

for i in range(NUM_CLASSES):
	print("\nTraining classifier {}".format(i+1))
	x0 = weight[:,i]
	param = minimize(NLL_logistic, x0, jac= NLL_logistic_gradient, method= 'BFGS', tol= TOLERANCE, options= {'disp': False, 'maxiter': MAX_ITER})
	weight[:,i] = param.x
	acc, pred_temp = predict_and_test(weight[:,i], test_data, test_label[:,i], regression= "logistic")
	predictions[:,i] = pred_temp
	print("Test_Acc= {:.2f}%".format(acc))

print("\nArgmax over different logistic values is considered as true label")
acc,_ = predict_and_test(weight, test_data, test_label, regression= "multiclass")
print("Test_Acc= {:.2f}%".format(acc))