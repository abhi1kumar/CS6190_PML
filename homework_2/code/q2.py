
import os
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from numpy import genfromtxt
from scipy.special import expit as logistic
from scipy.stats import norm
from scipy.optimize import minimize

from common import *

MAX_ITER = 100
TOLERANCE = 1e-5
REG = 1

def readcsv(path):
	my_data = genfromtxt(path, delimiter=',')
	return my_data

def train_model(train_data, train_label, test_data, test_label, reg= REG, max_iter= MAX_ITER, tolerance= TOLERANCE, init= "zeros", regression= "logistic", solver= "Newton"):	
	print("\n===================== Training {} Regression====================".format(regression))
	
	# Decide Initialization
	if init == "zeros":
		print("Weights initialized to zeros")
		weight = np.zeros((train_data.shape[1], 1))
	elif init == "random":
		print("Weights initialized to random normal")
		# Xavier initialization
		# Reference Understanding the difficulty of training deep feedforward neural networks, GLorot and Bengio, JMLR 
		weight = np.random.randn(train_data.shape[1],1)/np.sqrt(train_data.shape[1])

	# Decide Solver
	if solver== "Newton":
		print("Using Newton Raphson Solver\n")
		for i in range(max_iter):
			# Get training data prediction
			pred, dot_product = get_output(weight, train_data, regression= regression)
			# Get gradient and hessian
			gradient = get_gradient(phi= train_data, pred= pred, t= train_label[:, np.newaxis], dot_product= dot_product, weight= weight, reg= reg, regression= regression)
			hessian  = get_hessian (phi= train_data, pred= pred, t= train_label[:, np.newaxis], dot_product= dot_product, reg= reg, regression= regression)
			# Update weights
			weight_new = weight - np.matmul(np.linalg.inv(hessian), gradient)
			
			# Difference between weights
			diff = np.linalg.norm(weight_new- weight)

			# Get accuracy
			acc, _ = predict_and_test(weight_new, test_data, test_label, regression= regression)

			weight = weight_new

			print("Iteration= {:3d} Diff_in_weight= {:.5f} Test_Acc= {:.2f}%".format(i, diff, acc))
			if diff < tolerance:
				#print(weight)
				print("Training converged. Done.")
				break

	elif solver == "BFGS":
		print("Using BFGS Solver\n")
		if regression == "probit":

			x0 = weight[:,0]
			param_probit = minimize(NLL_Probit, x0, jac= NLL_Probit_Derivative, method='BFGS', tol= TOLERANCE, options={'disp': True, 'maxiter': MAX_ITER})
			weight[:,0] = param_probit.x
		# Get accuracy
		acc, _ = predict_and_test(weight, test_data, test_label, regression= regression)
		print("Test_Acc= {:.2f}%".format(acc))	

def NLL_Probit(w):
	# We want to minimize the minimum of log posterior
	# w gets passed as one dimnesional array
	output,_ = get_output(w, train_data, regression= "probit")
	output = np.log(output + TOLERANCE)
	NLL = -np.sum(np.multiply(train_label, output) +  np.multiply(1-train_label, 1-output)) + np.sum(w**2)
	return NLL

def NLL_Probit_Derivative(w):
	# w gets passed as one dimnesional array
	weight = w[:, np.newaxis]
	pred, dot_product = get_output(weight, train_data, regression= "probit")
	gradient = get_gradient(train_data, pred, train_label[:, np.newaxis], dot_product, weight, reg= 1, regression= "probit")
	return gradient[:,0]

#==============================================================================
# Main starts here
#==============================================================================
train = readcsv("bank_note/train.csv")
test  = readcsv("bank_note/test.csv")
train_label = train[:,-1]
test_label  = test [:,-1]
train_data  = np.hstack((train[:,0:-1], np.ones((train.shape[0],1))))
test_data   = np.hstack((test [:,0:-1], np.ones((test.shape[0], 1))))

np.random.seed(19)

regression = "logistic"
# Train using Newton Raphson
train_model(train_data, train_label, test_data, test_label, regression= regression, init= "zeros")
train_model(train_data, train_label, test_data, test_label, regression= regression, init= "random")

regression = "probit"
# Train using LBFGS
train_model(train_data, train_label, test_data, test_label, regression= regression, solver= "BFGS", init= "zeros")
train_model(train_data, train_label, test_data, test_label, regression= regression, solver= "BFGS", init= "random")

# Train using Newton Raphson
train_model(train_data, train_label, test_data, test_label, regression= regression, init= "zeros")
train_model(train_data, train_label, test_data, test_label, regression= regression, init= "random")
