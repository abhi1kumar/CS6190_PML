

import numpy as np
from matplotlib import pyplot as plt
from numpy import genfromtxt
from scipy.special import expit as logistic

TOLERANCE = 1e-5

def savefig(plt, path, show_message= True, tight_flag= True, newline= True):
    """
        Saves a plot object with showing message
    """
    if show_message:
        print("Saving to {}".format(path))
    if tight_flag:
        plt.savefig(path, bbox_inches='tight', pad_inches=0)
    else:
        plt.savefig(path)
    if newline:
        print("")

def readcsv(path, delimiter= ','):
    """
        Read a CSV file
    """
	my_data = genfromtxt(path, delimiter= delimiter)
	return my_data

def closest_point(points, ref):
    dist = np.sum((points - ref)**2, axis= 1)
    return np.argsort(dist)

def get_sigmoid(x):
    """
        Numerically stable version of sigmoid function. Taken from Assignment 2.
    """  
    output = np.zeros(x.shape)
    ind1 = (x >= 0)
    ind2 = (x  < 0)
    output[ind1] = 1 / (1 + np.exp(-x[ind1]))
    output[ind2] = np.divide(np.exp(x[ind2]), (1 + np.exp(x[ind2])))

    return output

def get_lambda_without_minus(xi):
    """
        Gets the lambda for the variational approximations
    """
    output = np.multiply(1/(2*(xi + TOLERANCE)), (get_sigmoid(xi) - 0.5))
    return output

def get_accuracy(pred, test_label, regression= "logistic"):
    """
        Gets accuracy in % for predictions. Taken from Assignment 2.
    """
	if regression == "multiclass":
		pred_max = np.argmax(pred, axis=1)
		gt_max   = np.argmax(test_label, axis=1)
		acc = np.sum(pred_max == gt_max)*100.0/pred.shape[0]
	elif regression == "logistic" or regression == "probit":
		if pred.ndim == 2:
			pred = pred[:,0]
		pred[pred >= 0.5] = 1.0
		pred[pred <  0.5] = 0.0
		acc = np.sum(pred == test_label)*100.0/pred.shape[0]

	return acc

def get_prediction_likelihood(test_data, test_label, w_map, hessian):
    """
        Returns prediction likelihood
    """
    # Get Prediction Likelihood
    pred_like = np.zeros((test_data.shape[0],))

    for i in range(test_data.shape[0]):
        test_pt = test_data[i]
        m = np.sqrt(2 * np.matmul(np.matmul(test_pt[np.newaxis,:], hessian), test_pt[:, np.newaxis]))
        c = np.sum (np.multiply(w_map, test_data[i]))
        likelihood_1 = gass_hermite_quad(affine_sigmoid, degree= 100, m= m, c= c)/np.sqrt(np.pi)
        pred_like[i] = likelihood_1**test_label[i] * (1-likelihood_1)**(1-test_label[i])

    return np.mean(pred_like)

def predict_and_test(weight, test_data, test_label, regression= "logistic"):
	# Predict on test data and get accuracy
	pred_test, _ = get_output(weight, test_data, regression= regression)
	acc          = get_accuracy(pred_test, test_label, regression= regression)
	return acc, pred_test

def get_output(weight, data, regression= "logistic"):
    """
        Output of different regression. Taken from Assignment 2.
    """
	dot_product = np.matmul(data,weight)
	if regression == "logistic":
		output = get_sigmoid(dot_product)
	elif regression == "probit":
		output = norm.cdf(dot_product)
	elif regression == "multiclass":
		output = softmax(dot_product, axis=1)

	return output, dot_product

def get_log_likelihood(phi, pred, t, dot_product, weight, reg= 1):
    """
        Returns log likelihood of the logistic regression
    """
    prior = -0.5* np.sum(np.power(weight, 2))
    likelihood = np.multiply(t[0], np.log(pred+TOLERANCE)) + np.multiply(1.0- t[0], np.log(1.0-pred+TOLERANCE))
    likelihood = np.sum(likelihood, axis= 0)

    return prior + likelihood

def get_hessian(phi, pred, t, dot_product, reg= 1, regression= "logistic"):
    """
        Hessian of different regression. Taken from Assignment 2.
    """
	R = np.eye(pred.shape[0])
	if regression == "logistic":
		for i in range(pred.shape[0]):
			R[i,i] = pred[i,0] * (1- pred[i,0])
	elif regression == "probit":
		for i in range(pred.shape[0]):
			y_n  = pred[i,0]
			t_n  = t[i,0] 
			dotp = dot_product[i, 0]
			pdf  = norm.pdf(dotp)

			term1 = 1/ (y_n * (1- y_n) + TOLERANCE)
			term2 = (y_n - t_n)/(y_n**2 * (1- y_n) + TOLERANCE)
			term3 = (y_n - t_n)/((1- y_n)**2 * y_n + TOLERANCE)
			term4 = (y_n - t_n)* dotp/(y_n * (1- y_n) * pdf + TOLERANCE)

			R[i,i] = (term1 - term2 + term3 - term4)*(pdf**2)

	# Add regularization			
	hessian = np.matmul(np.matmul(phi.T, R), phi) + np.eye(phi.shape[1])/reg
	return hessian

def get_affine(x, m, c):
    x = m*x + c
    return x

def affine_sigmoid(xin, m=10, c=3):
    if type(xin) != np.ndarray:
        x = np.array([xin])
    else:
        x = xin

    x = get_affine(x, m, c)                                
    output = np.zeros(x.shape)
    ind1 = (x >= 0)
    ind2 = (x  < 0)
    output[ind1] = 1 / (1 + np.exp(-x[ind1]))
    output[ind2] = np.divide(np.exp(x[ind2]), (1 + np.exp(x[ind2])))

    if type(xin) != np.ndarray:
        return output[0]
    else:
        return output

def gass_hermite_quad(f, degree, m, c):
    '''
    Calculate the integral (1) numerically.
    :param f: target function, takes a array as input x = [x0, x1,...,xn], and return a array of function values f(x) = [f(x0),f(x1), ..., f(xn)]
    :param degree: integer, >=1, number of points
    :return:
    '''

    points, weights = np.polynomial.hermite.hermgauss( degree)

    #function values at given points
    f_x = f(points, m= m, c= c)

    #weighted sum of function values
    F = np.sum( f_x  * weights)

    return F
