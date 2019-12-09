

import os
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

from util import *
np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

num_epochs = 100
#===============================================================================
# Main starts here
#===============================================================================
train = readcsv("data/bank-note/train.csv")
test  = readcsv("data/bank-note/test.csv")
train_label = train[:,-1]
test_label  = test [:,-1]
train_data  = np.hstack((train[:,0:-1], np.ones((train.shape[0],1))))
test_data   = np.hstack((test [:,0:-1], np.ones((test.shape[0], 1))))

print("Dataset stats...")
print(train_data.shape)
print(np.max(train_data, axis=0))
print(np.min(train_data, axis=0))
print("")
dim = train_data.shape[1]

delta = 0.05
wmax  = 5
wi    = np.arange(-wmax, wmax+ delta, delta)

wi1   = np.arange(-2.8, -2.6+delta, delta)
wi2   = np.arange(-1.7, -1.5+delta, delta)
wi3   = np.arange(-2.0, -1.8+delta, delta)
wi4   = np.arange(-0.2, -0.1+delta, delta)
wi5   = np.arange( 2.7,  2.9+delta, delta)
w_mesh= np.array(np.meshgrid(wi1, wi2, wi3, wi4, wi5)).T.reshape(-1,5)
print("")

#===============================================================================
# Full Laplace Approximation
#===============================================================================
max_log_likelihood = -100000
w_map = np.zeros((5, ))

# First get the mode
for i in range(w_mesh.shape[0]):
    pred, dot_product = get_output(w_mesh[i], train_data)
    likeli = get_log_likelihood(phi= train_data, pred= pred, t= train_label[:, np.newaxis], dot_product= dot_product, weight= w_mesh[i])
    if likeli > max_log_likelihood:
        max_log_likelihood = likeli
        w_map = w_mesh[i]
#w_map = np.array([-2.7,-1.59,-1.90,-0.17,2.8])

# Get training data prediction
pred, dot_product = get_output(w_map, train_data)

hessian  = get_hessian (phi= train_data, pred= pred[:, np.newaxis], t= train_label[:, np.newaxis], dot_product= dot_product)

print("\n=====================================================================")
print("\t\tFull Laplace Approximation ...")
print("=====================================================================")
print("MAP = ")
print(w_map)
print("Hessian = ")
print(hessian)

# Calculate on test data
pred_test, _ = get_output  (w_map, test_data)
acc          = get_accuracy(pred_test, test_label)
pred_likelihood = get_prediction_likelihood(test_data, test_label, w_map, hessian)
print("Test_data Pred_acc= {:.2f}, Pred_likelihood= {:.2f}".format(acc, pred_likelihood))


#===============================================================================
# Diagonal Laplace Approximation
#===============================================================================

print("\n=====================================================================")
print("\t\tDiagonal Laplace Approximation ...")
print("=====================================================================")
hessian = np.multiply(hessian, np.eye(hessian.shape[0]))
print("MAP = ")
print(w_map)
print("Hessian = ")
print(hessian)

# Calculate on test data
pred_test, _ = get_output  (w_map, test_data)
acc          = get_accuracy(pred_test, test_label)
pred_likelihood = get_prediction_likelihood(test_data, test_label, w_map, hessian)
print("Test_data Pred_acc= {:.2f}, Pred_likelihood= {:.2f}".format(acc, pred_likelihood))


#===============================================================================
# Variational Logistic Regression
#===============================================================================

print("\n=====================================================================")
print("\t\tLocal Variational Approximation ...")
print("=====================================================================")

pred_likelihood = -100
xi = -np.ones((train_data.shape[0],))
S0inv = np.linalg.inv(np.eye(dim))

for i in range(num_epochs):
    lambdaa = get_lambda_without_minus(xi)

    # Do the Expectation step
    phi_phi_transpose = np.matmul(train_data[:,:,np.newaxis], train_data[:,np.newaxis,:]) # N x 5 x 5
    phi_phi_transpose = np.multiply(lambdaa[:, np.newaxis, np.newaxis], phi_phi_transpose)
    SN = np.linalg.inv( S0inv + 2*np.sum(phi_phi_transpose, axis= 0))
    mN = np.matmul(SN, np.sum(np.multiply( (train_label[:, np.newaxis]-0.5), train_data), axis= 0))

    # Do the maximisation step
    temp = SN + np.matmul(mN[:, np.newaxis], mN[np.newaxis, :])
    xi = np.matmul( np.matmul( train_data[:, np.newaxis, :],  temp[np.newaxis, :, :]), train_data[:, :, np.newaxis]).flatten()
    xi = np.sqrt(xi)

    # Calculate on test data
    pred_test, _ = get_output  (mN, test_data)
    acc          = get_accuracy(pred_test, test_label)
    pred_likelihood_new = get_prediction_likelihood(test_data, test_label, mN, SN) 
    print("Epoch {:02d} Test_data Pred_acc= {:.2f}, Pred_likelihood= {:.2f}".format(i, acc, pred_likelihood_new))

    if np.abs(pred_likelihood_new - pred_likelihood) < 1e-3:
        break
    else:
        pred_likelihood = pred_likelihood_new

print("\nMean = ")
print(mN)
print("Hessian = ")
print(SN)


#===============================================================================
# Variational Logistic Regression with mean field update
#===============================================================================

print("\n=====================================================================")
print("\tLocal Variational Approximation with Factorised Posterior...")
print("=====================================================================")
pred_likelihood = -100
xi = -np.ones ((train_data.shape[0],))
mN =   np.zeros((dim, ))
mN_new = np.zeros((dim, ))
SN = np.zeros((dim, dim))

update_means_inplace = True
if update_means_inplace:
    print("Updating the means inplace")
else:
    print("Not updating the means inplace")

for i in range(num_epochs):
    lambdaa = get_lambda_without_minus(xi)

    phi_phi_transpose = np.matmul(train_data[:,:,np.newaxis], train_data[:,np.newaxis,:]) # N x 5 x 5
    phi_phi_transpose = np.multiply(lambdaa[:, np.newaxis, np.newaxis], phi_phi_transpose) # N x 5 x 5
    mean_term_1_all   = np.sum(np.multiply( (train_label[:, np.newaxis]-0.5), train_data), axis= 0) # 5
    
    # Do the Expectation step in multiple steps
    # Update one variable at a time.
    for j in range(dim):
        mean_term_1 = mean_term_1_all[j]
        all_except_one_index = np.delete(np.arange(dim), j)
        b = mN[np.newaxis, :]
        inner_sum         = np.sum( np.multiply(b[:,all_except_one_index], train_data[:,all_except_one_index]), axis=1) # N
        mean_term_2       = -2.0 * np.sum( np.multiply( np.multiply(lambdaa, train_data[:,j]), inner_sum), axis=0) # 1

        SN[j,j] = 1.0/( 1 + 2*np.sum(phi_phi_transpose[:,j,j], axis= 0))
        if update_means_inplace:
            mN[j]   = SN[j,j] * (mean_term_1 + mean_term_2)
        else:
            mN_new[j]   = SN[j,j] * (mean_term_1 + mean_term_2)

    if not update_means_inplace:
        mN = mN_new

    # Do the maximisation step
    temp = SN + np.matmul(mN[:, np.newaxis], mN[np.newaxis, :])
    xi = np.matmul( np.matmul( train_data[:, np.newaxis, :],  temp[np.newaxis, :, :]), train_data[:, :, np.newaxis]).flatten()
    xi = np.sqrt(xi)

    # Calculate on test data
    pred_test, _ = get_output  (mN, test_data)
    acc          = get_accuracy(pred_test, test_label)
    pred_likelihood_new = get_prediction_likelihood(test_data, test_label, mN, SN) 
    print("Epoch {:02d} Test_data Pred_acc= {:.2f}, Pred_likelihood= {:.2f}".format(i, acc, pred_likelihood_new))

    if np.abs(pred_likelihood_new - pred_likelihood) < 1e-3:
        break
    else:
        pred_likelihood = pred_likelihood_new

print("\nMean = ")
print(mN)
print("Hessian = ")
print(SN)
