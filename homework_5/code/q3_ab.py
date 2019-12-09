

import os
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import scipy.stats as stats

from util import *
np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

scale_KE = 1    
#===============================================================================
# Main starts here
#===============================================================================
train = readcsv("data/bank-note/train.csv")
test  = readcsv("data/bank-note/test.csv")
train_label = train[:,-1]
test_label  = test [:,-1]

train_data  = train[:,0:-1]
test_data   = test [:,0:-1]

# Normalize data so that each data column is in the range [0,1]
# Same normalization for train and test
#for i in range(train_data.shape[1]):
#    train_data[:,i] = (train_data[:,i] - np.min(train_data[:,i]))/ (np.max(train_data[:,i]) - np.min(train_data[:,i]))
#    test_data[:,i]  = (test_data[:,i]  - np.min(train_data[:,i]))/ (np.max(train_data[:,i]) - np.min(train_data[:,i]))

# Add ones for biases    
train_data  = np.hstack((train_data, np.ones((train.shape[0],1))))
test_data   = np.hstack((test_data , np.ones((test.shape[0] ,1))))

print("Dataset stats...")
print(train_data.shape)
print("Min value of each feature of training data")
print(np.min(train_data, axis=0))
print("Max value of each feature of training data")
print(np.max(train_data, axis=0))
print("")

dim = train_data.shape[1]

num_iterations       = 100000#//10
num_iterations_final = 10000 #//10
collect_final_sample_frequency = 10
display_frequency    = 5000

#===============================================================================
# Hybrid (Hamiltonian) Monte Carlo
#===============================================================================
print("\n=======================================================================")
print("\tHamiltonian Monte Carlo Sampling with Leapfrog")
print("=======================================================================")
epsilon_array = np.array([0.005, 0.01, 0.02, 0.05])
num_leapfrog_steps_array = np.array([10, 20, 50])

for i in range(epsilon_array.shape[0]):
    for j in range(num_leapfrog_steps_array.shape[0]):
        epsilon            = epsilon_array[i]
        num_leapfrog_steps = num_leapfrog_steps_array[j]
        print("\nBurnin stage, epsilon = {:.3f}, L= {}".format(epsilon, num_leapfrog_steps))
        w_init = np.zeros((dim, 1))
        _, _, _, w_new                          = hybrid_monte_carlo(train_data, train_label, z_init= w_init, num_iterations= num_iterations      , epsilon= epsilon, num_leapfrog_steps= num_leapfrog_steps, collect_final_sample_frequency= collect_final_sample_frequency, scale_KE= scale_KE)

        # Remember to initialize from new values
        print("Generating samples after burnin stage...")
        accepted, sampled, posterior_samples, _ = hybrid_monte_carlo(train_data, train_label, z_init= w_new , num_iterations= num_iterations_final, epsilon= epsilon, num_leapfrog_steps= num_leapfrog_steps, collect_final_sample_frequency= collect_final_sample_frequency, scale_KE= scale_KE)
        acceptance_rate = accepted.shape[0]/sampled.shape[0]
        test_on_posterior(test_data, test_label, posterior_samples)
        print("Acceptance rate= {:2f}".format(acceptance_rate))

#===============================================================================
# Gibbs Sampling with Variable Augmentation for Latent Probit Regression
#===============================================================================
print("\n=======================================================================")
print("Gibbs Sampling with Variable Augmentation for Latent Probit Regression")
print("=======================================================================")
w_init = np.zeros((dim, 1))

print("Burnin stage...")
_, w_new             = gibbs(train_data, train_label, w_init, num_iterations      , display_frequency, collect_final_sample_frequency)

# Remember to initialize from new values
print("Generating samples after burnin stage...")
posterior_samples, _ = gibbs(train_data, train_label, w_new , num_iterations_final, display_frequency, collect_final_sample_frequency)

test_on_posterior(test_data, test_label, posterior_samples)
