

import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import copy

from util import *

dpi = 200
fs = 20
lw = 1.75
matplotlib.rcParams.update({'font.size': fs})
hamiltonian_scale = 1e-10

def get_conditional_mean(mean1, Sigma12, Sigma22, mean2, z2):
    return mean1 + (Sigma12*(z2- mean2)/Sigma22)

def get_conditional_variance(Sigma11, Sigma22, Sigma12):
    return Sigma11 - (Sigma12*Sigma12/Sigma22)

#===============================================================================
# Main starts here
#===============================================================================
mean = np.array([0, 0])
covar= np.array([[3, 2.9], [2.9, 3]])
num_points = 500
num_iterations = 100
num_leapfrog_steps = 20
epsilon = 0.1

# Sample some data
x1, x2 = np.random.multivariate_normal(mean, covar, num_points).T

# get sample mean and covar
data = np.hstack((x1[:, np.newaxis], x2[:, np.newaxis]))

print("Getting sample mean and covariance")
sample_mean  = np.mean(data, axis= 0)
sample_covar = np.mean(np.multiply(data[:, :, np.newaxis], data[:, np.newaxis, :]), axis= 0)

# Initialization point
z_init = np.array([-4.0, -4.0])


#===============================================================================
# Gibbs Sampling
#===============================================================================
print("\n\nGibbs Sampling...")
samples = np.zeros((num_iterations+1, 2))
samples[0,:] = z_init
z1 = z_init[0]
z2 = z_init[1]

for i in range(num_iterations):
    # Now start sampling one variable keeping other fixed.
    m1 = get_conditional_mean    (sample_mean[0]   , sample_covar[0,1], sample_covar[1,1], sample_mean[1], z2)
    s1 = get_conditional_variance(sample_covar[0,0], sample_covar[1,1], sample_covar[0,1])
    z1 = np.random.normal(m1, s1, 1)[0]

    m2 = get_conditional_mean    (sample_mean[1]   , sample_covar[0,1], sample_covar[0,0], sample_mean[0], z1)
    s2 = get_conditional_variance(sample_covar[1,1], sample_covar[0,0], sample_covar[0,1])
    z2 = np.random.normal(m2, s2, 1)[0]

    # Append the samples
    samples[i+1,:] = np.array([z1, z2])

x = np.arange(num_iterations+1) 
fig = plt.figure(figsize= (8,8), dpi= dpi)
plt.scatter(x1, x2, marker= 'o', s= 8, color= 'dodgerblue')
plt.xlim((-6, 6))
plt.ylim((-6, 6))
plt.grid(True)
plt.xlabel('x')
plt.ylabel('y')
savefig(plt, "images/q2a.png", newline= False)

plt.plot(samples[:, 0], samples[:, 1], '--k', marker='o', markersize= 4, markerfacecolor= "r", markeredgecolor= "r")
savefig(plt, "images/q2b.png")
plt.close()


#===============================================================================
# Hamiltonian Monte Carlo Sampling with Leapfrog
#===============================================================================
print("\nHamiltonian Monte Carlo Sampling with Leapfrog")
samples = np.zeros((num_iterations+1, 2))
samples[0,:] = z_init
z = np.copy(z_init)
accepted = 0

for i in range(num_iterations):
    # draw random momentum vector from unit Gaussian which decides the energy
    # given out for exploration
    # random normal(mean, std deviation)
    p = np.random.normal(0.0, 1, 2)

    # Compute Hamiltonian
    H = get_hamiltonian(z, mean, covar, p, scale= hamiltonian_scale)
    old_grad = get_gradient_minus_log_joint(z, mean[np.newaxis, :], covar)

    new_z = np.copy(z)              # deep copy of array
    new_grad  = np.copy(old_grad)   # deep copy of array
    
    # Suggest new candidate using gradient + Hamiltonian dynamics.
    # Leapfrog
    for j in range(num_leapfrog_steps):  
        # Make first half step in p, full step in z and then again half step in p
        p        -= (epsilon/2.0)*new_grad
        new_z    += epsilon*p
        new_grad  = get_gradient_minus_log_joint(new_z, mean[np.newaxis, :], covar)
        p        -= (epsilon/2.0)*new_grad

    # Negate momentum
    p = -p
    # Compute Hamiltonian for the new point
    new_H  = get_hamiltonian(new_z, mean, covar, p, scale= hamiltonian_scale)

    if to_accept_without_log(np.exp(-H), np.exp(-new_H)):      
        z = new_z
        accepted += 1
    
    #print(H, new_H, z)
    samples[i+1,:] = z

print("Acceptance rate = "+str(accepted/float(num_iterations)))
x = np.arange(num_iterations+1) 
fig = plt.figure(figsize= (8,8), dpi= dpi)
plt.scatter(x1, x2, marker= 'o', s= 8, color= 'dodgerblue')
plt.xlim((-6, 6))
plt.ylim((-6, 6))
plt.grid(True)
plt.xlabel('x')
plt.ylabel('y')
plt.plot(samples[:, 0], samples[:, 1], '--k', marker='o', markersize= 4, markerfacecolor= "r", markeredgecolor= "r")

# Get the size of samples to display the frequency where a particle stays
#_, rev_ind, cnt = np.unique(samples, axis=0, return_counts= True, return_inverse= True)
#print(rev_ind)
#print(cnt)
#sizes_cnt = 4*cnt
#custom_sizes = sizes_cnt[rev_ind].tolist()
#plt.plot   (samples[:, 0], samples[:, 1], '--k')#  s= custom_sizes, )
#plt.scatter(samples[:, 0], samples[:, 1], s= 4, marker='o', color= "r", edgecolor= "r")
#plt.show()
savefig(plt, "images/q2c.png")
plt.close()
