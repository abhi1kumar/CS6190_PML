

import numpy as np
import os, sys
from matplotlib import pyplot as plt
import matplotlib
import copy
from util import *

dpi = 200
fs = 20
lw = 1.75
my_orange = [(1, 0.5, 0.06)]
matplotlib.rcParams.update({'font.size': fs})

# Transition model defines how to move
def transition(mu, variance):
    return np.random.normal(mu, variance, 1)

# Prior: Variance can not be negative
def prior(x):
    if(x[1] <=0):
        return 0
    return 1

def log_joint(z):
    # Get log joint
    return -z*z + np.log(affine_sigmoid(z, m=10, c=3) + 1e-6)

def get_gradient_log_joint(z):
    return -2*z + (1-affine_sigmoid(z, m=10, c=3))*10

# Metropolis Hastings Algorithm
def metropolis_hastings(likelihood_function, transition_function, acceptance_rule, prior, x_init, num_iterations, collect_final_sample_frequency= 50, display_frequency= 5000):
    # likelihood_function(x, data): returns the likelihood that these parameters generated the data
    # transition_function(x): a function that draws a sample from a symmetric distribution and returns it
    # acceptance_rule(x, x_new): decides whether to accept or reject the new sample
    # x_init: a starting sample
    # num_iterations: number of iterations
    # data: the data that we wish to model
    x = x_init

    accepted = [] # Keeps track of accepted samples
    sampled  = [] # Keeps track of all samples
    final    = [] # Keeps track of final samples which are sampled in a cyclic manner

    for i in range(num_iterations):
        x_new    = np.zeros(x.shape)
        x_new[0] = transition_function(x[0], x[1])  
        x_new[1] = tau

        # Get the log likelihoods
        x_likelihood     = likelihood_function(x[0]    )
        x_new_likelihood = likelihood_function(x_new[0])

        sampled.append(x_new[0])
        if (acceptance_rule(x_likelihood, x_new_likelihood )):            
            x = x_new
            accepted.append(x_new[0])
        
        if i % collect_final_sample_frequency == 0:
            # Sample from the current parameters
            final.append(x[0])
   
        if (i+1) % display_frequency == 0 or i == num_iterations-1:
            print("Iter {:5d} done".format(i+1))
    
    return np.array(accepted), np.array(sampled), np.array(final), x

def hybrid_monte_carlo(likelihood_function, gradient_function, acceptance_rule, z_init, num_iterations, epsilon, num_leapfrog_steps, collect_final_sample_frequency= 50, display_frequency= 5000):
    z = z_init

    accepted = [] # Keeps track of accepted samples
    sampled  = [] # Keeps track of all samples
    final    = [] # Keeps track of final samples which are sampled in a cyclic manner
    
    for i in range(num_iterations):
        # Old energy = -loglik and Old gradient
        old_PE   =  get_PE(likelihood_function  (z))
        old_grad = -gradient_function(z)

        new_z = copy.copy(z)              # deep copy of array
        new_grad  = copy.copy(old_grad)   # deep copy of array

        # draw random momentum vector from unit Gaussian which decides the energy
        # given out for exploration
        p = np.random.normal(0.0, 1.0, 1)

        # Compute Hamiltonian
        H = get_KE(p) + old_PE

        # Suggest new candidate using gradient + Hamiltonian dynamics.
        # Leapfrog
        for j in range(num_leapfrog_steps):  
            # Make first half step in p
            p        -= epsilon*new_grad/2.0
            new_z    += epsilon*p
            new_grad = -gradient_function(new_z)
            p        -= epsilon*new_grad/2.0

        # Negate momentum
        p = -p
        # Compute new Hamiltonian
        new_PE = get_PE(likelihood_function(new_z))
        new_H  = get_KE(p) + new_PE
        dH     = new_H - H

        # Accept new candidate in Monte-Carlo fashion.
        if (acceptance_rule(dH)):            
            z = new_z
            accepted.append(new_z)

        if i % collect_final_sample_frequency == 0:
            # Sample from the current parameters
            final.append(z)

        if (i+1) % display_frequency == 0 or i == num_iterations-1:
            print("Iter {:5d} done".format(i+1))
    
    return np.array(accepted), np.array(sampled), np.array(final), z   

#===============================================================================
# Main starts here
#===============================================================================
delta= 0.01
num_bins = 50
xmax = 5
num_iterations       = 100000
num_iterations_final = 50000
num_leapfrog_steps   = 10
collect_final_sample_frequency = 50
display_frequency    = 5000

z = np.arange(-xmax, xmax+delta, delta)
hist_bins = np.arange(-xmax, xmax, delta)

N = gass_hermite_quad(affine_sigmoid, degree= 100, m= 10, c= 3)
print("Normalization = {:.2f}".format(N))
pz = np.multiply(np.exp(-np.multiply(z, z)), affine_sigmoid(z))/N

#===============================================================================
# (a) Metropolis Hastings Algorithm
#===============================================================================
print("\n=======================================================================")
print("\tMetropolis Hastings Algorithm")
print("=======================================================================")
tau_array = np.array([0.01, 0.1, 0.2, 0.5, 1])
num_accepted_array = np.zeros(tau_array.shape)

for i in range(tau_array.shape[0]):
    tau = tau_array[i]
    print("\nBurnin stage, tau = {:.2f}".format(tau))
    x_init = np.array([0, tau])
    _, _, _, x_new = metropolis_hastings(likelihood_function= log_joint, transition_function= transition, acceptance_rule= to_accept, prior= prior, x_init= x_init, num_iterations= num_iterations)

    # Remember to initialize from new values
    print("Generating samples after burnin stage...")
    accepted, _, final, _        = metropolis_hastings(likelihood_function= log_joint, transition_function= transition, acceptance_rule= to_accept, prior= prior, x_init= x_new , num_iterations= num_iterations_final, collect_final_sample_frequency= collect_final_sample_frequency)
    num_accepted_array[i] = accepted.shape[0]
    
    fig = plt.figure(figsize= (9.6,6), dpi= dpi)
    plt.plot(z, pz, label='True', lw= lw, color= 'dodgerblue')
    plt.hist(final, bins= num_bins , density= 'True', alpha= 0.5, label= 'Sampled', color= my_orange)
    plt.grid(True)
    plt.xlabel('z')
    plt.ylabel('p(z)')
    plt.ylim((-0.05,1.0))
    plt.xlim((-5,5))
    plt.legend(loc= 'upper right')
    tau_rep = str(tau).replace(".", "_")
    path = "images/q1a_tau_" + tau_rep + ".png"
    savefig(plt, path= path)
    plt.close()

accepted_rate = num_accepted_array/float(num_iterations_final)

fig = plt.figure(figsize= (9.6,6), dpi= dpi)
plt.plot(tau_array, accepted_rate, lw= lw, marker='o', color= 'dodgerblue')
plt.xlabel(r'$\tau$')
plt.ylabel('Acceptance rate')
plt.xlim((0, 1.0))
plt.ylim((0, 1.0))
plt.grid(True)
path = "images/q1a_acceptance_rate_vs_tau.png"
savefig(plt, path= path)
plt.close()

#===============================================================================
# Hybrid (Hamiltonian) Monte Carlo
#===============================================================================
print("\n=======================================================================")
print("\tHamiltonian Monte Carlo Sampling with Leapfrog")
print("=======================================================================")
epsilon_array = np.array([0.005, 0.01, 0.1, 0.2, 0.5])
num_accepted_array = np.zeros(epsilon_array.shape)

for i in range(epsilon_array.shape[0]):
    epsilon = epsilon_array[i]
    print("\nBurnin stage, epsilon = {:.3f}".format(epsilon))
    z_init = np.array([0.0])
    _, _, _, z_new = hybrid_monte_carlo(likelihood_function= log_joint, gradient_function= get_gradient_log_joint, acceptance_rule= to_accept_hamiltonian, z_init= z_init, num_iterations= num_iterations, epsilon= epsilon, num_leapfrog_steps= num_leapfrog_steps)

    # Remember to initialize from new values
    print("Generating samples after burnin stage...")
    accepted, _, final, _        = hybrid_monte_carlo(likelihood_function= log_joint, gradient_function= get_gradient_log_joint, acceptance_rule= to_accept_hamiltonian, z_init= z_new , num_iterations= num_iterations_final, epsilon= epsilon, num_leapfrog_steps= num_leapfrog_steps, collect_final_sample_frequency= collect_final_sample_frequency)
    num_accepted_array[i] = accepted.shape[0]
    
    fig = plt.figure(figsize= (9.6,6), dpi= dpi)
    plt.plot(z, pz, label='True', lw= lw, color= 'dodgerblue')
    plt.hist(final, bins= num_bins , density= 'True', alpha= 0.5, label= 'Sampled', color= my_orange)
    plt.grid(True)
    plt.xlabel('z')
    plt.ylabel('p(z)')
    plt.ylim((-0.05,1.0))
    plt.xlim((-5,5))
    plt.legend(loc= 'upper right')
    epsilon_rep = str(epsilon).replace(".", "_")
    path = "images/q1b_epsilon_" + epsilon_rep + ".png"
    savefig(plt, path= path)
    plt.close()

accepted_rate = num_accepted_array/float(num_iterations_final)

fig = plt.figure(figsize= (9.6,6), dpi= dpi)
plt.plot(epsilon_array, accepted_rate, lw= lw, marker='o', color= 'dodgerblue')
plt.xlabel(r'$\epsilon$')
plt.ylabel('Acceptance rate')
plt.xlim((0, 1.0))
plt.ylim((0, 1.0))
plt.grid(True)
path = "images/q1b_acceptance_rate_vs_epsilon.png"
savefig(plt, path= path)
plt.close()
