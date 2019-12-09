

import os
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal

from util import *

dpi = 200
fs = 20
lw = 1.75
matplotlib.rcParams.update({'font.size': fs})


def E_step(pts, num_clusters, means, covar, mix_alpha):
    weight_matrix = np.zeros((pts.shape[0], num_clusters))
    for i in range(num_clusters):
        weight_matrix[:,i] = mix_alpha[i] * multivariate_normal.pdf(pts, mean= means[i], cov= covar[i])

    # Normalize them 
    weight_matrix = np.divide(weight_matrix, np.sum(weight_matrix, axis=1)[:,np.newaxis])

    return weight_matrix

def M_step(pts, num_clusters, weight_matrix):
    N  = pts.shape[0]
    dim= pts.shape[1]
    Nk = np.sum(weight_matrix, axis=0)

    mix_alpha = Nk/N
    means_new = np.zeros((num_clusters, pts.shape[1]))
    covar_new = np.zeros((num_clusters, pts.shape[1], pts.shape[1]))
 
    for i in range(num_clusters):
        weight_reshape_for_means = np.repeat(weight_matrix[:,i], dim)    .reshape(N, dim)      # N x 2
        weight_reshape_for_covar = np.repeat(weight_matrix[:,i], dim*dim).reshape(N, dim, dim) # N x 2 x 2 
        means_new[i] = np.sum(np.multiply(weight_reshape_for_means, pts), axis=0)/Nk[i]
        temp         = pts - means_new[i]  # N x 2
        outer_prod   = np.matmul(temp[:,:,np.newaxis], temp[:,np.newaxis,:]) # N x 2 x 2
        covar_new[i] = np.sum(np.multiply(weight_reshape_for_covar, outer_prod), axis= 0)/Nk[i] # 2 x 2

    return mix_alpha, means_new, covar_new

def visualize(pts, means, covar, weight_matrix, iter, label):

    saving_iter = np.array([1,2,5,100])-1

    if iter in saving_iter:
        fig = plt.figure(figsize=(8,8), dpi= dpi)
        
        # Each data point is assigned to the cluster that has a great posterior 
        # probability to include that data point
        index0 = weight_matrix[:, 0] >= weight_matrix[:, 1] 
        index1 = weight_matrix[:, 0]  < weight_matrix[:, 1]

        # Next plot the points in different colors
        plt.scatter(pts[index0, 0], pts[index0, 1],              c= 'r', s= 20, label= 'Cluster1')
        plt.scatter(pts[index1, 0], pts[index1, 1],              c= 'b', s= 20, label= 'Cluster2')

        # Next plot the centers
        plt.scatter(means[:, 0], means[:, 1]      , marker= '*', c= 'k', s= 150)

        plt.xlim((-1,1))
        plt.grid('True')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend(loc= 'upper left')
        path = "images/" + label + "_iter_" + str(iter+1) + ".png"
        savefig(plt, path, newline= False)

#===============================================================================
# Main starts here
#===============================================================================
pts = readcsv("data/faithful/faithful.txt", delimiter= ' ')
print("Shape of training data")
print(pts.shape)
print("")

# Normalize data so that each data column is in the range [0,1]
for i in range(pts.shape[1]):
    pts[:,i] = (pts[:,i] - np.mean(pts[:,i]))/ (np.max(pts[:,i]) - np.min(pts[:,i]))

num_clusters = 2
num_iter     = 100

#===============================================================================
# Q3(a)
#===============================================================================
means = np.array([[-1,1],[1,-1]])
covar = np.zeros((num_clusters, 2, 2))
covar[0] = 0.1*np.eye(2)
covar[1] = 0.1*np.eye(2)
# Equal mix_alpha
mix_alpha = (1/num_clusters)*np.ones((num_clusters, ))

for i in range(num_iter):
    weight_matrix           = E_step(pts, num_clusters, means, covar, mix_alpha)
    mix_alpha, means, covar = M_step(pts, num_clusters, weight_matrix)
    visualize(pts, means, covar, weight_matrix, iter= i, label= 'q3a')
print("")

#===============================================================================
# Q3(b)
#===============================================================================
means = np.array([[-1,-1],[1,1]])
covar = np.zeros((num_clusters, 2, 2))
covar[0] = 0.5*np.eye(2)
covar[1] = 0.5*np.eye(2)
# Equal mix_alpha
mix_alpha = (1/num_clusters)*np.ones((num_clusters, ))

for i in range(num_iter):
    weight_matrix           = E_step(pts, num_clusters, means, covar, mix_alpha)
    mix_alpha, means, covar = M_step(pts, num_clusters, weight_matrix)
    visualize(pts, means, covar, weight_matrix, iter= i, label= 'q3b')
