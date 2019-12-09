

import numpy as np
import os, sys
from matplotlib import pyplot as plt
import matplotlib

sys.path.append(os.getcwd())
sys.path.insert(0, 'data/example-code')
from gmq_example import gass_hermite_quad
#from data.example_code.gmq_example import gass_hermite_quad

m = 10
c = 3
dpi = 200
fs = 20
lw = 1.75
matplotlib.rcParams.update({'font.size': fs})


def savefig(plt, path, show_message= True, tight_flag= True, newline= True):
    if show_message:
        print("Saving to {}".format(path))
    if tight_flag:
        plt.savefig(path, bbox_inches='tight', pad_inches=0)
    else:
        plt.savefig(path)
    if newline:
        print("")

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

def get_lambda(xi):
    output = -1/(2*get_affine(xi, m, c)) * (affine_sigmoid(xi,m ,c) - 0.5)
    return output


N = gass_hermite_quad(affine_sigmoid, degree= 100)
print("Normalization = {:.2f}".format(N))

delta= 0.01
z = np.arange(-5, 5+delta, delta)
pz = np.multiply(np.exp(-np.multiply(z, z)), affine_sigmoid(z))/N

fig = plt.figure(figsize= (9.6,6), dpi= dpi)
plt.plot(z, pz, label='Ground', lw= lw)
plt.legend(loc= 'upper right')
plt.grid(True)
plt.xlabel('z')
plt.ylabel('p(z)')
plt.ylim((-0.05,1.0))
savefig(plt, "images/q1_ground.png")

#===============================================================================
# Laplace Approximation
#===============================================================================
print("Running Laplace Approximation...")
# Get the maximum of out
ind = pz.argmax()
# get the max value 
theta_0 = z[ind]
# get the double derivative inverse
double_derivative_inv = -(-2 - affine_sigmoid(theta_0) * (1- affine_sigmoid(theta_0)) * m * m)
laplace_z =  pz[ind] * np.exp(- 0.5 * np.power(z-theta_0, 2) * double_derivative_inv)
print("Mode/Mean = {:.2f}, Variance= {:.2f}".format(theta_0, 1.0/double_derivative_inv))

# Draw the curve now
plt.plot(z, laplace_z, label= 'Laplace', lw= lw)
plt.legend(loc= 'upper right')
savefig(plt, "images/q1_laplace.png")

#===============================================================================
# Get the local variational approximation
#===============================================================================
print("Running Local Variational Approximation...")
# Run EM algorithm
lva = np.zeros((z.shape))
# Initial value of xi
xi  = 0
for i in range(10000):
    # Expectation step
    first = np.exp(-np.multiply(z, z)) * affine_sigmoid(xi)
    second= np.exp(5*(z-xi) + get_lambda(xi) *np.multiply(10*(z-xi), 10*(z + xi) + 6))  
    lva = np.multiply(first, second)/N

    # Maximise step
    xi_new  = z[lva.argmax()]
    diff    = np.abs(xi_new - xi)
    print("Iter {}, xi= {:.4f}, Diff= {:.4f}".format(i, xi_new, diff))
    if diff < 1e-4:
        break
    else:
        xi = xi_new

# Draw the curve now
plt.plot(z, lva, label= 'LVA', lw= lw)
plt.legend(loc= 'upper right')
savefig(plt, "images/q1_laplace_lva.png")
