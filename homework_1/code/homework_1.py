import numpy as np
from scipy.stats import t as student_t
from scipy.stats import norm as gaussian
from scipy.stats import beta
from matplotlib import pyplot as plt
import matplotlib

lw = 1.5
DPI = 200
mu = 0
scale = 1
num_pts = 200
binwidth = 0.25
fs = 14
matplotlib.rcParams.update({'font.size': fs})

#==============================================================================
# Question 1
#==============================================================================
print("\n================ Question 1 ===================")
xmax = 10
xmin = -xmax
x = np.linspace(xmin, xmax, num_pts)

# Define the distribution parameters to be plotted
k_values = [0.1, 1, 10, 100, 10E6]
color_values = ['r', 'magenta', 'k', 'g', 'c']

# Plot the student's t distributions
plt.figure(figsize=(8, 6), dpi= DPI)

for k, col in zip(k_values, color_values):
    # Reference
    # https://www.astroml.org/book_figures/chapter3/fig_student_t_distribution.html
    dist = student_t(df= k, loc= mu, scale= scale)
    label = r't ($\mathrm{\nu}=%.1f$)' % k
    plt.plot(x, dist.pdf(x), lw= lw, c= col, label= label)

# Finally plot Gaussian
dist = gaussian(loc= mu, scale= scale)
plt.plot(x, dist.pdf(x), lw= lw, c= 'blue', label= 'Gaussian')

plt.xlim(-xmax, xmax)
plt.ylim(0.0, 0.4)
plt.xlabel('$x$')
plt.ylabel(r'$p(x)$')
plt.title(r'Students $t$ and Gaussian Distribution')

plt.legend()
plt.grid(True)
path = 'coding_1.png'
print("Saving to " + path)
plt.savefig(path)

#==============================================================================
# Question 2
#==============================================================================
print("\n================ Question 2 ===================")
xmax = 1
xmin = 0
x = np.linspace(xmin, xmax, num_pts)
color_values = ['r', 'b', 'k']

a_values = [1, 5, 10]
plt.figure(figsize=(8, 6), dpi= DPI)

for a, col in zip(a_values, color_values):
    dist = beta(a= a, b= a, loc= mu, scale= scale)
    label = r'Beta [$\mathrm{a}=%.0f, \mathrm{b}=%.0f$]' % (a, a)
    plt.plot(x, dist.pdf(x), lw= lw, c= col, label= label)

plt.xlim(xmin, xmax)
plt.ylim(0, 3.7)
plt.xlabel('$x$')
plt.ylabel(r'$p(x)$')
plt.title(r'Beta Distribution')

plt.legend(loc="upper right")
plt.grid(True)
path = 'coding_2a.png'
print("Saving to " + path)
plt.savefig(path)
plt.close()

a_values = [1, 5, 10]
b_values = [2, 6, 11]
plt.figure(figsize=(8, 6), dpi= DPI)

for a, b, col in zip(a_values, b_values, color_values):
    dist = beta(a= a, b= b, loc= mu, scale= scale)
    label = r'Beta [$\mathrm{a}=%.0f, \mathrm{b}=%.0f$]' % (a, b)
    plt.plot(x, dist.pdf(x), lw= lw, c= col, label= label)

plt.xlim(xmin, xmax)
plt.ylim(0, 3.7)
plt.xlabel('$x$')
plt.ylabel(r'$p(x)$')
plt.title(r'Beta Distribution')

plt.legend(loc="upper right")
plt.grid(True)
path = 'coding_2b.png'
print("Saving to " + path)
plt.savefig(path)
plt.close()

#==============================================================================
# Question 3
#==============================================================================
print("\n\n================ Question 3 ===================")

from scipy.optimize import minimize
import scipy.misc

mu = 0
variance = 2.0
sigma = np.sqrt(variance)

num_pts = 30
# Draw 30 samples from random distribution. Fix the seed to make the 
# experiment repeatable
np.random.seed(123)
s = np.random.normal(loc= mu, scale= sigma, size= num_pts)

def Gaussian(input, mu, sigma):
    return 1/(sigma * np.sqrt(2 * np.pi)) *np.exp( - (input - mu)**2 / (2 * sigma**2))

def MLE_Gaussian(x):
    # We minimize sum of negative of the log likelihood
    return np.sum(- np.log(Gaussian(s, x[0], x[1])))

def Student_t(input, k, mu, sigma):
    dist = student_t(df= k, loc= mu, scale= sigma)
    return dist.pdf(input)

def MLE_Student_t(x):
    # We minimize sum of negative of the log likelihood
    return np.sum(- np.log(Student_t(input= s, k= x[0], mu= x[1], sigma= x[2])))

def estimate_and_plot(s, path='coding_3a.png'):

    # Reference 
    # https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html#broyden-fletcher-goldfarb-shanno-algorithm-method-bfgs
    # If we do not supply our gradient, it is taken by finite difference methods
    # order [mu, sigma]
    x0 = np.array([0, 1.0])
    param_Gaussian = minimize(MLE_Gaussian, x0, method='BFGS', options={'disp': True})
    print(param_Gaussian.x)

    # order [k, mu, sigma]
    x0 = np.array([10, 0.0, 1.0])
    param_Student_t = minimize(MLE_Student_t, x0, method='BFGS', options={'disp': True})
    print(param_Student_t.x)


    x = np.linspace(-5, 5, num_pts)
    plt.figure(figsize=(8, 6), dpi= DPI)
    # Plot points
    plt.hist(s, bins=np.arange(-10, 10 + binwidth, binwidth), density=True)
    # Plot Gaussian
    dist = gaussian(loc= param_Gaussian.x[0], scale= param_Gaussian.x[1])
    plt.plot(x, dist.pdf(x), lw= lw, c= 'blue', label= 'Gaussian')
    # Plot Student t
    dist = student_t(df= param_Student_t.x[0], loc= param_Student_t.x[1], scale= param_Student_t.x[2])
    plt.plot(x, dist.pdf(x), lw= lw, c= 'red', label= 'Student_t')

    plt.ylim(0, 0.55)
    plt.xlim(-10, 10)
    plt.xlabel('$x$')
    plt.ylabel(r'$p(x)$')
    plt.title(r'Estimated Gaussian and Students $t$ Distribution')
    plt.legend(loc="upper right")
    plt.grid(True)
    print("Saving to " + path)
    plt.savefig(path)
    plt.close()

estimate_and_plot(s, path='coding_3a.png')

# Add noise
print("\nAdding noise to the data and then estimating again")
s= np.hstack((s, np.array([8,9,10.])))
estimate_and_plot(s, path='coding_3b.png')