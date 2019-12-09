import os
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import scipy as sp
from scipy.stats import multivariate_normal

np.set_printoptions(precision=2)

def get_func(x, w0, w1):
	y = w0 + w1*x
	return y

def plot_weights(mean, cov, log_space= False, likelihood= False, points= 0):
	# create a grid of (x,y) coordinates
	xlim = (-1, 1)
	ylim = (-1, 1)
	xres = 100
	yres = 100

	x1 = np.linspace(xlim[0], xlim[1], xres)
	y1 = np.linspace(ylim[0], ylim[1], yres)
	xx, yy = np.meshgrid(x1,y1)

	# evaluate prior at grid points
	xxyy = np.c_[xx.ravel(), yy.ravel()]

	# This is the prior
	k1 = sp.stats.multivariate_normal(mean= mean, cov= cov)
	posterior = k1.pdf(xxyy)
	
	if log_space:
		posterior = np.log(posterior)

	print()
	covariance = np.eye(2)*alpha
	mean  = np.zeros((2,1))

	if likelihood:
		print("Using likelihood of {} points".format(points))
		# Use likelihood
		if log_space:
			likelihood = np.zeros(posterior.shape)
		else:
			likelihood = np.ones(posterior.shape, dtype=np.float64)

		# Posterior mean calculation
		x_temp = np.hstack((x[0:points][:,np.newaxis], np.ones(points)[:,np.newaxis]))
		y_temp = y[0:points][:, np.newaxis]
		covariance = np.linalg.inv( covariance + beta* np.matmul(x_temp.T, x_temp))
		mean = beta*np.matmul(covariance, np.matmul(x_temp.T, y_temp))

		for i in range(points):
			new_xx = y_noisy[i] - np.matmul(xxyy, np.c_[x[i], np.ones(1)].T)
			k2 = sp.stats.multivariate_normal(mean= 0, cov= 1.0/beta)
			
			if log_space:
				likelihood =  likelihood + np.log(k2.pdf(new_xx))
			else:
				likelihood = likelihood*k2.pdf(new_xx)
		
		if log_space:
			posterior = posterior + likelihood
		else:
			posterior = posterior * likelihood

	else:
		print("Not using likelihood")
		covariance = np.linalg.inv( covariance)

	# reshape and plot image
	img = posterior.reshape((xres,yres))
	fig = plt.figure(figsize= (8,8), dpi= DPI)
	c= plt.imshow(img, cmap = 'jet')

	ax = plt.gca()
	x_ticks = np.array([0, xx.shape[0]//4, xx.shape[0]//2, xx.shape[0]*3//4, xx.shape[0]-1])
	y_ticks = np.array([0, yy.shape[0]//4, yy.shape[0]//2, yy.shape[0]*3//4, yy.shape[0]-1])
	ax.set_xticks(x_ticks)
	ax.set_yticks(y_ticks)
	ax.set_xticklabels(np.array([-1, -0.5, 0, 0.5, 1]))	
	ax.set_yticklabels(np.array([-1, -0.5, 0, 0.5, 1]))
	plt.xlabel(r'$w_1$')
	plt.ylabel(r'$w_0$')
	plt.title('Heatmap of weights')
	fig.colorbar(c, ax=ax)

	path = os.path.join(output_folder, "q1_heatmap_train_" + str(points) + ".png")
	print("Saving to "+ path)
	plt.savefig(path)
	plt.close()

	print("Mean and covariance of posterior are")
	print(mean)
	print(covariance)
	return mean[:,0], covariance

def sample_and_plot(x, y, m1, s1, points= 0):

	fig = plt.figure(figsize= (8,8), dpi= DPI)
	for i in range(num_points):
		w1, w0 = np.random.multivariate_normal(mean=m1, cov= s1)
		plt.plot(x, get_func(x, w0, w1), linewidth= lw)
	plt.plot(x, y, 'k--', linewidth= 2+lw, label= 'Ground Truth')	
	plt.xlim([-1, 1])
	plt.ylim([-4.5, 4.5])
	plt.xlabel('x')
	plt.ylabel('y')
	plt.grid(True)
	plt.legend(loc= 2)
	path = os.path.join(output_folder, "q1_fit_train_" + str(points) + ".png")
	print("Saving to "+ path)
	plt.savefig(path)
	plt.close()

#==============================================================================
# Main starts here
#==============================================================================
DPI= 200
fs = 20
lw = 1.5
matplotlib.rcParams.update({'font.size': fs})
w0_gd = -0.3
w1_gd = 0.5
num_points = 20
mu = 0
sigma = 0.2
alpha = 2
beta  = 25

output_folder = "output"
np.random.seed(123)
x = np.sort(np.random.uniform(-1,1, num_points))
y = get_func(x, w0_gd, w1_gd)
y_noisy = y + np.random.normal(mu, sigma, num_points)

# Draw Gaussian
m1 = np.array([0.,0])
s1 = alpha*np.eye(2)

mean, covariance = plot_weights(m1, s1)
sample_and_plot(x, y, mean, covariance, points= 0)

mean, covariance = plot_weights(m1, s1, likelihood= True, points= 1)
sample_and_plot(x, y, mean, covariance, points= 1)

mean, covariance = plot_weights(m1, s1, likelihood= True, points= 2)
sample_and_plot(x, y, mean, covariance, points= 2)

mean, covariance = plot_weights(m1, s1, likelihood= True, points= 5)
sample_and_plot(x, y, mean, covariance, points= 5)

"""
mean, covariance = plot_weights(m1, s1, likelihood= True, points= 10)
sample_and_plot(x, y, mean, covariance, points= 10)

mean, covariance = plot_weights(m1, s1, likelihood= True, points= 15)
sample_and_plot(x, y, mean, covariance, points= 15)
"""

mean, covariance = plot_weights(m1, s1, likelihood= True, points= 20)
sample_and_plot(x, y, mean, covariance, points= 20)