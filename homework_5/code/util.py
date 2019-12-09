

import numpy as np
from matplotlib import pyplot as plt
from numpy import genfromtxt
from scipy.special import expit as logistic
import scipy.stats as stats
import copy

from util import *

TOLERANCE = 1e-5
INFINITY = 1000000

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

def get_affine(x, m, c):
    """
        Affine Transformations of x
    """
    x = m*x + c
    return x

def affine_sigmoid(xin, m= 10, c= 3):
    """
        Calculates affine transformations of sigmoid
    """
    if type(xin) != np.ndarray:
        x = np.array([xin])
    else:
        x = xin

    x = get_affine(x, m, c)
    output = get_sigmoid(x)

    if type(xin) != np.ndarray:
        return output[0]
    else:
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

def get_prediction_likelihood_without_complications(test_data, test_label, weight):
    """
        Returns prediction likelihood on a sample weight without using any hessian
        test_data  = N x D
        test_label = N 
        weight     = D x 1
    """
    pred, _ = get_output(weight, test_data)
    pred = pred[:,0]
    pred_like = np.multiply(test_label, np.log(pred + TOLERANCE)) + np.multiply(1.0-test_label, np.log(1.0-pred+ TOLERANCE))
    return np.exp(np.mean(pred_like))

def get_output(weight, data, regression= "logistic"):
    """
        Output of different regression. Taken from Assignment 2.
        returns #examples x 1 arrays
    """
    dot_product = np.matmul(data,weight)
    if regression == "logistic":
        output = get_sigmoid(dot_product)
    elif regression == "probit":
        output = norm.cdf(dot_product)
    elif regression == "multiclass":
        output = softmax(dot_product, axis=1)

    return output, dot_product

def predict_and_test(weight, test_data, test_label, regression= "logistic"):
    """
        Predict and test on test data and get accuracy
    """
    pred_test, _ = get_output(weight, test_data, regression= regression)
    acc          = get_accuracy(pred_test, test_label, regression= regression)
    return acc, pred_test

def get_log_likelihood(phi, pred, t, dot_product, weight, reg= 1):
    """
        Returns log likelihood of the logistic regression
        t = N x 1
    """
    prior = -0.5* np.sum(np.multiply(weight, weight))
    likelihood = np.multiply(t, np.log(pred+TOLERANCE)) + np.multiply(1.0- t, np.log(1.0-pred+TOLERANCE))
    likelihood = np.sum(likelihood)

    return prior + likelihood

def get_minus_log_joint(z, mean, Sigma):
    """
        Returns minus log joint for 2D Gaussian
    """
    data_minus_mean = z[np.newaxis,:] - mean # 1 x 2
    Sigma_inv       = np.linalg.inv(Sigma)   # 2 x 2
    likelihood      = 0.5*np.matmul( np.matmul(data_minus_mean, Sigma_inv), data_minus_mean.T) # 1 x 1
    likelihood      = likelihood[0][0]
    return likelihood

def get_gradient(phi, pred, t, dot_product, weight, reg= 1, regression= "logistic"):
    """
        Returns log likelihood of the logistic regression. Taken from Assignment 2
        t = (N, 1)
        weight = (D, 1)
    """
    if regression == "logistic":
        gradient = np.matmul(phi.T, pred - t)
    elif regression == "probit":
        R = np.eye(pred.shape[0])
        for i in range(pred.shape[0]):
            y_n  = pred[i,0]
            dotp = dot_product[i, 0]
            pdf  = norm.pdf(dotp)
            R[i,i] = pdf/(y_n*(1-y_n) + TOLERANCE)
        gradient = np.matmul(np.matmul(phi.T, R), pred-t)
    elif regression == "multiclass":
        gradient = np.matmul(phi.T, pred - t)

    # Add regularization
    gradient += weight/ reg
    return gradient

def get_gradient_minus_log_joint(z, mean, Sigma):
    """
        Returns gradient of minus log joint for 2D Gaussian
    """
    data_minus_mean = z[np.newaxis,:] - mean# 1 x 2
    Sigma_inv       = np.linalg.inv(Sigma)  # 2 x 2
    gradient        = - np.matmul(Sigma_inv, data_minus_mean.T) #2 x 1
    gradient        = gradient[0]
    return gradient

def get_hessian(phi, pred, t, dot_product, reg= 1, regression= "logistic"):
    """
        Hessian of different regression. Taken from Assignment 2.
        t = true labels (N, 1)
        weight = (D, 1)
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

def gass_hermite_quad(f, degree, m, c):
    """
    Calculates the integral e^{-x^2} f(x) dx numerically.
    :param f: target function, takes a array as input x = [x0, x1,...,xn], and return a array of function values f(x) = [f(x0),f(x1), ..., f(xn)]
    :param degree: integer, >=1, number of points
    :return:
    """

    points, weights = np.polynomial.hermite.hermgauss( degree)

    #function values at given points
    f_x = f(points, m= m, c= c)

    #weighted sum of function values
    F = np.sum( f_x  * weights)

    return F

def to_accept(x, x_new):
    """
        Acceptance rule of samples with log inputs
    """
    if x_new>x:
        return True
    else:
        accept=np.random.uniform(0,1)
        # Since we did a log likelihood, we need to exponentiate in order to compare to the random number
        # less likely x_new are less likely to be accepted
        return (accept < (np.exp(x_new-x)))

def to_accept_without_log(x, x_new):
    """
        Acceptance rule without any log. 
    """
    if x_new>x:
        return True
    else:
        accept=np.random.uniform(0,1)
        return (accept < x_new/(x+TOLERANCE))

def to_accept_hamiltonian(dH):
    """
        Acceptance Rule for Hamiltonian
    """
    if (dH <= 0.0):
        return True
    else:
        u = np.random.uniform(0.0,1.0)
        if (u < np.exp(-dH)):
            return True
        else:
            return False

def get_PE(log_lld):
    """
        Returns PE from the log likelihood
    """
    return -log_lld
                    
def get_KE(p, scale= 1):
    """ 
        Returns KE from the momentum vector
    """
    p = p.flatten()
    return scale * 0.5*np.sum(np.multiply(p, p))

def get_hamiltonian(z, mean, covar, p, scale= 1):
    PE = get_minus_log_joint         (z, mean[np.newaxis, :], covar)
    KE = get_KE(p, scale= scale)
    return PE+KE

def get_prob_from_energy(energy):
    return np.exp(-energy)
        
def hybrid_monte_carlo(train_data, train_label, z_init, num_iterations, epsilon, num_leapfrog_steps, collect_final_sample_frequency= 10, display_frequency= 5000, scale_KE= 1):
    """
        Gets posterior samples for Bayes Logistic Regression using HMC algorithm
        z_int= (dim, 1)
    """
    dim = train_data.shape[1]
    z = z_init

    accepted = [] # Keeps track of accepted samples
    sampled  = [] # Keeps track of all samples
    final    = [] # Keeps track of final samples which are sampled in a cyclic manner
    
    for i in range(num_iterations):
        # Old energy = -loglik and Old gradient
        pred, dot_product = get_output(z, train_data)
        old_PE   =  -get_log_likelihood  (phi= train_data, pred= pred, t= train_label[:, np.newaxis], dot_product= dot_product, weight= z)
        
        # There is no minus since gradient function returns gradient of negative log likelihood
        old_grad =  get_gradient(phi= train_data, pred= pred, t= train_label[:, np.newaxis], dot_product= dot_product, weight= z)

        new_z = np.copy(z)              # deep copy of array
        new_grad  = np.copy(old_grad)   # deep copy of array

        # draw random momentum vector from unit Gaussian which decides the energy
        # given out for exploration
        p = np.random.normal(0.0, 1.0, (dim, 1))

        # Compute Hamiltonian
        H = get_KE(p, scale= scale_KE) + old_PE

        # Suggest new candidate using gradient + Hamiltonian dynamics.
        # Leapfrog
        for j in range(num_leapfrog_steps):  
            # Make first half step in p, full step in z and then again half step in p
            p        -= (epsilon/2.0)*new_grad
            new_z    += epsilon*p
            pred, dot_product = get_output(new_z, train_data)
            new_grad  = get_gradient(phi= train_data, pred= pred, t= train_label[:, np.newaxis], dot_product= dot_product, weight= new_z)
            p        -= (epsilon/2.0)*new_grad

        # Compute new Hamiltonian
        pred, dot_product = get_output(new_z, train_data)
        new_PE = -get_log_likelihood(phi= train_data, pred= pred, t= train_label[:, np.newaxis], dot_product= dot_product, weight= new_z)
        new_H  = get_KE(p, scale= scale_KE) + new_PE
        
        sampled.append(new_z)
        
        # Accept new candidate in Monte-Carlo fashion.
        if to_accept_without_log(get_prob_from_energy(H), get_prob_from_energy(new_H)):            
            z = new_z
            accepted.append(new_z)

        if i % collect_final_sample_frequency == 0:
            # Sample from the current parameters
            final.append(z)

        if (i+1) % display_frequency == 0 or i == num_iterations-1:
            print("Iter {:6d} done".format(i+1))
    
    return np.array(accepted), np.array(sampled), np.array(final), z

def gibbs(train_data, train_label, w_init, num_iterations, display_frequency, collect_final_sample_frequency):
    """
        Gets posterior samples for Bayes Logistic Regression using Gibbs Sampling
        z_int= (dim, 1)
    """        
    dim = train_data.shape[1]
    w_covariance = np.eye(dim) + np.sum(np.matmul(train_data[:, :, np.newaxis], train_data[:, np.newaxis, :]), axis= 0)
    w_covariance = np.linalg.inv(w_covariance)
    
    w = w_init
    _, z = get_output(w, train_data)
    sigma = 1
        
    sampled  = [] # Keeps track of all samples
    final    = [] # Keeps track of final samples which are sampled in a cyclic manner

    for i in range(num_iterations):
        # Sample weight
        w_new_mean = np.matmul(w_covariance, np.sum(z*train_data, axis= 0)[:, np.newaxis]) # dim x 1
        w          = np.random.multivariate_normal(w_new_mean[:, 0], w_covariance, 1).T # dim x 1
        
        # Now sample hidden variable
        _, z_new = get_output(w, train_data)
        lower=          np.zeros((train_data.shape[0], 1))
        upper= INFINITY*np.ones ((train_data.shape[0], 1))        
        lower[train_label < 0.5, :] = -INFINITY
        upper[train_label < 0.5, :] = 0


        X = stats.truncnorm((lower - z_new) / sigma, (upper - z_new) / sigma, loc= z_new, scale= sigma)        
        z_new = X.rvs((train_data.shape[0],1))
        
        z = copy.deepcopy(z_new)
        
        if i % collect_final_sample_frequency == 0:
            # Sample from the current parameters
            final.append(w)

        if (i+1) % display_frequency == 0 or i == num_iterations-1:
            print("Iter {:6d} done".format(i+1))
    
    return np.array(final), w  
    
def test_on_posterior(test_data, test_label, posterior_samples):
    """
        Returns stats on posterior samples
    """
    print("Testing on posterior samples...")
    num_posterior_samples = posterior_samples.shape[0]
    avg_pred_test    = np.zeros((num_posterior_samples, ))
    avg_pred_log_lld = np.zeros((num_posterior_samples, ))
                    
    for k in range(num_posterior_samples):
        # Use the posterior samples
        w_sampled = posterior_samples[k]
        
        # Get the hessian
        #pred, dot_product = get_output(w_sampled, train_data)
        #hessian  = get_hessian (phi= train_data, pred= pred[:, np.newaxis], t= train_label[:, np.newaxis], dot_product= dot_product)
        
        pred_test, _         = get_output  (w_sampled, test_data)
        acc                  = get_accuracy(pred_test, test_label) 
        pred_likelihood      = get_prediction_likelihood_without_complications(test_data, test_label, w_sampled) #get_prediction_likelihood(test_data, test_label, w_sampled, hessian)
        avg_pred_test[k]     = acc
        avg_pred_log_lld [k] = np.log(pred_likelihood)
        
        if (k+1)%100 == 0 or k== num_posterior_samples-1:
            print("{:5d} Posterior Weight samples Test_data Pred_acc= {:.2f}, Pred_log_likelihood= {:.2f}".format(k+1, np.mean(avg_pred_test[:k]), np.mean(avg_pred_log_lld[:k])))    
