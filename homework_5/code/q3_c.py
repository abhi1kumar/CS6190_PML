

"""
    Implements a Bayesian MLP
    References-
    Pytorch           https://github.com/pytorch/examples/blob/master/mnist/main.py
    Bayes by Backprop https://www.nitarshan.com/bayes-by-backprop/
"""
import numpy as np
import matplotlib.pyplot as plt
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from CustomDatasetFromCSV  import CustomDatasetFromCSV

TOLERANCE       = 1e-5
INPUT_DIM       = 5
NUM_CLASSES     = 2
TRAIN_SAMPLES   = 100 # Samples to draw for train
TEST_SAMPLES    = 100 # Samples to draw for test
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AverageMeter(object):
    """
        Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Gaussian(nn.Module):
    """
        A Gaussian Object which keeps track of the mean and sigma. Also implements
        the reparametrization trick
    """
    def __init__(self, mu, rho, trainable= True):
        super().__init__()

        self.trainable = trainable
        self.normal = torch.distributions.Normal(0,1)

        # Since we are initializing the tensors with requires_grad=True, we 
        # should wrap them into nn.Parameter, so that they will be properly 
        # registered in the state_dict and will be automatically pushed to the 
        # device, if we call model.to(device).
        # Reference https://discuss.pytorch.org/t/resolved-runtimeerror-expected-device-cpu-and-dtype-float-but-got-device-cuda-0-and-dtype-float/54783/8
        if self.trainable:
            self.mu     = nn.Parameter(0. + torch.Tensor(mu))
            self.rho    = nn.Parameter(0. + torch.Tensor(rho))
        else:
            self.mu     = torch.Tensor(mu ).to(DEVICE)
            self.rho    = torch.Tensor(rho).to(DEVICE)
            self.epsilon= self.normal.sample(self.rho.shape).type(self.mu.type())

    def sample(self):
        if self.trainable:
            # Sample a new epsilon
            epsilon = self.normal.sample(self.rho.shape).type(self.mu.type())
        else:
            # Use the saved epsilon now
            epsilon = self.epsilon

        # Reparametrization trick
        return self.mu + self.rho* epsilon

    def log_prob(self, input):
        output = (-math.log(math.sqrt(2 * math.pi)) - torch.log(self.rho) - ((input - self.mu) ** 2) / (2 * self.rho ** 2)).sum()
        return output

class BayesianLinear(nn.Module):
    """
        A Bayesian Linear Module.
        While training, it samples the weights.
    """
    def __init__(self, in_features, out_features):
        super(BayesianLinear, self).__init__()
        self.in_features  = in_features
        self.out_features = out_features

        scale = 1.0/np.sqrt(self.in_features)

        # Weight and bias parameters with standard normal
        self.weight_mu  = nn.Parameter(torch.zeros(out_features, in_features))
        self.weight_rho = nn.Parameter(scale *torch.ones (out_features, in_features))
        self.bias_mu    = nn.Parameter(torch.zeros(out_features, ))
        self.bias_rho   = nn.Parameter(scale *torch.ones(out_features, ))
        
        # Initializations
        self.weight = Gaussian(self.weight_mu, self.weight_rho)
        self.bias   = Gaussian(self.bias_mu  , self.bias_rho)

        # Prior distributions
        #self.weight_prior = Gaussian(self.weight_mu, self.weight_rho, trainable= False)
        #self.bias_prior   = Gaussian(self.bias_mu  , self.bias_rho, trainable= False)
        
        self.log_prior    = 0.0
        self.log_variational_posterior = 0.0

    def forward(self, input, sample=False, calculate_log_probs=False): 
        if self.training or sample:
            # While training, sample a weight bias
            weight = self.weight.sample()
            bias   = self.bias.sample()
        else:
            # While testing and sample is False, use the means
            weight = self.weight.mu
            bias   = self.bias.mu

        if self.training or calculate_log_probs:
            self.log_prior                 = 0     #self.weight_prior.log_prob(weight) + self.bias_prior.log_prob(bias)
            self.log_variational_posterior = self.weight.log_prob(weight)       + self.bias.log_prob(bias)
        else:
            self.log_prior, self.log_variational_posterior = 0.0, 0.0

        output = F.linear(input, weight, bias)
        return output

class BayesianMLP(nn.Module):
    """
        A Bayesian MLP network with different number of nodes and activations
    """
    def __init__(self, num_nodes= 20, activation= "relu"):
        super(BayesianMLP, self).__init__()
        self.num_nodes = num_nodes

        self.l1 = BayesianLinear(INPUT_DIM     , self.num_nodes)
        self.l2 = BayesianLinear(self.num_nodes, self.num_nodes)
        self.l3 = BayesianLinear(self.num_nodes, NUM_CLASSES)

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()

    def forward(self, x, sample= False):
        x = self.l1(x, sample= sample)
        x = self.activation(x)
        x = self.l2(x, sample= sample)
        x = self.activation(x)
        x = self.l3(x, sample= sample)
        x = F.log_softmax  (x, dim=1)
        return x

    def log_prior(self):
        return self.l1.log_prior + self.l2.log_prior + self.l3.log_prior

    def log_variational_posterior(self):
        return self.l1.log_variational_posterior + self.l2.log_variational_posterior + self.l2.log_variational_posterior

    def sample_elbo(self, input, target, samples= 100, batch_size= 1000):
        outputs                    = torch.zeros(samples, batch_size, NUM_CLASSES).to(DEVICE)
        log_priors                 = torch.zeros(samples).to(DEVICE)
        log_variational_posteriors = torch.zeros(samples).to(DEVICE)
        preds                      = torch.zeros(samples, batch_size).type(torch.LongTensor).to(DEVICE)
        corrects                   = torch.zeros(samples).to(DEVICE)

        for i in range(samples):
            outputs[i]                    = self.forward(input, sample= True)
            log_priors[i]                 = self.log_prior()
            log_variational_posteriors[i] = self.log_variational_posterior()
            preds[i]                      = outputs[i].argmax(dim= 1)
            corrects[i]                   = preds[i].eq(target.view_as(preds[i])).sum()

        log_prior                 = log_priors.mean()
        log_variational_posterior = log_variational_posteriors.mean()
        negative_log_likelihood   = F.nll_loss(outputs.mean(dim= 0), target, reduction='mean')
        loss                      = negative_log_likelihood #(log_variational_posterior - log_prior)+ negative_log_likelihood
        correct                   = corrects.mean()

        return loss, negative_log_likelihood, log_prior, log_variational_posterior, correct

def train(model, train_loader, optimizer, epoch, train_batch_size):
    # Meter objects to track the logs
    loss_meter = AverageMeter()
    nll_meter  = AverageMeter()
    acc_meter  = AverageMeter()

    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data   = data.to(DEVICE)
        target = target.to(DEVICE)

        model.zero_grad()
        loss, negative_log_likelihood, log_prior, log_variational_posterior, correct  = model.sample_elbo(data, target, samples= TRAIN_SAMPLES, batch_size= train_batch_size)

        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), data.shape[0])
        nll_meter .update(negative_log_likelihood.item(), data.shape[0])
        acc_meter .update(correct .item()/float(data.shape[0]), data.shape[0])

        if (batch_idx+1) % 100 == 0:
            print('Train Epoch= {:4d} [{}/{}] Loss= {:8.2f} NLL= {:8.2f} Acc= {:8.2f}%'.format(epoch, batch_idx, len(train_loader), loss_meter.avg, nll_meter.avg, 100.*acc_meter.avg))

    return loss_meter.avg, nll_meter.avg, acc_meter.avg

def test(model, test_loader, epoch, test_batch_size):
    # Meter objects to track the logs
    loss_meter = AverageMeter()
    nll_meter  = AverageMeter()
    acc_meter  = AverageMeter()

    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data   = data.to(DEVICE)
            target = target.to(DEVICE)

            loss, negative_log_likelihood, log_prior, log_variational_posterior, correct  =  model.sample_elbo(data, target, samples= TEST_SAMPLES, batch_size= test_batch_size)

            loss_meter.update(loss.item(), data.shape[0])
            nll_meter .update(negative_log_likelihood.item(), data.shape[0])
            acc_meter .update(correct .item()/float(data.shape[0]), data.shape[0])

            if (batch_idx+1) % 100 == 0:
                print('Test  Epoch= {:4d} [{}/{}] Loss= {:8.2f} NLL= {:8.2f} Acc= {:8.2f}%'.format(epoch, batch_idx, len(train_loader), loss_meter.avg, nll_meter.avg, 100.*acc_meter.avg))

    return loss_meter.avg, nll_meter.avg, acc_meter.avg

#===============================================================================
# Main starts here
#===============================================================================
epochs          = 1000
workers         = 4
train_batch_size= 872
test_batch_size = 500

#===============================================================================
# Get the data loaders
#===============================================================================
train_dataset = CustomDatasetFromCSV("data/bank-note/train.csv")
train_loader  = torch.utils.data.DataLoader(dataset= train_dataset,  batch_size= train_batch_size, shuffle= True , num_workers = workers)

val_dataset   = CustomDatasetFromCSV("data/bank-note/test.csv")
val_loader    = torch.utils.data.DataLoader(dataset= val_dataset  ,  batch_size= test_batch_size , shuffle= False, num_workers = workers)

lr_list         = [0.001]#, 0.0005, 0.0001, 0.00001]
num_nodes_list  = [10, 20, 50]
act_list        = ["relu", "tanh"]

for activation in act_list:
    for num_nodes in num_nodes_list:
        for lr in lr_list:
            print("\n===============================================================================");
            print("Using device: {}".format(DEVICE))
            print("epochs             = {}"    .format(epochs))
            print("Batch size         = {}"    .format(train_batch_size))
            print("Test Batch size    = {}"    .format(test_batch_size))
            print("lr                 = {:.5f}".format(lr))
            print("Activation         = {}"    .format(activation))
            print("Num_nodes          = {}"    .format(num_nodes))
            print("-------------------------------------------------------------------------------");
            #===============================================================================
            # Get the model and optimizer
            #===============================================================================
            model = BayesianMLP(num_nodes= num_nodes, activation= activation).to(DEVICE)
            optimizer = torch.optim.Adam(model.parameters(), lr = lr)

            for epoch in range(epochs):
                train_loss, train_nll, train_acc = train(model, train_loader, optimizer, epoch, train_batch_size)
                test_loss , test_nll , test_acc  = test (model, val_loader  , epoch, test_batch_size)
                print("Epoch= {:4d} \tTrain_loss= {:5.2f} Train_NLL= {:5.2f} Train_LL= {:5.2f} Train_acc= {:8.2f}% \tTest_loss= {:5.2f} Test_NLL= {:5.2f} Test_LL= {:5.2f} Test_acc= {:8.2f}%".format(epoch, train_loss, train_nll, -train_nll, 100.*train_acc, test_loss, test_nll, -test_nll, 100.*test_acc))


