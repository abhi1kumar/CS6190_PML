

import numpy as np
import csv
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from util import *

dpi = 200
fs = 20
lw = 2.5
matplotlib.rcParams.update({'font.size': fs})

input_file =  "logs/q3_c_20_tanh.log"
output_file = "logs/out.txt"

def remove_csv_columns(input_csv, output_csv):
    with open(input_csv) as file_in, open(output_csv, 'w') as file_out:
        reader = csv.reader(file_in, delimiter= " ", skipinitialspace=True)
        writer = csv.writer(file_out, delimiter= " ")
        writer.writerows([col.replace("%","") for idx, col in enumerate(row)  if idx %2 == 1] for row in reader)

remove_csv_columns(input_file, output_file)

# Load the processed file
data = np.genfromtxt(output_file)

print(data.shape)

# Plot train NLL (col 3), train acc (col 4), test NLL (col 7) and test acc (col 8)
epoch     = data[:,0]
train_nll = data[:,3]
train_acc = data[:,4]
test_nll  = data[:,7]
test_acc  = data[:,8]

fig = plt.figure(figsize= (9.6,6), dpi= dpi)
plt.plot(epoch, train_nll , lw= lw, label= 'Train  LL', color= 'dodgerblue' )
plt.xlabel('Epoch')
plt.ylabel('Log LLd')
plt.xlim((0, epoch.shape[0]))
plt.ylim((-1.0, 0.01))
plt.grid(True)
plt.legend(loc= 'upper left')
path = "images/q3_c_nll_acc_train.png"
savefig(plt, path= path)
plt.close()

fig = plt.figure(figsize= (9.6,6), dpi= dpi)
plt.plot(epoch, test_nll , lw= lw, label= 'Test  LL', color= 'orange' )
plt.xlabel('Epoch')
plt.ylabel('Log LLd')
plt.xlim((0, epoch.shape[0]))
plt.ylim((-1.0, 0.01))
plt.grid(True)
plt.legend(loc= 'upper left')
path = "images/q3_c_nll_acc_test.png"
savefig(plt, path= path)
plt.close()
