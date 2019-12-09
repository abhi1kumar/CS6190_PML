

"""
    Custom Dataloader in Pytorch
"""
import numpy as np
from util import *

import torch
from torch.utils.data.dataset import Dataset

class CustomDatasetFromCSV(Dataset):
    """
        Custom Dataset reader from CSV
    """
    def __init__(self, csv_path):
        """
            Args:
            csv_path (string): path to csv file
        """

        # Read the csv file separated by whitespace
        self.all_data = readcsv(csv_path)

        self.label_arr = self.all_data[:,-1]
        self.feature_arr  = np.hstack((self.all_data[:,0:-1], np.ones((self.all_data.shape[0],1))))
        self.data_len  = self.all_data.shape[0]


    def __getitem__(self, index):
        # Return feature and label
        # Feature as float
        feature_as_tensor = torch.tensor(self.feature_arr[index]).type(torch.FloatTensor)
        # Label as integer
        label_as_tensor   = torch.tensor(self.label_arr[index])  .type(torch.LongTensor)

        # Return feature and label
        return (feature_as_tensor, label_as_tensor)

    def __len__(self):
        return self.data_len

