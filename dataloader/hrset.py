"""
HR survey data Loading modules
"""
import os
import copy
import random
import numpy as np
import pandas as pd

from itertools import *

import torch
from torch.utils.data import Dataset


class IBMset(Dataset):
    """ IBM turnover dataset """
    def __init__(self, data_path="datasets", transform=None, sampling_rate=1.0, balanced=True):
        # Load IBM HR dataset
        ibm_path = os.path.join(data_path, 'IBM/ibmData.csv')
        ibm_data = pd.read_csv(ibm_path)
        
        self.rawx = ibm_data
        
        # Remove label
        raw_x = ibm_data.drop(columns=['Attrition'])
        # Remove invariant features 
        raw_x = raw_x.drop(columns=['StandardHours', 
                                    'EmployeeCount', 
                                    'Over18'
                                    ])
        
        # Reassign numerical values to binary categorical features
        for col, lab1, lab2 in [('Gender', 'Male', 'Female'),
                                ('MaritalStatus', 'Single', 'Married'),
                                ('OverTime', 'Yes', 'No')]:
            raw_x.loc[raw_x[col] == lab1, col] = 1
            raw_x.loc[raw_x[col] == lab2, col] = 0
        
        # Apply one-hot encoding
        for col in ['BusinessTravel', 'Department', 'EducationField', 'JobRole'] :
            one_hot = pd.get_dummies(raw_x[col], prefix=col)
            # Drop column 'col' as it is now encoded
            raw_x = raw_x.drop(col, axis = 1)
            # Join the encoded columns
            raw_x = raw_x.join(one_hot)
            
        # Normalize (Min-Max Scaling)
        nom_x = (raw_x - raw_x.min(numeric_only=True))/(raw_x.max(numeric_only=True) - raw_x.min(numeric_only=True))
        nom_x = nom_x.fillna(0)
        self.x = torch.tensor(nom_x.values.astype(np.float64))
        
        print("===IBM set===")
        print("#Original Samples: %d"%(len(raw_x)))
        
        # Get Labels
        self.y = [1 if raw_y == 'Yes' else 0 for raw_y in ibm_data['Attrition']]
        self.y = torch.tensor(self.y)#.unsqueeze(1).float()
        
        # Make the number of samples from each class identical
        if balanced :
            idx_pos = [i for i, e in enumerate(self.y) if e != 0]
            idx_neg = [i for i, e in enumerate(self.y) if e == 0]
            
            sc = max(len(idx_pos), len(idx_neg))
            idx_pos = torch.tensor(np.random.choice(idx_pos, sc, replace=True))
            idx_neg = torch.tensor(np.random.choice(idx_neg, sc, replace=True))
            
            indices = torch.cat((idx_pos, idx_neg))
            
            self.x = self.x[indices].clone().detach()
            self.y = self.y[indices].clone().detach()
        
        # Data Sampling
        if sampling_rate < 1.0 :
            num_samples = int(sampling_rate * len(self.y))
            indices = torch.tensor(random.sample(range(len(self.y)), num_samples))
            
            self.x = self.x[indices].clone().detach()
            self.y = self.y[indices].clone().detach()
            
        self.num_samples = self.x.shape[0]
        
    def numClasses(self):
        """ Return num of classes """
        return 2
        
    def getRawData(self):
        """ Return original survey data """
        return self.rawx
    
    def getProcessedData(self):
        """ Return pre-processed survey data """
        return self.rawx
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, item):
        imgs, labels = self.x[item], self.y[item]
        
        return imgs, labels
    

class FEVSset(Dataset):
    """ Federal Employee Viewpoint Survey dataset """
    def __init__(self, data_path="datasets", transform=None, sampling_rate=1.0, balanced=True):
        # Load IBM HR dataset
        data_path = os.path.join(data_path, 'FEVS/fevsSmall.csv')
        raw_data = pd.read_csv(data_path)
        
        num_ori_data = len(raw_data)
        
        self.rawx = raw_data
        
        # Remove Samples who have planned to retire (Class: D)
        raw_data = raw_data.loc[raw_data['DLEAVING'] != 'D']
        
        # Remove Samples having any 'Nan' values  
        raw_data = raw_data.dropna()
        
        # Drop rows that contain 'X' Values in question columns
        for col in raw_data.columns:
            raw_data = raw_data.loc[raw_data[col] != 'X']
        
        self.fx = copy.deepcopy(raw_data)
        
        # Remove label
        raw_x = raw_data.drop(columns=['DLEAVING'])
        # Remove invariant (ex. POSTWT, RANDOM) features or some categories with a huge range ('AGENCY', 'LEVEL1')
        raw_x = raw_x.drop(columns=['AGENCY', 
                                    'LEVEL1', 
                                    'POSTWT', 
                                    'RANDOM'
                                    ])
        
        # Reassign numerical values to binary categorical features
        for col, lab1, lab2 in [('DSEX', 'A', 'B'),
                                ('DSUPER', 'A', 'B'),
                                ('DMINORITY', 'A', 'B')]:
            raw_x.loc[raw_x[col] == lab1, col] = 1
            raw_x.loc[raw_x[col] == lab2, col] = 0
        
        # Apply one-hot encoding
        for col in ['DEDUC', 'DFEDTEN'] :
            one_hot = pd.get_dummies(raw_x[col], prefix=col)
            # Drop column 'col' as it is now encoded
            raw_x = raw_x.drop(col, axis = 1)
            # Join the encoded columns
            raw_x = raw_x.join(one_hot)
            
        print("===FEVS set===")
        print("#Original Samples: %d  #Dropped Samples: %d  #Trainable Samples: %d"
              %(num_ori_data, num_ori_data-len(raw_x), len(raw_x)))
            
        raw_x = raw_x.astype(str).astype(float)
        self.ppx = raw_x
        
        # Normalize (Min-Max Scaling)
        nom_x = (raw_x - raw_x.min(numeric_only=True))/(raw_x.max(numeric_only=True) - raw_x.min(numeric_only=True))
        self.x = torch.tensor(nom_x.values.astype(np.float64))
        
        # Get Labels
        self.y = [0 if raw_y == 'A' else 1 for raw_y in raw_data['DLEAVING']]
        self.y = torch.tensor(self.y)
        
        # Make the number of samples from each class identical
        if balanced :
            idx_pos = [i for i, e in enumerate(self.y) if e != 0]
            idx_neg = [i for i, e in enumerate(self.y) if e == 0]
            
            sc = max(len(idx_pos), len(idx_neg))
            idx_pos = torch.tensor(np.random.choice(idx_pos, sc, replace=True))
            idx_neg = torch.tensor(np.random.choice(idx_neg, sc, replace=True))
            
            indices = torch.cat((idx_pos, idx_neg))
            
            self.x = self.x[indices].clone().detach()
            self.y = self.y[indices].clone().detach()
        
        # Data Sampling
        if sampling_rate < 1.0 :
            num_samples = int(sampling_rate * len(self.y))
            indices = torch.tensor(random.sample(range(len(self.y)), num_samples))
            
            self.x = self.x[indices].clone().detach()
            self.y = self.y[indices].clone().detach()
            
        self.num_samples = self.x.shape[0]
        
    def numClasses(self):
        """ Return num of classes """
        return 2
        
    def getRawData(self):
        """ Return original survey data """
        return self.rawx
        
    def getFilteredData(self):
        """ Return data samples not including NaN (or 'X') values """
        return self.fx
    
    def getProcessedData(self):
        """ Return pre-processed survey data """
        return self.ppx
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, item):
        imgs, labels = self.x[item], self.y[item]
        
        return imgs, labels
    

class IBMset(Dataset):
    """ IBM turnover dataset """
    def __init__(self, data_path="datasets", transform=None, sampling_rate=1.0, balanced=True):
        # Load IBM HR dataset
        ibm_path = os.path.join(data_path, 'IBM/ibmData.csv')
        ibm_data = pd.read_csv(ibm_path)
        
        self.rawx = ibm_data
        
        # Remove label
        raw_x = ibm_data.drop(columns=['Attrition'])
        # Remove invariant features 
        raw_x = raw_x.drop(columns=['StandardHours', 
                                    'EmployeeCount', 
                                    'Over18'
                                    ])
        
        # Reassign numerical values to binary categorical features
        for col, lab1, lab2 in [('Gender', 'Male', 'Female'),
                                ('MaritalStatus', 'Single', 'Married'),
                                ('OverTime', 'Yes', 'No')]:
            raw_x.loc[raw_x[col] == lab1, col] = 1
            raw_x.loc[raw_x[col] == lab2, col] = 0
        
        # Apply one-hot encoding
        for col in ['BusinessTravel', 'Department', 'EducationField', 'JobRole'] :
            one_hot = pd.get_dummies(raw_x[col], prefix=col)
            # Drop column 'col' as it is now encoded
            raw_x = raw_x.drop(col, axis = 1)
            # Join the encoded columns
            raw_x = raw_x.join(one_hot)
            
        #self.rawx = raw_x
            
        # Normalize (Min-Max Scaling)
        nom_x = (raw_x - raw_x.min(numeric_only=True))/(raw_x.max(numeric_only=True) - raw_x.min(numeric_only=True))
        nom_x = nom_x.fillna(0)
        self.x = torch.tensor(nom_x.values.astype(np.float64))
        
        print("===IBM set===")
        print("#Original Samples: %d"%(len(raw_x)))
        
        # Get Labels
        self.y = [1 if raw_y == 'Yes' else 0 for raw_y in ibm_data['Attrition']]
        self.y = torch.tensor(self.y)
        
        # Make the number of samples from each class identical
        if balanced :
            idx_pos = [i for i, e in enumerate(self.y) if e != 0]
            idx_neg = [i for i, e in enumerate(self.y) if e == 0]
            
            sc = max(len(idx_pos), len(idx_neg))
            idx_pos = torch.tensor(np.random.choice(idx_pos, sc, replace=True))
            idx_neg = torch.tensor(np.random.choice(idx_neg, sc, replace=True))
            
            indices = torch.cat((idx_pos, idx_neg))
            
            self.x = self.x[indices].clone().detach()
            self.y = self.y[indices].clone().detach()
        
        # Data Sampling
        if sampling_rate < 1.0 :
            num_samples = int(sampling_rate * len(self.y))
            indices = torch.tensor(random.sample(range(len(self.y)), num_samples))
            
            self.x = self.x[indices].clone().detach()
            self.y = self.y[indices].clone().detach()
            
        self.num_samples = self.x.shape[0]
        
    def numClasses(self):
        """ Return num of classes """
        return 2
        
    def getRawData(self):
        """ Return original survey data """
        return self.rawx
    
    def getProcessedData(self):
        """ Return pre-processed survey data """
        return self.rawx
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, item):
        imgs, labels = self.x[item], self.y[item]
        
        return imgs, labels