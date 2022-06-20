# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 19:54:19 2022

@author: Joao Morais

Script inputs:
    - csv_path: absolute path to csv
    
    - split_ratio: percentage (in decimal) of dataset split in training, 
      validation and test sets. 
          - If split_ratio[0] is 0, no training is executed.
          - If split_ratio[2] is 0, no test is executed.
          - If split_ratio = [0, 0, 1], we assume the unlabelled challenge test 
            is used, thus labels are ignored.
      
    - trained_model_path: absolute or relative path to the pickled 
      trained model.
      
Example parameters for prototyping:
    - csv_path = train.csv
    - split_ratio = [0.7, 0.2, 0.1]
    - trained_model_path = trained_model.pkl
    
Example parameters for competition challenge:
    Run 1 - training:
        - csv_path = train.csv
        - split_ratio = [0.7, 0.3, 0]
        - trained_model_path = trained_model_test.pkl
    Run 2 - testing:
        - csv_path = test.csv
        - split_ratio = [0, 0, 1]
        - trained_model_path = trained_model_test.pkl
"""

import os
import sys
import utm
import pandas as pd
import numpy as np
import pickle

from tqdm import tqdm

import position_only_baseline_func as func


import matplotlib.pyplot as plt

csv_path = r'E:\Gouranga\DeepSense_V2I\final_folder_v5\dev_data\Image_Position\ml_challenge_dev_img_pos.csv'

split_ratio = [0.7, 0.2, 0.1] # train, val, test 

trained_model_path = f'trained_model_example.pkl'
#%% Read data from CSV

N_POS = 2 # 2 coords
N_BEAMS = 64

labels_available = split_ratio != [0, 0, 1]

file_sep = '/' if '/' in csv_path else '\\'
data_folder = file_sep.join(csv_path.split(file_sep)[:-1])

# check if numpy files exist already and load them if not (makes 2nd run faster)
npy_base_name = csv_path.split(file_sep)[-1].replace('.csv', '')
pos_array_name = npy_base_name + '_input_pos.npy'
beam_label_name = npy_base_name + '_label_beam.npy'

df = pd.read_csv(csv_path)
n_samples = df.index.stop

# Get all relative paths
pos1_rel_paths = df['unit2_loc_1'].values
pos2_rel_paths = df['unit2_loc_2'].values
pos_bs_rel_paths = df['unit1_loc'].values

# Get all absolute paths
pos1_abs_paths = [os.path.join(data_folder, path[2:]) for path in pos1_rel_paths]
pos2_abs_paths = [os.path.join(data_folder, path[2:]) for path in pos2_rel_paths]
pos_bs_abs_paths = [os.path.join(data_folder, path[2:]) for path in pos_bs_rel_paths]

# no. input sequences x no. samples per seq x sample dimension (lat and lon)
pos_input = np.zeros((n_samples, 2, N_POS)) 

pos_bs = np.zeros((n_samples, N_POS))

print('Loading all positions...')
for sample_idx in tqdm(range(n_samples)):
    # unit2 (UE) positions
    pos_input[sample_idx, 0, :] = np.loadtxt(pos1_abs_paths[sample_idx])
    pos_input[sample_idx, 1, :] = np.loadtxt(pos2_abs_paths[sample_idx])
    
    # unit1 (BS) position
    pos_bs[sample_idx] = np.loadtxt(pos_bs_abs_paths[sample_idx])
    
if labels_available:
    # Read beam labels (1-64 in the csv need to be set to 0-63)
    beam_label = df['unit1_beam'].values - 1

#%% Normalize positions (min-max XY position difference between UE and BS)

def xy_from_latlong(lat_long):
    """ 
    Requires lat and long, in decimal degrees, in the 1st and 2nd columns. 
    Returns same row vec/matrix on cartesian (XY) coords.
    """
    # utm.from_latlon() returns: (EASTING, NORTHING, ZONE_NUMBER, ZONE_LETTER)
    x, y, *_ = utm.from_latlon(lat_long[:,0], lat_long[:,1])
    return np.stack((x,y), axis=1)

pos_ue_stacked = np.vstack((pos_input[:, 0, :], pos_input[:, 1, :]))
pos_bs_stacked = np.vstack((pos_bs, pos_bs))

pos_ue_cart = xy_from_latlong(pos_ue_stacked)
pos_bs_cart = xy_from_latlong(pos_bs_stacked)

pos_diff = pos_ue_cart - pos_bs_cart

pos_min = np.min(pos_diff, axis=0)
pos_max = np.max(pos_diff, axis=0)

# Normalize and unstack
pos_stacked_normalized = (pos_diff - pos_min) / (pos_max - pos_min)
pos_input_normalized = np.zeros((n_samples, 2, 2))
pos_input_normalized[:, 0, :] = pos_stacked_normalized[:n_samples]
pos_input_normalized[:, 1, :] = pos_stacked_normalized[n_samples:]

#%% Data split

train_samples, val_samples, test_samples = func.data_split(split_ratio, n_samples)

x_train = pos_input_normalized[train_samples]
x_val= pos_input_normalized[val_samples]
x_test = pos_input_normalized[test_samples]

if labels_available:
    y_train = beam_label[train_samples]
    y_val = beam_label[val_samples]
    y_test = beam_label[test_samples]
    
    y_train = np.expand_dims(y_train, axis=1)
    y_val = np.expand_dims(y_val, axis=1)
    y_test = np.expand_dims(y_test, axis=1)
    

#%% Train model
if split_ratio[0] != 0: # if training set exists
    N = x_train.shape[-1] # size of each single input
    x_size = x_train.shape[1]
    
    # Instantiate the model (Note: beam labels should be between 0 and N_BEAMS)
    net = func.GruModelGeneral(in_features=N, num_classes=N_BEAMS, 
                               num_layers=1, hidden_size=64, embed_size=N_BEAMS, 
                               dropout=0.8)
    
    net.print_net_summary(x_size)
    
    val_loss, val_acc, predictions, trained_model, trained_model_path = \
        func.train_model(x_train, y_train, x_val, y_val, net, num_epoch=30)
    
    # Save trained model
    with open(trained_model_path, 'wb') as fp:
        pickle.dump(trained_model, fp)

#%% Test Model
if split_ratio[2] != 0: # if test set exists
    with open(trained_model_path, 'rb') as fp:
        trained_model = pickle.load(fp)
        
    y_pred = func.test_net(x_test, trained_model)
    
    func.save_pred_to_csv(y_pred, top_k=[1,2,3], target_csv='beam_pred.csv')
    
    if labels_available:
    
        acc = func.compute_acc(y_pred, y_test + 1, top_k=[1,3,5])
        print(f'Test accuracies: {acc}')
        
        score = func.compute_DBA_score(y_pred, y_test+1, max_k=3, delta=5)
        print(f'Competition score: {score:.6f}')
        #%%
        bin_edges = np.arange(N_BEAMS) + 1
        plt.figure()
        plt.hist(y_test+1, bins=bin_edges, label='y_true', alpha=.7)
        plt.hist(y_pred[:,0], bins=bin_edges, label='y_pred', alpha=.4)
        plt.legend()
        plt.title("Beam distribution")
