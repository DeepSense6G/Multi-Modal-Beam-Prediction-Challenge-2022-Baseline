# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 20:09:40 2022

@author: Joao Morais
"""

import os
import sys
import time
import copy
import numpy as np
import pandas as pd

import torch as t
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import datetime

from collections import OrderedDict
from pytorch_model_summary import summary

def compute_acc(y_pred, y_true, top_k=[1,3,5]):
    """ Computes top-k accuracy given prediction and ground truth labels."""
    n_top_k = len(top_k)
    total_hits = np.zeros(n_top_k)
    
    n_test_samples = len(y_true)
    if len(y_pred) != n_test_samples:
        raise Exception('Number of predicted beams does not match number of labels.')
    
    # For each test sample, count times where true beam is in k top guesses
    for samp_idx in range(len(y_true)):
        for k_idx in range(n_top_k):
            hit = np.any(y_pred[samp_idx,:top_k[k_idx]] == y_true[samp_idx, -1])
            total_hits[k_idx] += 1 if hit else 0
    
    # Average the number of correct guesses (over the total samples)
    return np.round(total_hits / len(y_true), 4)


def save_pred_to_csv(y_pred, top_k=[1,2,3], target_csv='beam_pred.csv'):
    """ 
    Saves the predicted beam results to a csv file. 
    Expects y_pred: n_samples x N_BEAMS, and saves the top_k columns only. 
    """
    
    cols = [f'top-{i} beam' for i in top_k]
    df = pd.DataFrame(data=y_pred[:, np.array(top_k)-1], columns=cols)
    df.index.name = 'index'
    df.to_csv(target_csv)
    

def compute_DBA_score(y_pred, y_true, max_k=3, delta=5):
    """ 
    The top-k MBD (Minimum Beam Distance) as the minimum distance
    of any beam in the top-k set of predicted beams to the ground truth beam. 
    
    Then we take the average across all samples.
    
    Then we average that number over all the considered Ks.
    """
    n_samples = y_pred.shape[0]
    n_beams = y_pred.shape[-1] 
    
    yk = np.zeros(max_k)
    for k in range(max_k):
        acc_avg_min_beam_dist = 0
        idxs_up_to_k = np.arange(k+1)
        for i in range(n_samples):
            aux1 = np.abs(y_pred[i, idxs_up_to_k] - y_true[i]) / delta
            # Compute min between beam diff and 1
            aux2 = np.min(np.stack((aux1, np.zeros_like(aux1)+1), axis=0), axis=0)
            acc_avg_min_beam_dist += np.min(aux2)
            
        yk[k] = 1 - acc_avg_min_beam_dist / n_samples
    
    return np.mean(yk)


def compute_avg_beam_dist(y_pred, y_true):
    return np.mean(np.abs(y_pred[:, 0] - y_true))


def data_split(data_split, n_samples, shuffle=False):
    """
    Takes in a ratio and outputs len(data_split) lists of random numbers 
    between 1 and n_samples, and the quantity of number in each list follows the
    data_split proportion. 
    
    data_split may be have 2 values (train vs val/test) or 3 values (train, val & test)
    """
    
    if not (-1e-9 < sum(data_split) - 1 < 1e-9):
        raise Exception('Impossible split. Sum of data_split must be 1.')
    
    if type(shuffle) == int:
        np.random.seed(shuffle)
        
    if shuffle:
        sample_list = np.random.permutation(n_samples)
    else:
        sample_list = np.arange(n_samples)
        
    last_training_sample = int(data_split[0] * n_samples)
    train_samples = sample_list[:last_training_sample]
        
    if len(data_split) == 2:
        val_samples = sample_list[last_training_sample:]
        return train_samples, val_samples, None
    
    if len(data_split) == 3:
        first_test_sample = int((1-data_split[2]) * n_samples)
        val_samples = sample_list[last_training_sample:first_test_sample]
        test_samples = sample_list[first_test_sample:]
        return train_samples, val_samples, test_samples
    else:
        raise Exception('data split needs to be a list with 2 or 3 elements.')
    


############################# NN-related FUNCTIONS ############################

class DataFeed(Dataset):
    """ Creates a torch datafeed from inputs and labels."""
    def __init__(self, x, y):
        
        self.inp_vals = x
        self.pred_vals = y
    
    def __len__(self):
        return len(self.inp_vals)
    
    def __getitem__(self, idx):

        inp_val = t.tensor(self.inp_vals[idx],  requires_grad=False)
        out_val = t.tensor(self.pred_vals[idx], requires_grad=False)
        
        return inp_val, out_val.long()


class GruModelGeneral(nn.Module):
    """ 
    Defines the GRU model for classification.
    """
    def __init__(self, in_features, num_classes, num_layers=1, hidden_size=64,
                 embed_size=64, dropout=0.8):
        super(GruModelGeneral, self).__init__()
        
        self.in_features = in_features
        if self.in_features == 2:
            self.embed = t.nn.Linear(in_features, embed_size)
        else:
            self.embed = t.nn.Embedding(num_embeddings=in_features,
                                        embedding_dim=embed_size)     
        
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.gru = t.nn.GRU(input_size=embed_size, hidden_size=hidden_size, 
                            num_layers=num_layers, dropout=dropout)
        self.fc = t.nn.Linear(hidden_size, num_classes)
        self.name = 'GruModelGeneral'
        self.dropout1 = nn.Dropout(0.5)

    def initHidden(self, batch_size):
        return t.zeros((self.num_layers, batch_size, self.hidden_size))

    def forward(self, x, h):
        y = self.embed(x)
        y = self.dropout1(y)
        y, h = self.gru(y, h)
        y = self.fc(y)
        return y, h

    def print_net_summary(self, seq_len=1):
        h = self.initHidden(1)
        if self.in_features == 2:
            # training on positions
            fake_input = t.zeros((seq_len, 1, self.in_features)) 
        else:
            # training on beams
            fake_input = t.zeros((seq_len, 1)).long()
        print(summary(self, fake_input, h))
                

def train_model(x_train, y_train, x_val, y_val, net,
                num_epoch=100, num_classes=64, train_batch_size=32, 
                val_batch_size=64, x_len=8, y_len=1):
    """
    This function will be used both to predict beams from positions and from
    beams. Only the content of x_train and the net change slightly. 
    
    Num classes - number of output classes/categories = # beams
    x_len - length of the input sequence, or the learning window of beams
    y_len - length of the output sequence. We default it to 1 beam, being
            one step or 10 steps into the future.
    N_LOSS - the length of the sequence we use for back propagation. 
    """
    num_classes = net.fc.out_features
    # competition locked to 1 output.
    # N_zeros = y_len - 1
    # N_LOSS = x_len + N_zeros
    N_zeros = 1
    N_LOSS = y_len
    
    train_loader = DataLoader(DataFeed(x_train, y_train), 
                              batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(DataFeed(x_val, y_val), 
                            batch_size=val_batch_size, shuffle=False)
    
    device = t.cuda.set_device(np.random.randint(4) if t.cuda.is_available() else 'cpu')

    now = datetime.datetime.now().strftime("%H_%M_%S")
    date = datetime.date.today().strftime("%y_%m_%d")

    # path to save the model
    CHECKPOINT_FOLDER = './checkpoint/' 
    best_model_path = CHECKPOINT_FOLDER + date + '_' + now + '_' + str(time.time()) + '.pth'
    
    if not os.path.exists(CHECKPOINT_FOLDER):
        os.mkdir(CHECKPOINT_FOLDER)
    
    # send model to GPU
    net.to(device)

    # set up loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = t.optim.Adam(net.parameters(), lr=0.01)
    scheduler = t.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40])

    best_epoch = 0
    best_top1 = 0
    # train model
    for epoch in range(num_epoch):  # loop over the dataset multiple times
        net.train()
        running_loss = 0.
        running_acc = 1.
        with tqdm(train_loader, unit="batch", file=sys.stdout) as tepoch:
            for i, (x, label) in enumerate(tepoch, 0):
                
                # x     is (# batches, # inputs, size of each input)
                # label is (# batches, # inputs)
                
                # TODO: move this outside the loop
                if len(x.shape) == 2:
                    x = x.long() # beams
                    x = x[:, :, None] # add an extra dimension for beams.
                else:
                    x = x.float() # positions!
                
                tepoch.set_description(f"Epoch {epoch+1}")
                x2 = t.transpose(x, 0, 1)
                
                # Pad if backpropagating to more than one output.
                if N_zeros > 0:
                    x2 = t.cat([x2, t.zeros_like(x2[:N_zeros, :])], dim=0) 
                x2 = x2.to(device)
                
                label = t.transpose(label,0,1).to(device)
                optimizer.zero_grad()
                
                h = net.initHidden(x2.shape[1]).to(device)
                outputs, _ = net(x2, h)
                outputs2 = outputs[-N_LOSS:, :]
                
                label_loss = label[-N_LOSS:, :]
                loss = criterion(outputs2.view(-1, num_classes), label_loss.flatten())
                prediction = t.argmax(outputs2, dim=-1)
                
                # Only the last beam matters for accuracy
                label_pred = label[-y_len:,:]
                prediction = prediction[-y_len:,:]
                acc = (prediction == label_pred).sum().item() / train_batch_size
                if np.any(label_pred.cpu().numpy() == -100):
                    raise Exception()
                loss.backward()
                optimizer.step()
                # print statistics
                running_loss = (loss.item() + i * running_loss) / (i + 1)
                running_acc = (acc + i * running_acc) / (i + 1)
                log = OrderedDict()
                log['loss'] = running_loss
                log['acc'] = running_acc
                tepoch.set_postfix(log)
            scheduler.step()
            # validation
            predictions = []
            net.eval()
            with t.no_grad():
                total = np.zeros((y_len,))
                top1_correct = np.zeros((y_len,))
                top3_correct = np.zeros((y_len,))
                top5_correct = np.zeros((y_len,))
                val_loss = 0
                for (x, label) in val_loader:
                    if len(x.shape) == 2:
                        x = x.long() # beams
                    else:
                        x = x.float()  # positions!
                    
                    x = t.transpose(x, 0, 1)
                    if N_zeros > 0:
                        x = t.cat([x, t.zeros_like(x[:N_zeros, :])], dim=0)
                    x = x.to(device)
                    
                    y_labels = t.transpose(label, 0, 1)
                    y_labels = y_labels.to(device)
                    
                    # forward + backward + optimize
                    h = net.initHidden(x.shape[1]).to(device)
                    outputs, _ = net(x, h)
                    outputs2 = outputs[-N_LOSS:, :]
                    # outputs is  (# input_seq_size, batch_size, output_size)
                    # outputs2 selects only the outputs we are using for training
                    
                    label_loss = y_labels[-N_LOSS:, :]
                    val_loss += nn.CrossEntropyLoss(reduction='sum')(outputs2.view(-1, num_classes), 
                                                                     label_loss.flatten()).item()
                    
                    prediction = t.argmax(outputs2, dim=-1)

                    prediction = prediction[-y_len:,:]
                    y = t.squeeze(y_labels, axis=-1)[-y_len:, :]
                    # y_labels is (# input_seq_size, batch_size, output_size)
                    
                    total += t.sum(y != -100, dim=-1).cpu().numpy()
                    
                    top1_correct += t.sum(prediction == y, dim=-1).cpu().numpy()
                    _, idx = t.topk(outputs2, 5, dim=-1)
                    idx = idx.cpu().numpy()
                    
                    for i in range(y.shape[0]):
                        for j in range(y.shape[1]):
                            top3_correct[i] += np.isin(y[i, j], idx[-(i+1), j, :3]).sum()
                            top5_correct[i] += np.isin(y[i, j], idx[-(i+1), j, :5]).sum()
                    predictions.append(prediction.cpu().numpy())
                val_loss /= total.sum()
                val_top1_acc = top1_correct / total
                val_top3_acc = top3_correct / total
                val_top5_acc = top5_correct / total
                print('Top-1,3,5 Accuracy:', flush=True)
                curr_acc = np.stack([val_top1_acc, val_top3_acc, val_top5_acc])
                print(curr_acc, flush=True)
                
                # Compute average # of beams of error
                if val_top1_acc > best_top1:
                    best_acc = curr_acc
                    best_top1 = val_top1_acc[0]
                    best_epoch = epoch
                    t.save(net.state_dict(), best_model_path)
                
    print(f'Finished Training! Best epoch = {best_epoch};')
    print(f'Best validation accuracy:\n {best_acc}')

    val_acc = {'top1': val_top1_acc, 'top3': val_top3_acc, 'top5': val_top5_acc}
    
    
    # Load weights of best trained model
    trained_model = copy.deepcopy(net)
    trained_model.load_state_dict(t.load(best_model_path))

    return val_loss, val_acc, predictions, trained_model, best_model_path

def test_net(x_test, net):
    test_loader = DataLoader(DataFeed(x_test, x_test), batch_size=64, shuffle=False)
    N_zeros = 0
    
    device = t.cuda.set_device(1 if t.cuda.is_available() else 'cpu')
    
    net.eval()
    predictions = []
    with t.no_grad():
        for (x, _) in test_loader:
            if len(x.shape) == 2:
                x = x.long() # beams
            else:
                x = x.float()  # positions!
            
            x = t.transpose(x, 0, 1)
            if N_zeros > 0:
                x = t.cat([x, t.zeros_like(x[:N_zeros, :])], dim=0)
            x = x.to(device)

            # forward + backward + optimize
            h = net.initHidden(x.shape[1]).to(device)
            outputs, _ = net(x, h)
            outputs = outputs[-1:, :]
            prediction = t.argsort(outputs, dim=-1, descending=True) + 1

            predictions.append(prediction.cpu().numpy())

        predictions = np.squeeze(np.concatenate(predictions, 1))

        return predictions
