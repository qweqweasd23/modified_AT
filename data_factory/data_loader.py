import torch
import os
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import collections
import numbers
import math
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

class PSMSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train", L=0.5):
        self.mode = mode
        self.step = step
        self.L = L
        self.win_size = win_size

        data = pd.read_csv(data_path + '/train.csv')
        data = np.nan_to_num(data.values[:, 1:])
        
        test_data = pd.read_csv(data_path + '/test.csv')
        self.test = np.nan_to_num(test_data.values[:, 1:])

        self.train = data
        data_len = len(self.train)
        self.train = self.train[:int(data_len * 0.8)]
        self.val = self.train[int(data_len * 0.8):]


        self.test_labels = pd.read_csv(data_path + '/test_label.csv').values[:, 1:]

        if self.mode == "train":
            self.data = self.train
            self.labels = np.zeros((len(self.train), 1), dtype=np.float32)  # train에는 라벨 없음
        elif self.mode == "val":
            self.data = self.val
            self.labels = np.zeros((len(self.val), 1), dtype=np.float32)
        elif self.mode == "test":
            self.data = self.test
            self.labels = self.test_labels

        # sanity check: label 길이 일치 보장
        if len(self.labels) != len(self.data):
            raise ValueError("Data and label length mismatch.")

    def __len__(self):
        return max(0, (len(self.data) - self.win_size) // self.step + 1)

    def __getitem__(self, index):
        index = index * self.step
        start = index
        end = index + self.win_size

        if end > len(self.data):
            raise IndexError("Index exceeds dataset length after sliding window.")

        seq = self.data[start:end]
        label = self.labels[start:end]

        if label.ndim == 1:
            label = label[:, np.newaxis]

        assert seq.shape[0] == self.win_size, f"Invalid seq shape: {seq.shape}"
        assert label.shape[0] == self.win_size, f"Invalid label shape: {label.shape}"

        return np.float32(seq), np.float32(label)



class MSLSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train", L=0.5):
        self.mode = mode
        self.step = step
        self.L = L
        self.win_size = win_size
        data = np.load(data_path + "/MSL_train.npy")
        self.test = np.load(data_path + "/MSL_test.npy")
        self.train = data
        data_len = len(self.train)
        self.train = self.train[:int(data_len * 0.8)]
        self.val = self.train[int(data_len * 0.8):]
        self.test_labels = np.load(data_path + "/MSL_test_label.npy")
                
        if self.mode == "train":
            self.data = self.train
            self.labels = np.zeros((len(self.train), 1), dtype=np.float32)
        elif self.mode == "val":
            self.data = self.val
            self.labels = np.zeros((len(self.val), 1), dtype=np.float32)
        elif self.mode == "test":
            self.data = self.test
            self.labels = self.test_labels

    def __len__(self):
        return (len(self.data) - self.win_size) // self.step + 1

    def __getitem__(self, index):
        index = index * self.step
        start = index
        end = index + self.win_size

        if end > len(self.data):
            raise IndexError("Index exceeds dataset length after sliding window.")
        seq = self.data[start:end]
        label = self.labels[start:end]
        if label.ndim == 1:
            label = label[:, np.newaxis]
        return np.float32(seq), np.float32(label)


class SMAPSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train", L=0.5):
        self.mode = mode
        self.step = step
        self.L = L
        self.win_size = win_size
        

        self.train = np.load(data_path + "/SMAP_train.npy")
        self.test = np.load(data_path + "/SMAP_test.npy")
        test_labels = np.load(data_path + "/SMAP_test_label.npy")

        self.test_labels = test_labels

        
        data_len = len(self.train)
        self.train = self.train[:int(data_len * 0.8)]
        self.val = self.train[int(data_len * 0.8):]

        
        if self.mode == "train":
            self.data = self.train
            self.labels = np.zeros((len(self.train), 1), dtype=np.float32)
        elif self.mode == "val":
            self.data = self.val
            self.labels = np.zeros((len(self.val), 1), dtype=np.float32)
        elif self.mode == "test":
            self.data = self.test
            self.labels = self.test_labels

        print(f"{self.mode} data shape:", self.data.shape)

    def __len__(self):
        return (len(self.data) - self.win_size) // self.step + 1

    def __getitem__(self, index):
        index = index * self.step
        start = index
        end = index + self.win_size

        if end > len(self.data):
            raise IndexError("Index exceeds dataset length after sliding window.")
        seq = self.data[start:end]
        label = self.labels[start:end]
        if label.ndim == 1:
            label = label[:, np.newaxis]
        return np.float32(seq), np.float32(label)
    
class SMDSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train", L=0.5):
        self.mode = mode
        self.step = step
        self.L = L
        self.win_size = win_size

        self.train = np.load(data_path + "/SMD_train.npy")
        self.test = np.load(data_path + "/SMD_test.npy")
        self.test_labels = np.load(data_path + "/SMD_test_label.npy")

        data_len = len(self.train)
        self.train = self.train[:int(data_len * 0.8)]
        self.val = self.train[int(data_len * 0.8):]

        
        if self.mode == "train":
            self.data = self.train
            self.labels = np.zeros((len(self.train), 1), dtype=np.float32) 
        elif self.mode == "val":
            self.data = self.val
            self.labels = np.zeros((len(self.val), 1), dtype=np.float32)  
        elif self.mode == "test":
            self.data = self.test
            self.labels = self.test_labels

    def __len__(self):
        return (len(self.data) - self.win_size) // self.step + 1

    def __getitem__(self, index):
        index = index * self.step
        start = index
        end = index + self.win_size

        if end > len(self.data):
            raise IndexError("Index exceeds dataset length after sliding window.")
        seq = self.data[start:end]
        label = self.labels[start:end]
        if label.ndim == 1:
            label = label[:, np.newaxis]
        return np.float32(seq), np.float32(label)


   



def get_loader_segment(data_path, batch_size, win_size=100, step=100, mode='train', dataset='KDD', L=0.5):
    if (dataset == 'SMD'):
        dataset = SMDSegLoader(data_path, win_size, step, mode, L)
    elif (dataset == 'MSL'):
        dataset = MSLSegLoader(data_path, win_size, 1, mode, L)
    elif (dataset == 'SMAP'):
        dataset = SMAPSegLoader(data_path, win_size, 1, mode, L)
    elif (dataset == 'PSM'):
        dataset = PSMSegLoader(data_path, win_size, 1, mode, L)

    shuffle = False
    if mode == 'train':
        shuffle = True

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=0)
    return data_loader
