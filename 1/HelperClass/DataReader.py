# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

""" 
Version 1.1
what's new: 
- add Normalization for X and Y
"""

import numpy as np
from pathlib import Path


class DataReader(object):
    def __init__(self, data_file):
        self.train_file_name = data_file
        self.num_train = 0
        self.XTrain = None  # 归一化后的x，如果没有归一化则与XRaw相同
        self.YTrain = None  # 归一化后的y，如果没有归一化则与YRaw相同
        self.XRaw = None    # raw x
        self.YRaw = None    # raw y

    # 从文件内加载数据
    def ReadData(self):
        train_file = Path(self.train_file_name)
        if train_file.exists():
            data = np.loadtxt(self.train_file_name,delimiter=",",skiprows=1)#np.load(self.train_file_name)
            self.num_train = data[:,0].shape[0]
            lie1 = np.array(data[:,0]).reshape(self.num_train, 1)#取第一列
            lie2 = np.array(data[:,1]).reshape(self.num_train, 1)
            self.XRaw = np.hstack((lie1, lie2)) # 出来的为拼接成列数=2的数据
            self.YRaw = np.array(data[:,2]).reshape(self.num_train, 1)
            self.XTrain = self.XRaw
            self.YTrain = self.YRaw
        else:
            raise Exception("Cannot find train file!!!")
        #end if

    # normalize data by extracting range from source data
    # return: X_new: normalized data with same shape
    # return: X_norm: N x 2
    #               [[min1, range1]
    #                [min2, range2]
    #                [min3, range3]]
    def NormalizeX(self):
        X_new = np.zeros(self.XRaw.shape)
        num_feature = self.XRaw.shape[1]
        self.X_norm = np.zeros((num_feature,2))
        # 按列归一化,即所有样本的同一特征值分别做归一化
        for i in range(num_feature):
            # get one feature from all examples
            col_i = self.XRaw[:,i]
            max_value = np.max(col_i)
            min_value = np.min(col_i)
            # min value
            self.X_norm[i,0] = min_value 
            # range value
            self.X_norm[i,1] = max_value - min_value 
            new_col = (col_i - self.X_norm[i,0])/(self.X_norm[i,1])
            # Min-Max 标准化，将数据映射到[0,1]区间
            X_new[:,i] = new_col
        #end for
        self.XTrain = X_new

    # 在预测时，把预测数据也做相同方式的标准化
    def NormalizePredicateData(self, X_raw):
        X_new = np.zeros(X_raw.shape)
        n = X_raw.shape[1]
        for i in range(n):
            col_i = X_raw[:,i]
            X_new[:,i] = (col_i - self.X_norm[i,0]) / self.X_norm[i,1]
        return X_new

    # 对Y标签值归一，防止loss爆炸
    def NormalizeY(self):
        self.Y_norm = np.zeros((1,2))
        max_value = np.max(self.YRaw)
        min_value = np.min(self.YRaw)
        # min value
        self.Y_norm[0, 0] = min_value 
        # range value
        self.Y_norm[0, 1] = max_value - min_value 
        y_new = (self.YRaw - min_value) / self.Y_norm[0, 1]
        self.YTrain = y_new

    # get batch training data
    def GetSingleTrainSample(self, iteration):
        x = self.XTrain[iteration]
        y = self.YTrain[iteration]
        return x, y

    # 返回训练用数据
    def GetBatchTrainSamples(self, batch_size, iteration):
        start = iteration * batch_size
        end = start + batch_size
        batch_X = self.XTrain[start:end,:]
        batch_Y = self.YTrain[start:end,:]
        return batch_X, batch_Y

    def GetWholeTrainSamples(self):
        return self.XTrain, self.YTrain

    # permutation only affect along the first axis, so we need transpose the array first
    # see the comment of this class to understand the data format
    def Shuffle(self):
        seed = np.random.randint(0,100)
        np.random.seed(seed)
        XP = np.random.permutation(self.XTrain)
        np.random.seed(seed)
        YP = np.random.permutation(self.YTrain)
        self.XTrain = XP
        self.YTrain = YP
