#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from numpy import array
import os
from HelperClass.NeuralNet import *
from HelperClass.DataReader import *

file_name = os.getcwd()+'/iris.csv'


if __name__ == '__main__':
    reader = DataReader(file_name)
    reader.ReadData()
    # print(reader.XTrainRaw)
    # print(reader.YTrainRaw)
    reader.NormalizeY(NetType.MultipleClassifier, base=1)
    reader.NormalizeX()
    reader.GenerateValidationSet()
    # print(reader.XTrain)
    # print(reader.YTrain)

    n_input = reader.num_feature
    n_hidden = 4#四个输入类型（四个隐藏神经元）
    n_output = 3#三输出
    eta, batch_size, max_epoch = 0.1, 5, 10000
    eps = 0.01#学习步长

    params = HyperParameters(n_input, n_hidden, n_output,
                             eta, max_epoch, batch_size, eps,
                             NetType.MultipleClassifier, InitialMethod.Xavier)

    net = NeuralNet(params, "Non-linear Classifier of Iris")
    net.train(reader, 10)
    net.ShowTrainingHistory()
    print("===输出===\nwb1.W = ", net.wb1.W,"\nwb1.B = ", net.wb1.B,"\nwb2.W = ", net.wb2.W,"\nwb2.B = ", net.wb2.B)