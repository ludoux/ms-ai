#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from numpy import array
import os
from HelperClass.NeuralNet import *
from mpl_toolkits.mplot3d import Axes3D

file_name = os.getcwd()+'/mlm.csv'


def showResult(reader, neural):
    X, Y = reader.GetWholeTrainSamples()
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter3D(X[:, 0], X[:, 1], Y[:, 0])
    x_dr = np.linspace(0,1,50)
    y_dr = np.linspace(0,1, 50)
    x_dr, y_dr = np.meshgrid(x_dr, y_dr)
    R = np.hstack((x_dr.ravel().reshape(2500, 1), y_dr.ravel().reshape(2500, 1)))
    z_dr = neural.Forward(R)
    z_dr = z_dr.reshape(-50, 50)
    ax.plot_surface(x_dr, y_dr, z_dr, cmap="rainbow")
    plt.show()

if __name__ == '__main__':
    
    reader = DataReader(file_name)
    reader.ReadData()#读入数据
    reader.NormalizeX()#归一化X
    reader.NormalizeY()#归一化Y
    # 具体的神经网络
    hp = HyperParameters(2, 1, eta=0.01, max_epoch=100, batch_size=10, eps=1e-5)
    net = NeuralNet(hp)
    net.train(reader, checkpoint=0.1)
    print("W = ", net.W)
    print("B = ", net.B)
    showResult(reader,net)
    