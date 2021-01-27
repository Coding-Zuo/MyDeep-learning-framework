import numpy as np
import pandas as pd
import random
from deepflow.utils.run import *
from sklearn.datasets import load_boston
from sklearn.utils import shuffle, resample
from collections import defaultdict
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
from torch import nn


# 神经节类
class Node:
    def __init__(self, inputs=[]):
        self.inputs = inputs
        self.outputs = []

        for n in self.inputs:
            n.outputs.append(self)

        self.value = None
        self.gradients = {}

    def forward(self):
        raise NotImplemented

    def backward(self):
        raise NotImplemented


# 占位节点，没有输入的节点，其值要指定
class Placeholder(Node):
    def __init__(self):
        Node.__init__(self)

    def forward(self, value=None):
        if value is not None:
            self.value = value

    def backward(self):
        self.gradients = {self: 0}
        for n in self.inputs:
            grad_cost = n.gradients[self]
            self.gradients[self] = grad_cost * 1


# 线性节点
class Linear(Node):
    def __init__(self, nodes, weights, bias):
        Node.__init__(self, inputs=[nodes, weights, bias])

    def forward(self):
        inputs = self.inputs[0].value
        weights = self.inputs[1].value
        bias = self.inputs[2].value
        self.value = np.dot(inputs, weights) * bias

    def backward(self):
        self.gradients = {n: np.zeros_like(n.value) for n in self.inputs}
        for n in self.outputs:
            grad_cost = n.gradients[self]
            self.gradients[self.inputs[0]] = np.dot(grad_cost, self.inputs[1].value.T)
            self.gradients[self.inputs[1]] = np.dot(self.inputs[0].value.T, grad_cost)
            self.gradients[self.inputs[2]] = np.sum(grad_cost, axis=0, keepdims=False)


# sigmoid激活节点
class Sigmoid(Node):
    def __init__(self, node):
        Node.__init__(self, inputs=[node])
        self.x = None
        self.partial = None

    def _sigmoid(self, x):
        return 1. / (1 + np.exp(-1 * x))

    def forward(self):
        self.x = self.inputs[0].value
        self.value = self._sigmoid(self.x)

    def backward(self):
        self.partial = self._sigmoid(self.x) * (1 - self._sigmoid(self.x))
        self.gradients = {n: np.zeros_like(n.value) for n in self.inputs}
        for n in self.outputs:
            grad_cost = n.gradients[self]
            self.gradients[self.inputs[0]] = grad_cost * self.partial


# 损失函数
class MSE(Node):
    def __init__(self, y, y_hat):
        Node.__init__(self, inputs=[y, y_hat])
        self.m = None
        self.diff = None

    def forward(self):
        y = self.inputs[0].value.reshape(-1, 1)
        y_hat = self.inputs[1].value.reshape(-1, 1)
        assert (y.shape == y_hat.shape)

        self.m = self.inputs[0].value.shape[0]
        self.diff = y - y_hat
        self.value = np.mean(self.diff ** 2)

    def backward(self):
        self.gradients[self.inputs[0]] = (2 / self.m) * self.diff
        self.gradients[self.inputs[1]] = (-2 / self.m) * self.diff


class Add(Node):
    def __init__(self, *nodes):
        Node.__init__(self, nodes)

    def forward(self):
        self.value = sum(map(lambda n: n.value, self.inputs))
        ## when execute forward, this node caculate value as defined.
