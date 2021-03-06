from deepflow.nn.core import *
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


# sigmoid函数
def sigmoid(x):
    return 1. / (1 + np.exp(-1 * x))


# 加载数据
def load_data():
    data = load_boston()
    X_ = data['data']
    Y_ = data['target']
    # 将数据归一化
    X_ = (X_ - np.mean(X_, axis=0)) / np.std(X_, axis=0)
    x_train, x_test, y_train, y_test = train_test_split(X_, Y_, test_size=0.2, random_state=666)
    return x_train, x_test, y_train, y_test


# 将节点数据转换成图
def convert_feed_dict_to_graph(feed_dict):
    computing_graph = defaultdict(list)
    nodes = [n for n in feed_dict]
    while nodes:
        n = nodes.pop()
        if isinstance(n, Placeholder):
            n.value = feed_dict[n]

        if n in computing_graph:
            continue

        for m in n.outputs:
            computing_graph[n].append(m)
            nodes.append(m)
    return computing_graph


# 将图进行拓扑排序，生成计算图
def toplogic(graph):
    sorted_nodes = []
    while len(graph) > 0:
        all_inputs = []
        all_outputs = []

        for n in graph:
            all_inputs += graph[n]
            all_outputs.append(n)

        all_inputs = set(all_inputs)
        all_outputs = set(all_outputs)

        need_remove = all_outputs - all_inputs

        if len(need_remove) > 0:
            node = random.choice(list(need_remove))
            need_to_visited = [node]

            if len(graph) == 1:
                need_to_visited += graph[node]

            graph.pop(node)
            sorted_nodes += need_to_visited

            for _, links in graph.items():
                if node in links:
                    links.remove(node)
        else:
            break
    return sorted_nodes


# 生成计算图
def toplogical_sort_feed_dict(feed_dict):
    graph = convert_feed_dict_to_graph(feed_dict)
    return toplogic(graph)


# 前向传播
def forward(graph):
    for n in graph:
        n.forward()


# 后向传播
def backward(graph):
    for n in graph:
        n.backward()


def forward_and_backward(graph):
    for n in graph:
        n.forward()

    for n in graph[::-1]:
        n.backward()


# 更新参数
def optimizer(trainables, learning_rate=1e-2):
    for t in trainables:
        t.value += -1 * learning_rate * t.gradients[t]


def MyFramePredict(w1_, b1_, w2_, b2_, losses, X_test, y_test):
    y1 = np.dot(X_test, w1_.value) + b1_.value
    s = sigmoid(y1)
    y2 = np.dot(s, w2_.value) + b2_.value

    # 用误差平方和评价
    sse = ((y2 - y_test) ** 2).sum()
    print('框架评分:{}'.format(sse))

    # 画图
    plt.figure()
    plt.plot(losses)
    plt.title("cost of model")
    plt.savefig('../output/FrameCost.png')
    plt.close()

    plt.figure()
    y2 = y2.flatten()
    delta = y2 - y_test
    plt.plot(delta, color='green')
    plt.savefig('../output/FrameResult.png')
    plt.close()


# pytorch预测
def PytorchPredict(model, losses, x_test, y_test):
    # 准备数据
    x_test = x_test.astype(np.float32)
    y_test = y_test.astype(np.float32)
    x_tensor = torch.from_numpy(x_test)
    y_tensor = torch.from_numpy(y_test)

    # 用模型进行预测
    y_pred = model(x_tensor).detach().numpy()
    sse = ((y_pred - y_test) ** 2).sum()
    print('pytorch评分:{}'.format(sse))

    # 画图
    plt.figure()
    plt.plot(losses)
    plt.title("cost of pytorch")
    plt.savefig("../output/PytorchCost.png")
    plt.close()

    plt.figure()
    y_pred = y_pred.flatten()
    delta = y_pred - y_test
    # print(y2.shape, y_test.shape, delta.shape, x.shape)
    plt.plot(delta, color="green")
    # plt.scatter(x, y_test, color = "red")
    plt.savefig("../output/PytorchResult.png")
    plt.close()


@timethis
def testMyFrame(X_, Y_):
    # 初始化参数
    n_features = X_.shape[1]
    n_hidden = 10
    W1_ = np.random.randn(n_features, n_hidden)
    b1_ = np.zeros(n_hidden)
    W2_ = np.random.randn(n_hidden, 1)
    b2_ = np.zeros(1)

    # 定义神经节点
    X, y = Placeholder(), Placeholder()
    W1, b1 = Placeholder(), Placeholder()
    W2, b2 = Placeholder(), Placeholder()

    # 定义模型
    l1 = Linear(X, W1, b1)
    s1 = Sigmoid(l1)
    l2 = Linear(s1, W2, b2)
    cost = MSE(y, l2)

    # 定义初始值
    feed_dict = {
        X: X_,
        y: Y_,
        W1: W1_,
        b1: b1_,
        W2: W2_,
        b2: b2_
    }

    # 定义超参数
    epochs = 200
    m = X_.shape[0]
    batch_size = 16
    steps_per_epoch = m // batch_size

    # 生成计算图
    graph = toplogical_sort_feed_dict(feed_dict)
    trainables = [W1, b1, W2, b2]

    print("样本总数{}".format(m))

    # 训练过程
    losses = []
    for i in range(epochs):
        loss = 0
        for j in range(steps_per_epoch):
            # 步骤1,对样本随机采样
            X_batch, y_batch = resample(X_, Y_, n_samples=batch_size)
            # 重置X,Y的输入值
            X.value = X_batch
            y.value = y_batch
            # 步骤2 前向和后向传播
            # _ = None
            forward_and_backward(graph)
            # 步骤3 更新参数
            rate = 1e-2
            optimizer(trainables, rate)
            loss += graph[-1].value

        # 输出
        if i % 100 == 0:
            # print("Epoch: {}, Loss: {:.3f}".format(i+1, loss/steps_per_epoch))
            losses.append(loss / steps_per_epoch)

    return W1, b1, W2, b2, losses


# 测试pytorch
@timethis
def testPytorch(x_train, y_train):
    n_features = x_train.shape[1]
    n_hidden = 10
    # 定义网络结构
    net = nn.Sequential(
        nn.Linear(n_features, n_hidden),
        nn.Sigmoid(),
        nn.Linear(n_hidden, 1)
    )

    # 定义损失函数和优化器
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=1e-2)

    # 定义训练参数
    epochs = 200
    m = n_features
    batch_size = 16
    steps_per_epoch = m // batch_size

    # 定义数据加载器
    x_train = x_train.astype(np.float32)
    y_train = y_train.astype(np.float32)
    x_tensor = torch.from_numpy(x_train)
    y_tensor = torch.from_numpy(y_train)
    train = torch.utils.data.TensorDataset(x_tensor, y_tensor)
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False)

    # 训练
    losses = []
    for i in range(epochs):
        for x, y in train_loader:
            y = y.view(-1, 1)
            optimizer.zero_grad()
            outputs = net(x)
            loss = loss_fn(outputs, y)
            loss.backward()
            optimizer.step()
        # 输出
        if i % 100 == 0:
            # print("Epoch: {}, Loss: {:.3f}".format(i+1, loss.data))
            losses.append(loss.data)
    return net, losses


if __name__ == '__main__':
    x_train, x_test, y_train, y_test = load_data()

    x_train_, x_test_, y_train_, y_test_ = x_train.copy(), x_test.copy(), y_train.copy(), y_test.copy()
    params = testMyFrame(x_train_, y_train_)
    MyFramePredict(params[0], params[1], params[2], params[3], params[4], x_test_, y_test_)

    x_train_, x_test_, y_train_, y_test_ = x_train.copy(), x_test.copy(), y_train.copy(), y_test.copy()
    model, losses = testPytorch(x_train_, y_train_)
    PytorchPredict(model, losses, x_test_, y_test_)
