#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms
import torch
import copy


def get_user_typep(x, setting_array=[7, 1, 1, 1]):
    a = 10 * setting_array[0]
    b = a + 10 * setting_array[1]
    c = b + 10 * setting_array[2]

    ############XXXX################
    if 0 <= x < a:  # 0 ~ 70
        return 1
    elif a <= x < b:  # 70 ~ 80
        return 2
    elif b <= x < c:  # 80 ~ 90
        return 3
    elif c <= x < 100:  # 90 ~ 100
        return 4
    else:
        return -1

# torch argsort: 0 being smallest, len(arr) -> largest
# torch where (condition, x(true), y(else))
# 0 ~ 39200 ~ 78400 ~ 117600 ~ 156800
#    4      3       2        1
def get_local_wmasks(ranks):
    local_masks = []
    mask = copy.deepcopy(ranks) * 0 + 1
    local_masks.append(mask.view(200, 784))
    mask0 = copy.deepcopy(ranks) * 0
    mask1 = copy.deepcopy(ranks) * 0 + 1
    # p2
    x = copy.deepcopy(ranks)
    mask = torch.where(torch.logical_and(x >= 78400, x < 117600), mask0, mask1)
    local_masks.append(mask.view(200, 784))
    # p3
    x = copy.deepcopy(ranks)
    mask = torch.where(torch.logical_and(x >= 39200, x < 78400), mask0, mask1)
    local_masks.append(mask.view(200, 784))
    # p4
    x = copy.deepcopy(ranks)
    mask = torch.where(x < 39200, mask0, mask1)
    local_masks.append(mask.view(200, 784))

    # p51
    x = copy.deepcopy(ranks)
    mask = torch.where(torch.logical_and(x >= 39200, x < 117600), mask0, mask1)
    local_masks.append(mask.view(200, 784))

    # p52
    x = copy.deepcopy(ranks)
    mask = torch.where(torch.logical_and(x >= 78400, x < 117600), mask0, mask1)
    mask = torch.where(torch.logical_and(x >= 0, x < 39200), mask0, mask)
    local_masks.append(mask.view(200, 784))

    # p53
    x = copy.deepcopy(ranks)
    mask = torch.where(x < 78400, mask0, mask1)
    local_masks.append(mask.view(200, 784))

    return local_masks


def get_local_bmasks(ranks):
    # [0,50][50,100][100,150][150,200]
    local_masks = []
    mask = copy.deepcopy(ranks) * 0 + 1
    local_masks.append(mask)
    mask0 = copy.deepcopy(ranks) * 0
    mask1 = copy.deepcopy(ranks) * 0 + 1
    # p2
    x = copy.deepcopy(ranks)
    mask = torch.where(torch.logical_and(x >= 100, x < 150), mask0, mask1)
    local_masks.append(mask)
    # p3
    x = copy.deepcopy(ranks)
    mask = torch.where(torch.logical_and(x >= 50, x < 100), mask0, mask1)
    local_masks.append(mask)
    # p4
    x = copy.deepcopy(ranks)
    mask = torch.where(x < 50, mask0, mask1)
    local_masks.append(mask)

    # p51
    x = copy.deepcopy(ranks)
    mask = torch.where(torch.logical_and(x >= 50, x < 150), mask0, mask1)
    local_masks.append(mask)
    # p52
    x = copy.deepcopy(ranks)
    mask = torch.where(torch.logical_and(x >= 0, x < 50), mask0, mask1)
    mask = torch.where(torch.logical_and(x >= 100, x < 150), mask0, mask)
    local_masks.append(mask)
    # p53

    x = copy.deepcopy(ranks)
    mask = torch.where(x >= 100, mask0, mask1)
    local_masks.append(mask)

    return local_masks


def get_mat(p_array, idx):
    x = np.ones(200)
    if idx == 1:
        return x
    for i in range(len(p_array)):
        if p_array[i] == idx:
            x[i] = 0
    return x


def get_matrxs(p_array):
    x = []
    for i in range(1, 5):
        x.append(get_mat(p_array, i))
    return x


def get_onehot_matrixs(rank):
    p_array = get_Pmat(rank)  # P[1,2,3,4] = [0,55,105,155]
    x = get_matrxs(p_array)  # [[1,1,1,1],[1,0,1,1],[1,1,0,1],[1,1,1,0]]
    return x


# Problem?
def get_Pmat(rank):
    def get_P(x):
        if x >= 0 and x < 50:
            return 1
        elif x > 49 and x < 100:
            return 2
        elif x > 99 and x < 150:
            return 3
        elif x > 149 and x < 200:
            return 4
        else:
            return -1

    x = np.zeros(200)
    for i in range(len(rank)):
        x[i] = get_P(rank[i])
    return x


def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    return dict_users


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


if __name__ == '__main__':
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)
