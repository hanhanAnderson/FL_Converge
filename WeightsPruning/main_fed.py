#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, get_user_typep, get_local_wmasks, get_local_bmasks
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img


def get_sub_paras(w_glob, wmask, bmask):
    w_l = copy.deepcopy(w_glob)
    w_l['layer_input.weight'] = w_l['layer_input.weight'] * wmask
    w_l['layer_input.bias'] = w_l['layer_input.bias'] * bmask

    return w_l


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            print("===============IID=DATA=======================")
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            print("===========NON==ID=DATA=======================")
            dict_users = mnist_noniid(dataset_train, args.num_users)
            args.epochs = 100
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)

        net_2 = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
        net_3 = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
        net_4 = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)

        net_51 = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
        net_52 = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
        net_53 = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)

    else:
        exit('Error: unrecognized model')
    print(net_glob)

    net_glob.train()
    net_2.train()
    net_3.train()
    net_4.train()
    net_51.train()
    net_52.train()
    net_53.train()
    print("***********MODIFIED: PRUNING WEIGHTS ONLY******************")
    # copy weights

    # Ranking the paras
    # w_glob['layer_input.weight'].view(-1).view(200, 784)  #156800 = 39200 * 4
    # Smallest  [0,39200], [39200,78400], [78400,117600],[117600, 156800] (largest)
    """
1 w_net      X              X                X               X
2 net_2      X              X                                X
3 net_3      X                               X               X
4 net_4                     X                X               X

5 net_51     X                                               X
6 net_52                    X                                X
7 net_53                                     X               X
    """
    #    [0,50]         [50,100]         [100,150]        [150,200]
    w_glob = net_glob.state_dict()
    starting_weights = copy.deepcopy(w_glob)

    # ABS OR NO ABS ?
    # wranks = torch.argsort(w_glob['layer_input.weight'].view(-1))
    wranks = torch.argsort(torch.absolute(w_glob['layer_input.weight'].view(-1)))
    local_w_masks = get_local_wmasks(wranks)
    # branks = torch.argsort(w_glob['layer_input.bias'])
    branks = torch.argsort(torch.absolute(w_glob['layer_input.bias']))
    local_b_masks = get_local_bmasks(branks)

    w_n2 = get_sub_paras(w_glob, local_w_masks[1], local_b_masks[1])
    net_2.load_state_dict(w_n2)
    w_n3 = get_sub_paras(w_glob, local_w_masks[2], local_b_masks[2])
    net_3.load_state_dict(w_n3)
    w_n4 = get_sub_paras(w_glob, local_w_masks[3], local_b_masks[3])
    net_4.load_state_dict(w_n4)

    w_n51 = get_sub_paras(w_glob, local_w_masks[4], local_b_masks[4])
    net_51.load_state_dict(w_n51)
    w_n52 = get_sub_paras(w_glob, local_w_masks[5], local_b_masks[5])
    net_52.load_state_dict(w_n52)
    w_n53 = get_sub_paras(w_glob, local_w_masks[6], local_b_masks[6])
    net_53.load_state_dict(w_n53)

    setting_arrays = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        #
        [1, 1, 1, 1, 1, 1, 4, 4, 4, 4],

        [1, 1, 1, 1, 1, 4, 4, 4, 4, 7],

        [1, 1, 1, 1, 2, 2, 3, 3, 4, 4],

        [1, 1, 1, 1, 2, 3, 4, 4, 4, 4],

        [1, 1, 1, 1, 4, 4, 4, 4, 7, 7],

        [1, 1, 1, 1, 2, 3, 4, 5, 6, 7],

        [1, 1, 1, 1, 5, 5, 6, 6, 7, 7],

        [1, 1, 1, 4, 5, 5, 6, 6, 7, 7],

        [1, 2, 3, 4, 5, 5, 6, 6, 7, 7],

        [1, 4, 5, 5, 6, 6, 6, 7, 7, 7],

        [2, 2, 3, 3, 4, 4, 5, 6, 7, 7]
    ]

    setting_array = setting_arrays[0]
    for setting_array in setting_arrays:
        net_glob.load_state_dict(starting_weights)
        # training
        loss_train = []
        cv_loss, cv_acc = [], []
        val_loss_pre, counter = 0, 0
        net_best = None
        best_loss = None
        val_acc_list, net_list = [], []

        # if args.all_clients:
        #     print("Aggregation over all clients")
        #     w_locals = [w_glob for i in range(args.num_users)]
        print(setting_array)
        setting = str(setting_array).replace(",", "").replace(" ", "").replace("[", "").replace("]", "")
        pic_name = './save/fed_{}_{}_{}_lep{}_iid{}_{}.png'.format(args.dataset, args.model, args.epochs, args.local_ep,
                                                                   args.iid, setting)
        txt_name = './save/fed_{}_{}_{}_lep{}_iid{}_{}.txt'.format(args.dataset, args.model, args.epochs, args.local_ep,
                                                                   args.iid, setting)
        npy_name = './save/fed_{}_{}_{}_lep{}_iid{}_{}.npy'.format(args.dataset, args.model, args.epochs, args.local_ep,
                                                                   args.iid, setting)

        for iter in range(args.epochs):

            if iter > 0:  # >=5 , %5, % 50, ==5
                w_glob = net_glob.state_dict()

                # ABS OR NO ABS
                wranks = torch.argsort(torch.absolute(w_glob['layer_input.weight'].view(-1)))
                # wranks = torch.argsort(w_glob['layer_input.weight'].view(-1))

                local_w_masks = get_local_wmasks(wranks)
                branks = torch.argsort(torch.absolute(w_glob['layer_input.bias']))
                local_b_masks = get_local_bmasks(branks)

                w_n2 = get_sub_paras(w_glob, local_w_masks[1], local_b_masks[1])
                net_2.load_state_dict(w_n2)
                w_n3 = get_sub_paras(w_glob, local_w_masks[2], local_b_masks[2])
                net_3.load_state_dict(w_n3)
                w_n4 = get_sub_paras(w_glob, local_w_masks[3], local_b_masks[3])
                net_4.load_state_dict(w_n4)

                w_n51 = get_sub_paras(w_glob, local_w_masks[4], local_b_masks[4])
                net_51.load_state_dict(w_n51)
                w_n52 = get_sub_paras(w_glob, local_w_masks[5], local_b_masks[5])
                net_52.load_state_dict(w_n52)
                w_n53 = get_sub_paras(w_glob, local_w_masks[6], local_b_masks[6])
                net_53.load_state_dict(w_n53)

                """
                net_glob : full net
                net_2~net_4: 75% net 
                net_51 ~ net_53 50% net     
                """
            loss_locals = []
            if not args.all_clients:
                w_locals = []
            m = max(int(args.frac * args.num_users), 1)
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)
            print(idxs_users)

            type_array = []

            for id, idx in enumerate(idxs_users):
                # typep = get_user_typep(idx, setting_array)
                typep = setting_array[id]
                local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
                if typep == 1:
                    type_array.append(1)
                    w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
                elif typep == 2:
                    type_array.append(2)
                    w, loss = local.train(net=copy.deepcopy(net_2).to(args.device))
                    w = get_sub_paras(w, local_w_masks[1], local_b_masks[1])
                elif typep == 3:
                    type_array.append(3)
                    w, loss = local.train(net=copy.deepcopy(net_3).to(args.device))
                    w = get_sub_paras(w, local_w_masks[2], local_b_masks[2])
                elif typep == 4:
                    type_array.append(4)
                    w, loss = local.train(net=copy.deepcopy(net_4).to(args.device))
                    w = get_sub_paras(w, local_w_masks[3], local_b_masks[3])


                elif typep == 5:
                    type_array.append(5)
                    w, loss = local.train(net=copy.deepcopy(net_51).to(args.device))
                    w = get_sub_paras(w, local_w_masks[4], local_b_masks[4])
                elif typep == 6:
                    type_array.append(6)
                    w, loss = local.train(net=copy.deepcopy(net_51).to(args.device))
                    w = get_sub_paras(w, local_w_masks[5], local_b_masks[5])
                elif typep == 7:
                    type_array.append(7)
                    w, loss = local.train(net=copy.deepcopy(net_51).to(args.device))
                    w = get_sub_paras(w, local_w_masks[6], local_b_masks[6])
                # if args.all_clients:
                #     # w_locals[idx] = copy.deepcopy(w)
                #     pass
                # else:
                w_locals.append(copy.deepcopy(w))
                loss_locals.append(copy.deepcopy(loss))

            # with open(txt_name, 'a+') as f:
            #     print(type_array, file=f)
            print(type_array)
            # FOR ITER

            # update global weights
            w_glob = FedAvg(w_locals, type_array, local_w_masks, local_b_masks)

            # copy weight to net_glob
            net_glob.load_state_dict(w_glob)
            # with open(txt_name, 'a+') as f:
            #     print(loss_locals, file=f)
            print(loss_locals)
            # print loss
            loss_avg = sum(loss_locals) / len(loss_locals)
            with open(txt_name, 'a+') as f:
                print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg), file=f)
            print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
            loss_train.append(loss_avg)

            net_glob.eval()
            acc_test, loss_test = test_img(net_glob, dataset_test, args)
            with open(txt_name, 'a+') as f:
                print('TRound {:3d}, Testing loss {:.3f}'.format(iter, loss_test), file=f)
                print('ARound {:3d}, Testing accuracy: {:.2f}'.format(iter, acc_test), file=f)
            print('LRound {:3d}, Testing loss {:.3f}'.format(iter, loss_test))
            print('ARound {:3d}, Testing accuracy: {:.2f}'.format(iter, acc_test))
            net_glob.train()

        # plot loss curve
        plt.figure()
        plt.plot(range(len(loss_train)), loss_train)
        plt.ylabel('train_loss')
        plt.savefig(pic_name)
        np.save(npy_name, loss_train)
        # testing
        net_glob.eval()
        acc_train, loss_train = test_img(net_glob, dataset_train, args)
        acc_test, loss_test = test_img(net_glob, dataset_test, args)
        with open(txt_name, 'a+') as f:
            print("Training accuracy: {:.2f}".format(acc_train), file=f)
            print("Testing accuracy: {:.2f}".format(acc_test), file=f)
        # np.save(npy_name, loss_train)
        print("Training accuracy: {:.2f}==================================================".format(acc_train))
        print("Testing accuracy: {:.2f}==================================================".format(acc_test))
