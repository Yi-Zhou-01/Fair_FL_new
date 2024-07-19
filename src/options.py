#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse
import os


def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--epochs', type=int, default=10,
                        help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=100,
                        help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.1,
                        help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=10,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=10,
                        help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')

    # model arguments
    parser.add_argument('--model', type=str, default='mlp', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9,
                        help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to \
                        use for convolution')
    parser.add_argument('--num_channels', type=int, default=1, help="number \
                        of channels of imgs")
    parser.add_argument('--norm', type=str, default='batch_norm',
                        help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32,
                        help="number of filters for conv nets -- 32 for \
                        mini-imagenet, 64 for omiglot.")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than \
                        strided convolutions")

    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name \
                        of dataset")
    parser.add_argument('--num_classes', type=int, default=10, help="number \
                        of classes")
    parser.add_argument('--gpu', default=None, help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--optimizer', type=str, default='sgd', help="type \
                        of optimizer")
    parser.add_argument('--iid', type=int, default=1,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--unequal', type=int, default=0,
                        help='whether to use unequal data splits for  \
                        non-i.i.d setting (use 0 for equal splits)')
    parser.add_argument('--stopping_rounds', type=int, default=10,
                        help='rounds of early stopping')
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    parser.add_argument('--seed', type=int, default=1, help='random seed')

    # Add partition file path 
    # default_path = os.getcwd()+ "/data/adult/partition/1/user_groups_10clients_0.5alpha_diri_income_adult_all_33col_70train_0.npy"
    parser.add_argument('--partition_idx', type=int, default=1, help='partition file path')

    # Choose which fl algorithm to perform
    parser.add_argument('--fl_new', type=bool, default=True, help='fl algorithm: new')

    parser.add_argument('--fl_fairfed', type=bool, default=True, help='fl algorithm: FairFed')

    parser.add_argument('--post_proc_cost', type=str, default='fpr', help='post-processing cost constraint.')

    parser.add_argument('--idx', type=int, help='Experiment index.')

    parser.add_argument('--beta', type=float, default=0.3, help='Beta parameter for Fairfed, "fairness budget".')
    
    parser.add_argument('--local_test_ratio', type=float, default=0.2, help='Local test set ratio')

    parser.add_argument('--plot_tpfp', type=bool, default=False, help='Local test set ratio')

    parser.add_argument('--fairfed_ep', type=int, default=20, help='Global training round for FairFed')
    
    parser.add_argument('--debias', type=str, default="pp", help='Local debias approaches for new fl_new: "pp" for post-processing and "ft" for final layer fine-tuning')
    
    parser.add_argument('--ft_alpha', type=float, default=1, help='parameter alpha used to calculate loss in final layer fine-tuning: loss = loss + alpha * loss_fairness')
    
    parser.add_argument('--ft_ep', type=int, default=5, help='Final layer fine-tuning epochs')
    
    parser.add_argument('--local_split', type=str, default="", help='File path if use saved local client train/test split; Empty if want a newly generated one.')
    
    parser.add_argument('--kaggle', type=bool, default=False, help='if runs on kaggle')
    

    args = parser.parse_args()
    return args

# def args_parser():
#     parser = argparse.ArgumentParser()

#     # federated arguments (Notation for the arguments followed from paper)
#     parser.add_argument('--epochs', type=int, default=10,
#                         help="number of rounds of training")
#     parser.add_argument('--num_users', type=int, default=100,
#                         help="number of users: K")
#     parser.add_argument('--frac', type=float, default=0.1,
#                         help='the fraction of clients: C')
#     parser.add_argument('--local_ep', type=int, default=10,
#                         help="the number of local epochs: E")
#     parser.add_argument('--local_bs', type=int, default=10,
#                         help="local batch size: B")
#     parser.add_argument('--lr', type=float, default=0.01,
#                         help='learning rate')
#     parser.add_argument('--momentum', type=float, default=0.5,
#                         help='SGD momentum (default: 0.5)')

#     # model arguments
#     parser.add_argument('--model', type=str, default='mlp', help='model name')
#     parser.add_argument('--kernel_num', type=int, default=9,
#                         help='number of each kind of kernel')
#     parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
#                         help='comma-separated kernel size to \
#                         use for convolution')
#     parser.add_argument('--num_channels', type=int, default=1, help="number \
#                         of channels of imgs")
#     parser.add_argument('--norm', type=str, default='batch_norm',
#                         help="batch_norm, layer_norm, or None")
#     parser.add_argument('--num_filters', type=int, default=32,
#                         help="number of filters for conv nets -- 32 for \
#                         mini-imagenet, 64 for omiglot.")
#     parser.add_argument('--max_pool', type=str, default='True',
#                         help="Whether use max pooling rather than \
#                         strided convolutions")

#     # other arguments
#     parser.add_argument('--dataset', type=str, default='mnist', help="name \
#                         of dataset")
#     parser.add_argument('--num_classes', type=int, default=10, help="number \
#                         of classes")
#     parser.add_argument('--gpu', default=None, help="To use cuda, set \
#                         to a specific GPU ID. Default set to use CPU.")
#     parser.add_argument('--optimizer', type=str, default='sgd', help="type \
#                         of optimizer")
#     parser.add_argument('--iid', type=int, default=1,
#                         help='Default set to IID. Set to 0 for non-IID.')
#     parser.add_argument('--unequal', type=int, default=0,
#                         help='whether to use unequal data splits for  \
#                         non-i.i.d setting (use 0 for equal splits)')
#     parser.add_argument('--stopping_rounds', type=int, default=10,
#                         help='rounds of early stopping')
#     parser.add_argument('--verbose', type=int, default=1, help='verbose')
#     parser.add_argument('--seed', type=int, default=1, help='random seed')
#     args = parser.parse_args()
#     return args
