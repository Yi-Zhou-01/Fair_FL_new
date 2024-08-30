#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from sampling import cifar_iid, cifar_noniid
from sampling import adult_iid, adult_noniid
from dataset import AdultDataset
import os


# def get_dataset(args):
#     """ Returns train and test datasets and a user group which is a dict where
#     the keys are the user index and the values are the corresponding data for
#     each of those users.
#     """

#     if  args.dataset == 'adult':

#         csv_file_train = os.getcwd()+'/data/adult/adult_encoded_80train.csv'
#         csv_file_test =  os.getcwd()+'/data/adult/adult_encoded_20test.csv'

#         # train_dataset = AdultDataset(csv_file_train)
#         test_dataset = AdultDataset(csv_file_test)

#         if args.iid:
#             # Sample IID user data from Mnist
#             train_dataset = AdultDataset(csv_file_train)
#             user_groups = adult_iid(train_dataset, args.num_users)
#         else:
#             # Sample Non-IID user data from Mnist
#             if args.unequal:
#                 # Chose uneuqal splits for every user
#                 raise NotImplementedError()
#             else:
#                 crop = 35600
#                 train_dataset = AdultDataset(csv_file_train, crop)
#                 # Chose euqal splits for every user
#                 user_groups = adult_noniid(train_dataset, args.num_users)


#     elif args.dataset == 'cifar':
#         data_dir = '../data/cifar/'
#         apply_transform = transforms.Compose(
#             [transforms.ToTensor(),
#              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#         train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
#                                        transform=apply_transform)

#         test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
#                                       transform=apply_transform)

#         # sample training data amongst users
#         if args.iid:
#             # Sample IID user data from Mnist
#             user_groups = cifar_iid(train_dataset, args.num_users)
#         else:
#             # Sample Non-IID user data from Mnist
#             if args.unequal:
#                 # Chose uneuqal splits for every user
#                 raise NotImplementedError()
#             else:
#                 # Chose euqal splits for every user
#                 user_groups = cifar_noniid(train_dataset, args.num_users)

#     elif args.dataset == 'mnist' or 'fmnist':
#         if args.dataset == 'mnist':
#             data_dir = '../data/mnist/'
#         else:
#             data_dir = '../data/fmnist/'

#         apply_transform = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.1307,), (0.3081,))])

#         train_dataset = datasets.MNIST(data_dir, train=True, download=True,
#                                        transform=apply_transform)

#         test_dataset = datasets.MNIST(data_dir, train=False, download=True,
#                                       transform=apply_transform)

#         # sample training data amongst users
#         if args.iid:
#             # Sample IID user data from Mnist
#             user_groups = mnist_iid(train_dataset, args.num_users)
#         else:
#             # Sample Non-IID user data from Mnist
#             if args.unequal:
#                 # Chose uneuqal splits for every user
#                 user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
#             else:
#                 # Chose euqal splits for every user
#                 user_groups = mnist_noniid(train_dataset, args.num_users)

#     return train_dataset, test_dataset, user_groups


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg



def get_fpr_diff(p, y, a):
    fpr = torch.abs(torch.sum(p * (1 - y) * a) / (torch.sum(a) + 1e-5) 
                    - torch.sum(p * (1 - y) * (1 - a)) / (torch.sum(1 - a) + 1e-5))
    return fpr

def get_tpr_diff(p, y, a):
    tpr = torch.abs(torch.sum(p * y * a) / (torch.sum(a) + 1e-5) 
                - torch.sum(p * y * (1 - a)) / (torch.sum(1 - a) + 1e-5))
    
    # fnr = torch.abs(torch.sum((1 - p) * y * a) / (torch.sum(a) + 1e-5) 
    #                 - torch.sum((1 - p) * y * (1 - a)) / (torch.sum(1 - a) + 1e-5))

    return tpr

def equalized_odds_diff(p, y, a):
    # return (get_fpr_diff(p, y, a)+get_tpr_diff(p, y, a))

    # return torch.max(get_fpr_diff(p, y, a), get_tpr_diff(p, y, a))
    return torch.mean(torch.tensor([get_fpr_diff(p, y, a), get_tpr_diff(p, y, a)]))


# max(np.abs(self.difference(self.false_positive_rate)), 
#                         np.abs(self.difference(self.true_positive_rate)))



def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return


def check_train_test_split(num_clients, pred_train_dic, pred_test_dic, save_dir=None):

    
    # print(pred_test_dic)
    # print(pred_test_dic.keys())
    # print(pred_test_dic)
    lines = []
    for i in range(num_clients):

        # Calculate tt;  tf; ff; ft
        # tt: Y=1 A=0
        test_len = len(pred_test_dic["labels"][i])
        train_len = len(pred_train_dic["labels"][i])

        YA_11 = ((pred_test_dic["labels"][i]) * (pred_test_dic["s_attr"][i]) )
        YA_10 = ((pred_test_dic["labels"][i]) * (1-pred_test_dic["s_attr"][i]) )
        YA_00 = ((1-pred_test_dic["labels"][i]) * (1-pred_test_dic["s_attr"][i]) )
        YA_01 = ((1-pred_test_dic["labels"][i]) * (pred_test_dic["s_attr"][i]) )

        YA_11_tr = ((pred_train_dic["labels"][i]) * (pred_train_dic["s_attr"][i]) )
        YA_10_tr = ((pred_train_dic["labels"][i]) * (1-pred_train_dic["s_attr"][i]) )
        YA_00_tr = ((1-pred_train_dic["labels"][i]) * (1-pred_train_dic["s_attr"][i]) )
        YA_01_tr = ((1-pred_train_dic["labels"][i]) * (pred_train_dic["s_attr"][i]) )

        tp_p = ((pred_test_dic["labels"][i]) * (pred_test_dic["pred_labels_fedavg"][i]) * (pred_test_dic["s_attr"][i]) )
        tp_unp = ((pred_test_dic["labels"][i]) * (pred_test_dic["pred_labels_fedavg"][i]) * (1-pred_test_dic["s_attr"][i]) )
        tp_p_tr = ((pred_train_dic["labels"][i]) * (pred_train_dic["pred_labels_fedavg"][i]) * (pred_train_dic["s_attr"][i]) )
        tp_unp_tr = ((pred_train_dic["labels"][i]) * (pred_train_dic["pred_labels_fedavg"][i]) * (1-pred_train_dic["s_attr"][i]) )

        # yYA_010 = torch.sum((pred_test_dic["labels"][i]) * (pred_test_dic["labels"][i]) * (1-pred_test_dic["s_attr"][i]) )
        # yYA_011
        # yYA_100
        # yYA_101
        
        tp_p_ = ((pred_test_dic["labels"][i]) * (pred_test_dic["pred_labels_pp"][i]) * (pred_test_dic["s_attr"][i]) )
        tp_unp_ = ((pred_test_dic["labels"][i]) * (pred_test_dic["pred_labels_pp"][i]) * (1-pred_test_dic["s_attr"][i]) )
        tp_p_tr_ = ((pred_train_dic["labels"][i]) * (pred_train_dic["pred_labels_pp"][i]) * (pred_train_dic["s_attr"][i]) )
        tp_unp_tr_ = ((pred_train_dic["labels"][i]) * (pred_train_dic["pred_labels_pp"][i]) * (1-pred_train_dic["s_attr"][i]) )
       



        # Number of prediction: 0 -> 1
        # Test Set
        flip_01 = (1 - pred_test_dic["pred_labels_fedavg"][i]) * (pred_test_dic["pred_labels_pp"][i])
        flip_10 = (pred_test_dic["pred_labels_fedavg"][i]) * (1 - pred_test_dic["pred_labels_pp"][i])
        
        flip_01_p = flip_01 * (pred_test_dic["s_attr"][i])
        flip_01_unp = flip_01 * (1-pred_test_dic["s_attr"][i])

        flip_10_p = flip_10 * (pred_test_dic["s_attr"][i])
        flip_10_unp = flip_10 * (1 - pred_test_dic["s_attr"][i])

        flip_01_p_Y1 = flip_01_p * (pred_test_dic["labels"][i])
        flip_01_unp_Y1 = flip_01_p * (pred_test_dic["labels"][i])

        flip_10_p_Y0 = flip_10_p * (1 -pred_test_dic["labels"][i])
        flip_10_unp_Y0 = flip_10_unp * (1 - pred_test_dic["labels"][i])

        # flip_01_YA_00 = flip_01 * YA_00
        # flip_01_YA_01 = flip_01 * YA_01

        # flip_10_YA_10 = flip_10 * YA_10
        # flip_10_YA_11 = flip_10 * YA_11

        lines.append("******** Client #{} ********".format(i+1))
        lines.append("           Test (%)   -   Train (%)")
        lines.append("#Samples:  {}        -    {}".format(test_len, train_len))
        lines.append("YA_11:     {:n} ({:.0%})     {:n} ({:.0%})".format(sum(YA_11), sum(YA_11)/test_len, sum(YA_11_tr),sum(YA_11_tr)/train_len ))
        lines.append("YA_00:     {:n} ({:.0%})     {:n} ({:.0%})".format(sum(YA_00), sum(YA_00)/test_len, sum(YA_00_tr),sum(YA_00_tr)/train_len))
        lines.append("YA_10:     {:n} ({:.0%})      {:n} ({:.0%})".format(sum(YA_10), sum(YA_10)/test_len, sum(YA_10_tr),sum(YA_10_tr)/train_len ))
        lines.append("YA_01:     {:n} ({:.0%})     {:n} ({:.0%})".format(sum(YA_01), sum(YA_01)/test_len, sum(YA_01_tr),sum(YA_01_tr)/train_len ))
        lines.append("-")
        lines.append("flip_01_p:  {:n}, Y=1: {:n}".format(sum(flip_01_p), sum(flip_01_p_Y1)))
        lines.append("flip_01_up: {:n}, Y=1: {:n}".format(sum(flip_01_unp), sum(flip_01_unp_Y1)))
        lines.append("flip_10_p:  {:n}, Y=0: {:n}".format(sum(flip_10_p), sum(flip_10_p_Y0)))
        lines.append("flip_10_up: {:n}, Y=0: {:n}".format(sum(flip_10_unp), sum(flip_10_unp_Y0)))
        lines.append("-")
        lines.append("Before post-processing ...")
        lines.append("             Test (%)    -    Train (%)")
        try:
            lines.append("TPR  p:     {:n}/{:n} ({:.0%})     {:n}/{:n} ({:.0%})".format(sum(tp_p),sum(YA_11), sum(tp_p)/sum(YA_11),\
                                                                                        sum(tp_p_tr),sum(YA_11_tr),sum(tp_p_tr)/sum(YA_11_tr) ))
            lines.append("TPR  unp:   {:n}/{:n} ({:.0%})     {:n}/{:n} ({:.0%})".format(sum(tp_unp), sum(YA_10), sum(tp_unp)/sum(YA_10),\
                                                                                        sum(tp_unp_tr),sum(YA_10_tr), sum(tp_unp_tr)/sum(YA_10_tr) ))
            lines.append("TPR Diff         ({:.0%})          ({:.0%})".format(sum(tp_unp)/sum(YA_10)-sum(tp_p)/sum(YA_11),\
                                                                            sum(tp_unp_tr)/sum(YA_10_tr)-sum(tp_p_tr)/sum(YA_11_tr) ))
            lines.append("-")   
            lines.append("After post-processing ...")
            lines.append("             Test (%)    -    Train (%)")
            lines.append("TPR  p:     {:n}/{:n} ({:.0%})     {:n}/{:n} ({:.0%})".format(sum(tp_p_),sum(YA_11), sum(tp_p_)/sum(YA_11),\
                                                                                        sum(tp_p_tr_),sum(YA_11_tr),sum(tp_p_tr_)/sum(YA_11_tr) ))
            lines.append("TPR  unp:   {:n}/{:n} ({:.0%})     {:n}/{:n} ({:.0%})".format(sum(tp_unp_), sum(YA_10), sum(tp_unp_)/sum(YA_10),\
                                                                                        sum(tp_unp_tr_),sum(YA_10_tr), sum(tp_unp_tr_)/sum(YA_10_tr) ))
            lines.append("TPR Diff         ({:.0%})          ({:.0%})".format(sum(tp_unp_)/sum(YA_10)-sum(tp_p_)/sum(YA_11),\
                                                                         sum(tp_unp_tr_)/sum(YA_10_tr)-sum(tp_p_tr_)/sum(YA_11_tr) ))
        except:
            lines.append("Scalar error!!")
        lines.append("-")   

        # Number of prediction: 0 -> 1 but true label is 0
        # count_false_01 =  torch.sum((1 - pred_test_dic["pred_labels_fedavg"]) * (pred_test_dic["pred_labels_pp"]) *  (1 - pred_test_dic["labels"]) )
        
        # # Number of prediction: 1 -> 0 but true label is 1
        # count_false_10 =  torch.sum((pred_test_dic["pred_labels_fedavg"]) * (1 - pred_test_dic["pred_labels_pp"]) *  (pred_test_dic["labels"]) )
        
        # torch.sum(p * y * a)
        # print(pred_test_dic)
        # print(pred_test_dic["pred_labels_fedavg"][i])
    
    for line in lines:
        print(line)
    
    if save_dir is not None:
        with open(save_dir + '/print_stats.txt', 'w') as f:
            for line in lines:
                f.write(f"{line}\n")

    # with open('somefile.txt', 'a') as the_file:
    # the_file.write('Hello\n')




# def check_stats_compas(part_idx=4):
    
    
#     local_test_ratio = 0.2
#     csv_file_train = "/Users/zhouyi/Desktop/Fair_FL_new/data/compas/compas_encoded_all.csv"
#     train_dataset = CompasDataset(csv_file_train)
    
#     if part_idx ==1 :
#         num_clients = 10
#     else:
#         num_clients = 4
#     partition_file = dataset.get_partition(part_idx, dataset="compas")
#     user_groups =  np.load(partition_file, allow_pickle=True).item()

#     local_set_ls = []
#     for i in range(num_clients):
#         local_idxs = user_groups[i]
#         local_dataset = update.LocalDataset(train_dataset, local_idxs, test_ratio=local_test_ratio)
#         local_set_ls.append(local_dataset)
        

#     target_list = list(train_dataset.df["two_year_recid"])
#     sens_list = list(train_dataset.df["sex"])
#     tt_sample_ls = []
#     ff_sample_ls = []
#     tf_sample_ls = []
#     ft_sample_ls = []
    
#     print("Partition #",part_idx, " -- All samples: ", len(target_list))
    
#     for c_idx in range(num_clients):
#         samples = user_groups[c_idx]
#         tt_sample = 0
#         ff_sample = 0
#         tf_sample = 0
#         ft_sample = 0
#         tt_sample_t = 0
#         ff_sample_t = 0
#         tf_sample_t = 0
#         ft_sample_t = 0
#         tt_sample_tt = 0
#         ff_sample_tt = 0
#         tf_sample_tt = 0
#         ft_sample_tt = 0
        
# #         tt_sample = [ samples[idx] for idx in range(len(samples)) if (sens_list[idx] and target_list[idx])]
# #         ff_sample = [ samples[idx] for idx in range(len(samples)) if (not sens_list[idx] and not target_list[idx])]
# #         tf_sample = [ samples[idx] for idx in range(len(samples)) if (sens_list[idx] and not target_list[idx])]
        
# #         tt_sample_ls.append(tt_sample)
# #         ff_sample_ls.append(ff_sample)
# #         tf_sample_ls.append(tf_sample)
        
#         # Y1A1 : tt
#         # Y1A0: tf
#         for idx in samples:
#             tt_sample += sens_list[idx] and target_list[idx]
#             ff_sample += (not sens_list[idx]) and (not target_list[idx])
#             tf_sample += (not sens_list[idx]) and ( target_list[idx])
#             ft_sample += ( sens_list[idx]) and ( not target_list[idx])
        
#         for idx in local_set_ls[c_idx].test_set_idxs:
#             tt_sample_t += sens_list[idx] and target_list[idx]
#             ff_sample_t += (not sens_list[idx]) and (not target_list[idx])
#             tf_sample_t += (not sens_list[idx]) and ( target_list[idx])
#             ft_sample_t += ( sens_list[idx]) and (not target_list[idx])
        
#         for idx in local_set_ls[c_idx].train_set_idxs:
#             tt_sample_tt += sens_list[idx] and target_list[idx]
#             ff_sample_tt += (not sens_list[idx]) and (not target_list[idx])
#             tf_sample_tt += (not sens_list[idx]) and ( target_list[idx])
#             ft_sample_tt += ( sens_list[idx]) and (not target_list[idx])
        
#         tt_sample_ls.append(tt_sample)
#         ff_sample_ls.append(ff_sample)
#         tf_sample_ls.append(tf_sample)
#         ft_sample_ls.append(ft_sample)
        
# #         print("******** Client # ", c_idx+1)
# #         print("      All - Test - Train")
# #         print("tt:  {:.2f} - {:.2f} - {:.2f} * {:.2f} - {:.2f}".format(
# #             (tt_sample)/len(samples), tt_sample_t/len(local_set_ls[c_idx].test_set_idxs),tt_sample_tt/len(local_set_ls[c_idx].train_set_idxs), tt_sample_t, len(samples)))
# #         print("ff:  {:.2f} - {:.2f} - {:.2f}".format(
# #             (ff_sample)/len(samples), ff_sample_t/len(local_set_ls[c_idx].test_set_idxs),ff_sample_tt/len(local_set_ls[c_idx].train_set_idxs)))
# #         print("tf:  {:.2f} - {:.2f} - {:.2f}".format(
# #             (tf_sample)/len(samples), tf_sample_t/len(local_set_ls[c_idx].test_set_idxs),tf_sample_tt/len(local_set_ls[c_idx].train_set_idxs)))
        
#         print("******** Client # ", c_idx+1, "****** #Samp: ", len(user_groups[c_idx]))
# #         print("Client samples: ", len(user_groups[c_idx]))
# #         print("Local size: ", local_set_ls[c_idx].size)
# #         print("Local idx len: ", len(local_set_ls[c_idx].local_idxs))
# #         print(len(local_set_ls[c_idx].train_set_idxs), len(local_set_ls[c_idx].test_set_idxs), len(local_set_ls[c_idx].val_set_idxs))
        
#         print("     Test - Train * #(tt/tf) - #Samp(Test) - #Samp(Train)")
        
#         print("tt:  {:.2f} - {:.2f}  *  {:.2f} - {:.2f} - {:.2f}".format(
#              tt_sample_t/len(local_set_ls[c_idx].test_set_idxs),tt_sample_tt/len(local_set_ls[c_idx].train_set_idxs),
#             tt_sample_t, len(local_set_ls[c_idx].test_set_idxs), len(local_set_ls[c_idx].train_set_idxs)))
        
#         print("ft:  {:.2f} - {:.2f}".format(
#              ft_sample_t/len(local_set_ls[c_idx].test_set_idxs),ft_sample_tt/len(local_set_ls[c_idx].train_set_idxs)))
# #         print("tt+ft:   {:.2f} - {:.2f} * {:.2f}".format(
# #              (ft_sample_t+tt_sample_t)/(len(local_set_ls[c_idx].test_set_idxs)),
# #              (ft_sample_tt+tt_sample_tt)/(len(local_set_ls[c_idx].train_set_idxs)),
# #             (ft_sample_t+tt_sample_t) ))
        
#         print("ff:  {:.2f} - {:.2f}".format(
#              ff_sample_t/len(local_set_ls[c_idx].test_set_idxs),ff_sample_tt/len(local_set_ls[c_idx].train_set_idxs)))
#         print("tf:  {:.2f} - {:.2f}  *  {:.2f} ".format(
#              tf_sample_t/len(local_set_ls[c_idx].test_set_idxs),tf_sample_tt/len(local_set_ls[c_idx].train_set_idxs),
#              tf_sample_t))

#     return tt_sample_ls, ff_sample_ls, tf_sample_ls, tt_sample_t, ff_sample_t, tf_sample_t