#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm
import os
import pandas as pd

import torch
from tensorboardX import SummaryWriter
# from torch.utils.tensorboard import SummaryWriter

import options
import dataset

# from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, MLPAdult, CNNMnist, CNNFashion_Mnist, CNNCifar, Plain_LR_Adult, MLPAdult2, MLPCompas
from utils import average_weights, exp_details
import os
import update
import plot
import pickle
import utils
import models
# import tracemalloc
# from memory_profiler import profile



from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.algorithms.postprocessing.calibrated_eq_odds_postprocessing import CalibratedEqOddsPostprocessing
from aif360.algorithms.postprocessing import EqOddsPostprocessing


# from dataset import get_dataset
# import dataset

# @profile
def main():
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = options.args_parser()
    exp_details(args)

    # if args.gpu_id:
    #     torch.cuda.set_device(args.gpu_id)
    # if args.gpu:
    #     torch.cuda.set_device(args.gpu)
    device = 'cuda' if args.gpu else 'cpu'
    print("Using devide: ", device)
    # print("Check device; ", device)
    # print("args.gpu: ", args.gpu)
    # # print("args.plot_tpfp: ", args.plot_tpfp)
    # if args.gpu == False:
    #     print("**&&")


    # Create folder to save training info
    if args.platform=="kaggle":
        save_to_root = "/kaggle/working"
    elif args.platform=="colab":
        save_to_root = "/content/drive/MyDrive/Fair_FL_new/save"
    elif args.platform=="azure":
        save_to_root = os.getcwd() + '/save'
    else:
        save_to_root =  os.getcwd() + '/save'

    all_fl = ""
    if args.fl_new:
        all_fl = all_fl + "new"
    if args.fl_fairfed:
        all_fl = all_fl + "fairfed"    
    statistics_dir = save_to_root+'/statistics/{}/{}_{}_{}_{}_frac{}_client{}_lr{}_part{}_beta{}_ep{}_{}_{}_ftep_{}_{}'.\
        format(args.idx, all_fl, args.debias, args.dataset, args.model, args.frac, args.num_users,
               args.lr, args.partition_idx, args.beta, args.epochs, args.local_ep, args.fairfed_ep, args.ft_ep, args.rep)    # <------------- iid tobeadded
        # Save to files ...
        # TBA
    os.makedirs(statistics_dir, exist_ok=True)

    # load dataset and user groups
    # user groups: different client datasets
    print("time: ", int(time.time() - start_time), int(time.time()))
    print("Getting dataset ... ")
    train_dataset, test_dataset, user_groups = dataset.get_dataset(args)

    print("time: ", int(time.time() - start_time), int(time.time()))

    # tracemalloc.start()

    # Split train/test for all local dataset
    print("Getting train/test split ... ")
    if args.local_split == "":
        local_set_ls = []
        for i in range(args.num_users):
            local_idxs = user_groups[i]
            # print("local_idxs: ", len(local_idxs))
            # print(local_idxs)
            # print(local_idxs)
            local_dataset = update.LocalDataset(train_dataset, local_idxs, test_ratio=args.local_test_ratio)
            local_set_ls.append(local_dataset)
            print("New local train/test split generated for Client {}: Train: {} | Test: {} | Total: {}".format(
                i, len(local_dataset.train_set_idxs), len(local_dataset.test_set_idxs), len(local_idxs)))
            # print("local_dataset.train_set_idxs: ", sorted(local_dataset.train_set_idxs))
            # print("local_dataset.test_set_idxs: ", sorted(local_dataset.test_set_idxs))
            # print("======================")
            # print("check shape")
            # print(local_set_ls[i].local_test_set.X.shape)
    else:
        with open(args.local_split, 'rb') as inp:
            local_set_ls = pickle.load(inp)
            print("Using saved local train/test split in: ", args.local_split)

    
    

    # BUILD MODEL
    img_size = train_dataset[0][0].shape
    global_model=models.get_model(args, img_size=img_size)



    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)

    # copy weights
    global_weights = global_model.state_dict()

    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0

    # Create statistics dictionary

    stat_keys = []
    set_split = ["train", "test"]
    # if args.plot_tpfp:
    #     local_metrics = ["acc", "eod", "tpr", "fpr"]
    # else:
    #     local_metrics = ["acc", "eod"]
    local_metrics = ["acc", "eod"]

    if args.fl_new:
        stat_keys += [ss+"_"+lm+"_"+"new" for ss in set_split for lm in local_metrics]
    # if args.fl_avg:
        stat_keys += [ss+"_"+lm+"_"+"fedavg" for ss in set_split for lm in local_metrics]
        if args.plot_tpfp:
            stat_keys += [ss+"_"+lm+"_"+"new" for ss in set_split for lm in  ["tpr", "fpr"]]
            stat_keys += [ss+"_"+lm+"_"+"fedavg" for ss in set_split for lm in  ["tpr", "fpr"]]
    if args.fl_fairfed:
        stat_keys += [ss+"_"+lm+"_"+"fairfed" for ss in set_split for lm in local_metrics]

    # stat_keys = ['train_acc_before','train_acc_after', 'test_acc_before', 'test_acc_after',
    #                 'train_eod_before', 'train_eod_after', 'test_eod_before', 'test_eod_after',
    #                 'train_fpr_before', 'train_fpr_after', 'test_fpr_before', 'test_fpr_after',
    #                 'train_tpr_before', 'train_tpr_after', 'test_tpr_before', 'test_tpr_after']
    
    # stat_dic = {k: [0]*args.num_users for k in stat_keys}
    # stat_dic = {k: [] for k in stat_keys}
    stat_dic = {k: np.zeros(args.num_users) for k in stat_keys}
    pred_train_dic = {}
    pred_test_dic = {}
    local_loss_all = []

    print("time: ", int(time.time() - start_time), int(time.time()))
    if args.fl_new:
        print("********* Start FedAvg/New Training **********")

        # For each (global) round of training
        for epoch in tqdm(range(args.epochs)):
            local_weights, local_losses = [], []
            print(f'\n | Global Training Round : {epoch+1} |\n')

            global_model.train()
            # Sample a subset of clients for training
            m = max(int(args.frac * args.num_users), 1)
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)

            # For each selected user do local_ep round of training
            for idx in idxs_users:
                local_dataset = local_set_ls[idx]
                split_idxs = (local_dataset.train_set_idxs,local_dataset.test_set_idxs,local_dataset.val_set_idxs)
                local_model = LocalUpdate(args=args, split_idxs=split_idxs, dataset=train_dataset,
                                        idxs=user_groups[idx], logger=logger)


                w, loss, _ = local_model.update_weights(
                    model=copy.deepcopy(global_model), global_round=epoch, client_idx=idx)
                local_weights.append(copy.deepcopy(w))
                local_losses.append(copy.deepcopy(loss))

            # update global weights
            global_weights = average_weights(local_weights)

            # update global weights
            global_model.load_state_dict(global_weights)

            loss_avg = sum(local_losses) / len(local_losses)
            train_loss.append(loss_avg)
            local_loss_all.append(local_losses)


            # print global training loss after every 'i' rounds
            print_every = 1
            if (epoch+1) % print_every == 0:
                print(f' \nAvg Training Stats after {epoch+1} global rounds:')
                print(f'Training Loss : {np.mean(np.array(train_loss))}')
                # print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))


        print("Start inference fairness: FedAvg train ...")
        list_acc, list_loss = [], []
        global_model.eval()
        all_local_train_y = []
        all_local_train_a = []
        all_local_train_pred = []
        all_local_train_eod = []
        pred_train_dic["pred_labels_fedavg"] = {}
        pred_train_dic["labels"] = {}
        pred_train_dic["s_attr"] = {}

        for c in range(args.num_users):
            print("time: ", int(time.time() - start_time), int(time.time()))
            local_dataset = local_set_ls[c]
            split_idxs = (local_dataset.train_set_idxs,local_dataset.test_set_idxs,local_dataset.val_set_idxs)
            local_model = LocalUpdate(args=args, split_idxs=split_idxs, dataset=train_dataset,
                                    idxs=user_groups[idx], logger=logger)

            # local_model = LocalUpdate(args=args, local_dataset=local_dataset, dataset=train_dataset,
            #                         idxs=user_groups[idx], logger=logger)
            
            # acc, loss = local_model.inference(model=global_model)
            if args.plot_tpfp:
                fairness_metrics = ["eod","tpr","fpr"]
            else:
                fairness_metrics = ["eod"]
            acc, loss, all_y, all_a, all_pred, local_fairness = local_model.inference_w_fairness(model=global_model, set="train", fairness_metric=fairness_metrics, client_idx=c)

            stat_dic["train_acc_fedavg"][c] = acc
            stat_dic["train_eod_fedavg"][c] = local_fairness["eod"]
            if args.plot_tpfp:
                stat_dic["train_tpr_fedavg"][c] = local_fairness["tpr"]
                stat_dic["train_fpr_fedavg"][c] = local_fairness["fpr"]
            
            all_local_train_y.append(all_y)
            all_local_train_a.append(all_a)
            all_local_train_pred.append(all_pred)

            pred_train_dic["pred_labels_fedavg"][c] = np.array([])
            pred_train_dic["labels"][c] = np.array([])
            pred_train_dic["s_attr"][c] = np.array([])

            pred_train_dic["pred_labels_fedavg"][c] = (np.asarray(all_y))
            pred_train_dic["labels"][c] = (np.asarray(all_a))
            pred_train_dic["s_attr"][c] =(np.asarray(all_pred))


            all_local_train_eod.append( local_fairness["eod"])

            list_acc.append(acc)
            list_loss.append(loss)

        # get test metrics
        print("Start inference fairness: FedAvg test ...")
        all_local_test_y = []
        all_local_test_a = []
        all_local_test_pred = []
        pred_test_dic["pred_labels_fedavg"] = {}
        pred_test_dic["labels"] = {}
        pred_test_dic["s_attr"] = {}
        for c in range(args.num_users):
            local_dataset = local_set_ls[c]
            split_idxs = (local_dataset.train_set_idxs,local_dataset.test_set_idxs,local_dataset.val_set_idxs)
            local_model = LocalUpdate(args=args, split_idxs=split_idxs, dataset=train_dataset,
                                    idxs=user_groups[idx], logger=logger)
            if args.plot_tpfp:
                fairness_metrics = ["eod","tpr","fpr"]
            else:
                fairness_metrics = ["eod"]
            acc, loss, all_y, all_a, all_pred, local_fairness = local_model.inference_w_fairness(model=global_model, set="test", fairness_metric=fairness_metrics, client_idx=c)
            stat_dic["test_acc_fedavg"][c] = acc
            stat_dic["test_eod_fedavg"][c] = local_fairness["eod"]
            if args.plot_tpfp:
                stat_dic["test_tpr_fedavg"][c] = local_fairness["tpr"]
                stat_dic["test_fpr_fedavg"][c] = local_fairness["fpr"]
            
            pred_test_dic["pred_labels_fedavg"][c] = np.array([])
            pred_test_dic["labels"][c] = np.array([])
            pred_test_dic["s_attr"][c] = np.array([])

            pred_test_dic["pred_labels_fedavg"][c]  = (np.asarray(all_y))
            pred_test_dic["labels"][c]  = (np.asarray(all_a))
            pred_test_dic["s_attr"][c]  =  (np.asarray(all_pred))

            all_local_test_y.append(all_y)
            all_local_test_a.append(all_a)
            all_local_test_pred.append(all_pred)

        print("---- stat_dic ----")
        print(stat_dic["test_acc_fedavg"])

        
        # Evaluation locally after training
        # print("********* Start Local Evaluation and Post-processing **********")
        plot_file_loss = statistics_dir + "/loss_plot.png"
        fig_titl_loss = statistics_dir.split("/")[-1] + "_exp" + str(args.idx)
        plot.plot_loss(local_loss_all, train_loss,title=fig_titl_loss, save_to=plot_file_loss)


        # Print weights of model
        weights_fedavg = global_model.state_dict()

        # print("weights_fedavg: ", weights_fedavg)
        with open(statistics_dir+"/weights.txt", "a") as w_file:
            w_file.write("FedAvg weights: \n")
            w_file.write("final_layer.weight: \n" + str(weights_fedavg["final_layer.weight"]) +"\n")
            w_file.write("final_layer.bias: \n"+str(weights_fedavg["final_layer.bias"]) +"\n")
        

        privileged_groups = [{"a": 1}]
        unprivileged_groups = [{"a": 0}]

        
        print("time: ", int(time.time() - start_time), int(time.time()))
        # Post-processing approach
        if "pp" in args.debias:
            # Apply post-processing locally at each client:
            print("******** Start post-processing ******** ")
            pred_train_dic['pred_labels_pp'] = [] # np.zeros(args.num_users) 
            pred_test_dic['pred_labels_pp'] = [] # np.zeros(args.num_users) 
            for idx in range(args.num_users):
                idxs = user_groups[idx]
                local_set = local_set_ls[idx]

                train_bld_prediction_dataset = dataset.get_bld_dataset_w_pred(a=all_local_train_a[idx], pred_labels=all_local_train_pred[idx])
                train_bld_original = dataset.get_bld_dataset_w_pred(a=all_local_train_a[idx], pred_labels=all_local_train_y[idx])
                test_bld_prediction_dataset = dataset.get_bld_dataset_w_pred(a=all_local_test_a[idx], pred_labels=all_local_test_pred[idx])
                test_bld_original = dataset.get_bld_dataset_w_pred(a=all_local_test_a[idx], pred_labels=all_local_test_y[idx])



                cost_constraint = args.post_proc_cost # "fpr" # "fnr", "fpr", "weighted"
                # cost_constraint = "fnr"
                randseed = 12345679 

                # Fit post-processing model
                
                # cpp = CalibratedEqOddsPostprocessing(privileged_groups = privileged_groups,
                #                                 unprivileged_groups = unprivileged_groups,
                #                                 cost_constraint=cost_constraint,
                #                                 seed=randseed)
                
                cpp = EqOddsPostprocessing(privileged_groups = privileged_groups,
                                                unprivileged_groups = unprivileged_groups,
                                                seed=randseed)
                cpp = cpp.fit(train_bld_original, train_bld_prediction_dataset)

                
                # Prediction after post-processing
                local_train_dataset_bld_prediction_debiased = cpp.predict(train_bld_prediction_dataset)
                local_test_dataset_bld_prediction_debiased = cpp.predict(test_bld_prediction_dataset)

                # Metrics after post-processing
                cm_pred_train_debiased = ClassificationMetric(train_bld_original, local_train_dataset_bld_prediction_debiased,
                            unprivileged_groups=unprivileged_groups,
                            privileged_groups=privileged_groups)
                
                
                pred_train_dic['pred_labels_pp'].append((np.asarray(local_train_dataset_bld_prediction_debiased.labels.flatten())))
                pred_test_dic['pred_labels_pp'].append((np.asarray(local_test_dataset_bld_prediction_debiased.labels.flatten())))
               
                
                # pred_train_dic['pred_labels_pp'].append(torch.from_numpy(np.asarray(local_train_dataset_bld_prediction_debiased.labels.flatten())))
                # pred_test_dic['pred_labels_pp'].append(torch.from_numpy(np.asarray(local_test_dataset_bld_prediction_debiased.labels.flatten())))
                # train_acc_new.append(cm_pred_train_debiased.accuracy())
                # train_eod_new.append(cm_pred_train_debiased.equalized_odds_difference())
                stat_dic['train_acc_new'][idx] = (cm_pred_train_debiased.accuracy())
                # stat_dic['train_eod_new'].append(cm_pred_train_debiased.average_abs_odds_difference())
                stat_dic['train_eod_new'][idx] = (cm_pred_train_debiased.equalized_odds_difference())
                if args.plot_tpfp:
                    stat_dic['train_tpr_new'][idx] = (cm_pred_train_debiased.true_positive_rate_difference())
                    stat_dic['train_fpr_new'][idx] = (cm_pred_train_debiased.false_positive_rate_difference())
                # stat_dic['train_eod_after'].append(cm_pred_train_debiased.average_abs_odds_difference())
                # stat_dic['train_fpr_new'].append(abs(cm_pred_train_debiased.difference(cm_pred_train_debiased.false_positive_rate)))
                # stat_dic['train_tpr_new'].append(abs( cm_pred_train_debiased.difference(cm_pred_train_debiased.true_positive_rate)))


                cm_pred_test_debiased = ClassificationMetric(test_bld_original, local_test_dataset_bld_prediction_debiased,
                            unprivileged_groups=unprivileged_groups,
                            privileged_groups=privileged_groups)

                # test_acc_new.append(cm_pred_test_debiased.accuracy())
                # test_eod_new.append(cm_pred_test_debiased.equalized_odds_difference())
                stat_dic['test_acc_new'][idx] = (cm_pred_test_debiased.accuracy())
                # stat_dic['test_eod_new'].append(cm_pred_test_debiased.average_abs_odds_difference())
                stat_dic['test_eod_new'][idx] = (cm_pred_test_debiased.equalized_odds_difference())
                if args.plot_tpfp:
                    stat_dic['test_tpr_new'][idx] = (cm_pred_test_debiased.true_positive_rate_difference())
                    stat_dic['test_fpr_new'][idx] = (cm_pred_test_debiased.false_positive_rate_difference())
                # stat_dic['test_eod_after'].append(cm_pred_test_debiased.average_abs_odds_difference())
                # stat_dic['test_fpr_new'].append(abs(cm_pred_test_debiased.difference(cm_pred_test_debiased.false_positive_rate)))
                # stat_dic['test_tpr_new'].append(abs( cm_pred_test_debiased.difference(cm_pred_test_debiased.true_positive_rate)))
        
        # print("stats dic: ")
        # print(stat_dic)

        print("time: ", int(time.time() - start_time), int(time.time()))
        # Apply final-layer fine-tuning
        if "ft" in args.debias:
            
            print("******** Final layer fine-tuning ******** ")
            ft_keys = ['test_acc_new_ft', 'test_eod_new_ft', 'test_tpr_new_ft','test_fpr_new_ft', \
                       'train_acc_new_ft','train_eod_new_ft','train_tpr_new_ft', 'train_fpr_new_ft']
            for k in ft_keys:
                stat_dic[k] = np.zeros(args.num_users) 

            all_epoch_loss = []
            for c in range(args.num_users):
                local_dataset = local_set_ls[c]
                split_idxs = (local_dataset.train_set_idxs,local_dataset.test_set_idxs,local_dataset.val_set_idxs)
                local_model = LocalUpdate(args=args, split_idxs=split_idxs, dataset=train_dataset,
                                        idxs=user_groups[idx], logger=logger)

                local_train_model = copy.deepcopy(global_model)
                weights, loss, epoch_loss = local_model.update_final_layer(
                    model=local_train_model, global_round=epoch, client_idx=c)
                
                all_epoch_loss.append(epoch_loss)
                # all_epoch_loss.append(loss)
                # print("ep loss: ", epoch_loss)
                
                local_train_model.load_state_dict(weights)

                # # Test acc & fairness
                # local_dataset_test = dataset.get_dataset_from_df(dataset_name=args.dataset, df=local_set_ls[i].test_set, kaggle=args.kaggle) 
                # local_dataset_train =  dataset.get_dataset_from_df(dataset_name=args.dataset, df=local_set_ls[i].train_set, kaggle=args.kaggle) 

                fairness_metric = ["eod", "tpr", "fpr"]
                # pred_labels, accuracy, local_fairness, _,_ = update.get_prediction_w_local_fairness(args.gpu, local_train_model, \
                #                                                                                     X=local_dataset.local_test_set.X, Y=local_dataset.local_test_set.y,
                #                                                                                       a=local_dataset.local_test_set.a, fairness_metric=fairness_metric)
                
                acc, _, _, _, _, local_fairness = local_model.inference_w_fairness(model=local_train_model, set="test", fairness_metric=fairness_metric, client_idx=c)

                
                stat_dic['test_acc_new_ft'][c] = (acc)
                stat_dic['test_eod_new_ft'][c] = (local_fairness["eod"])
                stat_dic['test_tpr_new_ft'][c] = (local_fairness["tpr"])
                stat_dic['test_fpr_new_ft'][c] = (local_fairness["fpr"])

                # pred_labels, accuracy, local_fairness,_,_ = update.get_prediction_w_local_fairness(args.gpu, local_train_model, \
                #                                                                                    X=local_dataset.local_train_set.X, Y=local_dataset.local_train_set.y,
                #                                                                                       a=local_dataset.local_train_set.a, fairness_metric=fairness_metric)
                
                acc, _, _, _, _, local_fairness = local_model.inference_w_fairness(model=local_train_model, set="train", fairness_metric=fairness_metric,  client_idx=c)

                
                stat_dic['train_acc_new_ft'][c] = (acc)
                stat_dic['train_eod_new_ft'][c] = (local_fairness["eod"])
                stat_dic['train_tpr_new_ft'][c] = (local_fairness["tpr"])
                stat_dic['train_fpr_new_ft'][c] = (local_fairness["fpr"])

            all_epoch_loss = np.asarray(all_epoch_loss)
            # print("all_epoch_loss: ", all_epoch_loss.size, len(all_epoch_loss))
            # print(all_epoch_loss)

            plot_file_loss_ft = statistics_dir + "/ft_loss_plot.png"
            fig_titl_loss_ft = statistics_dir.split("/")[-1] + "_exp" + str(args.idx)

            plot.plot_loss_ft((all_epoch_loss), title=fig_titl_loss_ft, save_to=plot_file_loss_ft)
                    

        # print("stats dic: ")
        # print(stat_dic)
    
    print("time: ", int(time.time() - start_time), int(time.time()))
    if args.fl_fairfed:
        print("********* Start FairFed Training **********")

        # # Initiaize local models
        # local_models = []
        # for i in range(args.num_users):
        #     local_dataset = local_set_ls[idx]
        #     split_idxs = (local_dataset.train_set_idxs,local_dataset.test_set_idxs,local_dataset.val_set_idxs)
        #     local_model = LocalUpdate(args=args, split_idxs=split_idxs, dataset=train_dataset,
        #                             idxs=user_groups[idx], logger=logger)
        #     local_models.append(local_model)
        

        # For each (global) round of training
        # local_weights_fair = []
        print("time: ", int(time.time() - start_time), int(time.time()))

        # Reinitialize model
        img_size = train_dataset[0][0].shape
        global_model=models.get_model(args, img_size=img_size)

        # Set the model to train and send it to device.
        global_model.to(device)
        global_model.train()

        all_loss_epoch = []
        for epoch in tqdm(range(args.fairfed_ep)):
            local_losses = []
            local_weights, local_losses = [], []
            print(f'\n | Global Training Round : {epoch+1} |\n')

            # Comppute local fairness and accuracy
            local_fairness_ls = []
            prediction_ls = []
            # for i in range(args.num_users):
            #     local_set = local_set_ls[i]
            #     # local_dataset =  dataset.get_dataset_from_df(dataset_name=args.dataset, df=local_set_df, kaggle=args.kaggle)

            #     pred_labels, accuracy, local_fairness,_,_ = update.get_prediction_w_local_fairness(args.gpu, global_model, \
            #                                                                                        X=local_set.local_train_set.X, Y=local_set.local_train_set.y, \
            #                                                                                          a=local_set.local_train_set.a, fairness_metric="eod")
            #     local_fairness_ls.append(local_fairness["eod"])
            #     prediction_ls.append(pred_labels)

            # Compute global fairness
            

            # print("all_local_train_a **")
            # print(all_local_train_a) 
            # print("type:::", type(all_local_train_a))
            global_acc, global_fairness = update.get_global_fairness_new(all_local_train_a, all_local_train_y, all_local_train_pred, "eod", "train")
            local_fairness_ls = all_local_train_eod
            # global_acc, global_fairness = update.get_global_fairness(local_set_ls, prediction_ls, "eod", "train")

            # Compute weighted mean metric gap
            metric_gap = [abs(global_fairness - lf) for lf in local_fairness_ls]
            metric_gap_avg = np.mean(metric_gap)




            print("FairFed: Round ", epoch)
            print("time: ", int(time.time() - start_time), int(time.time()))

            global_model.train()
            # For each selected user do local_ep round of training
            local_weights = []
            for idx in range(args.num_users):
                local_dataset = local_set_ls[idx]
                split_idxs = (local_dataset.train_set_idxs,local_dataset.test_set_idxs,local_dataset.val_set_idxs)
                local_model = LocalUpdate(args=args, split_idxs=split_idxs, dataset=train_dataset,
                                        idxs=user_groups[idx], logger=logger)

                # Update local model parameters
                w, loss, _ = local_model.update_weights(
                    model=copy.deepcopy(global_model), global_round=epoch, client_idx=idx)
                
                local_weights.append(copy.deepcopy(w))
                local_losses.append(copy.deepcopy(loss))

            # print(local_weights[0].keys())
            # print(type(local_weights[0]['layer_1.weight']))

            client_update_weight = []
            for idx in range(args.num_users):
                client_update_weight.append(np.exp( - args.beta * (metric_gap[idx] - metric_gap_avg)) * local_set_ls[idx].size / train_dataset.size)
            
            
            new_weights = copy.deepcopy(local_weights[0])
            for key in new_weights.keys():
                new_weights[key] = torch.zeros(size=local_weights[0][key].shape)

            for key in new_weights.keys():
                for idx in range(args.num_users):
                    new_weights[key] += ((client_update_weight[idx] / np.sum(client_update_weight)) * local_weights[idx][key].cpu())

            # new_weights = new_weights.to(device)
            global_model.load_state_dict(new_weights)
                # # Compute weighted local weights using FairFed formula
                # if epoch == 0:
                #     local_weights_original = (copy.deepcopy(w))
                #     local_weights_fair.append(local_weights_original)
                # else:

                #     for key in local_weights_fair[idx].keys():
                #         local_weights_fair[idx][key] = local_weights_fair[idx][key] - args.beta * (metric_gap[idx] - metric_gap_avg)
                #     # local_weights_fair[idx] = local_weights_fair[idx] - args.beta * (metric_gap[idx] - metric_gap_avg)
                
                # # Not sure about this
                # # local_weights.append(local_weights_fair[idx] / sum(local_weights_fair))
                
                # local_losses.append(copy.deepcopy(loss))

            # update global weights
            # # global_weights = average_weights(local_weights)
            # global_weights = average_weights(local_weights_fair)

            # # update global weights
            # global_model.load_state_dict(global_weights)

            loss_avg = sum(local_losses) / len(local_losses)
            train_loss.append(loss_avg)

            # Actually it is local test accuracy
            # Calculate avg training accuracy over all users at every epoch
            list_acc, list_loss = [], []
            global_model.eval()

            print("start inference: FairFed train...")
            print("time: ", int(time.time() - start_time), int(time.time()))

            for c in range(args.num_users):
                local_dataset = local_set_ls[c]
                split_idxs = (local_dataset.train_set_idxs,local_dataset.test_set_idxs,local_dataset.val_set_idxs)
                local_model = LocalUpdate(args=args, split_idxs=split_idxs, dataset=train_dataset,
                                        idxs=user_groups[idx], logger=logger)

                # local_model = LocalUpdate(args=args, local_dataset=local_dataset, dataset=train_dataset,
                #                         idxs=user_groups[idx], logger=logger)
                
                acc, loss, all_y, all_a, all_pred, local_fairness = local_model.inference_w_fairness(model=global_model, set="train", fairness_metric="eod", client_idx=c)
                stat_dic["train_acc_fairfed"][c] = acc
                stat_dic["train_eod_fairfed"][c] = local_fairness["eod"]

                list_acc.append(acc)
                list_loss.append(loss)

                # all_loss_epoch.append(np.array(list_loss)/len(list_loss))
                
            train_accuracy.append(sum(list_acc)/len(list_acc))
            all_loss_epoch.append(local_losses)
            # print("all_loss_epoch")
            # print(all_loss_epoch)

            

            # print global training loss after every 'i' rounds
            if (epoch+1) % print_every == 0:
                print(f' \nAvg Training Stats after {epoch+1} global rounds:')
                print(f'Training Loss : {np.mean(np.array(train_loss))}')
                print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))


        # print("all_loss_epoch")
        # print(np.asarray(all_loss_epoch))
        all_loss_epoch = np.asarray(all_loss_epoch).T
        print(all_loss_epoch)
        plot_file_loss_fairfed = statistics_dir + "/fairfed_loss_plot.png"
        fig_titl_loss_fairfed = statistics_dir.split("/")[-1] + "_exp" + str(args.idx)
        plot.plot_loss_ft((all_loss_epoch), fairfed=True, title=fig_titl_loss_fairfed, save_to=plot_file_loss_fairfed)
        
        # Check FairFed weights
        weights_fedavg = global_model.state_dict()

        # print("weights_FairFed: ", weights_fedavg)
        with open(statistics_dir+"/weights.txt", "a") as w_file:
            w_file.write("FairFed weights: \n")
            w_file.write("final_layer.weight: \n" + str(weights_fedavg["final_layer.weight"]) +"\n")
            w_file.write("final_layer.bias: \n"+str(weights_fedavg["final_layer.bias"]) +"\n")


        print("start inference: FairFed test...")
        print("time: ", int(time.time() - start_time), int(time.time()))
        for c in range(args.num_users):
            local_dataset = local_set_ls[c]
            split_idxs = (local_dataset.train_set_idxs,local_dataset.test_set_idxs,local_dataset.val_set_idxs)
            local_model = LocalUpdate(args=args, split_idxs=split_idxs, dataset=train_dataset,
                                    idxs=user_groups[idx], logger=logger)

            # local_model = LocalUpdate(args=args, local_dataset=local_dataset, dataset=train_dataset,
            #                         idxs=user_groups[idx], logger=logger)
            
            acc, _, all_y, all_a, all_pred, local_fairness = local_model.inference_w_fairness(model=global_model, set="test", fairness_metric="eod", client_idx=c)
            stat_dic["test_acc_fairfed"][c] = acc
            stat_dic["test_eod_fairfed"][c] = local_fairness["eod"]


        
        # Evaluation locally after training
        # print("********* Start Local Evaluation and Post-processing **********")

        # stat_keys = ['train_acc_before','train_acc_after', 'test_acc_before', 'test_acc_after',
        #             'train_eod_before', 'train_eod_after', 'test_eod_before', 'test_eod_after',
        #             'train_fpr_before', 'train_fpr_after', 'test_fpr_before', 'test_fpr_after',
        #             'train_tpr_before', 'train_tpr_after', 'test_tpr_before', 'test_tpr_after']
        # stat_dic = {k: [0]*args.num_users for k in stat_keys}         

        # local_fairness_ls = []
        # local_acc_ls = []
        # for i in range(args.num_users):
        #     # local_set_df = local_set_ls[i].train_set
        #     local_set = local_set_ls[i]
        #     # local_dataset =  dataset.get_dataset_from_df(dataset_name=args.dataset, df=local_set_df, kaggle=args.kaggle)
        #     pred_labels, accuracy, local_fairness,_,_ = update.get_prediction_w_local_fairness(args.gpu, global_model, \
        #                                                                                         X=local_set.local_train_set.X, Y=local_set.local_train_set.y, \
        #                                                                                              a=local_set.local_train_set.a, fairness_metric="eod")
        #     local_fairness_ls.append(local_fairness["eod"])
        #     local_acc_ls.append(accuracy)
        
        # print("TRAIN SET ----")
        # print("local_fairness_ls")
        # print(local_fairness_ls)
        # print("local acc ls")
        # print(local_acc_ls)
        # stat_dic["train_acc_fairfed"] = local_acc_ls
        # stat_dic["train_eod_fairfed"] = local_fairness_ls

        
        # local_fairness_ls = []
        # local_acc_ls = []
        # for i in range(args.num_users):
        #     # local_set_df = local_set_ls[i].test_set
        #     local_set = local_set_ls[i]
        #     # local_dataset =  dataset.get_dataset_from_df(dataset_name=args.dataset, df=local_set_df, kaggle=args.kaggle)
        #     pred_labels, accuracy, local_fairness,_,_ = update.get_prediction_w_local_fairness(args.gpu, global_model, \
        #                                                                                         X=local_set.local_test_set.X, Y=local_set.local_test_set.y, \
        #                                                                                              a=local_set.local_test_set.a, fairness_metric="eod")
        #     local_fairness_ls.append(local_fairness["eod"])
        #     local_acc_ls.append(accuracy)
        
        # print("TEST SET ----")
        # print("local_fairness_ls")
        # print(local_fairness_ls)
        # print("local acc ls")
        # print(local_acc_ls)
        # stat_dic["test_acc_fairfed"] = local_acc_ls
        # stat_dic["test_eod_fairfed"] = local_fairness_ls



    # print(stat_dic)
    # for key in stat_dic.keys():
    #     print(key, len(stat_dic[key]))
    # print(stat_dic.shape)
    print("time: ", int(time.time() - start_time), int(time.time()))
    print("Start saving...")
    stat_df = pd.DataFrame(stat_dic)
    stat_df.to_csv(statistics_dir + "/stats.csv")

    with open(statistics_dir+'/client_datasets.pkl', 'wb') as outp:
        pickle.dump(local_set_ls, outp, pickle.HIGHEST_PROTOCOL)
    
    with open(statistics_dir+'/pred_train_dic.pkl', 'wb') as outp:
        pickle.dump(pred_train_dic, outp, pickle.HIGHEST_PROTOCOL)
    
    with open(statistics_dir+'/pred_test_dic.pkl', 'wb') as outp:
        pickle.dump(pred_test_dic, outp, pickle.HIGHEST_PROTOCOL)
    
    print("Exp stats saved in dir: ", statistics_dir)

    # df = pd.DataFrame()
    # df["acc_before"] = local_acc_ls
    # df["acc_after"] = local_acc_ls_debiased
    # df["eod_before"] = local_eod_ls
    # df["eod_after"] = local_eod_ls_debiased
    # df.to_csv(statistics_dir + "/test_acc_eod.csv")

    # plot_file = statistics_dir + "/test_acc_eod_plot.png"
    # plot.plot_acc_eod(local_acc_ls, local_acc_ls_debiased,local_eod_ls, local_eod_ls_debiased, save_to=plot_file)

    # plot_file_train = statistics_dir + "/train_acc_eod_plot.png"
    # plot.plot_acc_eod(acc_before=local_train_acc_ls, acc_after=local_train_acc_ls_debiased,eod_before=local_train_eod_ls, eod_after=local_train_eod_ls_debiased, save_to=plot_file_train)


    fig_title = statistics_dir.split("/")[-1] + "_exp" + str(args.idx)
    plot_file_all = statistics_dir + "/all_acc_eod_plot.png"
    # plot.plot_all(stat_dic, title=fig_title,
    #             save_to=plot_file_all)
    
    plot.plot_multi_exp(stat_dic, args, new=args.fl_new, plot_tpfp=args.plot_tpfp,
                        title=fig_title, save_to=plot_file_all)


   


    # # Apply post-processing locally at each client:
    # if args.fl == "new":
    #     print("********* Start Post-processing Locally **********")     
    #     for idx in idxs_users:
    #         idxs = user_groups[idx] 

    #         # Use local training dataset to fit post-processing method (local train set as "dummy val sets")
    #         idxs_train =  idxs[:int(0.8*len(idxs))]     # <------------ Hard code train index
    #         train_set_df = train_dataset.df[train_dataset.df.index.isin(idxs_train)]
    #         local_train_dataset =  dataset.AdultDataset(csv_file="", df=train_set_df)
    #         local_train_prediction, local_train_acc = update.get_prediction(args, global_model, local_train_dataset)
    #         train_bld_prediction_dataset = dataset.get_bld_dataset_w_pred(local_train_dataset, local_train_prediction)


    #         cost_constraint = args.post_proc_cost # "fpr" # "fnr", "fpr", "weighted"
    #         randseed = 12345679 

    #         cpp = CalibratedEqOddsPostprocessing(privileged_groups = privileged_groups,
    #                                         unprivileged_groups = unprivileged_groups,
    #                                         cost_constraint=cost_constraint,
    #                                         seed=randseed)
    #         cpp = cpp.fit(local_train_dataset.bld, train_bld_prediction_dataset)

    #         # Test set with original prediction
    #         local_train_dataset_bld_prediction_debiased = cpp.predict(train_bld_prediction_dataset)
    #         local_test_dataset_bld_prediction_debiased = cpp.predict(local_test_bld_prediction_dataset)

    


    '''
    # Test inference after completion of training
    test_acc, test_loss = test_inference(args, global_model, test_dataset)

    prediction_test, _ = update.get_prediction(args, global_model, test_dataset.X, test_dataset.y)
    # X=local_set.local_test_set.X, Y=local_set.local_test_set.y)
    # print("prediction_test")
    # print(prediction_test)
    

    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))
    '''

    # Saving the objects train_loss and train_accuracy:
    file_name = statistics_dir + '/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.\
        format(args.dataset, args.model, args.epochs, args.frac, args.iid,
               args.local_ep, args.local_bs)

    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)

    print(file_name, " saved!")

    print('\n Total Run Time: {0:0.4f} s'.format(time.time()-start_time))


    # print("Check local set: ")
    # print(local_set_ls[0].local_train_set.X)
    # print(local_set_ls[0].local_train_set.y)
    # print(local_set_ls[0].local_train_set.a)
    # print("Check global set: ")
    # print(train_dataset.a)
    
    # Check statistics of training set
    utils.check_train_test_split(args.num_users, pred_train_dic, pred_test_dic, save_dir=statistics_dir)

    # print(tracemalloc.get_traced_memory())

    # stopping the library
    # tracemalloc.stop()  
    

    # PLOTTING (optional)
    # import matplotlib
    # import matplotlib.pyplot as plt
    # matplotlib.use('Agg')

    # Plot Loss curve
    # plt.figure()
    # plt.title('Training Loss vs Communication rounds')
    # plt.plot(range(len(train_loss)), train_loss, color='r')
    # plt.ylabel('Training loss')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))
    #
    # # Plot Average Accuracy vs Communication rounds
    # plt.figure()
    # plt.title('Average Accuracy vs Communication rounds')
    # plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    # plt.ylabel('Average Accuracy')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))


if __name__ == '__main__':
    main()