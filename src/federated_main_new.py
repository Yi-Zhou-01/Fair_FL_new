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
import algorithm
# import tracemalloc
# from memory_profiler import profile

import json


from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.algorithms.postprocessing.calibrated_eq_odds_postprocessing import CalibratedEqOddsPostprocessing
from aif360.algorithms.postprocessing import EqOddsPostprocessing


# from dataset import get_dataset
# import dataset

# @profile
def main():
    start_time = time.time()

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
    statistics_dir = save_to_root+'/statistics/{}/{}_{}_{}_{}_frac{}_client{}_lr{}_ftlr{}_part{}_beta{}_ep{}_{}_{}_ftep_{}_bs{}_ftbs{}_fta_{}{}_{}'.\
        format(args.idx, all_fl, args.debias, args.dataset, args.model, args.frac, args.num_users,
               args.lr, args.ft_lr, args.partition_idx, args.beta, args.epochs, args.local_ep, args.fairfed_ep, args.ft_ep, args.local_bs, args.ft_bs, args.ft_alpha,args.ft_alpha2, args.rep)    # <------------- iid tobeadded
        # Save to files ...
        # TBA
    os.makedirs(statistics_dir, exist_ok=True)


    # define paths
    logger = SummaryWriter(statistics_dir + '/logs')



    with open(statistics_dir+'/args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # with open('commandline_args.txt', 'r') as f:
    #     args.__dict__ = json.load(f)

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

    if args.fair_rep:
        train_dataset_rep = dataset.fair_rep_dataset(train_dataset, local_set_ls, args.lbd)
        # train_dataset = train_dataset_rep
    
    # BUILD MODEL
    print("args.use_saved_model: ", args.use_saved_model)
    img_size = train_dataset[0][0].shape
    if args.use_saved_model != "":
        global_model  = torch.load(args.use_saved_model)
        print("Using saved FedAvg model: ", args.use_saved_model)
    else:
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
        if args.fair_rep:
            stat_keys += [ss+"_"+lm+"_"+"new_rep" for ss in set_split for lm in local_metrics]
            stat_keys += [ss+"_"+lm+"_"+"fedavg_rep" for ss in set_split for lm in local_metrics]
        if args.plot_tpfp:
            stat_keys += [ss+"_"+lm+"_"+"new" for ss in set_split for lm in  ["tpr", "fpr"]]
            stat_keys += [ss+"_"+lm+"_"+"fedavg" for ss in set_split for lm in  ["tpr", "fpr"]]
            if args.fair_rep:
                stat_keys += [ss+"_"+lm+"_"+"fedavg_rep" for ss in set_split for lm in  ["tpr", "fpr"]]
    if args.fl_fairfed:
        stat_keys += [ss+"_"+lm+"_"+"fairfed" for ss in set_split for lm in local_metrics]
        stat_keys += [ss+"_"+lm+"_"+"fairfed_rep" for ss in set_split for lm in local_metrics]
        stat_keys += [ss+"_"+lm+"_"+"fairfed" for ss in set_split for lm in  ["tpr", "fpr"]]
        stat_keys += [ss+"_"+lm+"_"+"fairfed_rep" for ss in set_split for lm in  ["tpr", "fpr"]]

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
    time_point_1 = time.time()
    if args.fl_new and ( args.use_saved_model == ""):
        print("********* Start FedAvg/New Training **********")

        global_model = algorithm.fedavg_train(args, global_model, local_set_ls, train_dataset, statistics_dir, user_groups, logger)

        global_model, stat_dic, pred_train_dic, pred_test_dic = algorithm.fedavg_inference(args, global_model, local_set_ls, train_dataset, stat_dic, user_groups, logger)
        
        print("check stat_dic: ")
        print(stat_dic["train_acc_fedavg"])

        if args.fair_rep:
            print("********* Start [Fair Rep] FedAvg/New Training **********")
            global_model_rep = algorithm.fedavg_train(args, global_model, local_set_ls, train_dataset_rep, statistics_dir, user_groups, logger)

            global_model_rep, stat_dic, pred_train_dic_rep, pred_test_dic_rep = algorithm.fedavg_inference(args, global_model_rep, local_set_ls, train_dataset_rep, stat_dic, user_groups, logger, fair_rep=True)


        # Save trained model
        if args.save_avg_model:
            torch.save(global_model, statistics_dir+"/fedavg_model.pt")
            print("FedAvg trained model saved in: ", (statistics_dir+"/fedavg_model.pt"))


        
        print("time: ", int(time.time() - start_time), int(time.time()))
        time_point_2 =  time.time()
        # Post-processing approach
        if "pp" in args.debias:
            # Apply post-processing locally at each client:
            print("******** Start post-processing ******** ")
            pred_train_dic['pred_labels_pp'] = [] # np.zeros(args.num_users) 
            pred_test_dic['pred_labels_pp'] = [] # np.zeros(args.num_users) 

            stat_dic, pred_train_dic, pred_test_dic = algorithm.post_processing(args, stat_dic, pred_train_dic, pred_test_dic, fair_rep=False)
           
            time_point_3 =  time.time()
            if args.fair_rep:
                pred_train_dic_rep['pred_labels_pp'] = []
                pred_test_dic_rep['pred_labels_pp'] = []
                stat_dic, pred_train_dic_rep, pred_test_dic_rep = algorithm.post_processing(args, stat_dic, pred_train_dic_rep, pred_test_dic_rep, fair_rep=True)
           
            
           
            # for idx in range(args.num_users):
            #     idxs = user_groups[idx]
            #     local_set = local_set_ls[idx]

            #     # train_bld_prediction_dataset = dataset.get_bld_dataset_w_pred(a=all_local_train_a[idx], pred_labels=all_local_train_pred[idx])
            #     # train_bld_original = dataset.get_bld_dataset_w_pred(a=all_local_train_a[idx], pred_labels=all_local_train_y[idx])
            #     # test_bld_prediction_dataset = dataset.get_bld_dataset_w_pred(a=all_local_test_a[idx], pred_labels=all_local_test_pred[idx])
            #     # test_bld_original = dataset.get_bld_dataset_w_pred(a=all_local_test_a[idx], pred_labels=all_local_test_y[idx])

            #     train_bld_prediction_dataset = dataset.get_bld_dataset_w_pred(a=pred_train_dic["s_attr"][idx], pred_labels=pred_train_dic["pred_labels_fedavg"][idx])
            #     train_bld_original = dataset.get_bld_dataset_w_pred(a=pred_train_dic["s_attr"][idx], pred_labels=pred_train_dic["labels"][idx])
            #     test_bld_prediction_dataset = dataset.get_bld_dataset_w_pred(a=pred_test_dic["s_attr"][idx], pred_labels=pred_test_dic["pred_labels_fedavg"][idx])
            #     test_bld_original = dataset.get_bld_dataset_w_pred(a=pred_test_dic["s_attr"][idx], pred_labels=pred_test_dic["labels"][idx])



            #     cost_constraint = args.post_proc_cost # "fpr" # "fnr", "fpr", "weighted"
            #     # cost_constraint = "fnr"
            #     randseed = 12345679 

            #     # Fit post-processing model
                
            #     # cpp = CalibratedEqOddsPostprocessing(privileged_groups = privileged_groups,
            #     #                                 unprivileged_groups = unprivileged_groups,
            #     #                                 cost_constraint=cost_constraint,
            #     #                                 seed=randseed)
                
            #     cpp = EqOddsPostprocessing(privileged_groups = privileged_groups,
            #                                     unprivileged_groups = unprivileged_groups,
            #                                     seed=randseed)
            #     cpp = cpp.fit(train_bld_original, train_bld_prediction_dataset)

                
            #     # Prediction after post-processing
            #     local_train_dataset_bld_prediction_debiased = cpp.predict(train_bld_prediction_dataset)
            #     local_test_dataset_bld_prediction_debiased = cpp.predict(test_bld_prediction_dataset)

            #     # Metrics after post-processing
            #     cm_pred_train_debiased = ClassificationMetric(train_bld_original, local_train_dataset_bld_prediction_debiased,
            #                 unprivileged_groups=unprivileged_groups,
            #                 privileged_groups=privileged_groups)
                
                
            #     pred_train_dic['pred_labels_pp'].append((np.asarray(local_train_dataset_bld_prediction_debiased.labels.flatten())))
            #     pred_test_dic['pred_labels_pp'].append((np.asarray(local_test_dataset_bld_prediction_debiased.labels.flatten())))
               
                
            #     # pred_train_dic['pred_labels_pp'].append(torch.from_numpy(np.asarray(local_train_dataset_bld_prediction_debiased.labels.flatten())))
            #     # pred_test_dic['pred_labels_pp'].append(torch.from_numpy(np.asarray(local_test_dataset_bld_prediction_debiased.labels.flatten())))
            #     # train_acc_new.append(cm_pred_train_debiased.accuracy())
            #     # train_eod_new.append(cm_pred_train_debiased.equalized_odds_difference())
            #     stat_dic['train_acc_new'][idx] = (cm_pred_train_debiased.accuracy())
            #     # stat_dic['train_eod_new'].append(cm_pred_train_debiased.average_abs_odds_difference())
            #     stat_dic['train_eod_new'][idx] = (cm_pred_train_debiased.equalized_odds_difference())
            #     if args.plot_tpfp:
            #         stat_dic['train_tpr_new'][idx] = (cm_pred_train_debiased.true_positive_rate_difference())
            #         stat_dic['train_fpr_new'][idx] = (cm_pred_train_debiased.false_positive_rate_difference())
            #     # stat_dic['train_eod_after'].append(cm_pred_train_debiased.average_abs_odds_difference())
            #     # stat_dic['train_fpr_new'].append(abs(cm_pred_train_debiased.difference(cm_pred_train_debiased.false_positive_rate)))
            #     # stat_dic['train_tpr_new'].append(abs( cm_pred_train_debiased.difference(cm_pred_train_debiased.true_positive_rate)))


            #     cm_pred_test_debiased = ClassificationMetric(test_bld_original, local_test_dataset_bld_prediction_debiased,
            #                 unprivileged_groups=unprivileged_groups,
            #                 privileged_groups=privileged_groups)

            #     # test_acc_new.append(cm_pred_test_debiased.accuracy())
            #     # test_eod_new.append(cm_pred_test_debiased.equalized_odds_difference())
            #     stat_dic['test_acc_new'][idx] = (cm_pred_test_debiased.accuracy())
            #     # stat_dic['test_eod_new'].append(cm_pred_test_debiased.average_abs_odds_difference())
            #     stat_dic['test_eod_new'][idx] = (cm_pred_test_debiased.equalized_odds_difference())
            #     if args.plot_tpfp:
            #         stat_dic['test_tpr_new'][idx] = (cm_pred_test_debiased.true_positive_rate_difference())
            #         stat_dic['test_fpr_new'][idx] = (cm_pred_test_debiased.false_positive_rate_difference())
            #     # stat_dic['test_eod_after'].append(cm_pred_test_debiased.average_abs_odds_difference())
            #     # stat_dic['test_fpr_new'].append(abs(cm_pred_test_debiased.difference(cm_pred_test_debiased.false_positive_rate)))
            #     # stat_dic['test_tpr_new'].append(abs( cm_pred_test_debiased.difference(cm_pred_test_debiased.true_positive_rate)))
        
        # print("stats dic: ")
        # print(stat_dic)

        print("time: ", int(time.time() - start_time), int(time.time()))
        # Apply final-layer fine-tuning
        ft_keys = ['test_acc_new_ft', 'test_eod_new_ft', 'test_tpr_new_ft','test_fpr_new_ft', \
                       'train_acc_new_ft','train_eod_new_ft','train_tpr_new_ft', 'train_fpr_new_ft']
        for k in ft_keys:
            stat_dic[k] = np.zeros(args.num_users) 
        
        time_point_4 =  time.time()

        if ("ft" in args.debias) and  (args.ft_ep != 0):
            
            print("******** Final layer fine-tuning ******** ")
            
            stat_dic = algorithm.fine_tuning(args, global_model, local_set_ls, train_dataset, stat_dic, user_groups, logger, statistics_dir, fair_rep=False)
        time_point_5 =  time.time()

       
    
        # print("stats dic: ")
        # print(stat_dic)
    
    print("time: ", int(time.time() - start_time), int(time.time()))
    if args.fl_fairfed and (args.fairfed_ep != 0):
        print("********* Start FairFed Training **********")


        print("time: ", int(time.time() - start_time), int(time.time()))

        # Reinitialize model
        img_size = train_dataset[0][0].shape
        global_model=models.get_model(args, img_size=img_size)

        global_model, stat_dic = algorithm.fairfed_train(args, global_model, local_set_ls, train_dataset, stat_dic, user_groups, logger, statistics_dir)

        if args.fair_rep:
            print("********* Start [Fair Rep] FairFed Training **********")
            img_size = train_dataset_rep[0][0].shape
            global_model_rep=models.get_model(args, img_size=img_size)
            global_model_rep, stat_dic = algorithm.fairfed_train(args, global_model_rep, local_set_ls, train_dataset_rep, stat_dic, user_groups, logger, statistics_dir, fair_rep=True)
         
    time_point_6=time.time()

    with open(statistics_dir +"/time.txt", "a") as w_file:
        w_file.write("******** FedAvg ********\n")
        w_file.write(str(time_point_2-time_point_1) + "\n")
        w_file.write("******** FedAvg + PP ********\n")
        w_file.write(str(time_point_3-time_point_1) + "\n")           
        w_file.write("******** FedAvg + FT ********\n")
        w_file.write(str((time_point_2-time_point_1) + (time_point_5-time_point_4)) + "\n")    
        w_file.write("******** FairFed ********\n")
        w_file.write(str(time_point_6-time_point_5) + "\n")        

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

    fig_title = statistics_dir.split("/")[-1] + "_exp" + str(args.idx)
    plot_file_all = statistics_dir + "/all_acc_eod_plot.png"
    # plot.plot_all(stat_dic, title=fig_title,
    #             save_to=plot_file_all)
    
    plot.plot_multi_exp(stat_dic, args, new=args.fl_new, plot_tpfp=args.plot_tpfp, 
                        plot_fed = (args.fairfed_ep != 0),
                        plot_ft = (args.ft_ep != 0),
                        title=fig_title, save_to=plot_file_all)



    # Saving the objects train_loss and train_accuracy:
    file_name = statistics_dir + '/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.\
        format(args.dataset, args.model, args.epochs, args.frac, args.iid,
               args.local_ep, args.local_bs)

    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)

    print(file_name, " saved!")

    print('\n Total Run Time: {0:0.4f} s'.format(time.time()-start_time))



    # Check statistics of training set
    utils.check_train_test_split(args.num_users, pred_train_dic, pred_test_dic, save_dir=statistics_dir)




if __name__ == '__main__':
    main()