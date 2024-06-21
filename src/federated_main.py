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

import torch
from tensorboardX import SummaryWriter
# from torch.utils.tensorboard import SummaryWriter

import options
import dataset

# from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, MLPAdult, CNNMnist, CNNFashion_Mnist, CNNCifar
from utils import average_weights, exp_details
import os
import update


from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.algorithms.postprocessing.calibrated_eq_odds_postprocessing import CalibratedEqOddsPostprocessing


# from dataset import get_dataset
# import dataset

if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = options.args_parser()
    exp_details(args)

    # if args.gpu_id:
    #     torch.cuda.set_device(args.gpu_id)
    if args.gpu:
        torch.cuda.set_device(args.gpu)
    device = 'cuda' if args.gpu else 'cpu'

    # load dataset and user groups
    # user groups: different client datasets
    train_dataset, test_dataset, user_groups = dataset.get_dataset(args)

    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)

    elif args.model == 'mlp':
        # Multi-layer preceptron
        if args.dataset == 'adult':
            img_size = train_dataset[0][0].shape
            # print("img size: ", img_size)
            len_in = 1
            for x in img_size:
                len_in *= x
                global_model = MLPAdult(dim_in=len_in, dim_hidden=64,
                                dim_out=args.num_classes)

        else:
            img_size = train_dataset[0][0].shape
            len_in = 1
            for x in img_size:
                len_in *= x
                global_model = MLP(dim_in=len_in, dim_hidden=64,
                                dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')

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

    if args.fl == "new" or  args.fl == "fedavg":

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
                local_model = LocalUpdate(args=args, dataset=train_dataset,
                                        idxs=user_groups[idx], logger=logger)
                
                # print fairness eval
                if epoch == 0:
                    local_model.fairness_eval_spd()

                w, loss = local_model.update_weights(
                    model=copy.deepcopy(global_model), global_round=epoch)
                local_weights.append(copy.deepcopy(w))
                local_losses.append(copy.deepcopy(loss))

            # update global weights
            global_weights = average_weights(local_weights)

            # update global weights
            global_model.load_state_dict(global_weights)

            loss_avg = sum(local_losses) / len(local_losses)
            train_loss.append(loss_avg)

            # Actually it is local test accuracy
            # Calculate avg training accuracy over all users at every epoch
            list_acc, list_loss = [], []
            global_model.eval()
            for c in range(args.num_users):
                local_model = LocalUpdate(args=args, dataset=train_dataset,
                                        idxs=user_groups[idx], logger=logger)
                acc, loss = local_model.inference(model=global_model)
                list_acc.append(acc)
                list_loss.append(loss)
            train_accuracy.append(sum(list_acc)/len(list_acc))

            # print global training loss after every 'i' rounds
            if (epoch+1) % print_every == 0:
                print(f' \nAvg Training Stats after {epoch+1} global rounds:')
                print(f'Training Loss : {np.mean(np.array(train_loss))}')
                print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))
        
        # Evaluation locally after training
        print("********* Start Local Evaluation and Post-processing **********")
        local_acc_ls = []
        local_eod_ls = []

        local_acc_ls_debiased = []
        local_eod_ls_debiased = []


        for idx in idxs_users:
            idxs = user_groups[idx]
            idxs_test = idxs[int(0.9*len(idxs)):]        # <------------ Hard code index 
            idxs_train =  idxs[:int(0.8*len(idxs))]      # <------------ Hard code index 
            
            test_set_df = train_dataset.df[train_dataset.df.index.isin(idxs_test)]
            local_test_dataset =  dataset.AdultDataset(csv_file="", df=test_set_df)
            local_test_prediction, local_acc = update.get_prediction(args, global_model, local_test_dataset)
            local_test_bld_prediction_dataset = dataset.get_bld_dataset_w_pred(local_test_dataset, local_test_prediction)

            privileged_groups = [{train_dataset.s_attr: 1}]
            unprivileged_groups = [{train_dataset.s_attr: 0}]

            cm_pred_test = ClassificationMetric(local_test_dataset.bld, local_test_bld_prediction_dataset,
                            unprivileged_groups=unprivileged_groups,
                            privileged_groups=privileged_groups)
            abs_odds = cm_pred_test.average_abs_odds_difference()
            fp_diff, tp_diff = cm_pred_test.difference(cm_pred_test.false_positive_rate), cm_pred_test.difference(cm_pred_test.true_positive_rate)
            
            local_acc = cm_pred_test.accuracy()   #   --------  Or use this accuracy
            
            # More statistics could be added
            local_acc_ls.append(local_acc)
            local_eod_ls.append(abs_odds)


            # Apply post-processing locally at each client:
            if args.fl == "new":
                # Post-processing with local dataset
                train_set_df = train_dataset.df[train_dataset.df.index.isin(idxs_train)]
                local_train_dataset =  dataset.AdultDataset(csv_file="", df=train_set_df)
                local_train_prediction, local_train_acc = update.get_prediction(args, global_model, local_train_dataset)
                train_bld_prediction_dataset = dataset.get_bld_dataset_w_pred(local_train_dataset, local_train_prediction)
                
                cost_constraint = args.post_proc_cost # "fpr" # "fnr", "fpr", "weighted"
                randseed = 12345679 

                cpp = CalibratedEqOddsPostprocessing(privileged_groups = privileged_groups,
                                                unprivileged_groups = unprivileged_groups,
                                                cost_constraint=cost_constraint,
                                                seed=randseed)
                cpp = cpp.fit(local_train_dataset.bld, train_bld_prediction_dataset)

                # Test set with original prediction
                local_train_dataset_bld_prediction_debiased = cpp.predict(train_bld_prediction_dataset)
                local_test_dataset_bld_prediction_debiased = cpp.predict(local_test_bld_prediction_dataset)

                cm_pred_test_debiased = ClassificationMetric(local_test_dataset.bld, local_test_dataset_bld_prediction_debiased,
                            unprivileged_groups=unprivileged_groups,
                            privileged_groups=privileged_groups)
                abs_odds_debiased = cm_pred_test_debiased.average_abs_odds_difference()
                fp_diff_debiased, tp_diff_debiased = cm_pred_test_debiased.difference(cm_pred_test_debiased.false_positive_rate), cm_pred_test_debiased.difference(cm_pred_test_debiased.true_positive_rate)
                local_test_acc_debiased = cm_pred_test_debiased.accuracy()

                local_acc_ls_debiased.append(local_test_acc_debiased)
                local_eod_ls_debiased.append(abs_odds_debiased)


        print("Before post-processing ...")
        print(local_acc_ls)
        print(local_eod_ls)
        print("After post-processing ...")
        print(local_acc_ls_debiased)
        print(local_eod_ls_debiased)

        # Save to files ...
        # TBA


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

    



    # Test inference after completion of training
    test_acc, test_loss = test_inference(args, global_model, test_dataset)

    prediction_test, _ = update.get_prediction(args, global_model, test_dataset)
    # print("prediction_test")
    # print(prediction_test)
    

    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

    # Saving the objects train_loss and train_accuracy:
    file_name = os.getcwd() + '/save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.\
        format(args.dataset, args.model, args.epochs, args.frac, args.iid,
               args.local_ep, args.local_bs)

    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)

    print(file_name, " saved!")

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

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
