import os
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
import copy
import torch


from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.algorithms.postprocessing.calibrated_eq_odds_postprocessing import CalibratedEqOddsPostprocessing
from aif360.algorithms.postprocessing import EqOddsPostprocessing

import utils
import plot
from update import LocalUpdate
import update
import dataset


def fedavg_train(args, global_model, local_set_ls, train_dataset, statistics_dir, user_groups, logger):
        # For each (global) round of training
        local_loss_all = []
        train_loss=[]
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
            global_weights = utils.average_weights(local_weights)

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
        
        return global_model
        


def fedavg_inference(args, global_model, local_set_ls, train_dataset, stat_dic, user_groups, logger, fair_rep=False):
        pred_train_dic = {}
        pred_test_dic = {}
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
            print("time: ", int(time.time()))
            local_dataset = local_set_ls[c]
            split_idxs = (local_dataset.train_set_idxs,local_dataset.test_set_idxs,local_dataset.val_set_idxs)
            local_model = LocalUpdate(args=args, split_idxs=split_idxs, dataset=train_dataset,
                                    idxs=user_groups[c], logger=logger)

            if args.plot_tpfp:
                fairness_metrics = ["eod","tpr","fpr"]
            else:
                fairness_metrics = ["eod"]

            acc, loss, all_y, all_a, all_pred, local_fairness = local_model.inference_w_fairness(model=global_model, set="train", fairness_metric=fairness_metrics, client_idx=c)

            # print("check all acc: ", acc)
            # print("check all_pred: ", sum(all_pred), len(all_pred))

            if fair_rep:
                stat_dic["train_acc_fedavg_rep"][c] = acc
                stat_dic["train_eod_fedavg_rep"][c] = local_fairness["eod"]
                if args.plot_tpfp:
                    stat_dic["train_tpr_fedavg_rep"][c] = local_fairness["tpr"]
                    stat_dic["train_fpr_fedavg_rep"][c] = local_fairness["fpr"]
            else:
                # print("check all acc NOT REP: ", acc)
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

            pred_train_dic["pred_labels_fedavg"][c] = (np.asarray(all_pred))
            pred_train_dic["labels"][c] = (np.asarray(all_y))
            pred_train_dic["s_attr"][c] =(np.asarray(all_a))


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
                                    idxs=user_groups[c], logger=logger)
            if args.plot_tpfp:
                fairness_metrics = ["eod","tpr","fpr"]
            else:
                fairness_metrics = ["eod"]
            acc, loss, all_y, all_a, all_pred, local_fairness = local_model.inference_w_fairness(model=global_model, set="test", fairness_metric=fairness_metrics, client_idx=c)
            
            if fair_rep:
                stat_dic["test_acc_fedavg_rep"][c] = acc
                stat_dic["test_eod_fedavg_rep"][c] = local_fairness["eod"]
                if args.plot_tpfp:
                    stat_dic["test_tpr_fedavg_rep"][c] = local_fairness["tpr"]
                    stat_dic["test_fpr_fedavg_rep"][c] = local_fairness["fpr"]
            
            else:
                stat_dic["test_acc_fedavg"][c] = acc
                stat_dic["test_eod_fedavg"][c] = local_fairness["eod"]
                if args.plot_tpfp:
                    stat_dic["test_tpr_fedavg"][c] = local_fairness["tpr"]
                    stat_dic["test_fpr_fedavg"][c] = local_fairness["fpr"]
            
            pred_test_dic["pred_labels_fedavg"][c] = np.array([])
            pred_test_dic["labels"][c] = np.array([])
            pred_test_dic["s_attr"][c] = np.array([])

            pred_test_dic["pred_labels_fedavg"][c]  = (np.asarray(all_pred))
            pred_test_dic["labels"][c]  = (np.asarray(all_y))
            pred_test_dic["s_attr"][c]  =  (np.asarray(all_a))

            all_local_test_y.append(all_y)
            all_local_test_a.append(all_a)
            all_local_test_pred.append(all_pred)

        print("---- stat_dic ----: test_acc_fedavg | test_eod_fedavg")
        print(stat_dic["test_acc_fedavg"])
        print(stat_dic["test_eod_fedavg"])

        print("---- stat_dic ----: train_acc_fedavg | train_eod_fedavg")
        print(stat_dic["train_acc_fedavg"])
        print(stat_dic["train_eod_fedavg"])

        return global_model, stat_dic, pred_train_dic, pred_test_dic


def fairfed_train(args, global_model, local_set_ls, train_dataset, stat_dic, user_groups, logger, statistics_dir, fair_rep=False):


    # Set the model to train and send it to device.
    device = 'cuda' if args.gpu else 'cpu'

    global_model.to(device)
    global_model.train()

    all_loss_epoch = []
    train_loss, train_accuracy=[], []
    global_fairness_ls = []

    for epoch in tqdm(range(args.fairfed_ep)):
        local_losses = []
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        # Comppute local fairness and accuracy
        local_fairness_ls = []
        prediction_ls = []


        print("FairFed: Round ", epoch)
        print("time: ", int(time.time()))

        global_model.train()
        # For each selected user do local_ep round of training
        local_weights = []
        all_local_train_a = []
        all_local_train_y = []
        all_local_train_pred = []
        all_local_train_eod = []
        list_acc, list_loss = [], []
        for idx in range(args.num_users):
            local_dataset = local_set_ls[idx]
            split_idxs = (local_dataset.train_set_idxs,local_dataset.test_set_idxs,local_dataset.val_set_idxs)
            local_model = LocalUpdate(args=args, split_idxs=split_idxs, dataset=train_dataset,
                                    idxs=user_groups[idx], logger=logger)
            
            w, loss, _ = local_model.update_weights(model=copy.deepcopy(global_model), global_round=epoch, client_idx=idx)
            
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

            acc, loss, all_y, all_a, all_pred, local_fairness = local_model.inference_w_fairness(model=global_model, set="train", fairness_metric=["eod"], client_idx=idx)

            all_local_train_a.append(all_a)
            all_local_train_y.append(all_y)
            all_local_train_pred.append(all_pred)
            all_local_train_eod.append(local_fairness["eod"])

            list_acc.append(acc)
            list_loss.append(loss)

            if fair_rep:
                stat_dic["train_acc_fairfed_rep"][idx] = acc
                stat_dic["train_eod_fairfed_rep"][idx] = local_fairness["eod"]
            else:
                stat_dic["train_acc_fairfed"][idx] = acc
                stat_dic["train_eod_fairfed"][idx] = local_fairness["eod"]

            
        train_accuracy.append(sum(list_acc)/len(list_acc))
        all_loss_epoch.append(local_losses)
        # print("all_loss_epoch")
        # print(all_loss_epoch)


        # print global training loss after every 'i' rounds
        if (epoch+1) % 1 == 0:
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))


            # Update local model parameters


        # print(local_weights[0].keys())
        # print(type(local_weights[0]['layer_1.weight']))
    

        global_acc, global_fairness = update.get_global_fairness_new(all_local_train_a, all_local_train_y, all_local_train_pred, "eod", "train")
        local_fairness_ls = all_local_train_eod
        global_fairness_ls.append(global_fairness)
        # global_acc, global_fairness = update.get_global_fairness(local_set_ls, prediction_ls, "eod", "train")

        # Compute weighted mean metric gap
        metric_gap = [abs(global_fairness - lf) for lf in local_fairness_ls]
        metric_gap_avg = np.mean(metric_gap)




        client_update_weight = []
        for idx in range(args.num_users):
            client_update_weight.append(np.exp( - args.beta * (metric_gap[idx] - metric_gap_avg)) * local_set_ls[idx].size / train_dataset.size)
        
        
        new_weights = copy.deepcopy(local_weights[0])
        for key in new_weights.keys():
            new_weights[key] = torch.zeros(size=local_weights[0][key].shape)

        for key in new_weights.keys():
            for idx in range(args.num_users):
                new_weights[key] += ((client_update_weight[idx] / np.sum(client_update_weight)) * local_weights[idx][key].cpu())



        with open(statistics_dir+"/global_fairness_ls.txt", "a") as w_file:
            w_file.write("global_fairness: "+str(global_fairness) +"\n")
            w_file.write("local_fairness_ls: "+str(local_fairness_ls) +"\n")
            w_file.write("client_update_weight: "+str(client_update_weight) +"\n")
            w_file.write("local_weights: "+str(local_weights) +"\n")
            w_file.write("new_weights: "+str(new_weights) +"\n")
        

        # new_weights = new_weights.to(device)
        global_model.load_state_dict(new_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Actually it is local test accuracy
        # Calculate avg training accuracy over all users at every epoch
        
        global_model.eval()

        # print("start inference: FairFed train...")
        # print("time: ", int(time.time()))

        # for c in range(args.num_users):
        #     local_dataset = local_set_ls[c]
        #     split_idxs = (local_dataset.train_set_idxs,local_dataset.test_set_idxs,local_dataset.val_set_idxs)
        #     local_model = LocalUpdate(args=args, split_idxs=split_idxs, dataset=train_dataset,
        #                             idxs=user_groups[idx], logger=logger)


        #     acc, loss, all_y, all_a, all_pred, local_fairness = local_model.inference_w_fairness(model=global_model, set="train", fairness_metric="eod", client_idx=c)
        #     stat_dic["train_acc_fairfed"][c] = acc
        #     stat_dic["train_eod_fairfed"][c] = local_fairness["eod"]

        #     list_acc.append(acc)
        #     list_loss.append(loss)
            
        # train_accuracy.append(sum(list_acc)/len(list_acc))
        # all_loss_epoch.append(local_losses)
        # # print("all_loss_epoch")
        # # print(all_loss_epoch)


        # # print global training loss after every 'i' rounds
        # if (epoch+1) % 1 == 0:
        #     print(f' \nAvg Training Stats after {epoch+1} global rounds:')
        #     print(f'Training Loss : {np.mean(np.array(train_loss))}')
        #     print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))


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
    print("time: ", int(time.time()))
    for c in range(args.num_users):
        local_dataset = local_set_ls[c]
        split_idxs = (local_dataset.train_set_idxs,local_dataset.test_set_idxs,local_dataset.val_set_idxs)
        local_model = LocalUpdate(args=args, split_idxs=split_idxs, dataset=train_dataset,
                                idxs=user_groups[idx], logger=logger)

        # local_model = LocalUpdate(args=args, local_dataset=local_dataset, dataset=train_dataset,
        #                         idxs=user_groups[idx], logger=logger)
        
        acc, _, all_y, all_a, all_pred, local_fairness = local_model.inference_w_fairness(model=global_model, set="test", fairness_metric="eod", client_idx=c)
        
        if fair_rep:
            stat_dic["test_acc_fairfed_rep"][c] = acc
            stat_dic["test_eod_fairfed_rep"][c] = local_fairness["eod"]
        else:    
            print("check eod fairfed: ", local_fairness["eod"] )
            stat_dic["test_acc_fairfed"][c] = acc
            stat_dic["test_eod_fairfed"][c] = local_fairness["eod"]
    
    print("+++ global_fairness +++")
    print(global_fairness_ls)

    with open(statistics_dir+"/global_fairness_ls.txt", "a") as w_file:
        w_file.write("\n")
        w_file.write("+++ global_fairness +++ \n")
        w_file.write(str(global_fairness_ls) +"\n")
       

    return global_model, stat_dic



def post_processing(args, stat_dic, pred_train_dic, pred_test_dic, fair_rep=False):

    privileged_groups = [{"a": 1}]
    unprivileged_groups = [{"a": 0}]

    for idx in range(args.num_users):

        train_bld_prediction_dataset = dataset.get_bld_dataset_w_pred(a=pred_train_dic["s_attr"][idx], pred_labels=pred_train_dic["pred_labels_fedavg"][idx])
        train_bld_original = dataset.get_bld_dataset_w_pred(a=pred_train_dic["s_attr"][idx], pred_labels=pred_train_dic["labels"][idx])
        test_bld_prediction_dataset = dataset.get_bld_dataset_w_pred(a=pred_test_dic["s_attr"][idx], pred_labels=pred_test_dic["pred_labels_fedavg"][idx])
        test_bld_original = dataset.get_bld_dataset_w_pred(a=pred_test_dic["s_attr"][idx], pred_labels=pred_test_dic["labels"][idx])

        cost_constraint = args.post_proc_cost # "fpr" # "fnr", "fpr", "weighted"
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
        
        if fair_rep:
            stat_dic['train_acc_new_rep'][idx] = (cm_pred_train_debiased.accuracy())
            stat_dic['train_eod_new_rep'][idx] = (cm_pred_train_debiased.equalized_odds_difference())

        else:
            stat_dic['train_acc_new'][idx] = (cm_pred_train_debiased.accuracy())
            stat_dic['train_eod_new'][idx] = (cm_pred_train_debiased.equalized_odds_difference())
            if args.plot_tpfp:
                stat_dic['train_tpr_new'][idx] = (cm_pred_train_debiased.true_positive_rate_difference())
                stat_dic['train_fpr_new'][idx] = (cm_pred_train_debiased.false_positive_rate_difference())

        cm_pred_test_debiased = ClassificationMetric(test_bld_original, local_test_dataset_bld_prediction_debiased,
                    unprivileged_groups=unprivileged_groups,
                    privileged_groups=privileged_groups)

        if fair_rep:
            stat_dic['test_acc_new_rep'][idx] = (cm_pred_test_debiased.accuracy())
            stat_dic['test_eod_new_rep'][idx] = (cm_pred_test_debiased.equalized_odds_difference())
        else:
            stat_dic['test_acc_new'][idx] = (cm_pred_test_debiased.accuracy())
            stat_dic['test_eod_new'][idx] = (cm_pred_test_debiased.equalized_odds_difference())
            if args.plot_tpfp:
                stat_dic['test_tpr_new'][idx] = (cm_pred_test_debiased.true_positive_rate_difference())
                stat_dic['test_fpr_new'][idx] = (cm_pred_test_debiased.false_positive_rate_difference())

    return  stat_dic, pred_train_dic, pred_test_dic


def fine_tuning(args, global_model, local_set_ls, train_dataset, stat_dic, user_groups, logger, statistics_dir, fair_rep=False):

    all_epoch_loss = []
    for c in range(args.num_users):
        local_dataset = local_set_ls[c]
        split_idxs = (local_dataset.train_set_idxs,local_dataset.test_set_idxs,local_dataset.val_set_idxs)
        local_model = LocalUpdate(args=args, split_idxs=split_idxs, dataset=train_dataset,
                                idxs=user_groups[c], logger=logger)

        local_train_model = copy.deepcopy(global_model)
        weights, loss, epoch_loss = local_model.update_final_layer(
            model=local_train_model, global_round=args.epochs, client_idx=c)
        
        all_epoch_loss.append(epoch_loss)
        
        local_train_model.load_state_dict(weights)

        fairness_metric = ["eod", "tpr", "fpr"]

        acc, _, _, _, _, local_fairness = local_model.inference_w_fairness(model=local_train_model, set="train", fairness_metric=fairness_metric,  client_idx=c)

        stat_dic['train_acc_new_ft'][c] = (acc)
        stat_dic['train_eod_new_ft'][c] = (local_fairness["eod"])
        stat_dic['train_tpr_new_ft'][c] = (local_fairness["tpr"])
        stat_dic['train_fpr_new_ft'][c] = (local_fairness["fpr"])

        acc, _, _, _, _, local_fairness = local_model.inference_w_fairness(model=local_train_model, set="test", fairness_metric=fairness_metric, client_idx=c)
        
        stat_dic['test_acc_new_ft'][c] = (acc)
        stat_dic['test_eod_new_ft'][c] = (local_fairness["eod"])
        stat_dic['test_tpr_new_ft'][c] = (local_fairness["tpr"])
        stat_dic['test_fpr_new_ft'][c] = (local_fairness["fpr"])
    
    print("---- stat_dic ----: train_acc_new_ft | train_eod_new_ft")
    print(stat_dic["train_acc_new_ft"])
    print(stat_dic["train_eod_new_ft"])
    
    all_epoch_loss = np.asarray(all_epoch_loss)

    plot_file_loss_ft = statistics_dir + "/ft_loss_plot.png"
    fig_titl_loss_ft = statistics_dir.split("/")[-1] + "_exp" + str(args.idx)

    plot.plot_loss_ft((all_epoch_loss), title=fig_titl_loss_ft, save_to=plot_file_loss_ft)
    
    return stat_dic


