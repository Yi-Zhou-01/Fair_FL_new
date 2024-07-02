#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import dataset
import pandas as pd

from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
from aif360.datasets import BinaryLabelDataset

from sklearn.model_selection import train_test_split


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


class AdultDatasetWrapDf(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset_df):
        self.dataset_df = dataset_df.reset_index(drop=True)

        # print(self.dataset_df[:5])

    def __len__(self):
        return len(self.dataset_df)

    def __getitem__(self, idx):
        image, label = self.dataset_df.drop('income', axis=1).iloc[idx], self.dataset_df['income'].iloc[idx]
        return torch.tensor(image).to(torch.float32), torch.tensor(label).to(torch.float32)


class LocalDataset(object):
    def __init__(self, dataset, local_idxs, test_ratio=0.2):
        
        self.local_idxs = local_idxs
        self.local_dataset =  dataset.df[dataset.df.index.isin(local_idxs)]
        self.target_label = dataset.target


        self.test_ratio = test_ratio
        self.train_set, self.test_set, self.val_set, self.train_set_idxs, self.test_set_idxs, self.val_set_idxs  = self.train_test_split()
        
        self.train_len = len(self.train_set_idxs)
        self.test_len = len(self.test_set_idxs)
        self.val_len = len(self.val_set_idxs)
        # self.train_set, self.test_set, self.val_set = self.train_test_split()
        

    # Return df
    def train_test_split(self):

        # print("Local dataset")
        # print(self.local_dataset[:5].index)
        # print(self.local_dataset[-5:].index)

        
        X = self.local_dataset.drop(self.target_label, axis=1)
        y = self.local_dataset[self.target_label]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_ratio, stratify=y)

        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, stratify=y_test)

        X_train[self.target_label] = y_train
        X_test[self.target_label] = y_test
        X_val[self.target_label] = y_val

        # print("Local Train")
        # print(X_train[:5].index)

        # print("Local Test")
        # print(X_test[:5].index)

        return X_train, X_test, X_val, list(X_train.index), list(X_test.index), list(X_val.index)




class LocalUpdate(object):
    def __init__(self, args, split_idxs, dataset, idxs, logger,local_dataset=None):
        self.args = args
        self.logger = logger
        self.local_dataset = local_dataset

        # self.trainloader, self.validloader, self.testloader = self.train_val_test(dataset, list(idxs))
        self.trainloader, self.validloader, self.testloader = self.split_w_idxs(dataset, split_idxs)
        self.device = 'cuda' if args.gpu else 'cpu'
        # Default criterion set to NLL loss function
        # self.criterion = nn.NLLLoss().to(self.device)
        self.criterion = torch.nn.BCEWithLogitsLoss().to(self.device)
        self.dataset = dataset
    
    def split_w_idxs(self, dataset, idxs):
        train_idxs, test_idxs, val_idxs = idxs

        trainloader = DataLoader(DatasetSplit(dataset, train_idxs),
                                 batch_size=self.args.local_bs, shuffle=True)
        validloader = DataLoader(DatasetSplit(dataset, val_idxs),
                                 batch_size=int(len(val_idxs)/10), shuffle=False)
        testloader = DataLoader(DatasetSplit(dataset, test_idxs),
                                batch_size=int(len(test_idxs)/10), shuffle=False)
        
        return trainloader, validloader, testloader


        

    def train_val_test(self, local_dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """

        # trainloader = DataLoader(DatasetSplit(dataset, local_dataset.train_set_idxs),
        #                          batch_size=self.args.local_bs, shuffle=True)
        # validloader = DataLoader(DatasetSplit(dataset, local_dataset.val_set_idxs),
        #                          batch_size=int(len(idxs_val)/10), shuffle=False)
        # testloader = DataLoader(DatasetSplit(dataset, local_dataset.test_set_idxs),
        #                         batch_size=int(len(idxs_test)/10), shuffle=False)

        # -----------------------------------------
        trainloader = DataLoader(AdultDatasetWrapDf(self.local_dataset.train_set),
                                 batch_size=self.args.local_bs, shuffle=True)
        testloader = DataLoader(AdultDatasetWrapDf(self.local_dataset.test_set),
                                 batch_size=int(len(self.local_dataset.test_set)/10), shuffle=True)
        validloader = DataLoader(AdultDatasetWrapDf(self.local_dataset.val_set),
                                 batch_size=int(len(self.local_dataset.val_set)/10), shuffle=True)


        # -----------------------------------------

        # # split indexes for train, validation, and test (80, 10, 10)
        # idxs_train = idxs[:int(0.8*len(idxs))]
        # idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        # idxs_test = idxs[int(0.9*len(idxs)):]

        # # print("++++++++++ Local dataset statistics +++++++++++++++++++")
        # # target_ls = dataset.y
        # # print("Train: ", )

        # trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
        #                          batch_size=self.args.local_bs, shuffle=True)
        # validloader = DataLoader(DatasetSplit(dataset, idxs_val),
        #                          batch_size=int(len(idxs_val)/10), shuffle=False)
        # testloader = DataLoader(DatasetSplit(dataset, idxs_test),
        #                         batch_size=int(len(idxs_test)/10), shuffle=False)
        return trainloader, validloader, testloader

    def update_weights(self, model, global_round):
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                # log_probs = model(images)
                log_probs = model(images).squeeze()
                log_probs = log_probs.reshape(-1)
                loss = self.criterion(log_probs, labels)
                # loss = self.criterion(log_probs, labels.long())
                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 10 == 0):
                    if batch_idx % 50 == 0 and (iter<10 or iter%10 ==0):
                    # if self.args.local_ep <= 10 or (self.args.local_ep <=100 and self.args.local_ep % 10 == 0) or (self.args.local_ep % 50 == 0):
                        print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]  \tLoss: {:.6f}'.format(
                            global_round, iter, batch_idx * len(images),
                            len(self.trainloader.dataset),
                            100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def inference(self, model):
        """ Returns the inference accuracy and loss of local test data (?). 
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.validloader):
        # for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            # outputs = model(images)
            # batch_loss = self.criterion(outputs, labels)
            # batch_loss = self.criterion(outputs, labels.long())
            outputs = model(images).squeeze()
            outputs = outputs.reshape(-1)

            try:
                batch_loss = self.criterion(outputs, labels)
            except:
                print("DEBUG ====== ")
                print(outputs, outputs.size())
                print(labels,  labels.size())
                # print()
                print(len(outputs))
                print(len(labels))
            loss += batch_loss.item()

            # Prediction
            # _, pred_labels = torch.max(outputs, 1)
            pred_labels = torch.tensor([ int(pred >= 0.5) for pred in outputs])
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)


        # print("Inference pred labels: ", correct, "/", total, sum(labels))
        accuracy = correct/total
        return accuracy, loss
    
    # def fairness_eval_spd(self, set="train", target_label="income", s_attr="sex_1"):
    #     privileged_groups = [{s_attr: 1}]
    #     unprivileged_groups = [{s_attr: 0}]

    #     # statistical_parity_difference of local train data
    #     if set == "train":
    #         subset_df = self.dataset.df[self.dataset.df.index.isin(self.train_idx)]
    #     elif set == "test":
    #         subset_df = self.dataset.df[self.dataset.df.index.isin(self.test_idx)]
    #     dataset_bld = BinaryLabelDataset(df=subset_df, label_names=[target_label], protected_attribute_names=[s_attr])
        
    #     metric_orig_train = BinaryLabelDatasetMetric(dataset_bld, 
    #                             unprivileged_groups=unprivileged_groups,
    #                             privileged_groups=privileged_groups)
        
    #     print("Statistical parity difference= %f" % metric_orig_train.mean_difference())
    #     # print("Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_orig_train.mean_difference())
    #     print("pass")
    

    # def fairness_eval_eod(self, model):


    #     print("pass")


def test_inference(args, model, test_dataset):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = 'cuda' if args.gpu else 'cpu'
    # criterion = nn.NLLLoss().to(device)
    criterion = torch.nn.BCEWithLogitsLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images).squeeze()
        # batch_loss = criterion(outputs, labels)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        # _, pred_labels = torch.max(outputs, 1)
        pred_labels = torch.tensor([ int(pred >= 0.5) for pred in outputs])
        # pred_labels = outputs
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    return accuracy, loss

def get_prediction(args, model, test_dataset):

    """ Returns the test accuracy and loss.
    """

    model.eval()
    device = 'cuda' if args.gpu else 'cpu'
    # criterion = nn.NLLLoss().to(device)

    X_tensor = torch.tensor(test_dataset.X)
    Y_tensor  = torch.tensor(test_dataset.y)
    images, labels = X_tensor.to(device), Y_tensor.to(device)
    outputs = model(images).squeeze()
    pred_labels = [ int(pred >= 0.5) for pred in outputs]
  
    pred_labels = pred_labels = torch.tensor([ int(pred >= 0.5) for pred in outputs])
    pred_labels = pred_labels.view(-1)
    # print("Len of pred_labels set: ", len(pred_labels))
    # print(pred_labels)

    correct = torch.sum(torch.eq(pred_labels, labels)).item()
    print("predicted 1s / real 1s/ sample size: ", torch.sum(pred_labels), " / ", torch.sum(labels) ," / ", len(pred_labels))
    accuracy = correct/len(pred_labels)

    return pred_labels, accuracy


def get_prediction_w_local_fairness(gpu, model, test_dataset, metric="eod"):
    
    """ Returns the test accuracy and loss.
    """

    model.eval()
    device = 'cuda' if gpu else 'cpu'
    # criterion = nn.NLLLoss().to(device)

    X_tensor = torch.tensor(test_dataset.X)
    Y_tensor  = torch.tensor(test_dataset.y)
    images, labels = X_tensor.to(device), Y_tensor.to(device)
    outputs = model(images).squeeze()
    pred_labels = [ int(pred >= 0.5) for pred in outputs]
  
    pred_labels = pred_labels = torch.tensor([ int(pred >= 0.5) for pred in outputs])
    pred_labels = pred_labels.view(-1)
    # print("Len of pred_labels set: ", len(pred_labels))
    # print(pred_labels)

    train_bld_prediction_dataset = dataset.get_bld_dataset_w_pred(test_dataset, pred_labels)
                
    privileged_groups = [{test_dataset.s_attr: 1}]
    unprivileged_groups = [{test_dataset.s_attr: 0}]
    cm_pred_train = ClassificationMetric(test_dataset.bld, train_bld_prediction_dataset,
    unprivileged_groups=unprivileged_groups,
    privileged_groups=privileged_groups)

    accuracy = cm_pred_train.accuracy()
    if metric == "eod":
        local_fairness = cm_pred_train.equalized_odds_difference()

    # correct = torch.sum(torch.eq(pred_labels, labels)).item()
    print("predicted 1s / real 1s/ sample size: ", torch.sum(pred_labels), " / ", torch.sum(labels) ," / ", len(pred_labels))
    # accuracy = correct/len(pred_labels)

    return pred_labels, accuracy, local_fairness


def get_all_local_metrics(num_users, global_model, local_set_ls, gpu, set="test", fairness_metric="eod"):
    local_fairness_ls = []
    local_acc_ls = []

    for i in range(num_users):
        if set == "test":
            local_set_df = local_set_ls[i].test_set
        elif set == "train":
            local_set_df = local_set_ls[i].train_set

        local_dataset =  dataset.AdultDataset(csv_file="", df=local_set_df)
        pred_labels, accuracy, local_fairness = get_prediction_w_local_fairness(gpu, global_model, local_dataset, fairness_metric)
        local_fairness_ls.append(local_fairness)
        local_acc_ls.append(accuracy)
    
    return local_acc_ls, local_fairness_ls


def get_global_fairness(dataset, local_dataset_ls, prediction_ls, metric="eod", set="train"):
    '''
    Given local dataset split and predictions, return local fairness score
    '''

    rows = []
    for i in range(len(local_dataset_ls)):
        if set == "train":
            local_idxs = local_dataset_ls[i].train_set_idxs
        elif set == "test":
            local_idxs = local_dataset_ls[i].test_set_idxs
        
        for idx in  local_idxs:
            rows.append(dataset.df.iloc[idx])
    
    new_dataset = pd.DataFrame(rows)
    original_bld_dataset = BinaryLabelDataset(df=new_dataset, label_names=[dataset.target], protected_attribute_names=[dataset.s_attr])
    
    all_prediction =  [pred for ls in prediction_ls for pred in ls]
    prediction_dataset = new_dataset.copy(deep=True)
    prediction_dataset[dataset.target] = all_prediction
    prediction_bld_dataset = BinaryLabelDataset(df=prediction_dataset, label_names=[dataset.target], protected_attribute_names=[dataset.s_attr])


    privileged_groups = [{dataset.s_attr: 1}]
    unprivileged_groups = [{dataset.s_attr: 0}]
    cm_pred = ClassificationMetric(original_bld_dataset, prediction_bld_dataset,
    unprivileged_groups=unprivileged_groups,
    privileged_groups=privileged_groups)

    accuracy = cm_pred.accuracy()
    if metric == "eod":
        fairness = cm_pred.equalized_odds_difference()
    
    return accuracy, fairness
  

# def get_weighted_metric_gap(sample_size_ls, metric_gap):
