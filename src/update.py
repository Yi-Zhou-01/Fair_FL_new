#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import dataset
import pandas as pd
import torch.nn.functional as F
import utils
import sys
import time
import copy


from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
from aif360.datasets import BinaryLabelDataset


from sklearn.model_selection import train_test_split
import numpy as np

# from memory_profiler import profile

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

        # print(self.dataset[self.idxs[0]])

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        # image, label = self.dataset[self.idxs[item]]
        # image = self.dataset.X[self.idxs[item]]
        # label = self.dataset.df[self.dataset.target][self.idxs[item]]
        # s_attr = self.dataset[self.dataset.s_attr][self.idxs[item]]
        image, label, s_attr = self.dataset[self.idxs[item]]
        return torch.as_tensor(image), torch.as_tensor(label), torch.as_tensor(s_attr)


# class AdultDatasetWrapDf(Dataset):
#     """An abstract Dataset class wrapped around Pytorch Dataset class.
#     """

#     def __init__(self, dataset_df):
#         self.dataset_df = dataset_df.reset_index(drop=True)

#         # print(self.dataset_df[:5])

#     def __len__(self):
#         return len(self.dataset_df)

#     def __getitem__(self, idx):
#         image, label = self.dataset_df.drop('income', axis=1).iloc[idx], self.dataset_df['income'].iloc[idx]
#         return torch.tensor(image).to(torch.float32), torch.tensor(label).to(torch.float32)



class BatchDataloader:
    def __init__(self, *tensors, bs=1, mask=None):
        nonzero_idx, = np.nonzero(mask)
        self.tensors = tensors
        self.batch_size = bs
        self.mask = mask
        if nonzero_idx.size > 0:
            self.start_idx = min(nonzero_idx)
            self.end_idx = max(nonzero_idx)+1
        else:
            self.start_idx = 0
            self.end_idx = 0

    def __next__(self):
        if self.start == self.end_idx:
            raise StopIteration
        end = min(self.start + self.batch_size, self.end_idx)
        batch_mask = self.mask[self.start:end]
        while sum(batch_mask) == 0:
            self.start = end
            end = min(self.start + self.batch_size, self.end_idx)
            batch_mask = self.mask[self.start:end]
        batch = [np.array(t[self.start:end]) for t in self.tensors]
        self.start = end
        self.sum += sum(batch_mask)
        return [torch.tensor(b[batch_mask], dtype=torch.float32) for b in batch]

    def __iter__(self):
        self.start = self.start_idx
        self.sum = 0
        return self

    def __len__(self):
        count = 0
        start = self.start_idx
        while start != self.end_idx:
            end = min(start + self.batch_size, self.end_idx)
            batch_mask = self.mask[start:end]
            if sum(batch_mask) != 0:
                count += 1
            start = end
        return count


class LocalDataset(object):
    def __init__(self, global_dataset, local_idxs, test_ratio=0.2):
        
        self.local_idxs = np.asarray(list(local_idxs))
        self.test_ratio = test_ratio
        # all_X = global_dataset.X
        # all_y = global_dataset.y
        # all_a = global_dataset.a

        # self.local_dataset =  dataset.df[dataset.df.index.isin(local_idxs)]
        self.target_label = global_dataset.target
        self.s_attr =  global_dataset.s_attr
       
        # self.local_train_set, self.local_test_set, self.local_val_set=  self.train_test_split(global_dataset.name, global_dataset.X, global_dataset.y, global_dataset.a)


        # self.local_train_set, self.local_test_set, self.local_val_set, \
        #      self.train_set_idxs, self.test_set_idxs, self.val_set_idxs  =  self.train_test_split(global_dataset.name, global_dataset.X, global_dataset.y, global_dataset.a)
        
        # print("Check local type: ")
        # print(((self.local_idxs)))
        self.train_set_idxs, self.test_set_idxs, self.val_set_idxs  =  \
            self.train_test_split(global_dataset.name, global_dataset.X[(self.local_idxs)], \
                                  global_dataset.y[(self.local_idxs)], global_dataset.a[(self.local_idxs)])


        # self.train_set_X, self.train_set_Y, self.train_set_a, \
        #     self.test_set_X, self.test_set_Y, self.test_set_a, \
        #         self.val_set_X, self.val_set_Y, self.val_set_a, \
        #             self.train_set_idxs, self.test_set_idxs, self.val_set_idxs  = self.train_test_split()
        
        # self.train_len = len(self.train_set_idxs)
        # self.test_len = len(self.test_set_idxs)
        # self.val_len = len(self.val_set_idxs)
        
        self.size = len(self.local_idxs)
        # self.train_set, self.test_set, self.val_set = self.train_test_split()
        

    # Return df
    def train_test_split(self, name, X, y, a):

        
        # X = self.local_dataset.drop(self.target_label, axis=1)
        # y = self.local_dataset[self.target_label]
        # a = self.local_dataset[self.s_attr]

        # local_X = X[self.local_idxs]
        # local_y = y[self.]

        if name == "ptb-xl" or "nih-chest":
            dummy_X = np.array(range(len(X)))
            X_train, X_test, y_train, y_test, a_train, a_test  = train_test_split(pd.DataFrame(dummy_X), pd.DataFrame(y), pd.DataFrame(a), test_size=self.test_ratio, stratify=y)
            X_val = X_test
            train_set_idxs =  list(X_train.index)
            test_set_idxs = list(X_test.index)
            val_set_idxs = list(X_val.index)
            # print("train shape before", X[X_train].shape)

            X_train =np.squeeze(X[X_train])
            # print("train shape after", X_train.shape)
            X_test = np.squeeze(X[X_test])
            X_val =  np.squeeze(X[X_val])
            
            
        else:
            X_train, X_test, y_train, y_test, a_train, a_test  = train_test_split(pd.DataFrame(X), pd.DataFrame(y), pd.DataFrame(a), test_size=self.test_ratio, stratify=y)
            X_val = X_test
            train_set_idxs =  list(X_train.index)
            test_set_idxs = list(X_test.index)
            val_set_idxs = list(X_val.index)


        # X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, stratify=y_test)

        # X_train[self.target_label] = y_train
        # X_test[self.target_label] = y_test
    
        # X_val[self.target_label] = y_val
        
        # y_val = y_test
        # a_val = a_test
        # y_val = y_test

        # if name == "compas":
        #     local_train_set = dataset.CompasDataset(csv_file="", X=X_train, y=y_train,a=a_train)
        #     local_test_set = dataset.CompasDataset(csv_file="", X=X_test, y=y_test,a=a_test)
        #     local_val_set = local_test_set
        # elif name == "ptb-xl":
        #     # print("train shape before", X_train.shape)
        #     local_train_set = dataset.PTBDataset(csv_file="", X=X_train, y=y_train,a=a_train)
        #     # print("train shape after", local_train_set.X.shape)
        #     local_test_set = dataset.PTBDataset(csv_file="", X=X_test, y=y_test,a=a_test)
        #     local_val_set = local_test_set
        # elif name == "nih-chest":
        #     # print("train shape before", X_train.shape)
        #     local_train_set = dataset.NIHDataset(csv_file="", X=X_train, y=y_train,a=a_train)
        # else:
        #     print("!! ERROR: Dataset not implemented in LocalDataset: train_test_split")


        # return local_train_set, local_test_set, local_val_set, train_set_idxs, test_set_idxs, val_set_idxs
        # print("train_set_idxs: ", train_set_idxs)
        # print("test_set_idxs: ", test_set_idxs)
        return self.local_idxs[train_set_idxs], self.local_idxs[test_set_idxs], self.local_idxs[val_set_idxs]
    # X_train.values, y_train.values, a_train.values, X_test.values, y_test.values, a_test.values, X_val.values, y_val.values, a_val.values, \
            # list(X_train.index), list(X_test.index), list(X_val.index)



def get_mask_from_idx(data_size, train_idxs):

    mask = np.zeros(data_size, dtype=np.int8)
    for idx in train_idxs:
        mask[idx] = 1
    return mask


class LocalUpdate(object):
    def __init__(self, args, split_idxs, dataset, idxs, logger,local_dataset=None):
        self.args = args
        self.logger = logger
        self.local_dataset = local_dataset
        # self.ft = fine_tuning

        # self.trainloader, self.validloader, self.testloader = self.train_val_test(dataset, list(idxs))
        self.trainloader, self.validloader, self.testloader = self.split_w_idxs(dataset, split_idxs, args.local_bs, args.dataset)
        self.ft_trainloader, _, _ = self.split_w_idxs(dataset, split_idxs, args.ft_bs, args.dataset)
        self.device = 'cuda' if args.gpu else 'cpu'
        # Default criterion set to NLL loss function
        # self.criterion = nn.NLLLoss().to(self.device)
        # self.criterion = torch.nn.BCEWithLogitsLoss().to(self.device)
        self.criterion = torch.nn.BCELoss().to(self.device)
        self.dataset = dataset

    def split_w_idxs(self, dataset, idxs, batch_size, dataset_name=""):
        train_idxs, test_idxs, val_idxs = idxs

        # if self.args.dataset == "ptb-xl":
        #     data_size = dataset.size
        #     train_mask = get_mask_from_idx(data_size, train_idxs)
        #     val_mask = get_mask_from_idx(data_size, val_idxs)
        #     test_mask = get_mask_from_idx(data_size, test_idxs)
        #     trainloader = BatchDataloader(dataset.X, dataset.y, dataset.a, bs=self.args.local_bs, mask=train_mask)
        #     validloader = BatchDataloader(dataset.X, dataset.y, dataset.a, bs=self.args.local_bs, mask=val_mask)
        #     testloader = BatchDataloader(dataset.X, dataset.y, dataset.a, bs=self.args.local_bs, mask=test_mask)
        #     # print("ptb-xl Loader!")
        # else:

        test_bs = batch_size
        # if "nih" in dataset_name:
        #     test_bs = batch_size
        # else:
        #     test_bs = int(len(test_idxs)/10)

        trainloader = DataLoader(DatasetSplit(dataset, train_idxs),
                                batch_size=batch_size, shuffle=True)
        validloader = DataLoader(DatasetSplit(dataset, val_idxs),
                                batch_size=test_bs, shuffle=False)
        testloader = DataLoader(DatasetSplit(dataset, test_idxs),
                                batch_size=test_bs, shuffle=False)
        
        return trainloader, validloader, testloader
    

    def update_final_layer(self, model,global_round, client_idx=-1):
        model.train()
        model.set_grad(False)
        epoch_loss = []
        # hyperparameter for 100% fairness
        # optimizer = torch.optim.SGD(model.final_layer.parameters(), lr=1e-2,
                                        # momentum=0.9, weight_decay=5e-4)

        optimizer = torch.optim.SGD(model.final_layer.parameters(), lr=self.args.ft_lr,
                                        momentum=0.9, weight_decay=5e-4)
        
        # criterion = torch.nn.BCEWithLogitsLoss().to(self.device)
        criterion = torch.nn.BCELoss().to(self.device)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5, last_epoch=-1)
        # if self.args.optimizer == 'sgd':
        #     optimizer = torch.optim.SGD(model.final_layer.parameters(), lr=self.args.lr,
        #                                 momentum=0.5, weight_decay=1e-4)
            
        # elif self.args.optimizer == 'adam':
        #     optimizer = torch.optim.Adam(model.final_layer.parameters(), lr=self.args.lr,
        #                                 weight_decay=1evscode-webview://1hs7hefg7d9igdkj7jjs2g8ig9einv4iiof1dn2o9nvsvcaqja52/index.html?id=cfc1b4fd-dd94-41f2-b483-be71d2d91a99&origin=148850e9-0f3e-4859-a5fc-0508f77f5be3&swVersion=4&extensionId=vscode.media-preview&platform=electron&vscode-resource-base-authority=vscode-resource.vscode-cdn.net&parentOrigin=vscode-file%3A%2F%2Fvscode-app#-4)
        
        lowest_loss=1000
        # best_model
        for iter in range(self.args.ft_ep):
            batch_loss = []
            batch_loss_fairness = []
            batch_loss_1 = []
            for batch_idx, (images, labels, a) in enumerate(self.ft_trainloader):
                images, labels, a = images.to(self.device), labels.to(self.device), a.to(self.device)
                optimizer.zero_grad()  

                outputs = model(images).squeeze()
                # outputs = outputs.reshape(-1)

                # outputs = model.final_layer(model.get_features(images)).squeeze()
                # pred_labels = torch.tensor([ int(pred >= 0.5) for pred in outputs]).view(-1)
                # pred_labels =  (outputs > 0.5).to(torch.float32).view(-1)
                # if self.args.model == "plain":
                #     pred_labels = torch.max(outputs.data, 1)
                # else:

                loss_1 = criterion(outputs, labels)
               
                pred_labels = (outputs > 0.5).to(torch.float32)
                # print("pred_labels: ", type(pred_labels))
                eod_loss = utils.equalized_odds_diff(pred_labels, labels, a)
                # print("eod_loss: ", type(eod_loss))

                loss = loss_1*self.args.ft_alpha2 + self.args.ft_alpha * eod_loss
                # print("loss: ", type(loss))
                # loss = eod_loss

                loss.backward(retain_graph=True)
                optimizer.step()

                self.logger.add_scalar('loss', loss.item())
                
                # print('** Loss: {:.6f}  L1 - EOD:  {:.6f} | {:.6f}'.format(loss.item(), loss_1.item(), eod_loss.item()))

                batch_loss_fairness.append(eod_loss.item())
                batch_loss.append(loss.item())
                batch_loss_1.append(loss_1.item())

                # if batch_idx % 50 == 0:
                #     print('{} | # {} | Global Round : {} | Local Epoch : {} | Loss: {:.6f}  L1|EOD:  {:.6f} | {:.6f}'.format(
                #         int(time.time()), client_idx, global_round, iter, loss.item(), loss_1.item(), eod_loss.item()))
    
            print('{} | # {} | Global Round : {} | Local Epoch : {} | Loss: {:.6f}  L1|EOD:  {:.6f} | {:.6f}'.format(
                   int(time.time()), client_idx, global_round, iter, sum(batch_loss)/len(batch_loss), sum(batch_loss_1)/len(batch_loss_1), sum(batch_loss_fairness)/len(batch_loss_fairness)))
    
            if sum(batch_loss)/len(batch_loss) < lowest_loss:
                best_model = copy.deepcopy(model)
                lowest_loss = sum(batch_loss)/len(batch_loss)

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            # scheduler.step()

        # return best_model.state_dict(), sum(epoch_loss) / len(epoch_loss), np.asarray(epoch_loss)
        return best_model.state_dict(), sum(epoch_loss) / len(epoch_loss), np.asarray(epoch_loss)


    def update_weights(self, model, global_round, client_idx=-1):
        # Set mode to train model
        start_time = time.time()
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5, weight_decay=1e-4)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                        weight_decay=1e-4)

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels, _) in enumerate(self.trainloader):
                # [a,b,c] = images.shape
                # print("images shape: ", images.shape)
                # print(a,b,c)
                # images = torch.reshape(images, (b,c,a))
                # print("images shape: ", images.shape)
                # print("images shape new: ",torch.flatten(images, start_dim=1).shape)
                # images = torch.flatten(images, start_dim=1)
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                # log_probs = model(images)
                # def closure():

                #     log_probs = model(images).squeeze()
                #     log_probs = log_probs.reshape(-1)
                #     loss = self.criterion(log_probs, labels)
                #     optimizer.zero_grad()
                #     # loss = self.criterion(log_probs, labels.long())
                #     loss.backward()
                #     return loss
                
                # loss = closure()
                # optimizer.step(closure)
                log_probs = model(images).squeeze()
                log_probs = log_probs.reshape(-1)
                loss = self.criterion(log_probs, labels)
                # optimizer.zero_grad()
                # loss = self.criterion(log_probs, labels.long())
                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 10 == 0):
                    if batch_idx % 100 == 0 and (iter<10 or iter%10 ==0):
                    # if self.args.local_ep <= 10 or (self.args.local_ep <=100 and self.args.local_ep % 10 == 0) or (self.args.local_ep % 50 == 0):
                        # print("time: ", int(time.time() - start_time), int(time.time()))
                        print('{} | # {} | Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]  \tLoss: {:.6f}'.format(
                            int(time.time()), client_idx, global_round, iter, batch_idx * len(images),
                            len(self.trainloader.dataset),
                            100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss), batch_loss

    def inference(self, model):
        """ Returns the inference accuracy and loss of local test data (?). 
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels, _) in enumerate(self.validloader):
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
            pred_labels = torch.tensor([ int(pred >= self.args.threshold) for pred in outputs])
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)


        # print("Inference pred labels: ", correct, "/", total, sum(labels))
        accuracy = correct/total
        return accuracy, loss
    
    # @profile
    def inference_w_fairness(self, model, set="test", fairness_metric=["eod"], client_idx=-1):
        """ Returns the inference accuracy and loss of local test data (?). 
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0


        all_y = np.array([])
        all_a = np.array([])
        all_pred = np.array([])

        if set == "test":
            loader = self.testloader
        elif set == "val":
            loader = self.testloader
        else:
            loader = self.trainloader

        for batch_idx, (images, labels, a) in enumerate(loader):
        # for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels, a = images.to(self.device), labels.to(self.device),  a.to(self.device)

            outputs = model(images).squeeze()
            outputs = outputs.reshape(-1)

            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # pred_labels = torch.tensor([ int(pred >= 0.5) for pred in outputs])
            
            # if self.args.model == "plain":
            #     # print(outputs.shape)
            #     print("outputs: ", outputs)
            #     pred_labels = torch.max(outputs.data)
            #     print("pred_labels: ", pred_labels)
            # else:
            # pred_labels = (outputs > 0.5).to(torch.float32)
            pred_labels = (outputs > self.args.threshold).to(torch.float32)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
            all_y = np.append(all_y, labels.detach().cpu().numpy())
            all_a = np.append(all_a, a.detach().cpu().numpy())
            all_pred = np.append(all_pred, pred_labels.detach().cpu().numpy())
            # all_y = np.append(all_y, labels.item())
            # all_a = np.append(all_a, a.item())
            # all_pred = np.append(all_pred, pred_labels.item())
            # print("all_y", type(all_y[0]))


            if self.args.verbose and (batch_idx % 10 == 0):
                    if batch_idx % 100 == 0:
                    # if self.args.local_ep <= 10 or (self.args.local_ep <=100 and self.args.local_ep % 10 == 0) or (self.args.local_ep % 50 == 0):
                        # print("time: ", int(time.time() - start_time), int(time.time()))
                        print('{} | # {} | Global Round : .. | Local Epoch : .. | [{}/{} ({:.0f}%)]  \tLoss: {:.6f}'.format(
                            int(time.time()), client_idx, batch_idx * len(images),
                            len(loader.dataset),
                            100. * batch_idx / len(self.trainloader), batch_loss.item()))
        

        train_bld_prediction_dataset = dataset.get_bld_dataset_w_pred(all_a, all_pred)
        original_bld = dataset.get_bld_dataset_w_pred(all_a, all_y)
                    
        # privileged_groups = [{s_attr: 1}]
        # unprivileged_groups = [{s_attr: 0}]
        privileged_groups = [{"a": 1}]
        unprivileged_groups = [{"a": 0}]
        cm_pred_train = ClassificationMetric(original_bld, train_bld_prediction_dataset,
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups)

        accuracy = cm_pred_train.accuracy()
        # print("accuracy: ", accuracy.size, sys.getsizeof(accuracy))
        local_fairness = {}
        if "eod" in fairness_metric:
            local_fairness["eod"] = (cm_pred_train.equalized_odds_difference())
            # local_fairness["eod"] = (cm_pred_train.average_abs_odds_difference())
        if True:
            local_fairness["tpr"] = (cm_pred_train.true_positive_rate_difference())
            local_fairness["fpr"] = (cm_pred_train.false_positive_rate_difference())

        # if "tpr" in fairness_metric:
        #     local_fairness["tpr"] = (cm_pred_train.true_positive_rate_difference())
        # if "fpr" in fairness_metric:
        #     local_fairness["fpr"] = (cm_pred_train.false_positive_rate_difference())



        # print("Inference pred labels: ", correct, "/", total, sum(labels))
        accuracy = correct/total
        return accuracy, loss, all_y, all_a, all_pred, local_fairness

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

    for batch_idx, (images, labels, _) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images).squeeze()
        # batch_loss = criterion(outputs, labels)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        # _, pred_labels = torch.max(outputs, 1)
        pred_labels = torch.tensor([ int(pred >= args.threshold) for pred in outputs])
        # pred_labels = outputs
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    return accuracy, loss

def get_prediction(args, model, X, Y):

    """ Returns the test accuracy and loss.
    """

    model.eval()
    device = 'cuda' if args.gpu else 'cpu'
    # criterion = nn.NLLLoss().to(device)

    # X_tensor = torch.tensor(test_dataset.X)
    # Y_tensor  = torch.tensor(test_dataset.y)
    X_tensor = torch.tensor(X)
    Y_tensor  = torch.tensor(Y)
    images, labels = X_tensor.to(device), Y_tensor.to(device)
    outputs = model(images).squeeze()
    pred_labels = [ int(pred >= 0.5) for pred in outputs]
  
    pred_labels = torch.tensor([ int(pred >= 0.5) for pred in outputs])
    pred_labels = pred_labels.view(-1)
    # print("Len of pred_labels set: ", len(pred_labels))
    # print(pred_labels)

    correct = torch.sum(torch.eq(pred_labels, labels)).item()
    # print("predicted 1s / real 1s/ sample size: ", torch.sum(pred_labels), " / ", torch.sum(labels) ," / ", len(pred_labels))
    accuracy = correct/len(pred_labels)

    return pred_labels, accuracy

# instantiating the decorator
# @profile
# code for which memory has to
# be monitored
def get_prediction_w_local_fairness(gpu, model, X, Y, a, fairness_metric=["eod"]):
    
    """ Returns the test accuracy and loss.
    """

    model.eval()
    device = 'cuda' if gpu else 'cpu'
    # criterion = nn.NLLLoss().to(device)

    X_tensor = torch.tensor(X)
    Y_tensor  = torch.tensor(Y)
    # s_attr_ls = list(test_dataset.df[test_dataset.s_attr])
    s_attr_ls = np.array(a)
    print("s_attr_ls: ", s_attr_ls.size, sys.getsizeof(s_attr_ls), type(s_attr_ls))
    s_attr_ls = torch.tensor(a)
    print("s_attr_ls: ", s_attr_ls.size(), sys.getsizeof(s_attr_ls), type(s_attr_ls))
    images, labels = X_tensor.to(device), Y_tensor.to(device)
    outputs = model(images).squeeze()
    # print("images, labels, outputs : ", len(images), len(labels), len(outputs))
    print("labels: ", labels.size(), sys.getsizeof(labels), type(labels))
    # pred_labels = [ int(pred >= 0.5) for pred in outputs]
  
    pred_labels = torch.tensor([ int(pred >= 0.5) for pred in outputs])
    print("pred_labels: ", pred_labels.size())
    pred_labels = pred_labels.view(-1)
    print("pred_labels2: ", pred_labels.size(), sys.getsizeof(pred_labels))
    # print("Len of pred_labels set: ", len(pred_labels))
    # print(pred_labels)

    train_bld_prediction_dataset = dataset.get_bld_dataset_w_pred(a, pred_labels)
    original_bld = dataset.get_bld_dataset_w_pred(a, Y )
                
    # privileged_groups = [{s_attr: 1}]
    # unprivileged_groups = [{s_attr: 0}]
    privileged_groups = [{"a": 1}]
    unprivileged_groups = [{"a": 0}]
    cm_pred_train = ClassificationMetric(original_bld, train_bld_prediction_dataset,
    unprivileged_groups=unprivileged_groups,
    privileged_groups=privileged_groups)

    accuracy = cm_pred_train.accuracy()
    print("accuracy: ", accuracy.size, sys.getsizeof(accuracy))
    local_fairness = {}
    if "eod" in fairness_metric:
        local_fairness["eod"] = (cm_pred_train.equalized_odds_difference())
        # local_fairness["eod"] = (cm_pred_train.average_abs_odds_difference())
    if "tpr" in fairness_metric:
        local_fairness["tpr"] = (cm_pred_train.true_positive_rate_difference())
    if "fpr" in fairness_metric:
        local_fairness["fpr"] = (cm_pred_train.false_positive_rate_difference())
    
    print("local_fairness: ", sys.getsizeof(local_fairness))


    # correct = torch.sum(torch.eq(pred_labels, labels)).item()
    # print("predicted 1s / real 1s/ sample size: ", torch.sum(pred_labels), " / ", torch.sum(labels) ," / ", len(pred_labels))
    # accuracy = correct/len(pred_labels)

    check_ls = [pred_labels, accuracy, local_fairness, labels, s_attr_ls]
    print("all return mem:", sum([sys.getsizeof(it) for it in check_ls]))
    for it in check_ls:
    #     print("data type: ", type(it))
        print("mem size: ", sys.getsizeof(it))
    #     if not isinstance(it,dict):
    #         print("size: ", it.size)


    return pred_labels, accuracy, local_fairness, labels, s_attr_ls



# instantiating the decorator
# @profile
# code for which memory has to
# be monitored
def get_all_local_metrics(datasetname, num_users, global_model, local_set_ls, gpu, kaggle, set="test", fairness_metric=["eod"], return_label=False):
    local_fairness_ls = {}
    local_acc_ls = []
    labels_ls = []
    s_attr_ls = []
    pred_labels_ls=[]
    if "eod" in fairness_metric:
        local_fairness_ls["eod"] = []
    if "tpr" in fairness_metric:
        local_fairness_ls["tpr"] = []
    if "fpr" in fairness_metric:
        local_fairness_ls["fpr"] = []

    for i in range(num_users):
        if set == "test":
            # local_set_df = local_set_ls[i].test_set
            print("Test local set")
            print("len y, a : ", len(local_set_ls[i].local_test_set.y), len(local_set_ls[i].local_test_set.a))
            print("type of a: *** ", type(local_set_ls[i].local_test_set.a))
            pred_labels, accuracy, local_fairness, labels, s_attr = \
                get_prediction_w_local_fairness(gpu, global_model, local_set_ls[i].local_test_set.X, local_set_ls[i].local_test_set.y, local_set_ls[i].local_test_set.a, \
                                                 fairness_metric=fairness_metric)
        elif set == "train":
            # local_set_df = local_set_ls[i].train_set
            print("Train local set")
            print("len y, a : ", len(local_set_ls[i].local_train_set.y), len(local_set_ls[i].local_train_set.a))
            pred_labels, accuracy, local_fairness, labels, s_attr = \
                get_prediction_w_local_fairness(gpu, global_model, local_set_ls[i].local_train_set.X, local_set_ls[i].local_train_set.y, local_set_ls[i].local_train_set.a, \
                                                  fairness_metric=fairness_metric)
        else:
            print( "Wrong set in update.get_all_local_metrics ")

        # if datasetname == "adult":
        #     local_dataset =  dataset.AdultDataset(csv_file="", df=local_set_df)
        # elif datasetname == "compas":
        #     local_dataset =  dataset.CompasDataset(csv_file="", df=local_set_df)
        # elif datasetname == "wcld":
        #     local_dataset =  dataset.WCLDDataset(csv_file="", df=local_set_df)
        # elif datasetname == "ptb-xl":
        #     local_dataset =  dataset.PTBDataset(csv_file="", df=local_set_df, kaggle=kaggle)       


        # pred_labels, accuracy, local_fairness, labels, s_attr = get_prediction_w_local_fairness(gpu, global_model, local_dataset, fairness_metric)
        local_acc_ls.append(accuracy)
        if return_label:
            pred_labels_ls.append(pred_labels)
            labels_ls.append(labels)
            s_attr_ls.append((s_attr))
        if "eod" in fairness_metric:
            local_fairness_ls["eod"].append(local_fairness["eod"])
        if "tpr" in fairness_metric:
            local_fairness_ls["tpr"].append(local_fairness["tpr"])
        if "fpr" in fairness_metric:
            local_fairness_ls["fpr"].append(local_fairness["fpr"])
    
    if return_label:
        return local_acc_ls, local_fairness_ls, pred_labels_ls, labels_ls, s_attr_ls
    else:
        return local_acc_ls, local_fairness_ls


def get_global_fairness(local_dataset_ls, prediction_ls, metric="eod", set="train"):
    '''
    Given local dataset split and predictions, return local fairness score
    '''

    if set=="train":
        all_a = [x for i in range(len(local_dataset_ls)) for x in local_dataset_ls[i].local_train_set.a ]
        all_Y = [x for i in range(len(local_dataset_ls)) for x in local_dataset_ls[i].local_train_set.y ]
    
    elif set == "test":
        all_a = [x for i in range(len(local_dataset_ls)) for x in local_dataset_ls[i].local_test_set.a ]
        all_Y = [x for i in range(len(local_dataset_ls)) for x in local_dataset_ls[i].local_test_set.y ]


    # for i in range(len(local_dataset_ls)):
    #     if set == "train":
    #         local_idxs = local_dataset_ls[i].train_set_idxs
    #     elif set == "test":
    #         local_idxs = local_dataset_ls[i].test_set_idxs
        
    #     for idx in  local_idxs:
    #         rows.append(dataset.df.iloc[idx])
    
    # new_dataset = pd.DataFrame(rows)
    # original_bld_dataset = BinaryLabelDataset(df=new_dataset, label_names=[dataset.target], protected_attribute_names=[dataset.s_attr])
    original_bld_dataset = dataset.get_bld_dataset_w_pred(all_a, all_Y)
    
    all_prediction =  [pred for ls in prediction_ls for pred in ls]
    # prediction_dataset = new_dataset.copy(deep=True)
    # prediction_dataset[dataset.target] = all_prediction
    # prediction_bld_dataset = BinaryLabelDataset(df=prediction_dataset, label_names=[dataset.target], protected_attribute_names=[dataset.s_attr])
    prediction_bld_dataset = dataset.get_bld_dataset_w_pred(all_a, all_prediction)

    privileged_groups = [{"a": 1}]
    unprivileged_groups = [{"a": 0}]
    cm_pred = ClassificationMetric(original_bld_dataset, prediction_bld_dataset,
    unprivileged_groups=unprivileged_groups,
    privileged_groups=privileged_groups)

    accuracy = cm_pred.accuracy()
    if metric == "eod":
        # fairness = cm_pred.average_abs_odds_difference()
        fairness = cm_pred.equalized_odds_difference()
    
    return accuracy, fairness
  

# def get_weighted_metric_gap(sample_size_ls, metric_gap):


def get_global_fairness_new(local_a_ls, local_y_ls, prediction_ls, metric="eod", set="train"):

    # all_a = np.asarray(local_a_ls).flatten()
    # all_y = np.asarray(local_y_ls).flatten()
    # all_prediction =np.asarray(prediction_ls).flatten()

    all_a = np.concatenate(local_a_ls).ravel()
    all_y =  np.concatenate(local_y_ls).ravel()
    all_prediction = np.concatenate(prediction_ls).ravel()

    
    original_bld_dataset = dataset.get_bld_dataset_w_pred(all_a, all_y)
    prediction_bld_dataset = dataset.get_bld_dataset_w_pred(all_a, all_prediction)

    privileged_groups = [{"a": 1}]
    unprivileged_groups = [{"a": 0}]

    cm_pred = ClassificationMetric(original_bld_dataset, prediction_bld_dataset,
    unprivileged_groups=unprivileged_groups,
    privileged_groups=privileged_groups)

    accuracy = cm_pred.accuracy()
    if metric == "eod":
        # fairness = cm_pred.average_abs_odds_difference()
        fairness = cm_pred.equalized_odds_difference()    

    return accuracy, fairness