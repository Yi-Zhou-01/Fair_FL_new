#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
from aif360.datasets import BinaryLabelDataset

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


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, logger):
        self.args = args
        self.logger = logger
        self.trainloader, self.validloader, self.testloader, self.train_idx, self.valid_idx, self.test_idx = self.train_val_test(dataset, list(idxs))
        self.device = 'cuda' if args.gpu else 'cpu'
        # Default criterion set to NLL loss function
        # self.criterion = nn.NLLLoss().to(self.device)
        self.criterion = torch.nn.BCEWithLogitsLoss().to(self.device)
        self.dataset = dataset

        

    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and test (80, 10, 10)
        idxs_train = idxs[:int(0.8*len(idxs))]
        idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        idxs_test = idxs[int(0.9*len(idxs)):]

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True)
        validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                 batch_size=int(len(idxs_val)/10), shuffle=False)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=int(len(idxs_test)/10), shuffle=False)
        return trainloader, validloader, testloader, idxs_train, idxs_val, idxs_test

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
                loss = self.criterion(log_probs, labels)
                # loss = self.criterion(log_probs, labels.long())
                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 10 == 0):
                    if self.args.local_ep <= 10 or (self.args.local_ep <=100 and self.args.local_ep % 10 == 0) or (self.args.local_ep % 50 == 0):
                        print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
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
            batch_loss = self.criterion(outputs, labels)
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
    
    def fairness_eval_dataset(self, set="train", target_label="income", s_attr="sex_1"):
        privileged_groups = [{s_attr: 1}]
        unprivileged_groups = [{s_attr: 0}]

        # statistical_parity_difference of local train data
        if set == "train":
            subset_df = self.dataset.df[self.dataset.df.index.isin(self.train_idx)]
        elif set == "test":
            subset_df = self.dataset.df[self.dataset.df.index.isin(self.test_idx)]
        dataset_bld = BinaryLabelDataset(df=subset_df, label_names=[target_label], protected_attribute_names=[s_attr])
        
        metric_orig_train = BinaryLabelDatasetMetric(dataset_bld, 
                                unprivileged_groups=unprivileged_groups,
                                privileged_groups=privileged_groups)
        print("Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_orig_train.mean_difference())

        print("pass")
    
    def fairness_eval_predicction(self, model):


        print("pass")


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
    print("predicted 1s / sample size: ", torch.sum(pred_labels), " / ", len(pred_labels))

    return pred_labels
    
