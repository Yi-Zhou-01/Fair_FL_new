#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from torch import nn
import torch.nn.functional as F
import numpy as np

class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # print("x.shape", x.shape)
        # print("x.shape[1], x.shape[-2], x.shape[-1]", x.shape[1], x.shape[-2], x.shape[-1])
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return self.softmax(x)

# class MLPAdult(nn.Module):
#     def __init__(self, dim_in, dim_hidden, dim_out):
#         super(MLPAdult, self).__init__()
#         self.layer_input = nn.Linear(dim_in, dim_hidden)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout()
#         self.layer_hidden = nn.Linear(dim_hidden, dim_out)
#         self.softmax = nn.Softmax(dim=1)

#     def forward(self, x):
#         # print("x.shape", x.shape)
#         # print("x.shape[1], x.shape[-2], x.shape[-1]", x.shape[1], x.shape[-2], x.shape[-1])
#         # x = x.view(-1, x.shape[1]*x.shape[0])
#         x = self.layer_input(x)
#         x = self.dropout(x)
#         x = self.relu(x)
#         x = self.layer_hidden(x)
#         return self.softmax(x)

class MLPAdult(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLPAdult, self).__init__()
        self.layer_1 = nn.Linear(in_features=32, out_features=64)
        self.layer_2 = nn.Linear(in_features=64, out_features=128)
        self.layer_3 = nn.Linear(in_features=128, out_features=64)
        self.layer_4 = nn.Linear(in_features=64, out_features=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.dropout(self.relu(self.layer_1(x)))
        x = self.dropout(self.relu(self.layer_2(x)))
        x = self.dropout(self.relu(self.layer_3(x)))
        network = self.layer_4(x)
        return network
    



class MLPAdult2(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLPAdult2, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features=32, out_features=64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(in_features=64, out_features=128), 
            nn.ReLU(),
            nn.Dropout(0.1),
            # nn.Linear(in_features=128, out_features=64),
            # nn.ReLU(),
            # nn.Dropout(0.1),
        )
        self.final_layer = nn.Linear(in_features=128, out_features=1)

    def forward(self, x):
        x = self.mlp(x)
        network = self.final_layer(x)

        return network
    
    def get_features(self, x):
        # if norm:
        #     x = Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))(x)
        features = self.mlp(x)
        return features
        # return F.reshape(features, (features.shape[0], -1))

    def set_grad(self, val):
        for param in self.mlp.parameters():
            param.requires_grad = val



class MLPCompas(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLPCompas, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features=dim_in, out_features=64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(in_features=64, out_features=128), 
            nn.ReLU(),
            nn.Dropout(0.1),
            # nn.Linear(in_features=128, out_features=64),
            # nn.ReLU(),
            # nn.Dropout(0.1),
        )
        self.final_layer = nn.Linear(in_features=128, out_features=1)

    def forward(self, x):
        x = self.mlp(x)
        network = self.final_layer(x)

        return network
    
    def get_features(self, x):
        # if norm:
        #     x = Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))(x)
        features = self.mlp(x)
        return features
        # return F.reshape(features, (features.shape[0], -1))

    def set_grad(self, val):
        for param in self.mlp.parameters():
            param.requires_grad = val


def _padding(downsample, kernel_size):
    """Compute required padding"""
    padding = max(0, int(np.floor((kernel_size - downsample + 1) / 2)))
    return padding


def _downsample(n_samples_in, n_samples_out):
    """Compute downsample rate"""
    downsample = int(n_samples_in // n_samples_out)
    if downsample < 1:
        raise ValueError("Number of samples should always decrease\n but we got {} in and {} out.".format(n_samples_in, n_samples_out))
    if n_samples_in % n_samples_out != 0:
        raise ValueError("Number of samples for two consecutive blocks "
                         "should always decrease by an integer factor.")
    return downsample


class ResBlock1d(nn.Module):
    """Residual network unit for unidimensional signals."""

    def __init__(self, n_filters_in, n_filters_out, downsample, kernel_size, dropout_rate):
        if kernel_size % 2 == 0:
            raise ValueError("The current implementation only support odd values for `kernel_size`.")
        super(ResBlock1d, self).__init__()
        # Forward path
        padding = _padding(1, kernel_size)
        self.conv1 = nn.Conv1d(n_filters_in, n_filters_out, kernel_size, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(n_filters_out)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        padding = _padding(downsample, kernel_size)
        self.conv2 = nn.Conv1d(n_filters_out, n_filters_out, kernel_size,
                               stride=downsample, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(n_filters_out)
        self.dropout2 = nn.Dropout(dropout_rate)

        # Skip connection
        skip_connection_layers = []
        # Deal with downsampling
        if downsample > 1:
            maxpool = nn.MaxPool1d(downsample, stride=downsample)
            skip_connection_layers += [maxpool]
        # Deal with n_filters dimension increase
        if n_filters_in != n_filters_out:
            conv1x1 = nn.Conv1d(n_filters_in, n_filters_out, 1, bias=False)
            skip_connection_layers += [conv1x1]
        # Build skip conection layer
        if skip_connection_layers:
            self.skip_connection = nn.Sequential(*skip_connection_layers)
        else:
            self.skip_connection = None

    def forward(self, x, y):
        """Residual unit."""
        if self.skip_connection is not None:
            y = self.skip_connection(y)
        else:
            y = y
        # 1st layer
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)

        # 2nd layer
        x = self.conv2(x)
        x += y  # Sum skip connection and main connection
        y = x
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        return x, y

 
class ResNetPTB(nn.Module):
    def __init__(self, input_dim, blocks_dim, n_classes, kernel_size=17, dropout_rate=0.8):
        super(ResNetPTB, self).__init__()

        n_filters_in, n_filters_out = input_dim[0], blocks_dim[0][0]
        n_samples_in, n_samples_out = input_dim[1], blocks_dim[0][1]
        downsample = _downsample(n_samples_in, n_samples_out)
        padding = _padding(downsample, kernel_size)
        self.conv1 = nn.Conv1d(n_filters_in, n_filters_out, kernel_size, bias=False,
                               stride=downsample, padding=padding)
        self.bn1 = nn.BatchNorm1d(n_filters_out)

        # Residual block layers
        self.res_blocks = []
        for i, (n_filters, n_samples) in enumerate(blocks_dim):
            n_filters_in, n_filters_out = n_filters_out, n_filters
            n_samples_in, n_samples_out = n_samples_out, n_samples
            downsample = _downsample(n_samples_in, n_samples_out)
            resblk1d = ResBlock1d(n_filters_in, n_filters_out, downsample, kernel_size, dropout_rate)
            self.add_module('resblock1d_{0}'.format(i), resblk1d)
            self.res_blocks += [resblk1d]

        # Linear layer
        n_filters_last, n_samples_last = blocks_dim[-1]
        last_layer_dim = n_filters_last * n_samples_last
        self.final_layer = nn.Linear(last_layer_dim, n_classes)
        self.n_blk = len(blocks_dim)


    def forward(self, x):
        """Implement ResNet1d forward propagation"""
        # First layers
        x = self.conv1(x)
        x = self.bn1(x)

        # Residual blocks
        y = x
        for blk in self.res_blocks:
            x, y = blk(x, y)

        # Flatten array
        x = x.view(x.size(0), -1)

        # Fully conected layer
        x = self.final_layer(x)
        return x
    
    
    def get_features(self, x):
       # First layers
        x = self.conv1(x)
        x = self.bn1(x)

        # Residual blocks
        y = x
        for blk in self.res_blocks:
            x, y = blk(x, y)

        # Flatten array
        x = x.view(x.size(0), -1)

        return x
        # return F.reshape(features, (features.shape[0], -1))

    def set_grad(self, val):
        # Freeze all paramters and reactive final layer parameter
        for param in self.parameters():
            param.requires_grad = False
        for param in self.final_layer.parameters():
            param.requires_grad = True

        

class Plain_LR_Adult(nn.Module):
    def __init__(self, dim_in):
        super(Plain_LR_Adult, self).__init__()

        self.input_size = dim_in
        self.fc1 = nn.Linear(self.input_size, 1)

        # super(MLPAdult, self).__init__()
        # self.layer_1 = nn.Linear(in_features=32, out_features=64)
        # self.layer_2 = nn.Linear(in_features=64, out_features=128)
        # self.layer_3 = nn.Linear(in_features=128, out_features=64)
        # self.layer_4 = nn.Linear(in_features=64, out_features=1)
        # self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(0.1)
    def forward(self, x):
        prediction = nn.functional.sigmoid(self.fc1(x))

        return prediction

    # def forward(self, x):
    #     x = self.dropout(self.relu(self.layer_1(x)))
    #     x = self.dropout(self.relu(self.layer_2(x)))
    #     x = self.dropout(self.relu(self.layer_3(x)))
    #     network = self.layer_4(x)
    #     return network

    

# class Classification_AdultCensus(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.layer_1 = nn.Linear(in_features=34, out_features=64)
#         self.layer_2 = nn.Linear(in_features=64, out_features=128)
#         self.layer_3 = nn.Linear(in_features=128, out_features=64)
#         self.layer_4 = nn.Linear(in_features=64, out_features=1)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(0.1)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.dropout(self.relu(self.layer_1(x)))
#         x = self.dropout(self.relu(self.layer_2(x)))
#         x = self.dropout(self.relu(self.layer_3(x)))
#         network = self.layer_4(x)
#         return network



class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class CNNFashion_Mnist(nn.Module):
    def __init__(self, args):
        super(CNNFashion_Mnist, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7*7*32, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, args.num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

class modelC(nn.Module):
    def __init__(self, input_size, n_classes=10, **kwargs):
        super(AllConvNet, self).__init__()
        self.conv1 = nn.Conv2d(input_size, 96, 3, padding=1)
        self.conv2 = nn.Conv2d(96, 96, 3, padding=1)
        self.conv3 = nn.Conv2d(96, 96, 3, padding=1, stride=2)
        self.conv4 = nn.Conv2d(96, 192, 3, padding=1)
        self.conv5 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv6 = nn.Conv2d(192, 192, 3, padding=1, stride=2)
        self.conv7 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv8 = nn.Conv2d(192, 192, 1)

        self.class_conv = nn.Conv2d(192, n_classes, 1)


    def forward(self, x):
        x_drop = F.dropout(x, .2)
        conv1_out = F.relu(self.conv1(x_drop))
        conv2_out = F.relu(self.conv2(conv1_out))
        conv3_out = F.relu(self.conv3(conv2_out))
        conv3_out_drop = F.dropout(conv3_out, .5)
        conv4_out = F.relu(self.conv4(conv3_out_drop))
        conv5_out = F.relu(self.conv5(conv4_out))
        conv6_out = F.relu(self.conv6(conv5_out))
        conv6_out_drop = F.dropout(conv6_out, .5)
        conv7_out = F.relu(self.conv7(conv6_out_drop))
        conv8_out = F.relu(self.conv8(conv7_out))

        class_out = F.relu(self.class_conv(conv8_out))
        pool_out = F.adaptive_avg_pool2d(class_out, 1)
        pool_out.squeeze_(-1)
        pool_out.squeeze_(-1)
        return pool_out
