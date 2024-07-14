
import pandas as pd
import os
import torch
from torch.utils.data import Dataset
import numpy as np

from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from sampling import cifar_iid, cifar_noniid
from sampling import adult_iid, adult_noniid
import sampling

from sklearn.preprocessing import StandardScaler
from aif360.datasets import BinaryLabelDataset

class AdultDataset(Dataset):
    """Students Performance dataset."""

    def __init__(self, csv_file, df=None, crop=None, subset=None):
        """Initializes instance of Adult dataset.
        """

        if df is not None:
            self.df = df
        else:
            self.df = pd.read_csv(csv_file, index_col=False) #.drop("Unnamed: 0", axis=1)
        if crop:
            self.df = self.df[:crop]

        if subset:
            self.df = self.df[self.df.index.isin(subset)]

        self.target = "income"
        self.s_attr = "sex_1"
        self.bld = BinaryLabelDataset(df=self.df, label_names=[self.target], protected_attribute_names=[self.s_attr])

        self.X = self.df.drop(self.target, axis=1).to_numpy().astype(np.float32)
        self.y = self.df[self.target].to_numpy().astype(np.float32)
        self.a = self.df[self.s_attr].to_numpy().astype(np.float32)

        self.X = self.standardlize_X(self.X)
        self.size = len(self.y)

        # X = torch.from_numpy(X).type(torch.float) # better way of doing it 
        # y = torch.from_numpy(y).type(torch.float)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx, s_att=True):
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()
        # return [self.X.iloc[idx].values, self.y[idx]]
        if s_att:
            return [self.X[idx], self.y[idx], self.a[idx]]
        else:
            return [self.X[idx], self.y[idx]]
    
    def standardlize_X(self, X_data):
        # Define the columns to standardize
        # columns_to_standardize = [28, 29, 30, 31, 32, 33]
        columns_to_standardize = [26, 27, 28, 29, 30, 31]
        scaler = StandardScaler()
        scaler.fit(X_data[:, columns_to_standardize])
        X_data[:, columns_to_standardize] = scaler.transform(X_data[:, columns_to_standardize])

        return X_data

    # def get_bld_w_prediction(self, prediction_test):
    #     self.df[self.target] = prediction_test
    #     return BinaryLabelDataset(df=self.df, label_names=[self.target], protected_attribute_names=['sex_1'])



class CompasDataset(Dataset):
    """Students Performance dataset."""

    def __init__(self, csv_file, df=None, crop=None, subset=None):
        """Initializes instance of class Compas Dataset.
        """

        if df is not None:
            self.df = df
        else:
            self.df = pd.read_csv(csv_file, index_col=False) #.drop("Unnamed: 0", axis=1)
        if crop:
            self.df = self.df[:crop]

        if subset:
            self.df = self.df[self.df.index.isin(subset)]
        
        self.target = "two_year_recid"
        self.s_attr = "sex"
        self.bld = BinaryLabelDataset(df=self.df, label_names=[self.target], protected_attribute_names=[self.s_attr])

        self.X = self.df.drop(self.target, axis=1).to_numpy().astype(np.float32)
        self.y = self.df[self.target].to_numpy().astype(np.float32)
        self.a = self.df[self.s_attr].to_numpy().astype(np.float32)

        # self.X = self.standardlize_X(self.X)
        self.size = len(self.y)

        # X = torch.from_numpy(X).type(torch.float) # better way of doing it 
        # y = torch.from_numpy(y).type(torch.float)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx, s_att=True):
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()
        # return [self.X.iloc[idx].values, self.y[idx]]
        if s_att:
            return [self.X[idx], self.y[idx], self.a[idx]]
        else:
            return [self.X[idx], self.y[idx]]
    

def get_bld_dataset_w_pred(test_dataset, prediction_test):
    new_df = test_dataset.df.copy(deep=True)
    new_df[test_dataset.target] = prediction_test
    # print("Check sum prediction equal: ", sum(prediction_test), sum(new_df[test_dataset.target]))
    return BinaryLabelDataset(df=new_df, label_names=[test_dataset.target], protected_attribute_names=['sex_1'])


# def df_to_dataset(df, dataset_name="adult"):
#         csv_file_val =  os.getcwd()+'/data/adult/adult_dummy.csv'
#         train_dataset = AdultDataset(csv_file_train)

def get_partition(p_idx, dataset="adult"):
    path_root = '/Users/zhouyi/Desktop/Fair_FL_new/data/' + dataset + 'adult/partition/' + str(p_idx)
    file_ls = os.listdir(path_root)
    partition_file_ls = [file for file in file_ls if '.npy' in file]
    partition_file = path_root + '/' + partition_file_ls[0]

    return partition_file


def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if  args.dataset == 'adult':

        # csv_file_train = os.getcwd()+'/data/adult/adult_encoded_80train.csv'
        # csv_file_test =  os.getcwd()+'/data/adult/adult_encoded_20test.csv'

        # csv_file_train = os.getcwd()+'/data/adult/adult_all_33col_80train.csv'
        # csv_file_test =  os.getcwd()+'/data/adult/adult_all_33col_20test.csv'

        csv_file_train = os.getcwd()+'/data/adult/adult_all_33col.csv'
        # csv_file_train = os.getcwd()+'/data/adult/adult_all_33col_70train_0.csv'
        csv_file_test =  os.getcwd()+'/data/adult/adult_all_33col_20test_0.csv'
        csv_file_val =  os.getcwd()+'/data/adult/adult_all_33col_10val_0.csv'

        train_dataset = AdultDataset(csv_file_train)
        test_dataset = AdultDataset(csv_file_test)
        partition_file = get_partition(args.partition_idx, dataset=args.dataset)
        user_groups =  np.load(partition_file, allow_pickle=True).item()
    
    elif args.dataset == 'compas':

        csv_file_train = os.getcwd()+'/data/compas/compas_encoded_all.csv'

        train_dataset = CompasDataset(csv_file_train)
        test_dataset = None
        partition_file = get_partition(args.partition_idx, dataset=args.dataset)
        user_groups =  np.load(partition_file, allow_pickle=True).item()


    # elif args.dataset == 'cifar':
    #     data_dir = '../data/cifar/'
    #     apply_transform = transforms.Compose(
    #         [transforms.ToTensor(),
    #          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    #     train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
    #                                    transform=apply_transform)

    #     test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
    #                                   transform=apply_transform)

    #     # sample training data amongst users
    #     if args.iid:
    #         # Sample IID user data from Mnist
    #         user_groups = cifar_iid(train_dataset, args.num_users)
    #     else:
    #         # Sample Non-IID user data from Mnist
    #         if args.unequal:
    #             # Chose uneuqal splits for every user
    #             raise NotImplementedError()
    #         else:
    #             # Chose euqal splits for every user
    #             user_groups = cifar_noniid(train_dataset, args.num_users)

    # elif args.dataset == 'mnist' or 'fmnist':
    #     if args.dataset == 'mnist':
    #         data_dir = '../data/mnist/'
    #     else:
    #         data_dir = '../data/fmnist/'

    #     apply_transform = transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.1307,), (0.3081,))])

    #     train_dataset = datasets.MNIST(data_dir, train=True, download=True,
    #                                    transform=apply_transform)

    #     test_dataset = datasets.MNIST(data_dir, train=False, download=True,
    #                                   transform=apply_transform)

    #     # sample training data amongst users
    #     if args.iid:
    #         # Sample IID user data from Mnist
    #         user_groups = mnist_iid(train_dataset, args.num_users)
    #     else:
    #         # Sample Non-IID user data from Mnist
    #         if args.unequal:
    #             # Chose uneuqal splits for every user
    #             user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
    #         else:
    #             # Chose euqal splits for every user
    #             user_groups = mnist_noniid(train_dataset, args.num_users)

    return train_dataset, test_dataset, user_groups


# def get_val_dataset(args):
#     if  args.dataset == 'adult':

#         csv_file_val =  os.getcwd()+'/data/adult/adult_all_33col_10val_0.csv'
#         val_dataset = AdultDataset(csv_file_val)

#         return val_dataset

