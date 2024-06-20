
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

    def __init__(self, csv_file,crop=None, subset=None):
        """Initializes instance of class StudentsPerformanceDataset.

        Args:
            csv_file (str): Path to the csv file with the students data.

        """

        self.df = pd.read_csv(csv_file, index_col=False) #.drop("Unnamed: 0", axis=1)
        if crop:
            self.df = self.df[:crop]

        if subset:
            self.df = self.df[self.df.index.isin(subset)]

        self.bld = BinaryLabelDataset(df=self.df, label_names=['income'], protected_attribute_names=['sex_1'])
        self.target = "income"
        # self.sensitive = ""

        # self.X = self.df.drop(self.target, axis=1).astype(np.float32)
        # self.y = self.df[self.target].astype(np.float32)
        # self.train_labels = self.y
        # self.length = len(self.df)

        self.X = self.df.drop(self.target, axis=1).to_numpy().astype(np.float32)
        self.y = self.df[self.target].to_numpy().astype(np.float32)

        self.X = standardlize_X(self.X)

        # X = torch.from_numpy(X).type(torch.float) # better way of doing it 
        # y = torch.from_numpy(y).type(torch.float)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()
        # return [self.X.iloc[idx].values, self.y[idx]]
        return [self.X[idx], self.y[idx]]
    


def get_bld_predict(test_dataset, prediction_test):
    test_dataset.df["income"] = prediction_test
    return BinaryLabelDataset(df=test_dataset.df, label_names=['income'], protected_attribute_names=['sex_1'])

def standardlize_X(X_data):
    

    # Define the columns to standardize
    # columns_to_standardize = [28, 29, 30, 31, 32, 33]
    columns_to_standardize = [26, 27, 28, 29, 30, 31]

    # Create a StandardScaler object
    scaler = StandardScaler()

    # Convert X_train and X_test to NumPy arrays
    # X_train_np = X_train.numpy()
    # X_test_np = X_test.numpy()

    # Fit the scaler on the training data columns
    scaler.fit(X_data[:, columns_to_standardize])

    # Standardize the selected columns in both X_train and X_test
    X_data[:, columns_to_standardize] = scaler.transform(X_data[:, columns_to_standardize])
    # X_test_np[:, columns_to_standardize] = scaler.transform(X_test_np[:, columns_to_standardize])
    # print(X_train_np.head())

    # Convert back to PyTorch tensors
    # X_train = torch.from_numpy(X_train_np).type(torch.float) 
    # X_test = torch.from_numpy(X_test_np).type(torch.float)
    return X_data





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

        csv_file_train = os.getcwd()+'/data/adult/adult_all_33col_70train_0.csv'
        csv_file_test =  os.getcwd()+'/data/adult/adult_all_33col_20test_0.csv'
        csv_file_val =  os.getcwd()+'/data/adult/adult_all_33col_10val_0.csv'

        # train_dataset = AdultDataset(csv_file_train)
        test_dataset = AdultDataset(csv_file_test)

        if args.iid:
            # Sample IID user data from Mnist
            train_dataset = AdultDataset(csv_file_train)
            user_groups = adult_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                crop = 35600
                train_dataset = AdultDataset(csv_file_train, crop)
                # Chose euqal splits for every user
                # user_groups = adult_noniid(train_dataset, args.num_users)
                partition_file = os.getcwd() + "/data/adult/partition/user_groups_10clients_0alpha_diri_income_adult_all_33col_70train_0.npy" 
                user_groups = sampling.adult_noniid_new(train_dataset, args.num_users, partition_file)


    elif args.dataset == 'cifar':
        data_dir = '../data/cifar/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups = cifar_noniid(train_dataset, args.num_users)

    elif args.dataset == 'mnist' or 'fmnist':
        if args.dataset == 'mnist':
            data_dir = '../data/mnist/'
        else:
            data_dir = '../data/fmnist/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = mnist_noniid(train_dataset, args.num_users)

    return train_dataset, test_dataset, user_groups


def get_val_dataset(args):
    if  args.dataset == 'adult':

        csv_file_val =  os.getcwd()+'/data/adult/adult_all_33col_10val_0.csv'
        val_dataset = AdultDataset(csv_file_val)

        return val_dataset

