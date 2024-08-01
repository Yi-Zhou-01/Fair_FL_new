
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

import h5py
import cv2

class AdultDataset(Dataset):
    """Students Performance dataset."""

    def __init__(self, csv_file, X=None, y=None, a=None, df=None, crop=None, subset=None):
        """Initializes instance of Adult dataset.
        """
        self.target = "income"
        self.s_attr = "sex_1"
        self.name = "adult"

        if X is None:
            if df is not None:
                df = df
            else:
                df = pd.read_csv(csv_file, index_col=False) #.drop("Unnamed: 0", axis=1)
            if crop:
                df = df[:crop]

            if subset:
                df = df[df.index.isin(subset)]

            self.X = df.drop([self.target, self.s_attr], axis=1).to_numpy().astype(np.float32)
            self.X = self.standardlize_X(self.X)
            self.y = df[self.target].to_numpy().astype(np.float32)
            self.a = df[self.s_attr].to_numpy().astype(np.float32)

        else:
            self.X = X.to_numpy().astype(np.float32)
            self.X = self.standardlize_X(self.X)
            self.y = y.to_numpy().astype(np.float32).flatten()
            self.a = a.to_numpy().astype(np.float32).flatten()


        self.size = len(self.y)

        # X = torch.from_numpy(X).type(torch.float) # better way of doing it 
        # y = torch.from_numpy(y).type(torch.float)

    def __len__(self):
        return len(self.y)

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
        # columns_to_standardize = [26, 27, 28, 29, 30, 31]
        columns_to_standardize = list(range(25, len(self.X[0])))
        scaler = StandardScaler()
        scaler.fit(X_data[:, columns_to_standardize])
        X_data[:, columns_to_standardize] = scaler.transform(X_data[:, columns_to_standardize])

        return X_data

    # def get_bld_w_prediction(self, prediction_test):
    #     self.df[self.target] = prediction_test
    #     return BinaryLabelDataset(df=self.df, label_names=[self.target], protected_attribute_names=['sex_1'])



class CompasDataset(Dataset):
    """Students Performance dataset."""

    def __init__(self, csv_file, X=None, y=None, a=None, df=None, crop=None, subset=None):
        """Initializes instance of class Compas Dataset.
        """

        self.target = "two_year_recid"
        self.s_attr = "sex"
        self.name = "compas"

        if X is None:
            if df is not None:
                df = df
            else:
                df = pd.read_csv(csv_file, index_col=False) #.drop("Unnamed: 0", axis=1)
            
            if crop:
                df = df[:crop]

            if subset:
                df = df[df.index.isin(subset)]
            

            # self.bld = BinaryLabelDataset(df=self.df, label_names=[self.target], protected_attribute_names=[self.s_attr])

            # self.X = self.df.drop(self.target, axis=1).to_numpy().astype(np.float32)
            self.X = df.drop([self.target, self.s_attr], axis=1).to_numpy().astype(np.float32)
            self.X = self.standardlize_X(self.X)
            self.y = df[self.target].to_numpy().astype(np.float32)
            self.a = df[self.s_attr].to_numpy().astype(np.float32)
        
        else:
            self.X = X.to_numpy().astype(np.float32)
            self.X = self.standardlize_X(self.X)
            self.y = y.to_numpy().astype(np.float32).flatten()
            self.a = a.to_numpy().astype(np.float32).flatten()


        # new_df = pd.DataFrame()
        # new_df["y"] = self.y
        # new_df["a"] = self.a
        # self.bld =  BinaryLabelDataset(df=new_df, label_names=["y"], protected_attribute_names=["a"])

        self.size = len(self.y)

    def __len__(self):
        return len(self.y)

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
        # columns_to_standardize = [26, 27, 28, 29, 30, 31]
        columns_to_standardize = list(range(len(self.X[0]))) # standardize all
        scaler = StandardScaler()
        scaler.fit(X_data[:, columns_to_standardize])
        X_data[:, columns_to_standardize] = scaler.transform(X_data[:, columns_to_standardize])

        return X_data



class WCLDDataset(Dataset):
    """Students Performance dataset."""

    def __init__(self, csv_file, X=None, y=None, a=None, df=None, crop=None, subset=None):
        """Initializes instance of class Compas Dataset.
        """

        self.target = "recid_180d"
        self.s_attr = "sex"
        self.name = "wcld"

        if X is None:
            if df is not None:
                df = df
            else:
                df = pd.read_csv(csv_file, index_col=False) #.drop("Unnamed: 0", axis=1)
            if crop:
                df = df[:crop]
            if subset:
                df = df[df.index.isin(subset)]
        
            self.X = df.drop([self.target, self.s_attr], axis=1).to_numpy().astype(np.float32)
            self.X = self.standardlize_X(self.X)
            self.y = df[self.target].to_numpy().astype(np.float32)
            self.a = df[self.s_attr].to_numpy().astype(np.float32)
        

        else:
            self.X = X.to_numpy().astype(np.float32)
            self.X = self.standardlize_X(self.X)
            self.y = y.to_numpy().astype(np.float32).flatten()
            self.a = a.to_numpy().astype(np.float32).flatten()

        self.size = len(self.y)

    def __len__(self):
        return len(self.y)

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
        # columns_to_standardize = [26, 27, 28, 29, 30, 31]
        columns_to_standardize = list(range(9)) # standardize all
        scaler = StandardScaler()
        scaler.fit(X_data[:, columns_to_standardize])
        X_data[:, columns_to_standardize] = scaler.transform(X_data[:, columns_to_standardize])

        return X_data


class PTBDataset(Dataset):
    """Students Performance dataset."""

    def __init__(self, csv_file, X=None, y=None, a=None, kaggle=False, df=None, crop=None, subset=None, traces=True):
        """Initializes instance of class Compas Dataset.
        """
        self.target = "NORM"
        self.s_attr = "sex"
        self.name = "ptb-xl"

        if X is None:
            if df is not None:
                df = df
            else:
                df = pd.read_csv(csv_file, index_col=False)#[:1000] #.drop("Unnamed: 0", axis=1)
                # print("self.df: ", self.df[:5])
                columns = ["record_id", "ecg_id","patient_id","age","sex", "NORM"]
                df = df.loc[:, df.columns.isin(columns)]
            if crop:
                df = df[:crop]

            if subset:
                df = df[df.index.isin(subset)]
            

            # self.bld = BinaryLabelDataset(df=self.df, label_names=[self.target], protected_attribute_names=[self.s_attr])

            # self.X = self.df.drop(self.target, axis=1).to_numpy().astype(np.float32)
            if traces:
                if kaggle:
                    path_to_traces = "/kaggle/input/ptb-xl/ptbxl_all_clean_new_100hz.hdf5"
                else:
                    path_to_traces = os.getcwd() + "/data/ptb-xl/ptbxl_all_clean_new_100hz.hdf5"
                f = h5py.File(path_to_traces, 'r')
                self.X = np.array(f["tracings"][:])#[:1000] 
            else:
                self.X = df["record_id"].to_numpy().astype(np.float32)
            

            self.y = df[self.target].to_numpy().astype(np.float32)
            self.a = df[self.s_attr].to_numpy().astype(np.float32)

            # self.X = self.standardlize_X(self.X)
            


        else:
            if isinstance(X,(np.ndarray)):
                self.X = X.astype(np.float32)
            else:
                self.X = X.to_numpy().astype(np.float32)
            # self.X = self.standardlize_X(self.X)
            self.y = y.to_numpy().astype(np.float32).flatten()
            self.a = a.to_numpy().astype(np.float32).flatten()

        self.size = len(self.y)

        # X = torch.from_numpy(X).type(torch.float) # better way of doing it 
        # y = torch.from_numpy(y).type(torch.float)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx, s_att=True):
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()
        # return [self.X.iloc[idx].values, self.y[idx]]

        if s_att:
            return [self.X[idx], self.y[idx], self.a[idx]]
        else:
            return [self.X[idx], self.y[idx]]




class NIHDataset(Dataset):
    """Students Performance dataset."""

    def __init__(self, csv_file, X=None, y=None, a=None, kaggle=False, df=None, crop=None, subset=None, traces=True):
        """Initializes instance of class Compas Dataset.
        """
        self.target = "Disease"
        self.s_attr = "Patient Gender"
        self.name = "nih-chest"

        if X is None:
            if df is not None:
                df = df
            else:
                df = pd.read_csv(csv_file, index_col=False)#[:1000] #.drop("Unnamed: 0", axis=1)
                # print("self.df: ", self.df[:5])
                columns = ["Image Index", "Patient Gender","Disease","Multi_label", "folder_name"]
                df = df.loc[:, df.columns.isin(columns)]
            
            if not kaggle:
                crop = 1000

            if crop:
                df = df[:crop]

            if subset:
                df = df[df.index.isin(subset)]
            

            # self.bld = BinaryLabelDataset(df=self.df, label_names=[self.target], protected_attribute_names=[self.s_attr])

            # self.X = self.df.drop(self.target, axis=1).to_numpy().astype(np.float32)
            if traces:
                if kaggle:
                    self.path_to_traces = "/kaggle/input/data"
                else:
                    self.path_to_traces = "/Users/zhouyi/Desktop/Msc Project/nih-chest/png"
                # f = h5py.File(path_to_traces, 'r')
                # self.X = np.array(f["tracings"][:])#[:1000] 
            else:
                self.path_to_traces = None
            

            self.X = df["Image Index"].to_numpy()
            self.y = df[self.target].to_numpy().astype(np.float32)
            self.a = df[self.s_attr].to_numpy().astype(np.float32)
            self.folder_name = df["folder_name"].to_numpy()

            # self.X = self.standardlize_X(self.X)
            


        else:
            if isinstance(X,(np.ndarray)):
                self.X = X.astype(np.float32)
            else:
                self.X = X.to_numpy().astype(np.float32)
            # self.X = self.standardlize_X(self.X)
            self.y = y.to_numpy().astype(np.float32).flatten()
            self.a = a.to_numpy().astype(np.float32).flatten()

        self.size = len(self.y)

        # X = torch.from_numpy(X).type(torch.float) # better way of doing it 
        # y = torch.from_numpy(y).type(torch.float)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx, s_att=True):
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()
            print("list index !!")
        # return [self.X.iloc[idx].values, self.y[idx]]
 
        
        folder_name = self.folder_name[idx]
        img_name = self.X[idx]
        img_path = self.path_to_traces + "/" + folder_name + "/images/" + img_name
        # img_path = self.path_to_traces + "/" +  img_name
        
        # img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)

        if s_att:
            return [img.astype(np.float32), self.y[idx], self.a[idx]]
        else:
            return [img.astype(np.float32), self.y[idx]]





def get_bld_dataset_w_pred(a, pred_labels):

    new_df = pd.DataFrame()
    # new_df["X"] = X
    new_df["a"] = list(a)
    new_df["y"] =  list(pred_labels)
    # new_df = test_dataset.df.copy(deep=True)
    # new_df[test_dataset.target] = prediction_test
    # print("Check sum prediction equal: ", sum(prediction_test), sum(new_df[test_dataset.target]))
    return BinaryLabelDataset(df=new_df, label_names=["y"], protected_attribute_names=["a"])


# def df_to_dataset(df, dataset_name="adult"):
#         csv_file_val =  os.getcwd()+'/data/adult/adult_dummy.csv'
#         train_dataset = AdultDataset(csv_file_train)

def get_partition(kaggle, p_idx, dataset="adult"):
    if kaggle:
        path_root = "/kaggle/input/" + dataset + '/partition/' + str(p_idx)
    else:
        path_root = '/Users/zhouyi/Desktop/Fair_FL_new/data/' + dataset + '/partition/' + str(p_idx)
    file_ls = os.listdir(path_root)
    partition_file_ls = [file for file in file_ls if '.npy' in file]
    partition_file = path_root + '/' + partition_file_ls[0]

    return partition_file


def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if args.kaggle:
        data_path = "/kaggle/input"
    else:
        data_path = os.getcwd()+"/data"

    if  args.dataset == 'adult':

        # csv_file_train = os.getcwd()+'/data/adult/adult_encoded_80train.csv'
        # csv_file_test =  os.getcwd()+'/data/adult/adult_encoded_20test.csv'

        # csv_file_train = os.getcwd()+'/data/adult/adult_all_33col_80train.csv'
        # csv_file_test =  os.getcwd()+'/data/adult/adult_all_33col_20test.csv'

        csv_file_train = data_path+'/adult/adult_all_33col.csv'
        # csv_file_train = os.getcwd()+'/data/adult/adult_all_33col_70train_0.csv'
        csv_file_test =  data_path+'/adult/adult_all_33col_20test_0.csv'
        csv_file_val =  data_path+'/adult/adult_all_33col_10val_0.csv'

        train_dataset = AdultDataset(csv_file_train)
        test_dataset = AdultDataset(csv_file_test)
        partition_file = get_partition(args.kaggle, args.partition_idx, dataset=args.dataset)
        user_groups =  np.load(partition_file, allow_pickle=True).item()
    
    elif args.dataset == 'compas':

        csv_file_train =data_path+'/compas/compas_encoded_all.csv'

        train_dataset = CompasDataset(csv_file_train)
        test_dataset = train_dataset # Dummy test dataset: Not used for testing
        partition_file = get_partition(args.kaggle, args.partition_idx, dataset=args.dataset)
        user_groups =  np.load(partition_file, allow_pickle=True).item()

    elif args.dataset == 'wcld':

        csv_file_train =data_path+'/wcld/wcld_60000.csv'

        train_dataset = WCLDDataset(csv_file_train)
        test_dataset = train_dataset # Dummy test dataset: Not used for testing
        partition_file = get_partition(args.kaggle, args.partition_idx, dataset=args.dataset)
        user_groups =  np.load(partition_file, allow_pickle=True).item()
    
    elif args.dataset == 'ptb-xl':

        csv_file_train = data_path+'/ptb-xl/ptbxl_all_clean_new.csv'

        train_dataset = PTBDataset(csv_file_train, kaggle=args.kaggle)
        test_dataset = train_dataset # Dummy test dataset: Not used for testing
        partition_file = get_partition(args.kaggle, args.partition_idx, dataset=args.dataset)
        user_groups =  np.load(partition_file, allow_pickle=True).item()

    elif args.dataset == 'nih-chest':
        if not args.kaggle:
            data_path = "/Users/zhouyi/Desktop/Msc Project"
        csv_file_train = data_path+'/nih-chest/nih_chest_all_clean.csv'

        train_dataset = NIHDataset(csv_file_train, kaggle=args.kaggle)
        test_dataset = train_dataset # Dummy test dataset: Not used for testing
        partition_file = get_partition(args.kaggle, args.partition_idx, dataset=args.dataset)
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


# def get_dataset_from_df(dataset_name, df, kaggle):
#     """ Returns Dataset object
#     """

#     if  dataset_name == 'adult':
#         return AdultDataset(csv_file="", df=df)
#     elif  dataset_name == 'compas':
#         return CompasDataset(csv_file="", df=df)
#     elif  dataset_name == 'wcld':
#         return WCLDDataset(csv_file="", df=df)
#     elif  dataset_name == 'ptb-xl':
#         return PTBDataset(csv_file="", df=df,  kaggle=kaggle)
