import os
import logging
import numpy as np
import random
import argparse
import csv
import matplotlib.pyplot as plt
from dataset import AdultDataset, CompasDataset, WCLDDataset, PTBDataset
import json

def iid_sampling(dataset, num_clients):
    """
    Sample I.I.D. client data from Adult dataset
    Equallly divide data samples into N groups based on its original order
    :param dataset:
    :param num_clients:
    :return: dict of data index
    """
    num_items = int(len(dataset)/num_clients)
    dict_clients, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_clients):
        dict_clients[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_clients[i])
    
    return dict_clients



def dirichlet_sampling(dataset, num_clients, alpha, attr, ratio=None):
    """
    Sample Non-I.I.D. client data from Adult dataset
    Equallly divide data samples into N groups based on its original order
    :param dataset:
    :param num_clients:
    :param attr: imbalanced partition based on attr attribute
    :param dirichlet: value of alpha parameter of dirichlet distribution, None if dirichlet is not chosen
    :param ratio: ratio of attribute split where attr=1, None if quantity split is not chosen
    :return: dict of data index
    """

    # if dirichlet:
    # attr = dataset.target
    train_labels = dataset.y.astype(np.float32)
    n_classes = len(set(train_labels))

    label_distribution = np.random.dirichlet([alpha]*num_clients, n_classes)
    class_idcs = [np.argwhere(train_labels == y).flatten()
                for y in range(n_classes)]

    # 
    client_idcs = [[] for _ in range(num_clients)]
    # client_idcs =  {i: np.array([]) for i in range(num_clients)}
    for k_idcs, fracs in zip(class_idcs, label_distribution):

        for i, idcs in enumerate(np.split(k_idcs,
                                        (np.cumsum(fracs)[:-1]*len(k_idcs)).
                                        astype(int))):
            client_idcs[i] += [idcs]

    dict_clients = {i: np.concatenate(client_idcs[i]) for i in range(num_clients)}

    return dict_clients
    


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--partition', type=str, default='diri', help='the data partitioning strategy')
    parser.add_argument('--n_clients', type=int, default=10,  help='number of workers in a distributed cluster')
    # parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    # "./data/adult/adult_all_33col_70train_0.csv"
    # parser.add_argument('--data_path', type=str, required=False, default="/data/adult/adult_all_33col.csv", help="Data directory")
    # parser.add_argument('--save_to_dir', type=str, required=False, default="/data/adult/partition/", help="Output directory")
    parser.add_argument('--partition_idx', type=str, required=False, default="0", help="Output directory")
    
    parser.add_argument('--alpha', type=float, default=0.5, help='The parameter for the dirichlet distribution for data partitioning')
    # parser.add_argument('--save_to', type=str, default='', help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--target_attr', type=str, default=None, help='Sampling based on target_attr')
    
    parser.add_argument('--dataset', type=str, default='adult', help='The dataset for partition')
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    # target_attr = "education-num"
    # n_clients = 5
    # alpha = 0.5
    
    args = get_args()

    
    # '/data/adult/adult_all_33col_70train_0.csv'

    if args.dataset == "adult":
        csv_file_train = os.getcwd()+"/data/adult/adult_all_33col.csv"
        target_attr = "income"
        # save_to_dir = "/data/adult/partition/"
        train_data = AdultDataset(csv_file_train)
    elif args.dataset == "compas":
        csv_file_train = os.getcwd()+"/data/compas/compas_encoded_all.csv"
        target_attr = "two_year_recid"
        train_data = CompasDataset(csv_file_train)
    elif args.dataset == "wcld":
        csv_file_train = os.getcwd()+"/data/wcld/wcld_60000.csv"
        train_data = WCLDDataset(csv_file_train)
        target_attr = "recid_180d"
    elif args.dataset == "ptb-xl":
        csv_file_train = os.getcwd()+"/data/ptb-xl/ptbxl_all_clean_new.csv"
        train_data = PTBDataset(csv_file_train, traces=False)
        target_attr = "NORM"
        print("Read from ptb-xl!")
    else:
        print("ERROR: Dataset not found!")

    if args.target_attr:
        target_attr = args.target_attr
    labels = train_data.y
    classes = list(set(labels))
    n_classes = len(classes)

    if args.partition == "diri":
        client_idcs = dirichlet_sampling(dataset=train_data, num_clients=args.n_clients, attr=target_attr, alpha=args.alpha, ratio=None)
    elif args.partition == "iid":
        client_idcs = iid_sampling(dataset=train_data, num_clients=args.n_clients)


    print("client_idcs type: ", type(client_idcs))

    data_name = csv_file_train.split("/")[-1].split('.')[0]
    file_name = "user_groups_%dclients_%.1falpha_%s_%s_%s.npy" %(args.n_clients, args.alpha, args.partition, args.target_attr, data_name)
    target_dir = os.getcwd() + "/data/" + args.dataset + "/partition/" + args.partition_idx 
    os.makedirs(target_dir, exist_ok=True)
    save_to_file = target_dir + "/"  + file_name
    # json.dump(client_idcs, open(save_to_file,'w'))

    np.save(save_to_file,client_idcs)

    print("Data partition successfully saved in: ", save_to_file)

    # json.load(open("text.txt"))


    # plt.figure(figsize=(12, 8))
    # plt.hist([labels[client_idcs[i]]for i in range(len(client_idcs))], stacked=True,
    #          bins=np.arange(min(labels)-0.5, max(labels) + 1.5, 1),
    #          label=["Client {}".format(i) for i in range(n_clients)],
    #          rwidth=0.5)
    # plt.xticks(np.arange(n_classes), list(set(train_data.y)))
    # plt.xlabel("Label type")
    # plt.ylabel("Number of samples")
    # plt.legend(loc="upper right")
    # plt.title("Display Label Distribution on Different Clients")
    # plt.show()

    n_clients = args.n_clients
   
    plt.figure(figsize=(12, 8))
    label_distribution = [[] for _ in range(n_classes)]
    for c_id in range(len(client_idcs)):
        idc = client_idcs[c_id]
        for idx in idc:
            label_distribution[int(labels[int(idx)])].append(c_id)

    plt.hist(label_distribution, stacked=True,
                bins=np.arange(-0.5, n_clients + 1.5, 1),
                label=classes, rwidth=0.5)
    plt.xticks(np.arange(n_clients), ["Client %d" %
                                        (c_id+1) for c_id in range(n_clients)])
    plt.xlabel("Client ID")
    plt.ylabel("Number of samples")
    plt.legend()
    plt.title("Display Attribute Distribution on Different Clients - %s" % args.target_attr)
    # plt.show()
    plot_file_name = save_to_file.split(".npy")[0] + "_plot.png"
    plt.savefig(plot_file_name)

    # args = get_args()
    # num = -1
    # dataset = []
    # with open(args.datadir, newline='') as csvfile:
    #     reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    #     for row in reader:
    #         if num == -1:
    #             header = row
    #         else:
    #             dataset.append(row)
    #             for i in range(len(dataset[-1])):
    #                 dataset[-1][i] = eval(dataset[-1][i])
    #         num += 1
    
    # class_id = 0
    # for i in range(len(header)):
    #     if header[i] == "Class":
    #         class_id = i
    #         break
    # dataset = np.array(dataset)
    # num_class = int(np.max(dataset[:,class_id])) + 1

    # net_dataidx_map = partition_data(dataset, class_id, num_class, args.partition, args.n_parties, args.beta, args.init_seed)
    # mkdirs(args.outputdir)
    # for i in range(args.n_parties):
    #     file_name = args.outputdir+str(i)+'.csv'
    #     os.system("touch "+file_name)
    #     with open(file_name, 'w', newline='') as f:
    #         writer = csv.writer(f)
    #         writer.writerow(header)
    #         writer.writerows(dataset[net_dataidx_map[i]])