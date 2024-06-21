import os
import logging
import numpy as np
import random
import argparse
import csv
import matplotlib.pyplot as plt
from dataset import AdultDataset
import json

def adult_iid(dataset, num_clients):
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



def adult_noniid(dataset, num_clients, attr="income", dirichlet=None, ratio=None):
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

    if dirichlet:
        # Would one hot encoding be a problem? Would be for features with multiple classes
        # number of class for target label-- 2 for binary classification

        train_labels = dataset.df[attr].to_numpy().astype(np.float32)
        n_classes = len(set(train_labels))

        # n_classes = 2 # hard code for now
        # train_labels = dataset.y
        alpha = dirichlet
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
        # [np.concatenate(idcs) for idcs in client_idcs]

        # print("dict_clients")
        # print(dict_clients)
        return dict_clients
    


    # #----------------------------------------------------
    # num_shards, num_imgs = 178, 200
    # idx_shard = [i for i in range(num_shards)]
    # dict_users = {i: np.array([]) for i in range(num_users)}
    # idxs = np.arange(num_shards*num_imgs)
    # # labels = dataset.train_labels.numpy()
    # labels = np.array(dataset.train_labels)

    # # sort labels
    # idxs_labels = np.vstack((idxs, labels))
    # idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    # idxs = idxs_labels[0, :]

    # # divide and assign 2 shards/client
    # for i in range(num_users):
    #     # Random choose two shards for each client
    #     # print("idx_shard: ", len(idx_shard))
    #     rand_set = set(np.random.choice(idx_shard, 2, replace=False))
    #     idx_shard = list(set(idx_shard) - rand_set)
    #     # Add the data within the selected shard to the corresponding client
    #     for rand in rand_set:
    #         dict_users[i] = np.concatenate(
    #             (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    # # return dict_users

    # #----------------------------------------------------

    # np.random.seed(seed)
    # random.seed(seed)

    # n_train = dataset.shape[0]
    # y_train = dataset[:,class_id]

    # if partition == "noniid-labeldir":
    #     min_size = 0
    #     min_require_size = 10

    #     N = dataset.shape[0]
    #     net_dataidx_map = {}

    #     while min_size < min_require_size:
    #         idx_batch = [[] for _ in range(n_parties)]

    #         for k in range(K):
    #             idx_k = np.where(y_train == k)[0]
    #             np.random.shuffle(idx_k)
    #             proportions = np.random.dirichlet(np.repeat(beta, n_parties))
    #             # logger.info("proportions1: ", proportions)
    #             # logger.info("sum pro1:", np.sum(proportions))
    #             ## Balance
    #             proportions = np.array([p * (len(idx_j) < N / n_parties) for p, idx_j in zip(proportions, idx_batch)])
    #             # logger.info("proportions2: ", proportions)
    #             proportions = proportions / proportions.sum()
    #             # logger.info("proportions3: ", proportions)
    #             proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
    #             # logger.info("proportions4: ", proportions)
    #             idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
    #             min_size = min([len(idx_j) for idx_j in idx_batch])
    #             # if K == 2 and n_parties <= 10:
    #             #     if np.min(proportions) < 200:
    #             #         min_size = 0
    #             #         break


    #     for j in range(n_parties):
    #         np.random.shuffle(idx_batch[j])
    #         net_dataidx_map[j] = idx_batch[j]


    # return dict_clients


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--partition', type=str, default='diri', help='the data partitioning strategy')
    parser.add_argument('--n_clients', type=int, default=10,  help='number of workers in a distributed cluster')
    # parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    parser.add_argument('--data_path', type=str, required=False, default="./data/adult/adult_all_33col_70train_0.csv", help="Data directory")
    parser.add_argument('--save_to_dir', type=str, required=False, default="/data/adult/partition/", help="Output directory")
    parser.add_argument('--alpha', type=float, default=0.5, help='The parameter for the dirichlet distribution for data partitioning')
    # parser.add_argument('--save_to', type=str, default='', help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--target_attr', type=str, default='education-num', help='The parameter for the dirichlet distribution for data partitioning')
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    # target_attr = "education-num"
    # n_clients = 5
    # alpha = 0.5
    
    args = get_args()

    csv_file_train = os.getcwd()+'/data/adult/adult_all_33col_70train_0.csv'

    train_data = AdultDataset(csv_file_train)
    labels = train_data.df[args.target_attr].to_numpy()
    classes = list(set(labels))
    n_classes = len(classes)

    client_idcs = adult_noniid(dataset=train_data, num_clients=args.n_clients, attr=args.target_attr, dirichlet=args.alpha, ratio=None)
    
    print("client_idcs type: ", type(client_idcs))

    data_name = csv_file_train.split("/")[-1].split('.')[0]
    file_name = "user_groups_%dclients_%falpha_%s_%s_%s.npy" %(args.n_clients, args.alpha, args.partition, args.target_attr, data_name)
    save_to_file = os.getcwd() + args.save_to_dir + file_name
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