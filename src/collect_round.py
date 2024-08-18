import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import options
import pickle

def collect_rounds(args, rounds=[2]):

    if args.platform=="kaggle":
        stat_dir = "/kaggle/working/statistics"
    elif args.platform=="colab":
        stat_dir = "/content/drive/MyDrive/Fair_FL_new/save/statistics"
    elif args.platform=="azure":
        stat_dir = os.getcwd() + '/save/statistics'
    else:
        stat_dir =  os.getcwd() + '/save/statistics'

    exp_dir = stat_dir + "/" + str(args.idx)


    stats_rounds = {}
    stats_rounds["test_eod_new"] = []
    stats_rounds["test_eod_fedavg"] = []
    stats_rounds["test_eod_fairfed"] = []
    stats_rounds["test_acc_new"] = []
    stats_rounds["test_acc_fedavg"] = []
    stats_rounds["test_acc_fairfed"] = []
    # dict.fromkeys(["test_eod_new", "test_acc_new", "test_eod_fedavg", "test_acc_fedavg", "test_acc_fairfed", "test_eod_fairfed"], [])

    for round_idx in rounds:
        target_exp = 'stats_{}_{}_frac{}_client{}_lr{}_part{}_beta{}_ep{}_{}_{}_ftep_{}'.\
            format(args.dataset, args.model, args.frac, args.num_users,
                args.lr, args.partition_idx, args.beta, round_idx, args.local_ep, round_idx, args.ft_ep)    # <------------- iid tobeadded
        
        data_path = exp_dir + "/" + target_exp + "/stats_multi_exp.pkl"

        with open(data_path, 'rb') as inp:
            stats_all = pickle.load(inp)
            print("Round:", round_idx, " Loaded stats saved in: ", data_path)
        
        axis = 0
        print(stats_all["test_eod_new"])
        print(np.mean(stats_all["test_eod_new"], axis=axis))
       
        stats_rounds["test_eod_new"].append(np.mean(stats_all["test_eod_new"], axis=axis))
        stats_rounds["test_eod_fedavg"].append(np.mean(stats_all["test_eod_fedavg"], axis=axis))
        stats_rounds["test_eod_fairfed"].append(np.mean(stats_all["test_eod_fairfed"], axis=axis))
        stats_rounds["test_acc_new"].append(np.mean(stats_all["test_acc_new"], axis=axis))
        stats_rounds["test_acc_fedavg"].append(np.mean(stats_all["test_acc_fedavg"], axis=axis))
        stats_rounds["test_acc_fairfed"].append(np.mean(stats_all["test_acc_fairfed"], axis=axis))

    print(len(stats_rounds["test_eod_new"]))
    print((stats_rounds["test_eod_new"]))

    # Save all stats to object
    save_to_dir = "{}/round_{}".format(exp_dir, target_exp)
    os.makedirs(save_to_dir, exist_ok=True)
    save_to_file = save_to_dir+"/stats_round_exp.pkl"
    with open(save_to_file, 'wb') as outp:
        pickle.dump(stats_rounds, outp, pickle.HIGHEST_PROTOCOL)

    print("Successfully saved in: "+save_to_file )

    return save_to_file


def plot_mean_std_rounds(data_path, rounds, save_img=None, plot_ft=False):

    with open(data_path, 'rb') as inp:
        stats_all = pickle.load(inp)

    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10, 4))
    # X = list(range(1,len(stats_all["test_eod_new"])+1))

    X = [str(r) for r in rounds]

    line_wid = 1.5
    alpha = 0.2
    axis = 1

    target_data_1 = stats_all["test_eod_new"]
    ax1.plot(X, np.mean(target_data_1, axis=axis), color='red', label='Our method', linewidth = line_wid)
    ax1.fill_between(X, np.mean(target_data_1, axis=axis) - np.std(target_data_1, axis=axis),np.mean(target_data_1, axis=axis) + np.std(target_data_1, axis=axis), color='red', alpha=alpha)


    target_data_2 = stats_all["test_eod_fedavg"]
    ax1.plot(X, np.mean(target_data_2, axis=axis), color='blue', label='FedAvg', linewidth = line_wid)
    ax1.fill_between(X, np.mean(target_data_2, axis=axis) - np.std(target_data_2, axis=axis),np.mean(target_data_2, axis=axis) + np.std(target_data_2, axis=axis), color='navy', alpha=alpha)


    target_data_3 = stats_all["test_eod_fairfed"]
    ax1.plot(X, np.mean(target_data_3, axis=axis), color='green', label='FairFed', linewidth = line_wid)
    ax1.fill_between(X, np.mean(target_data_3, axis=axis) - np.std(target_data_3, axis=axis),np.mean(target_data_3, axis=axis) + np.std(target_data_3, axis=axis), color='green', alpha=alpha)



    target_data_4 = stats_all["test_acc_new"]
    ax2.plot(X, np.mean(target_data_4, axis=axis),color='red', label='Our method', linestyle='dashed', linewidth = line_wid)
    ax2.fill_between(X, np.mean(target_data_4, axis=axis) - np.std(target_data_4, axis=axis),np.mean(target_data_4, axis=axis) + np.std(target_data_4, axis=axis), color='pink', alpha=alpha)


    target_data_5 = stats_all["test_acc_fedavg"]
    ax2.plot(X, np.mean(target_data_5, axis=axis), color='blue', label='FedAvg',linestyle='dashed', linewidth = line_wid)
    ax2.fill_between(X, np.mean(target_data_5, axis=axis) - np.std(target_data_5, axis=axis),np.mean(target_data_5, axis=axis) + np.std(target_data_5, axis=axis), color='lightblue', alpha=alpha)


    target_data_6 = stats_all["test_acc_fairfed"]
    ax2.plot(X, np.mean(target_data_6, axis=axis), color='green', label='FairFed',linestyle='dashed', linewidth = line_wid)
    ax2.fill_between(X, np.mean(target_data_6, axis=axis) - np.std(target_data_6, axis=axis),np.mean(target_data_6, axis=axis) + np.std(target_data_6, axis=axis), color='lightgreen', alpha=alpha)


    title = data_path.split("/")[-2]
    fig.suptitle(title, fontsize=12)
    ax1.legend(loc='best')  
    ax1.set_ylabel("EOD")
    ax2.set_ylabel("Accuracy")
    ax1.set_xlabel("Round #")
    ax2.set_xlabel("Round #")

    ax2.axhline(0.8, color='lightsteelblue', alpha=0.6)
    ax1.axhline(0.1, color='orange', alpha=0.4)
    ax1.axhline(0.0, color='black', alpha=0.6)


    if save_img:
        save_to = data_path.split("stats_multi_exp")[0] + "rounds_exp_stats_plot.png"
        plt.savefig(save_to)
        print("Stats plot successfully saved in: {} !".format(save_to))



if __name__ == '__main__':
    args = options.args_parser()

    # rounds = [2,4,6,8,10]
    # rounds = [2,4,6,8,10, 12,14,16]
    # rounds=[10,20,28,36]
    # rounds=[18,24,30]
    # rounds=[10, 15, 20, 25, 30, 35]
    rounds=args.rounds_ls

    stats_path = collect_rounds(args, rounds=rounds)

    plot_mean_std_rounds(stats_path,rounds, save_img=True)
    
