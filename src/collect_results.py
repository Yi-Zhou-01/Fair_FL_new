import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import options
import pickle

def collect_n_save_results(args):

    if args.platform=="kaggle":
        stat_dir = "/kaggle/working/statistics"
    elif args.platform=="colab":
        stat_dir = "/content/drive/MyDrive/Fair_FL_new/save/statistics"
    elif args.platform=="azure":
        stat_dir = os.getcwd() + '/save/statistics'
    else:
        stat_dir =  os.getcwd() + '/save/statistics'

    exp_dir = stat_dir + "/" + str(args.idx)

    if args.example_folder:
        target_exp = args.example_folder
    else:
        target_exp = '{}_{}_frac{}_client{}_lr{}_part{}_beta{}_ep{}_{}_{}_ftep_{}'.\
            format(args.dataset, args.model, args.frac, args.num_users,
                args.lr, args.partition_idx, args.beta, args.epochs, args.local_ep, args.fairfed_ep, args.ft_ep)    # <------------- iid tobeadded
     
    print("target_exp: ", target_exp)
    
    exp_ls = [ exp_folder for exp_folder in os.listdir(exp_dir) if (target_exp in exp_folder) and ("stats" not in exp_folder)]
    
    print("Using results in folders: ")
    print(exp_ls)


    # read all stats
    test_eod_new = []
    test_acc_new = []
    test_eod_fedavg = []
    test_acc_fedavg = []

    test_acc_fairfed=[]
    test_eod_fairfed=[]

    test_eod_new_ft=[]
    test_acc_new_ft=[]

    for folder in exp_ls:
        csv_file = exp_dir + "/" + folder + "/stats.csv"

        df = pd.read_csv(csv_file)
        test_eod_new.append(df["test_eod_new"])
        test_acc_new.append(df["test_acc_new"])
        test_eod_fedavg.append(df["test_eod_fedavg"])
        test_acc_fedavg.append(df["test_acc_fedavg"])

        test_acc_fairfed.append(df["test_acc_fairfed"])
        test_eod_fairfed.append(df["test_eod_fairfed"])

        test_eod_new_ft.append(df["test_eod_new_ft"])
        test_acc_new_ft.append(df["test_acc_new_ft"])
        # print(df[:5])

    stats_multi_exp = dict.fromkeys(["test_eod_new", "test_acc_new", "test_eod_fedavg", "test_acc_fedavg"], [])
    stats_multi_exp["test_eod_new"] = test_eod_new
    stats_multi_exp["test_acc_new"] = test_acc_new
    stats_multi_exp["test_eod_fedavg"] = test_eod_fedavg
    stats_multi_exp["test_acc_fedavg"] = test_acc_fedavg

    stats_multi_exp["test_acc_fairfed"] = test_acc_fairfed
    stats_multi_exp["test_eod_fairfed"] = test_eod_fairfed

    stats_multi_exp["test_acc_new_ft"] = test_acc_new_ft
    stats_multi_exp["test_eod_new_ft"] = test_eod_new_ft

    print(np.array(test_acc_new))
    print(np.array(test_eod_new))
    print(np.array(test_eod_fedavg))
    # print(np.mean(test_acc_new_all, axis=0))
    # print(np.std(test_acc_new_all, axis=0))

    # Save all stats to object
    save_to_dir = "{}/{}_{}".format(exp_dir, "stats", target_exp)
    os.makedirs(save_to_dir, exist_ok=True)
    save_to_file = save_to_dir+"/stats_multi_exp.pkl"
    with open(save_to_file, 'wb') as outp:
        pickle.dump(stats_multi_exp, outp, pickle.HIGHEST_PROTOCOL)

    print("Successfully saved in: "+save_to_file )

    return save_to_file


    # # plot stats
    # fig, ax = plt.subplots(figsize=(8, 4))
    # X = list(range(len(test_acc_new_all)))
    # ax.plot(X, test_acc_new_all.mean(axis=0), alpha=0.5, color='blue', label='train', linewidth = 4.0)
    # # ax.plot(trees_grid, test_acc.mean(axis=1), alpha=0.5, color='red', label='cv', linewidth = 1.0)
    # # ax.fill_between(trees_grid, test_acc.mean(axis=1) - test_acc.std(axis=1), test_acc.mean(axis=1) + test_acc.std(axis=1), color='#888888', alpha=0.4)
    # ax.fill_between(X, test_acc_new_all.mean(axis=0) - 2*test_acc_new_all.std(axis=0), test_acc_new_all.mean(axis=0) + 2*test_acc_new_all.std(axis=0), color='#888888', alpha=0.2)
    # ax.legend(loc='best')
    # # ax.set_ylim([0.88,1.02])
    # ax.set_ylabel("Accuracy")
    # ax.set_xlabel("N_estimators")



    # print(os.listdir(exp_dir))

    # all_fl = ""
    # if args.fl_new:
    #     all_fl = all_fl + "new"
    # if args.fl_fairfed:
    #     all_fl = all_fl + "fairfed"    
    # statistics_dir = save_to_root+'/statistics/{}/{}_{}_{}_{}_frac{}_client{}_lr{}_part{}_beta{}_ep{}_{}_{}_ftep_{}_{}'.\
    #     format(args.idx, all_fl, args.debias, args.dataset, args.model, args.frac, args.num_users,
    #            args.lr, args.partition_idx, args.beta, args.epochs, args.local_ep, args.fairfed_ep, args.ft_ep, args.rep)    # <------------- iid tobeadded
     

    return None

def plot_mean_std(data_path, save_img=None, plot_ft=False):

    with open(data_path, 'rb') as inp:
        stats_all = pickle.load(inp)
        print("Loaded stats saved in: ", data_path)
    

    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10, 4))
    X = list(range(1,len(stats_all["test_eod_new"][0])+1))

    line_wid = 1.5
    alpha = 0.2

    target_data_1 = stats_all["test_eod_new"]
    ax1.plot(X, np.mean(target_data_1, axis=0), color='red', label='Our method', linewidth = line_wid)
    ax1.fill_between(X, np.mean(target_data_1, axis=0) - np.std(target_data_1, axis=0),np.mean(target_data_1, axis=0) + np.std(target_data_1, axis=0), color='red', alpha=alpha)


    target_data_2 = stats_all["test_eod_fedavg"]
    ax1.plot(X, np.mean(target_data_2, axis=0), color='blue', label='FedAvg', linewidth = line_wid)
    ax1.fill_between(X, np.mean(target_data_2, axis=0) - np.std(target_data_2, axis=0),np.mean(target_data_2, axis=0) + np.std(target_data_2, axis=0), color='navy', alpha=alpha)


    target_data_3 = stats_all["test_eod_fairfed"]
    ax1.plot(X, np.mean(target_data_3, axis=0), color='green', label='FairFed', linewidth = line_wid)
    ax1.fill_between(X, np.mean(target_data_3, axis=0) - np.std(target_data_3, axis=0),np.mean(target_data_3, axis=0) + np.std(target_data_3, axis=0), color='green', alpha=alpha)



    target_data_4 = stats_all["test_acc_new"]
    ax2.plot(X, np.mean(target_data_4, axis=0),color='red', label='Our method', linestyle='dashed', linewidth = line_wid)
    ax2.fill_between(X, np.mean(target_data_4, axis=0) - np.std(target_data_4, axis=0),np.mean(target_data_4, axis=0) + np.std(target_data_4, axis=0), color='pink', alpha=alpha)


    target_data_5 = stats_all["test_acc_fedavg"]
    ax2.plot(X, np.mean(target_data_5, axis=0), color='blue', label='FedAvg',linestyle='dashed', linewidth = line_wid)
    ax2.fill_between(X, np.mean(target_data_5, axis=0) - np.std(target_data_5, axis=0),np.mean(target_data_5, axis=0) + np.std(target_data_5, axis=0), color='lightblue', alpha=alpha)


    target_data_6 = stats_all["test_acc_fairfed"]
    ax2.plot(X, np.mean(target_data_6, axis=0), color='green', label='FairFed',linestyle='dashed', linewidth = line_wid)
    ax2.fill_between(X, np.mean(target_data_6, axis=0) - np.std(target_data_6, axis=0),np.mean(target_data_6, axis=0) + np.std(target_data_6, axis=0), color='lightgreen', alpha=alpha)


    title = data_path.split("/")[-2]
    # ax1.set_title(title)
    fig.suptitle(title, fontsize=12)
    ax1.legend(loc='best')
    # ax.set_ylim([0.88,1.02])
    ax1.set_ylabel("EOD")
    ax2.set_ylabel("Accuracy")
    ax1.set_xlabel("Client #")
    ax2.set_xlabel("Client #")
    ax1.set_xticks(np.arange(min(X), max(X)+1, 1.0))
    ax2.set_xticks(np.arange(min(X), max(X)+1, 1.0))

    
    if save_img:
        save_to = data_path.split("stats_multi_exp")[0] + "all_exp_stats_plot.png"
        plt.savefig(save_to)
        print("Stats plot successfully saved in: {} !".format(save_to))
    



if __name__ == '__main__':
    args = options.args_parser()

    stats_path = collect_n_save_results(args)

    plot_mean_std(stats_path, save_img=True)
    
