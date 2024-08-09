import numpy as np
import matplotlib.pyplot as plt
import os


def plot_mean_sd(ls):

    plt.figure(figsize=(12, 8))

    x = list(range(len(ls)))
    plt.plot(x, ls, 'k-')
    plt.fill_between(x, y-error, y+error)

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


def plot_before_after(before_ls, after_ls):

    plt.figure(figsize=(12, 8))

    x = list(range(len(before_ls)))
    plt.plot(x, before_ls, color='blue', label='before')
    plt.plot(x, after_ls,  color="red", label='after')
    plt.xlabel("Client ID")
    plt.ylabel("")
    plt.title("Statistics on Different Clients - %s", "" )
    plt.legend()
    plt.show()



def plot_acc_eod(acc_before, acc_after, eod_before, eod_after, save_to=None):

    plt.figure(figsize=(12, 8))

    x = list(range(len(acc_before)))
    plt.plot(x, acc_before, color='blue',  label='acc_before')
    plt.plot(x, acc_after,  color="red", label='acc_after')

    plt.plot(x, eod_before, color='blue', linestyle='dashed', label='eod_before')
    plt.plot(x, eod_after,  color="red", linestyle='dashed', label='eod_after')

    plt.xlabel("Client ID")
    plt.xticks(np.arange(min(x), max(x)+1, 1.0))
    plt.ylabel("")
    plt.title("Accuracy and EOD Across Clients - %s" % "Before and After Post-processing" )
    plt.legend()

    if save_to:
        plt.savefig(save_to)
    else:
        plt.show()


def plot_all(stat_dic, title=None, save_to=None):
    
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6) ) = plt.subplots(2, 3,figsize=(15, 10))

    x = list(range(len(stat_dic['test_acc_after'])))

    if title:
        fig.suptitle(title, fontsize=16)

    ax1.axhline(0.8, color='lightsteelblue', alpha=0.6)
    ax1.axhline(0.1, color='orange', alpha=0.4)
    ax1.plot(x, stat_dic['test_acc_before'], color='blue', marker='o', markersize=10, label='acc_before')
    ax1.plot(x, stat_dic['test_acc_after'],  color="red", marker='o',markersize=10,  label='acc_after')
    ax1.plot(x, stat_dic['test_eod_before'], color='blue', marker='o',markersize=10,  linestyle='dashed', label='eod_before')
    ax1.plot(x, stat_dic['test_eod_after'],  color="red",marker='o',markersize=10,  linestyle='dashed', label='eod_after')
    

    ax1.set_xlabel("Client ID")
    ax1.set_xticks(np.arange(min(x), max(x)+1, 1.0))
    ax1.set_title('Accuracy & EOD - Test')
    ax1.legend(loc="upper right")


    ax2.axhline(0.1, color='orange', alpha=0.4)
    ax2.plot(x, stat_dic['test_fpr_before'], color='blue',marker='o', linestyle='dashed', label='fpr_before')
    ax2.plot(x, stat_dic['test_fpr_after'],  color="red",marker='o', linestyle='dashed', label='fpr_after')
    ax2.set_xlabel("Client ID")
    ax2.set_xticks(np.arange(min(x), max(x)+1, 1.0))
    ax2.set_title('FPR - Test')
    ax2.legend()

    ax3.axhline(0.1, color='orange', alpha=0.4)
    ax3.plot(x, stat_dic['test_tpr_before'], color='blue',marker='o', linestyle='dashed', label='tpr_before')
    ax3.plot(x, stat_dic['test_tpr_after'],  color="red",marker='o', linestyle='dashed', label='tpr_after')
    ax3.set_xlabel("Client ID")
    ax3.set_xticks(np.arange(min(x), max(x)+1, 1.0))
    ax3.set_title('TPR - Test')
    ax3.legend()

    ax4.axhline(0.8, color='lightsteelblue', alpha=0.6)
    ax4.axhline(0.1, color='orange', alpha=0.4)
    ax4.plot(x, stat_dic['train_acc_before'], color='blue',marker='o',  label='acc_before')
    ax4.plot(x, stat_dic['train_acc_after'],  color="red",marker='o', label='acc_after')
    ax4.plot(x, stat_dic['train_eod_before'], color='blue',marker='o', linestyle='dashed', label='eod_before')
    ax4.plot(x, stat_dic['train_eod_after'],  color="red",marker='o', linestyle='dashed', label='eod_after')
    
    ax4.set_xlabel("Client ID")
    ax4.set_xticks(np.arange(min(x), max(x)+1, 1.0))
    ax4.set_title('Accuracy & EOD - Train')
    ax4.legend(loc="upper right")
    ax4.set_xticks(np.arange(min(x), max(x)+1, 1.0))

    ax5.axhline(0.1, color='orange', alpha=0.4)
    ax5.plot(x, stat_dic['train_fpr_before'], color='blue',marker='o', linestyle='dashed', label='fpr_before')
    ax5.plot(x, stat_dic['train_fpr_after'],  color="red",marker='o', linestyle='dashed', label='fpr_after')
    ax5.set_xlabel("Client ID")
    ax5.set_xticks(np.arange(min(x), max(x)+1, 1.0))
    ax5.set_title('FPR - Train')
    ax5.legend()

    ax6.axhline(0.1, color='orange', alpha=0.4)
    ax6.plot(x, stat_dic['train_tpr_before'], color='blue',marker='o', linestyle='dashed', label='tpr_before')
    ax6.plot(x, stat_dic['train_tpr_after'],  color="red",marker='o', linestyle='dashed', label='tpr_after')
    ax6.set_xlabel("Client ID")
    ax6.set_xticks(np.arange(min(x), max(x)+1, 1.0))
    ax6.set_title('TPR - Train')
    ax6.legend()


    if save_to:
        plt.savefig(save_to)
    else:
        plt.show()


def plot_multi_exp(stat_dic, args, new=True, plot_tpfp=True, title=None, save_to=None):

    # false_negative_rate_difference

    # stat_keys = []
    # set_split = ["train", "test"]
    # local_metrics = ["acc", "eod"]

    # if args.fl_new:
    #     stat_keys += [ss+"_"+lm+"_"+"new" for ss in set_split for lm in local_metrics]

    if plot_tpfp:
        # fig, ((ax1, ax2, ax3, ax30, ax31), (ax4, ax5, ax6, ax7, ax8) ) = plt.subplots(2, 5,figsize=(20, 8))
        fig, ((ax1, ax2, ax3, ax30), (ax4, ax5, ax6, ax7) ) = plt.subplots(2, 4,figsize=(16, 8))
    else:
        fig, ((ax1, ax2), (ax4, ax5) ) = plt.subplots(2, 2,figsize=(12, 10))

    x = [x + 1 for x in list(range(args.num_users))] 
    y_lim = [-0.05, 1.0]

    if title:
        fig.suptitle(title, fontsize=16)

    
    if plot_tpfp: 
        # Plot Test TPR FPR
        ax3.axhline(0.1, color='orange', alpha=0.4)
        ax3.axhline(0, color='black', alpha=0.4)
        ax3.axhline(-0.1, color='orange', alpha=0.4)
        ax3.plot(x, (stat_dic['test_tpr_fedavg']), color='blue', marker='o',  label='tpr_fedavg')
        ax3.plot(x, (stat_dic['test_tpr_new']),  color="red", marker='o', label='tpr_new')
        if "ft" in args.debias:
            ax3.plot(x, stat_dic['test_tpr_new_ft'], color='hotpink', marker='o',  label='tpr_new_ft')
            
        
        ax3.set_title('TPR - Test')
        ax3.set_xticks(np.arange(min(x), max(x)+1, 1.0))
        ax3.legend()

        ax30.axhline(0.1, color='orange', alpha=0.4)
        ax30.axhline(0, color='black', alpha=0.4)
        ax30.axhline(-0.1, color='orange', alpha=0.4)
        ax30.plot(x, (stat_dic['test_fpr_fedavg']), color='blue', marker='o', linestyle='dashed', label='fpr_fedavg')
        ax30.plot(x, (stat_dic['test_fpr_new']),  color="red", marker='o', linestyle='dashed', label='fpr_new')
        if "ft" in args.debias:
            ax30.plot(x, stat_dic['test_fpr_new_ft'], color='hotpink', marker='o',  linestyle='dashed',  label='fpr_new_ft')
        
        ax30.set_title('FPR - Test')
        ax30.set_xticks(np.arange(min(x), max(x)+1, 1.0))
        ax30.legend()


        # ax31.axhline(0.1, color='orange', alpha=0.4)
        # ax31.plot(x, stat_dic['test_eod_fedavg'], color='blue', marker='o', linestyle='dashed', label='eod_fedavg')
        # ax31.plot(x, stat_dic['test_eod_new'],  color="red", marker='o', linestyle='dashed', label='eod_new')
        # ax31.plot(x, stat_dic['test_eod_fairfed'],  color="green",  marker='o',linestyle='dashed', label='eod_fairfed')
        # ax31.set_xticks(np.arange(min(x), max(x)+1, 1.0))
        # ax31.legend()


        ax6.axhline(0.1, color='orange', alpha=0.4)
        ax6.axhline(0, color='black', alpha=0.4)
        ax6.axhline(-0.1, color='orange', alpha=0.4)
        ax6.plot(x, (stat_dic['train_tpr_fedavg']), color='blue', marker='o',  label='tpr_fedavg')
        ax6.plot(x, (stat_dic['train_tpr_new']),  color="red", marker='o', label='tpr_new')
        if "ft" in args.debias:
            ax6.plot(x, stat_dic['train_tpr_new_ft'], color='hotpink', marker='o',  label='tpr_new_ft')
            
        
        ax6.set_title('TPR - Train')
        ax6.set_xticks(np.arange(min(x), max(x)+1, 1.0))
        ax6.legend()

        ax7.axhline(0.1, color='orange', alpha=0.4)
        ax7.axhline(0, color='black', alpha=0.4)
        ax7.axhline(-0.1, color='orange', alpha=0.4)
        ax7.plot(x, (stat_dic['train_fpr_fedavg']), color='blue', marker='o', linestyle='dashed', label='fpr_fedavg')
        ax7.plot(x, (stat_dic['train_fpr_new']),  color="red", marker='o', linestyle='dashed', label='fpr_new')
        if "ft" in args.debias:
            ax7.plot(x, stat_dic['train_fpr_new_ft'], color='hotpink', marker='o', linestyle='dashed', label='fpr_new_ft')
        ax7.set_title('FPR - Train')
        ax7.set_xticks(np.arange(min(x), max(x)+1, 1.0))
        ax7.legend()

        # ax8.axhline(0.8, color='lightsteelblue', alpha=0.6)
        # ax8.plot(x, stat_dic['test_acc_fedavg'], color='blue', marker='o', label='eod_fedavg')
        # ax8.plot(x, stat_dic['test_acc_new'],  color="red", marker='o', label='eod_new')
        # ax8.plot(x, stat_dic['test_acc_fairfed'],  color="green",  marker='o', label='eod_fairfed')
        # ax8.set_xticks(np.arange(min(x), max(x)+1, 1.0))
        # ax8.legend()

    # ax1.axhline(0.8, color='lightsteelblue', alpha=0.6)
    ax1.axhline(0.1, color='orange', alpha=0.4)
    # ax1.plot(x, stat_dic['test_acc_fedavg'], color='blue', marker='o',  label='acc_fedavg')
    # ax1.plot(x, stat_dic['test_acc_new'],  color="red", marker='o', label='acc_new')
    ax1.plot(x, stat_dic['test_eod_fedavg'], color='blue', marker='o', linestyle='dashed', label='eod_fedavg')
    ax1.plot(x, stat_dic['test_eod_new'],  color="red", marker='o', linestyle='dashed', label='eod_new')
    ax1.plot(x, stat_dic['test_eod_fairfed'],  color="green",  marker='o',linestyle='dashed', label='eod_fairfed')
    # ax1.set_ylim(y_lim)
    

    ax1.set_xlabel("Client ID")
    ax1.set_xticks(np.arange(min(x), max(x)+1, 1.0))
    ax1.set_title('EOD - Test')
    ax1.legend(loc="upper right")


    # ax2.axhline(0.1, color='orange', alpha=0.4)
    ax2.axhline(0.8, color='lightsteelblue', alpha=0.6)
    ax2.plot(x, stat_dic['test_acc_fairfed'], color='green', marker='o', label='acc_fairfed')
    ax2.plot(x, stat_dic['test_acc_new'],  color="red", marker='o', label='acc_new')
    ax2.plot(x, stat_dic['test_acc_fedavg'], color='blue', marker='o',  label='acc_fedavg')
    # ax2.plot(x, stat_dic['test_eod_new_ft'], color='hotpink', marker='o', linestyle='dashed', label='eod_new_ft')

    # ax2.plot(x, stat_dic['test_eod_new'],  color="red", marker='o', linestyle='dashed', label='eod_new')
    # ax2.plot(x, stat_dic['test_eod_fedavg'], color='blue', marker='o', linestyle='dashed', label='eod_fedavg')
    # ax2.plot(x, stat_dic['test_eod_fairfed'],  color="green",  marker='o',linestyle='dashed', label='eod_fairfed')

    if not (args.dataset== "adult" or args.dataset== "compas"):
        ax2.plot(x, stat_dic['test_acc_new_ft'], color='hotpink', marker='o',  label='acc_new_ft')
        ax1.plot(x, stat_dic['test_eod_new_ft'], color='hotpink', marker='o', linestyle='dashed', label='eod_new_ft')

    ax2.set_xlabel("Client ID")
    ax2.set_xticks(np.arange(min(x), max(x)+1, 1.0))
    ax2.set_title('Acc - Test')
    ax2.legend(fontsize=6, loc="upper right")
    # ax2.set_ylim(y_lim)


    # ax4.axhline(0.8, color='lightsteelblue', alpha=0.6)
    ax4.axhline(0.1, color='orange', alpha=0.4)

    # ax4.plot(x, stat_dic['train_acc_fedavg'], color='blue', marker='o',  label='acc_fedavg')
    # ax4.plot(x, stat_dic['train_acc_new'],  color="red", marker='o', label='acc_new')
    ax4.plot(x, stat_dic['train_eod_fedavg'], color='blue', marker='o', linestyle='dashed', label='eod_fedavg')
    ax4.plot(x, stat_dic['train_eod_new'],  color="red", marker='o', linestyle='dashed', label='eod_new')
    ax4.plot(x, stat_dic['train_eod_fairfed'],  color="green", marker='o', linestyle='dashed', label='eod_fairfed')
    
    ax4.set_xlabel("Client ID")
    ax4.set_title('EOD - Train')
    ax4.legend(loc="upper right")
    ax4.set_xticks(np.arange(min(x), max(x)+1, 1.0))
    # ax4.set_ylim(y_lim)


    # ax5.axhline(0.1, color='orange', alpha=0.4)
    ax5.axhline(0.8, color='lightsteelblue', alpha=0.6)
    ax5.plot(x, stat_dic['train_acc_fairfed'], color='green', marker='o', label='acc_fairfed')
    ax5.plot(x, stat_dic['train_acc_new'],  color="red", marker='o', label='acc_new')
    ax5.plot(x, stat_dic['train_acc_fedavg'], color='blue', marker='o',  label='acc_fedavg')
    
    # ax5.plot(x, stat_dic['train_eod_new'],  color="red", marker='o', linestyle='dashed', label='eod_new')
    # ax5.plot(x, stat_dic['train_eod_fairfed'],  color="green", marker='o', linestyle='dashed', label='eod_fairfed')
    # ax5.plot(x, stat_dic['train_eod_fedavg'], color='blue', marker='o', linestyle='dashed', label='eod_fedavg')

    if not (args.dataset== "adult" or args.dataset== "compas"):
        ax5.plot(x, stat_dic['train_acc_new_ft'], color='hotpink', marker='o',  label='acc_new_ft')
        ax4.plot(x, stat_dic['train_eod_new_ft'], color='hotpink', marker='o', linestyle='dashed', label='eod_new_ft')



    ax5.set_xlabel("Client ID")
    ax5.set_xticks(np.arange(min(x), max(x)+1, 1.0))
    ax5.set_title('Acc- Train')
    ax5.legend(fontsize=6, loc="upper right")
    # ax5.set_ylim(y_lim)


    if save_to:
        plt.savefig(save_to)
    else:
        plt.show()


def plot_loss(local_loss_all, train_loss, new=True, plot_tpfp=True, title=None, save_to=None):
    
    local_loss_all_T = np.array(local_loss_all).transpose()
    # print("Check shape: ", np.array(local_loss_all).shape, local_loss_all_T.shape)

    
    fig, ((ax1, ax2)) = plt.subplots(1, 2,figsize=(12, 6))

    if title:
        fig.suptitle(title, fontsize=16)


    

    if train_loss is not None:
        x1 = list(range(1, len(train_loss) + 1))
        ax1.plot(x1, train_loss, color='blue', marker='o',  label='train_loss')
        ax1.legend(loc="upper right")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title('train_loss')
        if max(x1)>20:
            step = 2
        else:
            step = 1
        ax1.set_xticks(np.arange(min(x1), max(x1)+1, step))
        ax1.grid()

    if local_loss_all is not None:
        x2 = list(range(1, len(local_loss_all_T[0])+1))
        if max(x2)>20:
            step = 2
        else:
            step = 1
        for i in range(len(local_loss_all_T)):
            ax2.plot(x2, local_loss_all_T[i], marker='o',  label= ('random client '+str(i+1)))
        
        ax2.legend(loc="upper right")
        ax2.set_xticks(np.arange(min(x2), max(x2)+1, step))

    print("Plot finish")


    if save_to:
        plt.savefig(save_to)
    else:
        plt.show()


def plot_loss_ft(epoch_loss, fairfed=False, title=None, save_to=None):
    
    
    fig, axs = plt.subplots(2, 2,figsize=(8, 6))
    if fairfed:
        y_lim = [min(np.array(epoch_loss).flatten())-0.05, max(np.array(epoch_loss).flatten())+0.05]
    # y_lim for final_layer fine-tuning
    else:
        y_lim = [min(np.array(epoch_loss).flatten())-0.2, max(np.array(epoch_loss).flatten())+0.2]

    if title:
        fig.suptitle(title, fontsize=8)

    x1 = list(range(1, len(epoch_loss[0])+1))
    if max(x1)>39:
        step = 4
    elif max(x1)>20:
        step = 2
    else:
        step = 1

    for i, ax in enumerate(axs.ravel()):
        try:
            ax.plot(x1, epoch_loss[i], marker='o',  label= ('random client '+str(i+1)))
        except:
            print(i, ax, len(epoch_loss))
    
        ax.legend(loc="upper right")
        ax.set_xticks(np.arange(min(x1), max(x1)+1, step))
        ax.set_ylim(y_lim)
        ax.grid(axis = 'y')

    print("Plot finish")


    if save_to:
        plt.savefig(save_to)
    else:
        plt.show()

if __name__ == '__main__':

    # eod_before = [0.2, 0.10457516339869288, 0.4444444444444444, 0.02929292929292926, 0.047619047619047616, 0.03782238951624146, 0.032727272727272716, 0.14590932777488275, 0.14159811985898943, 0.12]

    # eod_after = [0.2, 0.3267973856209151, 0.4444444444444444, 0.011695906432748537, 0.09238095238095237, 0.3076923076923077, 0.032727272727272716, 0.19854090672225116, 0.1145710928319624, 0.13818181818181818]


    # acc_before = [0.7, 0.8686131386861314, 0.9369538077403246, 0.7906614785992218, 0.6049382716049383, 0.8911806543385491, 0.54, 0.757396449704142, 0.20577617328519857, 0.7064439140811456]

    # acc_after = [0.7, 0.8613138686131386, 0.9369538077403246, 0.7891050583657587, 0.5987654320987654, 0.887624466571835, 0.54, 0.7544378698224852, 0.20938628158844766, 0.7052505966587113]

    # plot_acc_eod(acc_before, acc_after, eod_before, eod_after)

    print("Nothing to plot!")

    # plot_mean_sd(sample_ls)
    # plot_before_after(sample_ls, sample_ls_2)

