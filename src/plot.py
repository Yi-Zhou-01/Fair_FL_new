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
    plt.ylabel("")
    plt.title("Accuracy and EOD Across Clients - %s" % "Before and After Post-processing" )
    plt.legend()

    if save_to:
        plt.savefig(save_to)
    else:
        plt.show()




if __name__ == '__main__':

    eod_before = [0.2, 0.10457516339869288, 0.4444444444444444, 0.02929292929292926, 0.047619047619047616, 0.03782238951624146, 0.032727272727272716, 0.14590932777488275, 0.14159811985898943, 0.12]

    eod_after = [0.2, 0.3267973856209151, 0.4444444444444444, 0.011695906432748537, 0.09238095238095237, 0.3076923076923077, 0.032727272727272716, 0.19854090672225116, 0.1145710928319624, 0.13818181818181818]


    acc_before = [0.7, 0.8686131386861314, 0.9369538077403246, 0.7906614785992218, 0.6049382716049383, 0.8911806543385491, 0.54, 0.757396449704142, 0.20577617328519857, 0.7064439140811456]

    acc_after = [0.7, 0.8613138686131386, 0.9369538077403246, 0.7891050583657587, 0.5987654320987654, 0.887624466571835, 0.54, 0.7544378698224852, 0.20938628158844766, 0.7052505966587113]

    plot_acc_eod(acc_before, acc_after, eod_before, eod_after)

    # plot_mean_sd(sample_ls)
    # plot_before_after(sample_ls, sample_ls_2)