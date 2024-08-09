
# RUn experiments on Adult dataset with Part #7 (Non-iid dirichlet split)
#

# Adult Partition:

# #12:  4 clients  -  IID (Random Split)
# #6:  4 clients  -  Non - IID (Dirichlet Split) - Client #2&3 have more samples
# #7:  4 clients  -  Non - IID (Dirichlet Split) - 1->4 Increasing number of sample

# Part #7

exp_idx=34

dataset=adult
partition_idx=7
frac=1.0

global_ep=8
fairfed_ep=10
ft_ep=1


platform=local


# # # Single run: Check Loss and weights
# rep_idx=0
# python src/federated_main.py --model=plain --dataset=$dataset --frac $frac \
# --epochs $global_ep --local_ep 1 --fairfed_ep $fairfed_ep --ft_ep $ft_ep  \
# --fl_new True --fl_fairfed True  --plot_tpfp True  --debias ppft \
# --idx ${exp_idx} --num_users 4  --partition_idx $partition_idx --beta 1.0  --local_bs 32  \
# --ft_alpha 10 --platform $platform --rep $rep_idx \


local_split=/Users/zhouyi/Desktop/Fair_FL_new/data/adult/partition/7/client_datasets_0.pkl
# local_split=/Users/zhouyi/Desktop/Fair_FL_new/data/adult/partition/6/client_datasets_0.pkl

for fairfed_ep in 8
do

    for global_ep in 8
    do

        for rep_idx in 0 1 2 3 4
        do
            python src/federated_main.py --model=plain --dataset=$dataset --frac $frac \
            --epochs $global_ep --local_ep 1 --fairfed_ep $fairfed_ep --ft_ep $ft_ep  \
            --fl_new True --fl_fairfed True  --plot_tpfp True  --debias ppft \
            --idx ${exp_idx} --num_users 4  --partition_idx $partition_idx --beta 1.0  --local_bs 32  \
            --ft_alpha 10 --platform $platform --rep $rep_idx --local_split $local_split
            # --gpu True
        done

        python src/collect_results.py --model=plain --dataset=$dataset --frac $frac \
        --epochs $global_ep --local_ep 1 --fairfed_ep $fairfed_ep --ft_ep $ft_ep   \
        --fl_new True --fl_fairfed True  --plot_tpfp True  --debias ppft \
        --idx ${exp_idx} --num_users 4  --partition_idx $partition_idx --beta 1.0  --local_bs 32  \
        --ft_alpha 10 --platform  $platform 
    done

done


# exp_idx=29
# python src/collect_results.py --idx ${exp_idx} \
# --example_folder newfairfed_ppft_ptb-xl_plain_frac0.1_client4_lr0.01_part1_beta1.0_ep4_1_8_ftep_8


# Feature Name
# workclass_1,workclass_2,workclass_3, workclass_4,workclass_5,workclass_6, marital-status_1,marital-status_2,
# marital-status_3,marital-status_4,marital-status_5,marital-status_6,new_occupation_1,new_occupation_2,new_occupation_3,new_occupation_4,
# relationship_1,relationship_2,relationship_3,relationship_4,relationship_5,race_1,race_2,race_3,
# race_4,sex_1,age,  fnlwgt,education-num,capital-gain, capital-loss,hours-per-week,  income