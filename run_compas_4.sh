
# RUn experiments on Comapas dataset with Part #4 (Non-iid dirichlet split)
#

# Compas Partition:
# #1:  10 clients   (X)
# #2:  4 clients  -  IID (Random Split)
# #3:  4 clients  -  Non - IID (Dirichlet Split) - More-imbalanced: Client #3 More samples
# #4:  4 clients  -  Non - IID (Dirichlet Split) - Less-imbalanced

# Part #4

exp_idx=32

# for rep_idx in 0
# do
#   python src/federated_main.py --model=plain --dataset=compas --frac 0.25 \
#   --epochs=2 --local_ep 1 --fairfed_ep 2 --ft_ep 2   \
#   --fl_new True --fl_fairfed True  --plot_tpfp True  --debias ppft \
#   --idx ${exp_idx} --num_users 4  --partition_idx 4 --beta 1.0  --local_bs 32  \
#   --ft_alpha 10 --platform "local" --rep $rep_idx \
#   # --gpu True
# done

# python src/collect_results.py --model=plain --dataset=compas --frac 1.0 \
# --epochs=2 --local_ep 1 --fairfed_ep 4 --ft_ep 2   \
# --fl_new True --fl_fairfed True  --plot_tpfp True  --debias ppft \
# --idx ${exp_idx} --num_users 4  --partition_idx 3 --beta 1.0  --local_bs 32  \
# --ft_alpha 10 --platform "local"  \

exp_idx=29
python src/collect_results.py --idx ${exp_idx} \
--example_folder newfairfed_ppft_ptb-xl_plain_frac0.1_client4_lr0.01_part1_beta1.0_ep4_1_8_ftep_8