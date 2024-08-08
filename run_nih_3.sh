
# Run experiments on NIH- Chest dataset with Part #3 (Non-iid dirichlet split)
#

# Part #2

exp_idx=100

for exp_idx in 0 1
do
  python src/federated_main.py --model=mobile --dataset=nih-chest-h5 --frac 0.25 \
  --epochs=2 --local_ep 1 --fairfed_ep 2 --ft_ep 2   \
  --fl_new True --fl_fairfed True  --plot_tpfp True  --debias ppft \
  --idx ${exp_idx} --num_users 4  --partition_idx 2 --beta 1.0  --local_bs 32 \
  --optimizer adam --ft_alpha 10 --platform azure --gpu True --rep 0 
done

