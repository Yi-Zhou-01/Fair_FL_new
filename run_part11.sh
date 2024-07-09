# Run partition 11 with different parameters and save to exp 13


python src/federated_main.py --model=mlp --dataset=adult --epochs=4 --num_users 1 --frac 0.1 --local_ep 1 --fl_new True  --fl_fairfed True --post_proc_cost "fnr" --idx 13 --partition_idx 11 --beta 1.0 --plot_tpfp True

python src/federated_main.py --model=mlp --dataset=adult --epochs=10 --num_users 1 --frac 0.1 --local_ep 1 --fl_new True  --fl_fairfed True --post_proc_cost "fnr" --idx 13 --partition_idx 11 --beta 1.0 --plot_tpfp True

python src/federated_main.py --model=mlp --dataset=adult --epochs=20 --num_users 1 --frac 0.1 --local_ep 1 --fl_new True  --fl_fairfed True --post_proc_cost "fnr" --idx 13 --partition_idx 11 --beta 1.0 --plot_tpfp True

python src/federated_main.py --model=mlp --dataset=adult --epochs=4 --num_users 1 --frac 0.1 --local_ep 4 --fl_new True  --fl_fairfed True --post_proc_cost "fnr" --idx 13 --partition_idx 11 --beta 1.0 --plot_tpfp True

python src/federated_main.py --model=mlp --dataset=adult --epochs=10 --num_users 1 --frac 0.1 --local_ep 4 --fl_new True  --fl_fairfed True --post_proc_cost "fnr" --idx 13 --partition_idx 11 --beta 1.0 --plot_tpfp True
