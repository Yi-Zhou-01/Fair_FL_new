# Run part6 with mlp and save to exp 17


# python src/federated_main.py --model=mlp --dataset=adult --epochs=4 --num_users 4 --frac 0.75 --local_ep 1 --fl_new True  --fl_fairfed True  --idx 17 --partition_idx 6 --beta 1.0 --plot_tpfp True

# python src/federated_main.py --model=mlp --dataset=adult --epochs=4 --num_users 4 --frac 0.75 --local_ep 2 --fl_new True  --fl_fairfed True  --idx 17 --partition_idx 6 --beta 1.0 --plot_tpfp True

# python src/federated_main.py --model=mlp --dataset=adult --epochs=10 --num_users 4 --frac 0.75 --local_ep 1 --fl_new True  --fl_fairfed True --idx 17 --partition_idx 6 --beta 1.0 --plot_tpfp True

# python src/federated_main.py --model=mlp --dataset=adult --epochs=20 --num_users 4 --frac 0.75 --local_ep 1 --fl_new True  --fl_fairfed True --idx 17 --partition_idx 6 --beta 1.0 --plot_tpfp True



# python src/federated_main.py --model=mlp --dataset=adult --epochs=4 --num_users 4 --frac 0.75 --local_ep 1  --fairfed_ep 20 --fl_new True  --fl_fairfed True --idx 18 --partition_idx 6 --beta 1.0 --plot_tpfp True

# python src/federated_main.py --model=mlp --dataset=adult --epochs=4 --num_users 4 --frac 0.75 --local_ep 1  --fairfed_ep 4 --fl_new True  --fl_fairfed True --idx 18 --partition_idx 6 --beta 1.0 --plot_tpfp True

# python src/federated_main.py --model=mlp --dataset=adult --epochs=20 --num_users 4 --frac 0.75 --local_ep 1  --fairfed_ep 20 --fl_new True  --fl_fairfed True --idx 18 --partition_idx 6 --beta 1.0 --plot_tpfp True

# python src/federated_main.py --model=mlp --dataset=adult --epochs=4 --num_users 4 --frac 0.75 --local_ep 1  --fairfed_ep 10 --fl_new True  --fl_fairfed True --idx 18 --partition_idx 6 --beta 1.0 --plot_tpfp True

# python src/federated_main.py --model=mlp --dataset=adult --epochs=10 --num_users 4 --frac 0.75 --local_ep 1  --fairfed_ep 20 --fl_new True  --fl_fairfed True --idx 18 --partition_idx 6 --beta 1.0 --plot_tpfp True


# python src/federated_main.py --model=mlp --dataset=adult --frac 0.75 --epochs=4 --local_ep 1  --fairfed_ep 8 --ft_ep 5 --fl_new True  --fl_fairfed True --idx 19 --num_users 4  --partition_idx 6 --beta 1.0 --plot_tpfp True --debias ppft

# python src/federated_main.py --model=mlp --dataset=adult --frac 0.75 --epochs=4 --local_ep 1  --fairfed_ep 8 --ft_ep 10 --fl_new True  --fl_fairfed True --idx 19 --num_users 4  --partition_idx 6 --beta 1.0 --plot_tpfp True --debias ppft

# python src/federated_main.py --model=mlp --dataset=adult --frac 0.75 --epochs=4 --local_ep 1  --fairfed_ep 8 --ft_ep 20 --fl_new True  --fl_fairfed True --idx 19 --num_users 4  --partition_idx 6 --beta 1.0 --plot_tpfp True --debias ppft

# python src/federated_main.py --model=mlp --dataset=adult --frac 0.75 --epochs=4 --local_ep 1  --fairfed_ep 8 --ft_ep 6 --fl_new True  --fl_fairfed True --idx 19 --num_users 4  --partition_idx 6 --beta 1.0 --plot_tpfp True --debias ppft

# python src/federated_main.py --model=mlp --dataset=adult --frac 0.75 --epochs=4 --local_ep 1  --fairfed_ep 8 --ft_ep 8 --fl_new True  --fl_fairfed True --idx 19 --num_users 4  --partition_idx 6 --beta 1.0 --plot_tpfp True --debias ppft


python src/federated_main.py --model=mlp --dataset=adult --frac 0.75 --epochs=4 --local_ep 1  --fairfed_ep 4 --ft_ep 117 --fl_new True  --fl_fairfed True --idx 21 --num_users 4  --partition_idx 6 --beta 1.0 --plot_tpfp True --debias ppft --ft_alpha 10
# python src/federated_main.py --model=mlp --dataset=adult --frac 0.75 --epochs=4 --local_ep 1  --fairfed_ep 4 --ft_ep 8 --fl_new True  --fl_fairfed True --idx 21 --num_users 4  --partition_idx 6 --beta 1.0 --plot_tpfp True --debias ppft --ft_alpha 0.5
# python src/federated_main.py --model=mlp --dataset=adult --frac 0.75 --epochs=4 --local_ep 1  --fairfed_ep 4 --ft_ep 16 --fl_new True  --fl_fairfed True --idx 21 --num_users 4  --partition_idx 6 --beta 1.0 --plot_tpfp True --debias ppft --ft_alpha 0.5
