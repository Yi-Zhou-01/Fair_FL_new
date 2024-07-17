python src/federated_main.py --model=mlp --dataset=wcld --frac 0.5 --epochs=4 --local_ep 1  --fairfed_ep 4 --ft_ep 80 --fl_new True  --fl_fairfed True --idx 24 --num_users 4  --partition_idx 1 --beta 1.0 --plot_tpfp True --debias ppft --ft_alpha 10

# python src/federated_main.py --model=mlp --dataset=wcld --frac 0.5 --epochs=4 --local_ep 1  --fairfed_ep 8 --ft_ep 40 --fl_new True  --fl_fairfed True --idx 24 --num_users 4  --partition_idx 1 --beta 1.0 --plot_tpfp True --debias ppft --ft_alpha 10

python src/federated_main.py --model=mlp --dataset=wcld --frac 0.5 --epochs=8 --local_ep 1  --fairfed_ep 8 --ft_ep 40 --fl_new True  --fl_fairfed True --idx 24 --num_users 4  --partition_idx 1 --beta 1.0 --plot_tpfp True --debias ppft --ft_alpha 10

python src/federated_main.py --model=mlp --dataset=wcld --frac 0.5 --epochs=8 --local_ep 1  --fairfed_ep 16 --ft_ep 40 --fl_new True  --fl_fairfed True --idx 24 --num_users 4  --partition_idx 1 --beta 1.0 --plot_tpfp True --debias ppft --ft_alpha 10

# python src/federated_main.py --model=mlp --dataset=wcld --frac 0.75 --epochs=16 --local_ep 1  --fairfed_ep 16 --ft_ep 80 --fl_new True  --fl_fairfed True --idx 24 --num_users 4  --partition_idx 1 --beta 1.0 --plot_tpfp True --debias ppft --ft_alpha 10

# python src/federated_main.py --model=mlp --dataset=wcld --frac 0.75 --epochs=16 --local_ep 1  --fairfed_ep 48 --ft_ep 80 --fl_new True  --fl_fairfed True --idx 24 --num_users 4  --partition_idx 1 --beta 1.0 --plot_tpfp True --debias ppft --ft_alpha 10
