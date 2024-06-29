
# ------------------------
# Centralized baseline
python src/baseline_main.py --model=mlp --dataset=mnist --epochs=3

python src/baseline_main.py --model=mlp --dataset=adult --epochs=3 --iid=1



# -----------------------
# FL Setting

python src/federated_main.py --model=mlp --dataset=mnist --iid=1 --epochs=3 --num_users 100 --frac 0.1 --local_ep 4

python src/federated_main.py --model=mlp --dataset=adult --iid=1 --epochs=2 --num_users 100 --frac 0.1 --local_ep 3

python src/federated_main.py --model=mlp --dataset=adult --iid=0 --epochs=2 --num_users 80 --frac 0.1 --local_ep 3
# ^ 89 clients maximum

python src/federated_main.py --model=mlp --dataset=adult --iid=1 --epochs=2 --num_users 80 --frac 0.1 --local_ep 1


# Client partition
python src/partition.py --partition=diri --n_clients=10 --target_attr=income --partition_idx 1 --alpha 0.9


# FL Non-iid trianing using saved partition file 
python src/federated_main.py --model=mlp --dataset=adult --iid=0 --epochs=2 --num_users 10 --frac 0.1 --local_ep 1 



# FL: with saved partition file; with post-processing and evaluation

python src/federated_main.py --model=mlp --dataset=adult --epochs=2 --num_users 10 --frac 0.1 --local_ep 1 --fl "new" --post_proc_cost "fpr"


python src/federated_main.py --model=mlp --dataset=adult --epochs=4 --num_users 10 --frac 0.1 --local_ep 6 --fl "new" --post_proc_cost "fnr" --idx 1 --partition_idx 5


python src/plot.py


python src/partition.py --partition=diri --n_clients=10 --target_attr=income --partition_idx 6 --alpha 0.9
