# Fair_FL_new

### Options


| Arguments | Description |   Default Option   | Other Options |
| --- | -----    |----     |----       |
|  ```--fl```   | Federated learning algorithm | 'new' |     'fedavg', 'fairfed'    |
|  ```--dataset```   | Dataset |  'adult'  |    ~~(images)~~    |
|  ```--model:``` | Model structure used for training | 'mlp' | 'mlp', 'cnn', ~~('dnn'?)~~|
| ```--gpu:``` | | None | ? |
| ```--epochs:```| | - | - |
| ```--lr:```| |0.01 | - | 
| ```--verbose:```| ? | 1 (activated) | 0 (deactivated)|
| ```--seed:```| ? | 1 | |
| ----- | -----    |----     |----       |
| ```--partition``` | File path containing data partition amongst clients.| **'TBA'** |   |
| ```--num_users:```Number of users. Default is 100.
| ```--frac:```     Fraction of users to be used for federated updates. Default is 0.1.
| ```--local_ep:``` Number of local training epochs in each user. Default is 10.
| ```--local_bs:``` Batch size of local updates in each user. Default is 10.
| ```--unequal:```  Used in non-iid setting. Option to split the data amongst users equally or unequally. Default set to 0 for equal splits. Set to 1 for unequal splits.


### Run commands

Run experiments using more balanced sample distribution (partition #5) and save results to experiment #100:

```python src/federated_main.py --model=mlp --dataset=adult --epochs=4 --num_users 10 --frac 0.1 --local_ep 6 --fl "new" --idx 100 --partition_idx 5```


Generate a new sample partition using Dirichlet distribution with alpha=0.9 and save to partition file #10:

```python src/partition.py --partition=diri --n_clients=10 --target_attr=income --partition_idx 10 --alpha 0.9```

<!-- 
* ```--dataset:```  Default: 'mnist'. Options: 'mnist', 'fmnist', 'cifar'
* ```--model:```    Default: 'mlp'. Options: 'mlp', 'cnn'
* ```--gpu:```      Default: None (runs on CPU). Can also be set to the specific gpu id.
* ```--epochs:```   Number of rounds of training.
* ```--lr:```       Learning rate set to 0.01 by default.
* ```--verbose:```  Detailed log outputs. Activated by default, set to 0 to deactivate.
* ```--seed:```     Random Seed. Default set to 1. -->

<!-- #### Federated Parameters
* ```--iid:```      Distribution of data amongst users. Default set to IID. Set to 0 for non-IID.
* ```--num_users:```Number of users. Default is 100.
* ```--frac:```     Fraction of users to be used for federated updates. Default is 0.1.
* ```--local_ep:``` Number of local training epochs in each user. Default is 10.
* ```--local_bs:``` Batch size of local updates in each user. Default is 10.
* ```--unequal:```  Used in non-iid setting. Option to split the data amongst users equally or unequally. Default set to 0 for equal splits. Set to 1 for unequal splits. -->

1. FL algorithm:
2. Dataset:
    - Adult
    - (image)
2. 