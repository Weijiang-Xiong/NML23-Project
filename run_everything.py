import subprocess

command_list = [
    "python train_net.py --max-epoch 500 --model tf",
    "python train_net.py --max-epoch 500 --model tf --from-raw --method binary",
    "python train_net.py --max-epoch 500 --model tf --from-raw --method svd",
    "python train_net.py --max-epoch 500 --model tf --from-raw --method binary+n2v",
    "python train_net.py --max-epoch 500 --model tf --from-raw --method feather",
    "python train_net.py --max-epoch 500 --model tf --from-raw --method feather+svd",
    "python train_net.py --max-epoch 500 --model tf --from-raw --method node2vec+svd",
    "python train_net.py --max-epoch 500 --model gcn",
    "python train_net.py --max-epoch 500 --model gcn --from-raw --method binary",
    "python train_net.py --max-epoch 500 --model gcn --from-raw --method svd",
    "python train_net.py --max-epoch 500 --model gcn --from-raw --method binary+n2v",
    "python train_net.py --max-epoch 500 --model gcn --from-raw --method feather",
    "python train_net.py --max-epoch 500 --model gcn --from-raw --method feather+svd",
    "python train_net.py --max-epoch 500 --model gcn --from-raw --method node2vec+svd",
]


for cmd in command_list:
    completed_process = subprocess.run(cmd, shell=True)