# Runs experiment on modal

import modal

gpu="a10g"
cloud="gcp" if gpu and gpu.startswith("a100") else "aws"
volume = modal.NetworkFileSystem.new().persisted("nanoGPT_softmax1_20230802_aws")
image = modal.Image.from_dockerfile("Dockerfile")
stub = modal.Stub("lob-run", image = image)

code_root = "/volume/nanoGPT"

@stub.function(
        cloud=cloud, 
        gpu=None,
        timeout=120,
        network_file_systems={"/volume": volume})
def ls():
    import os
    os.system(f"ls -lar")

@stub.function(
        cloud=cloud, 
        gpu=None,
        timeout=120,
        network_file_systems={"/volume": volume})
def init():
    import os
    import requests
    import tiktoken
    import numpy as np
    
    # define root
    os.system(f"mkdir -p {code_root}")
    os.chdir(code_root)

    # copy source to root
    os.system(f"cp -rf /nanoGPT .")

    # download the datasetcd 
    data_path = os.path.join('data', 'reddit')
    os.system(f"mkdir -p {data_path}")
    os.chdir(data_path)
    zip_filename = 'input.txt.tgz'
    input_filename = 'input.txt'

    if not os.path.exists(zip_filename):
        print("Input file doesn't exist. Downloading and extracting.")
        data_url_zipped = 'https://q1r1.c19.e2-5.dev/models/reddit_sydney_text_sample.tgz'
        os.system(f'wget -O {zip_filename} {data_url_zipped}')

    if not os.path.exists(input_filename):
        os.system(f'tar -xf {zip_filename}')

    with open(input_filename, 'r') as f:
        data = f.read()
    n = len(data)
    train_data = data[:int(n*0.9)]
    val_data = data[int(n*0.9):]

    # encode with tiktoken gpt2 bpe
    enc = tiktoken.get_encoding("gpt2")
    train_ids = enc.encode_ordinary(train_data)
    val_ids = enc.encode_ordinary(val_data)
    print(f"train has {len(train_ids):,} tokens")
    print(f"val has {len(val_ids):,} tokens")

    # export to bin files
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    train_ids.tofile('train.bin')
    val_ids.tofile('val.bin')


@stub.function(
        cloud=cloud, 
        gpu=gpu,
        timeout=7200,
        secret=modal.Secret.from_name("wandb-secret"),
        network_file_systems={"/volume": volume})
def train():
    import os
    os.chdir(code_root)
    params = \
"""
out_dir = 'out-reddit-softmax-original'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256 # context of up to 256 previous characters
n_layer = 6
n_head = 6
n_embd = 768
dropout = 0.2
learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 100000
lr_decay_iters = 100000
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small
warmup_iters = 100 # not super necessary potentially
eval_interval = 250
eval_iters = 200
log_interval = 10
always_save_checkpoint = True
wandb_log = True
wandb_project = 'nanoGPT_softmax1'
wandb_run_name = 'reddit-mini-gpt-original'
dataset = 'reddit'
"""
    with open('config/train_reddit.py', 'w') as f:
        f.write(params)

    os.system('python train.py config/train_reddit.py')