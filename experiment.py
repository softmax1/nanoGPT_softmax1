# Runs experiment on modal

import modal

gpu="a10g"

# This prevents cross-cloud file transfer, but you need to init again
# when the cloud changes.
cloud="gcp" if gpu and gpu.startswith("a100") else "aws"
volume_name = "nanoGPT_softmax1_20230802"
volume = modal.NetworkFileSystem.new().persisted(f"{volume_name}_{cloud}")
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
    os.system(f"cd /volume && ls -laR")

@stub.function(
        cloud=cloud, 
        gpu=None,
        network_file_systems={"/volume": volume},
        secret=modal.Secret.from_name("s3-secret"))
def upload_weights():
    import boto3
    import os    
    import sys
    bytes = 0
    def progress(chunk):
        nonlocal bytes
        bytes += chunk
        sys.stdout.write(f"\rUploaded: {bytes}")
        sys.stdout.flush()
    b3_session = boto3.Session()
    b3_client = b3_session.client('s3', endpoint_url=os.environ['AWS_ENDPOINT'])
    file = os.path.join(code_root, 'out-reddit-softmax-original', 'ckpt.pt')
    b3_client.upload_file(file, 'models', 'nanoGPT_softmax1/weights/softmax0/ckpt.pt', Callback=progress)
    file = os.path.join(code_root, 'out-reddit-softmax-one', 'ckpt.pt')
    b3_client.upload_file(file, 'models', 'nanoGPT_softmax1/weights/softmax1/ckpt.pt', Callback=progress)

# Warning there seems to be an issue where cp -rf silenty fails for files that
# already exist. It may be something to do with the chosen container. So if you change
# the source in the Docker container, you will need to delete the files on the volume
# first. I don't do this here as I would rather not accidently delete your model checkpoint.
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

    # download the dataset
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

params_common = \
"""
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
dataset = 'reddit'
"""

params_original = \
f"""
{params_common}
out_dir = 'out-reddit-softmax-original'
wandb_log = True
wandb_project = 'nanoGPT_softmax1'
wandb_run_name = 'reddit-mini-gpt-original'
"""

params_original_sample = \
f"""
{params_original}
start = "The best fried chicken restaurant I would recommend is"
max_new_tokens = 100
"""

@stub.function(
        cloud=cloud, 
        gpu=gpu,
        timeout=24 * 3600,
        secret=modal.Secret.from_name("wandb-secret"),
        network_file_systems={"/volume": volume})
def train():
    import os
    os.chdir(code_root)
    with open('config/train_reddit.py', 'w') as f:
        f.write(params_original)
    os.system('python train.py config/train_reddit.py')

@stub.function(
        cloud=cloud, 
        gpu=gpu,
        timeout=300,
        network_file_systems={"/volume": volume})
def sample():
    import os
    os.chdir(code_root)
    with open('config/sample_reddit.py', 'w') as f:
        f.write(params_original_sample)
    os.system('python sample.py config/sample_reddit.py')

params_softmax1 = \
f"""
{params_common}
out_dir = 'out-reddit-softmax-one'
wandb_log = True
wandb_project = 'nanoGPT_softmax1'
wandb_run_name = 'reddit-mini-gpt-softmax-one'
use_softmax1 = True
"""

@stub.function(
        cloud=cloud, 
        gpu=gpu,
        timeout=24 * 3600,
        secret=modal.Secret.from_name("wandb-secret"),
        network_file_systems={"/volume": volume})
def train_softmax1():
    import os
    os.chdir(code_root)    
    with open('config/train_reddit_softmax1.py', 'w') as f:
        f.write(params_softmax1)

    os.system('python train.py config/train_reddit_softmax1.py')

params_softmax1_sample = \
f"""
{params_softmax1}
start = "The best fried chicken restaurant I would recommend is"
max_new_tokens = 100
"""

@stub.function(
        cloud=cloud, 
        gpu=gpu,
        timeout=300,
        network_file_systems={"/volume": volume})
def sample_softmax1():
    import os
    os.chdir(code_root)
    with open('config/sample_reddit_softmax1.py', 'w') as f:
        f.write(params_softmax1_sample)
    os.system('python sample.py config/sample_reddit_softmax1.py')

def kurtosis_softmax1_code(out_dir: str, name: str):
    return \
    """
import torch
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out' # ignored if init_from is not 'resume'
start = "\\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 10 # number of samples to draw
max_new_tokens = 500 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cpu' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

""" + \
    f"ckpt_path='/volume/nanoGPT/{out_dir}/ckpt.pt'" + \
    """
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
print (f"Kurtosis for model softmax1")
for name, param in model.named_parameters():
    mean = param.data.mean()
    diffs = param.data - mean
    var = torch.mean(torch.pow(diffs, 2.0))
    std = torch.pow(var, 0.5)
    zscores = diffs / std
    skews = torch.mean(torch.pow(zscores, 3.0))
    kurtosis = torch.mean(torch.pow(zscores, 4.0)) - 3.0
    print ("{:<35}: {:>11.5f}".format(name, kurtosis))
    """

@stub.function(
        cloud=cloud, 
        gpu=None,
        timeout=60,
        network_file_systems={"/volume": volume})
def kurtosis():    
    import os
    os.chdir(code_root)
    with open('kurtosis_reddit.py', 'w') as f:
        f.write(kurtosis_softmax1_code('out-reddit-softmax-original', 'softmax original'))
    os.system('python kurtosis_reddit.py')

@stub.function(
        cloud=None, 
        gpu=None,
        timeout=60,
        network_file_systems={"/volume": volume})
def kurtosis_softmax1():    
    import os
    os.chdir(code_root)
    with open('kurtosis_reddit_softmax1.py', 'w') as f:
        f.write(kurtosis_softmax1_code('out-reddit-softmax-one', 'softmax1'))
    os.system('python kurtosis_reddit_softmax1.py')
