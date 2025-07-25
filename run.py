import torch
import argparse
from dataclasses import dataclass
from nanogpt.optim.dist_adam import DistAdam
from nanogpt.optim.utils import get_optimizer
from nanogpt.model import GPT
import random
import yaml
from nanogpt.utils import hash_config
import os
import torch.distributed as dist
from torch import  nn
import copy
import json
from nanogpt.train_gpt import train

parser = argparse.ArgumentParser(description='Train GPT-2 with optional config file.')
parser.add_argument('--config', type=str, help='Path to config file', default=None)
parser.add_argument('--suffix', type=str, help='Path to config file', default='')
args = parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU
set_seed(42)
# -----------------------------------------------------------------------------
# torchrun sets these env variables
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
# assert world_size == 8 # this code is designed for 8xH100
assert torch.cuda.is_available()
device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
torch.cuda.set_device(device)
dist.init_process_group(backend="nccl", device_id=device)
dist.barrier()
master_process = (rank == 0) # this process will do logging, checkpointing etc.

# Parse config
config_file = args.config
with open(config_file, 'r') as file:
    config = yaml.safe_load(file)
#config = load_config(get_default_config(), config_file)
outputname = config_file.replace("configs/","").replace('.yaml','')
output_dir = f"nanogpt/outputs/{outputname}"


if master_process:
    print(f"Loading configuration from {config_file}")
    os.makedirs(output_dir, exist_ok=True)   
# -----------------------------------------------------------------------------
# int main

@dataclass
class DefaultTrainingParams:
    # Data-related parameters
    train_files: str = "data/fineweb10B/fineweb_train_*.bin"
    val_files: str = "data/fineweb10B/fineweb_val_*.bin"
    val_tokens: int = 10485760
    train_seq_len: int = 6 * world_size * 1024  # For 8 GPUs, this gives 6 * 8 = 48
    val_seq_len: int = 4 * world_size * 1024  # For 8 GPUs, this gives 4 * 8 = 64

    # Optimization parameters
    num_iterations: int = 1750  # Number of iterations to run
    cooldown_frac: float = 0.45  # Fraction of training spent cooling down the learning rate

    # Evaluation and logging parameters
    val_loss_every: int = 125  # Every how many steps to evaluate val loss? 0 for only at the end
    save_checkpoint: bool = False

defaulttrainingparams = DefaultTrainingParams()

# Ensure all fields in defaulttrainingparams are present in config['training_params']
for field in vars(defaulttrainingparams):
    if field not in config['training_params']:
        config['training_params'][field] = getattr(defaulttrainingparams, field)



model: nn.Module = GPT(vocab_size=50257, num_layers=12, num_heads=6, model_dim=768, max_seq_len=max(config['training_params']["train_seq_len"], config['training_params']["val_seq_len"])).cuda()
for m in model.modules():
    if isinstance(m, nn.Embedding):
        m.bfloat16()
for param in model.parameters():
    dist.broadcast(param.detach(), 0)

training_params = config['training_params'] 
list_optimizer_params = config["optimizer_params"]
# Loop over optimizers
for opt_config in list_optimizer_params:
    for lr in opt_config['lr']:
        if master_process:
            print(f"Training with optimizer {opt_config['name']} and learning rate {lr}")
        # Generate hash for the current optimizer configuration
        opt_config_copy = copy.deepcopy(opt_config)
        opt_config_copy['lr'] = lr
        config_hash = hash_config(opt_config_copy, training_params)
        file_name = f"{opt_config['name']}-lr-{lr}-{config_hash}-world{world_size}"
        if args.suffix != '': file_name += f"-{args.suffix}"
        output_path = os.path.join(output_dir, file_name + '.json')

        # copy model to ensure consistency
        model_copy = copy.deepcopy(model).to(device)
        model_copy: nn.Module = torch.compile(model_copy, dynamic=False)
        # collect the parameters to optimize
        hidden_matrix_params = [p for n, p in model_copy.blocks.named_parameters() if p.ndim >= 2 and "embed" not in n]
        embed_params = [p for n, p in model_copy.named_parameters() if "embed" in n]
        scalar_params = [p for p in model_copy.parameters() if p.ndim < 2]
        head_params = [model_copy.lm_head.weight]
        # Get muon and version adam
        muon_optimizer_obj, muon_hyper_param  = get_optimizer(opt_config)   # NOT USING muon_hyper_param
        muon_optimizer = muon_optimizer_obj(hidden_matrix_params, lr=lr, momentum=opt_config['momentum'], weight_decay=opt_config['weight_decay'])
        adam_optimizer = DistAdam(scalar_params + head_params + embed_params, lr=opt_config['adam_lr'], betas=(0.8, 0.95), eps=1e-10, weight_decay=opt_config['weight_decay'])
        optimizers = [adam_optimizer, muon_optimizer]
        for opt in optimizers:
            for group in opt.param_groups:
                group["initial_lr"] = group["lr"]

        # Train and log of results
        logger = train(model_copy, optimizers, training_params, opt_config)

        # Save
        if master_process:
            logger.name = opt_config['name'] + '-lr-' + str(lr)
            if os.path.exists(output_path):
                print(f"File {output_path} already exists. Overwriting")
            with open(output_path, 'w') as file:
                json.dump(logger.__dict__, file)
            print(f"Saved output to {output_path}")
        del model_copy