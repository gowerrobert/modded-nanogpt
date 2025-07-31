import os
import sys
import time
import copy
from functools import lru_cache # Added partial for hook registration
from nanogpt.dataloader import distributed_data_generator
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
torch.empty(1, device="cuda", requires_grad=True).backward() # prevents a bug on some systems
from torch import Tensor, nn
import torch.nn.functional as F
import torch.distributed as dist

rank = int(os.environ["RANK"])
master_process = (rank == 0)
def print0(s, console=False):
    if master_process:
        print(s)

class Logging():
    def __init__(self):
        self.val_losses = []
        self.train_times = []
        self.step_times = []
        self.learning_rates = []
        self.max_memory_reserved =0
        self.max_memory_allocated =0

def train(model: nn.Module, optimizers: list[torch.optim.Optimizer],  train_config: dict, opt_config: dict) -> Logging:
    world_size = int(os.environ["WORLD_SIZE"])
    num_iterations = train_config["num_iterations"]

    def next_multiple_of_n(v: float | int, *, n: int):
        return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)
    # attention window size schedule: linearly increase
    @lru_cache(1)
    def get_window_size_blocks_helper(window_size: int):
        return torch.tensor(window_size // 128, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
    def get_window_size_blocks(step: int):
        x = step / num_iterations # progress in training
        assert 0 <= x <= 1
        # Linearly increase the block-wise sliding window size over training 128 -> 1792
        # increase by @fernbear.bsky.social; block-wise by @YouJiacheng
        window_size = next_multiple_of_n(1728 * x, n=128)
        return get_window_size_blocks_helper(window_size)

    ########################################
    #            Warmup kernels            #
    ########################################
    # Warmup the training kernels, then re-initialize the state so we aren't cheating
    logger = Logging()
    warmup_steps = 10
    initial_state = dict(model=copy.deepcopy(model.state_dict()),
                        optimizers=[copy.deepcopy(opt.state_dict()) for opt in optimizers]) # save the initial state
    train_loader = distributed_data_generator(train_config["train_files"], world_size * train_config["train_seq_len"], align_to_bos=True)
    for _ in range(warmup_steps):
        inputs, targets = next(train_loader)
        model(inputs, targets, get_window_size_blocks(1)).backward()
        for opt in optimizers:
            opt.step()
        model.zero_grad(set_to_none=True)
    model.load_state_dict(initial_state["model"])
    for opt, opt_state in zip(optimizers, initial_state["optimizers"]):
        opt.load_state_dict(opt_state)
    del train_loader, initial_state

    ########################################
    #  Learning rate schedule            #
    ########################################
    def get_lr(step: int):  
        x = step / num_iterations # progress in training
        assert 0 <= x <= 1
        if x < 1 - train_config['cooldown_frac']:
            return 1.0
        else:
            w = (1 - x) / train_config['cooldown_frac']
            return w * 1.0 + (1 - w) * 0.1

    ########################################
    #        Training and validation       #
    ########################################

    train_loader = distributed_data_generator(train_config["train_files"], world_size * train_config["train_seq_len"], align_to_bos=True)
    training_time_ms = 0
    # start the clock
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    # begin training
    for step in range(num_iterations+1):
        last_step = (step == num_iterations)

        # --------------- VALIDATION SECTION -----------------
        if last_step or (train_config["val_loss_every"] > 0 and step % train_config["val_loss_every"] == 0):
            # stop the clock
            torch.cuda.synchronize()
            training_time_ms += 1000 * (time.perf_counter() - t0)
            model.eval()
            val_batch_size = world_size * train_config["val_seq_len"]
            assert train_config["val_tokens"] % val_batch_size == 0
            val_steps = train_config["val_tokens"] // val_batch_size
            val_loader = distributed_data_generator(train_config["val_files"], val_batch_size, align_to_bos=False)
            val_loss = 0
            with torch.no_grad():
                for _ in range(val_steps):
                    inputs, targets = next(val_loader)
                    val_loss += model(inputs, targets, get_window_size_blocks(step))
            val_loss /= val_steps
            del val_loader
            dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
            print0(f"step:{step}/{num_iterations} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/max(step, 1):.2f}ms", console=True)
            logger.val_losses.append(val_loss.item())
            logger.train_times.append(training_time_ms)
            logger.step_times.append(training_time_ms / max(step, 1))
            model.train()
            # start the clock again
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        # --------------- TRAINING SECTION -----------------
        inputs, targets = next(train_loader)
        model(inputs, targets, get_window_size_blocks(step)).backward()
        # set optimization hyperparameters
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["initial_lr"] * get_lr(step)
            av_lr = sum(group["lr"] for group in opt.param_groups) / len(opt.param_groups)
            logger.learning_rates.append(av_lr)
        optimizer1, optimizer2 = optimizers
        for group in optimizers[1].param_groups:
            frac = min(step / 300, 1) # momentum warmup for muon
            group["momentum"] = (1 - frac) * 0.85 + frac * 0.95
        # step the optimizers
        for opt in optimizers:
            opt.step()
        # null the gradients
        model.zero_grad(set_to_none=True)
        # logging
        approx_training_time_ms = training_time_ms + 1000 * (time.perf_counter() - t0)
        print0(f"step:{step+1}/{num_iterations} train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms/(step + 1):.2f}ms", console=True)
        logger.train_times.append(training_time_ms)
        logger.step_times.append(approx_training_time_ms)
    print0(f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB", console=True)

    return logger