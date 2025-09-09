import os
import math
import glob
from tqdm.auto import tqdm
from itertools import cycle, islice
from dataclasses import dataclass
import numpy as np
import time
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from transformers import AutoTokenizer, set_seed

from stream_dataloader.dataset import SlidingTokenDataset
from model import GPTConfig, GPT
from utils import get_training_info

"""
Features:
    1. Muon + AdamW ZeRO-1 Optimizer
    2. Distributed framework: Pure ZeRO-1, use DTensor for auto overlap
    3. FP8 forward + (FP16) backward + BF16 optimizer + FP32 update
"""

@dataclass
class TrainerConfig:
    seed = 1337
    dataset_path = "../data/fineweb-edu-sample-10BT/"
    log_dir = "./log/"
    tokenizer_name = "gpt2"
    total_batch_size = 524288 # 2**19, ~0.5M, in number of tokens, range 0.5~4M, usually 1~2M
    B = 8 # micro batch size per device
    T = 4096 # sequence length
    shift = 1   # shift = 1 means next-token-prediction, > 1 means multi-token-prediction
    max_lr = 6e-4
    min_lr = max_lr * 0.1
    weight_decay=0.1
    grad_clip_value = 1.0
    warmup_steps = 1000     # 1000 steps as llama
    max_steps = None # 19073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens
    max_epochs = 1
    debug = True
    do_val = False
    do_inference = True
    split_rate=0.99 if do_val else 1.0
    val_every_steps = 250
    save_every_steps = 5000
    shift_every_steps = None
    use_compile = False


class Trainer:
    def _init_setup(self, config: TrainerConfig):
        set_seed(config.seed)
        int(os.environ.get('RANK', -1)) != -1
        init_process_group(backend='nccl')
        self.dp_rank = int(os.environ['RANK'])
        self.dp_local_rank = int(os.environ['LOCAL_RANK'])
        self.dp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{self.dp_local_rank}'
        torch.cuda.set_device(device)
        self.master_process = self.dp_rank == 0 # this process will do logging, checkpointing etc.
        
    def _init_dataset(self, config: TrainerConfig):
        self.train_dataset = SlidingTokenDataset(
            dataset_path=config.dataset_path, split="train", split_rate=config.split_rate, 
            seq_len=config.T, stride=config.T//2, batch_size=config.B*self.dp_world_size, 
            seed=config.seed, rank=self.dp_rank, world_size=self.dp_world_size)
        train_sampler = DistributedSampler(self.train_dataset, num_replicas=self.dp_world_size, rank=self.dp_rank, shuffle=False)
        self.train_loader = DataLoader(self.train_dataset, batch_size=config.B, shuffle=False, sampler=train_sampler, num_workers=0, pin_memory=True)
        if config.do_val:
            self.val_dataset = SlidingTokenDataset(
                dataset_path=config.dataset_path, split="validation", split_rate=config.split_rate, 
                seq_len=config.T, stride=config.T//2, batch_size=config.B*self.dp_world_size, 
                seed=config.seed, rank=self.dp_rank, world_size=self.dp_world_size)
            val_sampler = DistributedSampler(self.val_dataset, num_replicas=self.dp_world_size, rank=self.dp_rank, shuffle=False)
            self.val_loader = DataLoader(self.val_dataset, batch_size=config.B, shuffle=False, sampler=val_sampler, num_workers=0, pin_memory=True)
        else:
            self.val_dataset = self.val_loader = None

    def _init_model(self, config: TrainerConfig):
        torch.set_float32_matmul_precision('high')
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
        self.model_config = GPTConfig()
        model = GPT(self.model_config)
        self.model = DDP(model, device_ids=[self.dp_local_rank])
        self.raw_model = self.model.module

    def _init_optimizer(self, config: TrainerConfig):
        self.optimizer = torch.optim.AdamW(self.raw_model.parameters())

    def __init__(self, config: TrainerConfig):
        self.config = config
        self._init_setup(config)
        assert config.total_batch_size % (config.B * config.T * self.dp_world_size) == 0, "make sure total_batch_size is divisible by B * T * dp_world_size"
        self._init_dataset(config)
        self.training_info = get_training_info(
            config.B * len(self.train_loader), config.T, config.total_batch_size, config.B, self.dp_world_size, config.max_steps, config.max_epochs)
        if self.master_process:
            print(f"The training process will train {self.training_info['epochs']} epochs, {self.training_info['max_steps']} steps.")
            print(f"=> calculated gradient accumulation steps: {self.training_info['grad_accum_steps']}")
            print(f"=> calculated tokens per step: {self.training_info['total_tokens_per_step']}")
        self._init_model(config)
        self._init_optimizer(config)
        # create the log directory we will write checkpoints to and log to
        self.log_dir = os.path.join(
            config.log_dir,
            f"modelsize_{sum(p.numel() for p in self.raw_model.parameters())}_"
            f"lr{config.max_lr}_"
            f"B{config.total_batch_size}_"
            f"T{config.T}_"
            f"DP{self.dp_world_size}"
        )
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, f"log.txt")
        with open(self.log_file, "w") as f: # open for writing to clear the file
            pass

    def _lr_scheduler(self, it, max_steps, warmup_steps, max_lr, min_lr):
        # 1) linear warmup for warmup_iters steps
        if it < warmup_steps:
            return max_lr * (it+1) / warmup_steps
        # 2) if it > lr_decay_iters, return min learning rate
        if it > max_steps:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
        return min_lr + coeff * (max_lr - min_lr)
        
    def _one_training_micro_step(self, config: TrainerConfig, micro_step: int, data_batch: dict):
        x, y = data_batch["input_ids"], data_batch["labels"]
        x, y = x.to(f'cuda:{self.dp_local_rank}'), y.to(f'cuda:{self.dp_local_rank}')
        self.model.require_backward_grad_sync = (micro_step == self.training_info["grad_accum_steps"] - 1)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, loss = self.model(x, y)
        loss = loss / self.training_info["grad_accum_steps"]
        loss_accum += loss.detach()
        loss.backward()
        return loss_accum

    def _one_training_step(self, config: TrainerConfig, step: int):
        self.model.train()
        self.optimizer.zero_grad()
        loss_accum = 0.0
        for micro_step in range(self.training_info["grad_accum_steps"]):
            try:
                _, batch = next(self.train_loader_iter)
            except StopIteration:
                self.train_loader_iter = enumerate(self.train_loader)
                _, batch = next(self.train_loader_iter)
            loss_accum = self._one_training_micro_step(config, micro_step, batch)
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.grad_clip_value)
        lr = self._lr_scheduler(step, self.training_info["max_steps"], config.warmup_steps, config.max_lr, config.min_lr)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.optimizer.step()
        self.one_step_results["lr"] = lr
        self.one_step_results["loss"] = loss_accum
        self.one_step_results["grad_norm"] = norm
    
    def _resume_from_checkpoint(self, steps_per_epoch):
        pattern = os.path.join(self.log_dir, "model_*.pt")
        ckpts = sorted(glob.glob(pattern))
        if not ckpts:
            self.start_step = 0
            return
        ckpt_path = ckpts[-1]
        map_location = {'cuda:%d' % 0: 'cuda:%d' % self.dp_local_rank}
        ckpt = torch.load(ckpt_path, map_location=map_location)
        # 1) model
        self.raw_model.load_state_dict(ckpt['model'])
        # 2) optimizer
        if 'optimizer_state' in ckpt and ckpt['optimizer_state']:
            self.optimizer.load_state_dict(ckpt['optimizer_state'])
        # 3) RNG
        rng = ckpt.get('rng_state', None)
        if rng:
            torch.set_rng_state(rng['torch'])
            torch.cuda.set_rng_state(rng['cuda'], self.dp_local_rank)
            np.random.set_state(rng['numpy'])
        # 4) dataset state
        data_state = ckpt.get('dataset_state', {})
        if data_state:
            self._set_dataset_state(self.train_dataset, data_state.get('train', None))
            if self.val_dataset is not None:
                self._set_dataset_state(self.val_dataset, data_state.get('val', None))
        sampler_state = ckpt.get('sampler_state', {})
        epoch = sampler_state.get('epoch', 0)
        iter_idx = sampler_state.get('iter_idx', 0)
        if hasattr(self, 'train_sampler') and self.train_loader.sampler is not None:
            self.train_loader.sampler.set_epoch(epoch)
        self.train_loader_iter = enumerate(self.train_loader)
        if iter_idx > 0:
            self.train_loader_iter = enumerate(islice(self.train_loader, iter_idx, None), start=iter_idx)
        # 5) next step 
        step = ckpt.get('step', None)
        self.start_step = (step + 1) if (step is not None) else 0
        if self.master_process:
            print(f"=> Resumed from {os.path.basename(ckpt_path)} | next_step={self.start_step}, "
                f"sampler_epoch={epoch}, dataloader_iter_idx={iter_idx}")
    
    def train(self):
        self.results = {}
        steps_per_epoch = max(1, len(self.train_loader) // self.training_info['grad_accum_steps'])
        self.train_loader_iter = enumerate(self.train_loader)
        self._resume_from_checkpoint(steps_per_epoch)
        # training loop
        for step in tqdm(range(self.start_step, self.training_info["max_steps"]), desc="Train", disable=(self.dp_rank != 0)):
            self.one_step_results = {}
            t0 = time.time()
            last_step = (step == self.training_info["max_steps"] - 1)
            # 1) train
            self._one_training_step(self.config, step)
            torch.cuda.synchronize()
            # 2) eval
            if not self.config.debug and self.config.do_val and (step % self.config.val_every_steps == 0 or last_step):
                self.eval(step)
                if self.master_process:
                    tqdm.write(f"validation loss: {self.one_step_results['val_loss'].item():.4f}")
                with open(self.log_file, "a") as f:
                    f.write(f"{step} val {self.one_step_results["val_loss"].item():.4f}\n")
            # 3) save
            if not self.config.debug and step > 0 and (step % self.config.save_every_steps == 0 or last_step):
                if self.master_process:
                    self.save(step)
            # 4) print
            t1 = time.time()
            dt = t1 - t0 # time difference in seconds
            tokens_processed = self.train_dataset.batch_size * self.config.T * self.training_info["grad_accum_steps"] * self.dp_world_size
            tokens_per_sec = tokens_processed / dt
            if self.master_process:
                tqdm.write(f"step {step:5d} | loss: {self.one_step_results['loss'].item():.6f} | lr {self.one_step_results['lr']:.4e} | grad norm: {self.one_step_results['grad_norm']:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
                with open(self.log_file, "a") as f:
                    f.write(f"{step} train {self.one_step_results["loss"].item():.6f}\n")
            self.results[step] = self.one_step_results
        destroy_process_group()

    def eval(self):
        self.model.eval()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = self.config.B * len(self.val_loader) // (self.config.B * self.dp_world_size)
            for batch in tqdm(self.val_loader, desc="Val", disable=(self.dp_rank != 0)):
                x, y = batch["input_ids"], batch["labels"]
                x, y = x.to(f'cuda:{self.dp_local_rank}'), y.to(f'cuda:{self.dp_local_rank}')
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits, loss = self.model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        torch.cuda.synchronize()
        self.one_step_results["val_loss"] = val_loss_accum
    
    def save(self, step: int = None):
        # optionally write model checkpoints
        checkpoint_path = os.path.join(self.log_dir, f"model_{step:05d}.pt")
        steps_per_epoch = max(1, len(self.train_loader) // self.training_info['grad_accum_steps'])
        next_step = (step if step is not None else 0) + 1
        sampler_epoch_next = next_step // steps_per_epoch
        sampler_iter_idx_next = (next_step % steps_per_epoch) * self.training_info['grad_accum_steps']
        rng_state = {
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state(self.dp_local_rank),
            'numpy': np.random.get_state(),
        }
        checkpoint = {
            'config': self.config,
            'model': self.raw_model.state_dict(),
            'model_config': self.raw_model.config,
            'optimizer_state': self.optimizer.state_dict(),
            'step': step,
            'this_step_results': self.one_step_results,
            'dataset_state': {
                'train': self._get_dataset_state(self.train_dataset),
                'val': self._get_dataset_state(self.val_dataset) if self.val_dataset is not None else None,
            },
            'sampler_state': {
                'epoch': sampler_epoch_next,
                'iter_idx': sampler_iter_idx_next,
            },
            'rng_state': rng_state,
        }
        torch.save(checkpoint, checkpoint_path)