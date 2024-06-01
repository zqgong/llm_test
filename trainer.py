from __future__ import print_function

import argparse
import os
import time
import math
import glob
from pathlib import Path
import sys
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))
from typing import Any, Dict
from typing import Optional, Tuple, Union
import lightning as L
from lightning.pytorch.strategies import DeepSpeedStrategy, FSDPStrategy

from functools import partial
from lightning.pytorch.callbacks import ModelCheckpoint

from torch.utils.data import DataLoader
from torchmetrics.aggregation import RunningMean
from typing_extensions import Literal
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.loggers import CSVLogger

# from litgpt import Tokenizer
from litgpt.tokenizer_v2 import TTokenizer

from litgpt.args import EvalArgs, TrainArgs
from litgpt.config import name_to_config
from litgpt.data import DataModule, TinyLlama
from litgpt.model_v2 import GPT, Block, CausalSelfAttention, Config, LLaMAMLP
from litgpt.utils import (
    CLI,
    CycleIterator,
    capture_hparams,
    chunked_cross_entropy,
    copy_config_files,
    get_default_supported_precision,
    init_out_dir,
    num_parameters,
    parse_devices,
    reset_parameters,
    save_config,
    save_hyperparameters,
)
from deepspeed.ops.adam import FusedAdam, DeepSpeedCPUAdam
import deepspeed


MASTER_PORT = os.environ.get('MASTER_PORT', 29500)
MASTER_ADDR = os.environ.get('MASTER_ADDR', "localhost")
WORLD_SIZE = int(os.environ.get('WORLD_SIZE', 1))
RANK = os.environ.get('RANK', 0)
INT_RANK = int(RANK)
LOCAL_RANK = os.environ.get('LOCAL_RANK', 0)
INT_LOCAL_RANK = int(LOCAL_RANK)
print(
    "MASTER_ADDR:{}, MASTER_PORT:{}, WORLD_SIZE:{}, RANK:{}, LOCAL_RANK:{}".format(MASTER_ADDR, MASTER_PORT, WORLD_SIZE,
                                                                                   RANK, LOCAL_RANK))
class ParallelConfig:

    def __init__(self,
                optimizer = "AdamW", 
                cpu_checkpoint = False,  
                strategy_name = "deepspeed_2"):

        self.optimizer = optimizer 
        self.cpu_checkpoint = cpu_checkpoint  
        self.strategy_name = strategy_name

    # training
    # TODO
    # FusedAdam AdamW DeepSpeedCPUAdam
    # deepspeed_3 deepspeed_2 fsdp
    # deepspeed_3: DeepSpeedCPUAdam  cpu_checkpoint True batch_size 16
    # fsdp: AdamW  cpu_checkpoint False

    # optimizer = "AdamW" 
    # cpu_checkpoint = False  
    # strategy_name = "fsdp"  

    # optimizer = "DeepSpeedCPUAdam" 
    # cpu_checkpoint = True  
    # strategy_name = "deepspeed_2" 

    # optimizer = "AdamW" 
    # cpu_checkpoint = False  
    # strategy_name = "deepspeed_2" 

    # optimizer = "DeepSpeedCPUAdam" 
    # cpu_checkpoint = True  
    # strategy_name = "deepspeed_3"  



class LightningGPTModule(L.LightningModule):
    def __init__(self, 
                config: Config,
                train_args: TrainArgs, 
                eval_args: EvalArgs,
                parallel_config,
                warmup_iters, 
                max_iters,
                gradient_accumulation_iters) -> None:
        super().__init__()
        self.warmup_iters = warmup_iters
        self.max_iters = max_iters
        self.train_args = train_args
        self.eval_args = eval_args
        self.model_config = config
        self.parallel_config = parallel_config

        self.module: Optional[torch.nn.Module] = None
        self.flops_per_batch: Optional[int] = None

        self.running_loss = RunningMean(window=gradient_accumulation_iters, sync_on_compute=False)
        self.val_losses = []

    def configure_model(self) -> None:
        self.module = GPT(self.model_config)
        if self.model_config.rnn_type is None:
            self.module.apply(partial(initialize_weights, n_layer=self.model_config.n_layer, n_embd=self.model_config.n_embd))

        if self.train_args.tie_embeddings:
            self.module.transformer.wte.weight = self.module.lm_head.weight
        if self.train_args.max_seq_length:
            self.module.max_seq_length = self.train_args.max_seq_length

    def configure_optimizers(self) -> torch.optim.Optimizer:
        if self.parallel_config.optimizer == "AdamW":
            return torch.optim.AdamW(self.module.parameters(),
                lr=self.train_args.learning_rate, 
                weight_decay=self.train_args.weight_decay, 
                betas=(self.train_args.beta1, self.train_args.beta2), 
            )
        elif self.parallel_config.optimizer == "FusedAdam":
            return FusedAdam(self.module.parameters(),
                lr=self.train_args.learning_rate, 
                weight_decay=self.train.weight_decay, 
                betas=(self.train_args.beta1, self.train_args.beta2), 
            )
        else:
            return DeepSpeedCPUAdam(self.module.parameters(),
                lr=self.train_args.learning_rate, 
                weight_decay=self.train_args.weight_decay, 
                betas=(self.train_args.beta1, self.train_args.beta2), 
            )

    def on_fit_start(self) -> None:
        trainer = self.trainer

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> None:

        # determine and set the learning rate for this iteration
        lr = get_lr(learning_rate=self.train_args.learning_rate,
                    it=self.trainer.fit_loop.total_batch_idx, 
                    warmup_iters=self.warmup_iters, 
                    max_iters=self.max_iters, 
                    min_lr=self.train_args.min_lr)

        for optimizer in self.trainer.strategy.optimizers:
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
        self.log("lr", lr, on_step=True, on_epoch=False, prog_bar=False)

    def forward(self, idx):
        return self.module(idx)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        time1 = time.perf_counter()

        input_ids = batch[:, 0 : self.module.max_seq_length].contiguous().long()
        targets = batch[:, 1 : (self.module.max_seq_length + 1)].contiguous().long()

        logits = self(input_ids)
        loss = chunked_cross_entropy(logits, targets)

        self.running_loss.update(loss.detach())

        time2 = time.perf_counter()
        self.log("train_loss", self.running_loss.compute().item(), on_step=True, on_epoch=False, prog_bar=True)
        self.log("iter_time", time2 - time1, on_step=True, on_epoch=False, prog_bar=True)
        self.log("batch_idx", batch_idx, on_step=True, on_epoch=False, prog_bar=True)

        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        
        if batch_idx == 0:
            self.val_losses = []

        if batch_idx >= self.eval_args.max_iters:
            return

        input_ids = batch[:, 0 : self.module.max_seq_length].contiguous().long()
        targets = batch[:, 1 : (self.module.max_seq_length + 1)].contiguous().long()
        logits = self(input_ids)
        loss = chunked_cross_entropy(logits, targets)
        self.val_losses.append(loss)
        val_loss = torch.stack(self.val_losses).mean()

        val_loss = val_loss.item()
        
        self.log("val_loss", val_loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log("val_ppl", math.exp(val_loss), on_step=True, on_epoch=False, prog_bar=True)

def setup(
    num_nodes: int = 1,
    model_name: Optional[str] = None,
    model_config: Optional[Config] = None,
    out_dir: Path = Path("out/pretrain"),
    precision: Literal["bf16-true", "bf16-mixed", "32-true", None] = None,
    initial_checkpoint_dir: Optional[Path] = None,
    resume: Union[bool, Path] = False,
    data: Optional[DataModule] = None,
    train: TrainArgs = TrainArgs(
        save_interval=1000,
        log_interval=1,
        global_batch_size=512,
        micro_batch_size=4,
        max_tokens=int(3e12),  # 3 trillion
        learning_rate=4e-4,
        weight_decay=1e-1,
        beta1=0.9,
        beta2=0.95,
        max_norm=1.0,
        min_lr=4e-5,
        lr_warmup_steps=2000,
        tie_embeddings=False,
    ),
    eval: EvalArgs = EvalArgs(interval=1000, max_iters=100),
    devices: Union[int, str] = "auto",
    tokenizer_dir: Optional[Path] = None,
    logger_name: Literal["wandb", "tensorboard", "csv"] = "csv",
    seed: int = 42,
    parallel_config=ParallelConfig(),
    ):

    hparams = capture_hparams()
    data = TinyLlama() if data is None else data
    if model_config is not None and model_name is not None:
        raise ValueError("Only one of `model_name` or `model_config` can be set.")
    elif model_config is None and model_name is None:
        available_models = "\n".join(sorted(name_to_config))
        raise ValueError(f"Please specify --model_name <model_name>. Available values:\n{available_models}")
    config = Config.from_name(model_name) if model_config is None else model_config
    precision = precision or get_default_supported_precision(training=True)
    devices = parse_devices(devices)
    print("devices", devices)
    # print("train", train)
    out_dir = init_out_dir(out_dir)
    # in case the dataset requires the Tokenizer
    tokenizer = TTokenizer(tokenizer_dir) if tokenizer_dir is not None else None

    logger = CSVLogger(out_dir, name=f"pretrain-{config.name}", flush_logs_every_n_steps=train.log_interval)

    if devices > 1:
        if parallel_config.strategy_name == "deepspeed_3":
            strategy = DeepSpeedStrategy(
                stage=3,
                offload_optimizer=True,
                offload_parameters=True,
                cpu_checkpointing=False,
                pin_memory=True, 
            )
        elif parallel_config.strategy_name == "deepspeed_2":
            strategy = DeepSpeedStrategy(
                stage=2,
                offload_optimizer=parallel_config.cpu_checkpoint,
                pin_memory=parallel_config.cpu_checkpoint, 
                allgather_bucket_size=5e8,
                reduce_bucket_size=5e8
            )
        else:
            strategy = FSDPStrategy(
                auto_wrap_policy={Block},
                activation_checkpointing_policy=None,
                state_dict_type="full",
                limit_all_gathers=True,
                cpu_offload=parallel_config.cpu_checkpoint,
                sharding_strategy="HYBRID_SHARD", # 'FULL_SHARD', 'SHARD_GRAD_OP', 'NO_SHARD', 'HYBRID_SHARD'
            )
    else:
        strategy = "auto"

    world_size = devices * num_nodes
    max_tokens_per_device = train.max_tokens // world_size
    tokens_per_iter = train.micro_batch_size * train.max_seq_length
    max_iters = max_tokens_per_device // tokens_per_iter
    gradient_accumulation_iters = train.gradient_accumulation_iters(devices)
    log_iter_interval = train.log_interval * gradient_accumulation_iters

    validate_args(train, eval, initial_checkpoint_dir, resume)

    model_checkpoint = ModelCheckpoint(dirpath=out_dir, 
                                    every_n_train_steps=train.save_interval, 
                                    save_last=True, 
                                    verbose=True)
    trainer = L.Trainer(
        accelerator="gpu",
        devices=devices,
        strategy=strategy,
        num_nodes=num_nodes,
        precision=precision,
        logger=logger,
        callbacks=[model_checkpoint],
        max_steps=max_iters // gradient_accumulation_iters,
        limit_train_batches=max_iters,
        enable_checkpointing=True,
        use_distributed_sampler=True,
        limit_val_batches=eval.max_iters,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        accumulate_grad_batches=gradient_accumulation_iters,
        log_every_n_steps=train.log_interval,
        val_check_interval=eval.interval,
    )

    logger.log_hyperparams(hparams)

    train_dataloader, val_dataloader = get_dataloaders(data, tokenizer, train, train.max_seq_length)
    warmup_iters = train.warmup_iters(devices, max_iters, train_dataloader)

    trainer.print(f"Loading model with {config.__dict__}")
    t0 = time.perf_counter()
    model = LightningGPTModule(config, 
                                train, 
                                eval, 
                                parallel_config=parallel_config,
                                warmup_iters=warmup_iters, 
                                max_iters=max_iters,
                                gradient_accumulation_iters=gradient_accumulation_iters)

    trainer.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.")
    trainer.print(f"Total parameters: {num_parameters(model):,}")

    L.seed_everything(seed)

    # ================== save config
    save_hyperparameters(setup, out_dir)
    if tokenizer_dir is not None:
        copy_config_files(tokenizer_dir, out_dir)
    save_config(config, out_dir)
    # ================== 

    train_time = time.perf_counter()
    trainer.fit(model, train_dataloader, val_dataloader, ckpt_path="last")
    trainer.print(f"Training time: {(time.perf_counter()-train_time):.2f}s")
    if trainer.accelerator == "cuda":
        trainer.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")
    

def get_dataloaders(data: DataModule, tokenizer, train: TrainArgs, block_size: int
) -> Tuple[DataLoader, DataLoader]:
    data.connect(tokenizer=tokenizer, batch_size=train.micro_batch_size, max_seq_length=block_size)
    data.prepare_data()
    data.setup()
    train_dataloader = data.train_dataloader()
    val_dataloader = data.val_dataloader()
    return train_dataloader, val_dataloader


# learning rate decay scheduler (cosine with linear warmup)
def get_lr(learning_rate: float, it: int, warmup_iters: int, max_iters: int, min_lr: float) -> float:
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > max_iters, return min learning rate
    if it > max_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


def initialize_weights(model: GPT, n_layer: int, n_embd: int) -> None:
    """GPT-NeoX weight initialization (https://arxiv.org/abs/2204.06745)."""
    # Adapted from https://github.com/jzhang38/TinyLlama

    def init_weights(module, std):
        nn.init.normal_(module.weight, mean=0.0, std=std)
        if getattr(module, "bias", None) is not None:
            nn.init.zeros_(module.bias)

    for mod in model.modules():
        if isinstance(mod, (nn.Embedding, nn.Linear)):
            mod.reset_parameters = partial(init_weights, mod, std=math.sqrt(2.0 / 5 / n_embd))

    # need a separate loop because `mod.proj` below is a `nn.Linear` too
    for mod in model.modules():
        if isinstance(mod, (LLaMAMLP, CausalSelfAttention)):
            mod.proj.reset_parameters = partial(init_weights, mod.proj, std=(1 / math.sqrt(n_embd) / n_layer))


def validate_args(train: TrainArgs, eval: EvalArgs, initial_checkpoint_dir, resume) -> None:
    issues = []
    unsupported = [(train, ["max_steps", "epochs"]), (eval, ["max_new_tokens"])]
    for args, names in unsupported:
        for name in names:
            if getattr(args, name) is not None:
                issues.append(f"{__file__} doesn't support the {name!r} argument. This is set in {args}")
    required = [(train, ["max_tokens", "max_norm"])]
    for args, names in required:
        for name in names:
            if getattr(args, name) is None:
                issues.append(f"{__file__} requires the {name!r} argument. This is set in {args}")
    if initial_checkpoint_dir and resume:
        issues.append("Can't provide both `--resume` and `--initial_checkpoint_dir`. Choose one.")
    if issues:
        raise ValueError("\n".join(issues))



if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    CLI(setup)
