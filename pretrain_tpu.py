from typing import Optional, Any, Sequence, List
from dataclasses import dataclass
import os
import math
import yaml
import shutil
import copy

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader

import tqdm
import wandb
import coolname
import hydra
import pydantic
from omegaconf import DictConfig
from adam_atan2 import AdamATan2

# TPU-specific imports
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata
from utils.functions import load_model_class, get_model_source_path
from models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed
from models.ema import EMAHelper


class LossConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str


class ArchConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str
    loss: LossConfig


class EvaluatorConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")
    name: str


class PretrainConfig(pydantic.BaseModel):
    # Config
    arch: ArchConfig
    # Data
    data_paths: List[str]
    data_paths_test: List[str] = []
    # Evaluators
    evaluators: List[EvaluatorConfig] = []

    # Hyperparams
    global_batch_size: int
    epochs: int

    lr: float
    lr_min_ratio: float
    lr_warmup_steps: int

    weight_decay: float
    beta1: float
    beta2: float

    # Puzzle embedding
    puzzle_emb_lr: float
    puzzle_emb_weight_decay: float

    # Names
    project_name: Optional[str] = None
    run_name: Optional[str] = None
    load_checkpoint: Optional[str] = None
    checkpoint_path: Optional[str] = None

    # Extras
    seed: int = 0
    checkpoint_every_eval: bool = False
    eval_interval: Optional[int] = None
    min_eval_interval: Optional[int] = 0
    eval_save_outputs: List[str] = []

    ema: bool = False
    ema_rate: float = 0.999
    freeze_weights: bool = False


@dataclass
class TrainState:
    model: nn.Module
    optimizers: Sequence[torch.optim.Optimizer]
    optimizer_lrs: Sequence[float]
    carry: Any

    step: int
    total_steps: int


def move_to_device(obj, device):
    """Recursively move nested structures to device (non-mutating version)"""
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(move_to_device(v, device) for v in obj)
    elif hasattr(obj, '__dict__'):
        # Create a copy to avoid mutating shared objects
        new_obj = copy.copy(obj)
        for k, v in obj.__dict__.items():
            setattr(new_obj, k, move_to_device(v, device))
        return new_obj
    return obj


# ---------- Critical XLA helper ----------
def xla_all_reduce_sum(tensor):
    """Reduce a single tensor across replicas and return the reduced tensor."""
    return xm.all_reduce('sum', [tensor])[0]
# -----------------------------------------


def create_dataloader(config: PretrainConfig, split: str, rank: int, world_size: int, 
                     test_set_mode: bool = False, epochs_per_iter: Optional[int] = None, 
                     global_batch_size: Optional[int] = None):
    """Create dataloader with proper argument handling"""
    # Build dataset config kwargs
    dataset_kwargs = {
        'seed': config.seed,
        'dataset_paths': config.data_paths_test if len(config.data_paths_test) > 0 and split == "test" else config.data_paths,
        'rank': rank,
        'num_replicas': world_size
    }
    
    # Add optional parameters if they exist in PuzzleDatasetConfig
    if epochs_per_iter is not None:
        dataset_kwargs['epochs_per_iter'] = epochs_per_iter
    if global_batch_size is not None:
        dataset_kwargs['global_batch_size'] = global_batch_size
    
    dataset_config = PuzzleDatasetConfig(**dataset_kwargs)
    dataset = PuzzleDataset(dataset_config, split=split)
    
    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=1,
        prefetch_factor=8,
        pin_memory=False,
        persistent_workers=True
    )
    return dataloader, dataset.metadata


def create_model(config: PretrainConfig, train_metadata: PuzzleDatasetMetadata, rank: int, world_size: int, device):
    model_cfg = dict(
        **config.arch.__pydantic_extra__,
        batch_size=config.global_batch_size // world_size,
        vocab_size=train_metadata.vocab_size,
        seq_len=train_metadata.seq_len,
        num_puzzle_identifiers=train_metadata.num_puzzle_identifiers,
        causal=False
    )

    # Instantiate model with loss head
    model_cls = load_model_class(config.arch.name)
    loss_head_cls = load_model_class(config.arch.loss.name)

    # Create model directly on XLA device
    model: nn.Module = model_cls(model_cfg)
    print(f"[Rank {rank}] Model created")
    model = loss_head_cls(model, **config.arch.loss.__pydantic_extra__)
    
    # Move model to XLA device
    model = model.to(device)
    
    # torch.compile is not compatible with XLA
    print(f"[Rank {rank}] Note: torch.compile disabled for TPU compatibility")

    # Load checkpoint on rank 0 only
    if rank == 0:
        load_checkpoint(model, config, device)

    # FIXED: Proper parameter broadcast using XLA built-in helper if available
    xm.rendezvous("after_checkpoint_load")
    if world_size > 1:
        try:
            xm.broadcast_master_param(model)
        except Exception:
            # fallback: attempt to average parameters across replicas (best-effort)
            for p in list(model.parameters()) + list(model.buffers()):
                p.data = xla_all_reduce_sum(p.data) / float(world_size)

    # Optimizers
    if config.arch.puzzle_emb_ndim == 0:
        optimizers = [
            AdamATan2(
                model.parameters(),
                lr=0,
                weight_decay=config.weight_decay,
                betas=(config.beta1, config.beta2)
            )
        ]
        optimizer_lrs = [config.lr]
    elif config.freeze_weights:
        optimizers = [
            CastedSparseEmbeddingSignSGD_Distributed(
                model.model.puzzle_emb.buffers(),
                lr=0,
                weight_decay=config.puzzle_emb_weight_decay,
                world_size=world_size
            )
        ]
        optimizer_lrs = [config.puzzle_emb_lr]
    else:
        optimizers = [
            CastedSparseEmbeddingSignSGD_Distributed(
                model.model.puzzle_emb.buffers(),
                lr=0,
                weight_decay=config.puzzle_emb_weight_decay,
                world_size=world_size
            ),
            AdamATan2(
                model.parameters(),
                lr=0,
                weight_decay=config.weight_decay,
                betas=(config.beta1, config.beta2)
            )
        ]
        optimizer_lrs = [config.puzzle_emb_lr, config.lr]

    return model, optimizers, optimizer_lrs


def cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, base_lr: float, num_warmup_steps: int, num_training_steps: int, min_ratio: float = 0.0, num_cycles: float = 0.5
):
    if current_step < num_warmup_steps:
        return base_lr * float(current_step) / float(max(1, num_warmup_steps))

    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return base_lr * (min_ratio + max(0.0, (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))))


def init_train_state(config: PretrainConfig, train_metadata: PuzzleDatasetMetadata, rank: int, world_size: int, device):
    total_steps = int(config.epochs * train_metadata.total_groups * train_metadata.mean_puzzle_examples / config.global_batch_size)

    model, optimizers, optimizer_lrs = create_model(config, train_metadata, rank=rank, world_size=world_size, device=device)

    return TrainState(
        step=0,
        total_steps=total_steps,
        model=model,
        optimizers=optimizers,
        optimizer_lrs=optimizer_lrs,
        carry=None
    )


def save_train_state(config: PretrainConfig, train_state: TrainState, rank: int):
    if config.checkpoint_path is None:
        return

    if rank == 0:
        os.makedirs(config.checkpoint_path, exist_ok=True)
        cpu_state_dict = {k: v.cpu() for k, v in train_state.model.state_dict().items()}
        torch.save(cpu_state_dict, os.path.join(config.checkpoint_path, f"step_{train_state.step}"))


def load_checkpoint(model: nn.Module, config: PretrainConfig, device):
    if config.load_checkpoint is not None:
        print(f"Loading checkpoint {config.load_checkpoint}")

        state_dict = torch.load(config.load_checkpoint, map_location="cpu")

        puzzle_emb_name = "_orig_mod.model.inner.puzzle_emb.weights"
        expected_shape: torch.Size = model.model.puzzle_emb.weights.shape
        if puzzle_emb_name in state_dict:
            puzzle_emb = state_dict[puzzle_emb_name]
            if puzzle_emb.shape != expected_shape:
                print(f"Resetting puzzle embedding as shape is different. Found {puzzle_emb.shape}, Expected {expected_shape}")
                state_dict[puzzle_emb_name] = (
                    torch.mean(puzzle_emb, dim=0, keepdim=True).expand(expected_shape).contiguous()
                )
        
        # FIXED: Move state dict to target device before loading
        state_dict = {k: v.to(device) for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)


def compute_lr(base_lr: float, config: PretrainConfig, train_state: TrainState):
    return cosine_schedule_with_warmup_lr_lambda(
        current_step=train_state.step,
        base_lr=base_lr,
        num_warmup_steps=round(config.lr_warmup_steps),
        num_training_steps=train_state.total_steps,
        min_ratio=config.lr_min_ratio
    )


def create_evaluators(config: PretrainConfig, eval_metadata: PuzzleDatasetMetadata) -> List[Any]:
    data_paths = config.data_paths_test if len(config.data_paths_test) > 0 else config.data_paths
    evaluators = []
    for cfg in config.evaluators:
        for data_path in data_paths:
            cls = load_model_class(cfg.name, "evaluators.")(
                data_path=data_path, eval_metadata=eval_metadata, **cfg.__pydantic_extra__
            )
            evaluators.append(cls)
    return evaluators


def train_batch(config: PretrainConfig, train_state: TrainState, batch: Any, rank: int, world_size: int, device):
    train_state.step += 1
    if train_state.step > train_state.total_steps:
        return

    # FIXED: Init carry and ensure it's on the correct device
    if train_state.carry is None:
        train_state.carry = train_state.model.initial_carry(batch)
        train_state.carry = move_to_device(train_state.carry, device)

    # Forward
    train_state.carry, loss, metrics, _, _ = train_state.model(carry=train_state.carry, batch=batch, return_keys=[])

    # FIXED: Use global batch size from config, not from loader
    loss = loss / config.global_batch_size
    loss.backward()

    # FIXED: Manual gradient reduction (required before optimizer step)
    if world_size > 1:
        for param in train_state.model.parameters():
            if param.grad is not None:
                param.grad = xla_all_reduce_sum(param.grad)
    
    # FIXED: Robust optimizer step with fallback for custom optimizers
    lr_this_step = None
    for optim, base_lr in zip(train_state.optimizers, train_state.optimizer_lrs):
        lr_this_step = compute_lr(base_lr, config, train_state)
        for param_group in optim.param_groups:
            param_group['lr'] = lr_this_step
        
        # Try xm.optimizer_step, fallback to regular step for custom optimizers
        try:
            xm.optimizer_step(optim)
        except Exception:
            # Custom optimizers like AdamATan2 may not work with xm.optimizer_step
            optim.step()
        
        optim.zero_grad()
    
    # FIXED: Mark step immediately after optimizer updates
    xm.mark_step()

    # FIXED: Proper metric reduction
    if len(metrics):
        assert not any(v.requires_grad for v in metrics.values())
        metric_keys = list(sorted(metrics.keys()))
        metric_values = torch.stack([metrics[k] for k in metric_keys])
        
        if world_size > 1:
            metric_values = xla_all_reduce_sum(metric_values)

        if rank == 0:
            metric_values = metric_values.cpu().numpy()
            reduced_metrics = {k: metric_values[i] for i, k in enumerate(metric_keys)}
            count = max(reduced_metrics["count"], 1)
            reduced_metrics = {f"train/{k}": v / (config.global_batch_size if k.endswith("loss") else count) for k, v in reduced_metrics.items()}
            reduced_metrics["train/lr"] = lr_this_step
            return reduced_metrics


def evaluate(
    config: PretrainConfig,
    train_state: TrainState,
    eval_loader: torch.utils.data.DataLoader,
    eval_metadata: PuzzleDatasetMetadata,
    evaluators: List[Any],
    rank: int,
    world_size: int,
    device
):
    reduced_metrics = None

    with torch.inference_mode():
        return_keys = set(config.eval_save_outputs)
        for evaluator in evaluators:
            evaluator.begin_eval()
            return_keys.update(evaluator.required_outputs)

        set_ids = {k: idx for idx, k in enumerate(eval_metadata.sets)}
        save_preds = {}
        metric_keys = []
        metric_values = None
        carry = None
        processed_batches = 0
        
        for set_name, batch, global_batch_size in eval_loader:
            processed_batches += 1
            if rank == 0:
                print(f"Processing batch {processed_batches}: {set_name}")
            
            # FIXED: Ensure carry is on correct device
            carry = train_state.model.initial_carry(batch)
            carry = move_to_device(carry, device)

            # Forward
            inference_steps = 0
            while True:
                carry, loss, metrics, preds, all_finish = train_state.model(
                    carry=carry, batch=batch, return_keys=return_keys
                )
                inference_steps += 1
                if all_finish:
                    break

            if rank == 0:
                print(f"  Completed inference in {inference_steps} steps")

            # Save predictions to CPU
            for collection in (batch, preds):
                for k, v in collection.items():
                    if k in config.eval_save_outputs:
                        save_preds.setdefault(k, [])
                        save_preds[k].append(v.cpu())

            for evaluator in evaluators:
                evaluator.update_batch(batch, preds)

            # Aggregate metrics
            set_id = set_ids[set_name]

            if metric_values is None:
                metric_keys = list(sorted(metrics.keys()))
                metric_values = torch.zeros(
                    (len(set_ids), len(metrics.values())), dtype=torch.float32, device=device
                )

            metric_values[set_id] += torch.stack([metrics[k] for k in metric_keys])
            
            # Mark step for XLA graph execution
            xm.mark_step()

        # Concatenate save preds
        save_preds = {k: torch.cat(v, dim=0) for k, v in save_preds.items()}

        # Save preds
        if config.checkpoint_path is not None and len(save_preds):
            os.makedirs(os.path.dirname(config.checkpoint_path), exist_ok=True)
            torch.save(
                save_preds, os.path.join(config.checkpoint_path, f"step_{train_state.step}_all_preds.{rank}")
            )

        # FIXED: Proper metric reduction
        if metric_values is not None:
            if world_size > 1:
                metric_values = xla_all_reduce_sum(metric_values)

            if rank == 0:
                reduced_metrics = metric_values.cpu().numpy()
                reduced_metrics = {
                    set_name: {
                        metric_name: reduced_metrics[set_id, metric_id]
                        for metric_id, metric_name in enumerate(metric_keys)
                    }
                    for set_id, set_name in enumerate(set_ids)
                }

                for set_name, m in reduced_metrics.items():
                    count = m.pop("count")
                    reduced_metrics[set_name] = {k: v / count for k, v in m.items()}

        # Run evaluators
        if rank == 0:
            print(f"\nRunning {len(evaluators)} evaluator(s)...")
            
        for i, evaluator in enumerate(evaluators):
            if rank == 0:
                print(f"Running evaluator {i+1}/{len(evaluators)}: {evaluator.__class__.__name__}")
                
            evaluator_save_path = None
            if config.checkpoint_path is not None:
                evaluator_save_path = os.path.join(
                    config.checkpoint_path,
                    f"evaluator_{evaluator.__class__.__name__}_step_{train_state.step}",
                )
                if rank == 0:
                    os.makedirs(evaluator_save_path, exist_ok=True)

            metrics = evaluator.result(evaluator_save_path, rank=rank, world_size=world_size, group=None)
            if rank == 0 and metrics is not None:
                if reduced_metrics is None:
                    reduced_metrics = {}
                reduced_metrics.update(metrics)
                print(f"  Completed {evaluator.__class__.__name__}")
                
        if rank == 0:
            print("All evaluators completed!")

    return reduced_metrics


def save_code_and_config(config: PretrainConfig):
    if config.checkpoint_path is None or wandb.run is None:
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)

    code_list = [
        get_model_source_path(config.arch.name),
        get_model_source_path(config.arch.loss.name)
    ]
    for code_file in code_list:
        if code_file is not None:
            code_name = os.path.basename(code_file)
            shutil.copy(code_file, os.path.join(config.checkpoint_path, code_name))

    config_file = os.path.join(config.checkpoint_path, "all_config.yaml")
    with open(config_file, "wt") as f:
        yaml.dump(config.model_dump(), f)

    wandb.run.log_code(config.checkpoint_path)


def _mp_fn(index, config):
    """Main training function that runs on each TPU core"""
    # Get XLA device
    device = xm.xla_device()
    
    # Get rank and world size
    RANK = xm.get_ordinal()
    WORLD_SIZE = xm.xrt_world_size()
    
    print(f"[Rank {RANK}/{WORLD_SIZE}] Running on device: {device}")

    # FIXED: Use SAME seed on all ranks for identical initialization
    try:
        xm.set_rng_seed(int(config.seed))
    except Exception:
        pass
    torch.manual_seed(int(config.seed))

    # Dataset
    train_epochs_per_iter = config.eval_interval if config.eval_interval is not None else config.epochs
    total_iters = config.epochs // train_epochs_per_iter

    assert config.epochs % train_epochs_per_iter == 0, "Eval interval must be a divisor of total epochs."

    train_loader, train_metadata = create_dataloader(
        config, "train", rank=RANK, world_size=WORLD_SIZE,
        test_set_mode=False, epochs_per_iter=train_epochs_per_iter, 
        global_batch_size=config.global_batch_size
    )
    
    train_loader = pl.MpDeviceLoader(train_loader, device)
    
    try:
        eval_loader, eval_metadata = create_dataloader(
            config, "test", rank=RANK, world_size=WORLD_SIZE,
            test_set_mode=True, epochs_per_iter=1, 
            global_batch_size=config.global_batch_size
        )
        eval_loader = pl.MpDeviceLoader(eval_loader, device)
    except:
        print("NO EVAL DATA FOUND")
        eval_loader = eval_metadata = None

    try:
        evaluators = create_evaluators(config, eval_metadata)
    except:
        print("No evaluator found")
        evaluators = []

    # Train state
    train_state = init_train_state(config, train_metadata, rank=RANK, world_size=WORLD_SIZE, device=device)

    # Progress bar and logger (only on rank 0)
    progress_bar = None
    ema_helper = None
    if RANK == 0:
        progress_bar = tqdm.tqdm(total=train_state.total_steps)
        wandb.init(project=config.project_name, name=config.run_name, config=config.model_dump(), 
                   settings=wandb.Settings(_disable_stats=True))
        wandb.log({"num_params": sum(x.numel() for x in train_state.model.parameters())}, step=0)
        save_code_and_config(config)
    
    if config.ema:
        print('Setup EMA')
        ema_helper = EMAHelper(mu=config.ema_rate)
        ema_helper.register(train_state.model)

    # Training Loop
    for _iter_id in range(total_iters):
        print(f"[Rank {RANK}, World Size {WORLD_SIZE}]: Epoch {_iter_id * train_epochs_per_iter}")

        ############ Train Iter
        if RANK == 0:
            print("TRAIN")
        train_state.model.train()
        
        for set_name, batch, global_batch_size in train_loader:
            metrics = train_batch(config, train_state, batch, rank=RANK, world_size=WORLD_SIZE, device=device)

            if RANK == 0 and metrics is not None:
                wandb.log(metrics, step=train_state.step)
                progress_bar.update(train_state.step - progress_bar.n)
            
            if config.ema:
                ema_helper.update(train_state.model)

        if _iter_id >= config.min_eval_interval:
            ############ Evaluation
            if RANK == 0:
                print("EVALUATE")
            
            if config.ema:
                print("SWITCH TO EMA")
                train_state_eval = copy.deepcopy(train_state)
                train_state_eval.model = ema_helper.ema_copy(train_state_eval.model)
            else:
                train_state_eval = train_state
            
            train_state_eval.model.eval()
            metrics = evaluate(
                config, train_state_eval, eval_loader, eval_metadata, evaluators,
                rank=RANK, world_size=WORLD_SIZE, device=device
            )

            if RANK == 0 and metrics is not None:
                wandb.log(metrics, step=train_state.step)
                
            ############ Checkpointing
            if RANK == 0:
                print("SAVE CHECKPOINT")
            
            if RANK == 0 and (config.checkpoint_every_eval or (_iter_id == total_iters - 1)):
                save_train_state(config, train_state_eval, RANK)

            if config.ema:
                del train_state_eval

    # Finalize
    if RANK == 0:
        wandb.finish()
    
    xm.rendezvous("training_complete")


@hydra.main(config_path="config", config_name="cfg_pretrain", version_base=None)
def launch(hydra_config: DictConfig):
    """Main entry point - spawns processes for each TPU core"""
    os.environ['PJRT_DEVICE'] = 'TPU'
    
    # FIXED: Load config once on main process, pass to all workers
    config = PretrainConfig(**hydra_config)
    
    # Naming (only needed on main process)
    if config.project_name is None:
        config.project_name = f"{os.path.basename(config.data_paths[0]).capitalize()}-ACT-torch"
    if config.run_name is None:
        config.run_name = f"{config.arch.name.split('@')[-1]} {coolname.generate_slug(2)}"
    if config.checkpoint_path is None:
        config.checkpoint_path = os.path.join("checkpoints", config.project_name, config.run_name)
    
    # Check for interactive environment
    try:
        get_ipython()
        is_interactive = True
    except:
        is_interactive = False
    
    print(f"Launching training on TPU...")
    print(f"Interactive mode: {is_interactive}")
  
    # Spawn training processes with config as argument
    # CRITICAL: specify number of TPU cores (Kaggle v5e has 8)
    nprocs = 8
    if is_interactive:
        xmp.spawn(_mp_fn, args=(config,), nprocs=nprocs, start_method='fork')
    else:
        xmp.spawn(_mp_fn, args=(config,), nprocs=nprocs)


if __name__ == "__main__":
    launch()