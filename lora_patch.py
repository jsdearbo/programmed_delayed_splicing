import math
import torch
import torch.nn as nn
import torch.optim as optim
from peft import get_peft_model, LoraConfig, PeftModel
import re
import types
from copy import deepcopy
from typing import Callable, Optional
import pandas as pd
import logging

from grelu.lightning import LightningModel
from grelu.model.heads import ConvHead
from grelu.resources import load_model


def _unwrap_borzoi_model(model):
    """Return the underlying Borzoi model even if wrapped by PEFT or Lightning."""
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        return model.base_model.model
    if hasattr(model, "model"):
        return model.model
    return model

# ==========================================
# 1. Centralized Head Definitions
# ==========================================

class NonlinearHead(nn.Module):
    def __init__(self, in_ch=1920, hidden=512, out_ch=3):
        super().__init__()
        self.in_channels = in_ch
        self.n_tasks = out_ch
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden, out_ch, kernel_size=1),
        )

    def forward(self, x):
        return self.net(x)

class SplitHead(nn.Module):
    """
    Split head with shared nonlinear trunk + task-specific projection networks.
    Each task has its own normalization and nonlinear processing before final output.
    This architectural constraint forces tasks to learn different representations.
    
    Args:
        in_ch: Input channels (1920 for Borzoi)
        hidden: Shared trunk hidden dimension
        task_hidden: Task-specific projection hidden dimension
        out_ch: Number of tasks (output channels)
    """
    def __init__(self, in_ch=1920, hidden=512, task_hidden=256, out_ch=3):
        super().__init__()
        self.in_channels = in_ch
        self.n_tasks = out_ch
        
        # Shared nonlinear trunk
        self.shared = nn.Sequential(
            nn.Conv1d(in_ch, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=1),
            nn.GELU(),
        )
        
        # Task-specific projection networks (not just linear layers!)
        # Each task has its own BatchNorm to prevent internal collapse
        self.task_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(hidden, task_hidden, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm1d(task_hidden),  # Task-specific normalization (works with Conv1d)
                nn.Conv1d(task_hidden, 1, kernel_size=1),
            ) for _ in range(out_ch)
        ])

    def forward(self, x):
        shared = self.shared(x)
        # Apply each task-specific head to the shared representation -> (B, 1, L)
        outs = [head(shared) for head in self.task_heads]
        # Concatenate along channel dim: (B, out_ch, L)
        return torch.cat(outs, dim=1)

class ConditionalHead(nn.Module):
    """
    Conditional head: features (B, C, L) + cell_type_id (B,) -> (B, 1, L)
    """
    def __init__(self, in_ch=1920, hidden=512, cond_dim=32, n_celltypes=3, out_ch=1):
        super().__init__()
        self.in_channels = in_ch
        self.cell_embed = nn.Embedding(n_celltypes, cond_dim)
        
        # We need to reshape embedding to (B, cond_dim, 1) to broadcast over L
        # Input to conv will be in_ch + cond_dim
        self.net = nn.Sequential(
            nn.Conv1d(in_ch + cond_dim, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden, out_ch, kernel_size=1),
        )

    def forward(self, feats, cell_type_id):
        # feats: (B, C, L)
        # cell_type_id: (B,) long or (B,1)
        if cell_type_id.ndim == 0:
            cell_type_id = cell_type_id.unsqueeze(0)
        
        # Validate batch size consistency if possible
        if feats.shape[0] != cell_type_id.shape[0]:
            # Handle potential shape mismatch if one is scalar/broadcastable handled by caller
            pass

        cond = self.cell_embed(cell_type_id)          # (B, cond_dim)
        if cond.ndim == 2:
            cond = cond.unsqueeze(-1)                # (B, cond_dim, 1)
        
        # Broadcast conditioning over length dimension
        cond = cond.expand(-1, -1, feats.shape[-1])   # (B, cond_dim, L)
        
        # Concatenate
        x = torch.cat([feats, cond], dim=1)           # (B, C+cond_dim, L)
        return self.net(x)

def patch_model_forward_for_conditional_head(model_wrapper):
    """
    Patch model_wrapper.model.forward so it can accept:
      - x (Tensor): original behavior (mostly not used in cond training, but good for safety)
      - (x, cell_type_id): conditional behavior
    """
    base_model = _unwrap_borzoi_model(model_wrapper.model)
    # Avoid double patching
    if hasattr(base_model, "_conditional_forward_patched") and base_model._conditional_forward_patched:
        return

    orig_forward = base_model.forward

    # We need to bind this method to the instance
    def new_forward(self, x, *args, **kwargs):
        # Conditional route
        if isinstance(x, (tuple, list)) and len(x) == 2:
            seq, cell_type_id = x
            
            # Locate the trunk (grelu usually calls it 'embedding', borzoi might too)
            trunk = getattr(self, 'trunk', getattr(self, 'embedding', None))
            
            if trunk is not None and hasattr(self, 'head'):
                params_require_grad = any(param.requires_grad for param in trunk.parameters())
                with torch.set_grad_enabled(params_require_grad):
                    feats = trunk(seq)                    # (B, 1920, L)
                if not params_require_grad:
                    feats = feats.detach()

                # Verify head type before call
                if not isinstance(self.head, ConditionalHead):
                    if hasattr(self, 'conditional_head_ref'):
                        self.head = self.conditional_head_ref
                    else:
                        raise RuntimeError("Conditional head missing; ensure apply_custom_head ran")

                out = self.head(feats, cell_type_id)  # ConditionalHead expects (feats, cell_type_id)
                return out
            else:
                 # Fallback if structure is different?
                 # Assuming BorzoiModel structure from Grelu
                 pass

        # Default route
        return orig_forward(x, *args, **kwargs)

    base_model.forward = types.MethodType(new_forward, base_model)
    base_model._conditional_forward_patched = True

def patch_lightning_format_input(model_wrapper):
    """
    Patch LightningModel.format_input to allow tuples (x, t) to pass through.
    ALSO patch the __call__ method to ensure tuples reach the inner model unchanged.
    """
    # Patch format_input
    orig_format = model_wrapper.format_input

    def new_format_input(self, x):
        # Pass through tuple if len 2 (assume x, t)
        if isinstance(x, (tuple, list)) and len(x) == 2:
            return x
        return orig_format(x)
    
    model_wrapper.format_input = types.MethodType(new_format_input, model_wrapper)
    
    # ALSO override the forward method directly (not __call__) to handle conditional inputs
    # This is more reliable than patching __call__ which can be overridden by PyTorch internals
    orig_forward = model_wrapper.forward

    def new_forward(self, x, *args, **kwargs):
        # If x is a conditional tuple, route to the underlying Borzoi forward
        if isinstance(x, (tuple, list)) and len(x) == 2:
            base_model = _unwrap_borzoi_model(self.model)
            return base_model.forward(x, *args, **kwargs)
        # Otherwise use the original LightningModel forward
        return orig_forward(x, *args, **kwargs)

    model_wrapper.forward = types.MethodType(new_forward, model_wrapper)
    
def apply_custom_head(model_wrapper, n_tasks, model_params):
    """
    Applies the custom head logic (Nonlinear or Split) to a Grelu/Borzoi model.
    Used by both training and loading scripts to ensure consistency.
    """
    # 1. Standard Grelu head change (sets up internal state, creates a temporary linear head)
    # If conditional, we fundamentally change the task interface to 1 output at a time.
    is_conditional = model_params.get("conditional_head", False)
    logging.info(f"apply_custom_head: is_conditional={is_conditional}, n_tasks={n_tasks}")
    effective_n_tasks = 1 if is_conditional else n_tasks

    model_wrapper.change_head(
        n_tasks=effective_n_tasks,
        final_pool_func="identity", 
        final_act_func="linear"     
    )

    # 2. If config requests a custom structure, overwrite the head
    if model_params.get("nonlinear_head", False) or is_conditional:
        in_ch = 1920 # Borzoi standard
        hidden_dim = model_params.get("head_hidden", 512)
        
        if is_conditional:
             cond_dim = model_params.get("cond_dim", 32)
             # n_celltypes defaults to original n_tasks if not specified
             # Note: model_params might have 'n_celltypes' injected by main script
             n_cts = model_params.get("n_celltypes", n_tasks) 
             # If n_tasks was passed as 1 (due to conditional) but we need the real count:
             if n_cts == 1 and n_tasks > 1: n_cts = n_tasks 
             
             logging.info(f"[Head] Replacing with ConditionalHead (Hidden={hidden_dim}, Cond={cond_dim}, Types={n_cts})")
             custom_head = ConditionalHead(in_ch=in_ch, hidden=hidden_dim, cond_dim=cond_dim, n_celltypes=n_cts, out_ch=1)
             
             # Patch model forward
             patch_model_forward_for_conditional_head(model_wrapper)
             # New: Patch format_input to allow tuples
             patch_lightning_format_input(model_wrapper)

        elif model_params.get("split_head", False):
            task_hidden = model_params.get("task_hidden", 256)
            logging.info(f"[Head] Replacing with SplitHead (Hidden={hidden_dim}, TaskHidden={task_hidden}, Tasks={n_tasks})")
            custom_head = SplitHead(in_ch=in_ch, hidden=hidden_dim, task_hidden=task_hidden, out_ch=n_tasks)
        else:
            logging.info(f"[Head] Replacing with NonlinearHead (Hidden={hidden_dim}, Tasks={n_tasks})")
            custom_head = NonlinearHead(in_ch=in_ch, hidden=hidden_dim, out_ch=n_tasks)
        
        # Match device/dtype
        try:
            ref_param = next(model_wrapper.model.parameters())
            custom_head.to(device=ref_param.device, dtype=ref_param.dtype)
        except StopIteration:
            pass
            
        # Surgical Replacement
        model_wrapper.model.head = custom_head
        # Save reference for potential recovery from overwrite by grelu's change_head
        model_wrapper.model._custom_head_ref = custom_head
        if is_conditional:
            model_wrapper.model.conditional_head_ref = custom_head
    else:
        # Standard Linear Head checks
        # Verify pooling is identity if Grelu defaulted to Adaptive
        if hasattr(model_wrapper.model.head, 'pool'):
            current_pool = model_wrapper.model.head.pool
            if "Adaptive" in str(current_pool) or isinstance(current_pool, (nn.AdaptiveAvgPool1d, nn.AdaptiveMaxPool1d)):
                logging.info("[Head] Force-replacing AdaptivePool with Identity")
                if hasattr(model_wrapper.model.head.pool, 'layer'):
                     model_wrapper.model.head.pool.layer = nn.Identity()
                else:
                     model_wrapper.model.head.pool = nn.Identity()


# ==========================================
# 2. Locon Adapter (Low-Rank Conv) + LoRA / Optimizer Helpers
# ==========================================

class PyTorchLocon(nn.Module):
    """
    Low-rank adapter for Conv1d (Locon), matching Yuan et al. Genome Biology 2026:

        h = W₀ ⊗ x + (α/r) · lora_b ⊗ (lora_a ⊗ x)

    lora_a: same kernel_size/stride/dilation as the original conv  → captures local structure
    lora_b: pointwise (kernel=1) conv                               → projects back to out_channels

    Original weights are frozen; lora_a is He-initialized, lora_b is zero-initialized
    so the adapter starts as an exact identity (zero perturbation).
    """

    def __init__(self, original_layer: nn.Conv1d, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        if original_layer.groups != 1:
            raise ValueError(
                f"PyTorchLocon only supports standard (groups=1) convolutions; "
                f"got groups={original_layer.groups}"
            )
        self.original_layer = original_layer
        for p in self.original_layer.parameters():
            p.requires_grad = False

        in_ch  = original_layer.in_channels
        out_ch = original_layer.out_channels
        ksize  = original_layer.kernel_size[0]
        stride = original_layer.stride[0]
        pad    = original_layer.padding        # "same" or int tuple — passed through as-is
        dil    = original_layer.dilation[0]

        self.scale = alpha / rank

        # A: captures receptive field (same k, stride, dilation as original)
        self.lora_a = nn.Conv1d(in_ch, rank, kernel_size=ksize,
                                stride=stride, padding=pad,
                                dilation=dil, bias=False)
        # B: pointwise projection back to output channels
        self.lora_b = nn.Conv1d(rank, out_ch, kernel_size=1, bias=False)

        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.original_layer(x) + self.scale * self.lora_b(self.lora_a(x))


def _apply_locon_to_conv_blocks(model_wrapper, conv_select: int,
                                rank: int = 4, alpha: float = 1.0):
    """
    Replace the Conv1d in the last `conv_select` blocks of the Borzoi conv tower
    with PyTorchLocon adapters (original weights frozen; lora_a + lora_b trainable).

    Navigates through Lightning and PEFT wrappers automatically, so it can be called
    on the raw LightningModel at any wrapping state.
    """
    # Peel through Lightning → PEFT → BorzoiModel
    inner = model_wrapper
    if hasattr(inner, "model"):
        inner = inner.model                          # LightningModel → BorzoiModel or PeftModel
    if hasattr(inner, "base_model") and hasattr(inner.base_model, "model"):
        inner = inner.base_model.model               # PeftModel → BorzoiModel

    trunk = getattr(inner, "trunk", getattr(inner, "embedding", None))
    if trunk is None or not hasattr(trunk, "conv_tower"):
        logging.warning("[Locon] Could not locate conv_tower; skipping.")
        return

    # All blocks that have a direct .conv (Stem at [0] + ConvBlocks at [1+])
    conv_block_list = [(i, b) for i, b in enumerate(trunk.conv_tower.blocks)
                       if hasattr(b, "conv")]
    if conv_select > len(conv_block_list):
        logging.warning(f"[Locon] conv_select={conv_select} exceeds available blocks "
                        f"({len(conv_block_list)}); clamping.")
        conv_select = len(conv_block_list)

    selected    = conv_block_list[-conv_select:]
    ref         = next(trunk.conv_tower.parameters())   # device/dtype reference
    total_new   = 0

    for idx, block in selected:
        if isinstance(block.conv, PyTorchLocon):
            logging.debug(f"[Locon] blocks[{idx}].conv already wrapped; skipping.")
            continue
        if not isinstance(block.conv, nn.Conv1d):
            logging.warning(f"[Locon] blocks[{idx}].conv is {type(block.conv).__name__}; "
                            "expected Conv1d — skipping.")
            continue
        orig  = block.conv
        locon = PyTorchLocon(orig, rank=rank, alpha=alpha)
        locon.to(device=ref.device, dtype=ref.dtype)
        block.conv = locon
        n          = locon.lora_a.weight.numel() + locon.lora_b.weight.numel()
        total_new += n
        logging.info(f"  [Locon] blocks[{idx}].conv  "
                     f"{orig.in_channels}→{orig.out_channels}  k={orig.kernel_size[0]}"
                     f"  | +{n:,} params")

    logging.info(f"[Locon] {len(selected)} blocks wrapped, "
                 f"{total_new:,} new trainable params  "
                 f"(rank={rank}, α={alpha}, scale={alpha/rank:.3f})")


# --- New helper method (from PEFT docs) ---
def print_trainable_parameters(self):
    """
    Prints the number of trainable parameters in the model.
    (Bound to LightningModel instance)
    """
    trainable_params = 0
    all_param = 0
    # Check if the model is a PEFT model
    if hasattr(self, 'model') and hasattr(self.model, 'print_trainable_parameters'):
        self.model.print_trainable_parameters()
    else:
        # Fallback for non-PEFT models or manual check
        for _, param in self.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        logging.info(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
        )

def configure_optimizers_lora(self):
    """LoRA-aware implementation of configure_optimizers."""
    if getattr(self, "_is_lora_tuning", False):
        logging.info("Configuring optimizer for LoRA tuning...")
        base_lr = float(self.train_params.get("lr", 2e-4))
        head_lr = float(self.train_params.get("head_lr", 4e-4))
        weight_decay = float(self.train_params.get("weight_decay", 1e-6))
        beta1 = float(self.train_params.get("adam_beta1", 0.9))
        beta2 = float(self.train_params.get("adam_beta2", 0.999))
        warmup_steps = int(self.train_params.get("warmup_steps", 0))
        plateau_patience = int(self.train_params.get("lr_plateau_patience", 3))
        
        decay, no_decay, head_params, conv_params = [], [], [], []

        for name, param in self.model.named_parameters():
            if not param.requires_grad: continue

            # 'head' is the standard name.'task_heads' is our new Multitask container name. Both contain "head", so this condition works for both.
            if "head" in name:
                if "original_module" in name:
                    continue
                head_params.append(param)
            # Conv tower blocks unfrozen via conv_select (full fine-tuning, not LoRA)
            elif "conv_tower" in name:
                conv_params.append(param)
            # Biases and LayerNorm weights shouldn't have weight decay
            elif len(param.shape) == 1 or "norm" in name.lower():
                no_decay.append(param)
            # All other trainable params (i.e., LoRA weights)
            else:
                decay.append(param)

        param_groups = [
            {'params': head_params, 'weight_decay': weight_decay, 'lr': head_lr},
            {'params': conv_params, 'weight_decay': weight_decay, 'lr': base_lr},
            {'params': no_decay, 'weight_decay': 0.0, 'lr': base_lr},
            {'params': decay, 'weight_decay': weight_decay, 'lr': base_lr},
        ]
        # Drop empty groups to avoid AdamW warnings
        param_groups = [g for g in param_groups if g['params']]

        # Utility to count parameters in each group for logging
        def count_params(param_list):
            return sum(p.numel() for p in param_list)

        logging.info(f"Optimizer groups created:")
        logging.info(f"  - Head: {count_params(head_params)} params (lr={head_lr}, wd={weight_decay})")
        logging.info(f"  - Conv (full): {count_params(conv_params)} params (lr={base_lr}, wd={weight_decay})")
        logging.info(f"  - LoRA (decay): {count_params(decay)} params (lr={base_lr}, wd={weight_decay})")
        logging.info(f"  - No Decay: {count_params(no_decay)} params (lr={base_lr}, wd=0.0)")
        
        optimizer = optim.AdamW(param_groups, lr=base_lr, betas=(beta1, beta2))

        # --- LR schedule ---
        use_cosine = self.train_params.get("cosine_decay", False)
        min_lr_ratio = float(self.train_params.get("min_lr_ratio", 0.01))

        if use_cosine:
            total_steps = self.trainer.estimated_stepping_batches
            # cosine_T_epochs: fix the cosine decay period to N epochs instead of max_epochs.
            # After N epochs the LR stays at min_lr_ratio. Use this when HPs were tuned on a
            # short Optuna run (e.g. 3 epochs) but you want to keep training longer.
            cosine_T_epochs = self.train_params.get("cosine_T_epochs", None)
            if cosine_T_epochs is not None:
                max_epochs = self.trainer.max_epochs or 1
                steps_per_epoch = total_steps / max(1, max_epochs)
                cosine_steps = warmup_steps + int(cosine_T_epochs * steps_per_epoch)
                logging.info(
                    f"LR schedule: warmup({warmup_steps}) + cosine decay over "
                    f"{cosine_T_epochs} epochs ({cosine_steps} steps), "
                    f"then fixed at min_lr_ratio={min_lr_ratio} (total budget: {total_steps} steps)"
                )
            else:
                cosine_steps = total_steps
                logging.info(f"LR schedule: warmup({warmup_steps}) + cosine decay over {total_steps} total steps (min_lr_ratio={min_lr_ratio})")

            def warmup_cosine_lambda(step):
                if step < warmup_steps:
                    return float(step + 1) / float(max(1, warmup_steps))
                progress = float(step - warmup_steps) / float(max(1, cosine_steps - warmup_steps))
                progress = min(progress, 1.0)  # clamp so LR stays at min_lr_ratio after cosine_T_epochs
                return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_cosine_lambda)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
            }

        if warmup_steps > 0:
            logging.info(f"LR schedule: warmup({warmup_steps}) only — no decay")
            def warmup_lambda(step):
                return min(1.0, float(step + 1) / float(warmup_steps))

            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
            }

        ckpt_cfg = self.train_params.get("checkpoint", {})
        plateau_monitor = ckpt_cfg.get("monitor", "val_loss")
        plateau_mode = ckpt_cfg.get("mode", "min")
        logging.info(f"LR schedule: ReduceLROnPlateau monitoring {plateau_monitor} ({plateau_mode})")
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=plateau_mode, factor=0.5, patience=plateau_patience, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": plateau_monitor, "interval": "epoch", "frequency": 1},
        }
    else:
        # --- Original configure_optimizers logic ---
        logging.info("Configuring standard optimizer...")
        if self.train_params["optimizer"] == "adam":
            return optim.Adam(self.parameters(), lr=self.train_params["lr"])
        elif self.train_params["optimizer"] == "sgd":
            return optim.SGD(
                self.parameters(), lr=self.train_params["lr"], momentum=0.9
            )
        else:
            raise Exception("Unknown optimizer")

# --- 1. Define the LoRA-aware Training Method ---
def tune_on_dataset_lora(self, train_dataset, val_dataset, checkpoint_path=None, **kwargs):
    """LoRA-aware training entry point."""
    # Mirror grelu's tune_on_dataset contract: move pre-train data_params aside
    # so train_on_dataset starts with a clean slate.
    self.base_data_params = self.data_params.copy()
    self.data_params = {}

    # 1. Apply Head Architecture
    apply_custom_head(self, train_dataset.n_tasks, self.model_params)

    # 2. Apply LoRA
    if self.model_params.get("use_lora", False):
        logging.info("--- LoRA Tuning Enabled ---")
        self._is_lora_tuning = True
        self.wrap_with_lora()

    # 2b. Apply Locon adapters to the last N conv tower blocks (Yuan et al. 2026)
    conv_select = self.model_params.get("conv_select", 0)
    if conv_select > 0:
        locon_rank  = int(self.model_params.get("locon_rank", 4))
        locon_alpha = float(self.model_params.get("locon_alpha", 1.0))
        logging.info(f"--- Locon: last {conv_select} conv blocks "
                     f"(rank={locon_rank}, alpha={locon_alpha}) ---")
        _apply_locon_to_conv_blocks(self, conv_select, rank=locon_rank, alpha=locon_alpha)

    # 3. Freeze requested layers after wrapping so LoRA weights stay trainable
    freeze_embedding = kwargs.get("freeze_embedding", False) or self.model_params.get("freeze_embedding", False)
    layers_to_freeze = kwargs.get("layers_to_freeze")
    if layers_to_freeze is None:
        layers_to_freeze = self.model_params.get("layers_to_freeze", [])

    if freeze_embedding and hasattr(self.model, "embedding"):
        for param in self.model.embedding.parameters():
            param.requires_grad = False

    if layers_to_freeze:
        logging.info(f"Freezing layers matching: {layers_to_freeze}")
        for name, param in self.model.named_parameters():
            if any(layer in name for layer in layers_to_freeze):
                # Never freeze LoRA adapter weights or head modules_to_save
                if "lora_" in name or "modules_to_save" in name:
                    continue
                param.requires_grad = False

    if checkpoint_path:
        logging.info(f"Resuming training from checkpoint: {checkpoint_path}")
    return self.train_on_dataset(train_dataset, val_dataset, checkpoint_path=checkpoint_path)

def attach_lora_methods(model: LightningModel) -> LightningModel:
    def wrap_with_lora(self):
        rank = self.model_params.get('lora_rank', 8)
        logging.info(f"[LoRA] Wrapping model with Rank {rank}")
        target_modules = self.model_params.get("lora_target_modules", ["to_q", "to_v", "to_k", "to_out"])
        
        peft_config = LoraConfig(
            task_type=None, inference_mode=False, r=rank,
            lora_alpha=self.model_params.get("lora_alpha", 16),
            lora_dropout=0.05,
            target_modules=target_modules,
            modules_to_save=["head"], 
            bias="none"
        )
        self.model = get_peft_model(self.model, peft_config)
        
        # Explicit unfreeze of head (avoid double-counting original_module)
        for name, param in self.model.named_parameters():
            if "head" in name:
                if "modules_to_save" in name:
                    param.requires_grad = True
                elif "original_module" in name:
                    param.requires_grad = False

    model.wrap_with_lora = types.MethodType(wrap_with_lora, model)
    model.tune_on_dataset = types.MethodType(tune_on_dataset_lora, model)
    model.configure_optimizers = types.MethodType(configure_optimizers_lora, model)
    return model

# ==========================================
# 3. Robust Checkpoint Loader
# ==========================================

def load_finetuned_checkpoint(checkpoint_path):
    logging.info(f"[Loader] Loading checkpoint: {checkpoint_path}")
    
    # --- Fix for unpickling bound methods injected during training ---
    def _dummy_step(self, *args, **kwargs):
        pass
    for m in ['custom_training_step', 'custom_validation_step', 'configure_optimizers_lora', 
              'tune_on_dataset_lora', 'wrap_with_lora', 'tune_on_dataset']:
        if not hasattr(LightningModel, m):
            setattr(LightningModel, m, _dummy_step)
    # -----------------------------------------------------------------
    
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    hparams = ckpt["hyper_parameters"]
    
    # Extract params
    raw_model_params = hparams.get("model_params", {})
    raw_train_params = hparams.get("train_params", {})
    raw_data_params  = hparams.get("data_params",  {})
    model_params = deepcopy(raw_model_params)
    data_params  = deepcopy(raw_data_params) if raw_data_params else {}

    train_params = {}
    for k, v in raw_train_params.items():
        if k in ["logger", "callbacks", "trainer", "profiler", "strategy"]:
            train_params[k] = None
        else:
            try:
                train_params[k] = deepcopy(v)
            except Exception as e:
                logging.debug(f"[Loader] Skipping deepcopy for train_params['{k}'] due to error: {e}")
                train_params[k] = v

    # 1. Load Base Model
    model = load_model(project="borzoi", model_name="mouse_rep0")
    model = attach_lora_methods(model)
    model.model_params.update(model_params)
    model.train_params.update(train_params)
    # Restore data_params (task metadata) from checkpoint so that load_tasks()
    # returns the fine-tuned task list rather than the pretrained Borzoi tracks.
    if data_params:
        model.data_params.update(data_params)

    # 2. Detect if checkpoint has LoRA weights and whether it was saved from a wrapped model
    sd = ckpt["state_dict"]
    checkpoint_has_lora = any("lora" in k.lower() for k in sd.keys())
    checkpoint_is_wrapped = any(k.startswith("model.base_model.") for k in sd.keys())
    
    # 3. Reconstruct Head Architecture FIRST (before LoRA wrapping)
    n_tasks = model_params.get("n_tasks")
    if not n_tasks:
        # Fallback if n_tasks wasn't saved, though Grelu usually saves it
        logging.warning("[Loader] Warning: n_tasks not found in hparams, defaulting to config guess")
        n_tasks = 3 

    apply_custom_head(model, n_tasks, model_params)
    
    # 4. LOAD CHECKPOINT with the correct structure.
    # If the checkpoint keys are wrapped (model.base_model.*), wrap first.
    # If not, load first, then wrap.
    if checkpoint_has_lora and checkpoint_is_wrapped:
        logging.info("[Loader] Checkpoint uses wrapped key structure. Wrapping before load...")
        model.wrap_with_lora()
        
        # PEFT's modules_to_save creates two sub-copies of the head:
        #   head.original_module.*  (frozen copy)
        #   head.modules_to_save.default.*  (trainable copy)
        # But Lightning may save the checkpoint with bare head keys
        # (e.g. model.base_model.model.head.net.0.weight) which don't match
        # either sub-copy. Detect this and remap so both copies get loaded.
        head_prefix = "model.base_model.model.head."
        bare_head_keys = [
            k for k in sd
            if k.startswith(head_prefix)
            and "modules_to_save" not in k
            and "original_module" not in k
            and "conditional_head_ref" not in k
        ]
        if bare_head_keys:
            logging.info(f"[Loader] Remapping {len(bare_head_keys)} bare head keys to modules_to_save + original_module")
            for k in bare_head_keys:
                suffix = k[len(head_prefix):]
                sd[head_prefix + "modules_to_save.default." + suffix] = sd[k]
                sd[head_prefix + "original_module." + suffix] = sd[k]
                del sd[k]

    # 4b. Apply Locon adapters if the checkpoint was trained with them.
    # Must happen BEFORE loading state_dict so the model structure matches the saved keys.
    checkpoint_has_locon = any(".conv.lora_a." in k or ".conv.lora_b." in k for k in sd.keys())
    if checkpoint_has_locon:
        _conv_select = model_params.get("conv_select", 0)
        _locon_rank  = int(model_params.get("locon_rank", 4))
        _locon_alpha = float(model_params.get("locon_alpha", 1.0))
        if _conv_select > 0:
            logging.info(f"[Loader] Re-applying Locon (conv_select={_conv_select}, "
                         f"rank={_locon_rank}, alpha={_locon_alpha}) for checkpoint loading...")
            _apply_locon_to_conv_blocks(model, _conv_select,
                                        rank=_locon_rank, alpha=_locon_alpha)
        else:
            logging.warning("[Loader] Checkpoint has Locon weights but conv_select=0 "
                            "in model_params — conv keys will be missing!")

    logging.info("[Loader] Loading checkpoint weights...")
    missing, unexpected = model.load_state_dict(sd, strict=False)

    # 5. Verify loaded weights
    head_missing = [k for k in missing if "head" in k]
    other_missing = [k for k in missing if "head" not in k and "model." in k]
    
    if head_missing:
        logging.error(f"[Loader] ERROR: Failed to load {len(head_missing)} head weights!")
        logging.error(f"[Loader] Missing head keys: {head_missing[:5]} ...")
        logging.error("[Loader] This means the head is using RANDOM WEIGHTS!")
    elif missing:
        logging.info(f"[Loader] Note: {len(missing)} non-head keys missing (may be expected)")
        if other_missing:
            logging.debug(f"[Loader] Other missing keys: {other_missing[:3]} ...")
            
    if unexpected:
        logging.warning(f"[Loader] WARNING: {len(unexpected)} unexpected keys")
        logging.debug(f"[Loader] Unexpected keys: {unexpected[:5]} ...")
    
    if not head_missing:
        logging.info("[Loader] \033[92m✓ Head weights loaded successfully from checkpoint!\033[0m")
        # Verify weights are not random by checking some statistics
        base_model = _unwrap_borzoi_model(model)
        head = getattr(base_model, "head", None)
        if head is not None:
            total_params = sum(p.numel() for p in head.parameters())
            mean_abs_val = sum(p.abs().mean().item() * p.numel() for p in head.parameters()) / total_params if total_params > 0 else 0
            logging.info(f"[Loader] Head has {total_params:,} parameters, mean |weight| = {mean_abs_val:.6f}")
            if mean_abs_val < 1e-6:
                logging.warning("[Loader] WARNING: Head weights appear to be near zero!")
    logging.info("[Loader] \033[92mLoaded checkpoint successfully.\033[0m")
    
    # 6. If checkpoint had LoRA but was saved unwrapped, wrap after load
    if checkpoint_has_lora and not checkpoint_is_wrapped:
        logging.info("[Loader] Detected LoRA weights in checkpoint. Wrapping model with LoRA...")
        model.wrap_with_lora()
        logging.info("[Loader] \033[92mModel wrapped with LoRA. Loaded weights preserved in original_module.\033[0m")

    # 7. Re-patch the forward method if conditional head is detected
    # This is necessary because the forward patching happens during training setup,
    # but when loading from checkpoint, we need to re-apply it for inference
    # ALSO: After LoRA wrapping, we need to patch at MULTIPLE LEVELS:
    # - The base model (for internal structure)
    # - The PEFT wrapper (model.model)
    # - The LightningModel (model) since that's what gets called as ft_model()
    if model_params.get("conditional_head", False):
        logging.info("[Loader] Re-patching model forward for conditional head inference...")
        
        # Patch the base model first (preserves original structure)
        patch_model_forward_for_conditional_head(model)
        patch_lightning_format_input(model)
        
        # If model is a PEFT model, we also need to patch the top-level forward
        # to handle conditional tuple inputs before PEFT's forward logic runs
        if checkpoint_has_lora:
            logging.info("[Loader] Patching PEFT model forward for conditional inputs...")
            # model.model is the wrapped PeftModel (after wrap_with_lora())
            peft_model = model.model
            orig_peft_forward = peft_model.forward
            
            # Create a method that properly receives self and dispatches tuples
            def peft_conditional_forward(self, x, *args, **kwargs):
                # If x is a tuple (x, cell_type_id), route through the patched base model forward
                if isinstance(x, (tuple, list)) and len(x) == 2:
                    base_model = _unwrap_borzoi_model(self)
                    return base_model.forward(x, *args, **kwargs)
                # Otherwise use normal PEFT forward
                return orig_peft_forward(x, *args, **kwargs)
            
            # Bind as a method properly  
            peft_model.forward = types.MethodType(peft_conditional_forward, peft_model)

    # Replace data_params["tasks"] with the fine-tuned task list.
    # After loading, data_params["tasks"] still contains all base Borzoi tracks
    # (because change_head never updates it).  Downstream code (load_tasks, Aggregate)
    # uses task names to select output channels, so the mismatch between thousands
    # of Borzoi track names and the 1-output fine-tuned head causes CUDA index OOB.
    idx_to_celltype = model_params.get("idx_to_celltype", {})
    if idx_to_celltype:
        task_names = [idx_to_celltype[i] for i in sorted(idx_to_celltype)]
    else:
        n_out = model_params.get("n_tasks", 1)
        task_names = [f"task_{i}" for i in range(n_out)]
    model.data_params["tasks"] = {"name": task_names}
    logging.info(f"[Loader] data_params['tasks'] set to fine-tuned outputs: {task_names}")

    model.eval()
    return model