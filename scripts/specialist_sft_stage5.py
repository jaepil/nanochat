"""
Specialist SFT with PoE Stage 5 addition (Section 8.8 Elastic Depth).

Adds new layers to the pretrained PoE model and fine-tunes only the new stage
on specialist task data (chat, math, code, tool-calling, summarization, NER, ...).
Existing stages are frozen.

Usage:
    python -m scripts.specialist_sft_stage5
    torchrun --nproc_per_node=8 -m scripts.specialist_sft_stage5
"""

import os
import gc
import json
import time
import math
import random
import argparse

import torch
import torch.nn.functional as F
import torch.distributed as dist

from nanochat.gpt import GPT, GPTConfig, Block, has_ve
from nanochat.common import (compute_init, compute_cleanup, print0, DummyWandb,
                             print_banner, get_base_dir, autodetect_device_type,
                             COMPUTE_DTYPE, is_ddp_initialized)
from nanochat.tokenizer import get_tokenizer
from nanochat.checkpoint_manager import build_model, save_checkpoint, save_stage_delta, find_last_step

print_banner()

# CLI
parser = argparse.ArgumentParser(description="Chat SFT with PoE Stage 5")
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps")
parser.add_argument("--run", type=str, default="dummy", help="wandb run name")
# Model
parser.add_argument("--model-tag", type=str, default="poe_d24_r10")
parser.add_argument("--model-step", type=int, default=None)
parser.add_argument("--new-layers", type=int, default=6, help="Number of new stage layers")
parser.add_argument("--freeze-lm-head", action="store_true")
parser.add_argument("--dual-head", action="store_true",
                    help="Paper §6.5: attach a zero-init specialist lm_head_stage, "
                         "train it jointly with the new stage while the base lm_head stays "
                         "frozen. Implies --freeze-lm-head.")
parser.add_argument("--lm-head-stage-lr", type=float, default=None,
                    help="LR for lm_head_stage (dual-head only). Default: 0.25 × unembedding_lr (paper §6.5.2).")
parser.add_argument("--lm-head-stage-weight-decay", type=float, default=0.1,
                    help="Weight decay for lm_head_stage (paper §6.5.2 used 0.1).")
parser.add_argument("--case-augment", action="store_true",
                    help="Data augmentation: duplicate each conversation with the first "
                         "user turn's leading greeting case-perturbed. Helps mitigate "
                         "'Hi' vs 'hi' different-response issue on small models.")
parser.add_argument("--resume-from", type=int, default=None, help="Resume SFT from this step (loads SFT checkpoint instead of base model)")
# Data
parser.add_argument("--smoltalk-epochs", type=int, default=1)
parser.add_argument("--mmlu-epochs", type=int, default=3)
parser.add_argument("--gsm8k-epochs", type=int, default=4)
parser.add_argument("--code-alpaca-epochs", type=int, default=0)
parser.add_argument("--math-instruct-epochs", type=int, default=0)
parser.add_argument("--magicoder-epochs", type=int, default=0)
parser.add_argument("--glaive-fc-epochs", type=int, default=0)
parser.add_argument("--xlam-fc-epochs", type=int, default=0)
parser.add_argument("--xsum-epochs", type=int, default=0)
parser.add_argument("--cnndm-epochs", type=int, default=0)
parser.add_argument("--samsum-epochs", type=int, default=0)
parser.add_argument("--xlsum-en-epochs", type=int, default=0)
parser.add_argument("--conll-epochs", type=int, default=0)
parser.add_argument("--fewnerd-epochs", type=int, default=0)
parser.add_argument("--wikiann-epochs", type=int, default=0)
# Chain: optionally layer this stage on top of an existing SFT stage delta
# (instead of directly on the frozen base). The parent's lm_head_stage is
# folded into lm_head in-memory so downstream specialists compose additively.
parser.add_argument("--parent-stage-tag", type=str, default=None,
                    help="If set, load stage delta from chatsft_checkpoints/<tag>/ and stack on top of base")
parser.add_argument("--parent-stage-step", type=int, default=None,
                    help="Which step of the parent stage to load (default: latest)")
# Training
parser.add_argument("--device-batch-size", type=int, default=16)
parser.add_argument("--max-seq-len", type=int, default=2048)
parser.add_argument("--total-batch-size", type=int, default=-1)
parser.add_argument("--num-iterations", type=int, default=-1, help="-1 = one epoch")
parser.add_argument("--embedding-lr", type=float, default=None)
parser.add_argument("--unembedding-lr", type=float, default=None)
parser.add_argument("--matrix-lr", type=float, default=None)
parser.add_argument("--scalar-lr", type=float, default=None)
parser.add_argument("--init-lr-frac", type=float, default=0.2)
parser.add_argument("--warmup-ratio", type=float, default=0.05)
parser.add_argument("--warmdown-ratio", type=float, default=0.5)
parser.add_argument("--final-lr-frac", type=float, default=0.0)
# Eval & save
parser.add_argument("--eval-every", type=int, default=200)
parser.add_argument("--eval-tokens", type=int, default=40*524288)
parser.add_argument("--save-every", type=int, default=-1)
parser.add_argument("--output-tag", type=str, default=None)
parser.add_argument("--max-convs", type=int, default=-1,
                    help="Subsample training conversations to at most N (seed=42, deterministic). -1 = use all.")
parser.add_argument("--val-convs", type=int, default=512,
                    help="Held-out validation conversation count. Taken from shuffled all_conversations.")
args = parser.parse_args()

# Compute init
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None

# Wandb
use_dummy = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy else __import__("wandb").init(
    project="nanochat", name=args.run, config=vars(args))

# Load tokenizer
tokenizer = get_tokenizer()
vocab_size = tokenizer.get_vocab_size()
print0(f"Vocab size: {vocab_size}")

# ---------------------------------------------------------------------------
# Load model (fresh start or resume)
# ---------------------------------------------------------------------------
base_dir = get_base_dir()
new_layers = args.new_layers
resume_step = args.resume_from or 0

if resume_step > 0:
    # Resume from SFT checkpoint
    sft_dir = os.path.join(base_dir, "chatsft_checkpoints",
                           args.output_tag or f"sft_stage5_{args.model_tag}")
    print0(f"Resuming from SFT checkpoint: step {resume_step}")
    with open(os.path.join(sft_dir, f"meta_{resume_step:06d}.json")) as f:
        meta = json.load(f)
    mc = meta["model_config"]
    data = torch.load(os.path.join(sft_dir, f"model_{resume_step:06d}.pt"),
                       map_location=device)
    data = {k.removeprefix("_orig_mod."): v for k, v in data.items()}

    # Build model matching saved VE indices
    saved_ve = {int(k.split(".")[1]) for k in data if k.startswith("value_embeds.")}
    config = GPTConfig(**mc)
    orig_has_ve = has_ve
    import nanochat.gpt as _gpt_mod
    _gpt_mod.has_ve = lambda i, n: i in saved_ve
    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device=device)
    model.init_weights()
    model.load_state_dict(data, strict=True, assign=True)
    _gpt_mod.has_ve = orig_has_ve
    del data

    new_config = config
    new_n_layer = config.n_layer
    old_n_layer = new_n_layer - new_layers
    original_backout_layer = old_n_layer // 2
    ve_indices = sorted(saved_ve)
    print0(f"  Resumed d{new_n_layer}, frozen={old_n_layer}")
else:
    # Fresh start: load base model and expand
    checkpoint_dir = os.path.join(base_dir, "base_checkpoints", args.model_tag)
    step = args.model_step if args.model_step else find_last_step(checkpoint_dir)
    print0(f"Loading pretrained model: {args.model_tag} step {step}")
    model, _, meta = build_model(checkpoint_dir, step, device, phase="train")
    old_config = model.config
    old_n_layer = old_config.n_layer
    print0(f"  d{old_n_layer}, n_embd={old_config.n_embd}")

    # Chain: if a parent SFT stage is specified, expand base to parent's n_layer
    # and apply the parent delta (new transformer layers + lm_head_stage + wte).
    # The parent's lm_head_stage is then folded into lm_head in-memory so that
    # the new specialist stage's head composes additively on top.
    if args.parent_stage_tag:
        parent_stage_dir = os.path.join(base_dir, "chatsft_checkpoints", args.parent_stage_tag)
        parent_step = args.parent_stage_step if args.parent_stage_step else find_last_step(parent_stage_dir)
        parent_delta_path = os.path.join(parent_stage_dir, f"model_{parent_step:06d}.pt")
        parent_meta_path = os.path.join(parent_stage_dir, f"meta_{parent_step:06d}.json")
        with open(parent_meta_path, "r") as _pf:
            parent_meta = json.load(_pf)
        parent_n_layer = parent_meta["model_config"]["n_layer"]
        parent_dual_head = bool(parent_meta.get("dual_head", False))
        parent_new_layers = parent_n_layer - old_n_layer
        print0(f"Chain: stacking on parent stage {args.parent_stage_tag} step {parent_step} "
               f"(d{old_n_layer}->d{parent_n_layer}, dual_head={parent_dual_head})")

        # Build intermediate config for d{parent_n_layer} and add the stage
        # layers the parent trained. Initialization mirrors lines 166-177 below.
        _inter_config = GPTConfig(
            sequence_len=old_config.sequence_len,
            vocab_size=old_config.vocab_size,
            n_layer=parent_n_layer,
            n_head=old_config.n_head,
            n_kv_head=old_config.n_kv_head,
            n_embd=old_config.n_embd,
            window_pattern=old_config.window_pattern,
        )
        _n_embd = old_config.n_embd
        _s = 3**0.5 * _n_embd**-0.5
        for i in range(old_n_layer, parent_n_layer):
            block = Block(_inter_config, i).to(device)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
            torch.nn.init.uniform_(block.attn.c_q.weight, -_s, _s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -_s, _s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -_s, _s)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -_s * 0.4, _s * 0.4)
            if hasattr(block.attn, 've_gate') and block.attn.ve_gate is not None:
                block.attn.ve_gate = None
            model.transformer.h.append(block)
        with torch.no_grad():
            _new_resid = torch.cat([model.resid_lambdas.data,
                                    torch.ones(parent_new_layers, device=device)])
            model.resid_lambdas = torch.nn.Parameter(_new_resid)
            _new_x0 = torch.cat([model.x0_lambdas.data,
                                 torch.zeros(parent_new_layers, device=device)])
            model.x0_lambdas = torch.nn.Parameter(_new_x0)
        model.config = _inter_config
        model.window_sizes = model._compute_window_sizes(_inter_config)

        # Parent may have a dual-head lm_head_stage; create a placeholder so the
        # parent delta can load it, then fold into lm_head and discard.
        if parent_dual_head:
            _padded_vocab = model.lm_head.weight.size(0)
            from nanochat.gpt import Linear as _GPTLinear
            model.lm_head_stage = _GPTLinear(_n_embd, _padded_vocab, bias=False).to(device)
            with torch.no_grad():
                model.lm_head_stage.weight.zero_()

        parent_delta = torch.load(parent_delta_path, map_location=device, weights_only=False)
        parent_delta = {k.removeprefix("_orig_mod."): v for k, v in parent_delta.items()}
        _missing, _unexpected = model.load_state_dict(parent_delta, strict=False)
        if _unexpected:
            raise RuntimeError(f"Unexpected keys in parent stage delta: {_unexpected[:5]}")
        print0(f"  Parent delta loaded ({len(parent_delta)} tensors)")

        if parent_dual_head and model.lm_head_stage is not None:
            with torch.no_grad():
                model.lm_head.weight.data.add_(model.lm_head_stage.weight.data)
            # Drop the folded head; downstream dual-head init will create a
            # fresh zero-init lm_head_stage for this stage's own specialist.
            model.lm_head_stage = None
            print0(f"  Folded parent lm_head_stage into lm_head (new lm_head ready for additional stage head)")

        old_config = _inter_config
        old_n_layer = parent_n_layer

    # Expand model with Stage 5 (Section 8.8)
    new_n_layer = old_n_layer + new_layers
    ve_indices = sorted(int(k) for k in model.value_embeds.keys())
    print0(f"  VE indices (preserved): {ve_indices}")
    # Backout layer is anchored at the ORIGINAL base's midpoint to keep the
    # paper §8.8 fixed-index backout consistent with what the parent used. When
    # chaining on top of an existing stage, recover the base's n_layer from the
    # parent's frozen_layers (e.g., parent d26 with frozen_layers=24 -> base=24,
    # backout=12). Otherwise fall back to the current old_n_layer // 2.
    if args.parent_stage_tag:
        _base_n_layer = int(parent_meta.get("frozen_layers", old_n_layer))
        original_backout_layer = _base_n_layer // 2
    else:
        original_backout_layer = old_n_layer // 2

    new_config = GPTConfig(
        sequence_len=old_config.sequence_len,
        vocab_size=old_config.vocab_size,
        n_layer=new_n_layer,
        n_head=old_config.n_head,
        n_kv_head=old_config.n_kv_head,
        n_embd=old_config.n_embd,
        window_pattern=old_config.window_pattern,
    )

    n_embd = old_config.n_embd
    s = 3**0.5 * n_embd**-0.5
    for i in range(old_n_layer, new_n_layer):
        block = Block(new_config, i).to(device)
        torch.nn.init.zeros_(block.attn.c_proj.weight)
        torch.nn.init.zeros_(block.mlp.c_proj.weight)
        torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
        torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
        torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
        torch.nn.init.uniform_(block.mlp.c_fc.weight, -s * 0.4, s * 0.4)
        if hasattr(block.attn, 've_gate') and block.attn.ve_gate is not None:
            block.attn.ve_gate = None
        model.transformer.h.append(block)

    with torch.no_grad():
        new_resid = torch.cat([model.resid_lambdas.data, torch.ones(new_layers, device=device)])
        model.resid_lambdas = torch.nn.Parameter(new_resid)
        new_x0 = torch.cat([model.x0_lambdas.data, torch.zeros(new_layers, device=device)])
        model.x0_lambdas = torch.nn.Parameter(new_x0)

    model.config = new_config
    model.window_sizes = model._compute_window_sizes(new_config)
    print0(f"Expanded: d{new_n_layer} ({new_layers} new layers, frozen={old_n_layer})")
# Dual-head extension (paper §6.5): attach zero-init specialist head on top of
# the frozen base lm_head. The specialist composes additively with base at the
# final layer only; base head stays frozen so factual projections are preserved
# bit-identically (Washington Δlogit = 0.0000 across the specialist run).
if args.dual_head:
    if not args.freeze_lm_head:
        args.freeze_lm_head = True
        print0("  --dual-head -> auto-enabling --freeze-lm-head")
    # Mark model config so save_stage_delta and forward can detect dual-head.
    new_config.dual_head = True
    new_config.frozen_layers = old_n_layer
    model.config = new_config
    # Allocate specialist head with same shape as lm_head; zero-init so step 0
    # reproduces the head-frozen baseline exactly.
    padded_vocab_size = model.lm_head.weight.size(0)
    from nanochat.gpt import Linear as _GPTLinear
    model.lm_head_stage = _GPTLinear(new_config.n_embd, padded_vocab_size, bias=False).to(device)
    with torch.no_grad():
        model.lm_head_stage.weight.zero_()
    print0(f"  dual-head: lm_head_stage attached, shape={tuple(model.lm_head_stage.weight.shape)}, zero-init")

total_params = sum(p.numel() for p in model.parameters())
print0(f"  Total params: {total_params:,}")

# ---------------------------------------------------------------------------
# Freeze pretrained stages
# ---------------------------------------------------------------------------
# Freeze layers 0..old_n_layer-1
for i in range(old_n_layer):
    for p in model.transformer.h[i].parameters():
        p.requires_grad = False

# Warm-initialize chat special token embeddings from semantically related tokens.
# Chat tokens have norm ~30 (random init, never seen in pretraining) vs regular
# tokens norm ~200+. Without this, the chat structure is invisible to frozen stages.
# Skip when chaining on a parent stage — the parent has already trained these
# special-token embeddings, and overwriting would discard that work.
if args.parent_stage_tag:
    print0("  Skipping warm init (parent stage already has trained special tokens)")
else:
    with torch.no_grad():
        wte = model.transformer.wte.weight
        _enc = tokenizer.encode
        _sp = tokenizer.encode_special
        warm_map = {
            _sp("<|user_start|>"):      _enc("User")[0],
            _sp("<|user_end|>"):        _enc("\n")[0],
            _sp("<|assistant_start|>"): _enc("Assistant")[0],
            _sp("<|assistant_end|>"):   _enc("\n")[0],
            _sp("<|python_start|>"):    _enc("```")[0],
            _sp("<|python_end|>"):      _enc("```")[0],
            _sp("<|output_start|>"):    _enc("Output")[0],
            _sp("<|output_end|>"):      _enc("\n")[0],
        }
        for special_id, source_id in warm_map.items():
            wte[special_id] = wte[source_id].clone()
            print0(f"  Warm init: token {special_id} <- token {source_id}")

# Freeze embeddings, value_embeds, smear
for p in model.transformer.wte.parameters():
    p.requires_grad = False
for ve in model.value_embeds.values():
    for p in ve.parameters():
        p.requires_grad = False
model.smear_gate.weight.requires_grad = False
model.smear_lambda.requires_grad = False
model.backout_lambda.requires_grad = False

# Freeze resid/x0 lambdas for old layers (keep new ones trainable)
# These are single tensors, so we keep them trainable but the gradient
# for frozen indices will be zero (due to detach in forward)

if args.freeze_lm_head:
    for p in model.lm_head.parameters():
        p.requires_grad = False
    print0("  lm_head: frozen")
else:
    print0("  lm_head: trainable")

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print0(f"  Trainable: {trainable:,} / {total_params:,} ({100*trainable/total_params:.1f}%)")

# ---------------------------------------------------------------------------
# Monkey-patch forward to add freeze boundary + fixed backout
# ---------------------------------------------------------------------------
_original_forward = model.forward.__func__

def _patched_forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean',
                     poe_mode=None, poe_every=1, poe_alpha=0.0,
                     teacher_top_logits=None, teacher_top_indices=None,
                     kd_alpha=0.5, kd_temperature=2.0):
    B, T = idx.size()
    assert T <= self.cos.size(1)
    assert idx.device == self.cos.device
    T0 = 0 if kv_cache is None else kv_cache.get_pos()
    cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]

    x = self.transformer.wte(idx)
    x = x.to(COMPUTE_DTYPE)
    x = F.rms_norm(x, (x.size(-1),))

    # Smear
    if kv_cache is None:
        if T > 1:
            gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
            x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
    else:
        x_pre = kv_cache.prev_embedding
        kv_cache.prev_embedding = x[:, -1:, :]
        if T > 1:
            gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
            x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
        elif x_pre is not None:
            gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, :, :24]))
            x = x + gate * x_pre

    x0 = x
    n_layer = len(self.transformer.h)
    _backout_layer = original_backout_layer  # Fixed absolute index
    x_backout = None
    _frozen = old_n_layer

    for i, block in enumerate(self.transformer.h):
        x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        ve = self.value_embeds[str(i)](idx).to(x.dtype) if str(i) in self.value_embeds else None
        x = block(x, ve, cos_sin, self.window_sizes[i], kv_cache)
        if i == _backout_layer:
            x_backout = x
        # Freeze boundary
        if i == _frozen - 1:
            x = x.detach()
            x0 = x0.detach()

    if x_backout is not None:
        x = x - self.backout_lambda.to(x.dtype) * x_backout
    x = F.rms_norm(x, (x.size(-1),))

    softcap = 15
    # Dual-head composition at the final layer (paper §6.5). When the model
    # carries a specialist lm_head_stage, its zero-init-then-trained projection
    # is added to the frozen base projection before softcap.
    logits = self.lm_head(x)
    if getattr(self.config, "dual_head", False) and hasattr(self, "lm_head_stage"):
        logits = logits + self.lm_head_stage(x)
    logits = logits[..., :self.config.vocab_size]
    logits = logits.float()
    logits = softcap * torch.tanh(logits / softcap)

    if targets is not None:
        # Sum mode (explicit via loss_reduction='sum', or implicit when not training
        # e.g. evaluate_bpb which toggles model.eval()) returns raw unweighted CE sum
        # so val BPB remains comparable across tasks and runs. Training path applies
        # <|assistant_end|> down-weighting only (weight_scale was removed: it inflated
        # reported loss/bpb 3-16x and fed over-amplified gradients to the optimizer).
        flat_logits = logits.view(-1, logits.size(-1))
        flat_targets = targets.view(-1)
        if loss_reduction == 'sum' or not self.training:
            return F.cross_entropy(flat_logits, flat_targets,
                                   ignore_index=-1, reduction='sum')
        valid_mask = (flat_targets != -1).float()
        per_token_loss = F.cross_entropy(flat_logits, flat_targets,
                                         ignore_index=-1, reduction='none')
        END_TOKEN_ID = 32763  # <|assistant_end|>
        end_mask = (flat_targets == END_TOKEN_ID).float()
        per_token_weight = 1.0 - 0.5 * end_mask
        weighted_loss = per_token_loss * per_token_weight * valid_mask
        loss_sum = weighted_loss.sum()
        denom = torch.clamp((valid_mask * per_token_weight).sum(), min=1.0).to(loss_sum.dtype)
        return loss_sum / denom
    return logits

import types
model.forward = types.MethodType(_patched_forward, model)
print0("Patched forward with freeze boundary and fixed backout"
       + (" (+ dual-head composition)" if getattr(model.config, "dual_head", False) else ""))

# ---------------------------------------------------------------------------
# Identity verification
# ---------------------------------------------------------------------------
print0("Running identity verification...")
test_tokens = torch.tensor([[tokenizer.get_bos_token_id()] + tokenizer.encode("The")],
                           dtype=torch.long, device=device)
with torch.no_grad():
    logits = model(test_tokens)
    top1 = logits[0, -1].argmax().item()
    top1_val = logits[0, -1, top1].item()
print0(f"  top-1 token: {top1}, logit: {top1_val:.4f}")

# ---------------------------------------------------------------------------
# Optimizer (only trainable params)
# ---------------------------------------------------------------------------
# Inherit LRs from pretrain meta or use defaults
pretrain_config = meta.get("user_config", {})
emb_lr = args.embedding_lr or pretrain_config.get("embedding_lr") or 0.3
unemb_lr = args.unembedding_lr or pretrain_config.get("unembedding_lr") or 0.008
mat_lr = args.matrix_lr or pretrain_config.get("matrix_lr") or 0.02
scalar_lr_val = args.scalar_lr or pretrain_config.get("scalar_lr") or 0.5

# Scale by init_lr_frac
emb_lr *= args.init_lr_frac
unemb_lr *= args.init_lr_frac
mat_lr *= args.init_lr_frac
scalar_lr_val *= args.init_lr_frac

# SFT always uses weight_decay=0
# Build optimizer manually with ONLY trainable params (setup_optimizer includes
# frozen params, which causes grad=None errors in DistMuonAdamW all-reduce)
from nanochat.common import get_dist_info
from nanochat.optim import MuonAdamW, DistMuonAdamW
ddp_info = get_dist_info()
_ddp = ddp_info[0]
model_dim = new_config.n_embd
dmodel_lr_scale = (model_dim / 768) ** -0.5
print0(f"Scaling the LR for the AdamW parameters: {dmodel_lr_scale:.6f}")

# Collect only trainable params per group
new_matrix_params = [p for i in range(old_n_layer, new_n_layer)
                     for p in model.transformer.h[i].parameters() if p.requires_grad]
lm_head_params = [p for p in model.lm_head.parameters() if p.requires_grad]
resid_params = [model.resid_lambdas]  # single tensor, gradient zero for frozen indices via detach
x0_params = [model.x0_lambdas]

param_groups = []
if lm_head_params:
    param_groups.append(dict(kind='adamw', params=lm_head_params,
        lr=unemb_lr * dmodel_lr_scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.0))
# Dual-head: specialist head gets its own AdamW group (paper §6.5.2:
# specialist LR 0.25× backbone, weight_decay 0.1).
if args.dual_head and hasattr(model, "lm_head_stage"):
    stage_head_params = [p for p in model.lm_head_stage.parameters() if p.requires_grad]
    assert stage_head_params, "lm_head_stage has no trainable params — check dual_head init"
    stage_head_lr = args.lm_head_stage_lr if args.lm_head_stage_lr is not None \
        else 0.25 * unemb_lr  # paper §6.5.2 default
    param_groups.append(dict(kind='adamw', params=stage_head_params,
        lr=stage_head_lr * dmodel_lr_scale, betas=(0.8, 0.96), eps=1e-10,
        weight_decay=args.lm_head_stage_weight_decay))
    print0(f"  lm_head_stage: LR {stage_head_lr * dmodel_lr_scale:.2e}, "
           f"weight_decay {args.lm_head_stage_weight_decay}")
param_groups.append(dict(kind='adamw', params=resid_params,
    lr=scalar_lr_val * 0.01, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0))
param_groups.append(dict(kind='adamw', params=x0_params,
    lr=scalar_lr_val, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0))
# New stage matrix params grouped by shape
for shape in sorted({p.shape for p in new_matrix_params}):
    group_params = [p for p in new_matrix_params if p.shape == shape]
    param_groups.append(dict(kind='muon', params=group_params, lr=mat_lr,
        momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=0.0))

Factory = DistMuonAdamW if _ddp else MuonAdamW
optimizer = Factory(param_groups)
for group in optimizer.param_groups:
    group["initial_lr"] = group["lr"]
print0(f"Optimizer: {len(param_groups)} groups, matrix_lr={mat_lr}, unemb_lr={unemb_lr}")

# ---------------------------------------------------------------------------
# Data loading (SmolTalk SFT)
# ---------------------------------------------------------------------------
from tasks.smoltalk import SmolTalk

print0("Loading SFT data...")
train_tasks = []
for _ in range(args.smoltalk_epochs):
    train_tasks.append(SmolTalk(split="train"))
try:
    from tasks.mmlu import MMLU
    for _ in range(args.mmlu_epochs):
        train_tasks.append(MMLU(subset="all", split="auxiliary_train"))
except Exception as e:
    print0(f"  Skipping MMLU: {e}")
try:
    from tasks.gsm8k import GSM8K
    for _ in range(args.gsm8k_epochs):
        train_tasks.append(GSM8K(subset="main", split="train"))
except Exception as e:
    print0(f"  Skipping GSM8K: {e}")
try:
    from tasks.code_alpaca import CodeAlpaca
    for _ in range(args.code_alpaca_epochs):
        train_tasks.append(CodeAlpaca(split="train"))
except Exception as e:
    print0(f"  Skipping CodeAlpaca: {e}")
try:
    from tasks.mathinstruct import MathInstruct
    for _ in range(args.math_instruct_epochs):
        train_tasks.append(MathInstruct(split="train"))
except Exception as e:
    print0(f"  Skipping MathInstruct: {e}")
try:
    from tasks.magicoder import Magicoder
    for _ in range(args.magicoder_epochs):
        train_tasks.append(Magicoder(split="train"))
except Exception as e:
    print0(f"  Skipping Magicoder: {e}")
try:
    from tasks.glaive_function_calling import GlaiveFunctionCalling
    for _ in range(args.glaive_fc_epochs):
        train_tasks.append(GlaiveFunctionCalling(split="train"))
except Exception as e:
    print0(f"  Skipping GlaiveFunctionCalling: {e}")
try:
    from tasks.xlam_function_calling import XlamFunctionCalling
    for _ in range(args.xlam_fc_epochs):
        train_tasks.append(XlamFunctionCalling(split="train"))
except Exception as e:
    print0(f"  Skipping XlamFunctionCalling: {e}")
try:
    from tasks.summarization import XSum
    for _ in range(args.xsum_epochs):
        train_tasks.append(XSum(split="train"))
except Exception as e:
    print0(f"  Skipping XSum: {e}")
try:
    from tasks.summarization import CNNDailyMail
    for _ in range(args.cnndm_epochs):
        train_tasks.append(CNNDailyMail(split="train"))
except Exception as e:
    print0(f"  Skipping CNNDailyMail: {e}")
try:
    from tasks.summarization import SAMSum
    for _ in range(args.samsum_epochs):
        train_tasks.append(SAMSum(split="train"))
except Exception as e:
    print0(f"  Skipping SAMSum: {e}")
try:
    from tasks.summarization import XLSumEN
    for _ in range(args.xlsum_en_epochs):
        train_tasks.append(XLSumEN(split="train"))
except Exception as e:
    print0(f"  Skipping XLSumEN: {e}")
try:
    from tasks.ner import CoNLL2003
    for _ in range(args.conll_epochs):
        train_tasks.append(CoNLL2003(split="train"))
except Exception as e:
    print0(f"  Skipping CoNLL2003: {e}")
try:
    from tasks.ner import FewNERD
    for _ in range(args.fewnerd_epochs):
        train_tasks.append(FewNERD(split="train"))
except Exception as e:
    print0(f"  Skipping FewNERD: {e}")
try:
    from tasks.ner import WikiANNEn
    for _ in range(args.wikiann_epochs):
        train_tasks.append(WikiANNEn(split="train"))
except Exception as e:
    print0(f"  Skipping WikiANNEn: {e}")

# Build conversation list
all_conversations = []
for task in train_tasks:
    n = task.num_examples()
    print0(f"  {task.__class__.__name__}: {n} examples")
    for i in range(n):
        all_conversations.append(task.get_example(i))
print0(f"  Total: {len(all_conversations)} conversations")

# Deterministic shuffle + split: last `val_convs` reserved as held-out validation,
# then optional subsampling of the remainder to `max_convs` for the train set.
random.Random(42).shuffle(all_conversations)
if args.val_convs > 0 and len(all_conversations) > args.val_convs:
    val_conversations = all_conversations[-args.val_convs:]
    all_conversations = all_conversations[:-args.val_convs]
else:
    val_conversations = []
if args.max_convs > 0 and len(all_conversations) > args.max_convs:
    all_conversations = all_conversations[:args.max_convs]
    print0(f"  Subsampled train: {len(all_conversations):,} convs (seed=42)")
print0(f"  Train: {len(all_conversations):,} | Val (held-out): {len(val_conversations):,}")

# Case-augment first user turn's leading greeting so the model learns
# Hi == hi == HI == HiYa == Hey etc. map to the same chat register.
# Addresses short-prompt case sensitivity on small models.
_GREETING_PATTERNS = ("hi", "hello", "hey", "hiya", "yo",
                      "greetings", "good morning", "good afternoon", "good evening")

def _case_augment_first_user(conv):
    """Return a deep-copied conversation where the first user turn has its
    leading greeting (if any) case-perturbed. No-op if no greeting match."""
    import copy, random as _r
    new = copy.deepcopy(conv)
    msgs = new.get("messages", [])
    if not msgs or msgs[0].get("role") != "user":
        return None
    content = msgs[0].get("content")
    if not isinstance(content, str):
        return None
    stripped = content.lstrip()
    if not stripped:
        return None
    lower = stripped.lower()
    for g in _GREETING_PATTERNS:
        if lower.startswith(g):
            variant_fn = _r.choice([str.lower, str.upper, str.title, str.capitalize])
            new_greet = variant_fn(stripped[:len(g)])
            # If the chosen variant is byte-identical to the original slice,
            # signal "skip" so we don't duplicate the exact same conversation.
            if new_greet == stripped[:len(g)]:
                return None
            prefix = content[: len(content) - len(stripped)]
            msgs[0]["content"] = prefix + new_greet + stripped[len(g):]
            return new
    return None

# Best-fit packing SFT data generator
import random
def sft_data_generator(conversations, buffer_size=200):
    """Yield (inputs, targets) batches with best-fit packing and loss masking."""
    B = args.device_batch_size
    T = args.max_seq_len
    row_capacity = T + 1
    bos = tokenizer.get_bos_token_id()

    # Case augmentation: for each conv, yield original + (optional) case variant.
    effective_convs = conversations
    if getattr(args, "case_augment", False):
        augmented = []
        for c in conversations:
            augmented.append(c)
            v = _case_augment_first_user(c)
            if v is not None:
                augmented.append(v)
        print0(f"  Case-augmented: {len(conversations):,} -> {len(augmented):,} conversations")
        effective_convs = augmented

    # Tokenize all conversations
    tokenized = []
    for conv in effective_convs:
        ids, mask = tokenizer.render_conversation(conv, max_tokens=T)
        if len(ids) >= 2:
            tokenized.append((ids, mask))
    random.shuffle(tokenized)
    print0(f"  Tokenized: {len(tokenized)} conversations")

    buffer = list(tokenized)
    buffer.sort(key=lambda x: len(x[0]))

    while len(buffer) >= B:
        rows_ids = []
        rows_mask = []
        for _ in range(B):
            row_ids, row_mask = [], []
            remaining = row_capacity
            while remaining > 0 and buffer:
                best_idx = None
                for j in range(len(buffer) - 1, -1, -1):
                    if len(buffer[j][0]) <= remaining:
                        best_idx = j
                        break
                if best_idx is None:
                    break
                doc_ids, doc_mask = buffer.pop(best_idx)
                take = min(len(doc_ids), remaining)
                row_ids.extend(doc_ids[:take])
                row_mask.extend(doc_mask[:take])
                remaining -= take
            # Pad
            if len(row_ids) < row_capacity:
                pad = row_capacity - len(row_ids)
                row_ids.extend([bos] * pad)
                row_mask.extend([0] * pad)
            rows_ids.append(row_ids)
            rows_mask.append(row_mask)

        ids_t = torch.tensor(rows_ids, dtype=torch.long, device=device)
        mask_t = torch.tensor(rows_mask, dtype=torch.int8, device=device)
        inputs = ids_t[:, :-1]
        targets = ids_t[:, 1:].clone()
        targets[mask_t[:, 1:] == 0] = -1
        yield inputs, targets

# Compute total steps
tokens_per_step = args.device_batch_size * args.max_seq_len * ddp_world_size
total_tokens_est = sum(len(ids) for ids, _ in
    [tokenizer.render_conversation(c, args.max_seq_len) for c in all_conversations[:200]]
) / 200 * len(all_conversations)
num_steps = args.num_iterations if args.num_iterations > 0 else int(total_tokens_est / tokens_per_step)
print0(f"Training for {num_steps} steps ({tokens_per_step:,} tok/step)")

# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------
def get_lr_multiplier(progress):
    if args.warmup_ratio > 0 and progress < args.warmup_ratio:
        return progress / args.warmup_ratio
    if progress <= 1.0 - args.warmdown_ratio:
        return 1.0
    decay = (progress - (1.0 - args.warmdown_ratio)) / args.warmdown_ratio
    return (1 - decay) + decay * args.final_lr_frac

def get_muon_momentum(step):
    frac = min(step / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
output_tag = args.output_tag or f"sft_stage5_{args.model_tag}"
output_dir = os.path.join(base_dir, "chatsft_checkpoints", output_tag)
if master_process:
    os.makedirs(output_dir, exist_ok=True)

# Patch Muon's compiled kernel: the original muon_step_fused has a dtype bug
# where `1 - beta2` is bf16 but second_momentum_buffer is fp32, causing
# lerp_ to fail under torch.compile. Fix: cast beta2 to float32 for lerp_.
import nanochat.optim as _optim_module
@torch.compile(dynamic=False, fullgraph=True)
def _patched_muon_step_fused(stacked_grads, stacked_params, momentum_buffer,
                              second_momentum_buffer, momentum_t, lr_t, wd_t,
                              beta2_t, ns_steps, red_dim):
    from nanochat.common import COMPUTE_DTYPE as _CD
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)
    X = g.bfloat16() if _CD == torch.bfloat16 else g
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)
    coeffs = _optim_module.polar_express_coeffs
    if g.size(-2) > g.size(-1):
        for a, b, c in coeffs[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        for a, b, c in coeffs[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    g = X
    beta2 = beta2_t.float()  # FIX: keep beta2 as float32 for lerp_ compat
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)

_optim_module.muon_step_fused = _patched_muon_step_fused
print0("Patched muon_step_fused with float32 beta2 fix")

if device_type == "cuda":
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
    model = torch.compile(model)

@torch.no_grad()
def evaluate_bpb(val_convs):
    """Held-out bpb on DDP shard of val_convs, all-reduced. No case-aug, no packing.
    Each conversation padded/truncated to (T+1) tokens; loss averaged over assistant-mask tokens only."""
    if len(val_convs) < ddp_world_size * args.device_batch_size:
        return float("nan")
    model.eval()
    T = args.max_seq_len
    B = args.device_batch_size
    bos = tokenizer.get_bos_token_id()
    per_rank = len(val_convs) // ddp_world_size
    my_slice = val_convs[ddp_rank * per_rank : (ddp_rank + 1) * per_rank]
    total_loss = 0.0
    total_valid = 0
    for i in range(0, len(my_slice) - B + 1, B):
        rows_ids, rows_mask = [], []
        for c in my_slice[i:i + B]:
            ids, mask = tokenizer.render_conversation(c, max_tokens=T + 1)
            if len(ids) < 2:
                ids, mask = [bos, bos], [0, 0]
            if len(ids) < T + 1:
                pad = T + 1 - len(ids)
                ids = ids + [bos] * pad
                mask = mask + [0] * pad
            else:
                ids = ids[:T + 1]
                mask = mask[:T + 1]
            rows_ids.append(ids)
            rows_mask.append(mask)
        ids_t = torch.tensor(rows_ids, dtype=torch.long, device=device)
        mask_t = torch.tensor(rows_mask, dtype=torch.int8, device=device)
        inputs = ids_t[:, :-1]
        targets = ids_t[:, 1:].clone()
        valid_mask = mask_t[:, 1:] == 1
        targets[~valid_mask] = -1
        loss = model(inputs, targets, loss_reduction='sum')
        valid_count = int(valid_mask.sum().item())
        if valid_count > 0:
            total_loss += float(loss.item())
            total_valid += valid_count
    stats = torch.tensor([total_loss, float(total_valid)], dtype=torch.float64, device=device)
    if ddp_world_size > 1:
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    avg_loss = (stats[0] / stats[1].clamp(min=1)).item()
    val_bpb = avg_loss / math.log(2)
    model.train()
    return val_bpb

train_loader = sft_data_generator(all_conversations)

start_step = resume_step + 1
if resume_step > 0:
    print0(f"Skipping {resume_step} batches to resume position...")
    for _ in range(resume_step):
        try:
            next(train_loader)
        except StopIteration:
            train_loader = sft_data_generator(all_conversations)
            next(train_loader)
    print0(f"Resumed at step {start_step}")

print0(f"\nStarting SFT training from step {start_step}...")
t0 = time.time()
smooth_loss = None

for step in range(start_step, num_steps + 1):
    # Get batch
    try:
        inputs, targets = next(train_loader)
    except StopIteration:
        train_loader = sft_data_generator(all_conversations)
        inputs, targets = next(train_loader)

    # LR schedule
    progress = step / num_steps
    lr_mult = get_lr_multiplier(progress)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lr_mult

    # Muon momentum ramp
    momentum = get_muon_momentum(step)
    for group in optimizer.param_groups:
        if group.get("kind") == "muon":
            group["momentum"] = momentum

    # Forward + backward
    loss = model(inputs, targets)
    loss.backward()
    # Cast gradients to fp32 to match master weight dtype.
    # The frozen bf16 prefix produces bf16 activations; backward through
    # the new fp32 layers can yield bf16 gradients. Muon's compiled
    # kernel requires all tensors (params, grads, momentum) to match dtype.
    for p in model.parameters():
        if p.grad is not None and p.grad.dtype != p.dtype:
            p.grad = p.grad.to(p.dtype)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    loss_val = loss.item()
    smooth_loss = loss_val if smooth_loss is None else 0.95 * smooth_loss + 0.05 * loss_val

    if step % 50 == 0 or step == 1:
        synchronize()
        t1 = time.time()
        dt = t1 - t0
        tok_s = args.device_batch_size * args.max_seq_len * ddp_world_size * min(step, 50) / dt
        bpb = loss_val / math.log(2)
        smooth_bpb = smooth_loss / math.log(2)
        print0(f"step {step:6d}/{num_steps} | loss {loss_val:.4f} | bpb {bpb:.4f} | "
               f"smooth_bpb {smooth_bpb:.4f} | lr {lr_mult:.4f} | mom {momentum:.3f} | "
               f"{tok_s:.0f} tok/s")
        wandb_run.log({"sft/loss": loss_val, "sft/bpb": bpb, "sft/smooth_bpb": smooth_bpb,
                        "sft/lr": lr_mult, "sft/step": step})
        t0 = t1

    if args.eval_every > 0 and step % args.eval_every == 0 and len(val_conversations) > 0:
        synchronize()
        val_t0 = time.time()
        val_bpb = evaluate_bpb(val_conversations)
        synchronize()
        val_dt = time.time() - val_t0
        print0(f"Step {step:6d} | Validation bpb: {val_bpb:.6f} (eval {val_dt:.1f}s)")
        wandb_run.log({"sft/val_bpb": val_bpb, "sft/step": step})
        t0 = time.time()  # reset so eval time doesn't inflate next tok/s reading

    # Save checkpoint (delta-only when dual-head; full-sd otherwise for
    # backward compatibility with legacy single-head stage5 runs)
    if args.save_every > 0 and step % args.save_every == 0 and master_process:
        meta_save = {"step": step, "smooth_loss": smooth_loss,
                     "model_config": {"sequence_len": new_config.sequence_len,
                                      "vocab_size": new_config.vocab_size,
                                      "n_layer": new_n_layer,
                                      "n_head": new_config.n_head,
                                      "n_kv_head": new_config.n_kv_head,
                                      "n_embd": new_config.n_embd,
                                      "window_pattern": new_config.window_pattern,
                                      "dual_head": getattr(new_config, "dual_head", False),
                                      "frozen_layers": old_n_layer},
                     "user_config": vars(args)}
        if args.dual_head:
            base_dir = os.path.join(base_dir if False else "", "")  # placeholder  # noqa: F841
            save_stage_delta(output_dir, step, model,
                             base_model_dir=args.model_tag,  # tag so loader can find the base
                             meta_data=meta_save, rank=ddp_rank)
        else:
            model_data = {k.removeprefix("_orig_mod."): v for k, v in model.state_dict().items()}
            save_checkpoint(output_dir, step, model_data, None, meta_save, rank=ddp_rank)

# Final save
if master_process:
    meta_save = {"step": num_steps, "smooth_loss": smooth_loss,
                 "model_config": {"sequence_len": new_config.sequence_len,
                                  "vocab_size": new_config.vocab_size,
                                  "n_layer": new_n_layer,
                                  "n_head": new_config.n_head,
                                  "n_kv_head": new_config.n_kv_head,
                                  "n_embd": new_config.n_embd,
                                  "window_pattern": new_config.window_pattern,
                                  "dual_head": getattr(new_config, "dual_head", False),
                                  "frozen_layers": old_n_layer},
                 "user_config": vars(args)}
    if args.dual_head:
        save_stage_delta(output_dir, num_steps, model,
                         base_model_dir=args.model_tag,
                         meta_data=meta_save, rank=0)
    else:
        model_data = {k.removeprefix("_orig_mod."): v for k, v in model.state_dict().items()}
        save_checkpoint(output_dir, num_steps, model_data, None, meta_save, rank=0)
    print0(f"Saved final checkpoint to {output_dir}")
    # Upload to S3
    print0("Uploading to S3...")
    os.system(f"aws s3 sync {output_dir} s3://nanochat-checkpoint-transfer/sft_stage5_{args.model_tag}/ --exclude '*.pt' --include 'model_*.pt' --include 'meta_*.json'")
    print0("Upload complete")

if ddp:
    dist.barrier()
    compute_cleanup()

print0("SFT complete.")
