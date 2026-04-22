# nanochat-poe

A research fork of [karpathy/nanochat](https://github.com/karpathy/nanochat) that adds **Product of Experts (PoE) local learning** on top of the original training pipeline: each stage of the transformer trains independently through its own cross-entropy loss against a shared `lm_head`, and no gradient crosses stage boundaries. The fork tracks upstream nanochat (tokenizer, pretraining, SFT, evals, chat UI) and adds the training- and inference-time primitives needed to reproduce the experiments in the papers listed below.

Upstream nanochat's README (leaderboard, speedrun story, guides) is preserved in `dev/UPSTREAM_README.md` for reference.

## Papers

This repository is the reference implementation for:

1. **Product of Experts as Scalable Local Learning: Modular Construction at 1.3B Parameters** (Jeong, 2026). DOI: [10.5281/zenodo.19547653](https://doi.org/10.5281/zenodo.19547653). Production-scale validation of clustered PoE on a 1.3B-parameter GPT: 6.0% BPB gap vs. a matched-ratio backprop baseline on ClimbMix, WAND-style adaptive stage pruning (1.82x speedup, 100% top-1 agreement), Stage-1-as-drafter speculative decoding (1.87x), and compute-matched dual-head post-hoc SFT that preserves base-module predictions bit-identically while adding specialist capability.
2. **Clustered Local Learning: Bridging Biological Plausibility and Practical LLM Training** (Jeong, 2026). The 897M result that motivates the clustered (per-stage, not per-layer) factorization and quantifies the 6.6% BPB gap / 1.33x wall-clock trade-off at `poe_every=5`.
3. **Stateless Depth: Systems Advantages of Layer-Independent Transformer Training** (Jeong, 2026). The theoretical backbone: lossless prefix pruning, bubble-free pipeline parallelism, and elastic scaling, all derived from the absence of cross-layer backward dependencies.

## What's new vs. upstream nanochat

The diff against upstream is intentionally narrow. The GPT module, optimizer, data loader, and training loop are inherited; PoE additions are localized to a few well-defined hook points.

| Capability | Flag / Script | Location |
|------------|--------------|----------|
| Clustered PoE training | `--poe-mode=flat --poe-every=N` | `scripts/base_train.py`, ~30-line hook in `nanochat/gpt.py` |
| Per-stage loss aggregation exponent | `--poe-alpha=A` (`loss / n**(1-A)`) | `scripts/base_train.py` |
| Per-stage additive heads | `--per-stage-head` | `nanochat/gpt.py` (`GPTConfig.per_stage_head`) |
| PoE pipeline parallelism | `--pipeline-rank`, `--pipeline-peer-addr`, `--pipeline-port` | `scripts/base_train.py` |
| Post-hoc specialist stage (elastic depth) | `scripts/specialist_sft_new_stage.py` | freezes base + shared head; trains only the appended stage |
| Dual-head specialist SFT | `--dual-head` on `specialist_sft_new_stage.py` | base head preserved bit-identically; specialist head composes additively |
| KD from external teacher | `scripts/run_kd_experiment.sh` + `scripts/generate_teacher_logits.py` | top-K logit distillation into PoE student |
| FA3 / FA2 / SDPA cascade | auto-selected | `nanochat/flash_attention.py` |

The PoE surgery itself is a few lines in the transformer forward pass:

```python
for i, block in enumerate(self.transformer.h):
    if poe_mode == "flat" and i > 0 and i % poe_every == 0:
        x = x.detach()  # stage boundary
    x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
    x = block(x, ...)
    if (i + 1) % poe_every == 0 or i == n_layer - 1:
        poe_loss += checkpoint(self._poe_layer_loss, x, targets)
```

The shared `lm_head` adds zero parameters; `x0`-residual blending preserves a direct pathway from the embedding to every stage so per-stage detachment does not sever representational bandwidth.

## Setup

nanochat-poe uses [uv](https://docs.astral.sh/uv/) for dependency management, inherited from upstream:

```bash
uv sync --extra gpu    # CUDA (A100/H100/...)
uv sync --extra cpu    # CPU-only / Apple Silicon (MPS)
source .venv/bin/activate
```

For development (pytest, matplotlib, ipykernel, transformers, bitsandbytes, etc.):

```bash
uv sync --extra gpu --group dev
```

## Running the PoE experiments

All commands below assume you are in the repository root and have the venv activated.

### 1. Small-scale PoE vs. BP sanity check (`d20`, 8x A100/H100, single node)

Runs a matched-baseline pair: standard backprop first, then flat clustered PoE with `poe_every=5` (four stages of five layers). Produces two checkpoints plus full training logs under `~/poe_results/`.

```bash
bash scripts/run_poe_experiment.sh
```

Under the hood this calls:

```bash
uv run torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=20 \
    --poe-mode=flat \
    --poe-every=5 \
    --device-batch-size=8
```

Set `--poe-mode=none` to train the matched backprop baseline with everything else identical.

### 2. 1.3B PoE with cross-node pipeline parallelism (`d24`, 2 nodes x 4 GPUs)

PoE's detach boundaries double as pipeline split points: only forward activations cross the network, never gradients. This lets two commodity nodes (NVLink internal, modest cross-node bandwidth) train the Chinchilla-optimal d24 configuration.

On node 0 (first half, layers 0-11, embedding):

```bash
PIPELINE_PEER=<node1-ip> bash scripts/run_chinchilla_d24_pipeline.sh 0
```

On node 1 (second half, layers 12-23, `lm_head`):

```bash
PIPELINE_PEER=<node0-ip> bash scripts/run_chinchilla_d24_pipeline.sh 1
```

Tunable environment variables: `NPROC_PER_NODE` (default 4), `PIPELINE_PORT` (default 29600), `RESUME_FROM_STEP` (default -1, i.e. fresh run).

### 3. Post-hoc specialist SFT (elastic depth, new stage only)

Starts from a trained PoE base, appends N new transformer blocks as a fresh stage, and trains only those blocks (plus, with `--dual-head`, a dedicated additive head). The shared `lm_head` and all existing stages are frozen, so base-model predictions are preserved bit-identically.

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.specialist_sft_new_stage -- \
    --model-tag=poe_d24_r10 \
    --new-layers=6 \
    --dual-head \
    --smoltalk-epochs=1 \
    --mmlu-epochs=3 \
    --gsm8k-epochs=4 \
    --output-tag=sft_new_stage_poe_d24_dual_head
```

Key flags:

- `--model-tag` / `--model-step`: which base checkpoint to attach to (default: last step of the tag).
- `--new-layers`: how many transformer blocks to append (defaults to 6; the new stage index is inferred from the existing PoE layout).
- `--dual-head`: give the new stage its own trainable additive head on top of the frozen shared `lm_head` (the paper's recommended configuration; single-head SFT overwrites base predictions).
- `--freeze-lm-head`: freeze the shared head without enabling `--dual-head` (rarely what you want; documented for ablation).
- `--resume-from <step>`: resume a specialist-SFT run from a saved SFT checkpoint.
- `--s3-upload-prefix <s3://bucket/path>` (optional): sync the final checkpoint under `<prefix>/<output-tag>/`. Omit this flag to skip S3 upload entirely (the default for public use).
- Per-dataset epoch counts (`--smoltalk-epochs`, `--mmlu-epochs`, `--gsm8k-epochs`, `--code-alpaca-epochs`, `--glaive-fc-epochs`, `--xsum-epochs`, `--conll-epochs`, ...) control the specialist data mix. Set unwanted sources to 0.

### 4. KD experiment (external teacher -> PoE student)

Distills a large external teacher (e.g. Gemma-family) into a PoE student via top-K logit caching:

```bash
bash scripts/run_kd_experiment.sh
```

Phases: (0) data + tokenizer, (1) teacher logit generation (`scripts/generate_teacher_logits.py`, top-16 logits over 500M tokens), (2) student KD + PoE training (`--kd-logits-dir`, `--kd-alpha`, `--kd-temperature`). The script as shipped targets `google/gemma-4-31b-it` as teacher and a d32 PoE student with `poe_every=8`; edit the script in place to change teacher / depth / window pattern.

## Inference primitives unique to PoE

These follow directly from per-stage supervision and do not require retraining:

- **Lossless stage-prefix inference**: running only the first `k` stages through the shared head yields the exact predictions that prefix had in the full model (Proposition 4.1 of the paper). At `d24` / r=10 the first stage alone recovers 87.5% of full-model factual accuracy.
- **WAND-style adaptive stage pruning**: exit early on tokens whose top-1 margin exceeds the sum of remaining p99 logit-delta bounds. 1.82x wall-clock speedup at 100% top-1 agreement in the 1.3B run.
- **Stage-1-as-drafter speculative decoding**: Stage 1 drafts K tokens at 25% compute; the full stack verifies K+1 positions in parallel. 1.87x speedup, no separate drafter network.
- **Parallel composition of independently-trained branches**: two branches (e.g. `[1..4]` base, `[1..5]` dual-head specialist) log-sum their stage logits to produce a sharper joint distribution than either branch alone.

Chat CLI / web UI for talking to a trained checkpoint are unchanged from upstream:

```bash
python -m scripts.chat_cli              # terminal
python -m scripts.chat_web              # web UI on port 8000
```

## File structure (delta from upstream)

```
.
+-- nanochat/
|   +-- gpt.py                          # PoE hook + per-stage heads + frozen-prefix support
|   +-- flash_attention.py              # FA3/FA2/SDPA cascade
|   +-- checkpoint_manager.py           # save_stage_delta for dual-head specialist SFT
|   +-- ... (rest inherited from upstream)
+-- scripts/
|   +-- base_train.py                   # + PoE flags, pipeline flags, KD flags
|   +-- specialist_sft_new_stage.py     # post-hoc stage training (any stage count)
|   +-- generate_teacher_logits.py      # top-K teacher logit caching for KD
|   +-- run_poe_experiment.sh           # d20 PoE vs BP, single node
|   +-- run_chinchilla_d24.sh           # d24 single-node reference
|   +-- run_chinchilla_d24_pipeline.sh  # d24 2-node PoE pipeline
|   +-- run_kd_experiment.sh            # teacher -> PoE student KD
|   +-- gcp_*.sh                        # GCP cluster helpers (launch, sync, retry)
|   +-- ... (upstream scripts unchanged)
+-- tasks/                              # upstream task mix + ner.py (CoNLL/FewNerd/WikiAnn)
+-- runs/                               # upstream runs (speedrun.sh, miniseries.sh, ...)
+-- dev/                                # upstream dev assets (logo, leaderboard, notebooks)
```

Everything else (dataloader, tokenizer, optimizer, engine, Web UI, task definitions) is inherited from upstream nanochat.

## Precision / dtype

Precision management is inherited verbatim from upstream: a single global `COMPUTE_DTYPE` auto-detected per hardware, overridable via `NANOCHAT_DTYPE`. Weights are stored in fp32 for optimizer precision; `nanochat.gpt.Linear` casts to `COMPUTE_DTYPE` during forward.

| Hardware | Default | Reason |
|----------|---------|--------|
| CUDA SM 80+ (A100, H100) | `bfloat16` | Native bf16 tensor cores |
| CUDA SM < 80 (V100, T4)  | `float32` | No bf16; `NANOCHAT_DTYPE=float16` enables fp16 + GradScaler |
| CPU / MPS | `float32` | No reduced-precision tensor cores |

fp16 training automatically enables a `GradScaler` in `base_train.py`. SFT supports this; RL currently does not. Inference in fp16 works everywhere.

## Running on CPU / MPS

`runs/runcpu.sh` (inherited) shows a minimal-sized PoE run for CPU / Apple Silicon. Expect toy results; the script exists so the full pipeline can be exercised end-to-end on a laptop. MLX-port benchmarks (speculative decoding, parallel stage branching on M1 Ultra) are documented in the PoE paper (Section 7) but the MLX reimplementation lives outside this repository.

## Citing this fork

If you use nanochat-poe in your research, please cite both the relevant PoE paper and the upstream nanochat codebase:

```bibtex
@misc{jeong2026poe,
  author       = {Jaepil Jeong},
  title        = {Product of Experts as Scalable Local Learning: Modular Construction at 1.3B Parameters},
  year         = {2026},
  institution  = {Cognica, Inc.},
  doi          = {10.5281/zenodo.19547653},
  url          = {https://doi.org/10.5281/zenodo.19547653}
}

@misc{nanochat,
  author = {Andrej Karpathy},
  title = {nanochat: The best ChatGPT that \$100 can buy},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/karpathy/nanochat}
}
```

## Acknowledgements

- This fork builds directly on [Andrej Karpathy](https://github.com/karpathy)'s [nanochat](https://github.com/karpathy/nanochat); the base training recipe, tokenizer, evaluation harness, and web UI are his work.
- The PoE framework is due to Geoffrey Hinton (Hinton, 2002); the present work instantiates it as a local-learning objective at LLM scale.
- Thanks to [HuggingFace](https://huggingface.co/) for SmolTalk and ClimbMix-derived training data mixes used in upstream nanochat.

## License

MIT (inherited from upstream nanochat).
