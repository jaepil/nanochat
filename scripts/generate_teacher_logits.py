"""
Generate teacher logits for knowledge distillation.

Loads a HuggingFace teacher model (e.g., Gemma 4 31B-it) with 4-bit quantization,
processes ClimbMix training data through nanochat's tokenizer and packing,
and saves top-K teacher logits per token position to disk.

Usage:
    python -m scripts.generate_teacher_logits --teacher google/gemma-4-31b-it --top-k 16 --num-tokens 500000000

The output is a directory of .pt files, each containing a dict:
    {"input_ids": (B, T), "top_indices": (B, T, K), "top_logits": (B, T, K)}
"""

import os
import argparse
import time
import torch
import numpy as np
from pathlib import Path

from nanochat.tokenizer import get_tokenizer
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.common import get_base_dir, print_banner

print_banner()

parser = argparse.ArgumentParser(description="Generate teacher logits for KD")
parser.add_argument("--teacher", type=str, default="google/gemma-4-31b-it", help="HuggingFace teacher model name")
parser.add_argument("--top-k", type=int, default=16, help="number of top logits to save per token")
parser.add_argument("--num-tokens", type=int, default=500_000_000, help="total tokens to process")
parser.add_argument("--batch-size", type=int, default=4, help="batch size for teacher inference")
parser.add_argument("--seq-len", type=int, default=2048, help="sequence length")
parser.add_argument("--output-dir", type=str, default="", help="output directory (default: ~/.cache/nanochat/teacher_logits)")
parser.add_argument("--load-in-4bit", action="store_true", default=True, help="load teacher in 4-bit (default)")
parser.add_argument("--load-in-8bit", action="store_true", help="load teacher in 8-bit instead of 4-bit")
args = parser.parse_args()

output_dir = args.output_dir if args.output_dir else os.path.join(get_base_dir(), "teacher_logits")
os.makedirs(output_dir, exist_ok=True)

# ---- Load teacher model ----
print(f"Loading teacher: {args.teacher}")
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

if args.load_in_8bit:
    quant_config = BitsAndBytesConfig(load_in_8bit=True)
    print("Quantization: 8-bit")
else:
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )
    print("Quantization: 4-bit (nf4)")

teacher = AutoModelForCausalLM.from_pretrained(
    args.teacher,
    quantization_config=quant_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
teacher.eval()
teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher)
teacher_vocab_size = teacher.config.vocab_size
print(f"Teacher loaded. Vocab size: {teacher_vocab_size}, Device map: {teacher.hf_device_map}")

# ---- Load nanochat tokenizer and dataloader ----
nanochat_tokenizer = get_tokenizer()
nanochat_vocab_size = nanochat_tokenizer.get_vocab_size()
print(f"Nanochat vocab size: {nanochat_vocab_size}")

# We need to re-tokenize with the teacher's tokenizer since vocab differs.
# Strategy: load raw text from ClimbMix, tokenize with teacher tokenizer,
# pack into sequences, and run teacher forward.
from nanochat.dataset import list_parquet_files
import pyarrow.parquet as pq

parquet_paths = list_parquet_files()
assert len(parquet_paths) > 0, "No dataset parquet files found"
# Use all but last (last is validation)
train_paths = parquet_paths[:-1]
print(f"Data shards: {len(train_paths)}")

# ---- Process data ----
print(f"\nGenerating teacher logits: {args.num_tokens:,} tokens, top-{args.top_k}, batch={args.batch_size}, seq_len={args.seq_len}")
print(f"Output: {output_dir}")

tokens_processed = 0
shard_idx = 0
batch_buffer = []  # accumulate tokenized sequences

t0 = time.time()
file_counter = 0

for pq_path in train_paths:
    if tokens_processed >= args.num_tokens:
        break

    pf = pq.ParquetFile(pq_path)
    for rg_idx in range(pf.metadata.num_row_groups):
        if tokens_processed >= args.num_tokens:
            break

        table = pf.read_row_group(rg_idx, columns=["text"])
        texts = table.column("text").to_pylist()

        for text in texts:
            if tokens_processed >= args.num_tokens:
                break

            # Tokenize with teacher tokenizer
            encoded = teacher_tokenizer(
                text,
                max_length=args.seq_len,
                truncation=True,
                padding=False,
                return_tensors=None,
            )
            input_ids = encoded["input_ids"]

            if len(input_ids) < 32:  # skip very short docs
                continue

            # Truncate to seq_len
            if len(input_ids) > args.seq_len:
                input_ids = input_ids[:args.seq_len]

            batch_buffer.append(input_ids)

            # Process batch when full
            if len(batch_buffer) >= args.batch_size:
                # Pad batch to same length
                max_len = max(len(ids) for ids in batch_buffer[:args.batch_size])
                batch_ids = torch.zeros(args.batch_size, max_len, dtype=torch.long)
                attention_mask = torch.zeros(args.batch_size, max_len, dtype=torch.long)
                for i, ids in enumerate(batch_buffer[:args.batch_size]):
                    batch_ids[i, :len(ids)] = torch.tensor(ids)
                    attention_mask[i, :len(ids)] = 1

                batch_buffer = batch_buffer[args.batch_size:]

                # Teacher forward
                with torch.no_grad():
                    batch_ids = batch_ids.to(teacher.device)
                    attention_mask = attention_mask.to(teacher.device)
                    outputs = teacher(input_ids=batch_ids, attention_mask=attention_mask)
                    logits = outputs.logits.float()  # (B, T, V)

                    # Extract top-K
                    top_logits, top_indices = logits.topk(args.top_k, dim=-1)  # (B, T, K)

                # Save to disk
                save_dict = {
                    "input_ids": batch_ids.cpu().to(torch.int32),
                    "attention_mask": attention_mask.cpu().to(torch.bool),
                    "top_indices": top_indices.cpu().to(torch.int32),
                    "top_logits": top_logits.cpu().to(torch.float16),
                }
                save_path = os.path.join(output_dir, f"logits_{file_counter:06d}.pt")
                torch.save(save_dict, save_path)
                file_counter += 1

                batch_tokens = attention_mask.sum().item()
                tokens_processed += batch_tokens

                # Progress
                elapsed = time.time() - t0
                tps = tokens_processed / elapsed if elapsed > 0 else 0
                pct = 100 * tokens_processed / args.num_tokens
                eta = (args.num_tokens - tokens_processed) / tps / 3600 if tps > 0 else 0
                if file_counter % 10 == 0:
                    print(f"  [{pct:5.1f}%] {tokens_processed:>12,} / {args.num_tokens:,} tokens | {tps:,.0f} tok/s | {file_counter} files | eta: {eta:.1f}h")

elapsed = time.time() - t0
print(f"\nDone! {tokens_processed:,} tokens in {elapsed/3600:.1f} hours ({file_counter} files)")
print(f"Output: {output_dir}")

# Save metadata
import json
meta = {
    "teacher": args.teacher,
    "top_k": args.top_k,
    "num_tokens": tokens_processed,
    "num_files": file_counter,
    "batch_size": args.batch_size,
    "seq_len": args.seq_len,
    "teacher_vocab_size": teacher_vocab_size,
}
with open(os.path.join(output_dir, "meta.json"), "w") as f:
    json.dump(meta, f, indent=2)
print(f"Metadata saved to {output_dir}/meta.json")
