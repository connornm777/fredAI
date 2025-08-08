#!/usr/bin/env python
"""train.py – resume‑capable tiny Transformer (**DATA_DIR layout**)
=================================================================
Directory layout
----------------
.env contains a key
```
DATA_DIR=/absolute/or/relative/path/to/project_storage
```
Inside that folder this script expects / creates:
```
$DATA_DIR/
    data/          # corpus lives here (tiny.txt)
    checkpoints/   # model.pth, optim.pth, metrics, logs
```
Everything else (analytics, AMP, accumulation, resume) is unchanged.
"""
import os, json, csv, time, argparse, psutil, random, math
from dotenv import load_dotenv
import numpy as np
import torch, torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from model import MiniTransformer

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def set_seed(seed: int = 0):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def get_data_dir() -> str:
    load_dotenv("../.env")
    base = os.getenv("DATA_DIR")
    if not base:
        raise EnvironmentError("Please define DATA_DIR in your .env file")
    return base


def corpus_path(block: int) -> tuple[str, list[str]]:
    base = get_data_dir()
    path = os.path.join(base, "data", "tiny.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Corpus not found at {path}. Place tiny.txt there.")
    return path


def load_corpus(block: int):
    path = corpus_path(block)
    text = open(path, encoding="utf8").read()
    vocab = sorted(set(text))
    stoi  = {c: i for i, c in enumerate(vocab)}
    ids   = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    if len(ids) < block + 1:
        raise ValueError("Corpus shorter than block_size+1; add more text.")
    return ids, vocab

class SlidingDataset(torch.utils.data.Dataset):
    def __init__(self, ids, block):
        self.ids, self.block = ids, block
    def __len__(self):
        return len(self.ids) - self.block - 1
    def __getitem__(self, i):
        chunk = self.ids[i : i + self.block + 1]
        return chunk[:-1], chunk[1:]

# -----------------------------------------------------------------------------
# Training (resume‑aware)
# -----------------------------------------------------------------------------

def train(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ids, vocab = load_corpus(args.block)
    model = MiniTransformer(len(vocab), d_model=args.d_model, nhead=args.nhead).to(device)

    base_dir = get_data_dir()
    ckpt_dir = os.path.join(base_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    model_path = os.path.join(ckpt_dir, 'model.pth')
    optim_path = os.path.join(ckpt_dir, 'optim.pth')

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    start_step = 0
    if args.resume:
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"[resume] loaded {model_path}")
        if os.path.exists(optim_path):
            optim.load_state_dict(torch.load(optim_path, map_location=device))
            start_step = optim.state_dict().get('step', 0)
            print(f"[resume] optimiser @ step {start_step}")

    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)

    dataset = SlidingDataset(ids, args.block)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch, shuffle=True,
                                         drop_last=True, pin_memory=torch.cuda.is_available())
    loader_iter = iter(loader)

    csv_path = os.path.join(ckpt_dir, 'train_metrics.csv')
    if start_step == 0 and not args.resume:
        with open(csv_path, 'w', newline='') as f:
            csv.writer(f).writerow(['step','loss','acc','tok_per_sec','gpu_MB','cpu_MB','grad_norm'])

    tb = SummaryWriter(args.tb_logdir) if args.tb_logdir else None
    tokens_per_step = args.batch * args.block
    step = start_step

    while step < args.steps:
        try:
            src, tgt = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            src, tgt = next(loader_iter)

        src, tgt = src.to(device), tgt.to(device)
        t0 = time.time()

        with torch.amp.autocast('cuda', enabled=args.amp):
            logits = model(src)
            loss = F.cross_entropy(logits.reshape(-1, len(vocab)), tgt.reshape(-1)) / args.accum
        scaler.scale(loss).backward()

        grad_norm = 0.0
        if (step + 1) % args.accum == 0:
            if args.clip > 0:
                scaler.unscale_(optim)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip).item()
            scaler.step(optim); scaler.update(); optim.zero_grad()

        duration = time.time() - t0
        tok_sec = tokens_per_step / duration
        acc = (logits.argmax(-1) == tgt).float().mean().item()
        gpu_mem = torch.cuda.memory_allocated() / 2**20 if torch.cuda.is_available() else 0
        cpu_mem = psutil.Process().memory_info().rss / 2**20

        if (step + 1) % args.log_every == 0:
            full_loss = loss.item() * args.accum
            print(f"step {step+1:6d} | loss {full_loss:.4f} | acc {acc:.3f} | "
                  f"{tok_sec:7.0f} tok/s | gpu {gpu_mem:5.0f}MB | grad {grad_norm:.2f}")
            with open(csv_path, 'a', newline='') as f:
                csv.writer(f).writerow([step+1, full_loss, acc, tok_sec, gpu_mem, cpu_mem, grad_norm])
            if tb:
                tb.add_scalar('loss', full_loss, step+1)
                tb.add_scalar('accuracy', acc, step+0)
                tb.add_scalar('tok_per_sec', tok_sec, step+1)
        step += 1

    torch.save(model.state_dict(), model_path)
    torch.save(optim.state_dict(), optim_path)
    with open(os.path.join(ckpt_dir, 'vocab.json'), 'w') as f:
        json.dump(vocab, f)
    print("Saved model & optimiser to", ckpt_dir)
    if tb: tb.close()

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Train tiny Transformer (DATA_DIR edition)')
    ap.add_argument('--steps',      type=int,   default=2000)
    ap.add_argument('--block',      type=int,   default=32)
    ap.add_argument('--batch',      type=int,   default=32)
    ap.add_argument('--accum',      type=int,   default=1)
    ap.add_argument('--d-model',    type=int,   default=64)
    ap.add_argument('--nhead',      type=int,   default=4)
    ap.add_argument('--lr',         type=float, default=1e-3)
    ap.add_argument('--clip',       type=float, default=1.0)
    ap.add_argument('--amp',        action='store_true')
    ap.add_argument('--seed',       type=int,   default=0)
    ap.add_argument('--log-every',  type=int,   default=50)
    ap.add_argument('--tb-logdir',  default='')
    ap.add_argument('--resume',     action='store_true')
    args = ap.parse_args()
    train(args)
