#!/usr/bin/env python
"""chat.py – interactive sampler (DATA_DIR aware)
-------------------------------------------------
Loads `$DATA_DIR/checkpoints/model.pth` + `vocab.json` and completes text.
Usage:
    python chat.py --temp 0.7 --max-len 250
"""
import os, json, argparse, torch
from dotenv import load_dotenv
from model import MiniTransformer
from utils import checkpoints_dir

# -----------------------------------------------------------------------------
# loader helpers
# -----------------------------------------------------------------------------

def load_model(device):
    load_dotenv("../.env")
    ckpt_dir = checkpoints_dir()
    model_pth = os.path.join(ckpt_dir, 'model.pth')
    vocab_json = os.path.join(ckpt_dir, 'vocab.json')
    if not (os.path.exists(model_pth) and os.path.exists(vocab_json)):
        raise FileNotFoundError("Train first: model.pth / vocab.json missing in checkpoints/.")

    vocab = json.load(open(vocab_json))
    stoi  = {ch:i for i,ch in enumerate(vocab)}
    model = MiniTransformer(len(vocab))
    model.load_state_dict(torch.load(model_pth, map_location=device))
    model.to(device).eval()
    return model, stoi, vocab

@torch.no_grad()
def sample(model, stoi, itos, prompt, device, max_len=200, temp=0.8):
    ids = [stoi.get(c, 0) for c in prompt]
    x = torch.tensor(ids, device=device).unsqueeze(0)
    for _ in range(max_len):
        logits = model(x)[:, -1, :] / temp
        next_id = torch.multinomial(torch.softmax(logits, -1), 1).item()
        x = torch.cat([x, torch.tensor([[next_id]], device=device)], dim=1)
        if itos[next_id] == '\n':
            break
    return ''.join(itos[i] for i in x[0].tolist())

# -----------------------------------------------------------------------------
# REPL
# -----------------------------------------------------------------------------

def main(a):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, stoi, vocab = load_model(device)

    print("Chat ready! (Ctrl‑D to quit)")
    while True:
        try:
            prompt = input("\nYou: ")
        except EOFError:
            break
        out = sample(model, stoi, vocab, prompt, device, a.max_len, a.temp)
        print("Bot:", out[len(prompt):])

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--max-len', type=int, default=200)
    p.add_argument('--temp',    type=float, default=0.8)
    main(p.parse_args())
