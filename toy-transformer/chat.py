#!/usr/bin/env python
"""chat.py – interactive sampler for the tiny character Transformer
------------------------------------------------------------------
• Uses the same `$DATA_DIR` environment variable (see .env) as train.py.
• Loads `checkpoints/model.pth` and `vocab.json` from `$DATA_DIR/checkpoints`.
• Works on CPU or GPU; optional temperature flag.
"""
import os, json, argparse, torch
from dotenv import load_dotenv
from model import MiniTransformer

# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------

def get_paths():
    load_dotenv("../.env")
    base = os.getenv("DATA_DIR")
    if not base:
        raise EnvironmentError("Set DATA_DIR in your .env file before running chat.py")
    ckpt_dir = os.path.join(base, "checkpoints")
    model_pth = os.path.join(ckpt_dir, "model.pth")
    vocab_json = os.path.join(ckpt_dir, "vocab.json")
    if not (os.path.exists(model_pth) and os.path.exists(vocab_json)):
        raise FileNotFoundError("model.pth or vocab.json not found in $DATA_DIR/checkpoints. Train first.")
    return model_pth, vocab_json

@torch.no_grad()
def sample(model, stoi, itos, prompt, device, max_len=200, temp=0.8):
    ids = [stoi.get(c, 0) for c in prompt]
    x = torch.tensor(ids, device=device).unsqueeze(0)  # (1,T)
    for _ in range(max_len):
        logits = model(x)[:, -1, :] / temp
        next_id = torch.multinomial(torch.softmax(logits, -1), 1).item()
        x = torch.cat([x, torch.tensor([[next_id]], device=device)], dim=1)
        if itos[next_id] == '\n':
            break
    return ''.join(itos[i] for i in x[0].tolist())

# -----------------------------------------------------------------------------
# main REPL
# -----------------------------------------------------------------------------

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_pth, vocab_json = get_paths()

    vocab = json.load(open(vocab_json))
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = vocab

    model = MiniTransformer(len(vocab))
    model.load_state_dict(torch.load(model_pth, map_location=device))
    model.to(device).eval()

    print("Chat ready!  (Ctrl-D to exit)")
    while True:
        try:
            prompt = input("\nYou: ")
        except EOFError:
            break
        generated = sample(model, stoi, itos, prompt, device, args.max_len, args.temp)
        print("Bot:", generated[len(prompt):])

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--max-len', type=int, default=200)
    p.add_argument('--temp',    type=float, default=0.8)
    args = p.parse_args()
    main(args)
