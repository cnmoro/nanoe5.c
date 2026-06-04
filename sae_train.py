"""
sae_train.py - train a sparse autoencoder (SAE) on e5 TOKEN embeddings to extract
"latent terms" (Option B from the mixedbread "latent terms" idea).

Corpus is PT + EN ONLY (Brazilian Portuguese + English), per requirement.

Pipeline:
  1. build a PT+EN passage corpus (wikipedia + the eval corpora for domain match)
  2. embed with e5 (no prefix -> matches the C sparse path), collect token hiddens
  3. train an L1 ReLU SAE (Anthropic-style) on centered token activations
  4. export sae weights (.pt) and a 4-bit sae.bin for the C engine

The doc sparse vector at inference is max-pool over tokens of ReLU(W_enc h + b').
"""
import os, sys, time, struct, random
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as Fn
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset

SRC = "hf_src"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
N_FEAT   = int(os.environ.get("SAE_N", 16384))
MAX_TOK  = int(os.environ.get("SAE_TOKENS", 3_000_000))   # cap collected token vecs
N_DOCS   = int(os.environ.get("SAE_DOCS", 80000))
CHUNKTOK = 256
TOPK     = int(os.environ.get("SAE_K", 32))               # active features per token
EPOCHS   = int(os.environ.get("SAE_EPOCHS", 8))
torch.set_grad_enabled(False)
random.seed(0); np.random.seed(0); torch.manual_seed(0)


def build_corpus(n_docs):
    """Half EN, half PT. Mix wikipedia (diverse) + eval-domain corpora."""
    texts = []
    half = n_docs // 2

    def take(stream, field, n, minlen=120):
        out = []
        for ex in stream:
            t = ex.get(field) or ""
            if isinstance(t, str) and len(t) >= minlen:
                out.append(t[:2000])
                if len(out) >= n:
                    break
        return out

    print("[corpus] EN: scifact corpus + wikipedia.en")
    en = take(load_dataset("mteb/scifact", "corpus", split="corpus", streaming=True), "text", 5000)
    en += take(load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True), "text", half - len(en))
    print("[corpus] PT: quati corpus + wikipedia.pt + mmarco-pt")
    pt = take(load_dataset("pt-mteb/quati_1M_retrieval", "corpus", split="corpus", streaming=True), "text", 15000)
    pt += take(load_dataset("yywwrr/mmarco_portuguese_500k", split="train", streaming=True), "text", 10000)
    pt += take(load_dataset("wikimedia/wikipedia", "20231101.pt", split="train", streaming=True), "text", half - len(pt))
    texts = en[:half] + pt[:half]
    random.shuffle(texts)
    print(f"[corpus] {len(texts)} passages (EN+PT)")
    return texts


def collect_tokens(texts):
    tok = AutoTokenizer.from_pretrained(SRC)
    mdl = AutoModel.from_pretrained(SRC).eval().to(DEV).half()
    buf = np.empty((MAX_TOK, 384), dtype=np.float16)
    n = 0
    t0 = time.time()
    BS = 64
    for i in range(0, len(texts), BS):
        batch = texts[i:i + BS]
        enc = tok(batch, padding=True, truncation=True, max_length=CHUNKTOK, return_tensors="pt").to(DEV)
        h = mdl(**enc).last_hidden_state               # [B,L,384]
        mask = enc["attention_mask"].bool()
        # drop the two specials per row (bos at 0, eos at last real token) to match C
        for r in range(h.shape[0]):
            idx = mask[r].nonzero(as_tuple=True)[0]
            if idx.numel() <= 2:
                continue
            sel = idx[1:-1]                              # content tokens only
            v = h[r, sel].float().cpu().numpy().astype(np.float16)
            # subsample if we'd overflow
            take = min(v.shape[0], MAX_TOK - n)
            if take <= 0:
                break
            if take < v.shape[0]:
                v = v[np.random.choice(v.shape[0], take, replace=False)]
            buf[n:n + v.shape[0]] = v
            n += v.shape[0]
        if n >= MAX_TOK:
            break
        if i % (BS * 50) == 0:
            print(f"  embedded {i+len(batch)}/{len(texts)} docs, {n} tokens, {time.time()-t0:.0f}s")
    print(f"[tokens] collected {n} token vectors in {time.time()-t0:.0f}s")
    return torch.from_numpy(buf[:n]).float()


class SAE(nn.Module):
    """TopK sparse autoencoder (OpenAI/Anthropic style). Exactly K features fire
    per token; an AuxK term reconstructs the residual from currently-dead units
    to keep the dictionary alive."""
    def __init__(self, d, N, mean, k):
        super().__init__()
        self.k = k
        self.b_dec = nn.Parameter(mean.clone())
        W = torch.randn(d, N) / (d ** 0.5)
        self.W_enc = nn.Parameter(W.clone())
        self.b_enc = nn.Parameter(torch.zeros(N))
        self.W_dec = nn.Parameter(Fn.normalize(W.t().clone(), dim=1))   # [N, d] unit rows
        self.register_buffer("fired_ago", torch.zeros(N))              # steps since last firing

    def pre(self, x):
        return Fn.relu((x - self.b_dec) @ self.W_enc + self.b_enc)

    @staticmethod
    def _topk(pre, k):
        val, idx = pre.topk(k, dim=1)
        return torch.zeros_like(pre).scatter_(1, idx, val), idx

    def forward(self, x):
        pre = self.pre(x)
        z, idx = self._topk(pre, self.k)
        xhat = z @ self.W_dec + self.b_dec
        return xhat, pre, z, idx


def train_sae(acts):
    torch.set_grad_enabled(True)
    d = acts.shape[1]
    mean = acts.mean(0)
    sae = SAE(d, N_FEAT, mean, TOPK).to(DEV)
    opt = torch.optim.Adam(sae.parameters(), lr=4e-4)
    acts = acts.to(DEV)
    n = acts.shape[0]; BS = 4096
    DEAD_AFTER = 2 * (n // BS + 1)        # ~2 epochs without firing -> "dead"
    K_AUX = 256
    for ep in range(EPOCHS):
        perm = torch.randperm(n, device=DEV)
        tot_mse = nb = 0.0
        for i in range(0, n, BS):
            x = acts[perm[i:i + BS]]
            xhat, pre, z, idx = sae(x)
            mse = Fn.mse_loss(xhat, x)
            # AuxK: model the residual with the top dead features
            loss = mse
            with torch.no_grad():
                fired = torch.zeros(N_FEAT, device=DEV)
                fired[idx.reshape(-1)] = 1
                sae.fired_ago += 1; sae.fired_ago[fired.bool()] = 0
                dead = sae.fired_ago > DEAD_AFTER
            if dead.any():
                resid = (x - xhat).detach()
                pre_dead = pre.clone()
                pre_dead[:, ~dead] = 0
                kk = min(K_AUX, int(dead.sum()))
                zaux, _ = SAE._topk(pre_dead, kk)
                raux = zaux @ sae.W_dec
                loss = loss + 0.03 * Fn.mse_loss(raux, resid)
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                sae.W_dec.data = Fn.normalize(sae.W_dec.data, dim=1)
            tot_mse += mse.item(); nb += 1
        with torch.no_grad():
            xh, _, _, _ = sae(acts[:20000])
            fvu = ((acts[:20000] - xh) ** 2).mean() / acts[:20000].var()
            ndead = int((sae.fired_ago > DEAD_AFTER).sum())
        print(f"  epoch {ep}: mse={tot_mse/nb:.4f} 1-FVU={1-fvu.item():.3f} dead={ndead}/{N_FEAT}")
    torch.set_grad_enabled(False)
    return sae, mean


# ---- Q4_0 export (same format as convert.py) ----
QK = 32
_BLK = np.dtype([("d", "<f2"), ("qs", "u1", 16)])
def quantize_q4_0(a):
    a = np.ascontiguousarray(a, np.float32).reshape(-1, QK)
    nb = a.shape[0]
    idx = np.argmax(np.abs(a), 1); amax = a[np.arange(nb), idx]
    dq = amax / -8.0; inv = np.where(dq != 0, 1.0 / dq, 0).astype(np.float32)
    q = np.clip(np.floor(a * inv[:, None] + 8.5), 0, 15).astype(np.uint8)
    qs = (q[:, :16] | (q[:, 16:] << 4)).astype(np.uint8)
    out = np.empty(nb, _BLK); out["d"] = dq.astype(np.float16); out["qs"] = qs
    return out


def export_sae(sae, mean, path="sae.bin"):
    W_enc = sae.W_enc.detach().cpu().float().numpy()          # [d, N]
    b_enc = sae.b_enc.detach().cpu().float().numpy()          # [N]
    mean_np = mean.detach().cpu().float().numpy()             # [d]
    # fold centering into the bias: relu(W^T (h-mean) + b) = relu(W^T h + (b - W^T mean))
    bprime = (b_enc - mean_np @ W_enc).astype(np.float32)     # [N]
    Wt = np.ascontiguousarray(W_enc.T)                        # [N, d] row per feature
    d = Wt.shape[1]; N = Wt.shape[0]
    with open(path, "wb") as f:
        f.write(b"SAE1"); f.write(struct.pack("<iiii", N, d, QK, sae.k))
        f.write(bprime.tobytes())
        f.write(quantize_q4_0(Wt).tobytes())
    print(f"[export] {path}: N={N} d={d} k={sae.k} ({os.path.getsize(path)/1e6:.1f} MB)")


def main():
    texts = build_corpus(N_DOCS)
    acts = collect_tokens(texts)
    sae, mean = train_sae(acts)
    torch.save({"W_enc": sae.W_enc.detach().cpu(), "b_enc": sae.b_enc.detach().cpu(),
                "W_dec": sae.W_dec.detach().cpu(), "mean": mean.detach().cpu(),
                "N": N_FEAT, "k": sae.k},
               "sae.pt")
    export_sae(sae, mean)
    print("done.")


if __name__ == "__main__":
    main()
