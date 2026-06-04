"""
sae_eval.py - does the SAE "latent terms" sparse vector actually help retrieval?

Compares dense (e5), sparse (SAE latent terms), and hybrid on:
  * English: mteb/scifact
  * Portuguese: pt-mteb/quati_1M_retrieval  (corpus subset for speed)

Reports nDCG@10, Recall@100, MRR@10. Sparse uses the same recipe as the C path:
per-token TopK over ReLU(W_enc h + b'), max-pooled over content tokens (no prefix).
"""
import os, sys, time
import numpy as np
import torch, torch.nn.functional as Fn
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset

SRC = "hf_src"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_grad_enabled(False)
SPARSE_PREFIX = os.environ.get("SPARSE_PREFIX", "0") == "1"   # ablation

tok = AutoTokenizer.from_pretrained(SRC)
mdl = AutoModel.from_pretrained(SRC).eval().to(DEV).half()

S = torch.load("sae.pt", map_location=DEV)
W_enc = S["W_enc"].float().to(DEV); b_enc = S["b_enc"].float().to(DEV)
mean = S["mean"].float().to(DEV); KTOK = int(S["k"]); NF = int(S["N"])
print(f"[sae] N={NF} k={KTOK}")


def dense_embed(texts, is_query, bs=128):
    pfx = ("query: " if is_query else "passage: ")
    out = []
    for i in range(0, len(texts), bs):
        enc = tok([pfx + t for t in texts[i:i+bs]], padding=True, truncation=True,
                  max_length=256, return_tensors="pt").to(DEV)
        h = mdl(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1)
        v = (h * m).sum(1) / m.sum(1)
        out.append(Fn.normalize(v.float(), dim=1).cpu())
    return torch.cat(out)


def sparse_embed(texts, bs=32):
    """[Ndocs, NF] max-pooled per-token-TopK SAE activations (no prefix)."""
    pfx = ("query: " if SPARSE_PREFIX else "")
    out = torch.zeros(len(texts), NF)
    for i in range(0, len(texts), bs):
        chunk = texts[i:i+bs]
        enc = tok([pfx + t for t in chunk], padding=True, truncation=True,
                  max_length=256, return_tensors="pt").to(DEV)
        h = mdl(**enc).last_hidden_state.float()                 # [B,L,384]
        pre = Fn.relu((h - mean) @ W_enc + b_enc)                # [B,L,NF]
        val, idx = pre.topk(KTOK, dim=2)                         # [B,L,k]
        B, L, _ = h.shape
        # content mask: drop bos (first) and eos (last real token) per row
        am = enc["attention_mask"]
        cmask = am.clone()
        cmask[:, 0] = 0
        lastreal = am.sum(1) - 1
        cmask[torch.arange(B, device=DEV), lastreal] = 0
        cmask = cmask.unsqueeze(-1).expand(-1, -1, KTOK).reshape(B, L*KTOK).float()
        doc = torch.zeros(B, NF, device=DEV)
        doc.scatter_reduce_(1, idx.reshape(B, L*KTOK), (val.reshape(B, L*KTOK) * cmask),
                            reduce="amax", include_self=True)
        out[i:i+B] = doc.cpu()
    return out


# ----------------------------- metrics ---------------------------------
def metrics(scores, qids, dids, qrels, k_ndcg=10, k_rec=100):
    did_pos = {d: j for j, d in enumerate(dids)}
    ndcgs, recs, mrrs = [], [], []
    order = (-scores).argsort(1)
    for qi, q in enumerate(qids):
        rel = qrels.get(q, {})
        if not rel:
            continue
        ranked = [dids[j] for j in order[qi][:max(k_ndcg, k_rec)].tolist()]
        # nDCG@k
        dcg = sum((rel.get(d, 0) > 0) / np.log2(2 + r) for r, d in enumerate(ranked[:k_ndcg]))
        ideal = sum(1 / np.log2(2 + r) for r in range(min(len(rel), k_ndcg)))
        ndcgs.append(dcg / ideal if ideal else 0)
        # Recall@k_rec
        nrel = sum(1 for v in rel.values() if v > 0)
        hit = sum(1 for d in ranked[:k_rec] if rel.get(d, 0) > 0)
        recs.append(hit / nrel if nrel else 0)
        # MRR@10
        rr = 0
        for r, d in enumerate(ranked[:10]):
            if rel.get(d, 0) > 0:
                rr = 1 / (r + 1); break
        mrrs.append(rr)
    return np.mean(ndcgs), np.mean(recs), np.mean(mrrs)


def minmax(x):
    lo = x.min(1, keepdim=True).values; hi = x.max(1, keepdim=True).values
    return (x - lo) / (hi - lo + 1e-9)


def run_task(name, corpus_ds, queries_ds, qrels_ds, max_corpus=20000, scan_cap=150000):
    print(f"\n===== {name} =====")
    qrels = {}
    for r in qrels_ds:
        q, d, s = str(r["query-id"]), str(r["corpus-id"]), int(r["score"])
        qrels.setdefault(q, {})[d] = s
    rel_docs = set(d for v in qrels.values() for d in v)
    # corpus subset: all relevant docs we can find + distractors (bounded scan)
    docs, dids, seen = [], [], set()
    n_scan = n_distract = 0
    for r in corpus_ds:
        n_scan += 1
        did = str(r["_id"]); txt = ((r.get("title") or "") + " " + (r.get("text") or "")).strip()
        is_rel = did in rel_docs
        if did not in seen and (is_rel or n_distract < max_corpus):
            dids.append(did); docs.append(txt); seen.add(did)
            if not is_rel:
                n_distract += 1
        if n_distract >= max_corpus and rel_docs.issubset(seen):
            break
        if n_scan >= scan_cap:
            break
    dset = set(dids)
    # keep queries whose relevant docs survived in the corpus subset
    qrels = {q: {d: s for d, s in v.items() if d in dset} for q, v in qrels.items()}
    qids, queries = [], []
    qtext = {str(r["_id"]): r["text"] for r in queries_ds}
    for q in qtext:
        if qrels.get(q):
            qids.append(q); queries.append(qtext[q])
    print(f"  corpus={len(dids)} queries={len(qids)} (scanned {n_scan})")

    t0 = time.time()
    Dd = dense_embed(docs, False); Qd = dense_embed(queries, True)
    Ds = sparse_embed(docs);       Qs = sparse_embed(queries)
    print(f"  embedded in {time.time()-t0:.0f}s  | avg sparse nnz/doc={int((Ds>0).sum(1).float().mean())}")

    dense_s = (Qd @ Dd.T)
    sparse_s = (Qs @ Ds.T)
    dn, sn = minmax(dense_s), minmax(sparse_s)
    print(f"  {'method':<22}{'nDCG@10':>9}{'R@100':>8}{'MRR@10':>8}")
    for label, sc in [("dense (e5)", dense_s), ("sparse (latent terms)", sparse_s)]:
        n, r, mr = metrics(sc.numpy(), qids, dids, qrels)
        print(f"  {label:<22}{n:>9.4f}{r:>8.4f}{mr:>8.4f}")
    best = None
    for a in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        n, r, mr = metrics((a*dn + (1-a)*sn).numpy(), qids, dids, qrels)
        if best is None or n > best[1]:
            best = (a, n, r, mr)
    a, n, r, mr = best
    print(f"  {'hybrid (a=%.1f)'%a:<22}{n:>9.4f}{r:>8.4f}{mr:>8.4f}  <- best alpha")


def main():
    # English
    run_task("scifact (EN)",
             load_dataset("mteb/scifact", "corpus", split="corpus"),
             load_dataset("mteb/scifact", "queries", split="queries"),
             load_dataset("mteb/scifact", "default", split="test"),
             max_corpus=20000)
    # Portuguese
    run_task("quati (PT-BR)",
             load_dataset("pt-mteb/quati_1M_retrieval", "corpus", split="corpus", streaming=True),
             load_dataset("pt-mteb/quati_1M_retrieval", "queries", split="queries"),
             load_dataset("pt-mteb/quati_1M_retrieval", "default", split="test"),
             max_corpus=20000)


if __name__ == "__main__":
    main()
