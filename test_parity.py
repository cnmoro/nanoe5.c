"""
test_parity.py - validate the C engine against the reference HF model and
                 measure throughput.

  * tokenizer parity vs the fast HF tokenizer
  * embedding cosine similarity vs transformers (fp32 reference)
  * retrieval sanity (query vs passages)
  * throughput benchmark
"""
import os, time, sys
import numpy as np
from e5 import E5

SRC = os.environ.get("E5_SRC", "hf_src")
MODEL_PATH = os.environ.get("E5_MODEL")
VARIANT = os.environ.get("E5_VARIANT", "original")
TEXTSET = os.environ.get("E5_TEXTSET", "multilingual")


def _timeit(fn):
    t0 = time.time(); fn(); return time.time() - t0

TEXTS_BY_SET = {
    "multilingual": [
        "how much protein should a female eat",
        "Como funciona a fotossíntese nas plantas?",
        "The quick brown fox jumps over the lazy dog.",
        "机器学习是人工智能的一个分支。",
        "naïve café — ½ + ² = ?   multiple   spaces\tand\ttabs",
        "Was ist die Hauptstadt von Deutschland?",
    ],
    "enpt": [
        "how much protein should a female eat",
        "How do I cook rice without making it sticky?",
        "Como funciona a fotossíntese nas plantas?",
        "Quais alimentos têm mais proteína por porção?",
        "naïve café — ½ + ² = ?   multiple   spaces\tand\ttabs",
        "What is the capital of Germany?",
    ],
}
TEXTS = TEXTS_BY_SET[TEXTSET]
LONGS_BY_SET = {
    "multilingual": [
        ("Photosynthesis is the process used by plants to convert light. " * 60, False),
        ("机器学习是人工智能的一个分支。" * 100, False),
    ],
    "enpt": [
        ("Photosynthesis is the process used by plants to convert light. " * 60, False),
        ("A fotossíntese permite que as plantas transformem luz em energia. " * 80, False),
    ],
}


def ref_setup():
    import torch
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(SRC)
    mdl = AutoModel.from_pretrained(SRC).eval()
    torch.set_grad_enabled(False)

    def avg_pool(last, mask):
        last = last.masked_fill(~mask[..., None].bool(), 0.0)
        return last.sum(1) / mask.sum(1)[..., None]

    def embed(texts, is_query):
        pfx = "query: " if is_query else "passage: "
        batch = tok([pfx + t for t in texts], padding=True, truncation=True,
                    max_length=512, return_tensors="pt")
        out = mdl(**batch)
        emb = avg_pool(out.last_hidden_state, batch["attention_mask"])
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        return emb.numpy()
    return tok, embed, mdl


def _ref_window(tok, mdl, text, is_query, WMAX=510):
    """fp32 reference implementing the SAME sliding-window scheme as the engine."""
    import torch
    pfx = ("query: " if is_query else "passage: ") + text
    enc = tok(pfx, add_special_tokens=False)["input_ids"]
    bos, eos = tok.cls_token_id, tok.sep_token_id
    acc = np.zeros(384, np.float64); wsum = 0
    for a in range(0, max(1, len(enc)), WMAX):
        chunk = enc[a:a + WMAX]
        t = torch.tensor([[bos] + chunk + [eos]])
        out = mdl(input_ids=t, attention_mask=torch.ones_like(t)).last_hidden_state[0]
        w = max(1, len(chunk)); acc += w * out.mean(0).numpy(); wsum += w
    v = acc / wsum
    return (v / np.linalg.norm(v)).astype(np.float32)


def main():
    tok, ref_embed, ref_setup_model = ref_setup()
    m = E5(model_path=MODEL_PATH, variant=VARIANT)

    # ---- embedding parity (also validates the tokenizer end-to-end) -----
    print("== embedding cosine vs fp32 reference ==")
    worst = 1.0
    for is_q, label in [(True, "query"), (False, "passage")]:
        R = ref_embed(TEXTS, is_q)
        E = m.encode(TEXTS, is_q)
        for i, t in enumerate(TEXTS):
            cos = float(np.dot(R[i], E[i]))
            worst = min(worst, cos)
            flag = "" if cos > 0.985 else "  <-- check"
            print(f"  [{label}] cos={cos:.5f}{flag}  | {t[:42]!r}")
    print(f"  worst cosine = {worst:.5f}  (4-bit quantization; ~0.98-0.99 expected)")

    # ---- retrieval sanity ----------------------------------------------
    print("== retrieval sanity ==")
    q = m.query("how much protein should a female eat")
    docs = [
        "As a general guideline, the CDC's average requirement of protein for "
        "women ages 19 to 70 is 46 grams per day.",
        "Definition of summit for English Language Learners: the highest point "
        "of a mountain.",
    ]
    P = m.passage(docs)
    scores = (P @ q)
    print("  scores:", [round(float(s), 4) for s in scores],
          "-> best doc #", int(np.argmax(scores)))

    # ---- sliding window for >512-token inputs ---------------------------
    print("== sliding window (long inputs) ==")
    longs = LONGS_BY_SET[TEXTSET]
    win_worst = 1.0
    for txt, isq in longs:
        ntok = len(tok(("query: " if isq else "passage: ") + txt,
                       add_special_tokens=False)["input_ids"])
        ref = _ref_window(tok, ref_setup_model, txt, isq)
        eng = m.query(txt) if isq else m.passage(txt)
        cos = float(np.dot(ref, eng))
        win_worst = min(win_worst, cos)
        print(f"  cos={cos:.5f}  tokens={ntok:5d}  windows={ntok // 510 + 1}")
    print(f"  worst windowed cosine = {win_worst:.5f} (vs fp32 reference doing the same scheme)")

    # ---- speed ----------------------------------------------------------
    print("== speed ==")
    qtext = "how much protein should a female eat per day"
    for _ in range(5):
        m.query(qtext)
    lat = []
    for _ in range(50):
        t0 = time.time(); m.query(qtext); lat.append(time.time() - t0)
    print(f"  single-query latency (hot model): {np.median(lat)*1000:.2f} ms")
    raw = ["the cat sat on the mat number %d and learned to embed text" % i
           for i in range(512)]
    for _ in range(3):
        m.passage(raw[:64])                       # warm
    best = min(_timeit(lambda: m.passage(raw)) for _ in range(4))
    print(f"  batch throughput (512 passages): {512/best:.0f} texts/s")

    ok = worst > 0.98          # 4-bit Q4_0 tolerance vs the fp32 reference
    print("\nRESULT:", "PASS" if ok else "FAIL (cosine below 4-bit tolerance)")
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
