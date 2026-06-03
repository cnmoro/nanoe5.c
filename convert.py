#!/usr/bin/env python3
"""
convert.py  -  Pack intfloat/multilingual-e5-small into a single self-contained
               4-bit binary that the C engine (e5.c) consumes.

The output file `e5-small-q4.bin` contains EVERYTHING needed for inference:
  * model hyper-parameters
  * the full SentencePiece-unigram vocabulary (pieces + scores)
  * a per-codepoint normalization table dumped from the *real* HF normalizer
    (Precompiled NFKC-ish charsmap), so the C tokenizer is byte-faithful
  * every weight tensor, with the big matrices stored as Q4_0 (4-bit) blocks

No transformers / sentence-transformers code is needed at inference time.
"""

import json, os, struct
import numpy as np

SRC   = os.environ.get("E5_SRC", "hf_src")
OUT   = os.environ.get("E5_OUT", "e5-small-q4.bin")
QK    = 32                      # Q4_0 block size
MAGIC = b"E5S1"
ALIGN = 64

# ----------------------------------------------------------------------------
# Q4_0 quantization (identical layout to llama.cpp: fp16 scale + 32 nibbles)
#   block = { float16 d; uint8 qs[16]; }   -> 18 bytes / 32 weights
#   nibble pairing: byte j holds weight j (low) and weight j+16 (high)
# ----------------------------------------------------------------------------
_BLK = np.dtype([("d", "<f2"), ("qs", "u1", 16)])

def quantize_q4_0(a: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(a, dtype=np.float32).reshape(-1)
    if a.size % QK:
        raise ValueError(f"size {a.size} not divisible by {QK}")
    a = a.reshape(-1, QK)
    nb = a.shape[0]
    absa = np.abs(a)
    idx = np.argmax(absa, axis=1)
    amax = a[np.arange(nb), idx]                # signed value at max-abs position
    d = amax / -8.0
    inv = np.where(d != 0.0, 1.0 / d, 0.0).astype(np.float32)
    q = np.floor(a * inv[:, None] + 8.5)
    q = np.clip(q, 0, 15).astype(np.uint8)      # [nb, 32], values 0..15
    lo = q[:, :16]
    hi = q[:, 16:]
    qs = (lo | (hi << 4)).astype(np.uint8)
    out = np.empty(nb, dtype=_BLK)
    out["d"] = d.astype(np.float16)
    out["qs"] = qs
    return out

# ----------------------------------------------------------------------------
# helpers for writing the binary
# ----------------------------------------------------------------------------
class Blob:
    def __init__(self):
        self.tensors = []   # (name, dtype_id, n_logical, bytes)
    def add_q4(self, name, arr):
        b = quantize_q4_0(arr).tobytes()
        self.tensors.append((name, 1, int(arr.size), b))
    def add_f32(self, name, arr):
        a = np.ascontiguousarray(arr, dtype=np.float32)
        self.tensors.append((name, 0, int(a.size), a.tobytes()))

def w16(f, v): f.write(struct.pack("<H", v))
def w32(f, v): f.write(struct.pack("<i", v))
def wu32(f, v): f.write(struct.pack("<I", v))
def wu64(f, v): f.write(struct.pack("<Q", v))
def wf32(f, v): f.write(struct.pack("<f", v))
def wstr(f, s):
    b = s.encode("utf-8")
    w16(f, len(b)); f.write(b)

# ----------------------------------------------------------------------------
def main():
    cfg = json.load(open(os.path.join(SRC, "config.json")))
    H   = cfg["hidden_size"]
    NL  = cfg["num_hidden_layers"]
    NH  = cfg["num_attention_heads"]
    FF  = cfg["intermediate_size"]
    VOC = cfg["vocab_size"]
    MAXP = cfg["max_position_embeddings"]
    TVS = cfg["type_vocab_size"]
    EPS = float(cfg["layer_norm_eps"])
    print(f"[cfg] H={H} layers={NL} heads={NH} ffn={FF} vocab={VOC} maxpos={MAXP}")

    # ---- weights -----------------------------------------------------------
    from safetensors import safe_open
    st = safe_open(os.path.join(SRC, "model.safetensors"), "numpy")
    keys = set(st.keys())
    def get(name):
        # tolerate optional "embeddings."/"bert." prefixes
        for cand in (name, "bert." + name, "embeddings." + name):
            if cand in keys:
                return st.get_tensor(cand).astype(np.float32)
        raise KeyError(name + f"  (have e.g. {list(keys)[:4]})")

    blob = Blob()
    # token / position / type embeddings + embedding LayerNorm
    blob.add_q4 ("emb.word", get("embeddings.word_embeddings.weight"))     # [VOC,H]
    blob.add_f32("emb.pos",  get("embeddings.position_embeddings.weight")) # [MAXP,H]
    blob.add_f32("emb.type", get("embeddings.token_type_embeddings.weight"))# [TVS,H]
    blob.add_f32("emb.ln.w", get("embeddings.LayerNorm.weight"))
    blob.add_f32("emb.ln.b", get("embeddings.LayerNorm.bias"))

    for i in range(NL):
        p = f"encoder.layer.{i}."
        blob.add_q4 (f"l{i}.q.w",  get(p+"attention.self.query.weight"))
        blob.add_f32(f"l{i}.q.b",  get(p+"attention.self.query.bias"))
        blob.add_q4 (f"l{i}.k.w",  get(p+"attention.self.key.weight"))
        blob.add_f32(f"l{i}.k.b",  get(p+"attention.self.key.bias"))
        blob.add_q4 (f"l{i}.v.w",  get(p+"attention.self.value.weight"))
        blob.add_f32(f"l{i}.v.b",  get(p+"attention.self.value.bias"))
        blob.add_q4 (f"l{i}.ao.w", get(p+"attention.output.dense.weight"))
        blob.add_f32(f"l{i}.ao.b", get(p+"attention.output.dense.bias"))
        blob.add_f32(f"l{i}.aln.w",get(p+"attention.output.LayerNorm.weight"))
        blob.add_f32(f"l{i}.aln.b",get(p+"attention.output.LayerNorm.bias"))
        blob.add_q4 (f"l{i}.fi.w", get(p+"intermediate.dense.weight"))      # [FF,H]
        blob.add_f32(f"l{i}.fi.b", get(p+"intermediate.dense.bias"))
        blob.add_q4 (f"l{i}.fo.w", get(p+"output.dense.weight"))            # [H,FF]
        blob.add_f32(f"l{i}.fo.b", get(p+"output.dense.bias"))
        blob.add_f32(f"l{i}.oln.w",get(p+"output.LayerNorm.weight"))
        blob.add_f32(f"l{i}.oln.b",get(p+"output.LayerNorm.bias"))
    print(f"[weights] packed {len(blob.tensors)} tensors")

    # ---- tokenizer ---------------------------------------------------------
    tj = json.load(open(os.path.join(SRC, "tokenizer.json")))
    model = tj["model"]
    assert model["type"] == "Unigram", model["type"]
    vocab = model["vocab"]                       # list of [piece, score]
    unk_id = int(model.get("unk_id", 3))
    pieces = [p for p, _ in vocab]
    scores = np.array([s for _, s in vocab], dtype=np.float32)
    # special ids (XLM-R): <s>=0 <pad>=1 </s>=2 <unk>=3
    sid = {p: i for i, p in enumerate(pieces)}
    BOS = sid["<s>"]; EOS = sid["</s>"]; PAD = sid["<pad>"]; UNK = unk_id
    print(f"[tok] vocab={len(pieces)} bos={BOS} eos={EOS} pad={PAD} unk={UNK}")

    # per-codepoint normalization table from the REAL normalizer
    from tokenizers import Tokenizer
    norm = Tokenizer.from_file(os.path.join(SRC, "tokenizer.json")).normalizer
    nmap = []   # (cp, [out_cp,...])
    for cp in range(0x0, 0x30000):
        if 0xD800 <= cp <= 0xDFFF:
            continue
        s = chr(cp)
        r = norm.normalize_str(s)
        if r != s:
            nmap.append((cp, [ord(c) for c in r]))
    print(f"[tok] normalization entries: {len(nmap)}")

    # ---- write file --------------------------------------------------------
    with open(OUT, "wb") as f:
        f.write(MAGIC); wu32(f, 1)               # magic + version
        for v in (H, NL, NH, FF, VOC, MAXP, TVS): w32(f, v)
        wf32(f, EPS); w32(f, QK)

        # tokenizer header
        w32(f, len(pieces)); w32(f, BOS); w32(f, EOS); w32(f, PAD); w32(f, UNK)
        # pieces blob: concat with offsets + lengths
        blobbytes = b"".join(p.encode("utf-8") for p in pieces)
        offs = np.zeros(len(pieces) + 1, dtype=np.uint32)
        acc = 0
        for i, p in enumerate(pieces):
            offs[i] = acc; acc += len(p.encode("utf-8"))
        offs[len(pieces)] = acc
        wu64(f, len(blobbytes))
        f.write(offs.tobytes())                  # (n+1) uint32 offsets
        f.write(scores.tobytes())                # n float32 scores
        f.write(blobbytes)
        # normalization map
        w32(f, len(nmap))
        nm = bytearray()
        for cp, outs in nmap:
            nm += struct.pack("<IB", cp, len(outs))
            for o in outs: nm += struct.pack("<I", o)
        wu32(f, len(nm)); f.write(bytes(nm))

        # tensor directory
        w32(f, len(blob.tensors))
        # first pass: compute data offsets (aligned). We need to know where the
        # data region starts, which depends on the directory size -> two passes.
        # Build directory entries with placeholder offsets, measure, then fix.
        # Simpler: write directory with sizes, then a marker, then aligned data.
        dir_start = f.tell()
        # reserve directory: name + dtype + n_logical + offset(u64) + nbytes(u64)
        for name, dt, n, b in blob.tensors:
            wstr(f, name); f.write(struct.pack("<B", dt)); wu32(f, n)
            wu64(f, 0); wu64(f, len(b))           # offset patched later
        # align data region
        pos = f.tell()
        pad = (-pos) % ALIGN
        f.write(b"\x00" * pad)
        data_start = f.tell()
        # write data, recording offsets
        offsets = []
        for name, dt, n, b in blob.tensors:
            pos = f.tell()
            pad = (-pos) % ALIGN
            f.write(b"\x00" * pad)
            offsets.append(f.tell())
            f.write(b)
        end = f.tell()
        # patch directory offsets
        f.seek(dir_start)
        for (name, dt, n, b), off in zip(blob.tensors, offsets):
            wstr(f, name); f.write(struct.pack("<B", dt)); wu32(f, n)
            wu64(f, off); wu64(f, len(b))
        f.seek(end)

    print(f"[out] wrote {OUT}  ({os.path.getsize(OUT)/1e6:.1f} MB)")

if __name__ == "__main__":
    main()
