# nanoE5.c

**A blazing-fast, dependency-free CPU engine for [`multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small) text embeddings.**

> A tiny C core (the `.c` is the whole point) packaged for one-command use:
> `pip install nanoe5` from Python, or a single self-contained server binary.

Two 4-bit models are **bundled** — there is nothing to download or configure:

* `original` → `intfloat/multilingual-e5-small`
* `enpt` → [`cnmoro/portuguese-multilingual-e5-small`](https://huggingface.co/cnmoro/portuguese-multilingual-e5-small)

Use it from Python in two lines, or run an OpenAI-compatible server from a
single self-contained binary.

```bash
pip install nanoe5
```
```python
import nanoe5
q = nanoe5.query("how much protein per day")     # 384-dim, L2-normalized
P = nanoe5.passage(["doc a", "doc b"])           # (2, 384)
scores = P @ q                                   # cosine similarity
```

…or run an **OpenAI-compatible server** (works with the official `openai` client):

```bash
nanoe5-serve --port 8000           # OpenAI-compatible embeddings API
```

No PyTorch. No transformers. No ONNX. No BLAS. Just C, `libm`, and OpenMP.

---

## Why

* **One file to deploy.** Both bundled 4-bit models are linked *inside* the
  `./e5` binary. Copy it to a server and run — nothing to download, install, or mount.
* **Fast where it counts.** ~2 ms to embed a single query on a desktop CPU —
  about **7× faster than `sentence-transformers`** for one-at-a-time serving.
* **Tiny.** 72 MB 4-bit model vs 471 MB fp32. Instant startup (mmap).
* **Faithful.** Real XLM-RoBERTa SentencePiece tokenizer + exact BERT forward
  pass; cosine **0.98–0.99** vs the fp32 reference, retrieval rankings preserved.
* **Handles long text.** Inputs over 512 tokens are windowed automatically and
  transparently, in bounded memory.

---

## Install

### From PyPI (Python)

```bash
pip install nanoe5
```

That's it — the 4-bit model is inside the package. The tiny C engine compiles on
install (needs a C compiler with OpenMP, e.g. `gcc`), then everything runs with
**no ML dependencies** (just NumPy). Requires an x86-64 CPU with AVX2 for the
fast path; other CPUs fall back to a portable scalar build automatically.

### From source (server binary + CLI)

```bash
# 1. download + quantize the original model -> e5-small-q4.bin  (one-time, ~72 MB)
make convert        # pip install torch transformers safetensors tokenizers numpy

# optional: quantize the bundled EN+PT-pruned model -> e5-small-enpt-q4.bin
make convert-enpt

# 2a. build the self-contained server/CLI binary  ->  ./e5
make server

# 2b. (optional) build the Python shared library   ->  libe5.so
make lib
```

`make convert` / `make convert-enpt` are the only steps that touch the Python ML
stack. After that, the binary runs with **no ML dependencies at all**.

---

## Use it: the OpenAI-compatible server

Start a server with one command — **works with the official `openai` Python
client out of the box** (verified against `openai>=1.0`):

```bash
pip install nanoe5
nanoe5-serve --port 8000 --variant original   # or --variant enpt
```

```python
from openai import OpenAI                       # the official OpenAI client

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

resp = client.embeddings.create(
    model="e5-query",                           # see "Query vs passage" below
    input=["how much protein per day", "best protein sources"],
)
embeddings = [d.embedding for d in resp.data]   # two 384-dim vectors
```

Both `encoding_format="float"` and the client's default `"base64"` path are
supported, so nothing in your existing OpenAI code needs to change — just point
`base_url` at the server.

> Prefer a **single dependency-free binary**? `make server` builds `./e5`, which
> embeds the model and serves the same API with zero Python:
> `./e5 --server --port 8000`.

…or hit it with plain `curl`:

```bash
curl http://localhost:8000/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"input": ["doc one", "doc two"], "input_type": "passage"}'
```

```jsonc
{
  "object": "list",
  "data": [
    {"object": "embedding", "index": 0, "embedding": [0.031, -0.044, ...]},
    {"object": "embedding", "index": 1, "embedding": [0.018,  0.007, ...]}
  ],
  "model": "multilingual-e5-small-q4",
  "usage": {"prompt_tokens": 8, "total_tokens": 8}
}
```

### Endpoints

| Method & path | Purpose |
|---|---|
| `POST /v1/embeddings` | Create embeddings (string or array of strings). |
| `GET /v1/models` | List the served model. |
| `GET /health` | Liveness check → `{"status":"ok"}`. |

### Request fields

| Field | Values | Default |
|---|---|---|
| `input` | a string **or** an array of strings | *required* |
| `encoding_format` | `"float"` or `"base64"` | `"float"` |
| `input_type` | `"query"`, `"passage"` (alias `"document"`) | server default |
| `model` | any string; if it contains `query`/`passage`/`doc` it sets the modality | — |

`encoding_format: "base64"` returns each embedding as base64-encoded
little-endian float32 — this is what the official OpenAI Python client requests
by default, and it's fully supported.

### Server flags

Both server forms take the same flags:

```
nanoe5-serve  [--host H] [--port P] [--threads N] [--default-type query|passage] [--variant original|enpt] [--model FILE]
./e5 --server [--host H] [--port P] [--threads N] [--default-type query|passage] [--variant original|enpt] [--model FILE]
```

* `--threads N` caps OpenMP threads (default: all cores).
* `--default-type` sets the modality when a request doesn't specify one
  (default `query`).
* `--variant` selects which bundled model to serve:
  * `original` → `multilingual-e5-small-q4`
  * `enpt` → `portuguese-multilingual-e5-small-q4`
* `--model FILE` loads an external `e5-small-q4.bin` (the binary otherwise uses
  the selected bundled copy; the pip server uses the bundled one).

---

## Use it: from Python

The simplest form uses module-level helpers backed by a shared, hot model
(loaded once, reused for every call):

```python
import nanoe5

q = nanoe5.query("how much protein per day")     # (384,)
docs = nanoe5.passage([                            # (N, 384)
    "The recommended protein intake for adult women is about 46 g/day.",
    "Mount Everest is the highest mountain above sea level.",
])

scores = docs @ q          # already L2-normalized -> dot product = cosine
print(scores.argmax())     # -> 0
```

Or hold an explicit handle (e.g. to cap threads):

```python
from nanoe5 import E5
model = E5(num_threads=8, variant="enpt")      # or variant="original"
model.query("...");  model.passage(["...", "..."])
```

You can also select the variant through the module helpers:

```python
import nanoe5

q = nanoe5.query("quanta proteína por dia", variant="enpt")
```

That's the whole API:

| Call | Prefix added | Returns |
|---|---|---|
| `nanoe5.query(text \| list)` / `model.query(...)` | `query: ` | `(384,)` or `(N, 384)` `float32` |
| `nanoe5.passage(text \| list)` / `model.passage(...)` | `passage: ` | `(384,)` or `(N, 384)` `float32` |
| `nanoe5.encode(x, is_query=False)` / `model.encode(...)` | either | generic form |

A single text is parallelized across all CPU cores (low latency); a list is
parallelized across texts (high throughput).

### Bundled variants

* `variant="original"` keeps the full multilingual tokenizer and weights from
  `intfloat/multilingual-e5-small`.
* `variant="enpt"` uses the bundled pruned tokenizer/weights from
  `cnmoro/portuguese-multilingual-e5-small`, keeping English + Portuguese tokens.

If `model_path=...` is given, it overrides `variant`.

---

## Query vs passage

`multilingual-e5-small` is trained with two prefixes, and you should use the
right one:

* **`query:`** — short search queries / questions.
* **`passage:`** — documents you want to retrieve.

Embed your documents with `passage`, your search queries with `query`, then rank
documents by cosine similarity (a plain dot product, since outputs are
normalized).

* **Python:** `model.query(...)` vs `model.passage(...)`.
* **Server:** set `"input_type": "query"` or `"passage"` per request (or name
  the model `e5-query` / `e5-passage`), otherwise the server's `--default-type`
  is used.

---

## Long inputs (automatic)

The base model maxes out at 512 tokens. Instead of truncating, nanoE5.c slides a
window over longer text: it splits into ≤510-token windows, embeds each, and
returns the **token-count-weighted average** (then re-normalizes). This is
mathematically equivalent to mean-pooling over the whole document and needs **no
API change** — just pass a long string. Memory stays bounded (~350 MB) even for
million-token inputs.

---

## Sparse "latent terms" & hybrid retrieval (optional)

A single dense vector has a fixed capacity; a high‑dimensional **sparse** vector
can encode complementary lexical signal and improves recall. Inspired by
mixedbread's [*latent terms*](https://www.mixedbread.com/blog/latent-terms),
nanoE5.c can attach a **sparse head**: a TopK sparse autoencoder (`sae.bin`,
3.6 MB) trained on e5 token embeddings that maps each token to a 16,384‑dim
sparse code, max‑pooled over the document. It was trained on **Portuguese +
English only**.

```python
from nanoe5 import E5
m = E5()                          # auto-loads the bundled sae.bin
m.has_sparse                      # True
v = m.sparse("quanta proteína por dia")          # dense (16384,) float32 (default)
S = m.sparse(docs, fmt="scipy")                  # (N, 16384) scipy.sparse.csr_matrix
i, w = m.sparse(text, fmt="indices")             # raw (feature_id, weight) arrays
```

`m.sparse(...)` returns a **numpy array by default** — `(sparse_dim,)` for one
text, `(N, sparse_dim)` for a list — or a `scipy.sparse.csr_matrix` with
`fmt="scipy"` (use this to index large corpora).

On standard benchmarks, **hybrid (dense + sparse) beats dense alone** — small but
consistent across both languages (best at dense‑weight ≈ 0.8):

| | nDCG@10 | Recall@100 |
|---|---|---|
| scifact (EN) dense | 0.654 | 0.917 |
| scifact (EN) **hybrid** | **0.668** | **0.930** |
| quati (PT‑BR) dense | 0.387 | 0.796 |
| quati (PT‑BR) **hybrid** | **0.392** | **0.816** |

### How to use it in a retrieval pipeline

Two patterns — pick based on whether you want better **ranking** or better **recall**:

**1. Hybrid retrieval (recommended — improves recall).** Index *both*
representations and fuse at query time. Sparse catches exact/rare‑term matches
the dense vector structurally cannot.

```python
import numpy as np

# --- index time ---
D = m.passage(docs)                  # (N, 384)  dense
S = m.sparse(docs, fmt="scipy")      # (N, 16384) sparse  (use "numpy" for small corpora)

# --- query time ---
qd, qs = m.query(query), m.sparse(query)
dense  = D @ qd                      # cosine (vectors are normalized)
sparse = np.asarray(S @ qs).ravel()  # sparse dot
def mm(x): return (x - x.min()) / (np.ptp(x) + 1e-9)
score  = 0.8 * mm(dense) + 0.2 * mm(sparse)       # or Reciprocal Rank Fusion
```

At scale, put dense in an ANN index (HNSW/FAISS) and the sparse vectors in an
**inverted index** (feature_id → postings); the SAE feature ids behave like
terms. Both are first‑stage retrievers whose candidate sets you union, then fuse.

**2. Dense‑retrieve → sparse rerank (cheaper, improves ranking only).** Take the
dense top‑K, re‑score those K with `0.8·dense + 0.2·sparse`, reorder. This is
what you proposed and it's the lightest option — but note a reranker can only
reorder what dense already found, so it improves ordering, **not recall**. Most
of the measured gain above is in Recall@100, which needs pattern 1.

> The sparse head is optional: without `sae.bin`, `m.has_sparse` is `False` and
> everything else works unchanged. Retrain it with `python sae_train.py` (uses a
> GPU; PT+EN corpus only) and evaluate with `python sae_eval.py`.

---

## CLI

The same binary is also a quick CLI:

```bash
./e5 query   "how much protein should a female eat"
./e5 --variant enpt query "quanta proteína devo comer por dia"
./e5 passage "a document to index"
./e5 --model e5-small-q4.bin query "use an external model file"
```

---

## How it works (short version)

* **4-bit weights (Q4_0).** Every large matrix is stored in 32-weight blocks
  with an fp16 scale (~4.5 bits/weight) — ~10× less memory traffic than fp32.
* **int8 × int4 matmul.** Activations are quantized to int8 and multiplied
  against the 4-bit weights with AVX2 integer MACs — no fp32 dequant in the hot
  loop. Scalar fallback included for non-AVX CPUs.
* **One pass per batch.** All tokens of a batch share a single matmul per layer,
  so weights stream once; attention runs per text.
* **OpenMP** across matrix rows / texts; deterministic regardless of thread count.
* **Faithful tokenizer.** XLM-RoBERTa SentencePiece-unigram (Viterbi) with the
  real Precompiled normalizer baked in as a per-codepoint table.

The model is packed into one binary blob by `convert.py`; `e5.c` is the entire
engine (loader, tokenizer, BERT, quantized matmul); `server.c` adds the HTTP
server and CLI; `e5.py` is the ctypes wrapper.

---

## Performance

On a Ryzen 7 5800X3D (8 cores / 16 threads, AVX2):

| | nanoE5.c (4-bit) | sentence-transformers (fp32) |
|---|---|---|
| single-query latency (hot) | **~2 ms** | ~13 ms |
| batch throughput | ~190–340 texts/s | ~280 texts/s |
| model size | **72 MB** | 471 MB |
| dependencies | libc, libm, OpenMP | torch + transformers |
| cold start | instant (mmap) | seconds |

For online serving (one query at a time, model hot) nanoE5.c is **~7× faster**
per call. For huge offline batch jobs, PyTorch's oneDNN GEMM edges ahead on raw
throughput — but at 1/6th the footprint and zero dependencies.

---

## Validate & stress

```bash
make test                                   # original parity vs HF reference + speed
E5_SRC=hf_src_enpt E5_MODEL=e5-small-enpt-q4.bin E5_VARIANT=enpt E5_TEXTSET=enpt python3 test_parity.py
python3 test_variants.py                    # constructor/server variant wiring
python3 bench_variants.py                   # original vs enpt speed
env E5_VARIANT=original python3 stress_test.py
env E5_VARIANT=enpt python3 stress_test.py
```

`make stress` throws adversarial inputs at every layer and asserts: no crashes,
no hangs, finite & unit-norm outputs, determinism, `batch == single` (exact),
`server == binding` parity, `base64 == float` parity, real OpenAI-client
compatibility, correct `4xx` handling for malformed requests, survival of a raw
garbage barrage, and concurrent requests with zero errors or races.

---

## Files

```
e5.c / e5.h      the entire inference engine
server.c         OpenAI-compatible HTTP server + CLI
convert.py       build e5-small-q4.bin / e5-small-enpt-q4.bin from HF checkpoints
sae_train.py     train the sparse "latent terms" head -> sae.bin (PT+EN, GPU)
sae_eval.py      dense vs sparse vs hybrid retrieval eval (scifact + quati)
nanoe5/          the pip package (engine + both 4-bit models + sae.bin bundled)
pyproject.toml   / setup.py   packaging (compiles the engine, bundles the model)
e5.py            standalone ctypes wrapper (repo-local use)
test_parity.py   parity vs HF reference + benchmark
stress_test.py   hard stress / edge-case suite
Makefile
```

## License

The code here is yours to use. The bundled model weights are
`intfloat/multilingual-e5-small` (MIT) and
`cnmoro/portuguese-multilingual-e5-small` (MIT-compatible derivative of the same base) —
see the model cards for details.
