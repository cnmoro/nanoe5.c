# nanoE5.c

**A blazing-fast, dependency-free CPU engine for [`multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small) text embeddings.**

> A tiny C core (the `.c` is the whole point) packaged for one-command use:
> `pip install nanoe5` from Python, or a single self-contained server binary.

The 4-bit model is **bundled** — there is nothing to download or configure.
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

…or the server, one file and zero dependencies:

```bash
./e5 --server --port 8000          # OpenAI-compatible embeddings API
```

No PyTorch. No transformers. No ONNX. No BLAS. Just C, `libm`, and OpenMP.

---

## Why

* **One file to deploy.** The 4-bit model is linked *inside* the `./e5` binary
  (~69 MB). Copy it to a server and run — nothing to download, install, or mount.
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
# 1. download + quantize the model -> e5-small-q4.bin  (one-time, ~72 MB)
make convert        # pip install torch transformers safetensors tokenizers numpy

# 2a. build the self-contained server/CLI binary  ->  ./e5
make server

# 2b. (optional) build the Python shared library   ->  libe5.so
make lib
```

`make convert` is the only step that touches the Python ML stack. After it, the
binary runs with **no ML dependencies at all**.

---

## Use it: the server

Start it (the model is already inside the binary):

```bash
./e5 --server --host 0.0.0.0 --port 8000
```

It speaks the **OpenAI embeddings API**, so any OpenAI client works unchanged:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

resp = client.embeddings.create(
    model="e5-query",                       # see "Query vs passage" below
    input=["how much protein per day", "best protein sources"],
)
embeddings = [d.embedding for d in resp.data]   # two 384-dim vectors
```

…or just `curl`:

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

```
./e5 --server [--host H] [--port P] [--threads N]
              [--default-type query|passage] [--model FILE]
```

* `--threads N` caps OpenMP threads (default: all cores).
* `--default-type` sets the modality when a request doesn't specify one
  (default `query`).
* `--model FILE` loads an external `e5-small-q4.bin` instead of the embedded one.

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
model = E5(num_threads=8)
model.query("...");  model.passage(["...", "..."])
```

That's the whole API:

| Call | Prefix added | Returns |
|---|---|---|
| `nanoe5.query(text \| list)` / `model.query(...)` | `query: ` | `(384,)` or `(N, 384)` `float32` |
| `nanoe5.passage(text \| list)` / `model.passage(...)` | `passage: ` | `(384,)` or `(N, 384)` `float32` |
| `nanoe5.encode(x, is_query=False)` / `model.encode(...)` | either | generic form |

A single text is parallelized across all CPU cores (low latency); a list is
parallelized across texts (high throughput).

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

## CLI

The same binary is also a quick CLI:

```bash
./e5 query   "how much protein should a female eat"
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
make test     # cosine parity vs the fp32 HF reference + speed
make stress   # hard edge-case / concurrency / server suite
```

`make stress` throws adversarial inputs at every layer and asserts: no crashes,
no hangs, finite & unit-norm outputs, determinism, `batch == single` (exact),
`server == binding` parity, `base64 == float` parity, real OpenAI-client
compatibility, correct `4xx` handling for malformed requests, survival of a raw
garbage barrage, and 400 concurrent requests with zero errors or races.

---

## Files

```
e5.c / e5.h      the entire inference engine
server.c         OpenAI-compatible HTTP server + CLI
convert.py       build e5-small-q4.bin from the HF checkpoint (one-time)
nanoe5/          the pip package (engine + 4-bit model bundled)
pyproject.toml   / setup.py   packaging (compiles the engine, bundles the model)
e5.py            standalone ctypes wrapper (repo-local use)
test_parity.py   parity vs HF reference + benchmark
stress_test.py   hard stress / edge-case suite
Makefile
```

## License

The code here is yours to use. The model weights are
`intfloat/multilingual-e5-small` (MIT) — see the model card for details.
