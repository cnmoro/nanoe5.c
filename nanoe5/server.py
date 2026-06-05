"""
nanoe5.server - an OpenAI-compatible embeddings server.

Run it after ``pip install nanoe5``::

    nanoe5-serve --port 8000              # console script
    python -m nanoe5.server --port 8000   # equivalent

Then use the *official* OpenAI Python client unchanged::

    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
    r = client.embeddings.create(model="e5-query", input=["how much protein per day"])

Endpoints:
    POST /v1/embeddings   string or array ``input``; ``encoding_format`` "float"|"base64"
    GET  /v1/models
    GET  /health
"""
import argparse
import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from ._core import E5

MODEL_NAMES = {
    "original": "multilingual-e5-small-q4",
    "enpt": "portuguese-multilingual-e5-small-q4",
}


def _modality(req, default_query):
    """Resolve query vs passage from input_type / model name, else default."""
    it = req.get("input_type")
    if it == "query":
        return True
    if it in ("passage", "document"):
        return False
    m = req.get("model")
    if isinstance(m, str):
        if "query" in m:
            return True
        if "passage" in m or "doc" in m:
            return False
    return default_query


class _Handler(BaseHTTPRequestHandler):
    model = None            # set by serve()
    model_name = MODEL_NAMES["original"]
    lock = threading.Lock()
    default_query = True
    protocol_version = "HTTP/1.1"

    # -- helpers -----------------------------------------------------------
    def _json(self, code, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _err(self, code, msg, typ="invalid_request_error"):
        self._json(code, {"error": {"message": msg, "type": typ, "code": None}})

    def log_message(self, *a):       # keep the server quiet
        pass

    # -- routes ------------------------------------------------------------
    def do_GET(self):
        if self.path in ("/", "/health"):
            self._json(200, {"status": "ok"})
        elif self.path == "/v1/models":
            self._json(200, {"object": "list", "data": [
                {"id": self.model_name, "object": "model", "owned_by": "nanoE5.c"}]})
        else:
            self._err(404, "unknown route")

    def do_POST(self):
        if self.path not in ("/v1/embeddings", "/embeddings"):
            self._err(404, "unknown route")
            return
        try:
            n = int(self.headers.get("Content-Length", 0) or 0)
        except ValueError:
            n = 0
        raw = self.rfile.read(n) if n > 0 else b""
        try:
            req = json.loads(raw.decode("utf-8") or "{}")
        except Exception:
            self._err(400, "invalid JSON body")
            return
        if not isinstance(req, dict):
            self._err(400, "body must be a JSON object")
            return

        inp = req.get("input")
        if inp is None:
            self._err(400, "missing 'input'")
            return
        if isinstance(inp, str):
            texts = [inp]
        elif isinstance(inp, list):
            if not all(isinstance(x, str) for x in inp):
                self._err(400, "'input' array must contain strings")
                return
            texts = inp
        else:
            self._err(400, "'input' must be a string or array of strings")
            return
        if not texts:
            self._err(400, "'input' must not be empty")
            return

        is_q = _modality(req, self.default_query)
        b64 = req.get("encoding_format") == "base64"

        try:
            with self.lock:          # one inference at a time -> all cores, no oversubscription
                embs = self.model.encode(texts, is_q)
                total = sum(self.model.token_count(t, is_q) for t in texts)
        except Exception as e:       # never crash the server on a bad request
            self._err(500, "inference failed: %s" % e, "server_error")
            return

        embs = np.atleast_2d(np.asarray(embs, dtype=np.float32))
        data = []
        for i, e in enumerate(embs):
            emb = base64.b64encode(e.tobytes()).decode("ascii") if b64 else [float(x) for x in e]
            data.append({"object": "embedding", "index": i, "embedding": emb})
        self._json(200, {
            "object": "list",
            "data": data,
            "model": self.model_name,
            "usage": {"prompt_tokens": total, "total_tokens": total},
        })


def serve(host="0.0.0.0", port=8000, default_type="query",
          num_threads=None, model_path=None, variant="original"):
    """Start the blocking OpenAI-compatible embeddings server."""
    _Handler.model = E5(model_path=model_path, num_threads=num_threads, variant=variant)
    _Handler.model_name = MODEL_NAMES.get(variant, "custom-e5-q4")
    _Handler.default_query = (default_type == "query")
    httpd = ThreadingHTTPServer((host, port), _Handler)
    print("nanoe5 OpenAI-compatible server on http://%s:%d  (dim %d, default %s, variant %s)"
          % (host, port, _Handler.model.dim, default_type, variant), flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.shutdown()


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="nanoe5-serve", description="OpenAI-compatible embeddings server (multilingual-e5-small, 4-bit)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--default-type", choices=["query", "passage"], default="query",
                    help="modality when a request doesn't specify input_type")
    ap.add_argument("--threads", type=int, default=None, help="cap OpenMP threads")
    ap.add_argument("--model", default=None, help="path to an external e5-small-q4.bin")
    ap.add_argument("--variant", choices=sorted(MODEL_NAMES), default="original",
                    help="bundled model variant to load when --model is not given")
    a = ap.parse_args(argv)
    serve(a.host, a.port, a.default_type, a.threads, a.model, a.variant)


if __name__ == "__main__":
    main()
