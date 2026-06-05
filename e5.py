"""
e5.py - thin ctypes wrapper around libe5.so.

Load the model ONCE; keep it hot in RAM and call `.query()` / `.passage()`
as many times as you like with zero reload overhead. The heavy lifting
(tokenization + 4-bit BERT forward pass) happens entirely in C.

    from e5 import E5
    model = E5()                                  # loads e5-small-q4.bin, stays hot
    q = model.query("how much protein per day")   # (384,) float32, L2-normalized
    P = model.passage(["doc a", "doc b", "doc c"]) # (3, 384)
    scores = P @ q                                 # cosine similarities
"""
import ctypes as C
import os
import numpy as np

_DIR = os.path.dirname(os.path.abspath(__file__))
_MODELS = {
    "original": os.path.join(_DIR, "e5-small-q4.bin"),
    "enpt": os.path.join(_DIR, "e5-small-enpt-q4.bin"),
}


class E5:
    def __init__(self, model_path=None, lib_path=None, num_threads=None, variant="original"):
        if model_path is None:
            if variant not in _MODELS:
                raise ValueError(f"unknown variant {variant!r}; expected one of {sorted(_MODELS)}")
            model_path = _MODELS[variant]
        lib_path = lib_path or os.path.join(_DIR, "libe5.so")
        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"{lib_path} not found - run `make lib`")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"{model_path} not found - run `make convert`")
        if num_threads:
            os.environ["OMP_NUM_THREADS"] = str(num_threads)

        lib = C.CDLL(lib_path)
        lib.e5_load.restype = C.c_void_p
        lib.e5_load.argtypes = [C.c_char_p]
        lib.e5_free.argtypes = [C.c_void_p]
        lib.e5_dim.restype = C.c_int
        lib.e5_dim.argtypes = [C.c_void_p]
        lib.e5_embed.restype = C.c_int
        lib.e5_embed.argtypes = [C.c_void_p, C.c_char_p, C.c_int,
                                 C.POINTER(C.c_float)]
        lib.e5_embed_batch.restype = C.c_int
        lib.e5_embed_batch.argtypes = [C.c_void_p, C.POINTER(C.c_char_p),
                                       C.c_int, C.c_int, C.POINTER(C.c_float)]
        self._lib = lib
        self._m = lib.e5_load(model_path.encode())
        if not self._m:
            raise RuntimeError("e5_load failed")
        self.dim = lib.e5_dim(self._m)

    def _encode(self, texts, is_query):
        single = isinstance(texts, str)
        if single:
            texts = [texts]
        n = len(texts)
        out = np.empty((n, self.dim), dtype=np.float32)
        outp = out.ctypes.data_as(C.POINTER(C.c_float))
        if n == 1:
            self._lib.e5_embed(self._m, texts[0].encode("utf-8"),
                               int(is_query), outp)
        else:
            arr = (C.c_char_p * n)(*[t.encode("utf-8") for t in texts])
            self._lib.e5_embed_batch(self._m, arr, n, int(is_query), outp)
        return out[0] if single else out

    def query(self, texts):
        """Embed text(s) with the 'query: ' prefix."""
        return self._encode(texts, True)

    def passage(self, texts):
        """Embed text(s) with the 'passage: ' prefix."""
        return self._encode(texts, False)

    # generic alias
    def encode(self, texts, is_query=False):
        return self._encode(texts, is_query)

    def __del__(self):
        if getattr(self, "_m", None):
            self._lib.e5_free(self._m)
            self._m = None


if __name__ == "__main__":
    m = E5()
    q = m.query("how much protein should a female eat")
    docs = [
        "As a general guideline, the CDC's average requirement of protein for "
        "women ages 19 to 70 is 46 grams per day.",
        "Definition of summit for English Language Learners: the highest point "
        "of a mountain.",
    ]
    P = m.passage(docs)
    print("dim:", m.dim)
    print("scores:", (P @ q).tolist())
