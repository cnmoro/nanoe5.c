"""
nanoe5._core - ctypes binding to the compiled C engine.

Both the compiled engine (``_engine*.so``) and the 4-bit model
(``e5-small-q4.bin``) are bundled inside this package, so there is nothing to
download or configure: construct :class:`E5` and embed.
"""
import ctypes
import glob
import os

import numpy as np

_PKG = os.path.dirname(os.path.abspath(__file__))
_MODEL = os.path.join(_PKG, "e5-small-q4.bin")


def _find_engine():
    pats = ("_engine*.so", "_engine*.pyd", "_engine*.dylib", "_engine*.dll")
    for pat in pats:
        hits = glob.glob(os.path.join(_PKG, pat))
        if hits:
            return hits[0]
    raise ImportError(
        "nanoe5: compiled engine not found in %s. "
        "Reinstall the package (it must be built with a C compiler)." % _PKG
    )


class E5:
    """Hot, in-process embedder for multilingual-e5-small (4-bit).

    >>> from nanoe5 import E5
    >>> m = E5()
    >>> q = m.query("how much protein per day")        # (384,) float32, L2-normalized
    >>> P = m.passage(["doc a", "doc b"])              # (2, 384)
    >>> scores = P @ q
    """

    def __init__(self, model_path=None, lib_path=None, num_threads=None):
        model_path = model_path or _MODEL
        lib_path = lib_path or _find_engine()
        if not os.path.exists(model_path):
            raise FileNotFoundError("nanoe5: model file not found: %s" % model_path)
        if num_threads:
            os.environ["OMP_NUM_THREADS"] = str(int(num_threads))

        lib = ctypes.CDLL(lib_path)
        lib.e5_load.restype = ctypes.c_void_p
        lib.e5_load.argtypes = [ctypes.c_char_p]
        lib.e5_free.argtypes = [ctypes.c_void_p]
        lib.e5_dim.restype = ctypes.c_int
        lib.e5_dim.argtypes = [ctypes.c_void_p]
        lib.e5_embed.restype = ctypes.c_int
        lib.e5_embed.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int,
                                 ctypes.POINTER(ctypes.c_float)]
        lib.e5_embed_batch.restype = ctypes.c_int
        lib.e5_embed_batch.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_char_p),
                                       ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_float)]
        self._lib = lib
        self._m = lib.e5_load(model_path.encode())
        if not self._m:
            raise RuntimeError("nanoe5: failed to load model")
        self.dim = lib.e5_dim(self._m)

    def _encode(self, texts, is_query):
        single = isinstance(texts, str)
        if single:
            texts = [texts]
        else:
            texts = list(texts)
        n = len(texts)
        out = np.empty((max(n, 1), self.dim), dtype=np.float32)
        if n == 0:
            return out[:0]
        outp = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        if n == 1:
            self._lib.e5_embed(self._m, texts[0].encode("utf-8"), int(is_query), outp)
        else:
            arr = (ctypes.c_char_p * n)(*[t.encode("utf-8") for t in texts])
            self._lib.e5_embed_batch(self._m, arr, n, int(is_query), outp)
        return out[0] if single else out[:n]

    def query(self, texts):
        """Embed text(s) with the ``query: `` prefix."""
        return self._encode(texts, True)

    def passage(self, texts):
        """Embed text(s) with the ``passage: `` prefix."""
        return self._encode(texts, False)

    def encode(self, texts, is_query=False):
        """Generic embed; ``is_query`` selects the prefix."""
        return self._encode(texts, is_query)

    def __del__(self):
        m = getattr(self, "_m", None)
        if m:
            self._lib.e5_free(m)
            self._m = None
