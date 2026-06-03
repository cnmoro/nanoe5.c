"""
nanoE5.c - blazing-fast 4-bit CPU embeddings for multilingual-e5-small.

Quick start::

    import nanoe5
    q = nanoe5.query("how much protein per day")     # (384,) float32, normalized
    P = nanoe5.passage(["doc a", "doc b"])           # (2, 384)
    scores = P @ q                                    # cosine similarity

or keep an explicit handle::

    from nanoe5 import E5
    model = E5()
    model.query("...");  model.passage([...])

Everything (engine + 4-bit model) ships inside this package; nothing to
download. Use ``query`` for search queries and ``passage`` for documents.
"""
from ._core import E5

__version__ = "0.1.0"
__all__ = ["E5", "query", "passage", "encode", "get_model", "dim"]

_default = None


def get_model():
    """Return the lazily-created, process-wide hot model."""
    global _default
    if _default is None:
        _default = E5()
    return _default


def query(texts):
    """Embed text(s) with the ``query: `` prefix using the shared hot model."""
    return get_model().query(texts)


def passage(texts):
    """Embed text(s) with the ``passage: `` prefix using the shared hot model."""
    return get_model().passage(texts)


def encode(texts, is_query=False):
    """Embed text(s); ``is_query`` selects the prefix."""
    return get_model().encode(texts, is_query)


def dim():
    """Embedding dimensionality (384)."""
    return get_model().dim
