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

__version__ = "0.3.0"
__all__ = ["E5", "query", "passage", "encode", "sparse", "get_model", "dim", "serve"]


def serve(*args, **kwargs):
    """Start the OpenAI-compatible embeddings server (see nanoe5.server.serve)."""
    from .server import serve as _serve
    return _serve(*args, **kwargs)

_default = None


def get_model(variant="original"):
    """Return the lazily-created, process-wide hot model."""
    global _default
    if _default is None or getattr(_default, "variant", "original") != variant:
        _default = E5(variant=variant)
    return _default


def query(texts, variant="original"):
    """Embed text(s) with the ``query: `` prefix using the shared hot model."""
    return get_model(variant=variant).query(texts)


def passage(texts, variant="original"):
    """Embed text(s) with the ``passage: `` prefix using the shared hot model."""
    return get_model(variant=variant).passage(texts)


def encode(texts, is_query=False, variant="original"):
    """Embed text(s); ``is_query`` selects the prefix."""
    return get_model(variant=variant).encode(texts, is_query)


def sparse(texts, top_k=256, fmt="numpy", variant="original"):
    """Sparse "latent terms" vector(s) using the shared hot model."""
    return get_model(variant=variant).sparse(texts, top_k=top_k, fmt=fmt)


def dim(variant="original"):
    """Embedding dimensionality (384)."""
    return get_model(variant=variant).dim
