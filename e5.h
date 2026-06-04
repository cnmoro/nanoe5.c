/* e5.h - blazing-fast CPU inference for multilingual-e5-small (4-bit, pure C) */
#ifndef E5_H
#define E5_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct e5_model e5_model;

/* Load the packed 4-bit model produced by convert.py. Returns NULL on error. */
e5_model *e5_load(const char *path);
/* Load a model from an in-memory image (e.g. embedded in the binary). The
 * buffer must outlive the model; it is not copied or freed. */
e5_model *e5_load_mem(const void *data, size_t size);
void      e5_free(e5_model *m);

/* Token count (incl. specials) for usage reporting. */
int e5_token_count(e5_model *m, const char *text, int is_query);

/* Embedding dimension (384). */
int       e5_dim(const e5_model *m);

/* Embed one text. `is_query` selects the "query: " (1) or "passage: " (0)
 * prefix described in the model card. `out` must hold e5_dim() floats; it is
 * filled with the L2-normalized sentence embedding. Returns 0 on success.
 * Internally parallelized across CPU cores (low latency for a single text). */
int e5_embed(e5_model *m, const char *text, int is_query, float *out);

/* Embed `n` texts at once, writing n*e5_dim() floats (row-major) into `out`.
 * Parallelized across texts for maximum throughput. Returns 0 on success. */
int e5_embed_batch(e5_model *m, const char **texts, int n, int is_query,
                   float *out);

/* --- sparse "latent terms" via a trained TopK sparse autoencoder (SAE) -----
 * Load a sparse head (sae.bin from sae_train.py); then e5_embed_sparse maps the
 * encoder's token hidden states through the SAE and max-pools over tokens into
 * a high-dimensional sparse vector for hybrid (dense + sparse) retrieval. */

/* Attach a sparse head from sae.bin. Returns 0 on success. */
int e5_load_sae(e5_model *m, const char *path);

/* Number of sparse features (SAE latent dimension), or 0 if no SAE is loaded. */
int e5_sparse_dim(const e5_model *m);

/* Top-k sparse vector for `text`. Requires e5_load_sae. `out_idx`/`out_val`
 * must hold `top_k` entries; returns the number of nonzeros (<= top_k), sorted
 * by weight descending, or -1 if no SAE is loaded. Sparse uses the raw text
 * (no query:/passage: prefix); the same recipe is used for queries and docs. */
int e5_embed_sparse(e5_model *m, const char *text, int top_k,
                    int32_t *out_idx, float *out_val);

#ifdef __cplusplus
}
#endif
#endif /* E5_H */
