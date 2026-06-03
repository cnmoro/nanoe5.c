/* e5.h - blazing-fast CPU inference for multilingual-e5-small (4-bit, pure C) */
#ifndef E5_H
#define E5_H

#include <stddef.h>

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

/* Same as above but with an explicit prefix already chosen per call mode.
 * `is_query`: 1 -> "query: ", 0 -> "passage: ". Provided for completeness. */

#ifdef __cplusplus
}
#endif
#endif /* E5_H */
