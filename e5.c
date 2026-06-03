/* e5.c - blazing-fast CPU inference for multilingual-e5-small.
 *
 * Pure C. No transformers, no sentence-transformers, no BLAS. Just:
 *   - mmap a single packed 4-bit model file (see convert.py)
 *   - a faithful XLM-RoBERTa SentencePiece-unigram tokenizer
 *   - a hand-rolled BERT encoder with Q4_0 (4-bit) weight matmuls
 *   - mean pooling + L2 normalization
 *
 * Parallelism: OpenMP. Single-text calls parallelize across matrix rows /
 * attention heads (low latency). Batch calls parallelize across texts, and
 * because nested parallelism is capped to one active level the inner matmuls
 * then run serially -> one thread per text -> maximum throughput.
 */
#include "e5.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

#ifdef _OPENMP
#include <omp.h>
#endif
#if defined(__AVX2__)
#include <immintrin.h>
#endif

#define RESTRICT __restrict__
#define QK 32

/* ----------------------------- fp16 -> fp32 ----------------------------- */
static inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t man  = h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
        if (man == 0) { f = sign; }
        else {
            exp = 1;
            while (!(man & 0x400)) { man <<= 1; exp++; }
            man &= 0x3FF;
            f = sign | ((uint32_t)(127 - 15 - exp + 1) << 23) | (man << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7F800000u | (man << 13);
    } else {
        f = sign | ((exp - 15 + 127) << 23) | (man << 13);
    }
    float out; memcpy(&out, &f, 4); return out;
}

/* Q4_0 block: fp16 scale + 32 nibbles (byte j -> weight j low, weight j+16 high) */
typedef struct __attribute__((packed)) { uint16_t d; uint8_t qs[16]; } block_q4_0;

/* dequantize a full Q4 row (nin weights) into out[nin] */
static void deq_row(const block_q4_0 *RESTRICT b, int nin, float *RESTRICT out) {
    int nb = nin / QK;
    for (int i = 0; i < nb; i++) {
        float d = fp16_to_fp32(b[i].d);
        const uint8_t *qs = b[i].qs;
        float *o = out + i * QK;
        for (int j = 0; j < 16; j++) {
            int q0 = qs[j] & 0x0F, q1 = qs[j] >> 4;
            o[j]      = (float)(q0 - 8) * d;
            o[j + 16] = (float)(q1 - 8) * d;
        }
    }
}

/* ---- int8 x int4 matmul (the hot path) -------------------------------
 * Activations are quantized per 32-element block to int8 (Q8_0): scale d_a
 * and the block's int-sum. Weights stay 4-bit (Q4_0). Each output is
 *   sum_blocks  d_w * d_a * ( <w_nibble, a_int8> - 8 * sum(a_int8) )
 * computed with AVX2 maddubs integer MACs -- no fp32 dequant at all. */

#if defined(__AVX2__)
static inline float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v), hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 1));
    return _mm_cvtss_f32(lo);
}
/* unpack one Q4 block's 32 nibbles to 32 unsigned bytes (idx 0..31) */
static inline __m256i load_w(const block_q4_0 *RESTRICT w) {
    __m128i bytes = _mm_loadu_si128((const __m128i *)w->qs);
    __m128i lo = _mm_and_si128(bytes, _mm_set1_epi8(0x0F));            /* w0..15  */
    __m128i hi = _mm_and_si128(_mm_srli_epi16(bytes, 4), _mm_set1_epi8(0x0F)); /* w16..31 */
    return _mm256_set_m128i(hi, lo);
}
#endif

/* Quantize activations X[L,nin] to per-block int8 (Q8_0). */
static void quantize_acts(const float *RESTRICT X, int L, int nin,
                          int8_t *RESTRICT q, float *RESTRICT d, int32_t *RESTRICT s) {
    int nb = nin / QK;
#pragma omp parallel for schedule(static) if (!omp_in_parallel())
    for (int t = 0; t < L; t++) {
        for (int b = 0; b < nb; b++) {
            const float *x = X + (size_t)t * nin + b * QK;
            float amax = 0.f;
            for (int j = 0; j < QK; j++) { float a = fabsf(x[j]); if (a > amax) amax = a; }
            float dd = amax / 127.f, id = dd > 0.f ? 1.f / dd : 0.f;
            int8_t *qq = q + ((size_t)t * nb + b) * QK;
            int32_t sum = 0;
            for (int j = 0; j < QK; j++) {
                int v = (int)lrintf(x[j] * id);
                if (v > 127) v = 127; else if (v < -128) v = -128;
                qq[j] = (int8_t)v; sum += v;
            }
            d[(size_t)t * nb + b] = dd;
            s[(size_t)t * nb + b] = sum;
        }
    }
}

/* Y[L,nout] = X * W^T + bias,  W is Q4_0 [nout,nin]; X already quantized.
 * 4 output rows are computed together so each activation block is loaded once
 * and reused across 4 weight rows (4x less activation memory traffic). The
 * int32 block partials accumulate straight into float vectors (one hsum per
 * output); the -8 nibble offset is folded in through the activation sums. */
#define RO 8
static void matmul_q4q8(const block_q4_0 *W, const float *bias,
                        const int8_t *aq, const float *ad, const int32_t *as,
                        float *Y, int L, int nout, int nin) {
    int nb = nin / QK;
#pragma omp parallel if (!omp_in_parallel())
    {
        /* per-row-group weights pre-expanded to int8 once, reused over all L
         * tokens: the inner token loop then runs pure int8 x int8 maddubs. */
        int8_t *wexp = (int8_t *)aligned_alloc(32, (size_t)RO * nin);
#pragma omp for schedule(static)
        for (int oo = 0; oo < nout; oo += RO) {
            int ro = nout - oo < RO ? nout - oo : RO;
            float dwf[RO][64];
            for (int r = 0; r < ro; r++) {
                const block_q4_0 *wr = W + (size_t)(oo + r) * nb;
                int8_t *we = wexp + (size_t)r * nin;
                for (int b = 0; b < nb; b++) {
                    dwf[r][b] = fp16_to_fp32(wr[b].d);
                    const uint8_t *qs = wr[b].qs; int8_t *o = we + b * QK;
                    for (int j = 0; j < 16; j++) { o[j] = (qs[j] & 0x0F); o[j + 16] = (qs[j] >> 4); }
                }
            }
            for (int t = 0; t < L; t++) {
                const int8_t *a = aq + (size_t)t * nin;
                const float *dT = ad + (size_t)t * nb;
                const int32_t *sT = as + (size_t)t * nb;
#if defined(__AVX2__)
                __m256 facc[RO]; float corr[RO];
                for (int r = 0; r < ro; r++) { facc[r] = _mm256_setzero_ps(); corr[r] = 0.f; }
                const __m256i ones = _mm256_set1_epi16(1);
                for (int b = 0; b < nb; b++) {
                    __m256i av = _mm256_loadu_si256((const __m256i *)(a + b * QK));
                    float da = dT[b]; float sa = (float)sT[b];
                    for (int r = 0; r < ro; r++) {
                        __m256i wv = _mm256_loadu_si256((const __m256i *)(wexp + (size_t)r * nin + b * QK));
                        __m256i s = _mm256_madd_epi16(_mm256_maddubs_epi16(wv, av), ones);
                        float scale = dwf[r][b] * da;
                        facc[r] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(s), _mm256_set1_ps(scale), facc[r]);
                        corr[r] += scale * sa;
                    }
                }
                for (int r = 0; r < ro; r++)
                    Y[(size_t)t * nout + oo + r] =
                        (bias ? bias[oo + r] : 0.f) + hsum256(facc[r]) - 8.f * corr[r];
#else
                for (int r = 0; r < ro; r++) {
                    const int8_t *we = wexp + (size_t)r * nin; float acc = 0.f;
                    for (int b = 0; b < nb; b++) {
                        const int8_t *wb = we + b * QK; const int8_t *ab = a + b * QK; int isum = 0;
                        for (int j = 0; j < QK; j++) isum += wb[j] * (int)ab[j];
                        acc += dwf[r][b] * dT[b] * (float)(isum - 8 * sT[b]);
                    }
                    Y[(size_t)t * nout + oo + r] = (bias ? bias[oo + r] : 0.f) + acc;
                }
#endif
            }
        }
        free(wexp);
    }
}
#undef RO

/* x[L,H] = layernorm(x + y).  If y==NULL, layernorm(x) in place. */
static void add_layernorm(float *x, const float *y, const float *w,
                          const float *b, int L, int H, float eps) {
#pragma omp parallel for schedule(static) if (!omp_in_parallel())
    for (int t = 0; t < L; t++) {
        float *r = x + (size_t)t * H;
        const float *yy = y ? y + (size_t)t * H : NULL;
        float mean = 0.f;
        for (int i = 0; i < H; i++) { if (yy) r[i] += yy[i]; mean += r[i]; }
        mean /= H;
        float var = 0.f;
        for (int i = 0; i < H; i++) { float d = r[i] - mean; var += d * d; }
        var /= H;
        float inv = 1.f / sqrtf(var + eps);
        for (int i = 0; i < H; i++) r[i] = (r[i] - mean) * inv * w[i] + b[i];
    }
}

static inline float gelu(float x) {
    return 0.5f * x * (1.f + erff(x * 0.70710678118654752440f));
}

/* ============================== model ================================== */
typedef struct {
    const block_q4_0 *qw, *kw, *vw, *aow, *fiw, *fow;
    const float *qb, *kb, *vb, *aob, *aln_w, *aln_b, *fib, *fob, *oln_w, *oln_b;
} layer_t;

typedef struct {
    /* unigram vocab */
    int n_pieces, bos, eos, pad, unk;
    float unk_score;
    int max_piece_bytes;
    uint32_t *offs;      /* n_pieces+1 byte offsets into blob (heap copy) */
    float    *scores;    /* n_pieces (heap copy) */
    const char *blob;    /* piece bytes (mmap) */
    int *ht; int ht_mask;/* open-addressing: piece_index+1, 0 = empty */
    /* normalization map (sorted by cp) */
    int n_norm;
    uint32_t *norm_cp;   /* n_norm sorted codepoints */
    uint32_t *norm_off;  /* n_norm+1 offsets into norm_out */
    uint32_t *norm_out;  /* flattened replacement codepoints */
} tokenizer;

struct e5_model {
    int H, NL, NH, FF, VOC, MAXP, TVS;
    float eps;
    int qk, hd, nb_word;
    const block_q4_0 *word;
    const float *pos, *type, *eln_w, *eln_b;
    layer_t *layers;
    tokenizer tok;
    void *map; size_t map_size;   /* mmap region */
};

/* --------------------------- file directory --------------------------- */
typedef struct { char name[24]; const void *ptr; int dtype; int n; } tentry;

static const void *find_tensor(tentry *dir, int ndir, const char *name) {
    for (int i = 0; i < ndir; i++)
        if (strcmp(dir[i].name, name) == 0) return dir[i].ptr;
    fprintf(stderr, "e5: missing tensor '%s'\n", name);
    return NULL;
}

/* cursor reads (unaligned-safe) */
typedef struct { const uint8_t *p; } cur;
static int32_t rd_i32(cur *c){ int32_t v; memcpy(&v,c->p,4); c->p+=4; return v; }
static uint32_t rd_u32(cur *c){ uint32_t v; memcpy(&v,c->p,4); c->p+=4; return v; }
static uint64_t rd_u64(cur *c){ uint64_t v; memcpy(&v,c->p,8); c->p+=8; return v; }
static float   rd_f32(cur *c){ float v; memcpy(&v,c->p,4); c->p+=4; return v; }
static uint16_t rd_u16(cur *c){ uint16_t v; memcpy(&v,c->p,2); c->p+=2; return v; }

/* ----------------------------- tokenizer ------------------------------ */
static uint64_t fnv1a(const char *s, int n) {
    uint64_t h = 1469598103934665603ULL;
    for (int i = 0; i < n; i++) { h ^= (uint8_t)s[i]; h *= 1099511628211ULL; }
    return h;
}
static int tok_lookup(const tokenizer *t, const char *s, int n) {
    uint32_t i = (uint32_t)(fnv1a(s, n) & t->ht_mask);
    for (;;) {
        int v = t->ht[i];
        if (v == 0) return -1;
        int idx = v - 1;
        int len = (int)(t->offs[idx + 1] - t->offs[idx]);
        if (len == n && memcmp(t->blob + t->offs[idx], s, n) == 0) return idx;
        i = (i + 1) & t->ht_mask;
    }
}
/* binary search normalization map; returns index or -1 */
static int norm_find(const tokenizer *t, uint32_t cp) {
    int lo = 0, hi = t->n_norm - 1;
    while (lo <= hi) {
        int mid = (lo + hi) >> 1;
        uint32_t v = t->norm_cp[mid];
        if (v == cp) return mid;
        if (v < cp) lo = mid + 1; else hi = mid - 1;
    }
    return -1;
}

/* UTF-8 decode one codepoint; returns bytes consumed (1 on error) */
static int utf8_decode(const char *s, int n, uint32_t *cp) {
    uint8_t c = (uint8_t)s[0];
    if (c < 0x80) { *cp = c; return 1; }
    if ((c >> 5) == 0x6 && n >= 2) { *cp = ((c & 0x1F) << 6) | (s[1] & 0x3F); return 2; }
    if ((c >> 4) == 0xE && n >= 3) {
        *cp = ((c & 0x0F) << 12) | ((s[1] & 0x3F) << 6) | (s[2] & 0x3F); return 3;
    }
    if ((c >> 3) == 0x1E && n >= 4) {
        *cp = ((c & 0x07) << 18) | ((s[1] & 0x3F) << 12) | ((s[2] & 0x3F) << 6) | (s[3] & 0x3F);
        return 4;
    }
    *cp = c; return 1;
}
static int utf8_encode(uint32_t cp, char *o) {
    if (cp < 0x80) { o[0] = cp; return 1; }
    if (cp < 0x800) { o[0] = 0xC0 | (cp >> 6); o[1] = 0x80 | (cp & 0x3F); return 2; }
    if (cp < 0x10000) {
        o[0] = 0xE0 | (cp >> 12); o[1] = 0x80 | ((cp >> 6) & 0x3F); o[2] = 0x80 | (cp & 0x3F);
        return 3;
    }
    o[0] = 0xF0 | (cp >> 18); o[1] = 0x80 | ((cp >> 12) & 0x3F);
    o[2] = 0x80 | ((cp >> 6) & 0x3F); o[3] = 0x80 | (cp & 0x3F); return 4;
}

#define META 0x2581u   /* the SentencePiece metaspace marker U+2581 */

/* Run normalizer + metaspace pipeline; produce normalized UTF-8 bytes (buf),
 * and char-start offset array cs[0..nc]. Caller frees buf/cs. Returns the
 * content tokens (NO <s>/</s>), in forward order, capped to max_out. */
static int tok_content(const tokenizer *t, const char *text,
                       int *out_ids, int max_out) {
    int tn = (int)strlen(text);
    /* decode -> apply per-codepoint normalization map */
    uint32_t *cps = (uint32_t *)malloc(sizeof(uint32_t) * (size_t)(tn + 2) * 4 + 64);
    int nc = 0;
    for (int i = 0; i < tn; ) {
        uint32_t cp; int adv = utf8_decode(text + i, tn - i, &cp); i += adv;
        int ni = norm_find(t, cp);
        if (ni < 0) { cps[nc++] = cp; }
        else {
            for (uint32_t k = t->norm_off[ni]; k < t->norm_off[ni + 1]; k++)
                cps[nc++] = t->norm_out[k];
        }
    }
    /* collapse runs of 2+ spaces (U+0020) into one */
    int w = 0;
    for (int i = 0; i < nc; i++) {
        if (cps[i] == 0x20 && w > 0 && cps[w - 1] == 0x20) continue;
        cps[w++] = cps[i];
    }
    nc = w;
    /* metaspace: prepend space if not already leading-space, then ' ' -> META */
    int start = 0;
    char *buf = (char *)malloc((size_t)(nc + 1) * 4 + 8);
    int blen = 0;
    /* leading metaspace */
    if (nc == 0 || cps[0] != 0x20) blen += utf8_encode(META, buf + blen);
    int *cs = (int *)malloc(sizeof(int) * (size_t)(nc + 2));
    /* we need char boundaries on the *byte* buffer for Viterbi */
    int nchar = 0;
    if (nc == 0 || cps[0] != 0x20) cs[nchar++] = 0;     /* the prepended META */
    for (int i = start; i < nc; i++) {
        cs[nchar++] = blen;
        uint32_t cp = cps[i] == 0x20 ? META : cps[i];
        blen += utf8_encode(cp, buf + blen);
    }
    cs[nchar] = blen;

    /* Viterbi over Unigram pieces */
    int n = nchar;
    float *best = (float *)malloc(sizeof(float) * (size_t)(n + 1));
    int *bp = (int *)malloc(sizeof(int) * (size_t)(n + 1));   /* piece id, -1 = unk */
    int *bf = (int *)malloc(sizeof(int) * (size_t)(n + 1));   /* from char index */
    for (int i = 0; i <= n; i++) best[i] = -1e30f;
    best[0] = 0.f;
    for (int i = 0; i < n; i++) {
        if (best[i] <= -1e29f) continue;
        int bi = cs[i];
        /* try every piece length in chars while byte span fits */
        for (int e = i + 1; e <= n; e++) {
            int len = cs[e] - bi;
            if (len > t->max_piece_bytes) break;
            int idx = tok_lookup(t, buf + bi, len);
            if (idx >= 0 && idx != t->unk) {
                float sc = best[i] + t->scores[idx];
                if (sc > best[e]) { best[e] = sc; bp[e] = idx; bf[e] = i; }
            }
        }
        /* single-char unknown fallback */
        float us = best[i] + t->unk_score;
        if (us > best[i + 1]) { best[i + 1] = us; bp[i + 1] = -1; bf[i + 1] = i; }
    }
    /* backtrack -> content tokens (no specials), forward order, head-capped */
    int *tmp = (int *)malloc(sizeof(int) * (size_t)(n + 1));
    int nt = 0;
    for (int i = n; i > 0; ) {
        int id = bp[i] < 0 ? t->unk : bp[i];
        tmp[nt++] = id;
        i = bf[i];
    }
    int ncontent = nt < max_out ? nt : max_out;
    for (int j = 0; j < ncontent; j++) out_ids[j] = tmp[nt - 1 - j];

    free(tmp);
    free(best); free(bp); free(bf);
    free(cs); free(buf); free(cps);
    return ncontent;
}

/* ------------------------------- forward ------------------------------
 * The whole batch is processed with the tokens of every text concatenated
 * into one [T,H] activation matrix. The token-wise matmuls (Q/K/V/O/FFN)
 * then stream each weight ONCE for the entire batch (high arithmetic
 * intensity, the key to throughput). Attention runs per text segment.
 *
 * ids   : concatenated token ids, length T
 * pos   : per-token position id (resets to 0 at each segment start)
 * sstart/send[gi] : the [start,end) token range of gi's segment
 * segoff: nseg+1 segment offsets; out holds nseg*H embeddings */
static int forward_batch(e5_model *m, const int *ids, const int *pos,
                         const int *sstart, const int *send,
                         const int *segoff, int nseg, int T, float *out) {
    int H = m->H, FF = m->FF, NH = m->NH, hd = m->hd;
    size_t TH = (size_t)T * H;
    float *x   = (float *)malloc(TH * sizeof(float));
    float *q   = (float *)malloc(TH * sizeof(float));
    float *k   = (float *)malloc(TH * sizeof(float));
    float *v   = (float *)malloc(TH * sizeof(float));
    float *ctx = (float *)malloc(TH * sizeof(float));
    float *ao  = (float *)malloc(TH * sizeof(float));
    float *mid = (float *)malloc((size_t)T * FF * sizeof(float));
    float *fo  = (float *)malloc(TH * sizeof(float));
    int nbmax = FF / QK;
    int8_t  *aq = (int8_t *)malloc((size_t)T * FF);
    float   *ad = (float *)malloc((size_t)T * nbmax * sizeof(float));
    int32_t *as = (int32_t *)malloc((size_t)T * nbmax * sizeof(int32_t));
#define QUANT(X, NIN) quantize_acts((X), T, (NIN), aq, ad, as)
#define MAT(W, B, Y, NOUT, NIN) matmul_q4q8((W), (B), aq, ad, as, (Y), T, (NOUT), (NIN))

    /* embeddings: word + position + token_type(0), then LayerNorm */
#pragma omp parallel for schedule(static)
    for (int t = 0; t < T; t++) {
        deq_row(m->word + (size_t)ids[t] * m->nb_word, H, x + (size_t)t * H);
        const float *pp = m->pos + (size_t)pos[t] * H;
        float *r = x + (size_t)t * H;
        for (int i = 0; i < H; i++) r[i] += pp[i] + m->type[i];
    }
    add_layernorm(x, NULL, m->eln_w, m->eln_b, T, H, m->eps);

    float scale = 1.f / sqrtf((float)hd);
    for (int l = 0; l < m->NL; l++) {
        layer_t *ly = &m->layers[l];
        QUANT(x, H);                       /* Q/K/V share the same input */
        MAT(ly->qw, ly->qb, q, H, H);
        MAT(ly->kw, ly->kb, k, H, H);
        MAT(ly->vw, ly->vb, v, H, H);
        /* scaled dot-product attention, restricted to each text's own tokens */
#pragma omp parallel
        {
            float *prob = (float *)malloc(sizeof(float) * (size_t)m->MAXP);
#pragma omp for collapse(2) schedule(static)
            for (int t = 0; t < T; t++)
                for (int h = 0; h < NH; h++) {
                    int s0 = sstart[t], s1 = send[t];
                    const float *qh = q + (size_t)t * H + h * hd;
                    float mx = -1e30f;
                    for (int s = s0; s < s1; s++) {
                        const float *kh = k + (size_t)s * H + h * hd;
                        float d = 0.f;
                        for (int e = 0; e < hd; e++) d += qh[e] * kh[e];
                        d *= scale; prob[s - s0] = d; if (d > mx) mx = d;
                    }
                    float sum = 0.f;
                    for (int s = s0; s < s1; s++) { float e = expf(prob[s - s0] - mx); prob[s - s0] = e; sum += e; }
                    float inv = 1.f / sum;
                    float *ch = ctx + (size_t)t * H + h * hd;
                    for (int e = 0; e < hd; e++) ch[e] = 0.f;
                    for (int s = s0; s < s1; s++) {
                        float p = prob[s - s0] * inv;
                        const float *vh = v + (size_t)s * H + h * hd;
                        for (int e = 0; e < hd; e++) ch[e] += p * vh[e];
                    }
                }
            free(prob);
        }
        QUANT(ctx, H); MAT(ly->aow, ly->aob, ao, H, H);
        add_layernorm(x, ao, ly->aln_w, ly->aln_b, T, H, m->eps);   /* x = LN(x+attn) */

        QUANT(x, H); MAT(ly->fiw, ly->fib, mid, FF, H);
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < (size_t)T * FF; i++) mid[i] = gelu(mid[i]);
        QUANT(mid, FF); MAT(ly->fow, ly->fob, fo, H, FF);
        add_layernorm(x, fo, ly->oln_w, ly->oln_b, T, H, m->eps);   /* x = LN(x+ffn) */
    }

    /* mean pool each segment, then L2 normalize */
#pragma omp parallel for schedule(static)
    for (int sgi = 0; sgi < nseg; sgi++) {
        float *o = out + (size_t)sgi * H;
        for (int i = 0; i < H; i++) o[i] = 0.f;
        int a = segoff[sgi], b = segoff[sgi + 1];
        for (int t = a; t < b; t++) {
            const float *r = x + (size_t)t * H;
            for (int i = 0; i < H; i++) o[i] += r[i];
        }
        /* raw mean-pool only; L2-normalization happens after windows are
         * recombined in embed_n (so sliding-window texts pool correctly). */
        float invL = 1.f / (float)(b - a);
        for (int i = 0; i < H; i++) o[i] *= invL;
    }

    free(x); free(q); free(k); free(v); free(ctx); free(ao); free(mid); free(fo);
    free(aq); free(ad); free(as);
    return 0;
#undef QUANT
#undef MAT
}

/* Soft cap on tokens processed per forward pass, so that arbitrarily long
 * inputs (many sliding windows) run in bounded memory instead of allocating
 * one giant activation buffer. Windows are flushed in chunks up to this size;
 * the weighted combine is additive across chunks, so results are identical. */
#define E5_CHUNK_TOKENS 16384

/* Embed n texts with transparent sliding windows for inputs longer than the
 * model's 512-token limit. A long text is split (no overlap) into windows of
 * up to MAXP-2 content tokens; each window is run as a full bos+chunk+eos
 * input, and the per-text result is the token-count-weighted average of the
 * windows' (pre-normalization) mean-pooled vectors, finally L2-normalized.
 * This equals mean-pooling over the whole document, modulo windowed attention.
 * Texts that fit in one window behave exactly as a plain single pass. */
static int embed_n(e5_model *m, const char **texts, int n, int is_query, float *out) {
    const char *pfx = is_query ? "query: " : "passage: ";
    size_t plen = strlen(pfx);
    int H = m->H, WMAX = m->MAXP - 2;        /* content tokens per window */

    /* 1. tokenize each text's full content (prefix included, no truncation),
     *    and enumerate every window as (text, content range). */
    int **content = (int **)malloc(sizeof(int *) * n);
    int  *clen    = (int *)malloc(sizeof(int) * n);
    int W = 0;                               /* total windows across all texts */
    for (int i = 0; i < n; i++) {
        size_t tl = strlen(texts[i]);
        char *buf = (char *)malloc(plen + tl + 1);
        memcpy(buf, pfx, plen); memcpy(buf + plen, texts[i], tl + 1);
        int capc = (int)(plen + tl) + 4;     /* #tokens <= #bytes, safe bound */
        content[i] = (int *)malloc(sizeof(int) * (capc > 1 ? capc : 1));
        clen[i] = tok_content(&m->tok, buf, content[i], capc);
        free(buf);
        int nw = clen[i] / WMAX + (clen[i] % WMAX ? 1 : 0);
        if (nw < 1) nw = 1;
        W += nw;
    }
    int *wt = (int *)malloc(sizeof(int) * W);   /* window -> text id      */
    int *wa = (int *)malloc(sizeof(int) * W);   /* window content [a, b)  */
    int *wb = (int *)malloc(sizeof(int) * W);
    int W2 = 0;
    for (int i = 0; i < n; i++) {
        int nw = clen[i] / WMAX + (clen[i] % WMAX ? 1 : 0); if (nw < 1) nw = 1;
        for (int w = 0; w < nw; w++) {
            int a = w * WMAX, b = a + WMAX; if (b > clen[i]) b = clen[i];
            wt[W2] = i; wa[W2] = a; wb[W2] = b; W2++;
        }
    }

    /* 2. accumulate per-text weighted sums, processing windows in chunks */
    double *wsum = (double *)calloc(n, sizeof(double));
    for (int i = 0; i < n; i++) { float *d = out + (size_t)i * H; for (int kk = 0; kk < H; kk++) d[kk] = 0.f; }

    int budget = E5_CHUNK_TOKENS;
    if (budget < m->MAXP) budget = m->MAXP;     /* always fit >=1 window */
    int rc = 0;
    for (int w0 = 0; w0 < W; ) {
        /* gather windows into a chunk under the token budget (>=1 window) */
        int w1 = w0; long Tc = 0;
        while (w1 < W) {
            int len = (wb[w1] - wa[w1]) + 2;
            if (w1 > w0 && Tc + len > budget) break;
            Tc += len; w1++;
        }
        int segc = w1 - w0;
        int *ids = (int *)malloc(sizeof(int) * (size_t)Tc);
        int *pos = (int *)malloc(sizeof(int) * (size_t)Tc);
        int *segoff = (int *)malloc(sizeof(int) * (segc + 1));
        int *sstart = (int *)malloc(sizeof(int) * (size_t)Tc);
        int *send   = (int *)malloc(sizeof(int) * (size_t)Tc);
        int T2 = 0; segoff[0] = 0;
        for (int s = 0; s < segc; s++) {
            int wi = w0 + s, a = wa[wi], b = wb[wi], p = 0;
            ids[T2] = m->tok.bos; pos[T2] = p++; T2++;
            for (int j = a; j < b; j++) { ids[T2] = content[wt[wi]][j]; pos[T2] = p++; T2++; }
            ids[T2] = m->tok.eos; pos[T2] = p++; T2++;
            segoff[s + 1] = T2;
        }
        for (int s = 0; s < segc; s++)
            for (int t = segoff[s]; t < segoff[s + 1]; t++) { sstart[t] = segoff[s]; send[t] = segoff[s + 1]; }

        float *segmean = (float *)malloc(sizeof(float) * (size_t)segc * H);
        rc |= forward_batch(m, ids, pos, sstart, send, segoff, segc, (int)Tc, segmean);

        for (int s = 0; s < segc; s++) {
            int wi = w0 + s, i = wt[wi];
            float wgt = (float)((wb[wi] - wa[wi]) > 0 ? (wb[wi] - wa[wi]) : 1);
            float *d = out + (size_t)i * H;
            const float *sm = segmean + (size_t)s * H;
            for (int kk = 0; kk < H; kk++) d[kk] += wgt * sm[kk];
            wsum[i] += wgt;
        }
        free(ids); free(pos); free(segoff); free(sstart); free(send); free(segmean);
        w0 = w1;
    }

    /* 3. divide by total weight and L2-normalize each text */
    for (int i = 0; i < n; i++) {
        float *d = out + (size_t)i * H;
        float inv = wsum[i] > 0 ? (float)(1.0 / wsum[i]) : 1.f, nrm = 0.f;
        for (int kk = 0; kk < H; kk++) { d[kk] *= inv; nrm += d[kk] * d[kk]; }
        nrm = 1.f / sqrtf(nrm > 1e-12f ? nrm : 1e-12f);
        for (int kk = 0; kk < H; kk++) d[kk] *= nrm;
    }

    for (int i = 0; i < n; i++) free(content[i]);
    free(content); free(clen); free(wt); free(wa); free(wb); free(wsum);
    return rc;
}

int e5_embed(e5_model *m, const char *text, int is_query, float *out) {
    return embed_n(m, &text, 1, is_query, out);
}

int e5_embed_batch(e5_model *m, const char **texts, int n, int is_query, float *out) {
    return embed_n(m, texts, n, is_query, out);
}

int e5_dim(const e5_model *m) { return m->H; }

/* ------------------------------- loader -------------------------------
 * Parse a model image living at `base` (length `size`). `map`/`map_size`,
 * when non-NULL, are an owned mmap region to munmap on free; for an embedded
 * (in-binary) image they are NULL and nothing is unmapped. */
static e5_model *e5_parse(const uint8_t *base, size_t size, void *map, size_t map_size) {
    if (size < 8 || memcmp(base, "E5S1", 4) != 0) {
        fprintf(stderr, "e5: bad magic / truncated model\n");
        if (map) munmap(map, map_size);
        return NULL;
    }
    e5_model *m = (e5_model *)calloc(1, sizeof(e5_model));
    m->map = map; m->map_size = map_size;
    cur c = { base };
    c.p += 4; rd_u32(&c); /* magic + version (magic already checked) */
    m->H = rd_i32(&c); m->NL = rd_i32(&c); m->NH = rd_i32(&c); m->FF = rd_i32(&c);
    m->VOC = rd_i32(&c); m->MAXP = rd_i32(&c); m->TVS = rd_i32(&c);
    m->eps = rd_f32(&c); m->qk = rd_i32(&c);
    m->hd = m->H / m->NH; m->nb_word = m->H / QK;

    /* tokenizer */
    tokenizer *t = &m->tok;
    t->n_pieces = rd_i32(&c);
    t->bos = rd_i32(&c); t->eos = rd_i32(&c); t->pad = rd_i32(&c); t->unk = rd_i32(&c);
    uint64_t bloblen = rd_u64(&c);
    size_t offs_bytes = (size_t)(t->n_pieces + 1) * 4;
    t->offs = (uint32_t *)malloc(offs_bytes);   memcpy(t->offs, c.p, offs_bytes); c.p += offs_bytes;
    size_t sc_bytes = (size_t)t->n_pieces * 4;
    t->scores = (float *)malloc(sc_bytes);      memcpy(t->scores, c.p, sc_bytes); c.p += sc_bytes;
    t->blob = (const char *)c.p; c.p += bloblen;
    /* normalization map */
    t->n_norm = rd_i32(&c);
    uint32_t nm_bytes = rd_u32(&c);
    const uint8_t *nm = c.p; c.p += nm_bytes;
    t->norm_cp  = (uint32_t *)malloc((size_t)t->n_norm * 4);
    t->norm_off = (uint32_t *)malloc((size_t)(t->n_norm + 1) * 4);
    {   /* parse (cp,u8 nout,outs...) -> arrays; count total outs */
        const uint8_t *p = nm; int total = 0;
        for (int i = 0; i < t->n_norm; i++) {
            uint32_t cp; memcpy(&cp, p, 4); p += 4; uint8_t no = *p++; p += (size_t)no * 4;
            total += no;
        }
        t->norm_out = (uint32_t *)malloc((size_t)total * 4);
        p = nm; int acc = 0;
        for (int i = 0; i < t->n_norm; i++) {
            uint32_t cp; memcpy(&cp, p, 4); p += 4; uint8_t no = *p++;
            t->norm_cp[i] = cp; t->norm_off[i] = acc;
            for (int j = 0; j < no; j++) { memcpy(&t->norm_out[acc], p, 4); p += 4; acc++; }
        }
        t->norm_off[t->n_norm] = acc;
    }
    /* unk score = min piece score - 10 ; max piece byte length */
    float mins = 1e30f; int maxb = 1;
    for (int i = 0; i < t->n_pieces; i++) {
        if (t->scores[i] < mins) mins = t->scores[i];
        int len = (int)(t->offs[i + 1] - t->offs[i]);
        if (len > maxb) maxb = len;
    }
    t->unk_score = mins - 10.f;
    t->max_piece_bytes = maxb;
    /* build piece hash table */
    int hbits = 1; while ((1 << hbits) < t->n_pieces * 2) hbits++;
    int hsz = 1 << hbits; t->ht_mask = hsz - 1;
    t->ht = (int *)calloc(hsz, sizeof(int));
    for (int i = 0; i < t->n_pieces; i++) {
        int len = (int)(t->offs[i + 1] - t->offs[i]);
        if (len == 0) continue;                 /* skip empty (shouldn't happen) */
        uint32_t h = (uint32_t)(fnv1a(t->blob + t->offs[i], len) & t->ht_mask);
        while (t->ht[h]) h = (h + 1) & t->ht_mask;
        t->ht[h] = i + 1;
    }

    /* tensor directory */
    int ndir = rd_i32(&c);
    tentry *dir = (tentry *)malloc(sizeof(tentry) * ndir);
    for (int i = 0; i < ndir; i++) {
        int nl = rd_u16(&c);
        memset(dir[i].name, 0, sizeof(dir[i].name));
        memcpy(dir[i].name, c.p, nl < 23 ? nl : 23); c.p += nl;
        dir[i].dtype = *c.p++; dir[i].n = rd_u32(&c);
        uint64_t off = rd_u64(&c); rd_u64(&c); /* nbytes unused */
        dir[i].ptr = base + off;
    }

    m->word  = (const block_q4_0 *)find_tensor(dir, ndir, "emb.word");
    m->pos   = (const float *)find_tensor(dir, ndir, "emb.pos");
    m->type  = (const float *)find_tensor(dir, ndir, "emb.type");
    m->eln_w = (const float *)find_tensor(dir, ndir, "emb.ln.w");
    m->eln_b = (const float *)find_tensor(dir, ndir, "emb.ln.b");
    m->layers = (layer_t *)calloc(m->NL, sizeof(layer_t));
    char nm2[24];
    for (int l = 0; l < m->NL; l++) {
        layer_t *ly = &m->layers[l];
#define T(field, suffix, type) \
        snprintf(nm2, sizeof(nm2), "l%d." suffix, l); ly->field = (type)find_tensor(dir, ndir, nm2);
        T(qw, "q.w", const block_q4_0 *)  T(qb, "q.b", const float *)
        T(kw, "k.w", const block_q4_0 *)  T(kb, "k.b", const float *)
        T(vw, "v.w", const block_q4_0 *)  T(vb, "v.b", const float *)
        T(aow, "ao.w", const block_q4_0 *) T(aob, "ao.b", const float *)
        T(aln_w, "aln.w", const float *)  T(aln_b, "aln.b", const float *)
        T(fiw, "fi.w", const block_q4_0 *) T(fib, "fi.b", const float *)
        T(fow, "fo.w", const block_q4_0 *) T(fob, "fo.b", const float *)
        T(oln_w, "oln.w", const float *)  T(oln_b, "oln.b", const float *)
#undef T
    }
    free(dir);

#ifdef _OPENMP
    omp_set_max_active_levels(1);   /* batch parallel -> inner matmuls serial */
#endif
    return m;
}

e5_model *e5_load(const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { fprintf(stderr, "e5: cannot open %s\n", path); return NULL; }
    struct stat sb; fstat(fd, &sb);
    void *map = mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (map == MAP_FAILED) { fprintf(stderr, "e5: mmap failed\n"); return NULL; }
    return e5_parse((const uint8_t *)map, (size_t)sb.st_size, map, (size_t)sb.st_size);
}

e5_model *e5_load_mem(const void *data, size_t size) {
    return e5_parse((const uint8_t *)data, size, NULL, 0);
}

/* Number of tokens (incl <s>/</s>) the model would feed for `text`; used for
 * usage reporting. Counts the un-windowed content length + 2. */
int e5_token_count(e5_model *m, const char *text, int is_query) {
    const char *pfx = is_query ? "query: " : "passage: ";
    size_t plen = strlen(pfx), tl = strlen(text);
    char *buf = (char *)malloc(plen + tl + 1);
    memcpy(buf, pfx, plen); memcpy(buf + plen, text, tl + 1);
    int cap = (int)(plen + tl) + 4;
    int *tmp = (int *)malloc(sizeof(int) * (cap > 1 ? cap : 1));
    int n = tok_content(&m->tok, buf, tmp, cap);
    free(buf); free(tmp);
    return n + 2;
}

void e5_free(e5_model *m) {
    if (!m) return;
    free(m->tok.offs); free(m->tok.scores); free(m->tok.ht);
    free(m->tok.norm_cp); free(m->tok.norm_off); free(m->tok.norm_out);
    free(m->layers);
    if (m->map) munmap(m->map, m->map_size);
    free(m);
}

/* ------------------------------- CLI ---------------------------------- */
#ifdef E5_MAIN
int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s model.bin <query|passage> [text...]\n", argv[0]); return 1; }
    e5_model *m = e5_load(argv[1]);
    if (!m) return 1;
    int is_q = strcmp(argv[2], "query") == 0;
    int D = e5_dim(m);
    float *out = (float *)malloc(sizeof(float) * D);
    const char *text = argc > 3 ? argv[3] : "hello world";
    e5_embed(m, text, is_q, out);
    printf("dim=%d  [", D);
    for (int i = 0; i < 8; i++) printf("%.5f ", out[i]);
    printf("...]\n");
    free(out); e5_free(m);
    return 0;
}
#endif
