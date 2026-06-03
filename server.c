/* server.c - self-contained OpenAI-compatible embeddings server for nanoE5.c.
 *
 *   ./e5 --server --host 0.0.0.0 --port 8000
 *
 * The 4-bit model is embedded directly in the binary (see Makefile / ld), so a
 * single file is all a client needs. Also keeps the plain CLI:
 *
 *   ./e5 query "some text"          # uses the embedded model
 *   ./e5 --model x.bin passage "y"  # external model file
 *
 * Endpoints:
 *   POST /v1/embeddings   - OpenAI embeddings API (string or array input,
 *                           "float" and "base64" encoding_format)
 *   GET  /v1/models       - lists the model
 *   GET  /  /health       - health check
 */
#define _GNU_SOURCE
#include "e5.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdarg.h>
#include <strings.h>
#include <errno.h>
#include <unistd.h>
#include <pthread.h>
#include <semaphore.h>
#include <signal.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <sys/time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/* the model image, injected by `ld -r -b binary model.bin` */
#ifdef E5_EMBED
extern const unsigned char _binary_model_bin_start[];
extern const unsigned char _binary_model_bin_end[];
#endif

#define MAX_BODY     (256u * 1024u * 1024u)   /* 256 MB request cap */
#define MAX_CONN     256                       /* concurrent connection cap */
#define JSON_MAXDEPTH 64

static e5_model *G_model;
static int       G_dim;
static int       G_default_query = 1;          /* default modality: query */
static const char *G_model_name = "multilingual-e5-small-q4";
static pthread_mutex_t G_compute = PTHREAD_MUTEX_INITIALIZER;
static sem_t     G_slots;

/* ============================ dynamic buffer =========================== */
typedef struct { char *p; size_t len, cap; } buf;
static void buf_reserve(buf *b, size_t need) {
    if (b->len + need <= b->cap) return;
    size_t nc = b->cap ? b->cap * 2 : 4096;
    while (nc < b->len + need) nc *= 2;
    b->p = (char *)realloc(b->p, nc); b->cap = nc;
}
static void buf_put(buf *b, const void *s, size_t n) { buf_reserve(b, n); memcpy(b->p + b->len, s, n); b->len += n; }
static void buf_puts(buf *b, const char *s) { buf_put(b, s, strlen(s)); }
static void buf_putc(buf *b, char c) { buf_reserve(b, 1); b->p[b->len++] = c; }
static void buf_printf(buf *b, const char *fmt, ...) {
    char tmp[256]; va_list ap; va_start(ap, fmt);
    int n = vsnprintf(tmp, sizeof(tmp), fmt, ap); va_end(ap);
    if (n < 0) return;
    if ((size_t)n < sizeof(tmp)) { buf_put(b, tmp, (size_t)n); return; }
    buf_reserve(b, (size_t)n + 1);
    va_start(ap, fmt); vsnprintf(b->p + b->len, (size_t)n + 1, fmt, ap); va_end(ap);
    b->len += (size_t)n;
}

/* ================================ base64 =============================== */
static const char B64[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
static void b64_encode(buf *b, const uint8_t *d, size_t n) {
    size_t i = 0;
    for (; i + 3 <= n; i += 3) {
        uint32_t v = (d[i] << 16) | (d[i + 1] << 8) | d[i + 2];
        buf_putc(b, B64[(v >> 18) & 63]); buf_putc(b, B64[(v >> 12) & 63]);
        buf_putc(b, B64[(v >> 6) & 63]);  buf_putc(b, B64[v & 63]);
    }
    if (n - i == 1) {
        uint32_t v = d[i] << 16;
        buf_putc(b, B64[(v >> 18) & 63]); buf_putc(b, B64[(v >> 12) & 63]);
        buf_putc(b, '='); buf_putc(b, '=');
    } else if (n - i == 2) {
        uint32_t v = (d[i] << 16) | (d[i + 1] << 8);
        buf_putc(b, B64[(v >> 18) & 63]); buf_putc(b, B64[(v >> 12) & 63]);
        buf_putc(b, B64[(v >> 6) & 63]); buf_putc(b, '=');
    }
}

/* ============================== JSON parse ============================= */
typedef struct { const char *p, *end; int depth; } jp;
static void jskip_ws(jp *j) {
    while (j->p < j->end) {
        char c = *j->p;
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') j->p++;
        else break;
    }
}
static int jhex(int c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}
static int utf8_put(buf *b, uint32_t cp) {
    if (cp < 0x80) buf_putc(b, (char)cp);
    else if (cp < 0x800) { buf_putc(b, 0xC0 | (cp >> 6)); buf_putc(b, 0x80 | (cp & 0x3F)); }
    else if (cp < 0x10000) {
        buf_putc(b, 0xE0 | (cp >> 12)); buf_putc(b, 0x80 | ((cp >> 6) & 0x3F)); buf_putc(b, 0x80 | (cp & 0x3F));
    } else {
        buf_putc(b, 0xF0 | (cp >> 18)); buf_putc(b, 0x80 | ((cp >> 12) & 0x3F));
        buf_putc(b, 0x80 | ((cp >> 6) & 0x3F)); buf_putc(b, 0x80 | (cp & 0x3F));
    }
    return 0;
}
/* parse a JSON string at j->p (which must point at the opening quote) into out.
 * Decodes escapes incl. \uXXXX surrogate pairs; drops embedded NUL () so
 * the NUL-terminated engine never truncates. Returns 0 on success. */
static int jstring(jp *j, buf *out) {
    if (j->p >= j->end || *j->p != '"') return -1;
    j->p++;
    while (j->p < j->end) {
        unsigned char c = (unsigned char)*j->p++;
        if (c == '"') return 0;
        if (c == '\\') {
            if (j->p >= j->end) return -1;
            char e = *j->p++;
            switch (e) {
                case '"': buf_putc(out, '"'); break;
                case '\\': buf_putc(out, '\\'); break;
                case '/': buf_putc(out, '/'); break;
                case 'b': buf_putc(out, '\b'); break;
                case 'f': buf_putc(out, '\f'); break;
                case 'n': buf_putc(out, '\n'); break;
                case 'r': buf_putc(out, '\r'); break;
                case 't': buf_putc(out, '\t'); break;
                case 'u': {
                    if (j->end - j->p < 4) return -1;
                    int h0 = jhex(j->p[0]), h1 = jhex(j->p[1]), h2 = jhex(j->p[2]), h3 = jhex(j->p[3]);
                    if ((h0 | h1 | h2 | h3) < 0) return -1;
                    uint32_t cp = (h0 << 12) | (h1 << 8) | (h2 << 4) | h3; j->p += 4;
                    if (cp >= 0xD800 && cp <= 0xDBFF) {        /* high surrogate */
                        if (j->end - j->p >= 6 && j->p[0] == '\\' && j->p[1] == 'u') {
                            int g0 = jhex(j->p[2]), g1 = jhex(j->p[3]), g2 = jhex(j->p[4]), g3 = jhex(j->p[5]);
                            if ((g0 | g1 | g2 | g3) >= 0) {
                                uint32_t lo = (g0 << 12) | (g1 << 8) | (g2 << 4) | g3;
                                if (lo >= 0xDC00 && lo <= 0xDFFF) {
                                    cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                                    j->p += 6;
                                } else cp = 0xFFFD;
                            } else cp = 0xFFFD;
                        } else cp = 0xFFFD;
                    } else if (cp >= 0xDC00 && cp <= 0xDFFF) cp = 0xFFFD; /* stray low */
                    if (cp != 0) utf8_put(out, cp);            /* drop NUL */
                    break;
                }
                default: return -1;
            }
        } else if (c == 0) {
            /* raw NUL inside body: drop it */
        } else {
            buf_putc(out, (char)c);
        }
    }
    return -1; /* unterminated */
}
/* recursively skip any JSON value (depth-limited) */
static int jskip(jp *j) {
    if (++j->depth > JSON_MAXDEPTH) return -1;
    jskip_ws(j);
    if (j->p >= j->end) { j->depth--; return -1; }
    char c = *j->p;
    int rc = 0;
    if (c == '"') { buf t = {0}; rc = jstring(j, &t); free(t.p); }
    else if (c == '{' || c == '[') {
        char close = c == '{' ? '}' : ']'; j->p++;
        jskip_ws(j);
        if (j->p < j->end && *j->p == close) { j->p++; j->depth--; return 0; }
        for (;;) {
            jskip_ws(j);
            if (c == '{') {                       /* key */
                buf k = {0}; if (jstring(j, &k)) { free(k.p); rc = -1; break; }
                free(k.p); jskip_ws(j);
                if (j->p >= j->end || *j->p != ':') { rc = -1; break; }
                j->p++;
            }
            if (jskip(j)) { rc = -1; break; }
            jskip_ws(j);
            if (j->p >= j->end) { rc = -1; break; }
            if (*j->p == ',') { j->p++; continue; }
            if (*j->p == close) { j->p++; break; }
            rc = -1; break;
        }
    } else { /* number / true / false / null */
        while (j->p < j->end) {
            char d = *j->p;
            if (d == ',' || d == '}' || d == ']' || d == ' ' || d == '\t' || d == '\n' || d == '\r') break;
            j->p++;
        }
    }
    j->depth--;
    return rc;
}

/* parsed request */
typedef struct {
    char **inputs; int ninputs, cap;
    int is_query;        /* -1 = use server default */
    int base64;
    int err;             /* 1 = malformed/unsupported */
    const char *errmsg;
} req;
static void req_push(req *r, char *s) {
    if (r->ninputs == r->cap) { r->cap = r->cap ? r->cap * 2 : 8; r->inputs = realloc(r->inputs, sizeof(char *) * r->cap); }
    r->inputs[r->ninputs++] = s;
}
static void req_free(req *r) {
    for (int i = 0; i < r->ninputs; i++) free(r->inputs[i]);
    free(r->inputs);
}
static char *jstring_dup(jp *j) {
    buf t = {0}; if (jstring(j, &t)) { free(t.p); return NULL; }
    buf_putc(&t, '\0');           /* NUL terminate for the engine */
    return t.p;
}
/* parse the top-level embeddings request object */
static void parse_req(const char *body, size_t len, req *r) {
    r->is_query = -1; r->base64 = 0; r->err = 0; r->errmsg = NULL;
    jp j = { body, body + len, 0 };
    jskip_ws(&j);
    if (j.p >= j.end || *j.p != '{') { r->err = 1; r->errmsg = "body must be a JSON object"; return; }
    j.p++;
    jskip_ws(&j);
    int got_input = 0;
    if (j.p < j.end && *j.p == '}') { j.p++; r->err = 1; r->errmsg = "missing 'input'"; return; }
    for (;;) {
        jskip_ws(&j);
        buf key = {0};
        if (jstring(&j, &key)) { free(key.p); r->err = 1; r->errmsg = "malformed JSON (key)"; return; }
        buf_putc(&key, '\0');
        jskip_ws(&j);
        if (j.p >= j.end || *j.p != ':') { free(key.p); r->err = 1; r->errmsg = "malformed JSON (colon)"; return; }
        j.p++; jskip_ws(&j);

        if (strcmp(key.p, "input") == 0) {
            got_input = 1;
            if (j.p < j.end && *j.p == '"') {
                char *s = jstring_dup(&j);
                if (!s) { free(key.p); r->err = 1; r->errmsg = "malformed input string"; return; }
                req_push(r, s);
            } else if (j.p < j.end && *j.p == '[') {
                j.p++; jskip_ws(&j);
                if (j.p < j.end && *j.p == ']') { j.p++; }
                else for (;;) {
                    jskip_ws(&j);
                    if (j.p >= j.end || *j.p != '"') { free(key.p); r->err = 1; r->errmsg = "input array must contain strings"; return; }
                    char *s = jstring_dup(&j);
                    if (!s) { free(key.p); r->err = 1; r->errmsg = "malformed input string"; return; }
                    req_push(r, s);
                    jskip_ws(&j);
                    if (j.p < j.end && *j.p == ',') { j.p++; continue; }
                    if (j.p < j.end && *j.p == ']') { j.p++; break; }
                    free(key.p); r->err = 1; r->errmsg = "malformed input array"; return;
                }
            } else { free(key.p); r->err = 1; r->errmsg = "'input' must be string or array of strings"; return; }
        } else if (strcmp(key.p, "encoding_format") == 0) {
            char *s = jstring_dup(&j);
            if (s) { r->base64 = (strcmp(s, "base64") == 0); free(s); }
            else { free(key.p); r->err = 1; r->errmsg = "bad encoding_format"; return; }
        } else if (strcmp(key.p, "input_type") == 0) {
            char *s = jstring_dup(&j);
            if (s) {
                if (strcmp(s, "query") == 0) r->is_query = 1;
                else if (strcmp(s, "passage") == 0 || strcmp(s, "document") == 0) r->is_query = 0;
                free(s);
            } else { free(key.p); r->err = 1; r->errmsg = "bad input_type"; return; }
        } else if (strcmp(key.p, "model") == 0) {
            char *s = jstring_dup(&j);
            if (s) {
                if (strstr(s, "query")) r->is_query = 1;
                else if (strstr(s, "passage") || strstr(s, "doc")) r->is_query = 0;
                free(s);
            } else { free(key.p); r->err = 1; r->errmsg = "bad model"; return; }
        } else {
            if (jskip(&j)) { free(key.p); r->err = 1; r->errmsg = "malformed JSON value"; return; }
        }
        free(key.p);
        jskip_ws(&j);
        if (j.p < j.end && *j.p == ',') { j.p++; continue; }
        if (j.p < j.end && *j.p == '}') { j.p++; break; }
        r->err = 1; r->errmsg = "malformed JSON object"; return;
    }
    if (!got_input) { r->err = 1; r->errmsg = "missing 'input'"; }
}

/* ================================ HTTP ================================ */
static void writen(int fd, const char *p, size_t n) {
    while (n) {
        ssize_t w = write(fd, p, n);
        if (w <= 0) { if (errno == EINTR) continue; return; }
        p += w; n -= (size_t)w;
    }
}
static void send_resp(int fd, int code, const char *status, const char *ctype, const char *body, size_t blen) {
    buf h = {0};
    buf_printf(&h, "HTTP/1.1 %d %s\r\nContent-Type: %s\r\nContent-Length: %zu\r\nConnection: close\r\n\r\n",
               code, status, ctype, blen);
    writen(fd, h.p, h.len); free(h.p);
    if (blen) writen(fd, body, blen);
}
static void send_error(int fd, int code, const char *status, const char *type, const char *msg) {
    buf b = {0};
    buf_puts(&b, "{\"error\":{\"message\":\"");
    for (const char *s = msg; *s; s++) { if (*s == '"' || *s == '\\') buf_putc(&b, '\\'); buf_putc(&b, *s); }
    buf_printf(&b, "\",\"type\":\"%s\",\"code\":null}}", type);
    send_resp(fd, code, status, "application/json", b.p, b.len);
    free(b.p);
}

static void handle_embeddings(int fd, const char *body, size_t len) {
    req r = {0};
    parse_req(body, len, &r);
    if (r.err) { send_error(fd, 400, "Bad Request", "invalid_request_error", r.errmsg ? r.errmsg : "bad request"); req_free(&r); return; }
    if (r.ninputs == 0) { send_error(fd, 400, "Bad Request", "invalid_request_error", "'input' must not be empty"); req_free(&r); return; }

    int is_q = r.is_query >= 0 ? r.is_query : G_default_query;
    int n = r.ninputs, D = G_dim;
    float *emb = (float *)malloc((size_t)n * D * sizeof(float));
    if (!emb) { send_error(fd, 500, "Internal Server Error", "server_error", "out of memory"); req_free(&r); return; }

    /* serialize the heavy compute so each request gets all cores and we never
     * oversubscribe OpenMP across concurrent connections */
    pthread_mutex_lock(&G_compute);
    int rc = e5_embed_batch(G_model, (const char **)r.inputs, n, is_q, emb);
    long total_tok = 0;
    for (int i = 0; i < n; i++) total_tok += e5_token_count(G_model, r.inputs[i], is_q);
    pthread_mutex_unlock(&G_compute);

    if (rc) { free(emb); send_error(fd, 500, "Internal Server Error", "server_error", "inference failed"); req_free(&r); return; }

    buf out = {0};
    buf_puts(&out, "{\"object\":\"list\",\"data\":[");
    for (int i = 0; i < n; i++) {
        if (i) buf_putc(&out, ',');
        buf_puts(&out, "{\"object\":\"embedding\",\"index\":");
        buf_printf(&out, "%d,\"embedding\":", i);
        const float *e = emb + (size_t)i * D;
        if (r.base64) {
            buf_putc(&out, '"');
            b64_encode(&out, (const uint8_t *)e, (size_t)D * sizeof(float));
            buf_putc(&out, '"');
        } else {
            buf_putc(&out, '[');
            for (int k = 0; k < D; k++) { if (k) buf_putc(&out, ','); buf_printf(&out, "%.7g", e[k]); }
            buf_putc(&out, ']');
        }
        buf_putc(&out, '}');
    }
    buf_printf(&out, "],\"model\":\"%s\",\"usage\":{\"prompt_tokens\":%ld,\"total_tokens\":%ld}}",
               G_model_name, total_tok, total_tok);
    send_resp(fd, 200, "OK", "application/json", out.p, out.len);
    free(out.p); free(emb); req_free(&r);
}

static void handle_connection(int fd) {
    struct timeval tv = { 30, 0 };
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

    /* read until end of headers (track scan pos to stay linear) */
    buf in = {0};
    size_t hdr_end = 0, scan = 0;
    for (;;) {
        buf_reserve(&in, 65536);
        ssize_t got = read(fd, in.p + in.len, in.cap - in.len);
        if (got <= 0) { free(in.p); return; }
        in.len += (size_t)got;
        for (; scan + 4 <= in.len; scan++)
            if (memcmp(in.p + scan, "\r\n\r\n", 4) == 0) { hdr_end = scan + 4; break; }
        if (hdr_end) break;
        if (in.len > 1u << 20) { free(in.p); return; }   /* header too large */
    }
    /* method + path */
    char method[8] = {0}, path[256] = {0};
    sscanf(in.p, "%7s %255s", method, path);
    /* Content-Length (case-insensitive) */
    size_t clen = 0;
    for (size_t i = 0; i + 15 < hdr_end; i++) {
        if (strncasecmp(in.p + i, "Content-Length:", 15) == 0) {
            clen = strtoul(in.p + i + 15, NULL, 10); break;
        }
    }
    if (clen > MAX_BODY) { send_error(fd, 413, "Payload Too Large", "invalid_request_error", "request body too large"); free(in.p); return; }

    /* routes that need no body */
    if (strcmp(method, "GET") == 0) {
        if (strcmp(path, "/") == 0 || strcmp(path, "/health") == 0) {
            const char *ok = "{\"status\":\"ok\"}";
            send_resp(fd, 200, "OK", "application/json", ok, strlen(ok));
        } else if (strcmp(path, "/v1/models") == 0) {
            buf b = {0};
            buf_printf(&b, "{\"object\":\"list\",\"data\":[{\"id\":\"%s\",\"object\":\"model\",\"owned_by\":\"nanoE5.c\"}]}", G_model_name);
            send_resp(fd, 200, "OK", "application/json", b.p, b.len); free(b.p);
        } else send_error(fd, 404, "Not Found", "invalid_request_error", "unknown route");
        free(in.p); return;
    }
    if (strcmp(method, "POST") != 0) { send_error(fd, 405, "Method Not Allowed", "invalid_request_error", "use POST"); free(in.p); return; }

    /* read the rest of the body */
    size_t have = in.len - hdr_end;
    while (have < clen) {
        buf_reserve(&in, clen - have);
        ssize_t got = read(fd, in.p + in.len, clen - have);
        if (got <= 0) { free(in.p); return; }      /* client gave up */
        in.len += (size_t)got; have += (size_t)got;
    }
    const char *body = in.p + hdr_end;

    if (strcmp(path, "/v1/embeddings") == 0 || strcmp(path, "/embeddings") == 0)
        handle_embeddings(fd, body, clen);
    else
        send_error(fd, 404, "Not Found", "invalid_request_error", "unknown route");
    free(in.p);
}

typedef struct { int fd; } conn;
static void *worker(void *arg) {
    int fd = ((conn *)arg)->fd; free(arg);
    handle_connection(fd);
    shutdown(fd, SHUT_RDWR); close(fd);
    sem_post(&G_slots);
    return NULL;
}

static int run_server(const char *host, int port, int threads) {
    signal(SIGPIPE, SIG_IGN);
#ifdef _OPENMP
    if (threads > 0) omp_set_num_threads(threads);
#endif
    sem_init(&G_slots, 0, MAX_CONN);

    int s = socket(AF_INET, SOCK_STREAM, 0);
    int one = 1; setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET; addr.sin_port = htons(port);
    if (inet_pton(AF_INET, host, &addr.sin_addr) != 1) addr.sin_addr.s_addr = INADDR_ANY;
    if (bind(s, (struct sockaddr *)&addr, sizeof(addr)) < 0) { perror("bind"); return 1; }
    if (listen(s, 512) < 0) { perror("listen"); return 1; }
    fprintf(stderr, "e5 server listening on http://%s:%d  (model: %s, dim %d, default %s)\n",
            host, port, G_model_name, G_dim, G_default_query ? "query" : "passage");

    for (;;) {
        sem_wait(&G_slots);
        int fd = accept(s, NULL, NULL);
        if (fd < 0) { sem_post(&G_slots); if (errno == EINTR) continue; continue; }
        int one2 = 1; setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one2, sizeof(one2));
        conn *c = (conn *)malloc(sizeof(conn)); c->fd = fd;
        pthread_t th;
        if (pthread_create(&th, NULL, worker, c) != 0) { close(fd); free(c); sem_post(&G_slots); continue; }
        pthread_detach(th);
    }
    return 0;
}

/* ================================ main ================================ */
static e5_model *load_default(const char *model_path) {
    if (model_path) return e5_load(model_path);
#ifdef E5_EMBED
    size_t sz = (size_t)(_binary_model_bin_end - _binary_model_bin_start);
    return e5_load_mem(_binary_model_bin_start, sz);
#else
    fprintf(stderr, "e5: no embedded model in this build; pass --model PATH\n");
    return NULL;
#endif
}

static void usage(const char *a0) {
    fprintf(stderr,
        "usage:\n"
        "  %s --server [--host H] [--port P] [--threads N] [--default-type query|passage] [--model FILE]\n"
        "  %s [--model FILE] <query|passage> \"text\"\n", a0, a0);
}

int main(int argc, char **argv) {
    const char *host = "0.0.0.0", *model_path = NULL;
    int port = 8000, server = 0, threads = 0;
    const char *cli_mode = NULL, *cli_text = NULL;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--server")) server = 1;
        else if (!strcmp(argv[i], "--host") && i + 1 < argc) host = argv[++i];
        else if (!strcmp(argv[i], "--port") && i + 1 < argc) port = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--model") && i + 1 < argc) model_path = argv[++i];
        else if (!strcmp(argv[i], "--default-type") && i + 1 < argc) G_default_query = strcmp(argv[++i], "query") == 0;
        else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) { usage(argv[0]); return 0; }
        else if (!cli_mode) cli_mode = argv[i];
        else if (!cli_text) cli_text = argv[i];
    }

    G_model = load_default(model_path);
    if (!G_model) return 1;
    G_dim = e5_dim(G_model);

    if (server) return run_server(host, port, threads);

    if (!cli_mode) { usage(argv[0]); e5_free(G_model); return 1; }
    int is_q = strcmp(cli_mode, "query") == 0;
    int is_p = strcmp(cli_mode, "passage") == 0;
    if (!is_q && !is_p) { usage(argv[0]); e5_free(G_model); return 1; }
    const char *text = cli_text ? cli_text : "";
    float *out = (float *)malloc(sizeof(float) * G_dim);
    e5_embed(G_model, text, is_q, out);
    printf("dim=%d  [", G_dim);
    for (int i = 0; i < 8 && i < G_dim; i++) printf("%.5f ", out[i]);
    printf("...]\n");
    free(out); e5_free(G_model);
    return 0;
}
