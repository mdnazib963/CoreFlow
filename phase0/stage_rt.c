#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <stdatomic.h>
#include <immintrin.h>
#include "cf_model.h"
#include "cf_json.h"
extern unsigned long long cf_cyc_gemv, cf_cyc_act, cf_cyc_attn, cf_cyc_head,
                          cf_cyc_rope, cf_cyc_scor, cf_cyc_soft, cf_cyc_val;
#ifdef _WIN32
#include <windows.h>
#else
#include <time.h>
#include <pthread.h>
#endif

#define MAX_D_MSG 1024
#define RING_CAP 32
#define CORE_ORDER_MAX 8

typedef struct {
    int32_t sid, tok, kind, pos;
    float x[MAX_D_MSG];
} msg_t;

typedef struct {
    _Alignas(128) _Atomic size_t head;
    char pad0[128 - sizeof(size_t)];
    _Alignas(128) _Atomic size_t tail;
    char pad1[128 - sizeof(size_t)];
    char *slots;
    size_t stride, cap, mask;
} ring_t;

typedef struct {
    int lo, hi;
    long cost_mac;
} part_t;

typedef struct {
    int embed, head;
    long bytes;
} unit_ir_t;

typedef struct {
    int off, len;
} range_t;

typedef struct rt rt_t;
typedef struct group {
    rt_t *rt;
    int gid, lo, hi, core;
    int embed, head;
    cf_scratch_t sc;
    ring_t *in, *out;
    int pf_r, pf_i;
} group_t;

struct rt {
    cf_model_t m;
    cf_stream_t *st;
    int K;
    uint16_t *data;
    size_t ndata;
    int prompt;
    int *c_prompt_i, *c_pos, *c_inflight, *c_pending, *c_di;
    double *t_push;
    ring_t *rings;
    ring_t rout;
    group_t *grp;
    int N;
    int units_n;
    unit_ir_t *units;
    part_t *part;
    range_t *all_ranges;
    int *unit_range_off;
    volatile int stop;
    int *lat;
    int nlat, nlat_cap;
    int decode_tok, prefill_tok;
    int cores[CORE_ORDER_MAX];
    int ncores;
    void *th[CORE_ORDER_MAX];
    int nth;
    int no_pf;
    unsigned long long l3_bytes;
    long long home_bytes;
    int *conv_ix;
    int *conv_blk;
    int nconv_b;
    float *cvs;
    int *cvs_pos;
    int quiet;
    long sp_drafted, sp_accepted, sp_flushes, sp_cycles;
    double sp_draft_s, sp_cycle_s;
};

static int ring_init(ring_t *r) {
    r->stride = (sizeof(msg_t) + 63) & ~(size_t)63;
    r->cap = RING_CAP;
    r->mask = RING_CAP - 1;
    r->slots = (char *)calloc(r->cap, r->stride);
    if (!r->slots) return -1;
    atomic_init(&r->head, 0);
    atomic_init(&r->tail, 0);
    return 0;
}

static int ring_push(ring_t *r, const msg_t *m) {
    size_t h = atomic_load_explicit(&r->head, memory_order_relaxed);
    if (h - atomic_load_explicit(&r->tail, memory_order_acquire) >= r->cap)
        return 0;
    memcpy(r->slots + (h & r->mask) * r->stride, m, sizeof(msg_t));
    atomic_store_explicit(&r->head, h + 1, memory_order_release);
    return 1;
}

static int ring_pop(ring_t *r, msg_t *m) {
    size_t t = atomic_load_explicit(&r->tail, memory_order_relaxed);
    if (t >= atomic_load_explicit(&r->head, memory_order_acquire))
        return 0;
    memcpy(m, r->slots + (t & r->mask) * r->stride, sizeof(msg_t));
    atomic_store_explicit(&r->tail, t + 1, memory_order_release);
    return 1;
}

static void push_wait(rt_t *rt, ring_t *r, const msg_t *m) {
    while (!ring_push(r, m)) {
        if (rt->stop) return;
        _mm_pause();
    }
}

static int pop_wait(rt_t *rt, ring_t *r, msg_t *m) {
    for (;;) {
        if (ring_pop(r, m)) return 1;
        if (rt->stop) return 0;
        _mm_pause();
    }
}

static void prefetch_step(group_t *G) {
    rt_t *rt = G->rt;
    volatile long long sink = 0;
    int r0 = rt->unit_range_off[G->lo];
    int r1 = rt->unit_range_off[G->hi + 1];
    int n = 0;
    if (rt->no_pf) {
        _mm_pause();
        return;
    }
    if (r1 <= r0) {
        _mm_pause();
        return;
    }
    while (n < 512) {
        int r = r0 + G->pf_r;
        int off, len;
        if (G->pf_r >= r1 - r0) {
            G->pf_r = 0;
            G->pf_i = 0;
            continue;
        }
        off = rt->all_ranges[r].off;
        len = rt->all_ranges[r].len;
        if (G->pf_i >= len) {
            G->pf_r++;
            G->pf_i = 0;
            continue;
        }
        sink += rt->m.base[off + G->pf_i];
        G->pf_i += 64;
        n++;
    }
    (void)sink;
}

static void run_blocks(group_t *G, cf_stream_t *s, int pos, float *x) {
    int nb = (int)G->rt->m.h.nblocks;
    int b_lo = G->lo == 0 ? 0 : G->lo - 1;
    int b_hi = G->hi < nb ? G->hi : nb;
    if (b_lo < b_hi)
        cf_blocks(&G->rt->m, s, &G->sc, pos, b_lo, b_hi, x);
}

static size_t cv_bytes(const cf_model_t *m) {
    return (size_t)m->h.d * (m->h.k - 1) * sizeof(float);
}

static void cvs_save(rt_t *rt, int block, int pos, const float *cbuf) {
    int ci = rt->conv_ix[block];
    int sl = pos & 7;
    size_t cb = cv_bytes(&rt->m);
    if (ci < 0 || ci >= rt->nconv_b || block < 0 || block >= (int)rt->m.h.nblocks)
        return;
    memcpy((char *)rt->cvs + ((size_t)ci * 8 + sl) * cb, cbuf, cb);
    rt->cvs_pos[ci * 8 + sl] = pos;
}

static void cvs_restore(rt_t *rt, int pos) {
    int j, sl;
    size_t cb;
    if (pos < 0) return;
    cb = cv_bytes(&rt->m);
    sl = pos & 7;
    for (j = 0; j < rt->nconv_b; j++)
        if (rt->cvs_pos[j * 8 + sl] == pos)
            memcpy(rt->st[0].bs[rt->conv_blk[j]].cbuf,
                   (char *)rt->cvs + ((size_t)j * 8 + sl) * cb, cb);
}

static void cvs_save_u(rt_t *rt, int lo, int hi, int pos, cf_stream_t *s) {
    int u;
    for (u = lo; u <= hi; u++) {
        int b;
        if (u < 1 || u > rt->units_n - 2) continue;
        b = u - 1;
        if (rt->conv_ix[b] >= 0) cvs_save(rt, b, pos, s->bs[b].cbuf);
    }
}

static void group_body(group_t *G) {
    rt_t *rt = G->rt;
    cf_pin(G->core);
    if (G->gid == 0) {
        int rr = 0;
        for (;;) {
            int did = 0, feeds = 0;
            msg_t om;
            if (rt->stop) return;
            while (ring_pop(&rt->rout, &om)) {
                int sid = om.sid;
                rt->c_inflight[sid] = 0;
                rt->c_pending[sid] = om.tok;
                if (rt->c_prompt_i[sid] >= rt->prompt &&
                    rt->nlat < rt->nlat_cap) {
                    rt->lat[rt->nlat++] =
                        (int)((cf_now() - rt->t_push[sid]) * 1e6);
                    rt->decode_tok++;
                }
                did = 1;
            }
            while (feeds < 2) {
                int k, fed = 0;
                for (k = 0; k < rt->K; k++) {
                    int sid = (rr + k) % rt->K;
                    int tok, is_prompt;
                    msg_t out;
                    if (rt->c_inflight[sid]) continue;
                    is_prompt = rt->c_prompt_i[sid] < rt->prompt;
                    if (!is_prompt && rt->c_pos[sid] >= (int)rt->m.h.ctx) {
                        cf_stream_reset(&rt->st[sid], &rt->m);
                        rt->c_prompt_i[sid] = 0;
                        rt->c_pos[sid] = 0;
                        rt->c_di[sid] += (int)rt->m.h.ctx;
                        if (rt->c_di[sid] + rt->prompt + 4 > (int)rt->ndata)
                            rt->c_di[sid] = 0;
                        is_prompt = 1;
                    }
                    if (is_prompt)
                        tok = rt->data[rt->c_di[sid] + rt->c_prompt_i[sid]++];
                    else
                        tok = rt->c_pending[sid];
                    rt->st[sid].pos = rt->c_pos[sid];
                    rt->t_push[sid] = cf_now();
                    if (G->embed) cf_embed(&rt->m, tok, G->sc.x);
                    run_blocks(G, &rt->st[sid], rt->c_pos[sid], G->sc.x);
                    memset(&out, 0, sizeof(out));
                    out.sid = sid;
                    out.pos = rt->c_pos[sid];
                    memcpy(out.x, G->sc.x, (size_t)rt->m.h.d * sizeof(float));
                    push_wait(rt, rt->rings + G->gid, &out);
                    rt->c_inflight[sid] = 1;
                    rt->c_pos[sid]++;
                    if (is_prompt) rt->prefill_tok++;
                    rr = (sid + 1) % rt->K;
                    feeds++;
                    fed = 1;
                    break;
                }
                if (!fed) break;
            }
            if (!did && feeds == 0) {
                for (;;) {
                    if (ring_pop(&rt->rout, &om)) break;
                    if (rt->stop) return;
                    prefetch_step(G);
                }
                rt->c_inflight[om.sid] = 0;
                rt->c_pending[om.sid] = om.tok;
                if (rt->c_prompt_i[om.sid] >= rt->prompt &&
                    rt->nlat < rt->nlat_cap) {
                    rt->lat[rt->nlat++] =
                        (int)((cf_now() - rt->t_push[om.sid]) * 1e6);
                    rt->decode_tok++;
                }
            }
        }
    }
    for (;;) {
        msg_t m, o;
        cf_stream_t *s;
        if (rt->stop) return;
        if (!ring_pop(G->in, &m)) {
            prefetch_step(G);
            continue;
        }
        s = &rt->st[m.sid];
        s->pos = m.pos;
        if (m.sid == 0) cvs_save_u(rt, G->lo, G->hi, m.pos, s);
        run_blocks(G, s, m.pos, m.x);
        if (G->head) {
            cf_head(&rt->m, s, &G->sc, m.x);
            memset(&o, 0, sizeof(o));
            o.sid = m.sid;
            o.pos = m.pos;
            o.tok = cf_argmax(s->logits, (int)rt->m.h.vocab);
            push_wait(rt, &rt->rout, &o);
        } else {
            push_wait(rt, G->out, &m);
        }
    }
}

#ifdef _WIN32
static DWORD WINAPI group_thread(LPVOID p) {
    group_body((group_t *)p);
    return 0;
}
#else
static void *group_thread(void *p) {
    group_body((group_t *)p);
    return NULL;
}
#endif

static void start_groups(rt_t *rt, int from, int to) {
    int g;
    rt->stop = 0;
    for (g = from; g < to; g++) {
#ifdef _WIN32
        rt->th[g] = CreateThread(NULL, 0, group_thread, &rt->grp[g], 0, NULL);
#else
        rt->th[g] = (void *)1;
        {
            pthread_t t;
            pthread_create(&t, NULL, group_thread, &rt->grp[g]);
            rt->th[g] = (void *)t;
        }
#endif
    }
    rt->nth = to;
}

static void stop_groups(rt_t *rt) {
    int g;
    rt->stop = 1;
    for (g = 0; g < rt->nth; g++) {
#ifdef _WIN32
        if (rt->th[g]) {
            WaitForSingleObject((HANDLE)rt->th[g], 5000);
            CloseHandle((HANDLE)rt->th[g]);
            rt->th[g] = NULL;
        }
#else
        rt->th[g] = NULL;
#endif
    }
    rt->nth = 0;
}

static char *slurp(const char *path, size_t *out_n) {
    FILE *f = fopen(path, "rb");
    long sz;
    char *b;
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return NULL; }
    fseek(f, 0, SEEK_END);
    sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    b = (char *)malloc((size_t)sz + 1);
    if (!b) { fclose(f); return NULL; }
    if (fread(b, 1, (size_t)sz, f) != (size_t)sz) { free(b); fclose(f); return NULL; }
    fclose(f);
    b[sz] = 0;
    if (out_n) *out_n = (size_t)sz;
    return b;
}

static uint16_t *load_bin(const char *bin, size_t *out_n) {
    FILE *f = fopen(bin, "rb");
    long sz;
    size_t n;
    uint16_t *data;
    if (!f) { fprintf(stderr, "cannot open %s\n", bin); return NULL; }
    fseek(f, 0, SEEK_END);
    sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    n = (size_t)sz / 2;
    data = (uint16_t *)cf_xmalloc(n * 2);
    if (fread(data, 2, n, f) != n) { free(data); fclose(f); return NULL; }
    fclose(f);
    *out_n = n;
    return data;
}

static int load_ir_buf(rt_t *rt, char *src, int N) {
    char err[256];
    js_val *root, *units, *parts, *arch, *arr;
    int i;
    if (!src) return -1;
    root = js_parse(src, err, sizeof(err));
    free(src);
    if (!root) { fprintf(stderr, "json: %s\n", err); return -1; }
    arch = js_obj_get(root, "arch");
    if (!arch ||
        (int)js_num(js_obj_get(arch, "d")) != (int)rt->m.h.d ||
        (int)js_num(js_obj_get(arch, "vocab")) != (int)rt->m.h.vocab ||
        (int)js_num(js_obj_get(arch, "ctx")) != (int)rt->m.h.ctx ||
        (int)js_num(js_obj_get(arch, "nblocks")) != (int)rt->m.h.nblocks) {
        fprintf(stderr, "IR arch mismatch vs model\n");
        js_free(root);
        return -1;
    }
    units = js_obj_get(root, "units");
    if (!units || units->t != JS_ARR) { js_free(root); return -1; }
    rt->units_n = units->n;
    rt->units = (unit_ir_t *)cf_xmalloc(sizeof(unit_ir_t) * (size_t)units->n);
    rt->unit_range_off = (int *)cf_xmalloc(sizeof(int) * (size_t)(units->n + 1));
    {
        int total = 0, idx = 0;
        for (i = 0; i < units->n; i++) {
            js_val *u = js_arr_get(units, i);
            js_val *secs = js_obj_get(u, "sections");
            int j;
            rt->units[i].embed = js_bool(js_obj_get(u, "embed"));
            rt->units[i].head = js_bool(js_obj_get(u, "head"));
            rt->units[i].bytes = 0;
            for (j = 0; secs && j < secs->n; j++) {
                js_val *rgs = js_obj_get(js_arr_get(secs, j), "ranges");
                total += rgs->n;
            }
        }
        rt->all_ranges = (range_t *)cf_xmalloc(sizeof(range_t) * (size_t)(total ? total : 1));
        for (i = 0; i < units->n; i++) {
            js_val *u = js_arr_get(units, i);
            js_val *secs = js_obj_get(u, "sections");
            int j;
            rt->unit_range_off[i] = idx;
            for (j = 0; secs && j < secs->n; j++) {
                js_val *rgs = js_obj_get(js_arr_get(secs, j), "ranges");
                int r;
                for (r = 0; r < rgs->n; r++) {
                    js_val *rg = js_arr_get(rgs, r);
                    rt->all_ranges[idx].off = (int)js_num(js_obj_get(rg, "off"));
                    rt->all_ranges[idx].len = (int)js_num(js_obj_get(rg, "len"));
                    rt->units[i].bytes += rt->all_ranges[idx].len;
                    idx++;
                }
            }
        }
        rt->unit_range_off[units->n] = idx;
    }
    parts = js_obj_get(root, "partitions");
    if (!parts || parts->t != JS_OBJ) { js_free(root); return -1; }
    {
        char key[16];
        int g;
        snprintf(key, sizeof(key), "%d", N);
        arr = js_obj_get(parts, key);
        if (!arr || arr->t != JS_ARR || (int)arr->n < N) {
            fprintf(stderr, "no partition for N=%d\n", N);
            js_free(root);
            return -1;
        }
        rt->part = (part_t *)cf_xmalloc(sizeof(part_t) * (size_t)N);
        for (g = 0; g < N; g++) {
            js_val *pg = js_arr_get(arr, g);
            rt->part[g].lo = (int)js_num(js_obj_get(pg, "lo"));
            rt->part[g].hi = (int)js_num(js_obj_get(pg, "hi"));
            rt->part[g].cost_mac = (long)js_num(js_obj_get(pg, "cost_mac"));
        }
    }
    js_free(root);
    return 0;
}

static int load_ir(rt_t *rt, const char *path, int N) {
    char *src = slurp(path, NULL);
    if (!src) return -1;
    return load_ir_buf(rt, src, N);
}

static int ends_with(const char *s, const char *suf) {
    size_t a = strlen(s), b = strlen(suf);
    return a >= b && memcmp(s + a - b, suf, b) == 0;
}

static int cf_container_load(rt_t *rt, const char *path, int N) {
    size_t cn = 0;
    char *cb = slurp(path, &cn);
    uint32_t ver = 0, nent = 0;
    uint64_t iroff = 0, irsz = 0, moff = 0, msz = 0;
    int have_ir = 0, have_model = 0;
    uint32_t i;
    if (!cb) return -1;
    if (cn < 16 || memcmp(cb, "CORECF01", 8) != 0) {
        fprintf(stderr, "%s: bad container magic\n", path);
        free(cb);
        return -1;
    }
    memcpy(&ver, cb + 8, 4);
    memcpy(&nent, cb + 12, 4);
    if (ver != 1 || nent == 0 || nent > 16 ||
        (uint64_t)cn < 16ULL + 24ULL * nent) {
        fprintf(stderr, "%s: bad container header\n", path);
        free(cb);
        return -1;
    }
    for (i = 0; i < nent; i++) {
        const char *e = cb + 16 + 24 * i;
        uint64_t off = 0, sz = 0;
        memcpy(&off, e + 8, 8);
        memcpy(&sz, e + 16, 8);
        if (off + sz > (uint64_t)cn) {
            fprintf(stderr, "%s: entry out of range\n", path);
            free(cb);
            return -1;
        }
        if (!memcmp(e, "ir", 2)) {
            iroff = off; irsz = sz; have_ir = 1;
        } else if (!memcmp(e, "model", 5)) {
            moff = off; msz = sz; have_model = 1;
        }
    }
    if (!have_ir || !have_model || !irsz || !msz) {
        fprintf(stderr, "%s: missing ir/model entry\n", path);
        free(cb);
        return -1;
    }
    if (cf_model_load_mem(&rt->m, cb + moff, (size_t)msz) != 0) {
        free(cb);
        return -1;
    }
    {
        char *irb = (char *)malloc((size_t)irsz + 1);
        if (!irb) { free(cb); return -1; }
        memcpy(irb, cb + iroff, (size_t)irsz);
        irb[irsz] = 0;
        free(cb);
        return load_ir_buf(rt, irb, N);
    }
}

static void do_touch(group_t *G) {
    rt_t *rt = G->rt;
    volatile long long sink = 0;
    int u;
    cf_pin(G->core);
    for (u = G->lo; u <= G->hi; u++) {
        int r;
        for (r = rt->unit_range_off[u]; r < rt->unit_range_off[u + 1]; r++) {
            int off = rt->all_ranges[r].off;
            int len = rt->all_ranges[r].len;
            int i;
            for (i = 0; i < len; i += 64)
                sink += rt->m.base[off + i];
        }
    }
    (void)sink;
}

static int sync_feed(rt_t *rt, group_t *G0, int sid, int tok, int *out_tok) {
    msg_t out, om;
    rt->st[sid].pos = rt->c_pos[sid];
    if (rt->N == 1) {
        cf_forward(&rt->m, &rt->st[sid], &G0->sc, tok);
        *out_tok = cf_argmax(rt->st[sid].logits, (int)rt->m.h.vocab);
        rt->c_pos[sid]++;
        return 0;
    }
    if (sid == 0) cvs_save_u(rt, G0->lo, G0->hi, rt->c_pos[sid], &rt->st[sid]);
    if (G0->embed) cf_embed(&rt->m, tok, G0->sc.x);
    run_blocks(G0, &rt->st[sid], rt->c_pos[sid], G0->sc.x);
    memset(&out, 0, sizeof(out));
    out.sid = sid;
    out.pos = rt->c_pos[sid];
    memcpy(out.x, G0->sc.x, (size_t)rt->m.h.d * sizeof(float));
    push_wait(rt, rt->rings, &out);
    if (!pop_wait(rt, &rt->rout, &om)) return -1;
    rt->c_pos[sid]++;
    *out_tok = om.tok;
    return 0;
}

static int do_gen(rt_t *rt, const char *binpath, int ntok) {
    group_t *G0 = &rt->grp[0];
    size_t n = 0;
    uint16_t *data = load_bin(binpath, &n);
    int *seq;
    int p, i, pending = 0, count = 0;
    int prompt = rt->prompt;
    double t0 = 0;
    if (!data) return 1;
    seq = (int *)cf_xmalloc(sizeof(int) * (size_t)(ntok + prompt + 4));
    cf_stream_reset(&rt->st[0], &rt->m);
    rt->c_pos[0] = 0;
    if (rt->N > 1) start_groups(rt, 1, rt->N);
    cf_pin(G0->core);
    for (p = 0; p < prompt && p + 1 < (int)n; p++) {
        int o;
        if (sync_feed(rt, G0, 0, (int)data[p], &o)) goto done;
        pending = o;
    }
    t0 = cf_now();
    for (i = 0; i < ntok && rt->c_pos[0] < (int)rt->m.h.ctx; i++) {
        int o;
        seq[count++] = pending;
        if (sync_feed(rt, G0, 0, pending, &o)) goto done;
        pending = o;
    }
done:
    if (rt->N > 1) stop_groups(rt);
    if (!rt->quiet) {
        printf("{\"mode\":\"gen\",\"N\":%d,\"prompt\":%d,\"tok_s\":%.2f,\"tokens\":[",
               rt->N, prompt,
               (count && cf_now() - t0 > 0) ? count / (cf_now() - t0) : 0.0);
        for (i = 0; i < count; i++)
            printf("%s%d", i ? "," : "", seq[i]);
        printf("]}\n");
    }
    free(seq);
    free(data);
    return 0;
}

#define SUITE_MAX_NEW 24
#define SUITE_STOP(t) ((t) == 0 || (t) == 1 || (t) == 2 || (t) == 13)

static int do_suite(rt_t *rt, const char *pf, const char *of) {
    group_t *G0 = &rt->grp[0];
    FILE *f, *fo;
    uint32_t np = 0, i;
    uint32_t first_ok = 0, parity_ok = 0, skipped = 0;
    long long emitted = 0;
    double t0 = 0;
    int rc = 1;
    f = fopen(pf, "rb");
    if (!f) {
        fprintf(stderr, "suite: cannot open %s\n", pf);
        return 1;
    }
    fo = fopen(of, "wb");
    if (!fo) {
        fprintf(stderr, "suite: cannot open %s\n", of);
        fclose(f);
        return 1;
    }
    if (fread(&np, 4, 1, f) != 1) {
        fprintf(stderr, "suite: bad header\n");
        goto out;
    }
    fwrite(&np, 4, 1, fo);
    if (rt->N > 1) start_groups(rt, 1, rt->N);
    cf_pin(G0->core);
    t0 = cf_now();
    for (i = 0; i < np; i++) {
        uint32_t plen = 0, alen = 0, tlen = 0, j, ne = 0, tok, first = 0;
        uint16_t *p, *g, *th, emit[SUITE_MAX_NEW];
        int o = 0;
        if (fread(&plen, 4, 1, f) != 1 || fread(&alen, 4, 1, f) != 1 ||
            fread(&tlen, 4, 1, f) != 1)
            goto done;
        if (plen > 8192 || alen > 64 || tlen > 64) {
            fprintf(stderr, "suite: record %u out of range\n", i);
            goto done;
        }
        p = (uint16_t *)cf_xmalloc(plen * 2 + 2);
        g = (uint16_t *)cf_xmalloc(alen * 2 + 2);
        th = (uint16_t *)cf_xmalloc(tlen * 2 + 2);
        if (fread(p, 2, plen, f) != plen || fread(g, 2, alen, f) != alen ||
            fread(th, 2, tlen, f) != tlen) {
            free(p);
            free(g);
            free(th);
            fprintf(stderr, "suite: record %u truncated\n", i);
            goto done;
        }
        if (plen + SUITE_MAX_NEW + 1 > (uint32_t)rt->m.h.ctx) {
            skipped++;
            first = 0;
            ne = 0;
            goto record;
        }
        cf_stream_reset(&rt->st[0], &rt->m);
        rt->c_pos[0] = 0;
        for (j = 0; j < plen; j++) {
            if (sync_feed(rt, G0, 0, (int)p[j], &o)) {
                free(p);
                free(g);
                free(th);
                goto done;
            }
        }
        tok = (uint32_t)o;
        first = tok;
        for (j = 0; j < SUITE_MAX_NEW; j++) {
            if (SUITE_STOP(tok)) break;
            emit[ne++] = (uint16_t)tok;
            if (sync_feed(rt, G0, 0, (int)tok, &o)) {
                free(p);
                free(g);
                free(th);
                goto done;
            }
            tok = (uint32_t)o;
        }
    record:
        if (alen && first == g[0]) first_ok++;
        if (ne == tlen && !memcmp(emit, th, ne * 2)) parity_ok++;
        emitted += ne;
        {
            uint32_t hdr[3] = { i, first, ne };
            fwrite(hdr, 4, 3, fo);
            fwrite(emit, 2, ne, fo);
        }
        free(p);
        free(g);
        free(th);
    }
    rc = 0;
done:
    if (rt->N > 1) stop_groups(rt);
    if (rc == 0) {
        double el = cf_now() - t0;
        printf("{\"mode\":\"suite\",\"N\":%d,\"prompts\":%u,\"skipped\":%u,"
               "\"first_ok\":%u,\"first_acc\":%.4f,\"torch_parity\":%u,"
               "\"parity_acc\":%.4f,\"emitted\":%lld,\"sec\":%.3f}\n",
               rt->N, np, skipped, first_ok,
               (double)first_ok / (double)(np ? np : 1), parity_ok,
               (double)parity_ok / (double)(np ? np : 1), emitted, el);
    }
out:
    fclose(f);
    fclose(fo);
    return rc;
}

static int do_ask(rt_t *rt, const char *pf, int ntok) {
    group_t *G0 = &rt->grp[0];
    FILE *f;
    uint32_t plen = 0, j, emitted = 0, tok;
    uint16_t *p;
    int o = 0;
    double t0 = 0;
    int rc = 1;
    const char *stopr = "length";
    f = fopen(pf, "rb");
    if (!f) {
        fprintf(stderr, "ask: cannot open %s\n", pf);
        return 1;
    }
    if (fread(&plen, 4, 1, f) != 1 || plen > 8192) {
        fprintf(stderr, "ask: bad prompt\n");
        fclose(f);
        return 1;
    }
    p = (uint16_t *)cf_xmalloc(plen * 2 + 2);
    if (fread(p, 2, plen, f) != plen) {
        fprintf(stderr, "ask: truncated prompt\n");
        fclose(f);
        free(p);
        return 1;
    }
    fclose(f);
    if (plen + 2 > (uint32_t)rt->m.h.ctx) {
        fprintf(stderr, "ask: prompt %u exceeds ctx %d\n", plen,
                (int)rt->m.h.ctx);
        free(p);
        return 1;
    }
    if (ntok < 1) ntok = 1;
    if (plen + (uint32_t)ntok + 1 > (uint32_t)rt->m.h.ctx)
        ntok = (int)rt->m.h.ctx - (int)plen - 1;
    if (rt->N > 1) start_groups(rt, 1, rt->N);
    cf_pin(G0->core);
    cf_stream_reset(&rt->st[0], &rt->m);
    rt->c_pos[0] = 0;
    for (j = 0; j < plen; j++)
        if (sync_feed(rt, G0, 0, (int)p[j], &o)) goto done;
    free(p);
    p = NULL;
    t0 = cf_now();
    tok = (uint32_t)o;
    while (emitted < (uint32_t)ntok) {
        if (SUITE_STOP(tok)) {
            stopr = tok == 2 ? "eos" : tok == 13 ? "newline"
                     : tok == 0 ? "pad" : "bos";
            break;
        }
        printf("{\"i\":%u,\"t\":%u,\"el\":%.6f}\n", emitted, tok,
               cf_now() - t0);
        fflush(stdout);
        if (sync_feed(rt, G0, 0, (int)tok, &o)) goto done;
        tok = (uint32_t)o;
        emitted++;
    }
    rc = 0;
done:
    if (rt->N > 1) stop_groups(rt);
    if (rc == 0) {
        double el = cf_now() - t0;
        printf("{\"done\":1,\"tokens\":%u,\"prompt\":%u,\"sec\":%.6f,"
               "\"tok_s\":%.2f,\"stop\":\"%s\"}\n",
               emitted, plen, el, el > 0 ? emitted / el : 0.0, stopr);
    }
    free(p);
    return rc;
}

typedef struct {
    cf_stream_t s;
} draft_t;

static int cmp_dbl(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y ? 1 : 0;
}

static void draft_init(draft_t *dt, rt_t *rt) {
    cf_stream_init(&dt->s, &rt->m);
    cf_stream_reset(&dt->s, &rt->m);
}

static void draft_snap(draft_t *dt, rt_t *rt) {
    int i;
    size_t cb = cv_bytes(&rt->m);
    for (i = 0; i < (int)rt->m.h.nblocks; i++) {
        if (rt->m.blk[i].is_attn) {
            dt->s.bs[i].ks = rt->st[0].bs[i].ks;
            dt->s.bs[i].vs = rt->st[0].bs[i].vs;
            dt->s.bs[i].kq = rt->st[0].bs[i].kq;
            dt->s.bs[i].vq = rt->st[0].bs[i].vq;
        } else {
            memcpy(dt->s.bs[i].cbuf, rt->st[0].bs[i].cbuf, cb);
        }
    }
    dt->s.pos = rt->st[0].pos;
}

static void cvs_drop(rt_t *rt, int from, int to) {
    int j, p;
    for (p = from; p <= to; p++)
        for (j = 0; j < rt->nconv_b; j++)
            rt->cvs_pos[j * 8 + (p & 7)] = -1;
}

static void spec_feed(rt_t *rt, group_t *G0, int tok, int pos) {
    msg_t out;
    rt->st[0].pos = pos;
    cvs_save_u(rt, G0->lo, G0->hi, pos, &rt->st[0]);
    if (G0->embed) cf_embed(&rt->m, tok, G0->sc.x);
    run_blocks(G0, &rt->st[0], pos, G0->sc.x);
    memset(&out, 0, sizeof(out));
    out.sid = 0;
    out.pos = pos;
    memcpy(out.x, G0->sc.x, (size_t)rt->m.h.d * sizeof(float));
    push_wait(rt, rt->rings, &out);
}

static int do_gen_spec(rt_t *rt, const char *binpath, int ntok, int kmax,
                       float tau) {
    group_t *G0 = &rt->grp[0];
    size_t n = 0;
    uint16_t *data = load_bin(binpath, &n);
    int *seq, *dtoks, *ov;
    int p, i, pending = 0, count = 0;
    int prompt = rt->prompt;
    long drafted = 0, accepted = 0, flushed = 0, cycles = 0;
    double draft_s = 0, cyc_s = 0, t0 = 0;
    double *cyc;
    draft_t ds;
    if (!data) return 1;
    if (kmax < 1) kmax = 1;
    if (kmax > 8) kmax = 8;
    seq = (int *)cf_xmalloc(sizeof(int) * (size_t)(ntok + prompt + kmax + 8));
    dtoks = (int *)cf_xmalloc(sizeof(int) * (size_t)(kmax + 4));
    ov = (int *)cf_xmalloc(sizeof(int) * (size_t)(kmax + 4));
    cyc = (double *)cf_xmalloc(sizeof(double) * (size_t)(ntok + kmax + 8));
    cf_stream_reset(&rt->st[0], &rt->m);
    rt->c_pos[0] = 0;
    draft_init(&ds, rt);
    start_groups(rt, 1, rt->N);
    cf_pin(G0->core);
    for (p = 0; p < prompt && p + 1 < (int)n; p++) {
        int o;
        if (sync_feed(rt, G0, 0, (int)data[p], &o)) goto done;
        pending = o;
    }
    t0 = cf_now();
    while (count < ntok && rt->c_pos[0] < (int)rt->m.h.ctx) {
        int P = rt->c_pos[0], kp = 1, e = 0, rej = -1;
        double c0 = cf_now(), d0;
        msg_t om;
        cvs_restore(rt, P);
        d0 = cf_now();
        dtoks[0] = pending;
        draft_snap(&ds, rt);
        for (i = 1; i < kmax && P + kp < (int)rt->m.h.ctx; i++) {
            float conf;
            ds.s.pos = P + kp - 1;
            dtoks[kp] = cf_draft(&rt->m, &ds.s, &G0->sc, dtoks[kp - 1]);
            conf = cf_top1_prob(ds.s.logits, (int)rt->m.h.vocab);
            kp++;
            if (conf < tau) {
                kp--;
                break;
            }
        }
        draft_s += cf_now() - d0;
        for (i = 0; i < kp; i++)
            spec_feed(rt, G0, dtoks[i], P + i);
        for (i = 0; i < kp; i++) {
            if (!pop_wait(rt, &rt->rout, &om)) goto done;
            ov[i] = om.tok;
        }
        if (count < ntok) seq[count++] = pending;
        e = 1;
        for (i = 0; i + 1 < kp; i++) {
            if (ov[i] == dtoks[i + 1]) {
                if (count < ntok) seq[count++] = dtoks[i + 1];
                e++;
                accepted++;
            } else {
                rej = i;
                break;
            }
        }
        if (rej < 0) {
            pending = ov[kp - 1];
        } else {
            if (count < ntok) seq[count++] = ov[rej];
            e++;
            cvs_restore(rt, P + rej + 1);
            cvs_drop(rt, P + rej + 2, P + kp - 1);
            spec_feed(rt, G0, ov[rej], P + rej + 1);
            if (!pop_wait(rt, &rt->rout, &om)) goto done;
            pending = om.tok;
        }
        cycles++;
        drafted += kp - 1;
        if (rej >= 0) flushed += (kp - 1) - rej;
        rt->c_pos[0] = P + e;
        cyc[cycles - 1] = (cf_now() - c0) * 1000.0;
        cyc_s += cf_now() - c0;
    }
done:
    stop_groups(rt);
    rt->sp_drafted = drafted;
    rt->sp_accepted = accepted;
    rt->sp_flushes = flushed;
    rt->sp_cycles = cycles;
    rt->sp_draft_s = draft_s;
    rt->sp_cycle_s = cyc_s;
    if (cycles > 1)
        qsort(cyc, (size_t)cycles, sizeof(double), cmp_dbl);
    if (!rt->quiet) {
        printf("{\"mode\":\"spec\",\"N\":%d,\"prompt\":%d,\"kmax\":%d,\"tau\":%.3f,"
               "\"cycles\":%ld,\"drafted\":%ld,\"accepted\":%ld,\"flushed\":%ld,"
               "\"acceptance\":%.4f,\"cycle_ms\":%.4f,\"cycle_p50_ms\":%.4f,"
               "\"cycle_p99_ms\":%.4f,\"draft_ms\":%.4f,\"draft_pct\":%.2f,"
               "\"tok_s\":%.2f,\"tokens\":[",
               rt->N, prompt, kmax, tau, cycles, drafted, accepted, flushed,
               drafted ? (double)accepted / (double)drafted : 0.0,
               cycles ? cyc_s * 1000.0 / cycles : 0.0,
               cycles ? cyc[cycles / 2] : 0.0,
               cycles ? cyc[(int)(0.99 * (cycles - 1))] : 0.0,
               cycles ? draft_s * 1000.0 / cycles : 0.0,
               cyc_s > 0 ? 100.0 * draft_s / cyc_s : 0.0,
               (cycles && cf_now() - t0 > 0) ? count / (cf_now() - t0) : 0.0);
        for (i = 0; i < count; i++)
            printf("%s%d", i ? "," : "", seq[i]);
        printf("]}\n");
    }
    free(seq);
    free(dtoks);
    free(ov);
    free(cyc);
    free(data);
    return 0;
}

static int do_draftprobe(rt_t *rt, const char *binpath, int steps) {
    group_t *G0 = &rt->grp[0];
    size_t n = 0;
    uint16_t *data = load_bin(binpath, &n);
    draft_t ds;
    int i;
    if (!data) return 1;
    if (steps < 1) steps = 32;
    if ((size_t)steps + 1 >= n) steps = (int)n - 1;
    cf_stream_reset(&rt->st[0], &rt->m);
    draft_init(&ds, rt);
    cf_pin(G0->core);
    printf("{\"mode\":\"draftprobe\",\"steps\":%d,\"pairs\":[", steps);
    for (i = 0; i < steps; i++) {
        int d;
        float conf;
        ds.s.pos = i;
        d = cf_draft(&rt->m, &ds.s, &G0->sc, (int)data[i]);
        conf = cf_top1_prob(ds.s.logits, (int)rt->m.h.vocab);
        printf("%s[%d,%d,%d,%.4f]", i ? "," : "", i, (int)data[i + 1], d, conf);
    }
    printf("]}\n");
    free(data);
    return 0;
}

static double probe_window(rt_t *rt, group_t *G0, uint16_t *data, size_t n,
                           int ntok, double *first_ms) {
    int p, pending = 0, i, done = 0;
    double t0, tw = -1;
    cf_stream_reset(&rt->st[0], &rt->m);
    rt->c_pos[0] = 0;
    for (p = 0; p < rt->prompt && p + 1 < (int)n; p++) {
        int o;
        if (sync_feed(rt, G0, 0, (int)data[p], &o)) break;
        pending = o;
    }
    t0 = cf_now();
    for (i = 0; i < ntok && rt->c_pos[0] < (int)rt->m.h.ctx; i++) {
        int o;
        double s0 = cf_now();
        if (sync_feed(rt, G0, 0, pending, &o)) break;
        if (i == 0) tw = (cf_now() - s0) * 1000.0;
        pending = o;
        done++;
    }
    *first_ms = tw;
    return cf_now() - t0;
}

static int do_probe(rt_t *rt, const char *binpath, int cycles, int ntok,
                    size_t evmb) {
    group_t *G0 = &rt->grp[0];
    size_t n = 0, i;
    uint16_t *data = load_bin(binpath, &n);
    double wt = 0, ct = 0, wf = 0, cfv = 0;
    int c;
    char *evbuf;
    if (!data) return 1;
    start_groups(rt, 1, rt->N);
    evbuf = (char *)cf_xmalloc(evmb * 1024 * 1024);
    for (i = 0; i < evmb * 1024 * 1024; i++) evbuf[i] = (char)i;
    cf_pin(G0->core);
    for (c = 0; c < cycles; c++) {
        double ts, fm;
        size_t di = (size_t)(c * 137) % (n / 2);
        ts = probe_window(rt, G0, data + di, n - di, ntok, &fm);
        wt += ts;
        wf += fm;
        for (i = 0; i < evmb * 1024 * 1024; i += 64)
            evbuf[i] = (char)(evbuf[i] + 1);
        ts = probe_window(rt, G0, data + di, n - di, ntok, &fm);
        ct += ts;
        cfv += fm;
    }
    stop_groups(rt);
    printf("{\"mode\":\"probe\",\"N\":%d,\"cycles\":%d,\"tok_per_cycle\":%d,"
           "\"evict_mb\":%zu,\"warm_tok_s\":%.2f,\"cold_tok_s\":%.2f,"
           "\"ratio_cold_over_warm\":%.3f,\"warm_first_ms\":%.4f,"
           "\"cold_first_ms\":%.4f,\"first_ratio\":%.3f}\n",
           rt->N, cycles, ntok, evmb,
           (double)cycles * ntok / wt, (double)cycles * ntok / ct,
           (ct / cycles) / (wt / cycles),
           wf / cycles, cfv / cycles,
           (cfv / cycles) / (wf / cycles));
    free(evbuf);
    free(data);
    return 0;
}

static int do_bench(rt_t *rt, const char *binpath, double seconds) {
    size_t n = 0;
    uint16_t *data = load_bin(binpath, &n);
    int i, *cp, ntok;
    double t0, el, mean = 0;
    unsigned long long c0g, c0a, c0t, c0h, c0r, c0s, c0f, c0v, tsch = 0;
    if (!data) return 1;
    rt->data = data;
    rt->ndata = n;
    cf_cyc_gemv = cf_cyc_act = cf_cyc_attn = cf_cyc_head = 0;
    cf_cyc_rope = cf_cyc_scor = cf_cyc_soft = cf_cyc_val = 0;
    {
        unsigned long long a = __rdtsc();
        Sleep(120);
        tsch = (__rdtsc() - a) / 120;
    }
    c0g = cf_cyc_gemv;
    c0a = cf_cyc_act;
    c0t = cf_cyc_attn;
    c0h = cf_cyc_head;
    c0r = cf_cyc_rope;
    c0s = cf_cyc_scor;
    c0f = cf_cyc_soft;
    c0v = cf_cyc_val;
    t0 = cf_now();
    if (rt->N == 1) {
        group_t *G0 = &rt->grp[0];
        int pending = 0;
        cf_pin(G0->core);
        while (cf_now() - t0 < seconds) {
            int is_prompt, tok, o;
            double f0;
            is_prompt = rt->c_prompt_i[0] < rt->prompt;
            if (!is_prompt && rt->c_pos[0] >= (int)rt->m.h.ctx) {
                cf_stream_reset(&rt->st[0], &rt->m);
                rt->c_prompt_i[0] = 0;
                rt->c_pos[0] = 0;
                rt->c_di[0] += (int)rt->m.h.ctx;
                if (rt->c_di[0] + rt->prompt + 4 > (int)rt->ndata)
                    rt->c_di[0] = 0;
                is_prompt = 1;
            }
            if (is_prompt)
                tok = (int)rt->data[rt->c_di[0] + rt->c_prompt_i[0]++];
            else
                tok = pending;
            f0 = cf_now();
            if (sync_feed(rt, G0, 0, tok, &o)) break;
            if (is_prompt) {
                rt->prefill_tok++;
            } else if (rt->nlat < rt->nlat_cap) {
                rt->lat[rt->nlat++] = (int)((cf_now() - f0) * 1e6);
                rt->decode_tok++;
            }
            pending = o;
        }
    } else {
        start_groups(rt, 0, rt->N);
        while (cf_now() - t0 < seconds)
            Sleep(20);
        stop_groups(rt);
    }
    el = cf_now() - t0;
    ntok = rt->nlat;
    if (ntok < 8) {
        printf("{\"mode\":\"stage\",\"error\":\"no tokens\",\"decode_tok\":%d}\n",
               rt->decode_tok);
        free(data);
        return 1;
    }
    for (i = 0; i < ntok; i++) mean += rt->lat[i];
    mean /= ntok;
    cp = (int *)cf_xmalloc(sizeof(int) * (size_t)ntok);
    memcpy(cp, rt->lat, sizeof(int) * (size_t)ntok);
    qsort(cp, (size_t)ntok, sizeof(int), cf_cmp_int);
    printf("{\"mode\":\"stage\",\"N\":%d,\"streams\":%d,\"tokens\":%d,"
           "\"sec\":%.3f,\"tok_s\":%.2f,\"mean_ms\":%.4f,\"p50_ms\":%.4f,"
           "\"p99_ms\":%.4f,\"max_ms\":%.4f,\"prefill_tok\":%d,"
           "\"barriers_per_token\":0,\"prefetch\":%d,\"cores\":[",
           rt->N, rt->K, ntok, el, ntok / el, mean / 1000.0,
           cp[ntok / 2] / 1000.0, cp[(int)(0.99 * ntok)] / 1000.0,
           cp[ntok - 1] / 1000.0, rt->prefill_tok, !rt->no_pf);
    for (i = 0; i < rt->N; i++)
        printf("%s%d", i ? "," : "", rt->grp[i].core);
    printf("],\"stage_costs\":[");
    for (i = 0; i < rt->N; i++)
        printf("%s%ld", i ? "," : "", rt->part[i].cost_mac);
    printf("],\"us_tok\":{\"gemv\":%.2f,\"act\":%.2f,\"attn\":%.2f,"
           "\"head\":%.2f,\"rope\":%.2f,\"scor\":%.2f,\"soft\":%.2f,"
           "\"val\":%.2f,\"sum\":%.2f,\"wall\":%.2f,\"tsc_ghz\":%.3f}}\n",
           (double)(cf_cyc_gemv - c0g) / (double)ntok / (double)tsch * 1e3,
           (double)(cf_cyc_act - c0a) / (double)ntok / (double)tsch * 1e3,
           (double)(cf_cyc_attn - c0t) / (double)ntok / (double)tsch * 1e3,
           (double)(cf_cyc_head - c0h) / (double)ntok / (double)tsch * 1e3,
           (double)(cf_cyc_rope - c0r) / (double)ntok / (double)tsch * 1e3,
           (double)(cf_cyc_scor - c0s) / (double)ntok / (double)tsch * 1e3,
           (double)(cf_cyc_soft - c0f) / (double)ntok / (double)tsch * 1e3,
           (double)(cf_cyc_val - c0v) / (double)ntok / (double)tsch * 1e3,
           (double)(cf_cyc_gemv - c0g + cf_cyc_act - c0a +
                    cf_cyc_attn - c0t + cf_cyc_head - c0h) /
               (double)ntok / (double)tsch * 1e3,
           mean, (double)tsch / 1e3);
    free(cp);
    free(data);
    return 0;
}

static void clflush_range(const uint8_t *p, size_t n) {
    size_t i;
    for (i = 0; i < n; i += 64) _mm_clflush((const char *)(p + i));
    _mm_sfence();
}

static void cold_prompt(rt_t *rt, const uint16_t *data, int *o) {
    int pi;
    cf_stream_reset(&rt->st[0], &rt->m);
    rt->c_pos[0] = 0;
    for (pi = 0; pi < rt->prompt; pi++) {
        rt->st[0].pos = rt->c_pos[0];
        cf_forward(&rt->m, &rt->st[0], &rt->grp[0].sc, (int)data[pi]);
        *o = cf_argmax(rt->st[0].logits, (int)rt->m.h.vocab);
        rt->c_pos[0]++;
    }
}

static void cold_arm(rt_t *rt, const uint16_t *data, int reps, int flush,
                     int *lat, unsigned long long tsch, double *batch_us,
                     double ctr[7]) {
    int i, dec = 0, o = 0;
    unsigned long long acc = 0, mark = 0;
    unsigned long long g0, a0, s0, f0, v0, h0, r0;
    cold_prompt(rt, data, &o);
    g0 = cf_cyc_gemv;
    a0 = cf_cyc_act;
    r0 = cf_cyc_rope;
    s0 = cf_cyc_scor;
    f0 = cf_cyc_soft;
    v0 = cf_cyc_val;
    h0 = cf_cyc_head;
    for (i = 0; i < reps; i++) {
        unsigned long long a, b;
        if (dec >= 900) {
            if (mark) {
                acc += __rdtsc() - mark;
                mark = 0;
            }
            cold_prompt(rt, data, &o);
            dec = 0;
        }
        if (flush) {
            if (mark) {
                acc += __rdtsc() - mark;
                mark = 0;
            }
            clflush_range(rt->m.base, rt->m.size);
        }
        if (!mark) mark = __rdtsc();
        a = __rdtsc();
        rt->st[0].pos = rt->c_pos[0];
        cf_forward(&rt->m, &rt->st[0], &rt->grp[0].sc, o);
        o = cf_argmax(rt->st[0].logits, (int)rt->m.h.vocab);
        rt->c_pos[0]++;
        b = __rdtsc();
        lat[i] = (int)((double)(b - a) / (double)tsch * 1e3);
        dec++;
    }
    if (mark) acc += __rdtsc() - mark;
    *batch_us = (double)acc / (double)tsch * 1e3 / (double)reps;
    ctr[0] = (double)(cf_cyc_gemv - g0) / (double)reps / (double)tsch * 1e3;
    ctr[1] = (double)(cf_cyc_act - a0) / (double)reps / (double)tsch * 1e3;
    ctr[2] = (double)(cf_cyc_scor - s0) / (double)reps / (double)tsch * 1e3;
    ctr[3] = (double)(cf_cyc_soft - f0) / (double)reps / (double)tsch * 1e3;
    ctr[4] = (double)(cf_cyc_val - v0) / (double)reps / (double)tsch * 1e3;
    ctr[5] = (double)(cf_cyc_head - h0) / (double)reps / (double)tsch * 1e3;
    ctr[6] = (double)(cf_cyc_rope - r0) / (double)reps / (double)tsch * 1e3;
}

static int cold_cmp(const void *a, const void *b) {
    int x = *(const int *)a, y = *(const int *)b;
    return (x > y) - (x < y);
}

static void cold_stats(int *v, int n, int *med, double *mean, int *p99) {
    int i, *c = (int *)cf_xmalloc(sizeof(int) * (size_t)n);
    double s = 0;
    memcpy(c, v, sizeof(int) * (size_t)n);
    qsort(c, (size_t)n, sizeof(int), cold_cmp);
    for (i = 0; i < n; i++) s += v[i];
    *med = c[n / 2];
    *mean = s / n;
    *p99 = c[(int)(0.99 * (n - 1))];
    free(c);
}

static int do_cold(rt_t *rt, const char *binpath, int cold_reps, int warm_reps) {
    size_t n = 0;
    uint16_t *data = load_bin(binpath, &n);
    int *wl, *cl, wmed, wp99, cmed, cp99;
    double wavg, cavg, flush_us = 0, wbatch, cbatch;
    double wctr[7], cctr[7];
    unsigned long long tsch, fa;
    if (!data) return 1;
    if (n < (size_t)rt->prompt + 4) {
        fprintf(stderr, "cold: bin too small\n");
        free(data);
        return 1;
    }
    if (cold_reps < 16) cold_reps = 16;
    if (warm_reps < cold_reps) warm_reps = cold_reps * 10;
    cf_pin(rt->grp[0].core);
    {
        unsigned long long a = __rdtsc();
        Sleep(120);
        tsch = (__rdtsc() - a) / 120;
    }
    wl = (int *)cf_xmalloc(sizeof(int) * (size_t)warm_reps);
    cl = (int *)cf_xmalloc(sizeof(int) * (size_t)cold_reps);
    cold_arm(rt, data, warm_reps, 0, wl, tsch, &wbatch, wctr);
    fa = __rdtsc();
    clflush_range(rt->m.base, rt->m.size);
    flush_us = (double)(__rdtsc() - fa) / (double)tsch * 1e3;
    cold_arm(rt, data, cold_reps, 1, cl, tsch, &cbatch, cctr);
    cold_stats(wl, warm_reps, &wmed, &wavg, &wp99);
    cold_stats(cl, cold_reps, &cmed, &cavg, &cp99);
    printf("{\"mode\":\"cold\",\"N\":1,\"prompt\":%d,\"warm_reps\":%d,"
           "\"cold_reps\":%d,\"flush_scope\":\"model_weights\","
           "\"reset_cadence\":900,"
           "\"warm_us\":{\"median\":%d,\"mean\":%.1f,\"p99\":%d,"
           "\"batch\":%.1f},"
           "\"cold_us\":{\"median\":%d,\"mean\":%.1f,\"p99\":%d,"
           "\"batch\":%.1f},"
           "\"median_delta_us\":%d,\"median_ratio\":%.3f,"
           "\"batch_ratio\":%.3f,"
           "\"warm_tok_s\":%.1f,\"cold_tok_s\":%.1f,"
           "\"warm_ctr_us\":{\"gemv\":%.1f,\"act\":%.1f,\"scor\":%.1f,"
           "\"soft\":%.1f,\"val\":%.1f,\"head\":%.1f,\"rope\":%.1f},"
           "\"cold_ctr_us\":{\"gemv\":%.1f,\"act\":%.1f,\"scor\":%.1f,"
           "\"soft\":%.1f,\"val\":%.1f,\"head\":%.1f,\"rope\":%.1f},"
           "\"flush_bytes\":%zu,\"flush_us\":%.1f,\"flush_lines\":%zu,"
           "\"tsc_ghz\":%.3f,\"cores\":[%d]}\n",
           rt->prompt, warm_reps, cold_reps,
           wmed, wavg, wp99, wbatch, cmed, cavg, cp99, cbatch,
           cmed - wmed, (double)cmed / (double)wmed, cbatch / wbatch,
           1e6 / wavg, 1e6 / cavg,
           wctr[0], wctr[1], wctr[2], wctr[3], wctr[4], wctr[5], wctr[6],
           cctr[0], cctr[1], cctr[2], cctr[3], cctr[4], cctr[5], cctr[6],
           rt->m.size, flush_us, rt->m.size / 64,
           (double)tsch / 1e3, rt->grp[0].core);
    free(wl);
    free(cl);
    free(data);
    return 0;
}

static int do_info(rt_t *rt) {
    int i, u;
    long bytes;
    printf("{\"model_bytes\":%zu,\"d\":%u,\"units\":%d,\"N\":%d,\"groups\":[",
           rt->m.size, rt->m.h.d, rt->units_n, rt->N);
    for (i = 0; i < rt->N; i++) {
        bytes = 0;
        for (u = rt->part[i].lo; u <= rt->part[i].hi; u++)
            bytes += rt->units[u].bytes;
        printf("%s{\"lo\":%d,\"hi\":%d,\"bytes\":%ld,\"cost_mac\":%ld,\"core\":%d}",
               i ? "," : "", rt->part[i].lo, rt->part[i].hi, bytes,
               rt->part[i].cost_mac, rt->cores[i]);
    }
    printf("]}\n");
    return 0;
}

static int gen_cmd(rt_t *rt, const char *bin, int ntok, int spec, int auto_m,
                   int kmax, float tau) {
    int r;
    if (auto_m) {
        double wp, ws, t0;
        int use_spec;
        const char *reason;
        rt->quiet = 1;
        t0 = cf_now();
        r = do_gen(rt, bin, 128);
        wp = cf_now() - t0;
        if (r) return r;
        t0 = cf_now();
        r = do_gen(rt, bin, 128);
        wp = (wp + (cf_now() - t0)) * 0.5;
        if (r) return r;
        t0 = cf_now();
        r = do_gen_spec(rt, bin, 128, kmax, tau);
        ws = cf_now() - t0;
        if (r) return r;
        t0 = cf_now();
        r = do_gen_spec(rt, bin, 128, kmax, tau);
        ws = (ws + (cf_now() - t0)) * 0.5;
        if (r) return r;
        if (!rt->sp_drafted ||
            (double)rt->sp_accepted / (double)rt->sp_drafted < 0.40) {
            use_spec = 0;
            reason = "d8-acceptance";
        } else if (ws < wp * 0.90) {
            use_spec = 1;
            reason = "probe-speed";
        } else {
            use_spec = 0;
            reason = "probe-speed";
        }
        printf("{\"mode\":\"auto\",\"plain_s\":%.4f,\"spec_s\":%.4f,"
               "\"acceptance\":%.4f,\"chosen\":\"%s\",\"reason\":\"%s\"}\n",
               wp, ws,
               rt->sp_drafted
                   ? (double)rt->sp_accepted / (double)rt->sp_drafted
                   : 0.0,
               use_spec ? "spec" : "plain", reason);
        rt->quiet = 0;
        if (use_spec) return do_gen_spec(rt, bin, ntok, kmax, tau);
        return do_gen(rt, bin, ntok);
    }
    if (spec) r = do_gen_spec(rt, bin, ntok, kmax, tau);
    else r = do_gen(rt, bin, ntok);
    return r;
}

static const char *flagv(int argc, char **argv, int *i, const char *name) {
    size_t n = strlen(name);
    const char *a = argv[*i];
    if (!strncmp(a, name, n) && a[n] == '=') return a + n + 1;
    if (!strcmp(a, name) && *i + 1 < argc) {
        (*i)++;
        return argv[*i];
    }
    return NULL;
}

static int query_topology(int *cores_out, int maxn, unsigned long long *l3b,
                          char *err, size_t errlen, int *nphys_out) {
    DWORD len = 0, i;
    SYSTEM_LOGICAL_PROCESSOR_INFORMATION *info;
    ULONG_PTR l3mask = 0;
    DWORD l3size = 0;
    int phys[16], sib[16], ncore = 0, n = 0, k;
    *l3b = 0;
    if (nphys_out) *nphys_out = 0;
    GetLogicalProcessorInformation(NULL, &len);
    if (!len) {
        snprintf(err, errlen, "topology query failed");
        return -1;
    }
    info = (SYSTEM_LOGICAL_PROCESSOR_INFORMATION *)malloc(len);
    if (!info || !GetLogicalProcessorInformation(info, &len)) {
        free(info);
        snprintf(err, errlen, "topology query failed");
        return -1;
    }
    for (i = 0; i < len / sizeof(*info); i++) {
        if (info[i].Relationship == RelationCache && info[i].Cache.Level == 3) {
            l3mask |= info[i].ProcessorMask;
            if (info[i].Cache.Size > l3size) l3size = info[i].Cache.Size;
        }
        if (info[i].Relationship == RelationProcessorCore && ncore < 16) {
            ULONG_PTR m = info[i].ProcessorMask;
            int b, first = -1;
            phys[ncore] = -1;
            sib[ncore] = -1;
            for (b = 0; b < 64; b++)
                if (m & ((ULONG_PTR)1 << b)) {
                    if (first < 0) {
                        first = b;
                        phys[ncore] = b;
                    } else if (sib[ncore] < 0) {
                        sib[ncore] = b;
                    }
                }
            if (phys[ncore] >= 0) ncore++;
        }
    }
    free(info);
    if (!l3mask || !ncore) {
        snprintf(err, errlen, "no L3 or core topology found");
        return -1;
    }
    for (k = 0; k < ncore; k++)
        if (!(l3mask & ((ULONG_PTR)1 << phys[k]))) {
            snprintf(err, errlen,
                     "cross-CCX: core %d (LP %d) outside L3 domain", k,
                     phys[k]);
            return -1;
        }
    for (k = 0; k < ncore && n < maxn; k++)
        cores_out[n++] = phys[k];
    for (k = 0; k < ncore && n < maxn; k++)
        if (sib[k] >= 0 && (l3mask & ((ULONG_PTR)1 << sib[k])))
            cores_out[n++] = sib[k];
    if (nphys_out) *nphys_out = ncore;
    *l3b = l3size;
    return n;
}

static int setup(rt_t *rt, const char *model, const char *irpath, int N,
                 int K, const char *cores_csv) {
    int g, i;
    memset(rt, 0, sizeof(*rt));
    if (ends_with(model, ".cf")) {
        if (cf_container_load(rt, model, N) != 0) return -1;
    } else {
        if (cf_model_load(&rt->m, model) != 0) return -1;
        if (load_ir(rt, irpath, N) != 0) return -1;
    }
    if (rt->m.h.d > MAX_D_MSG) {
        fprintf(stderr, "d > %d\n", MAX_D_MSG);
        return -1;
    }
    if (N < 1 || N > rt->units_n) {
        fprintf(stderr, "N must be 1..%d\n", rt->units_n);
        return -1;
    }
    rt->N = N;
    rt->K = K;
    rt->prompt = 65;
    rt->nlat_cap = 4000000;
    rt->lat = (int *)cf_xmalloc(sizeof(int) * (size_t)rt->nlat_cap);
    rt->st = (cf_stream_t *)cf_xmalloc(sizeof(cf_stream_t) * (size_t)K);
    for (i = 0; i < K; i++) {
        cf_stream_init(&rt->st[i], &rt->m);
        cf_stream_reset(&rt->st[i], &rt->m);
    }
    rt->c_prompt_i = (int *)cf_xmalloc(sizeof(int) * (size_t)K);
    rt->c_pos = (int *)cf_xmalloc(sizeof(int) * (size_t)K);
    rt->c_inflight = (int *)cf_xmalloc(sizeof(int) * (size_t)K);
    rt->c_pending = (int *)cf_xmalloc(sizeof(int) * (size_t)K);
    rt->c_di = (int *)cf_xmalloc(sizeof(int) * (size_t)K);
    rt->t_push = (double *)cf_xmalloc(sizeof(double) * (size_t)K);
    for (i = 0; i < K; i++)
        rt->c_di[i] = i * ((int)rt->m.h.ctx + rt->prompt + 8);
    {
        int nb = (int)rt->m.h.nblocks, b;
        rt->conv_ix = (int *)cf_xmalloc(sizeof(int) * (size_t)(nb ? nb : 1));
        rt->conv_blk = (int *)cf_xmalloc(sizeof(int) * (size_t)(nb ? nb : 1));
        rt->nconv_b = 0;
        for (b = 0; b < nb; b++) {
            rt->conv_ix[b] = -1;
            if (!rt->m.blk[b].is_attn) {
                rt->conv_ix[b] = rt->nconv_b;
                rt->conv_blk[rt->nconv_b++] = b;
            }
        }
        rt->cvs = (float *)cf_xmalloc(8 * (size_t)(rt->nconv_b ? rt->nconv_b : 1) *
                                      cv_bytes(&rt->m));
        rt->cvs_pos = (int *)cf_xmalloc(sizeof(int) * 8 *
                                        (size_t)(rt->nconv_b ? rt->nconv_b : 1));
        for (i = 0; i < 8 * (rt->nconv_b ? rt->nconv_b : 1); i++)
            rt->cvs_pos[i] = -1;
    }
    {
        char terr[128];
        int nc = query_topology(rt->cores, CORE_ORDER_MAX, &rt->l3_bytes,
                                terr, sizeof(terr), NULL);
        if (nc < 0) {
            fprintf(stderr, "topology: %s\n", terr);
            return -1;
        }
        rt->ncores = nc;
    }
    if (cores_csv && cores_csv[0]) {
        char buf[128];
        char *tok;
        snprintf(buf, sizeof(buf), "%s", cores_csv);
        rt->ncores = 0;
        tok = strtok(buf, ",");
        while (tok && rt->ncores < CORE_ORDER_MAX) {
            rt->cores[rt->ncores++] = atoi(tok);
            tok = strtok(NULL, ",");
        }
    }
    {
        long long home = 0;
        int u;
        for (u = 0; u < rt->units_n; u++) home += rt->units[u].bytes;
        rt->home_bytes = home;
        rt->no_pf = 1;
        (void)rt->l3_bytes;
    }
    if (N > rt->ncores) {
        fprintf(stderr, "N=%d > available cores %d\n", N, rt->ncores);
        return -1;
    }
    rt->grp = (group_t *)cf_xmalloc(sizeof(group_t) * (size_t)N);
    rt->rings = (ring_t *)cf_xmalloc(sizeof(ring_t) * (size_t)(N - 1));
    for (g = 0; g < N - 1; g++)
        if (ring_init(&rt->rings[g]) != 0) return -1;
    if (ring_init(&rt->rout) != 0) return -1;
    for (g = 0; g < N; g++) {
        group_t *G = &rt->grp[g];
        G->rt = rt;
        G->gid = g;
        G->lo = rt->part[g].lo;
        G->hi = rt->part[g].hi;
        G->core = rt->cores[g];
        G->embed = rt->units[G->lo].embed;
        G->head = rt->units[G->hi].head;
        G->in = g > 0 ? &rt->rings[g - 1] : NULL;
        G->out = g < N - 1 ? &rt->rings[g] : NULL;
        G->pf_r = 0;
        G->pf_i = 0;
        cf_scratch_init(&G->sc, &rt->m);
        do_touch(G);
    }
    return 0;
}

static void unpin_main(void) {
#ifdef _WIN32
    SetThreadAffinityMask(GetCurrentThread(), 0xFF);
#endif
}

int main(int argc, char **argv) {
    rt_t rt;
    const char *model, *irpath, *cmd;
    int N = 4, K = 1;
    const char *cores = NULL;
    int nopf, pff, spec = 0, auto_m = 0, kmax = 4;
    float tau = 0.5f;
    int i;
    if (argc >= 3 && !strcmp(argv[1], "run")) {
        const char *modelf = argv[2];
        const char *bin = "data/val.bin";
        int rN = -1, rntok = 900, rspec = 0, rauto = 0, rkmax = 4;
        float rtau = 0.5f;
        int cores_order[CORE_ORDER_MAX], nphys = 0, nc;
        unsigned long long l3 = 0;
        char terr[128];
        for (i = 3; i < argc; i++) {
            const char *a = argv[i], *v;
            if (!strcmp(a, "--spec")) { rspec = 1; continue; }
            if (!strcmp(a, "--auto")) { rauto = 1; continue; }
            v = flagv(argc, argv, &i, "--cores");
            if (v) { rN = strcmp(v, "auto") ? atoi(v) : -1; continue; }
            v = flagv(argc, argv, &i, "--ntok");
            if (v) { rntok = atoi(v); continue; }
            v = flagv(argc, argv, &i, "--bin");
            if (v) { bin = v; continue; }
            v = flagv(argc, argv, &i, "--kmax");
            if (v) { rkmax = atoi(v); continue; }
            v = flagv(argc, argv, &i, "--tau");
            if (v) { rtau = (float)atof(v); continue; }
            fprintf(stderr, "run: unknown flag %s\n", a);
            return 1;
        }
        if (!ends_with(modelf, ".cf")) {
            fprintf(stderr, "run expects model.cf (create with cf_pack.py)\n");
            return 1;
        }
        if (rN < 0) {
            nc = query_topology(cores_order, CORE_ORDER_MAX, &l3, terr,
                                sizeof(terr), &nphys);
            if (nc < 0) {
                fprintf(stderr, "topology: %s\n", terr);
                return 1;
            }
            rN = nphys >= 4 ? 4 : (nphys < 1 ? 1 : nphys);
            fprintf(stderr, "cores=auto: nphys=%d -> N=%d\n", nphys, rN);
        }
        if (rkmax < 1) rkmax = 1;
        if (rkmax > 8) rkmax = 8;
        if (!(rtau > 0.f) || rtau > 1.f) rtau = 0.5f;
        if (rN == 1 && (rspec || rauto)) {
            fprintf(stderr, "spec/auto requires N>=2, running plain\n");
            rspec = 0;
            rauto = 0;
        }
        if (setup(&rt, modelf, NULL, rN, 1, NULL) != 0) return 1;
        return gen_cmd(&rt, bin, rntok, rspec, rauto, rkmax, rtau);
    }
    if (argc < 4) {
        fprintf(stderr,
                "usage: %s run <model.cf> [--cores auto|N] [--ntok N] [--bin path]"
                " [--spec] [--auto] [--kmax=N] [--tau=X]\n"
                "   or: %s <model.i8|model.cf> <stages.json|-> info|gen|bench|cold"
                "|probe|draftprobe [N] [bin] [ntok] [spec] [auto] [kmax=N] [tau=X]"
                " [cores=a,b,c]\n",
                argv[0], argv[0]);
        return 1;
    }
    model = argv[1];
    irpath = argv[2];
    cmd = argv[3];
    if (argc > 4) N = atoi(argv[4]);
    if (!strcmp(cmd, "bench") && argc > 7 && argv[7][0] >= '0' &&
        argv[7][0] <= '9')
        K = atoi(argv[7]);
    for (i = 5; i < argc; i++)
        if (!strncmp(argv[i], "cores=", 6)) cores = argv[i] + 6;
    nopf = 0;
    pff = 0;
    for (i = 4; i < argc; i++) {
        if (!strcmp(argv[i], "nopf")) nopf = 1;
        if (!strcmp(argv[i], "pf")) pff = 1;
        if (!strcmp(argv[i], "spec")) spec = 1;
        if (!strcmp(argv[i], "auto")) auto_m = 1;
        if (!strncmp(argv[i], "kmax=", 5)) kmax = atoi(argv[i] + 5);
        if (!strncmp(argv[i], "tau=", 4)) tau = (float)atof(argv[i] + 4);
    }
    if (kmax < 1) kmax = 1;
    if (kmax > 8) kmax = 8;
    if (!(tau > 0.f) || tau > 1.f) tau = 0.5f;
    if (N == 1 && (spec || auto_m)) {
        fprintf(stderr, "spec/auto requires N>=2, running plain\n");
        spec = 0;
        auto_m = 0;
    }
    if (setup(&rt, model, irpath, N, K, cores) != 0) return 1;
    if (pff) rt.no_pf = 0;
    if (nopf) rt.no_pf = 1;
    if (!strcmp(cmd, "info"))
        return do_info(&rt);
    if (!strcmp(cmd, "gen")) {
        const char *bin = argc > 5 ? argv[5] : "data/val.bin";
        int ntok = argc > 6 ? atoi(argv[6]) : 32;
        return gen_cmd(&rt, bin, ntok, spec, auto_m, kmax, tau);
    }
    if (!strcmp(cmd, "suite")) {
        const char *pf = argc > 5 ? argv[5] : "suite_prompts.bin";
        const char *of = argc > 6 ? argv[6] : "suite_out.bin";
        unpin_main();
        return do_suite(&rt, pf, of);
    }
    if (!strcmp(cmd, "ask")) {
        const char *pf = argc > 5 ? argv[5] : "ask_prompt.bin";
        int ntok = argc > 6 ? atoi(argv[6]) : 200;
        unpin_main();
        return do_ask(&rt, pf, ntok);
    }
    if (!strcmp(cmd, "draftprobe")) {
        const char *bin = argc > 5 ? argv[5] : "data/val.bin";
        int steps = argc > 6 ? atoi(argv[6]) : 32;
        return do_draftprobe(&rt, bin, steps);
    }
    if (!strcmp(cmd, "bench")) {
        const char *bin = argc > 5 ? argv[5] : "data/val.bin";
        double sec = argc > 6 ? atof(argv[6]) : 30.0;
        unpin_main();
        return do_bench(&rt, bin, sec);
    }
    if (!strcmp(cmd, "cold")) {
        const char *bin = argc > 5 ? argv[5] : "data/val.bin";
        int creps = argc > 6 ? atoi(argv[6]) : 1000;
        int wreps = argc > 7 ? atoi(argv[7]) : 5000;
        unpin_main();
        return do_cold(&rt, bin, creps, wreps);
    }
    if (!strcmp(cmd, "probe")) {
        const char *bin = argc > 5 ? argv[5] : "data/val.bin";
        int cycles = argc > 6 ? atoi(argv[6]) : 6;
        int ntok = argc > 7 ? atoi(argv[7]) : 16;
        size_t evmb = argc > 8 ? (size_t)atoi(argv[8]) : 32;
        unpin_main();
        return do_probe(&rt, bin, cycles, ntok, evmb);
    }
    fprintf(stderr, "unknown cmd %s\n", cmd);
    return 1;
}
