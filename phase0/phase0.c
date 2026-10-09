#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "cf_model.h"
#ifdef _WIN32
#include <windows.h>
#else
#include <time.h>
#include <pthread.h>
#endif

typedef struct {
    volatile int stop;
    size_t bytes;
    int mode;
} evict_t;

#ifdef _WIN32
static DWORD WINAPI evict_fn(LPVOID p) {
#else
static void *evict_fn(void *p) {
#endif
    evict_t *e = (evict_t *)p;
    size_t n = e->bytes, i;
    volatile char *buf = (volatile char *)cf_xmalloc(n);
    volatile long sink = 0;
    cf_pin(1);
    while (!e->stop) {
        if (e->mode == 2) {
            for (i = 0; i < n; i += 64) sink += buf[i];
        } else {
            for (i = 0; i < n; i += 64) buf[i] = (char)(buf[i] + 1);
        }
    }
    (void)sink;
    free((void *)buf);
#ifdef _WIN32
    return 0;
#else
    return NULL;
#endif
}

static void start_evict(evict_t *e, size_t bytes, int mode, void **th) {
    e->stop = 0;
    e->bytes = bytes;
    e->mode = mode;
#ifdef _WIN32
    *th = CreateThread(NULL, 0, evict_fn, e, 0, NULL);
#else
    *th = (void *)1;
    {
        pthread_t t;
        pthread_create(&t, NULL, evict_fn, e);
        (void)t;
    }
#endif
}

static void stop_evict(evict_t *e, void *th) {
    e->stop = 1;
#ifdef _WIN32
    if (th) {
        WaitForSingleObject((HANDLE)th, 10000);
        CloseHandle((HANDLE)th);
    }
#else
    (void)th;
#endif
}

static int do_info(cf_model_t *M) {
    size_t ws = cf_working_set(M);
    double l3 = 4096.0 * 1024.0;
    printf("{\"file_bytes\":%zu,\"d\":%u,\"dff\":%u,\"nconv\":%u,\"natt\":%u,"
           "\"heads\":%u,\"kv\":%u,\"ctx\":%u,\"vocab\":%u,\"softcap\":%.1f,"
           "\"working_set\":%zu,\"l3\":4096,\"ws_pct_l3\":%.1f,\"fit_half_l3\":%s}\n",
           M->size, M->h.d, M->h.dff, M->h.nconv, M->h.natt, M->h.heads,
           M->h.kv_heads, M->h.ctx, M->h.vocab, M->h.softcap, ws,
           100.0 * ws / l3, ws * 2 <= (size_t)l3 ? "true" : "false");
    return 0;
}

static int do_stream(double seconds) {
    size_t n = 128ull * 1024 * 1024, i, pass = 0;
    char *buf = (char *)cf_xmalloc(n);
    double t0, el;
    volatile long sink = 0;
    for (i = 0; i < n; i++) buf[i] = (char)i;
    cf_pin(0);
    t0 = cf_now();
    while (cf_now() - t0 < seconds) {
        long s = 0;
        for (i = 0; i < n; i += 64) s += buf[i];
        sink += s;
        pass++;
    }
    el = cf_now() - t0;
    printf("{\"mode\":\"stream\",\"sec\":%.3f,\"passes\":%zu,\"gb_s\":%.2f}\n",
           el, pass, (double)pass * n / el / 1e9);
    free(buf);
    (void)sink;
    return 0;
}

static int do_bench(cf_model_t *M, cf_stream_t *S, cf_scratch_t *T,
                    const uint16_t *data, size_t ndata,
                    double seconds, int prompt, int do_evict, size_t evict_mb) {
    int vocab = (int)M->h.vocab, ctx = (int)M->h.ctx;
    int *lat = (int *)cf_xmalloc(sizeof(int) * 4000000);
    int nlat = 0, cycles = 0, cap = 3999000;
    int prefill_tok = 0;
    double t0, el, prefill_s = 0, mean = 0;
    size_t di = 0;
    evict_t ev;
    void *th = NULL;
    int i;
    int *cp;
    cf_pin(0);
    if (do_evict) start_evict(&ev, evict_mb * 1024 * 1024, do_evict, &th);

    t0 = cf_now();
    while (cf_now() - t0 < seconds && nlat < cap) {
        int pos = 0, p, pending, t;
        double p0, s0;
        cf_stream_reset(S, M);
        p0 = cf_now();
        for (p = 0; p < prompt && p < (int)ndata - 2; p++) {
            cf_forward(M, S, T, (int)data[di + p]);
            pos++;
        }
        prefill_s += cf_now() - p0;
        prefill_tok += pos;
        pending = cf_argmax(S->logits, vocab);
        while (pos < ctx && nlat < cap && cf_now() - t0 < seconds) {
            s0 = cf_now();
            cf_forward(M, S, T, pending);
            pos++;
            t = cf_argmax(S->logits, vocab);
            lat[nlat++] = (int)((cf_now() - s0) * 1e6);
            pending = t;
        }
        cycles++;
        di += (size_t)ctx;
        if (di + (size_t)ctx + 4 > ndata) di = 0;
    }
    if (do_evict) stop_evict(&ev, th);
    el = cf_now() - t0;
    if (nlat < 8) {
        printf("{\"mode\":\"clean\",\"error\":\"no tokens\"}\n");
        free(lat);
        return 1;
    }
    for (i = 0; i < nlat; i++) mean += lat[i];
    mean /= nlat;
    cp = (int *)cf_xmalloc(sizeof(int) * nlat);
    memcpy(cp, lat, sizeof(int) * nlat);
    qsort(cp, nlat, sizeof(int), cf_cmp_int);
    printf("{\"mode\":\"%s\",\"tokens\":%d,\"sec\":%.3f,\"tok_s\":%.2f,"
           "\"mean_ms\":%.4f,\"p50_ms\":%.4f,\"p99_ms\":%.4f,\"max_ms\":%.4f,"
           "\"cycles\":%d,\"prefill_tok_s\":%.0f}\n",
           do_evict == 2 ? "evict_read" : (do_evict ? "evict_write" : "clean"), nlat, el, nlat / el,
           mean / 1000.0, cp[nlat / 2] / 1000.0, cp[(int)(0.99 * nlat)] / 1000.0,
           cp[nlat - 1] / 1000.0, cycles,
           prefill_s > 0 ? prefill_tok / prefill_s : 0.0);
    free(cp);
    free(lat);
    return 0;
}

static int do_ppl(cf_model_t *M, cf_stream_t *S, cf_scratch_t *T,
                  const char *bin, int windows, int seq) {
    FILE *f = fopen(bin, "rb");
    long sz;
    size_t ntok, w;
    uint16_t *data;
    double tot = 0;
    long cnt = 0;
    int wn = 0, t;
    if (!f) { fprintf(stderr, "cannot open %s\n", bin); return 1; }
    fseek(f, 0, SEEK_END);
    sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    ntok = (size_t)sz / 2;
    data = (uint16_t *)cf_xmalloc(ntok * 2);
    if (fread(data, 2, ntok, f) != ntok) { fclose(f); return 1; }
    fclose(f);
    cf_pin(0);
    for (w = 0; w < (size_t)windows; w++) {
        size_t base = w * (size_t)seq;
        if (base + (size_t)seq + 1 > ntok) break;
        cf_stream_reset(S, M);
        for (t = 0; t < seq; t++) {
            cf_forward(M, S, T, (int)data[base + t]);
            if (t + 1 < seq) {
                int i;
                float mx = S->logits[0];
                double s = 0;
                for (i = 1; i < (int)M->h.vocab; i++)
                    if (S->logits[i] > mx) mx = S->logits[i];
                for (i = 0; i < (int)M->h.vocab; i++)
                    s += exp((double)S->logits[i] - mx);
                tot += (mx + log(s)) - (double)S->logits[data[base + t + 1]];
                cnt++;
            }
        }
        wn++;
    }
    printf("{\"windows\":%d,\"seq\":%d,\"ntok\":%ld,\"nll\":%.6f,\"ppl\":%.6f}\n",
           wn, seq, cnt, tot / cnt, exp(tot / cnt));
    free(data);
    return 0;
}

static int prefill_at(cf_model_t *M, cf_stream_t *S, cf_scratch_t *T,
                      const uint16_t *data, size_t di, size_t ndata,
                      int prompt, int *pending) {
    int p;
    cf_stream_reset(S, M);
    for (p = 0; p < prompt && di + p + 1 < ndata; p++)
        cf_forward(M, S, T, (int)data[di + p]);
    *pending = (int)data[di + p];
    return S->pos;
}

static double decode_n(cf_model_t *M, cf_stream_t *S, cf_scratch_t *T,
                       int *pending, int ntok, double *first) {
    int i, ctx = (int)M->h.ctx;
    double t0 = cf_now(), f = -1;
    for (i = 0; i < ntok && S->pos < ctx; i++) {
        int nxt;
        double s0 = cf_now();
        cf_forward(M, S, T, *pending);
        nxt = cf_argmax(S->logits, (int)M->h.vocab);
        *pending = nxt;
        if (i == 0) f = (cf_now() - s0) * 1000.0;
    }
    if (first) *first = f;
    return cf_now() - t0;
}

static int do_probe(cf_model_t *M, cf_stream_t *S, cf_scratch_t *T,
                    const uint16_t *data, size_t ndata,
                    int cycles, int ntok, int prompt, size_t evmb) {
    size_t evn = evmb * 1024 * 1024, i;
    char *evbuf = (char *)cf_xmalloc(evn);
    double warm_tot = 0, cold_tot = 0, warm_f = 0, cold_f = 0;
    size_t di = 0;
    int c, nw = 0, nc = 0;
    for (i = 0; i < evn; i++) evbuf[i] = (char)i;
    cf_pin(0);
    for (c = 0; c < cycles; c++) {
        int pending;
        double t, f;
        prefill_at(M, S, T, data, di, ndata, prompt, &pending);
        t = decode_n(M, S, T, &pending, ntok, &f);
        warm_tot += t;
        warm_f += f;
        nw++;
        prefill_at(M, S, T, data, di, ndata, prompt, &pending);
        for (i = 0; i < evn; i += 64) evbuf[i] = (char)(evbuf[i] + 1);
        t = decode_n(M, S, T, &pending, ntok, &f);
        cold_tot += t;
        cold_f += f;
        nc++;
        di += 64;
        if (di + (size_t)prompt + 8 > ndata) di = 0;
    }
    printf("{\"mode\":\"probe\",\"cycles\":%d,\"tok_per_cycle\":%d,"
           "\"evict_mb\":%zu,\"warm_tok_s\":%.2f,\"cold_tok_s\":%.2f,"
           "\"ratio_cold_over_warm\":%.3f,"
           "\"warm_first_ms\":%.4f,\"cold_first_ms\":%.4f,"
           "\"first_ratio\":%.3f}\n",
           cycles, ntok, evmb, nw * ntok / warm_tot, nc * ntok / cold_tot,
           (cold_tot / nc) / (warm_tot / nw),
           warm_f / nw, cold_f / nc, (cold_f / nc) / (warm_f / nw));
    free(evbuf);
    return 0;
}

static uint16_t *load_bin(const char *bin, size_t *out_n);

static int do_gen(cf_model_t *M, cf_stream_t *S, cf_scratch_t *T,
                  const char *bin, int ntok) {
    size_t ntok_total = 0;
    uint16_t *data = load_bin(bin, &ntok_total);
    int *seq;
    int p, i, pending = 0, count = 0, prompt = 65;
    if (!data) return 1;
    seq = (int *)cf_xmalloc(sizeof(int) * (size_t)(ntok + prompt + 4));
    cf_stream_reset(S, M);
    for (p = 0; p < prompt && p + 1 < (int)ntok_total; p++) {
        cf_forward(M, S, T, (int)data[p]);
        pending = cf_argmax(S->logits, (int)M->h.vocab);
    }
    for (i = 0; i < ntok && S->pos < (int)M->h.ctx; i++) {
        seq[count++] = pending;
        cf_forward(M, S, T, pending);
        pending = cf_argmax(S->logits, (int)M->h.vocab);
    }
    printf("{\"mode\":\"gen\",\"N\":1,\"prompt\":%d,\"tokens\":[", prompt);
    for (i = 0; i < count; i++)
        printf("%s%d", i ? "," : "", seq[i]);
    printf("]}\n");
    free(seq);
    free(data);
    return 0;
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

int main(int argc, char **argv) {
    cf_model_t M;
    cf_stream_t S;
    cf_scratch_t T;
    const char *cmd;
    memset(&M, 0, sizeof(M));
    memset(&S, 0, sizeof(S));
    memset(&T, 0, sizeof(T));
    if (argc < 3) {
        fprintf(stderr, "usage: %s <model.i8> info|stream|ppl|bench [args]\n", argv[0]);
        return 1;
    }
    if (cf_model_load(&M, argv[1]) != 0) return 1;
    cf_stream_init(&S, &M);
    cf_scratch_init(&T, &M);
    cf_stream_reset(&S, &M);
    cmd = argv[2];
    if (!strcmp(cmd, "info")) return do_info(&M);
    if (!strcmp(cmd, "stream")) return do_stream(argc > 3 ? atof(argv[3]) : 2.0);
    if (!strcmp(cmd, "ppl"))
        return do_ppl(&M, &S, &T, argc > 3 ? argv[3] : "data/val.bin",
                      argc > 4 ? atoi(argv[4]) : 80, argc > 5 ? atoi(argv[5]) : 1024);
    if (!strcmp(cmd, "gen")) {
        const char *bin = argc > 3 ? argv[3] : "data/val.bin";
        int ntok = argc > 4 ? atoi(argv[4]) : 32;
        return do_gen(&M, &S, &T, bin, ntok);
    }
    if (!strcmp(cmd, "bench")) {
        const char *bin = argc > 3 ? argv[3] : "data/val.bin";
        double s = argc > 4 ? atof(argv[4]) : 60.0;
        int prompt = argc > 5 ? atoi(argv[5]) : 65;
        int ev = argc > 6 ? atoi(argv[6]) : 0;
        size_t evmb = argc > 7 ? (size_t)atoi(argv[7]) : 64;
        size_t ntok = 0;
        uint16_t *data = load_bin(bin, &ntok);
        int r;
        if (!data) return 1;
        r = do_bench(&M, &S, &T, data, ntok, s, prompt, ev, evmb);
        free(data);
        return r;
    }
    if (!strcmp(cmd, "probe")) {
        const char *bin = argc > 3 ? argv[3] : "data/val.bin";
        int cycles = argc > 4 ? atoi(argv[4]) : 8;
        int ntok = argc > 5 ? atoi(argv[5]) : 16;
        int prompt = argc > 6 ? atoi(argv[6]) : 65;
        size_t evmb = argc > 7 ? (size_t)atoi(argv[7]) : 32;
        size_t n = 0;
        uint16_t *data = load_bin(bin, &n);
        int r;
        if (!data) return 1;
        r = do_probe(&M, &S, &T, data, n, cycles, ntok, prompt, evmb);
        free(data);
        return r;
    }
    fprintf(stderr, "unknown cmd %s\n", cmd);
    return 1;
}
