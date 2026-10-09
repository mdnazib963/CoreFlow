#define _CRT_SECURE_NO_WARNINGS
#include "cf_model.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <time.h>
#include <pthread.h>
#endif

double cf_now(void) {
#ifdef _WIN32
    static LARGE_INTEGER f;
    LARGE_INTEGER c;
    if (!f.QuadPart) QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)f.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
#endif
}

void cf_pin(int core) {
#ifdef _WIN32
    SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)1 << core);
#else
    cpu_set_t s;
    CPU_ZERO(&s);
    CPU_SET(core, &s);
    pthread_setaffinity_np(pthread_self(), sizeof(s), &s);
#endif
}

void *cf_xmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) { fprintf(stderr, "oom %zu\n", n); exit(2); }
    memset(p, 0, n);
    return p;
}

int cf_cmp_int(const void *a, const void *b) {
    int x = *(const int *)a, y = *(const int *)b;
    return (x > y) - (x < y);
}

unsigned long long cf_cyc_gemv = 0, cf_cyc_act = 0, cf_cyc_attn = 0,
                   cf_cyc_head = 0, cf_cyc_rope = 0, cf_cyc_scor = 0,
                   cf_cyc_soft = 0, cf_cyc_val = 0;

static inline unsigned long long cf_cyc(void) {
#ifdef _WIN32
    return (unsigned long long)__rdtsc();
#else
    return 0;
#endif
}

static inline float dot_i8(const int8_t *w, const float *x, int n) {
    float acc = 0.f;
    int i = 0;
#if defined(__AVX2__)
    __m256 va = _mm256_setzero_ps();
    for (; i + 8 <= n; i += 8) {
        __m128i qw = _mm_loadl_epi64((const __m128i *)(w + i));
        __m256i qi = _mm256_cvtepi8_epi32(qw);
        __m256 wf = _mm256_cvtepi32_ps(qi);
        va = _mm256_fmadd_ps(wf, _mm256_loadu_ps(x + i), va);
    }
    {
        float tmp[8];
        int t;
        _mm256_storeu_ps(tmp, va);
        for (t = 0; t < 8; t++) acc += tmp[t];
    }
#endif
    for (; i < n; i++) acc += (float)w[i] * x[i];
    return acc;
}

#define CF_XQ_MAX 512

static inline int32_t idot_i8(const int8_t *w, const int16_t *x, int n) {
    int i = 0;
    int32_t acc = 0;
#if defined(__AVX2__)
    __m256i a0 = _mm256_setzero_si256();
    __m256i a1 = _mm256_setzero_si256();
    __m256i a2 = _mm256_setzero_si256();
    __m256i a3 = _mm256_setzero_si256();
    for (; i + 64 <= n; i += 64) {
        a0 = _mm256_add_epi32(a0, _mm256_madd_epi16(
                 _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(w + i))),
                 _mm256_loadu_si256((const __m256i *)(x + i))));
        a1 = _mm256_add_epi32(a1, _mm256_madd_epi16(
                 _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(w + i + 16))),
                 _mm256_loadu_si256((const __m256i *)(x + i + 16))));
        a2 = _mm256_add_epi32(a2, _mm256_madd_epi16(
                 _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(w + i + 32))),
                 _mm256_loadu_si256((const __m256i *)(x + i + 32))));
        a3 = _mm256_add_epi32(a3, _mm256_madd_epi16(
                 _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(w + i + 48))),
                 _mm256_loadu_si256((const __m256i *)(x + i + 48))));
    }
    for (; i + 16 <= n; i += 16)
        a0 = _mm256_add_epi32(a0, _mm256_madd_epi16(
                 _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(w + i))),
                 _mm256_loadu_si256((const __m256i *)(x + i))));
    {
        __m256i s = _mm256_add_epi32(_mm256_add_epi32(a0, a1),
                                     _mm256_add_epi32(a2, a3));
        __m128i h = _mm_add_epi32(_mm256_castsi256_si128(s),
                                  _mm256_extracti128_si256(s, 1));
        h = _mm_add_epi32(h, _mm_srli_si128(h, 8));
        h = _mm_add_epi32(h, _mm_srli_si128(h, 4));
        acc += _mm_cvtsi128_si32(h);
    }
#endif
    for (; i < n; i++) acc += (int32_t)w[i] * (int32_t)x[i];
    return acc;
}

static void gemv_q(const cf_mat_t *m, const float *x, float *y) {
    uint32_t r;
    int c = (int)m->cols, i;
    if (c > CF_XQ_MAX) {
        for (r = 0; r < m->rows; r++)
            y[r] = m->s[r] * dot_i8(m->q + (size_t)r * c, x, c);
        return;
    }
    {
        float mx = 0.f, sc;
        int16_t xq[CF_XQ_MAX];
#if defined(__AVX2__)
        {
            const __m256 absmsk =
                _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
            __m256 vm = _mm256_setzero_ps();
            float tmp[8];
            int k;
            for (i = 0; i + 8 <= c; i += 8)
                vm = _mm256_max_ps(vm, _mm256_and_ps(
                         _mm256_loadu_ps(x + i), absmsk));
            _mm256_storeu_ps(tmp, vm);
            for (k = 0; k < 8; k++)
                if (tmp[k] > mx) mx = tmp[k];
            for (; i < c; i++) {
                float a = fabsf(x[i]);
                if (a > mx) mx = a;
            }
        }
#else
        for (i = 0; i < c; i++) {
            float a = fabsf(x[i]);
            if (a > mx) mx = a;
        }
#endif
        if (mx <= 0.f) {
            for (r = 0; r < m->rows; r++) y[r] = 0.f;
            return;
        }
        sc = mx / 127.f;
        {
            float inv = 127.f / mx;
#if defined(__AVX2__)
            __m256 vinv = _mm256_set1_ps(inv);
            for (i = 0; i + 16 <= c; i += 16) {
                __m256i d0 = _mm256_cvtps_epi32(
                    _mm256_mul_ps(_mm256_loadu_ps(x + i), vinv));
                __m256i d1 = _mm256_cvtps_epi32(
                    _mm256_mul_ps(_mm256_loadu_ps(x + i + 8), vinv));
                __m256i p = _mm256_packs_epi32(d0, d1);
                p = _mm256_permute4x64_epi64(p, _MM_SHUFFLE(3, 1, 2, 0));
                _mm256_storeu_si256((__m256i *)(xq + i), p);
            }
            for (; i < c; i++)
                xq[i] = (int16_t)lrintf(x[i] * inv);
#else
            for (i = 0; i < c; i++)
                xq[i] = (int16_t)lrintf(x[i] * inv);
#endif
        }
        for (r = 0; r < m->rows; r++) {
            const int8_t *w = m->q + (size_t)r * c;
            _mm_prefetch((const char *)(w + c), _MM_HINT_T0);
            y[r] = m->s[r] * sc * (float)idot_i8(w, xq, c);
        }
    }
}

static void gemv(const cf_mat_t *m, const float *x, float *y) {
    unsigned long long t0 = cf_cyc();
    gemv_q(m, x, y);
    cf_cyc_gemv += cf_cyc() - t0;
}

static void rms(const float *x, const float *w, float *o, int d) {
    float m = 0.f;
    int i;
    for (i = 0; i < d; i++) m += x[i] * x[i];
    m = 1.f / sqrtf(m / (float)d + CF_EPS);
    for (i = 0; i < d; i++) o[i] = x[i] * m * w[i];
}

#if defined(__AVX2__)
static inline __m256 expf256(__m256 x) {
    const __m256 ln2 = _mm256_set1_ps(0.6931471805599453f);
    const __m256 invln2 = _mm256_set1_ps(1.4426950408889634f);
    __m256 t, r, r2;
    __m256i k;
    x = _mm256_max_ps(x, _mm256_set1_ps(-87.3f));
    x = _mm256_min_ps(x, _mm256_set1_ps(88.f));
    t = _mm256_mul_ps(x, invln2);
    k = _mm256_cvtps_epi32(t);
    t = _mm256_cvtepi32_ps(k);
    r = _mm256_sub_ps(x, _mm256_mul_ps(t, ln2));
    r2 = _mm256_mul_ps(r, r);
    {
        __m256 c7 = _mm256_set1_ps(1.0f / 5040.f);
        __m256 c6 = _mm256_set1_ps(1.0f / 720.f);
        __m256 c5 = _mm256_set1_ps(1.0f / 120.f);
        __m256 c4 = _mm256_set1_ps(1.0f / 24.f);
        __m256 c3 = _mm256_set1_ps(1.0f / 6.f);
        __m256 c2 = _mm256_set1_ps(0.5f);
        __m256 p = c7;
        p = _mm256_add_ps(c6, _mm256_mul_ps(p, r));
        p = _mm256_add_ps(c5, _mm256_mul_ps(p, r));
        p = _mm256_add_ps(c4, _mm256_mul_ps(p, r));
        p = _mm256_add_ps(c3, _mm256_mul_ps(p, r));
        p = _mm256_add_ps(c2, _mm256_mul_ps(p, r));
        p = _mm256_add_ps(_mm256_set1_ps(1.f), _mm256_mul_ps(p, r));
        p = _mm256_add_ps(_mm256_set1_ps(1.f), _mm256_mul_ps(p, r));
        r = p;
    }
    r2 = _mm256_castsi256_ps(
        _mm256_slli_epi32(_mm256_add_epi32(k, _mm256_set1_epi32(127)), 23));
    return _mm256_mul_ps(r, r2);
}
#else
#define expf256(x) (expf(x))
#endif

static void mlp(const cf_model_t *m, const cf_blkw_t *b, cf_scratch_t *t,
                float *x) {
    int d = (int)m->h.d, dff = (int)m->h.dff, i;
    gemv(&b->f1, t->hn, t->fa);
    gemv(&b->f2, t->hn, t->fb);
    {
        unsigned long long t0 = cf_cyc();
        i = 0;
#if defined(__AVX2__)
        for (i = 0; i + 8 <= dff; i += 8) {
            __m256 s = _mm256_loadu_ps(t->fa + i);
            __m256 e = expf256(_mm256_sub_ps(_mm256_setzero_ps(), s));
            __m256 f = _mm256_div_ps(s, _mm256_add_ps(_mm256_set1_ps(1.f), e));
            _mm256_storeu_ps(t->fa + i,
                             _mm256_mul_ps(f, _mm256_loadu_ps(t->fb + i)));
        }
#endif
        for (; i < dff; i++) {
            float s = t->fa[i];
            t->fa[i] = s / (1.f + expf(-s)) * t->fb[i];
        }
        cf_cyc_act += cf_cyc() - t0;
    }
    gemv(&b->f3, t->fa, t->fout);
    for (i = 0; i < d; i++) x[i] += t->fout[i];
}

static void rope_apply(float *v, int nchunks, int hd, const float *cs,
                       const float *sn) {
    int half = hd >> 1, c, i;
    for (c = 0; c < nchunks; c++) {
        float *u = v + (size_t)c * hd;
        for (i = 0; i < half; i++) {
            float a = u[i], b = u[i + half];
            u[i] = a * cs[i] - b * sn[i];
            u[i + half] = b * cs[i] + a * sn[i];
        }
    }
}

static void conv_step(const cf_model_t *m, const cf_blkw_t *b,
                      cf_bstate_t *st, cf_scratch_t *t, float *c_out) {
    int d = (int)m->h.d, k = (int)m->h.k, n, i;
    for (n = 0; n < d; n++) {
        float *buf = st->cbuf + (size_t)n * (k - 1);
        float acc = 0.f;
        const int8_t *w = b->dw.q + (size_t)n * k;
        float sc = b->dw.s[n];
        for (i = 0; i < k - 1; i++) t->hv[i] = buf[i];
        t->hv[k - 1] = t->hn[n];
        for (i = 0; i < k; i++) acc += (float)w[i] * t->hv[i];
        c_out[n] = acc * sc;
        for (i = 0; i < k - 2; i++) buf[i] = buf[i + 1];
        buf[k - 2] = t->hn[n];
    }
}

static void store_kv(const cf_model_t *m, const cf_blkw_t *b, cf_bstate_t *st,
                     cf_scratch_t *t, int pos) {
    int kvh = (int)m->h.kv_heads, hd = (int)m->h.d / (int)m->h.heads, kk, p;
    float *k = t->qkvbuf + m->h.d;
    float *v = k + (size_t)kvh * hd;
    for (kk = 0; kk < kvh; kk++) {
        float *kr = k + (size_t)kk * hd;
        float *vr = v + (size_t)kk * hd;
        float mx = 0.f, s;
        for (p = 0; p < hd; p++) {
            float a = fabsf(kr[p]);
            if (a > mx) mx = a;
        }
        s = mx > 0.f ? mx / 127.f : 1.f;
        st->ks[(size_t)pos * kvh + kk] = s;
        for (p = 0; p < hd; p++)
            st->kq[((size_t)pos * kvh + kk) * hd + p] = (int8_t)lrintf(kr[p] / s);
        mx = 0.f;
        for (p = 0; p < hd; p++) {
            float a = fabsf(vr[p]);
            if (a > mx) mx = a;
        }
        s = mx > 0.f ? mx / 127.f : 1.f;
        st->vs[(size_t)pos * kvh + kk] = s;
        for (p = 0; p < hd; p++)
            st->vq[((size_t)pos * kvh + kk) * hd + p] = (int8_t)lrintf(vr[p] / s);
    }
}

void cf_embed(const cf_model_t *m, int tok, float *x) {
    int d = (int)m->h.d, i;
    const int8_t *er = m->wte.q + (size_t)tok * d;
    float es = m->wte.s[tok];
    for (i = 0; i < d; i++) x[i] = es * (float)er[i];
}

void cf_blocks(const cf_model_t *m, cf_stream_t *s, cf_scratch_t *t,
               int pos, int lo, int hi, float *x) {
    int d = (int)m->h.d, i, j;
    for (i = lo; i < hi; i++) {
        const cf_blkw_t *b = &m->blk[i];
        cf_bstate_t *bs = &s->bs[i];
        if (!b->is_attn) {
            rms(x, b->n1.p, t->hn, d);
            conv_step(m, b, bs, t, t->cc);
            gemv(&b->gate, t->hn, t->g);
            {
                unsigned long long t0 = cf_cyc();
                j = 0;
#if defined(__AVX2__)
                for (j = 0; j + 8 <= d; j += 8) {
                    __m256 s = _mm256_loadu_ps(t->g + j);
                    __m256 e = expf256(_mm256_sub_ps(_mm256_setzero_ps(), s));
                    __m256 f =
                        _mm256_div_ps(s, _mm256_add_ps(_mm256_set1_ps(1.f), e));
                    _mm256_storeu_ps(t->g + j,
                                     _mm256_mul_ps(f, _mm256_loadu_ps(t->cc + j)));
                }
#endif
                for (; j < d; j++) {
                    float sc = t->g[j];
                    t->g[j] = (sc / (1.f + expf(-sc))) * t->cc[j];
                }
                cf_cyc_act += cf_cyc() - t0;
            }
            gemv(&b->mix, t->g, t->cc);
            for (j = 0; j < d; j++) x[j] += t->cc[j];
            rms(x, b->n2.p, t->hn, d);
            mlp(m, b, t, x);
        } else {
            int hd = d / (int)m->h.heads, heads = (int)m->h.heads;
            int kvh = (int)m->h.kv_heads, g = heads / kvh, hs, p, ii;
            float cs[128], sn[128];
            rms(x, b->n1.p, t->hn, d);
            gemv(&b->qkv, t->hn, t->qkvbuf);
            {
                unsigned long long t0 = cf_cyc(), t1;
                for (ii = 0; ii < m->rope_half; ii++) {
                    float a = (float)pos * m->rope_inv[ii];
                    cs[ii] = cosf(a);
                    sn[ii] = sinf(a);
                }
                rope_apply(t->qkvbuf, heads, hd, cs, sn);
                memcpy(t->qh, t->qkvbuf, (size_t)d * sizeof(float));
                rope_apply(t->qkvbuf + m->h.d, kvh, hd, cs, sn);
                store_kv(m, b, bs, t, pos);
                cf_cyc_rope += cf_cyc() - t0;
                for (hs = 0; hs < heads; hs++) {
                    float *qh = t->qh + (size_t)hs * hd;
                    int kk = hs / g;
                    float mx = -1e30f, inv = 1.f / sqrtf((float)hd);
                    int16_t qhq[CF_XQ_MAX];
                    float qmx = 0.f, qsc;
                    t1 = cf_cyc();
                    for (p = 0; p < hd; p++) {
                        float a = fabsf(qh[p]);
                        if (a > qmx) qmx = a;
                    }
                    qsc = qmx > 0.f ? 127.f / qmx : 0.f;
                    for (p = 0; p < hd; p++)
                        qhq[p] = (int16_t)lrintf(qh[p] * qsc);
                    qsc = qmx > 0.f ? qmx / 127.f : 0.f;
                    for (j = 0; j <= pos; j++) {
                        const int8_t *kr =
                            bs->kq + ((size_t)j * kvh + kk) * hd;
                        float sc = bs->ks[(size_t)j * kvh + kk];
                        int32_t acc = idot_i8(kr, qhq, hd);
                        t->scores[j] =
                            (qsc * sc * inv) * (float)acc;
                        if (t->scores[j] > mx) mx = t->scores[j];
                    }
                    cf_cyc_scor += cf_cyc() - t1;
                    t1 = cf_cyc();
                    {
                        float sum = 0.f, invs;
                        int jj;
#if defined(__AVX2__)
                        for (j = 0; j + 8 <= pos + 1; j += 8) {
                            __m256 v = _mm256_loadu_ps(t->scores + j);
                            _mm256_storeu_ps(
                                t->scores + j,
                                expf256(_mm256_sub_ps(v, _mm256_set1_ps(mx))));
                        }
                        for (jj = j; jj <= pos; jj++)
                            t->scores[jj] =
                                expf(t->scores[jj] - mx);
#else
                        j = 0;
                        for (jj = 0; jj <= pos; jj++)
                            t->scores[jj] =
                                expf(t->scores[jj] - mx);
#endif
                        for (j = 0; j <= pos; j++) sum += t->scores[j];
                        invs = 1.f / sum;
                        cf_cyc_soft += cf_cyc() - t1;
                        t1 = cf_cyc();
                        for (p = 0; p < hd; p++)
                            t->o[(size_t)hs * hd + p] = 0.f;
                        for (j = 0; j <= pos; j++) {
                            const int8_t *vr =
                                bs->vq + ((size_t)j * kvh + kk) * hd;
                            float sc = bs->vs[(size_t)j * kvh + kk];
                            float cj = t->scores[j] * invs * sc;
                            __m256 vc = _mm256_set1_ps(cj);
                            int q = 0;
#if defined(__AVX2__)
                            for (; q + 16 <= hd; q += 16) {
                                __m256i i16v = _mm256_cvtepi8_epi16(
                                    _mm_loadu_si128((const __m128i *)(vr + q)));
                                __m256 fl = _mm256_cvtepi32_ps(
                                    _mm256_cvtepi16_epi32(
                                        _mm256_castsi256_si128(i16v)));
                                __m256 fh = _mm256_cvtepi32_ps(
                                    _mm256_cvtepi16_epi32(
                                        _mm256_extracti128_si256(i16v, 1)));
                                __m256 ol = _mm256_loadu_ps(
                                    t->o + (size_t)hs * hd + q);
                                __m256 oh = _mm256_loadu_ps(
                                    t->o + (size_t)hs * hd + q + 8);
                                _mm256_storeu_ps(
                                    t->o + (size_t)hs * hd + q,
                                    _mm256_fmadd_ps(vc, fl, ol));
                                _mm256_storeu_ps(
                                    t->o + (size_t)hs * hd + q + 8,
                                    _mm256_fmadd_ps(vc, fh, oh));
                            }
#endif
                            for (; q < hd; q++)
                                t->o[(size_t)hs * hd + q] +=
                                    cj * (float)vr[q];
                        }
                        cf_cyc_val += cf_cyc() - t1;
                    }
                }
                cf_cyc_attn += cf_cyc() - t0;
            }
            gemv(&b->out, t->o, t->g);
            for (j = 0; j < d; j++) x[j] += t->g[j];
            rms(x, b->n2.p, t->hn, d);
            mlp(m, b, t, x);
        }
    }
}

void cf_head(const cf_model_t *m, cf_stream_t *s, cf_scratch_t *t,
             const float *x) {
    int d = (int)m->h.d, i;
    rms(x, m->nf.p, t->hn, d);
    gemv(&m->wte, t->hn, s->logits);
    if (m->h.softcap > 0.f) {
        float sc = m->h.softcap;
        unsigned long long t0 = cf_cyc();
        i = 0;
#if defined(__AVX2__)
        for (; i + 8 <= (int)m->h.vocab; i += 8) {
            __m256 z = _mm256_div_ps(_mm256_loadu_ps(s->logits + i),
                                     _mm256_set1_ps(sc));
            __m256 e = expf256(_mm256_add_ps(z, z));
            __m256 th = _mm256_div_ps(
                _mm256_sub_ps(e, _mm256_set1_ps(1.f)),
                _mm256_add_ps(e, _mm256_set1_ps(1.f)));
            _mm256_storeu_ps(s->logits + i, _mm256_mul_ps(_mm256_set1_ps(sc), th));
        }
#endif
        for (; i < (int)m->h.vocab; i++)
            s->logits[i] = sc * tanhf(s->logits[i] / sc);
        cf_cyc_head += cf_cyc() - t0;
    }
}

void cf_forward(const cf_model_t *m, cf_stream_t *s, cf_scratch_t *t, int tok) {
    cf_embed(m, tok, t->x);
    cf_blocks(m, s, t, s->pos, 0, (int)m->h.nblocks, t->x);
    cf_head(m, s, t, t->x);
    s->pos++;
}

int cf_argmax(const float *v, int n) {
    int i, b = 0;
    float mx = v[0];
    for (i = 1; i < n; i++)
        if (v[i] > mx) { mx = v[i]; b = i; }
    return b;
}

int cf_draft(const cf_model_t *m, cf_stream_t *s, cf_scratch_t *t, int tok) {
    int d = (int)m->h.d, ff = (int)m->dm1.rows, i;
    if (m->dhead_depth <= 0) return -1;
    cf_embed(m, tok, t->x);
    cf_blocks(m, s, t, s->pos, 0, m->dhead_depth, t->x);
    rms(t->x, m->dn1.p, t->hn, d);
    gemv(&m->dm1, t->hn, t->fa);
    gemv(&m->dm2, t->hn, t->fb);
    for (i = 0; i < ff; i++) {
        float a = t->fa[i];
        t->fa[i] = a / (1.f + expf(-a)) * t->fb[i];
    }
    gemv(&m->dm3, t->fa, t->fout);
    for (i = 0; i < d; i++) t->fout[i] += t->x[i];
    rms(t->fout, m->dn2.p, t->hn, d);
    gemv(&m->wte, t->hn, s->logits);
    if (m->h.softcap > 0.f) {
        float sc = m->h.softcap;
        for (i = 0; i < (int)m->h.vocab; i++)
            s->logits[i] = sc * tanhf(s->logits[i] / sc);
    }
    return cf_argmax(s->logits, (int)m->h.vocab);
}

float cf_top1_prob(const float *v, int n) {
    int i;
    float mx = v[0], sum = 0.f;
    for (i = 1; i < n; i++)
        if (v[i] > mx) mx = v[i];
    for (i = 0; i < n; i++) sum += expf(v[i] - mx);
    return 1.f / sum;
}

static cf_mat_t grab_i8(const uint8_t *base, cf_sec_t s) {
    cf_mat_t m;
    m.q = (const int8_t *)(base + s.data_off);
    m.s = (const float *)(base + s.scale_off);
    m.rows = s.rows;
    m.cols = s.cols;
    return m;
}

static int expect(const cf_sec_t *tab, int *si, int nsec, uint32_t tag) {
    if (*si >= nsec) {
        fprintf(stderr, "section exhausted expecting tag %u\n", tag);
        return -1;
    }
    if (tab[*si].tag != tag) {
        fprintf(stderr, "section %d tag %u expected %u\n", *si, tab[*si].tag, tag);
        return -1;
    }
    (*si)++;
    return 0;
}

static int cf_model_parse(cf_model_t *m) {
    cf_hdr_t *h;
    cf_sec_t *tab;
    int si = 0, bi, nsec;
    int d, kvh, heads, hd;
    memcpy(&m->h, m->base, sizeof(cf_hdr_t));
    h = &m->h;
    if (h->magic != CF_MAGIC) { fprintf(stderr, "bad magic\n"); return -1; }
    nsec = (int)h->nsections;
    d = (int)h->d;
    kvh = (int)h->kv_heads;
    heads = (int)h->heads;
    hd = d / heads;
    tab = (cf_sec_t *)(m->base + h->sections_off);

    m->blk = (cf_blkw_t *)cf_xmalloc(sizeof(cf_blkw_t) * h->nblocks);
    m->rope_half = hd >> 1;
    m->rope_inv = (float *)cf_xmalloc((size_t)m->rope_half * sizeof(float));
    {
        int i;
        for (i = 0; i < m->rope_half; i++)
            m->rope_inv[i] = powf(10000.f, -(2.f * (float)i) / (float)hd);
    }

    if (tab[si].tag != CF_TAG_WTE) { fprintf(stderr, "first section not wte\n"); return -1; }
    m->wte = grab_i8(m->base, tab[si]);
    si++;

    for (bi = 0; bi < (int)h->nblocks; bi++) {
        cf_blkw_t *b = &m->blk[bi];
        memset(b, 0, sizeof(*b));
        b->n1.p = (const float *)(m->base + tab[si].data_off);
        b->n1.rows = tab[si].rows;
        if (expect(tab, &si, nsec, CF_TAG_N1)) return -1;
        b->n2.p = (const float *)(m->base + tab[si].data_off);
        b->n2.rows = tab[si].rows;
        if (expect(tab, &si, nsec, CF_TAG_N2)) return -1;
        if (si >= nsec) return -1;
        if (tab[si].tag == CF_TAG_DW) {
            b->is_attn = 0;
            b->dw = grab_i8(m->base, tab[si]);
            if (expect(tab, &si, nsec, CF_TAG_DW)) return -1;
            b->gate = grab_i8(m->base, tab[si]);
            if (expect(tab, &si, nsec, CF_TAG_GATE)) return -1;
            b->mix = grab_i8(m->base, tab[si]);
            if (expect(tab, &si, nsec, CF_TAG_MIX)) return -1;
        } else if (tab[si].tag == CF_TAG_QKV) {
            b->is_attn = 1;
            b->qkv = grab_i8(m->base, tab[si]);
            if (expect(tab, &si, nsec, CF_TAG_QKV)) return -1;
            b->out = grab_i8(m->base, tab[si]);
            if (expect(tab, &si, nsec, CF_TAG_OUT)) return -1;
        } else {
            fprintf(stderr, "unknown block at %d tag %u\n", si, tab[si].tag);
            return -1;
        }
        b->f1 = grab_i8(m->base, tab[si]);
        if (expect(tab, &si, nsec, CF_TAG_F1)) return -1;
        b->f2 = grab_i8(m->base, tab[si]);
        if (expect(tab, &si, nsec, CF_TAG_F2)) return -1;
        b->f3 = grab_i8(m->base, tab[si]);
        if (expect(tab, &si, nsec, CF_TAG_F3)) return -1;
    }
    m->nf.p = (const float *)(m->base + tab[si].data_off);
    m->nf.rows = tab[si].rows;
    if (expect(tab, &si, nsec, CF_TAG_NF)) return -1;
    m->dhead_depth = 0;
    if (si < nsec) {
        m->dhead_depth = (int)tab[si].cols;
        m->dn1.p = (const float *)(m->base + tab[si].data_off);
        m->dn1.rows = tab[si].rows;
        if (expect(tab, &si, nsec, CF_TAG_DN1)) return -1;
        m->dm1 = grab_i8(m->base, tab[si]);
        if (expect(tab, &si, nsec, CF_TAG_DM1)) return -1;
        m->dm2 = grab_i8(m->base, tab[si]);
        if (expect(tab, &si, nsec, CF_TAG_DM2)) return -1;
        m->dm3 = grab_i8(m->base, tab[si]);
        if (expect(tab, &si, nsec, CF_TAG_DM3)) return -1;
        m->dn2.p = (const float *)(m->base + tab[si].data_off);
        m->dn2.rows = tab[si].rows;
        if (expect(tab, &si, nsec, CF_TAG_DN2)) return -1;
    }
    if (si != nsec) {
        fprintf(stderr, "section walk %d != %d\n", si, nsec);
        return -1;
    }
    (void)kvh;
    (void)hd;
    return 0;
}

int cf_model_load(cf_model_t *m, const char *path) {
    FILE *f = fopen(path, "rb");
    long sz;
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return -1; }
    fseek(f, 0, SEEK_END);
    sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    m->size = (size_t)sz;
    m->base = (uint8_t *)cf_xmalloc(m->size);
    if (fread(m->base, 1, m->size, f) != m->size) { fclose(f); return -1; }
    fclose(f);
    return cf_model_parse(m);
}

int cf_model_load_mem(cf_model_t *m, const void *buf, size_t n) {
    if (!buf || !n) return -1;
    m->size = n;
    m->base = (uint8_t *)cf_xmalloc(n);
    memcpy(m->base, buf, n);
    return cf_model_parse(m);
}

void cf_stream_init(cf_stream_t *s, const cf_model_t *m) {
    int d = (int)m->h.d, kvh = (int)m->h.kv_heads, hd = d / (int)m->h.heads;
    int i;
    s->bs = (cf_bstate_t *)cf_xmalloc(sizeof(cf_bstate_t) * m->h.nblocks);
    for (i = 0; i < (int)m->h.nblocks; i++) {
        cf_bstate_t *bs = &s->bs[i];
        if (m->blk[i].is_attn) {
            bs->kq = (int8_t *)cf_xmalloc((size_t)m->h.ctx * kvh * hd);
            bs->vq = (int8_t *)cf_xmalloc((size_t)m->h.ctx * kvh * hd);
            bs->ks = (float *)cf_xmalloc((size_t)m->h.ctx * kvh * sizeof(float));
            bs->vs = (float *)cf_xmalloc((size_t)m->h.ctx * kvh * sizeof(float));
        } else {
            bs->cbuf = (float *)cf_xmalloc((size_t)d * (m->h.k - 1) * sizeof(float));
        }
    }
    s->logits = (float *)cf_xmalloc((size_t)m->h.vocab * sizeof(float));
    s->pos = 0;
}

void cf_stream_reset(cf_stream_t *s, const cf_model_t *m) {
    int d = (int)m->h.d, kvh = (int)m->h.kv_heads, hd = d / (int)m->h.heads, i;
    for (i = 0; i < (int)m->h.nblocks; i++) {
        cf_bstate_t *bs = &s->bs[i];
        if (bs->cbuf)
            memset(bs->cbuf, 0, (size_t)d * (m->h.k - 1) * sizeof(float));
        if (bs->kq) {
            memset(bs->kq, 0, (size_t)m->h.ctx * kvh * hd);
            memset(bs->vq, 0, (size_t)m->h.ctx * kvh * hd);
            memset(bs->ks, 0, (size_t)m->h.ctx * kvh * sizeof(float));
            memset(bs->vs, 0, (size_t)m->h.ctx * kvh * sizeof(float));
        }
    }
    s->pos = 0;
}

void cf_scratch_init(cf_scratch_t *t, const cf_model_t *m) {
    int d = (int)m->h.d, kvh = (int)m->h.kv_heads;
    int hd = d / (int)m->h.heads;
    t->x = (float *)cf_xmalloc((size_t)d * sizeof(float));
    t->hn = (float *)cf_xmalloc((size_t)d * sizeof(float));
    t->g = (float *)cf_xmalloc((size_t)d * sizeof(float));
    t->cc = (float *)cf_xmalloc((size_t)d * sizeof(float));
    t->hv = (float *)cf_xmalloc((size_t)m->h.k * sizeof(float));
    t->qkvbuf = (float *)cf_xmalloc((size_t)(d + 2 * kvh * hd) * sizeof(float));
    t->qh = (float *)cf_xmalloc((size_t)d * sizeof(float));
    t->o = (float *)cf_xmalloc((size_t)d * sizeof(float));
    t->scores = (float *)cf_xmalloc((size_t)m->h.ctx * sizeof(float));
    t->fa = (float *)cf_xmalloc((size_t)m->h.dff * sizeof(float));
    t->fb = (float *)cf_xmalloc((size_t)m->h.dff * sizeof(float));
    t->fout = (float *)cf_xmalloc((size_t)d * sizeof(float));
}

size_t cf_working_set(const cf_model_t *m) {
    size_t w = m->size, i;
    int d = (int)m->h.d;
    for (i = 0; i < m->h.nblocks; i++) {
        if (m->blk[i].is_attn) {
            w += (size_t)m->h.ctx * m->h.kv_heads * (d / (int)m->h.heads) * 2;
            w += (size_t)m->h.ctx * m->h.kv_heads * 2 * sizeof(float);
        } else {
            w += (size_t)d * (m->h.k - 1) * sizeof(float);
        }
    }
    w += cf_stream_ws(m);
    return w;
}

size_t cf_stream_ws(const cf_model_t *m) {
    size_t w = 0;
    int d = (int)m->h.d;
    w += (size_t)d * 6 * sizeof(float);
    w += (size_t)(d + 2 * m->h.kv_heads * (d / (int)m->h.heads)) * sizeof(float);
    w += (size_t)(m->h.dff * 2 + d) * sizeof(float);
    w += (size_t)m->h.ctx * sizeof(float);
    w += (size_t)m->h.vocab * sizeof(float);
    w += (size_t)m->rope_half * sizeof(float);
    return w;
}
