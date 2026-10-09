#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>
#include <windows.h>

static inline float dot_i8f(const int8_t *w, const float *x, int n) {
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
        int32_t t[32];
        int k;
        _mm256_storeu_si256((__m256i *)(t + 0), a0);
        _mm256_storeu_si256((__m256i *)(t + 8), a1);
        _mm256_storeu_si256((__m256i *)(t + 16), a2);
        _mm256_storeu_si256((__m256i *)(t + 24), a3);
        for (k = 0; k < 32; k++) acc += t[k];
    }
#endif
    for (; i < n; i++) acc += (int32_t)w[i] * (int32_t)x[i];
    return acc;
}

static double now_ms(void) {
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1000.0 / (double)f.QuadPart;
}

int main(int argc, char **argv) {
    enum { RMAX = 8192, CMAX = 512 };
    static int8_t w[RMAX * CMAX];
    static int16_t xq[CMAX];
    static float xf[CMAX];
    int c = argc > 1 ? atoi(argv[1]) : 396;
    int rows = argc > 2 ? atoi(argv[2]) : 4096;
    int reps = argc > 3 ? atoi(argv[3]) : 64;
    int r, i, rep;
    volatile int32_t sink = 0;
    volatile float sinkf = 0.f;
    double t0, t1;
    long long macs;

    srand(7);
    for (r = 0; r < RMAX; r++)
        for (i = 0; i < CMAX; i++)
            w[r * CMAX + i] = (int8_t)(rand() % 255 - 127);
    for (i = 0; i < CMAX; i++) {
        xq[i] = (int16_t)(rand() % 255 - 127);
        xf[i] = (float)xq[i];
    }

    t0 = now_ms();
    for (rep = 0; rep < reps; rep++)
        for (r = 0; r < rows; r++)
            sink += idot_i8(w + (size_t)r * CMAX, xq, c);
    t1 = now_ms();
    macs = (long long)reps * rows * c;
    printf("int  : %8.3f ms  %6.2f ns/dot  %6.3f ps/MAC  %7.1f MMAC/s  %7.1f Mdot/s\n",
           t1 - t0, (t1 - t0) * 1e6 / (reps * rows),
           (t1 - t0) * 1e12 / macs, macs / (t1 - t0) / 1e6,
           (double)reps * rows / (t1 - t0));
    (void)sink;

    t0 = now_ms();
    for (rep = 0; rep < reps; rep++)
        for (r = 0; r < rows; r++)
            sinkf += dot_i8f(w + (size_t)r * CMAX, xf, c);
    t1 = now_ms();
    printf("float: %8.3f ms  %6.2f ns/dot  %6.3f ps/MAC  %7.1f MMAC/s  %7.1f Mdot/s\n",
           t1 - t0, (t1 - t0) * 1e6 / (reps * rows),
           (t1 - t0) * 1e12 / macs, macs / (t1 - t0) / 1e6,
           (double)reps * rows / (t1 - t0));
    (void)sinkf;
    return 0;
}
