#ifndef CF_MODEL_H
#define CF_MODEL_H

#include <stddef.h>
#include <stdint.h>

#define CF_MAGIC 0x30465043u
#define CF_K_I8_ROWS 0
#define CF_TAG_WTE 0
#define CF_TAG_N1 1
#define CF_TAG_N2 2
#define CF_TAG_DW 3
#define CF_TAG_GATE 4
#define CF_TAG_MIX 5
#define CF_TAG_F1 6
#define CF_TAG_F2 7
#define CF_TAG_F3 8
#define CF_TAG_QKV 9
#define CF_TAG_OUT 10
#define CF_TAG_NF 11
#define CF_TAG_DN1 12
#define CF_TAG_DM1 13
#define CF_TAG_DM2 14
#define CF_TAG_DM3 15
#define CF_TAG_DN2 16
#define CF_EPS 1e-6f

#pragma pack(push, 1)
typedef struct {
    uint32_t magic, version, vocab, d, dff, nconv, natt, heads, kv_heads, ctx,
             nblocks, nsections, k;
    float softcap;
    uint32_t sections_off, table_size;
} cf_hdr_t;
typedef struct {
    uint32_t kind, rows, cols, tag;
    uint64_t data_off, scale_off;
} cf_sec_t;
#pragma pack(pop)

typedef struct {
    const int8_t *q;
    const float *s;
    uint32_t rows, cols;
} cf_mat_t;
typedef struct {
    const float *p;
    uint32_t rows;
} cf_fmat_t;

typedef struct {
    int is_attn;
    cf_fmat_t n1, n2;
    cf_mat_t dw, gate, mix, qkv, out, f1, f2, f3;
} cf_blkw_t;

typedef struct {
    cf_hdr_t h;
    uint8_t *base;
    size_t size;
    cf_mat_t wte;
    cf_fmat_t nf;
    cf_mat_t dm1, dm2, dm3;
    cf_fmat_t dn1, dn2;
    int dhead_depth;
    cf_blkw_t *blk;
    float *rope_inv;
    int rope_half;
} cf_model_t;

typedef struct {
    float *cbuf;
    int8_t *kq, *vq;
    float *ks, *vs;
} cf_bstate_t;

typedef struct {
    cf_bstate_t *bs;
    int pos;
    float *logits;
} cf_stream_t;

typedef struct {
    float *x, *hn, *g, *cc, *hv, *qkvbuf, *qh, *o, *scores;
    float *fa, *fb, *fout;
} cf_scratch_t;

double cf_now(void);
void cf_pin(int core);
void *cf_xmalloc(size_t n);
int cf_cmp_int(const void *a, const void *b);

int cf_model_load(cf_model_t *m, const char *path);
int cf_model_load_mem(cf_model_t *m, const void *buf, size_t n);
void cf_stream_init(cf_stream_t *s, const cf_model_t *m);
void cf_stream_reset(cf_stream_t *s, const cf_model_t *m);
void cf_scratch_init(cf_scratch_t *t, const cf_model_t *m);

void cf_embed(const cf_model_t *m, int tok, float *x);
void cf_blocks(const cf_model_t *m, cf_stream_t *s, cf_scratch_t *t,
               int pos, int lo, int hi, float *x);
void cf_head(const cf_model_t *m, cf_stream_t *s, cf_scratch_t *t,
             const float *x);
void cf_forward(const cf_model_t *m, cf_stream_t *s, cf_scratch_t *t, int tok);
int cf_argmax(const float *v, int n);
int cf_draft(const cf_model_t *m, cf_stream_t *s, cf_scratch_t *t, int tok);
float cf_top1_prob(const float *v, int n);

size_t cf_working_set(const cf_model_t *m);
size_t cf_stream_ws(const cf_model_t *m);

#endif
