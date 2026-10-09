#define _CRT_SECURE_NO_WARNINGS
#include "cf_json.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct {
    const char *p, *end;
    char *err;
    size_t errlen;
} pctx_t;

static js_val *parse_value(pctx_t *c);

static js_val *new_val(js_type t) {
    js_val *v = (js_val *)calloc(1, sizeof(js_val));
    if (!v) { fprintf(stderr, "oom\n"); exit(2); }
    v->t = t;
    return v;
}

static void skip_ws(pctx_t *c) {
    while (c->p < c->end && (*c->p == ' ' || *c->p == '\t' || *c->p == '\n' ||
                             *c->p == '\r'))
        c->p++;
}

static int match(pctx_t *c, char ch) {
    skip_ws(c);
    if (c->p < c->end && *c->p == ch) { c->p++; return 1; }
    return 0;
}

static void utf8_enc(char **o, unsigned cp) {
    if (cp < 0x80) { *(*o)++ = (char)cp; }
    else if (cp < 0x800) {
        *(*o)++ = (char)(0xC0 | (cp >> 6));
        *(*o)++ = (char)(0x80 | (cp & 0x3F));
    } else {
        *(*o)++ = (char)(0xE0 | (cp >> 12));
        *(*o)++ = (char)(0x80 | ((cp >> 6) & 0x3F));
        *(*o)++ = (char)(0x80 | (cp & 0x3F));
    }
}

static char *parse_string_raw(pctx_t *c) {
    const char *s;
    char *out, *o;
    size_t n = 0;
    if (!match(c, '"')) return NULL;
    s = c->p;
    while (c->p < c->end && *c->p != '"') {
        if (*c->p == '\\') {
            c->p++;
            if (c->p >= c->end) return NULL;
            switch (*c->p) {
            case 'n': case 't': case 'r': case 'b': case 'f': case '"':
            case '\\': case '/': n++; break;
            case 'u': n += 3; c->p += 4; break;
            default: return NULL;
            }
        } else n++;
        c->p++;
    }
    if (c->p >= c->end) return NULL;
    c->p++;
    out = (char *)malloc(n * 3 + 1);
    if (!out) { fprintf(stderr, "oom\n"); exit(2); }
    o = out;
    {
        const char *q = s;
        while (q < c->p - 1) {
            if (*q == '\\') {
                q++;
                switch (*q) {
                case 'n': *o++ = '\n'; break;
                case 't': *o++ = '\t'; break;
                case 'r': *o++ = '\r'; break;
                case 'b': *o++ = '\b'; break;
                case 'f': *o++ = '\f'; break;
                case 'u': {
                    unsigned cp = 0;
                    int i;
                    for (i = 1; i <= 4; i++) {
                        char h = q[i];
                        cp <<= 4;
                        if (h >= '0' && h <= '9') cp |= (unsigned)(h - '0');
                        else if (h >= 'a' && h <= 'f') cp |= (unsigned)(h - 'a' + 10);
                        else if (h >= 'A' && h <= 'F') cp |= (unsigned)(h - 'A' + 10);
                    }
                    utf8_enc(&o, cp);
                    q += 4;
                    break;
                }
                default: *o++ = *q; break;
                }
                q++;
            } else {
                *o++ = *q++;
            }
        }
    }
    *o = 0;
    return out;
}

static js_val *parse_number(pctx_t *c) {
    char *endp;
    double d;
    skip_ws(c);
    d = strtod(c->p, &endp);
    if (endp == c->p) return NULL;
    c->p = endp;
    {
        js_val *v = new_val(JS_NUM);
        v->num = d;
        return v;
    }
}

static js_val *parse_array(pctx_t *c) {
    js_val *v;
    if (!match(c, '[')) return NULL;
    v = new_val(JS_ARR);
    skip_ws(c);
    if (match(c, ']')) return v;
    for (;;) {
        js_val *e = parse_value(c);
        if (!e) { js_free(v); return NULL; }
        v->items = (js_val **)realloc(v->items, sizeof(js_val *) * (v->n + 1));
        v->items[v->n++] = e;
        skip_ws(c);
        if (match(c, ',')) continue;
        if (match(c, ']')) return v;
        js_free(v);
        return NULL;
    }
}

static js_val *parse_object(pctx_t *c) {
    js_val *v;
    if (!match(c, '{')) return NULL;
    v = new_val(JS_OBJ);
    skip_ws(c);
    if (match(c, '}')) return v;
    for (;;) {
        char *k;
        js_val *e;
        skip_ws(c);
        k = parse_string_raw(c);
        if (!k) { js_free(v); return NULL; }
        if (!match(c, ':')) { free(k); js_free(v); return NULL; }
        e = parse_value(c);
        if (!e) { free(k); js_free(v); return NULL; }
        v->items = (js_val **)realloc(v->items, sizeof(js_val *) * (v->n + 1));
        v->keys = (char **)realloc(v->keys, sizeof(char *) * (v->n + 1));
        v->keys[v->n] = k;
        v->items[v->n] = e;
        v->n++;
        skip_ws(c);
        if (match(c, ',')) continue;
        if (match(c, '}')) return v;
        js_free(v);
        return NULL;
    }
}

static js_val *parse_value(pctx_t *c) {
    skip_ws(c);
    if (c->p >= c->end) return NULL;
    if (*c->p == '{') return parse_object(c);
    if (*c->p == '[') return parse_array(c);
    if (*c->p == '"') {
        js_val *v = new_val(JS_STR);
        v->str = parse_string_raw(c);
        if (!v->str) { js_free(v); return NULL; }
        return v;
    }
    if (!strncmp(c->p, "true", 4) && c->p + 4 <= c->end) {
        js_val *v = new_val(JS_BOOL);
        v->boolean = 1;
        c->p += 4;
        return v;
    }
    if (!strncmp(c->p, "false", 5) && c->p + 5 <= c->end) {
        js_val *v = new_val(JS_BOOL);
        c->p += 5;
        return v;
    }
    if (!strncmp(c->p, "null", 4) && c->p + 4 <= c->end) {
        js_val *v = new_val(JS_NULL);
        c->p += 4;
        return v;
    }
    return parse_number(c);
}

js_val *js_parse(const char *src, char *err, size_t errlen) {
    pctx_t c;
    js_val *v;
    c.p = src;
    c.end = src + strlen(src);
    c.err = err;
    c.errlen = errlen;
    v = parse_value(&c);
    if (err && errlen && !v) snprintf(err, errlen, "json parse error near offset %td",
                                      c.p - src);
    return v;
}

void js_free(js_val *v) {
    int i;
    if (!v) return;
    free(v->str);
    for (i = 0; i < v->n; i++) {
        if (v->keys) free(v->keys[i]);
        js_free(v->items[i]);
    }
    free(v->keys);
    free(v->items);
    free(v);
}

js_val *js_obj_get(const js_val *o, const char *key) {
    int i;
    if (!o || o->t != JS_OBJ) return NULL;
    for (i = 0; i < o->n; i++)
        if (!strcmp(o->keys[i], key)) return o->items[i];
    return NULL;
}

js_val *js_arr_get(const js_val *a, int i) {
    if (!a || a->t != JS_ARR || i < 0 || i >= a->n) return NULL;
    return a->items[i];
}

const char *js_str(const js_val *v) {
    return v && v->t == JS_STR ? v->str : NULL;
}

double js_num(const js_val *v) {
    return v && v->t == JS_NUM ? v->num : NAN;
}

int js_bool(const js_val *v) {
    if (!v) return 0;
    if (v->t == JS_BOOL) return v->boolean;
    if (v->t == JS_NUM) return v->num != 0;
    return 0;
}
