#ifndef CF_JSON_H
#define CF_JSON_H

#include <stddef.h>

typedef enum { JS_NULL, JS_BOOL, JS_NUM, JS_STR, JS_ARR, JS_OBJ } js_type;

typedef struct js_val js_val;
struct js_val {
    js_type t;
    double num;
    int boolean;
    char *str;
    js_val **items;
    char **keys;
    int n;
};

js_val *js_parse(const char *src, char *err, size_t errlen);
void js_free(js_val *v);
js_val *js_obj_get(const js_val *o, const char *key);
js_val *js_arr_get(const js_val *a, int i);
const char *js_str(const js_val *v);
double js_num(const js_val *v);
int js_bool(const js_val *v);

#endif
