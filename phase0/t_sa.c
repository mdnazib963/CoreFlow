#include <stdatomic.h>
int main(void) {
    atomic_size_t x;
    atomic_store(&x, 1);
    return (int)atomic_load(&x) - 1;
}
