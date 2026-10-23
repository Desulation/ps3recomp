/* The SSE spu_fi must match the lane-loop reference bit for bit: random
 * estimates and interpolants, plus the exponent edges (0, 1, 254, 255) where
 * the zero, underflow and top-of-range rules apply. Build with -msse4.1. */
#include "../spu_helpers.h"
#include <stdio.h>
#include <stdlib.h>

static uint32_t rnd32(void) { return ((uint32_t)rand() << 17) ^ ((uint32_t)rand() << 6) ^ (uint32_t)rand(); }

int main(void)
{
#if !defined(SPU_SSE_PERM)
    puts("test_spu_fi: built without SSE4.1, nothing to compare");
    return 0;
#else
    static const uint32_t edge_exp[] = { 0, 1, 2, 127, 254, 255 };
    srand(777);
    int fails = 0;
    for (int it = 0; it < 2000000; it++) {
        u128 a, b;
        for (int i = 0; i < 4; i++) {
            a._u32[i] = rnd32();
            b._u32[i] = rnd32();
            if (it & 1)
                b._u32[i] = (b._u32[i] & ~0x7F800000u) | (edge_exp[rand() % 6] << 23);
            if ((it & 7) == 2) a._u32[i] |= 0x7FFFFu;           /* largest Y */
            if ((it & 7) == 4) b._u32[i] |= 0x3FFu;             /* largest step */
        }
        u128 x = spu_fi(a, b), y = spu_fi_ref(a, b);
        if (memcmp(&x, &y, 16) && fails++ < 10)
            printf("test_spu_fi: MISMATCH a=%08X b=%08X got %08X want %08X\n",
                   a._u32[0], b._u32[0], x._u32[0], y._u32[0]);
    }
    if (fails) { printf("test_spu_fi: %d mismatches\n", fails); return 1; }
    puts("test_spu_fi: ok (2000000 random cases)");
    return 0;
#endif
}
