/* The AVX2 cflts/cfltu/csflt/cuflt must match their lane-loop references bit
 * for bit for every scale (i8 0..255) and for words covering every exponent,
 * zeros, INT_MIN and the saturation edges. Build with -mavx2. */
#include "../spu_helpers.h"
#include <stdio.h>
#include <stdlib.h>

static uint32_t rnd32(void) { return ((uint32_t)rand() << 17) ^ ((uint32_t)rand() << 6) ^ (uint32_t)rand(); }

static int fails = 0;
static void check(const char* what, int i8, u128 a, u128 x, u128 y)
{
    if (!memcmp(&x, &y, 16)) return;
    if (fails++ < 12)
        for (int i = 0; i < 4; i++) if (x._u32[i] != y._u32[i])
            printf("test_spu_cvt: %s(%08X, %d) = %08X, want %08X\n", what, a._u32[i], i8, x._u32[i], y._u32[i]);
}

int main(void)
{
#if !defined(__AVX2__)
    puts("test_spu_cvt: built without AVX2, nothing to compare");
    return 0;
#else
    static const uint32_t special[] = { 0, 0x80000000u, 0x7FFFFFFFu, 0xFFFFFFFFu, 1, 0x00800000u,
                                        0x7F800000u, 0xFF800000u, 0x4F000000u, 0xCF000000u, 0x4F800000u };
    srand(2718);
    for (int it = 0; it < 40000; it++) {
        u128 a;
        for (int i = 0; i < 4; i++) {
            a._u32[i] = rnd32();
            if (rand() % 4 == 0) a._u32[i] = special[rand() % 11];
            else if (rand() % 3 == 0) a._u32[i] >>= rand() % 32;           /* small magnitudes */
        }
        for (int i8 = 0; i8 < 256; i8++) {
            check("cflts", i8, a, spu_cflts(a, i8), spu_cflts_ref(a, i8));
            check("cfltu", i8, a, spu_cfltu(a, i8), spu_cfltu_ref(a, i8));
            check("csflt", i8, a, spu_csflt(a, i8), spu_csflt_ref(a, i8));
            check("cuflt", i8, a, spu_cuflt(a, i8), spu_cuflt_ref(a, i8));
        }
    }
    if (fails) { printf("test_spu_cvt: %d mismatches\n", fails); return 1; }
    puts("test_spu_cvt: ok (40000 x 256 scales)");
    return 0;
#endif
}
