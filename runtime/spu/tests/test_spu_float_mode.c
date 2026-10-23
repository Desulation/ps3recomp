/* In the SPU single-precision FPU mode (spu_fp_enter_spu_mode) fa/fs/fm/fma/
 * fms/fnms take a native path; they must match the exact references bit for
 * bit, and the double-precision ops must give the same results in and out of
 * the mode. Operands cover every exponent class: 0 (zero on the SPU,
 * denormal on x86), 1, 254 and 255 (extended range on the SPU, Inf/NaN on
 * x86), and ordinary values close enough together that sums cancel. Build
 * with -mavx2 -mfma so the native path is the one under test. */
#include "../spu_helpers.h"
#include <stdio.h>
#include <stdlib.h>

static uint32_t rnd32(void) { return ((uint32_t)rand() << 17) ^ ((uint32_t)rand() << 6) ^ (uint32_t)rand(); }

static uint32_t operand(void)
{
    static const uint32_t exps[] = { 0, 1, 2, 100, 126, 127, 128, 150, 200, 253, 254, 255 };
    uint32_t w = rnd32();
    switch (rand() % 4) {
    case 0: return w;                                                         /* anything */
    case 1: return (w & ~0x7F800000u) | (exps[rand() % 12] << 23);            /* edge exponents */
    case 2: return (w & 0x807FFFFFu) | ((uint32_t)(120 + rand() % 16) << 23); /* near 1.0: cancellation */
    default: return rand() % 8 ? (w & 0x80000000u) : 0;                       /* zeros */
    }
}

/* Exponent 200..255: with these the native result is often not the SPU's (an operand or the
 * result at exponent 255, or a result past FLT_MAX) and the helpers take the widened path. */
static uint32_t big_operand(void)
{
    return (rnd32() & 0x807FFFFFu) | ((uint32_t)(200 + rand() % 56) << 23);
}

static int fails = 0;
static void check(const char* what, u128 x, u128 y, u128 a, u128 b, u128 c)
{
    if (!memcmp(&x, &y, 16)) return;
    if (fails++ < 12)
        for (int i = 0; i < 4; i++) if (x._u32[i] != y._u32[i])
            printf("test_spu_float_mode: %s(%08X, %08X, %08X) = %08X, want %08X\n",
                   what, a._u32[i], b._u32[i], c._u32[i], x._u32[i], y._u32[i]);
}

int main(void)
{
#if !defined(SPU_FP_MODE_BITS)
    puts("test_spu_float_mode: no SPU FPU mode on this target, nothing to compare");
    return 0;
#else
    srand(9001);
    for (int it = 0; it < 1500000; it++) {
        u128 a, b, c;
        if (it < 1000000) {
            for (int i = 0; i < 4; i++) { a._u32[i] = operand(); b._u32[i] = operand(); c._u32[i] = operand(); }
        } else {    /* the widened path, with every kind of partner */
            for (int i = 0; i < 4; i++) {
                a._u32[i] = big_operand(); b._u32[i] = rand() % 2 ? big_operand() : operand();
                c._u32[i] = rand() % 2 ? big_operand() : operand();
            }
            if (it & 1) { const u128 t = a; a = b; b = t; }
        }
        /* Make some fma addends cancel the product exactly or nearly. */
        if (it % 5 == 0) c = spu_fm_ref(a, b), c._u32[it & 3] ^= 0x80000000u;
        if (it % 7 == 0) c._u32[1] = spu_fm_ref(a, b)._u32[1] ^ 0x80000001u;

        spu_fp_leave_spu_mode();
        const u128 ra = spu_fa_ref(a, b), rs = spu_fs_ref(a, b), rm = spu_fm_ref(a, b);
        const u128 rfa = spu_fma_ref(a, b, c), rfs = spu_fms_ref(a, b, c), rfn = spu_fnms_ref(a, b, c);
        const u128 da = spu_dfa(a, b), dm = spu_dfm(a, b), dma = spu_dfma(a, b, c), rd = spu_frds(a);

        spu_fp_enter_spu_mode();
        check("fa", spu_fa(a, b), ra, a, b, b);
        check("fs", spu_fs(a, b), rs, a, b, b);
        check("fm", spu_fm(a, b), rm, a, b, b);
        check("fma", spu_fma(a, b, c), rfa, a, b, c);
        check("fms", spu_fms(a, b, c), rfs, a, b, c);
        check("fnms", spu_fnms(a, b, c), rfn, a, b, c);
        check("fa_ref", spu_fa_ref(a, b), ra, a, b, b);      /* references unaffected by the mode */
        check("fma_ref", spu_fma_ref(a, b, c), rfa, a, b, c);
        check("dfa", spu_dfa(a, b), da, a, b, b);
        check("dfm", spu_dfm(a, b), dm, a, b, b);
        check("dfma", spu_dfma(a, b, c), dma, a, b, c);
        check("frds", spu_frds(a), rd, a, a, a);
        if ((_mm_getcsr() & SPU_FP_MODE_BITS) != SPU_FP_MODE_BITS && fails++ < 12)
            puts("test_spu_float_mode: a helper left the SPU mode");
    }
    spu_fp_leave_spu_mode();
    if (fails) { printf("test_spu_float_mode: %d mismatches\n", fails); return 1; }
    puts("test_spu_float_mode: ok (1500000 random cases, 500000 of them mostly at exponents 200-255)");
    return 0;
#endif
}
