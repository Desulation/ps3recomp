/* The SSE quadword byte shifts/rotates and insertion-control generators must
 * match their byte-loop references for every shift count and position. Build
 * with -msse4.1 on x86 so the SIMD versions are the ones under test. */
#include "../spu_helpers.h"
#include <stdio.h>
#include <stdlib.h>

static int fails = 0;

static void check(const char* what, int arg, u128 x, u128 y)
{
    if (memcmp(&x, &y, 16) && fails++ < 10)
        printf("test_spu_byteperm: MISMATCH %s(%d)\n", what, arg);
}

int main(void)
{
#if !defined(SPU_SSE_PERM)
    puts("test_spu_byteperm: built without SSE4.1, nothing to compare");
    return 0;
#else
    srand(4242);
    for (int it = 0; it < 2000; it++) {
        u128 a;
        for (int i = 0; i < 16; i++) a._u8[i] = (uint8_t)rand();
        /* Every immediate a 7-bit field can carry, and register counts with
         * junk in the bits the instruction ignores. */
        for (int sh = -64; sh < 64; sh++) {
            u128 b; for (int i = 0; i < 16; i++) b._u8[i] = (uint8_t)rand();
            b._u32[0] = (uint32_t)sh ^ ((uint32_t)rand() << 8 & 0xFFFFFF00u);
            check("shlqbyi", sh, spu_shlqbyi(a, sh), spu_shlqbyi_ref(a, sh));
            check("rotqbyi", sh, spu_rotqbyi(a, sh), spu_rotqbyi_ref(a, sh));
            check("rotqmbyi", sh, spu_rotqmbyi(a, sh), spu_rotqmbyi_ref(a, sh));
            check("shlqby", sh, spu_shlqby(a, b), spu_shlqby_ref(a, b));
            check("rotqby", sh, spu_rotqby(a, b), spu_rotqby_ref(a, b));
            check("shlqbybi", sh, spu_shlqbybi(a, b), spu_shlqbybi_ref(a, b));
            check("rotqbybi", sh, spu_rotqbybi(a, b), spu_rotqbybi_ref(a, b));
            check("rotqmby", sh, spu_rotqmby(a, b), spu_rotqmby_ref(a, b));
            check("rotqmbybi", sh, spu_rotqmbybi(a, b), spu_rotqmbybi_ref(a, b));
            /* Counts that reach the bit-shifted forms' upper bits. */
            b._u32[0] = (uint32_t)(sh * 8) ^ ((uint32_t)rand() << 12 & 0xFFFFF000u);
            check("shlqbybi8", sh, spu_shlqbybi(a, b), spu_shlqbybi_ref(a, b));
            check("rotqbybi8", sh, spu_rotqbybi(a, b), spu_rotqbybi_ref(a, b));
            check("rotqmbybi8", sh, spu_rotqmbybi(a, b), spu_rotqmbybi_ref(a, b));
        }
    }
    for (int pos = -300; pos < 300; pos++) {
        check("cbd", pos, spu_cbd_pos(pos), spu_cbd_pos_ref(pos));
        check("chd", pos, spu_chd_pos(pos), spu_chd_pos_ref(pos));
        check("cwd", pos, spu_cwd_pos(pos), spu_cwd_pos_ref(pos));
        check("cdd", pos, spu_cdd_pos(pos), spu_cdd_pos_ref(pos));
        u128 a; a._u32[0] = (uint32_t)pos; a._u32[1] = a._u32[2] = a._u32[3] = 0xDEADBEEFu;
        check("cwd-insn", pos, spu_cwd(a, 3), spu_cwd_ref(a, 3));
    }
    if (fails) { printf("test_spu_byteperm: %d mismatches\n", fails); return 1; }
    puts("test_spu_byteperm: ok");
    return 0;
#endif
}
