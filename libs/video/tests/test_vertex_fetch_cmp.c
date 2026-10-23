/* rsx_fetch_attrib on CELL_GCM_VERTEX_CMP (type 6): one big-endian 32-bit
 * word holding signed-normalised x:11, y:11, z:10 (low bits first), w = 1.
 * Games store normals this way; reading the word as a float gave garbage.
 *
 * Then rsx_fetch_vertex (a per-draw plan) against rsx_fetch_attrib on all 16
 * slots, over random attribute layouts: every type, size and stride,
 * frequency divide/modulo, LOCAL and MAIN arrays, disabled slots. The local
 * resolver adds a bias and the main one swaps 4 KB halves of each 8 KB block,
 * so a plan that resolved MAIN once per draw, or LOCAL through the wrong
 * space, reads the wrong bytes.
 * Build: clang-cl /I.. /I../../../include test_vertex_fetch_cmp.c ../rsx_vertex_fetch.c */
#include "../rsx_vertex_fetch.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GUEST_SIZE (1u << 20)
#define LOCAL_BIAS 0x100u
static u8* guest;
u8* vm_base;
u32 cellGcmResolveOffset(u32 offset) { return offset; }
u32 cellGcmResolveLocated(int location, u32 offset)
{
    return location ? offset + LOCAL_BIAS : (offset ^ 0x1000u) % (GUEST_SIZE - 0x10000u);
}

static rsx_state st;
static int failures;

static u32 pack(s32 x, s32 y, s32 z)
{
    return ((u32)x & 0x7FFu) | (((u32)y & 0x7FFu) << 11) | (((u32)z & 0x3FFu) << 22);
}

static void check(u32 word, float ex, float ey, float ez)
{
    u8* w = guest + LOCAL_BIAS;
    w[0] = (u8)(word >> 24); w[1] = (u8)(word >> 16);
    w[2] = (u8)(word >> 8);  w[3] = (u8)word;
    float v[4];
    rsx_fetch_attrib(&st, 2, 0, v);
    const float want[4] = { ex, ey, ez, 1.0f };
    for (int k = 0; k < 4; k++)
        if (fabsf(v[k] - want[k]) > 1e-3f) {
            printf("FAIL word=%08X comp %d: got %f want %f\n", word, k, v[k], want[k]);
            failures++;
        }
}

static void plan_vs_attrib(void)
{
    static const u32 types[] = { 1, 2, 3, 4, 5, 6, 7, 9 };   /* 9: unhandled, read as float */
    srand(31337);
    for (int layout = 0; layout < 2000; layout++) {
        memset(&st, 0, sizeof st);
        for (int i = 0; i < 16; i++) {
            for (int k = 0; k < 4; k++) st.vertex_data4f[i][k] = (float)(rand() % 1000) / 7.0f;
            rsx_vertex_attrib* a = &st.vertex_attribs[i];
            a->enabled = rand() % 3 != 0;
            a->type = types[rand() % 8];
            a->size = (u32)(rand() % 5);                        /* 0 means 4 */
            a->stride = (rand() % 8 == 0) ? 0 : (u32)(4 + 4 * (rand() % 16));
            a->offset = (u32)(rand() % 0x40000) | ((rand() & 1) ? 0x80000000u : 0);
            a->frequency = (rand() % 4 == 0) ? (u32)(rand() % 5) : 0;
        }
        st.frequency_divider_op = (u32)rand() & 0xFFFFu;
        rsx_fetch_plan plan;
        rsx_fetch_plan_init(&st, &plan);
        for (u32 vi = 0; vi < 40; vi++) {
            float got[16][4], want[16][4];
            rsx_fetch_vertex(&plan, vi * 7, got);
            for (int i = 0; i < 16; i++) rsx_fetch_attrib(&st, i, vi * 7, want[i]);
            if (memcmp(got, want, sizeof got) && failures++ < 10)
                printf("FAIL plan layout %d vertex %u\n", layout, vi * 7);
        }
    }
}

int main(void)
{
    guest = (u8*)malloc(GUEST_SIZE);
    for (u32 i = 0; i < GUEST_SIZE; i++) guest[i] = (u8)(i * 2654435761u >> 13);
    vm_base = guest;

    rsx_vertex_attrib* a = &st.vertex_attribs[2];
    a->enabled = 1; a->type = 6; a->size = 1; a->stride = 4; a->offset = 0;
    st.vertex_data4f[2][0] = 0.25f; st.vertex_data4f[2][3] = 0.5f;

    check(pack(1023, 0, 0),       1.0f,  0.0f,  0.0f);
    check(pack(0, 1023, 0),       0.0f,  1.0f,  0.0f);
    check(pack(0, 0, 511),        0.0f,  0.0f,  1.0f);
    check(pack(-1023, -1023, -511), -1.0f, -1.0f, -1.0f);
    check(pack(512, -512, 256),   512.0f / 1023, -512.0f / 1023, 256.0f / 511);

    for (u32 i = 0; i < GUEST_SIZE; i++) guest[i] = (u8)(i * 2654435761u >> 13);
    plan_vs_attrib();
    if (!failures) printf("test_vertex_fetch_cmp: OK\n");
    return failures ? 1 : 0;
}
