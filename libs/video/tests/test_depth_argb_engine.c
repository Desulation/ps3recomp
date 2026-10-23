/* The draw engine's GPU route for a Z24S8 depth target sampled as A8R8G8B8
 * (rsx_draw_backend.depth_as_argb), through its public methods with a
 * synthetic backend: when it is taken, what it is asked for, when it is asked
 * again, and every case that must still go through guest memory. No game
 * programs, original assets or graphics API are used; the resolve's texels
 * themselves are test_vk_depth_argb.c's.
 *
 *   clang -std=c11 -Wno-comment -I include -I libs/video \
 *     libs/video/tests/test_depth_argb_engine.c libs/video/rsx_draw_engine.c \
 *     libs/video/rsx_dispatch.c libs/video/rsx_vertex_fetch.c \
 *     libs/video/rsx_vertex_compact.c libs/video/rsx_texture_layout.c \
 *     libs/video/rsx_vp_decompiler.c libs/video/rsx_fp_decompiler.c \
 *     -lm -o /tmp/test_depth_argb_engine
 */
#define main original_engine_test_main
#include "test_rsx_draw_engine.c"
#undef main

#define ZW 12u
#define ZH 2u
#define ZPITCH 80u
#define REMAP 0xAA93u
#define OTHER_ZETA 0xA0000u

static struct { int n; u32 depth, w, h, rgba; u32 handle; } argb;
static struct { int n; u32 surface, remap, fmt; } view;
static int argb_fails;

static u32 stub_argb(void* u, u32 depth, u32 w, u32 h, int rgba)
{
    (void)u;
    argb.n++; argb.depth = depth; argb.w = w; argb.h = h; argb.rgba = (u32)rgba;
    if (argb_fails) return 0;
    argb.handle = 0x7000u + (u32)argb.n;
    return argb.handle;
}
static u32 stub_view(void* u, u32 surface, u32 remap, u32 fmt)
{
    (void)u;
    view.n++; view.surface = surface; view.remap = remap; view.fmt = fmt;
    return surface >= 0x7000u ? surface + 0x100u : surface;
}
static int flat_depth(void* u, u32 depth, u32 x, u32 y, u32 w, u32 h, float* out, u32 pitch)
{
    (void)u; (void)depth; (void)x; (void)y;
    stub.n_depth_readback++;
    for (u32 r = 0; r < h; r++)
        for (u32 c = 0; c < w; c++) *(float*)((u8*)out + r * pitch + c * 4u) = 0.5f;
    return 0;
}

typedef struct {
    const char* name;
    int  hook;          /* depth_as_argb registered */
    u32  zfmt;          /* SURFACE_FORMAT depth field: 2 Z24S8, 1 Z16 */
    u32  tex_fmt;       /* SET_TEXTURE_FORMAT byte */
    u32  tex_w, tex_pitch;
    int  zeta_bound;    /* the sampling draw still has it as its zeta */
} argb_case;

/* Write depth through a zeta at ZETA_OFFSET, then sample that address with
 * texture unit 0 as the case describes, from a pass with another zeta (as
 * Warhawk's lighting pass has) unless the case keeps it. Returns the handle
 * bound. */
static u32 depth_then_sample(const argb_case* c)
{
    m(M_TEX_UNIT + 0x0C, 0);
    m(M_ZETA_OFFSET, ZETA_OFFSET);
    m(M_COLOR_A_OFFSET, ZETA_OFFSET); m(M_COLOR_MASK, 0);
    m(M_DEPTH_TEST, 1); m(M_DEPTH_WRITE, 1);
    draw_triangle();
    if (!c->zeta_bound) m(M_ZETA_OFFSET, OTHER_ZETA);
    m(M_COLOR_A_OFFSET, 0x90000u); m(M_COLOR_MASK, 0x01010101);
    m(M_DEPTH_TEST, 0); m(M_DEPTH_WRITE, 0);
    m(M_TEX_UNIT + 0x0C, 0x80000000u);
    draw_triangle();
    return stub.bound_tex[0];
}

static void setup(const argb_case* c, rsx_draw_backend* be)
{
    *be = g_stub_backend;
    be->depth_readback = flat_depth;
    be->surface_view = stub_view;
    be->depth_as_argb = c->hook ? stub_argb : NULL;
    memset(&stub, 0, sizeof stub); memset(&argb, 0, sizeof argb); memset(&view, 0, sizeof view);
    rsx_draw_engine_set_backend(be);
    CHECK(rsx_draw_engine_init(32, 8) == 0, "%s: engine initializes", c->name);
    m(M_CONTEXT_DMA_COLOR_A, DMA_LOCAL); m(M_SURFACE_CLIP_H, ZW << 16); m(M_SURFACE_CLIP_V, ZH << 16);
    m(M_VTXFMT, 2u | (4u << 4) | (16u << 8)); m(M_VTXBUF_OFFSET, 0x80000000u | VTX_OFFSET);
    m(M_CONTEXT_DMA_ZETA, DMA_LOCAL); m(M_SURFACE_FORMAT, 8u | (c->zfmt << 5));
    m(M_ZETA_OFFSET, ZETA_OFFSET); m(M_ZETA_PITCH, ZPITCH); m(M_COLOR_A_OFFSET, 0x90000u);
    memset(vm_base + GUEST_LOCAL_EA + ZETA_OFFSET - 16, 0xA6, ZPITCH * 3 + 16);
    bind_texture_unit(RSX_LOCATION_LOCAL, ZETA_OFFSET);
    m(M_TEX_UNIT + 0x04, 1u | (2u << 4) | (c->tex_fmt << 8) | (1u << 16));
    m(M_TEX_UNIT + 0x10, REMAP);
    m(M_TEX_UNIT + 0x18, (c->tex_w << 16) | ZH);
    m(0x1840, c->tex_pitch);
}

static int guest_untouched(void)
{
    const u8* p = vm_base + GUEST_LOCAL_EA + ZETA_OFFSET;
    for (u32 i = 0; i < ZPITCH * ZH; i++) if (p[i] != 0xA6) return 0;
    return 1;
}

/* The route is taken: one resolve per depth write, nothing in guest memory. */
static void taken(void)
{
    const argb_case c = { "taken", 1, 2u, 0x85u | 0x20u, ZW, ZPITCH, 0 };
    rsx_draw_backend be;
    setup(&c, &be);
    const u32 bound = depth_then_sample(&c);
    CHECK(argb.n == 1 && argb.w == ZW && argb.h == ZH && argb.rgba == 0,
          "taken: one resolve of the used %ux%u rectangle, ARGB order (got %d, %ux%u)", ZW, ZH, argb.n, argb.w, argb.h);
    CHECK(argb.depth != 0, "taken: the resolve names the depth target");
    CHECK(view.surface == argb.handle && view.remap == REMAP && view.fmt == (0x85u | 0x20u),
          "taken: sampled through a view with the unit's crossbar and format byte");
    CHECK(bound == argb.handle + 0x100u, "taken: the unit binds that view (bound %X)", bound);
    CHECK(stub.n_depth_readback == 0 && guest_untouched(),
          "taken: no readback, guest memory not written (%d readbacks)", stub.n_depth_readback);
    draw_triangle();
    CHECK(argb.n == 1 && stub.bound_tex[0] == bound, "taken: unchanged depth reuses the resolve");
    depth_then_sample(&c);
    CHECK(argb.n == 2 && stub.n_depth_readback == 0, "taken: a new depth write resolves again");
    m(M_ZETA_OFFSET, ZETA_OFFSET);
    m(M_ZSTENCIL_CLEAR, 0xFFFFFF00u); m(M_CLEAR_BUFFERS, 1u);
    m(M_ZETA_OFFSET, OTHER_ZETA);
    draw_triangle();
    CHECK(argb.n == 3 && stub.n_depth_readback == 0, "taken: so does a depth clear");
    engine_down();
}

/* Every case the resolve does not reproduce goes through guest memory. */
static void falls_back(const argb_case* c, int expect_asked)
{
    rsx_draw_backend be;
    setup(c, &be);
    depth_then_sample(c);
    CHECK(stub.n_depth_readback == 1 && !guest_untouched(),
          "%s: stored into guest memory (%d readbacks)", c->name, stub.n_depth_readback);
    CHECK(argb.n == expect_asked, "%s: resolve asked %d time(s) (got %d)", c->name, expect_asked, argb.n);
    engine_down();
}

int main(void)
{
    vm_base = calloc(1, GUEST_BYTES);
    if (!vm_base) return 1;
    ppu_vm_size = GUEST_BYTES;
    for (u32 i = 0; i < 12; i++) { u8* p = vm_base + VTX_OFFSET + i * 4; p[0] = 0x3F; p[1] = 0x80; }

    taken();
    const argb_case no_hook  = { "no hook",          0, 2u, 0x85u | 0x20u, ZW,      ZPITCH,      0 };
    const argb_case swizzled = { "swizzled texture", 1, 2u, 0x85u,         ZW,      ZPITCH,      0 };
    const argb_case narrower = { "narrower texture", 1, 2u, 0x85u | 0x20u, ZW - 1u, ZPITCH,      0 };
    const argb_case pitch    = { "other pitch",      1, 2u, 0x85u | 0x20u, ZW,      ZPITCH + 16, 0 };
    const argb_case z16      = { "Z16 zeta",         1, 1u, 0x85u | 0x20u, ZW,      ZPITCH,      0 };
    const argb_case bound    = { "zeta still bound", 1, 2u, 0x85u | 0x20u, ZW,      ZPITCH,      1 };
    falls_back(&no_hook, 0);
    falls_back(&swizzled, 0);
    falls_back(&narrower, 0);
    falls_back(&pitch, 0);
    falls_back(&z16, 0);
    falls_back(&bound, 0);
    argb_fails = 1;
    const argb_case refused  = { "backend refuses",  1, 2u, 0x85u | 0x20u, ZW,      ZPITCH,      0 };
    falls_back(&refused, 1);
    argb_fails = 0;

    free(vm_base);
    printf("depth as A8R8G8B8: %d failures\n", g_failures);
    return g_failures ? 1 : 0;
}
