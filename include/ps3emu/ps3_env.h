/*
 * ps3_env -- getenv cached per call site.
 *
 * The RSX drain is full of debug switches read with a bare getenv() on
 * per-draw and per-vertex paths. The CRT's getenv takes a lock and scans the
 * whole environment on every call; in Warhawk's match that was ~37% of the
 * drain thread, more than the vertex fetch itself. The environment does not
 * change once the runtime is up (the harness sets its defaults in main()
 * before any thread starts), so each call site reads it once.
 *
 * A file includes this after <stdlib.h> and every getenv("LITERAL") in it
 * becomes cached. The cache is per call site, so the name must be the same
 * on every call: never pass a variable name through it.
 */
#ifndef PS3EMU_PS3_ENV_H
#define PS3EMU_PS3_ENV_H

#include <stdlib.h>

#define PS3_GETENV_ONCE(name) ({                                              \
    static const char* _ps3_env_v;                                            \
    static int _ps3_env_k;                                                    \
    if (!__atomic_load_n(&_ps3_env_k, __ATOMIC_ACQUIRE)) {                    \
        _ps3_env_v = (getenv)(name);                                          \
        __atomic_store_n(&_ps3_env_k, 1, __ATOMIC_RELEASE);                   \
    }                                                                         \
    _ps3_env_v; })

#define getenv(name) PS3_GETENV_ONCE(name)

#endif
