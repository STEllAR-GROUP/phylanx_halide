#include <hpx/config.hpp>

#include <Halide.h>
#include <HalideRuntime.h>

#include <hpx/hpx_init.hpp>
#include <hpx/iostream.hpp>
#include <hpx/include/parallel_for_loop.hpp>

extern "C" int hpx_halide_do_par_for(void *ctx, int (*f)(void *, int, uint8_t *),
                              int min, int extent, uint8_t *closure) {
  hpx::for_loop(hpx::execution::par, min, min + extent,
                [&](int i) { f(ctx, i, closure); });
  return 0;
}

// Make sure to register the HPX backend functionalities
struct on_load
{
    on_load()
    {
        // register halide custom handlers
        ::halide_set_custom_do_par_for(&hpx_halide_do_par_for);
    }
} cfg;
