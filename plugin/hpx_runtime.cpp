#include <hpx/hpx_init.hpp>
#include <hpx/iostream.hpp>

#include <hpx/include/parallel_for_loop.hpp>

#include <Halide.h>
#include <HalideRuntime.h>

extern "C" int hpx_halide_do_par_for(void *ctx, int (*f)(void *, int, uint8_t *),
                              int min, int extent, uint8_t *closure) {
  hpx::for_loop(hpx::execution::par, min, min + extent,
                [&](int i) { f(ctx, i, closure); });
  return 0;
}

///////////////////////////////////////////////////////////////////////////
// Allow applications to add configuration settings if HPX_MAIN is set
std::vector<std::string>(*prev_user_main_config_function)(
    std::vector<std::string> const&) = nullptr;

std::vector<std::string> user_main_config(
    std::vector<std::string> const& config)
{
    // register halide custom handlers
    halide_set_custom_do_par_for(&hpx_halide_do_par_for);

    // If there was another config function registered, call it
    if (prev_user_main_config_function)
    {
        return prev_user_main_config_function(config);
    }
    return config;
}

// Make sure our configuration information is injected into the startup
// procedure
struct register_user_main_config
{
    register_user_main_config()
    {
        prev_user_main_config_function =
            hpx_startup::user_main_config_function;
        hpx_startup::user_main_config_function = &user_main_config;
    }
} cfg;
