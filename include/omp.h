#ifndef __OMP_H__
# define __OMP_H__

# include <stddef.h>

/**
 * omp_depend_t -- implementation-defined depend object.
 * Each ABI (XKOMP, KMP, GOMP) interprets the pointee differently.
 */
typedef void * omp_depend_t;

/**
 * For each omp_<suffix> function, declare both the omp_<suffix> and
 * xkomp_<suffix> prototypes.  The actual symbol versioning is handled by
 * the EXPORT_OMP_ABI macro in the .cc files; the declarations here let
 * callers use either name.
 */
#define DECLARE_OMP_ABI(ret, suffix, ...) \
    ret omp_##suffix(__VA_ARGS__);        \
    ret xkomp_##suffix(__VA_ARGS__)

#ifdef __cplusplus
extern "C"
{
# endif /* __cplusplus */
    // threading
    DECLARE_OMP_ABI(int, get_thread_num, void);
    DECLARE_OMP_ABI(int, get_num_threads, void);
    DECLARE_OMP_ABI(int, get_max_threads, void);

    // timing
    DECLARE_OMP_ABI(double, get_wtime, void);

    // devices
    DECLARE_OMP_ABI(void, set_default_device, int device);
    DECLARE_OMP_ABI(int,  get_default_device, void);
    DECLARE_OMP_ABI(int,  get_num_devices, void);
    DECLARE_OMP_ABI(int,  get_device_num, void);
    DECLARE_OMP_ABI(int,  get_initial_device, void);
    DECLARE_OMP_ABI(int,  is_initial_device, void);

    // target memory
    DECLARE_OMP_ABI(int, xkomp_target_memcpy_async,
        void *dst,
        const void *src,
        size_t length,
        size_t dst_offset,
        size_t src_offset,
        int dst_device_num,
        int src_device_num,
        int depobj_count,
        omp_depend_t *depobj_list
    );

    // TODO: currently in libomptarget, should be moved to xkomp
    void * omp_get_mapped_ptr(const void * ptr, int device_num);

    typedef enum    omp_sync_hint_t
    {
        omp_sync_hint_none           = 0,
        omp_lock_hint_none           = omp_sync_hint_none,
        omp_sync_hint_uncontended    = (1 << 0),
        omp_lock_hint_uncontended    = omp_sync_hint_uncontended,
        omp_sync_hint_contended      = (1 << 1),
        omp_lock_hint_contended      = omp_sync_hint_contended,
        omp_sync_hint_nonspeculative = (1 << 2),
        omp_lock_hint_nonspeculative = omp_sync_hint_nonspeculative,
        omp_sync_hint_speculative    = (1 << 3),
        omp_lock_hint_speculative    = omp_sync_hint_speculative,
        kmp_lock_hint_hle            = (1 << 16),
        kmp_lock_hint_rtm            = (1 << 17),
        kmp_lock_hint_adaptive       = (1 << 18)

    }               omp_sync_hint_t;

    typedef void * omp_event_handle_t;

#ifdef __cplusplus
};
#endif /* __cplusplus */

#undef DECLARE_OMP_ABI

#endif /* __OMP_H__ */
