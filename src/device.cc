# include <omp.h>
# include <xkomp/xkomp.h>
# include <assert.h>

/* In OpenMP, non-host devices are indexed from 0 to n-1 where 'n' is the value
 * returned by 'omp_get_num_devices' */

extern "C"
void
xkomp_set_default_device(int device)
{
    LOGGER_NOT_IMPLEMENTED();
}
EXPORT_OMP_ABI(set_default_device);

extern "C"
int
xkomp_get_default_device(void)
{
    return 0;
}
EXPORT_OMP_ABI(get_default_device);

extern "C"
int
xkomp_get_num_devices(void)
{
    xkomp_t * xkomp = xkomp_get();
    return xkomp->runtime.get_ndevices() - 1;
}
EXPORT_OMP_ABI(get_num_devices);

extern "C"
int xkomp_get_initial_device(void);

extern "C"
int
xkomp_get_device_num(void)
{
    thread_t * thread = thread_t::get_tls();
    return (int) (thread->device_unique_id == XKRT_HOST_DEVICE_UNIQUE_ID ? xkomp_get_initial_device() : thread->device_unique_id - 1);
}
EXPORT_OMP_ABI(get_device_num);

extern "C"
int
xkomp_get_initial_device(void)
{
    return xkomp_get_num_devices();
}
EXPORT_OMP_ABI(get_initial_device);

extern "C"
int
xkomp_is_initial_device(void)
{
    return xkomp_get_device_num() == xkomp_get_initial_device();
}
EXPORT_OMP_ABI(is_initial_device);

extern "C"
void *
xkomp_target_alloc(size_t size, int device_num)
{
   return NULL;
}
EXPORT_OMP_ABI(target_alloc);

extern "C"
void
xkomp_target_free(void * device_ptr, int device_num)
{
}
EXPORT_OMP_ABI(target_free);


/////////////////////////////
// TARGET MEMORY TRANSFERS //
/////////////////////////////

extern "C"
int
xkomp_target_memcpy_async(
    void * dst,
    const void * src,
    size_t length,
    size_t dst_offset,
    size_t src_offset,
    int dst_device_num,
    int src_device_num,
    int depobj_count,
    omp_depend_t * depobj_list
) {
    LOGGER_FATAL("Not implemented");
    return -1;
}
EXPORT_OMP_ABI(target_memcpy_async);

/////////////////////
// XKOMP EXTENSION //
/////////////////////

extern "C"
xkrt_device_unique_id_t
omp_device_id_to_xkomp(int device_id)
{
    static_assert(XKRT_HOST_DEVICE_UNIQUE_ID == 0);
    if (device_id == xkomp_get_initial_device())
        return XKRT_HOST_DEVICE_UNIQUE_ID;
    return 1 + (device_id % xkomp_get_num_devices());
}

extern "C"
int
xkomp_device_unique_id_to_omp(xkrt_device_unique_id_t device_unique_id)
{
    return (device_unique_id == XKRT_HOST_DEVICE_UNIQUE_ID) ? xkomp_get_num_devices() : (int) (device_unique_id - 1);
}
