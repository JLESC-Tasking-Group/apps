# include <stdio.h>
# include <stdint.h>
# include <stdlib.h>
# include <time.h>

uint64_t
power_get_nanotime(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)(ts.tv_sec * 1000000000) + (uint64_t) ts.tv_nsec;
}

# if CUDA_ENERGY
#  include <nvml.h>
static nvmlDevice_t nvdevice;

void
power_init(void)
{
    // Initialize Cuda energy counters
    nvmlReturn_t result;
    unsigned long long energy;

    // Initialize NVML
    result = nvmlInit();
    if (NVML_SUCCESS != result) {
        printf("Failed to initialize NVML: %s\n", nvmlErrorString(result));
        exit(1);
    }

    // Get the handle for the first device
    result = nvmlDeviceGetHandleByIndex(0, &nvdevice);
    if (NVML_SUCCESS != result) {
        printf("Failed to get handle for device 0: %s\n", nvmlErrorString(result));
        nvmlShutdown();
        exit(1);
    }

    // Description
    // Retrieves total energy consumption for this GPU in millijoules (mJ) since the driver was last reloaded
    // For Volta or newer fully supported devices.
    result = nvmlDeviceGetTotalEnergyConsumption(nvdevice, &energy);
    if (NVML_SUCCESS != result) {
        printf("Failed to get total energy consumption: %s\n", nvmlErrorString(result));
        nvmlShutdown();
        exit(1);
    }

    printf("Initialized nvml (CUDA_ENERGY=1) - initial energy is `%llu`\n", energy);
}

void
power_deinit(void)
{
    nvmlShutdown();
}

# elif ROCM_ENERGY
#include "rocm_smi/rocm_smi.h"
static RSMI_POWER_TYPE rsmi_type = RSMI_CURRENT_POWER;
static uint32_t num_devices;
static int rsmi_device = 0;

void
power_init(void)
{
    rsmi_init(0);
    rsmi_num_monitor_devices(&num_devices);
}

void
power_deinit(void)
{
}

# elif ZES_ENERGY
# include <zes_api.h>
static zes_driver_handle_t zes_driver;
static uint32_t zes_driver_count;

static zes_device_handle_t zes_device;
static uint32_t zes_device_count;

static zes_pwr_handle_t zes_pwr_handle;
static uint32_t zes_pwr_handle_count;

# define ZE_SAFE_CALL(X)                \
    do {                                \
        ze_result_t r = X;              \
        if (r != ZE_RESULT_SUCCESS)     \
            abort();                    \
    } while (0)

void
power_init(void)
{
    zes_init_flags_t flags = ZES_INIT_FLAG_PLACEHOLDER;
    ZE_SAFE_CALL(zesInit(flags));

    zes_driver_count = 1;
    ZE_SAFE_CALL(zesDriverGet(&zes_driver_count, &zes_driver));

    zes_device_count = 1;
    ZE_SAFE_CALL(zesDeviceGet(zes_driver, &zes_device_count, &zes_device));

    zes_pwr_handle_count = 1;
    ZE_SAFE_CALL(zesDeviceEnumPowerDomains(zes_device, &zes_pwr_handle_count, &zes_pwr_handle));
}

void
power_deinit(void)
{
}

# else /* ZES_ENERGY */

void
power_init(void)
{
}


void
power_deinit(void)
{
}


# endif

static uint64_t t1, t2;

# if CUDA_ENERGY
static unsigned long long mj1, mj2;
# endif

# if ROCM_ENERGY
static double e1, e2;
# endif

# if ZES_ENERGY
static zes_power_energy_counter_t e1, e2;
# endif

void
power_start(void)
{
    # if CUDA_ENERGY
    nvmlDeviceGetTotalEnergyConsumption(nvdevice, &mj1);
    # endif

    # if ROCM_ENERGY
    uint64_t mj;
    float res;
    uint64_t timestamp;
    rsmi_dev_energy_count_get(rsmi_device, &mj, &res, &timestamp);
    e1 = (double) mj * res * 0.000001;
# endif

    # if ZES_ENERGY
    ZE_SAFE_CALL(zesPowerGetEnergyCounter(zes_pwr_handle, &e1));
    # endif

    t1 = power_get_nanotime();
}

double
power_stop(void)
{
    t2 = power_get_nanotime();
    double s = (t2 - t1) / (double) 1e9;

    double J = 0.0;

    # if CUDA_ENERGY
    nvmlDeviceGetTotalEnergyConsumption(nvdevice, &mj2);
    J = (mj2 - mj1) / (double)1e3;
    # endif

    # if ROCM_ENERGY
    uint64_t mj;
    float res;
    uint64_t timestamp;
    rsmi_dev_energy_count_get(rsmi_device, &mj, &res, &timestamp);
    e2 = (double) mj * res * 0.000001;
    J = (e2 - e1);
    # endif

    # if ZES_ENERGY
    ZE_SAFE_CALL(zesPowerGetEnergyCounter(zes_pwr_handle, &e2));
    double uJ = (double) (e2.energy - e1.energy);
    J = uJ / (double)1e6;
    # endif

    double P = J / s;
    return P;
}
