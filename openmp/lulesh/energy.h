# ifndef __POWER_LIB_H__
#  define __POWER_LIB_H__
# include <stdint.h>

uint64_t power_get_nanotime(void);
void power_init(void);
void power_deinit(void);
void   power_start(void);
double power_stop(void);

#endif /* __POWER_LIB_H__ */
