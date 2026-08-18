#ifndef __TIMER_H__
#define __TIMER_H__

#ifdef __cplusplus
extern "C" {
#endif

// Initialize 'nb_timers' timers
#define TIMERS_INIT(nb_timers) timers_init(nb_timers)

// Start timer 'timer_id'
#define TICK(timer_id) timers_tick(timer_id)

// Stop timer 'timer_id' and returns its elapsed time
#define TOCK(timer_id) timers_tock(timer_id)

// Returns timer 'timer_id' elapsed time
#define GET_TIME(timer_id) timers_get_time(timer_id)

void timers_init(unsigned int nb_timers);
void timers_tick(unsigned int timer_id);
double timers_tock(unsigned int timer_id);
double timers_get_time(unsigned int timer_id);

#ifdef __cplusplus
}
#endif

#endif // __TIMER_H__