#include <stdio.h>
#include <stdlib.h>
#include <time.h>


#ifdef __cplusplus
extern "C" {
#endif


struct llm_timer_t {
    struct timespec start;
    struct timespec stop;
    int is_running;
};


struct llm_timer_t *_timers = NULL;
unsigned int _nb_timers = 0;

void timer_init(struct llm_timer_t *timer) { timer->is_running = 0; }

double timer_get_time(struct llm_timer_t *timer) {
    double t_s = 0.;
    if (!timer->is_running) {
        t_s = (timer->stop.tv_sec - timer->start.tv_sec) + (timer->stop.tv_nsec - timer->start.tv_nsec) / 1e9;
    } else {
        printf("TIMER WARNING : Timer is still running\n");
    }
    return t_s;
}

void timer_tick(struct llm_timer_t *timer) {
    if (!timer->is_running) {
        clock_gettime(CLOCK_MONOTONIC, &timer->start);
        timer->is_running = 1;
    } else
        printf("TIMER WARNING : Timer was already running\n");
}

double timer_tock(struct llm_timer_t *timer) {
    double t_s = 0.;
    if (timer->is_running) {
        clock_gettime(CLOCK_MONOTONIC, &timer->stop);
        timer->is_running = 0;
        t_s = timer_get_time(timer);
    } else
        printf("TIMER WARNING : Timer was not running\n");
    return t_s;
}

void _timers_destroy() {
    if (_timers != NULL) {
        free(_timers);
        _timers = NULL;
        _nb_timers = 0;
    }
}

void timers_init(unsigned int nb_timers) 
{
    if (_timers == NULL) 
    {
        #if __cplusplus
        _timers = new llm_timer_t[nb_timers];
        #else
        _timers = malloc(nb_timers * sizeof(struct llm_timer_t));
        #endif
        _nb_timers = nb_timers;
        for (unsigned int i = 0; i < nb_timers; i++)
            timer_init(_timers + i);
        atexit(_timers_destroy);
    } else {
        printf("TIMER WARNING : Timers were already initialized\n");
    }
}

void timers_tick(unsigned int timer_id) {
    if (_timers == NULL) {
        printf("TIMER WARNING : No timers to tick\n");
        return;
    }
    if (timer_id >= _nb_timers) {
        printf("TIMER_WARNING : Given timer id is out of range\n");
        return;
    }

    timer_tick(_timers + timer_id);
}

double timers_tock(unsigned int timer_id) {
    if (_timers == NULL) {
        printf("TIMER WARNING : No timers to tock\n");
        return 0.;
    }
    if (timer_id >= _nb_timers) {
        printf("TIMER_WARNING : Given timer id is out of range\n");
        return 0.;
    }

    return timer_tock(_timers + timer_id);
}

double timers_get_time(unsigned int timer_id) {
    if (_timers == NULL) {
        printf("TIMER WARNING : No timers to get time from\n");
        return 0.;
    }
    if (timer_id >= _nb_timers) {
        printf("TIMER_WARNING : Given timer id is out of range\n");
        return 0.;
    }

    return timer_get_time(_timers + timer_id);
}

#ifdef __cplusplus
}
#endif
