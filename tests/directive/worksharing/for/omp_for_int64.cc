// xkomp: supported  (__kmpc_for_static_init_8)
//
// Signed 64-bit loop counters lower to __kmpc_for_static_init_8, the last of
// the four static worksharing entry points to be implemented in src/kmp/for.cc.
// The unsigned 64-bit path (_8u) is covered by omp_for_unsigned.cc, the signed
// 32-bit one (_4) by omp_for.cc.

#include "common.h"

int
main(void)
{
    const long long N = LOOPCOUNT;
    long long total = 0;

    #pragma omp parallel num_threads(4) shared(total)
    {
        #pragma omp for
        for (long long i = 0; i < N; ++i)
        {
            #pragma omp atomic
            total += 1;
        }
    }

    CHECK_EQ(total, N);

    TEST_PASS();
    return 0;
}
