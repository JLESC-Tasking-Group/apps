# include <xkomp/xkomp.h>
# include <xkomp/kmp.h>

extern "C"
void
__kmpc_for_static_fini(
    ident_t * loc,
    kmp_int32 global_tid
) {
}

extern "C"
void
__kmpc_for_static_init_8(
    ident_t * loc,
    kmp_int32 gtid,
    kmp_int32 schedtype,
    kmp_int32 *plastiter,
    kmp_int64 *plower,
    kmp_int64 *pupper,
    kmp_int64 *pstride,
    kmp_int64 incr,
    kmp_int64 chunk
) {
    assert(schedtype == kmp_sch_static);
    team_t::parallel_for_thread_bounds(plastiter, plower, pupper, incr);
}

extern "C"
void
__kmpc_for_static_init_4(
    ident_t *loc,
    kmp_int32 gtid,
    kmp_int32 schedtype,
    kmp_int32 *plastiter,
    kmp_int32 *plower,
    kmp_int32 *pupper,
    kmp_int32 *pstride,
    kmp_int32 incr,
    kmp_int32 chunk
) {
    assert(schedtype == kmp_sch_static);
    team_t::parallel_for_thread_bounds(plastiter, plower, pupper, incr);
}

extern "C"
void
__kmpc_for_static_init_8u(
    ident_t * loc,
    kmp_int32 gtid,
    kmp_int32 schedtype,
    kmp_int32 * plastiter,
    kmp_uint64 * plower,
    kmp_uint64 * pupper,
    kmp_int64 * pstride,
    kmp_int64 incr,
    kmp_int64 chunk
) {
    assert(schedtype == kmp_sch_static);
    team_t::parallel_for_thread_bounds(plastiter, plower, pupper, incr);
}

extern "C"
void
__kmpc_for_static_init_4u(
    ident_t * loc,
    kmp_int32 gtid,
    kmp_int32 schedtype,
    kmp_int32 * plastiter,
    kmp_uint32 * plower,
    kmp_uint32 * pupper,
    kmp_int32 * pstride,
    kmp_int32 incr,
    kmp_int32 chunk)
{
    assert(schedtype == kmp_sch_static);
    team_t::parallel_for_thread_bounds(plastiter, plower, pupper, incr);
}
