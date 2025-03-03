#ifndef COMMON_H
#define COMMON_H

#include <stdint.h>

/* define feature_t, sum_t, count_t differently for host vs DPU: */
#ifdef __UPMEM__
  /* DPU side => short for features, long for sums, etc. */
  typedef int16_t  feature_t;
  typedef int64_t  sum_t;
  typedef uint32_t count_t;
#else
  /* host side => double for features, sums, etc. */
  typedef double   feature_t;
  typedef double   sum_t;
  typedef unsigned int count_t;
#endif

// /* basic DPU defaults */
// #ifndef NR_DPUS
// #define NR_DPUS 8
// #endif
// #ifndef NR_TASKLETS
// #define NR_TASKLETS 16
// #endif

/* structure broadcast from host to DPU each iteration */
typedef struct {
    uint32_t dpu_points;
    uint32_t nfeatures;
    uint32_t nclusters;
    uint32_t max_points_per_dpu;
} dpu_arguments_t;

/* default to "t_features" in the DPU kernel. */
#ifndef DPU_MRAM_HEAP_POINTER_NAME
#define DPU_MRAM_HEAP_POINTER_NAME "t_features"
#endif

/* 8-byte alignment helper */
static inline size_t align8(size_t x) {
    return (x + 7UL) & ~7UL;
}

/* block size for the DPU DMA buffer */
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

#endif /* COMMON_H */