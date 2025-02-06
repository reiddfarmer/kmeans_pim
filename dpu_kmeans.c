/**
 * dpu_kmeans.c
 * A multi-DPU K-Means kernel with ONE TASKLET per DPU.
 *
 * Each DPU sees a subset of points in t_features[], plus global centroids in c_clusters[].
 * Then it accumulates partial sums + counts => centers_sum_mram, centers_count_mram.
 */

#include <defs.h>
#include <mram.h>
#include <barrier.h>
#include <alloc.h>
#include <mutex.h>
#include <string.h>
#include <stdint.h>

/** 
 * data types (8-byte each) to avoid alignment errors:
 * - 'dpu_feature_t' for feature values (double)
 * - 'dpu_sum_t' for partial sums (double)
 * - 'dpu_count_t' for counts (uint64_t)
 */
typedef double   dpu_feature_t;
typedef double   dpu_sum_t;
typedef uint64_t dpu_count_t;

// assuming single tasklet => no leftover logic within each DPU
#ifndef NR_TASKLETS
#define NR_TASKLETS 1
#endif

//"max" bounds for safety
#ifndef MAX_POINTS_DPU
#define MAX_POINTS_DPU 65536
#endif
#ifndef MAX_FEATURES
#define MAX_FEATURES 16
#endif
#ifndef MAX_CLUSTERS
#define MAX_CLUSTERS 16
#endif

// local arrays in MRAM/WRAM:

//each DPU gets a chunk of points in t_features[] (dpu_points*nfeatures)
__mram_noinit dpu_feature_t t_features[MAX_POINTS_DPU * MAX_FEATURES];

// centroids (host region, no offset math needed)
__host dpu_feature_t c_clusters[MAX_CLUSTERS * MAX_FEATURES];

//partial sums+counts in separate MRAM arrays
__mram_noinit dpu_sum_t   centers_sum_mram[MAX_CLUSTERS * MAX_FEATURES];
__mram_noinit dpu_count_t centers_count_mram[MAX_CLUSTERS];

// arguments from the host
typedef struct {
    uint32_t dpu_points;  // #points assigned to this DPU
    uint32_t nfeatures;   // #features
    uint32_t nclusters;   // #clusters
} dpu_arguments_t;

__host dpu_arguments_t DPU_INPUT_ARGUMENTS;

// accumulate partial sums in WRAM
static dpu_sum_t   local_sums [MAX_CLUSTERS * MAX_FEATURES];
static dpu_count_t local_counts[MAX_CLUSTERS];

int main(void)
{
    // single tasklet => no leftover logic
    uint32_t dpu_points = DPU_INPUT_ARGUMENTS.dpu_points;
    uint32_t nfeatures  = DPU_INPUT_ARGUMENTS.nfeatures;
    uint32_t nclusters  = DPU_INPUT_ARGUMENTS.nclusters;

    // init accumulators
    memset(local_sums,   0, sizeof(local_sums));
    memset(local_counts, 0, sizeof(local_counts));

    // for each point in [0, dpu_points), read from t_features, find nearest cluster, accumulate
    for (uint32_t i = 0; i < dpu_points; i++) {
        // read the point
        dpu_feature_t point[MAX_FEATURES];
        mram_read(&t_features[i * nfeatures],
                  point,
                  nfeatures * sizeof(dpu_feature_t));

        // find nearest centroid
        double best_dist = 1e30;
        int best_cl = -1;
        for (uint32_t c = 0; c < nclusters; c++) {
            double dist_c = 0.0;
            for (uint32_t f = 0; f < nfeatures; f++) {
                double diff = point[f] - c_clusters[c*nfeatures + f];
                dist_c += diff*diff;
            }
            if (dist_c < best_dist) {
                best_dist = dist_c;
                best_cl   = c;
            }
        }

        // accumulate
        local_counts[best_cl]++;
        for (uint32_t f = 0; f < nfeatures; f++) {
            local_sums[best_cl*nfeatures + f] += point[f];
        }
    }

    // write partial sums + counts to MRAM
    // sums: nclusters*nfeatures (each 8 bytes)
    uint32_t sums_bytes = nclusters*nfeatures*sizeof(dpu_sum_t);
    mram_write(local_sums, centers_sum_mram, sums_bytes);

    // counts: nclusters (each 8 bytes)
    uint32_t counts_bytes = nclusters*sizeof(dpu_count_t);
    mram_write(local_counts, centers_count_mram, counts_bytes);

    return 0;
}