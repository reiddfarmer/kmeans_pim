#include <defs.h>
#include <mram.h>
#include <barrier.h>
#include <alloc.h>
#include <mutex.h>
#include <string.h>
#include <stdint.h>

// data types (8 bytes each)
typedef double   dpu_feature_t;
typedef double   dpu_sum_t;
typedef uint64_t dpu_count_t;

// allowed maximum
#ifndef NR_TASKLETS
#define NR_TASKLETS 8  // e.g. 8
#endif

#ifndef MAX_POINTS_DPU
#define MAX_POINTS_DPU 65536
#endif
#ifndef MAX_FEATURES
#define MAX_FEATURES 16
#endif
#ifndef MAX_CLUSTERS
#define MAX_CLUSTERS 16
#endif

// MRAM arrays
__mram_noinit dpu_feature_t t_features[MAX_POINTS_DPU * MAX_FEATURES];
__host       dpu_feature_t c_clusters[MAX_CLUSTERS   * MAX_FEATURES];
__mram_noinit dpu_sum_t     centers_sum_mram[MAX_CLUSTERS  * MAX_FEATURES];
__mram_noinit dpu_count_t   centers_count_mram[MAX_CLUSTERS];

// arguments from the host
typedef struct {
    uint32_t dpu_points;  // #points on this DPU
    uint32_t nfeatures;
    uint32_t nclusters;
} dpu_arguments_t;

__host dpu_arguments_t DPU_INPUT_ARGUMENTS;

// ==================== LOCAL BUFFERS ======================
__dma_aligned dpu_sum_t   tasklet_sums[NR_TASKLETS][MAX_CLUSTERS * MAX_FEATURES];
__dma_aligned dpu_count_t tasklet_counts[NR_TASKLETS][MAX_CLUSTERS];

// barrier for synchronization
BARRIER_INIT(my_barrier, NR_TASKLETS);

/**
 * Each tasklet processes a sub-range of the DPU's local points.
 */
int main() {
    // read arguments
    uint32_t dpu_points = DPU_INPUT_ARGUMENTS.dpu_points;
    uint32_t nfeatures  = DPU_INPUT_ARGUMENTS.nfeatures;
    uint32_t nclusters  = DPU_INPUT_ARGUMENTS.nclusters;

    // get the ID of this tasklet (0..NR_TASKLETS-1)
    uint32_t t_id = me();

    // init this tasklet's accumulators
    memset(tasklet_sums[t_id],   0, sizeof(dpu_sum_t   ) * nclusters * nfeatures);
    memset(tasklet_counts[t_id], 0, sizeof(dpu_count_t ) * nclusters);

    // figure out sub-range of points for THIS tasklet
    // e.g. block distribution
    uint32_t points_per_tasklet = dpu_points / NR_TASKLETS;
    uint32_t remainder = dpu_points % NR_TASKLETS;
    // a typical approach: let first 'remainder' tasklets each handle +1 point
    uint32_t start, end;
    if (t_id < remainder) {
        start = t_id * (points_per_tasklet + 1);
        end   = start + (points_per_tasklet + 1);
    } else {
        start = remainder * (points_per_tasklet + 1)
                + (t_id - remainder) * points_per_tasklet;
        end   = start + points_per_tasklet;
    }
    if (end > dpu_points) {
        end = dpu_points; // safety
    }

    // process each point in [start, end)
    for (uint32_t i = start; i < end; i++) {
        // read the point from MRAM
        dpu_feature_t point[MAX_FEATURES];
        mram_read(&t_features[i * nfeatures],
                  point,
                  nfeatures * sizeof(dpu_feature_t));

        // find nearest centroid
        double best_dist = 1e30;
        int best_cl = -1;
        for (uint32_t c = 0; c < nclusters; c++) {
            double dist_c = 0.0;
            // read the centroid from the c_clusters[] (host region)
            // in practice, c_clusters is loaded into WRAM automatically
            // by your push_xfer, so you can access it directly as array
            for (uint32_t f = 0; f < nfeatures; f++) {
                double diff = point[f] - c_clusters[c*nfeatures + f];
                dist_c += diff*diff;
            }
            if (dist_c < best_dist) {
                best_dist = dist_c;
                best_cl   = c;
            }
        }
        // accumulate into this tasklet's partial sums
        tasklet_counts[t_id][best_cl]++;
        for (uint32_t f = 0; f < nfeatures; f++) {
            tasklet_sums[t_id][best_cl*nfeatures + f] += point[f];
        }
    }

    // barrier: wait for all tasklets to finish partial sums
    barrier_wait(&my_barrier);

    // reduce everything into tasklet 0
    if (t_id == 0) {
        // we can reuse, for instance, tasklet_sums[0] as the aggregator
        for (uint32_t tid = 1; tid < NR_TASKLETS; tid++) {
            for (uint32_t c = 0; c < nclusters; c++) {
                // add counts
                tasklet_counts[0][c] += tasklet_counts[tid][c];
                // add sums
                for (uint32_t f = 0; f < nfeatures; f++) {
                    tasklet_sums[0][c*nfeatures + f] +=
                        tasklet_sums[tid][c*nfeatures + f];
                }
            }
        }

        // now write these aggregated results to MRAM
        // sums
        uint32_t sums_bytes   = nclusters * nfeatures * sizeof(dpu_sum_t);
        mram_write(&tasklet_sums[0][0], centers_sum_mram, sums_bytes);

        // counts
        uint32_t counts_bytes = nclusters * sizeof(dpu_count_t);
        mram_write(&tasklet_counts[0][0], centers_count_mram, counts_bytes);
    }

    return 0;
}