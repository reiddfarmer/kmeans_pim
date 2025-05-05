/**
 * dpu_kmeans.c
 *
 *
 * Each tasklet processes its assigned points in bursts of up to 2048 bytes per DMA transfer
 * With NR_TASKLETS tasklets, each gets its own 2 KB buffer, for a total of 2KB*NR_TASKLETS extra WRAM. (CURRENT MAXIMUM OF 8 TASKLETS)
 */

 #include <defs.h>
 #include <mram.h>
 #include <barrier.h>
 #include <alloc.h>
 #include <string.h>
 #include <stdint.h>
 
 // data types
 typedef double   dpu_feature_t;    /* each feature is 8 bytes */
 typedef double   dpu_sum_t;
 typedef uint64_t dpu_count_t;
 
// limits
 #ifndef NR_TASKLETS
 #define NR_TASKLETS 8
 #endif
 
 #ifndef MAX_POINTS_DPU
 #define MAX_POINTS_DPU 65536
 #endif
 
 #ifndef MAX_FEATURES
 #define MAX_FEATURES 16
 #endif
 
 #ifndef MAX_CLUSTERS
 #define MAX_CLUSTERS 20
 #endif

 #define MIN(a, b) (( (a) < (b) ) ? (a) : (b))
 
// mram arrays
 __mram_noinit dpu_feature_t t_features[MAX_POINTS_DPU * MAX_FEATURES];
 __host       dpu_feature_t c_clusters[MAX_CLUSTERS   * MAX_FEATURES];
 __mram_noinit dpu_sum_t     centers_sum_mram[MAX_CLUSTERS * MAX_FEATURES];
 __mram_noinit dpu_count_t   centers_count_mram[MAX_CLUSTERS];
 
 //host arguments
 typedef struct {
     uint32_t dpu_points;  // # points stored in t_features
     uint32_t nfeatures;   // dimension of each point
     uint32_t nclusters;   // k in k-means
 } dpu_arguments_t;
 __host dpu_arguments_t DPU_INPUT_ARGUMENTS;
 
// wram arrays
 __dma_aligned dpu_sum_t   tasklet_sums[NR_TASKLETS][MAX_CLUSTERS * MAX_FEATURES];
 __dma_aligned dpu_count_t tasklet_counts[NR_TASKLETS][MAX_CLUSTERS];
 
// each tasklet gets its own scratch buffer in WRAM for DMA (16 * 16 features = 256 dpu_feature_t (double) * 8 bytes = 2048 bytes)
 __dma_aligned dpu_feature_t tasklet_buf[NR_TASKLETS][16 * MAX_FEATURES];
 
 BARRIER_INIT(my_barrier, NR_TASKLETS);
 
 #define MIN(a, b) (( (a) < (b) ) ? (a) : (b))
 
 int main(void)
 {
     // read host args
     const uint32_t dpu_points = DPU_INPUT_ARGUMENTS.dpu_points;
     const uint32_t nfeatures  = DPU_INPUT_ARGUMENTS.nfeatures;
     const uint32_t nclusters  = DPU_INPUT_ARGUMENTS.nclusters;
     const uint32_t t_id       = me();  
 
     // clear accumulators
     memset(tasklet_sums[t_id],   0, nclusters * nfeatures * sizeof(dpu_sum_t));
     memset(tasklet_counts[t_id], 0, nclusters * sizeof(dpu_count_t));
 
     //compute sub-ranges for this tasklet
     const uint32_t pts_per_tasklet = dpu_points / NR_TASKLETS;
     const uint32_t rem             = dpu_points % NR_TASKLETS;
     const uint32_t start = t_id * pts_per_tasklet + MIN(t_id, rem);
     const uint32_t end   = start + pts_per_tasklet + (t_id < rem ? 1 : 0);
 
     // dma constants
     const uint32_t point_bytes      = nfeatures * sizeof(dpu_feature_t);
     const uint32_t max_pts_per_read = 2048 / point_bytes;  
 
     // each tasklet gets its own chunk
     uint32_t idx = start;
     while (idx < end) {
         // # of points in this batch
         uint32_t batch = MIN(max_pts_per_read, end - idx);
 
         // ach tasklet uses its own scratch area from tasklet_buf
         mram_read(&t_features[idx * nfeatures],
                   tasklet_buf[t_id],
                   batch * point_bytes);
 
        // process each point from my private buffer 
         for (uint32_t p = 0; p < batch; p++) {
             dpu_feature_t *point = &tasklet_buf[t_id][p * nfeatures];
 
             // find the nearest centroid 
             double best_dist = 1e30;
             uint32_t best_cl = 0;
             for (uint32_t c = 0; c < nclusters; c++) {
                 const dpu_feature_t *cent = &c_clusters[c * nfeatures];
                 double dist = 0.0;
                 for (uint32_t f = 0; f < nfeatures; f++) {
                     double diff = point[f] - cent[f];
                     dist += diff * diff;
                 }
                 if (dist < best_dist) {
                     best_dist = dist;
                     best_cl = c;
                 }
             }
 
             //accumulate partial sums and counts 
             tasklet_counts[t_id][best_cl]++;
             dpu_sum_t *sum_vec = &tasklet_sums[t_id][best_cl * nfeatures];
             for (uint32_t f = 0; f < nfeatures; f++) {
                 sum_vec[f] += point[f];
             }
         }
 
         idx += batch;
     }
 
     // intra-DPU reduction
     barrier_wait(&my_barrier);
     if (t_id == 0) {
         for (uint32_t tid = 1; tid < NR_TASKLETS; tid++) {
             for (uint32_t c = 0; c < nclusters; c++) {
                 tasklet_counts[0][c] += tasklet_counts[tid][c];
                 dpu_sum_t *dst = &tasklet_sums[0][c * nfeatures];
                 dpu_sum_t *src = &tasklet_sums[tid][c * nfeatures];
                 for (uint32_t f = 0; f < nfeatures; f++) {
                     dst[f] += src[f];
                 }
             }
         }
 
         // write aggregated results to MRAM
         mram_write(tasklet_sums[0], centers_sum_mram, nclusters * nfeatures * sizeof(dpu_sum_t));
         mram_write(tasklet_counts[0], centers_count_mram, nclusters * sizeof(dpu_count_t));
     }
 
     return 0;
 }