/**
 * host_kmeans.c
 *
 * UPMEM K-Means code that:
 *   - loads data (points) from a text file
 *   - runs a CPU reference K-Means solution for comparison
 *   - rushes data to the DPU
 *   - iterates K-Means on the DPU (one iteration per dpu_launch)
 *   - reads partial sums from MRAM and updates centroids on the host
 */
#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <float.h>
#include <math.h>
#include <time.h>         
#include <dpu.h>
 
// Must match the DPU code
typedef double   feature_t;  // 8 bytes
typedef double   sum_t;      // 8 bytes
typedef uint64_t count_t;    // 8 bytes
 
// DPU arguments struct
typedef struct {
     uint32_t dpu_points;
     uint32_t nfeatures;
     uint32_t nclusters;
} dpu_arguments_t;
 
 /**
  * read_data_from_file
  * format:
  *   #points #features #clusters
  *   (p1_f1 p1_f2 ... p1_fN)
  *   ...
  */
//  static feature_t *read_data_from_file(const char *filename,
//                                        unsigned *p_points,
//                                        unsigned *p_features,
//                                        unsigned *p_clusters)
//  {
//      FILE *fp = fopen(filename, "r");
//      if (!fp) {
//          fprintf(stderr, "Could not open file '%s'\n", filename);
//          exit(1);
//      }
//      if (fscanf(fp, "%u %u %u", p_points, p_features, p_clusters) != 3) {
//          fprintf(stderr, "Error reading #points,#features,#clusters\n");
//          exit(1);
//      }
//      unsigned n_points   = *p_points;
//      unsigned n_features = *p_features;
 
//      feature_t *data = malloc(n_points * n_features * sizeof(feature_t));
//      if (!data) {
//          fprintf(stderr, "Memory alloc error for data\n");
//          exit(1);
//      }
 
//      for (unsigned i = 0; i < n_points; i++) {
//          for (unsigned f = 0; f < n_features; f++) {
//              int temp;
//              fscanf(fp, "%d", &temp);
//              data[i*n_features + f] = (feature_t) temp;
//          }
//      }
//      fclose(fp);
//      return data;
//  }

 /**
  * generate_data
  * format:
  *   #points #features #clusters
  *   (p1_f1 p1_f2 ... p1_fN)
  *   ...
  */
#define MAX_NUMBER 99
 static feature_t *generate_data(
    unsigned *p_points,
    unsigned *p_features)
{
unsigned n_points   = *p_points;
unsigned n_features = *p_features;

feature_t *data = malloc(n_points * n_features * sizeof(feature_t));
if (!data) {
fprintf(stderr, "Memory alloc error for data\n");
exit(1);
}

for (unsigned i = 0; i < n_points; i++) {
for (unsigned f = 0; f < n_features; f++) {
data[i * n_features + f] = (feature_t)(rand() % MAX_NUMBER);
}}
return data;

}
 
 /**
  * CPU reference K-Means for comparison
  */
 static void cpu_reference_kmeans(const feature_t *points,
                                  feature_t       *centroids,
                                  unsigned n_points,
                                  unsigned n_features,
                                  unsigned n_clusters,
                                  unsigned iters)
 {
     sum_t   *acc_sums   = calloc(n_clusters * n_features, sizeof(sum_t));
     count_t *acc_counts = calloc(n_clusters, sizeof(count_t));
     if (!acc_sums || !acc_counts) {
         fprintf(stderr, "Memory alloc error in cpu_reference_kmeans\n");
         exit(1);
     }
 
     for (unsigned iter = 0; iter < iters; iter++) {
         // reset accumulators
         memset(acc_sums,   0, n_clusters * n_features * sizeof(sum_t));
         memset(acc_counts, 0, n_clusters * sizeof(count_t));
 
         // assign each point to nearest cluster
         for (unsigned i = 0; i < n_points; i++) {
             double best_dist = DBL_MAX;
             int best_cl = -1;
             for (unsigned c = 0; c < n_clusters; c++) {
                 double dist_c = 0.0;
                 for (unsigned f = 0; f < n_features; f++) {
                     double diff = points[i*n_features + f]
                                 - centroids[c*n_features + f];
                     dist_c += diff*diff;
                 }
                 if (dist_c < best_dist) {
                     best_dist = dist_c;
                     best_cl = c;
                 }
             }
             if (best_cl < 0) {
                 fprintf(stderr, "Error: negative cluster assignment!\n");
                 exit(1);
             }
             acc_counts[best_cl]++;
             for (unsigned f = 0; f < n_features; f++) {
                 acc_sums[best_cl*n_features + f] += points[i*n_features + f];
             }
         }
 
         // update centroids
         for (unsigned c = 0; c < n_clusters; c++) {
             if (acc_counts[c] > 0) {
                 for (unsigned f = 0; f < n_features; f++) {
                     centroids[c*n_features + f] =
                       acc_sums[c*n_features + f] / (double)acc_counts[c];
                 }
             }
         }
     }
 
     free(acc_sums);
     free(acc_counts);
 }
 
 // simple frobenius norm to measure shift
 static double frob_norm(const feature_t *oldc,
                         const feature_t *newc,
                         unsigned n_clusters,
                         unsigned n_features)
 {
     double sum = 0.0;
     for (unsigned c = 0; c < n_clusters; c++) {
         for (unsigned f = 0; f < n_features; f++) {
             double diff = newc[c*n_features + f] - oldc[c*n_features + f];
             sum += diff * diff;
         }
     }
     return sqrt(sum);
 }
 
//  print centroids
 static void print_centroids(const char* label,
                             const feature_t *ctds,
                             unsigned n_clusters,
                             unsigned n_features)
 {
     printf("%s:\n", label);
     for (unsigned c = 0; c < n_clusters; c++) {
         printf(" cluster %u => (", c);
         for (unsigned f = 0; f < n_features; f++) {
             printf("%.2f", ctds[c*n_features + f]);
             if (f < n_features - 1) printf(", ");
         }
         printf(")\n");
     }
 }
 
 int main(int argc, char *argv[])
 {
    printf("Version 1\n");
    //default
    unsigned n_points   = 1024;
    unsigned n_features = 2;
    unsigned n_clusters = 5;

    if (argc == 4) {
        /* Parse command-line integers. */
        n_points   = (unsigned)atoi(argv[1]);
        n_features = (unsigned)atoi(argv[2]);
        n_clusters = (unsigned)atoi(argv[3]);
    }
    else if (argc != 1) {
        fprintf(stderr, "Usage: %s <n_points> <n_features> <n_clusters>\n", argv[0]);
        return 1;
    }
 
    //  feature_t *points = read_data_from_file(argv[1],
    //                                          &n_points, &n_features, &n_clusters);

    feature_t *points = generate_data(
        &n_points, &n_features);
    printf("Loaded dataset: %u points, %u features, %u clusters\n",
            n_points, n_features, n_clusters);
 
    //  CPU reference for 10 iters
     feature_t *cpu_ctds = malloc(n_clusters * n_features * sizeof(feature_t));
     // init centroids
    for (unsigned c = 0; c < n_clusters; c++) {
        for (unsigned f = 0; f < n_features; f++) {
            cpu_ctds[c*n_features + f] = 10.0 * c;
        }
    }
    // run cpu and get time
    struct timespec cpu_start, cpu_finish;
    clock_gettime(CLOCK_MONOTONIC, &cpu_start);
    cpu_reference_kmeans(points, cpu_ctds, n_points, n_features, n_clusters, 20);
    print_centroids("CPU final (10 iters)", cpu_ctds, n_clusters, n_features);
    clock_gettime(CLOCK_MONOTONIC, &cpu_finish);

    double cpu_elapsed = (cpu_finish.tv_sec - cpu_start.tv_sec) * 1e3 +
                      (cpu_finish.tv_nsec - cpu_start.tv_nsec) / 1e6;
 
    //  now, run the DPU-based K-means

     // set up dpu timers
     struct timespec dpu_set_up, dpu_set_up2;
     struct timespec dpu_start, dpu_finish;
     struct timespec dpu_calc1, dpu_calc2;
     struct timespec dpu_read1, dpu_read2;
    
     // variables to store overall times over loops
     double dpu_calc_time = 0;
     double dpu_read_time = 0;

     // start dpu set_up timer
     clock_gettime(CLOCK_MONOTONIC, &dpu_set_up);
 
     // allocate + load DPUs
     struct dpu_set_t dpus;
     DPU_ASSERT(dpu_alloc(NR_DPUS, NULL, &dpus));
     DPU_ASSERT(dpu_load(dpus, "./bin/kmeans_dpu", NULL));
 
     // retrieve # of DPUs
     uint32_t nr_of_dpus;
     DPU_ASSERT(dpu_get_nr_dpus(dpus, &nr_of_dpus));
     printf("Number of DPUs: %i\n", nr_of_dpus);
 
     // partition the points
     typedef struct {
         uint32_t num_points;
         uint32_t offset;
     } partition_t;
     partition_t *partitions = malloc(nr_of_dpus * sizeof(partition_t));
 
     uint32_t base_points = n_points / nr_of_dpus;
     uint32_t remainder   = n_points % nr_of_dpus;
     uint32_t current_offset = 0;
     for (uint32_t i = 0; i < nr_of_dpus; i++) {
         uint32_t count = base_points + (i < remainder ? 1 : 0);
         partitions[i].num_points = count;
         partitions[i].offset     = current_offset;
         current_offset += count;
     }
 
     // for each DPU, push only its subarray + set the arguments
     dpu_arguments_t *all_args = malloc(nr_of_dpus * sizeof(dpu_arguments_t));
     {
         struct dpu_set_t dpu;
         uint32_t i = 0;
         DPU_FOREACH(dpus, dpu, i) {
             uint32_t offset_in_points = partitions[i].offset;
             uint32_t count_points     = partitions[i].num_points;
 
             // prepare subarray
             feature_t *partition_start = points + offset_in_points * n_features;
             size_t partition_bytes = count_points * n_features * sizeof(feature_t);
 
             // fill argument struct
             all_args[i].dpu_points = count_points;
             all_args[i].nfeatures  = n_features;
             all_args[i].nclusters  = n_clusters;
 
             // transfer the subset of points to MRAM
             DPU_ASSERT(dpu_prepare_xfer(dpu, partition_start));
             DPU_ASSERT(dpu_push_xfer(dpu, DPU_XFER_TO_DPU,
                           "t_features", 0,
                           partition_bytes,
                           DPU_XFER_DEFAULT));
 
             // debug print
            //  printf("[Host] DPU %u => offset=%u, count=%u\n", i, offset_in_points, count_points);
         }
 
         // push the arguments
         i = 0;
         DPU_FOREACH(dpus, dpu, i) {
             DPU_ASSERT(dpu_prepare_xfer(dpu, &all_args[i]));
         }
         DPU_ASSERT(dpu_push_xfer(dpus, DPU_XFER_TO_DPU,
                   "DPU_INPUT_ARGUMENTS", 0,
                   sizeof(dpu_arguments_t),
                   DPU_XFER_DEFAULT));
     }
     clock_gettime(CLOCK_MONOTONIC, &dpu_set_up2);
     double dpu_set_up_elapsed = (dpu_set_up2.tv_sec - dpu_set_up.tv_sec) * 1e3 +
                      (dpu_set_up2.tv_nsec - dpu_set_up.tv_nsec) / 1e6;
 
     // prepare centroids on host side
     feature_t *dpu_ctds = malloc(n_clusters * n_features * sizeof(feature_t));
     // same init as CPU centroids
     for (unsigned c = 0; c < n_clusters; c++) {
         for (unsigned f = 0; f < n_features; f++) {
             dpu_ctds[c*n_features + f] = 10.0 * c;
         }
     }
 
     unsigned MAX_ITER = 20;
     double threshold  = 0.01;
     unsigned iter     = 0;
     double shift      = 99999.0;
 
     //start overall dpu timer
     clock_gettime(CLOCK_MONOTONIC, &dpu_start);
     
     while ((iter < MAX_ITER) && (shift > threshold)) {
         iter++;
 
         // save old centroids
         feature_t *old_ctds = malloc(n_clusters * n_features * sizeof(feature_t));
         memcpy(old_ctds, dpu_ctds, n_clusters * n_features * sizeof(feature_t));
 
         // push centroids to each DPU
         size_t cbytes = n_clusters * n_features * sizeof(feature_t);
         {
             struct dpu_set_t each;
             DPU_FOREACH(dpus, each) {
                 DPU_ASSERT(dpu_prepare_xfer(each, dpu_ctds));
             }
             DPU_ASSERT(dpu_push_xfer(dpus, DPU_XFER_TO_DPU,
                           "c_clusters", 0,
                           cbytes,
                           DPU_XFER_DEFAULT));
         }
 
         // launch DPUs
         clock_gettime(CLOCK_MONOTONIC, &dpu_calc1);
         DPU_ASSERT(dpu_launch(dpus, DPU_SYNCHRONOUS));
         clock_gettime(CLOCK_MONOTONIC, &dpu_calc2);
         dpu_calc_time += (dpu_calc2.tv_sec - dpu_calc1.tv_sec) * 1e3 +
                      (dpu_calc2.tv_nsec - dpu_calc1.tv_nsec) / 1e6;

 
         // allocate global accumulators
         sum_t   *acc_sums_global   = calloc(n_clusters * n_features, sizeof(sum_t));
         count_t *acc_counts_global = calloc(n_clusters, sizeof(count_t));
 
        // read partial sums and counts from each DPU and accumulate
        size_t sum_bytes   = n_clusters * n_features * sizeof(sum_t);
        size_t count_bytes = n_clusters * sizeof(count_t);

        struct dpu_set_t each;
        DPU_FOREACH(dpus, each) {
            sum_t   *acc_sums_local   = calloc(n_clusters * n_features, sizeof(sum_t));
            count_t *acc_counts_local = calloc(n_clusters, sizeof(count_t));
            
            //time incoming reads from dpu to host
            clock_gettime(CLOCK_MONOTONIC, &dpu_read1);
            DPU_ASSERT(dpu_prepare_xfer(each, acc_sums_local));
            DPU_ASSERT(dpu_push_xfer(each, DPU_XFER_FROM_DPU,
                        "centers_sum_mram", 0,
                        sum_bytes,
                        DPU_XFER_DEFAULT));

            DPU_ASSERT(dpu_prepare_xfer(each, acc_counts_local));
            DPU_ASSERT(dpu_push_xfer(each, DPU_XFER_FROM_DPU,
                        "centers_count_mram", 0,
                        count_bytes,
                        DPU_XFER_DEFAULT));

            clock_gettime(CLOCK_MONOTONIC, &dpu_read2);
            dpu_read_time += (dpu_read2.tv_sec - dpu_read1.tv_sec) * 1e3 +
                      (dpu_read2.tv_nsec - dpu_read1.tv_nsec) / 1e6;

            for (unsigned c = 0; c < n_clusters; c++) {
                acc_counts_global[c] += acc_counts_local[c];
                for (unsigned f = 0; f < n_features; f++) {
                    acc_sums_global[c*n_features + f] += acc_sums_local[c*n_features + f];
                }
            }

            free(acc_sums_local);
            free(acc_counts_local);
        }
    
 
         // update centroids on host using the aggregated results
         for (unsigned c = 0; c < n_clusters; c++) {
             if (acc_counts_global[c] > 0) {
                 for (unsigned f = 0; f < n_features; f++) {
                     double val = (double)acc_sums_global[c*n_features + f] / (double)acc_counts_global[c];
                     dpu_ctds[c*n_features + f] = val;
                 }
             }
         }
         free(acc_sums_global);
         free(acc_counts_global);
 
         shift = frob_norm(old_ctds, dpu_ctds, n_clusters, n_features);
        //  printf("Iteration %u: shift=%.2f\n", iter, shift);
 
         free(old_ctds);
     }
 
     // end timing
     clock_gettime(CLOCK_MONOTONIC, &dpu_finish);
     double dpu_elapsed = (dpu_finish.tv_sec - dpu_start.tv_sec) * 1e3 +
                    (dpu_finish.tv_nsec - dpu_start.tv_nsec) / 1e6;
     
 
     printf("DPU final after %u iteration(s):\n", iter);
     for (unsigned c = 0; c < n_clusters; c++) {
         printf(" cluster %u => (", c);
         for (unsigned f = 0; f < n_features; f++) {
             printf("%.2f", dpu_ctds[c*n_features + f]);
             if (f < n_features - 1) printf(", ");
         }
         printf(")\n");
     }

    //print all timings
    printf("CPU Implementation elapsed time: %.3f ms.\n", cpu_elapsed);
    printf("DPU Set Up Time: %.3f ms.\n", dpu_set_up_elapsed);
    printf("DPU Calculation time: %.3f ms.\n", dpu_calc_time);
    printf("Read from DPU to Host time: %.3f ms.\n", dpu_read_time);
    printf("DPU Implementation elapsed time (without set up): %.3f ms.\n", dpu_elapsed);

    // cleanup
    DPU_ASSERT(dpu_free(dpus));
    free(points);
    free(cpu_ctds);
    free(dpu_ctds);
    free(partitions);
    free(all_args);
    return 0;
 }