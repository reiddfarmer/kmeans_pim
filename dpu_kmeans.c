/**
 * dpu_kmeans.c  –  INT16 version with batching
 */
#include <defs.h>
#include <mram.h>
#include <barrier.h>
#include <alloc.h>
#include <string.h>
#include <stdint.h>

/* data types */
typedef int16_t   dpu_feature_t;
typedef int32_t   dpu_sum_t;
typedef uint64_t  dpu_count_t;

/* limits */
#ifndef NR_TASKLETS
# define NR_TASKLETS 8
#endif
#ifndef MAX_POINTS_DPU
# define MAX_POINTS_DPU 65536
#endif
#ifndef MAX_FEATURES
# define MAX_FEATURES 16
#endif
#ifndef MAX_CLUSTERS
# define MAX_CLUSTERS 20
#endif
#define MIN(a,b) ((a)<(b)?(a):(b))

/* MRAM symbols */
__mram_noinit dpu_feature_t t_features[MAX_POINTS_DPU * MAX_FEATURES];
__host        dpu_feature_t c_clusters[MAX_CLUSTERS   * MAX_FEATURES];
__mram_noinit dpu_sum_t     centers_sum_mram [MAX_CLUSTERS * MAX_FEATURES];
__mram_noinit dpu_count_t   centers_count_mram[MAX_CLUSTERS];

/* host arguments */
typedef struct { uint32_t dpu_points,nfeatures,nclusters; } dpu_arguments_t;
__host dpu_arguments_t DPU_INPUT_ARGUMENTS;

/* WRAM scratch */
#define DMA_BYTES      2048
#define BUF_ELEMS      (DMA_BYTES / sizeof(dpu_feature_t))          /* 1024 */
#define BUF_POINTS_MAX (DMA_BYTES / (MAX_FEATURES * sizeof(dpu_feature_t)))/*128*/

__dma_aligned dpu_sum_t   task_sum [NR_TASKLETS][MAX_CLUSTERS * MAX_FEATURES];
__dma_aligned dpu_count_t task_cnt [NR_TASKLETS][MAX_CLUSTERS];
__dma_aligned dpu_feature_t buf[NR_TASKLETS][BUF_ELEMS];

BARRIER_INIT(bar,NR_TASKLETS);

int main(void)
{
    const uint32_t P  = DPU_INPUT_ARGUMENTS.dpu_points;
    const uint32_t D  = DPU_INPUT_ARGUMENTS.nfeatures;
    const uint32_t K  = DPU_INPUT_ARGUMENTS.nclusters;
    const uint32_t tid = me();

    memset(task_sum[tid],0,K*D*sizeof(dpu_sum_t));
    memset(task_cnt[tid],0,K  *sizeof(dpu_count_t));

    /* my slice of points */
    const uint32_t per = P/NR_TASKLETS, rem = P%NR_TASKLETS;
    const uint32_t start = tid*per + MIN(tid,rem);
    const uint32_t end   = start + per + (tid<rem);

    const uint32_t bytes_pt = D*sizeof(dpu_feature_t);
    const uint32_t max_pts_dma = DMA_BYTES / bytes_pt;   /* ≤ BUF_POINTS_MAX */

    for(uint32_t idx = start; idx < end; ){
        uint32_t batch = MIN(max_pts_dma, end-idx);
        mram_read(&t_features[idx*D], buf[tid], batch*bytes_pt);

        for(uint32_t p=0;p<batch;++p){
            dpu_feature_t *pt=&buf[tid][p*D];
            int64_t best=INT64_MAX; uint32_t bestk=0;

            for(uint32_t k=0;k<K;++k){
                const dpu_feature_t *cent=&c_clusters[k*D];
                int64_t dsq=0;
                for(uint32_t f=0;f<D;++f){
                    int32_t diff=(int32_t)pt[f] - cent[f];
                    dsq += (int64_t)diff*diff;
                }
                if(dsq<best){best=dsq; bestk=k;}
            }

            task_cnt[tid][bestk]++;
            dpu_sum_t *sv=&task_sum[tid][bestk*D];
            for(uint32_t f=0;f<D;++f) sv[f]+=pt[f];
        }
        idx += batch;
    }

    /* reduction */
    barrier_wait(&bar);
    if(tid==0){
        for(uint32_t t=1;t<NR_TASKLETS;++t){
            for(uint32_t k=0;k<K;++k){
                task_cnt[0][k] += task_cnt[t][k];
                dpu_sum_t *dst=&task_sum[0][k*D];
                dpu_sum_t *src=&task_sum[t][k*D];
                for(uint32_t f=0;f<D;++f) dst[f]+=src[f];
            }
        }
        mram_write(task_sum[0],  centers_sum_mram,
                   K*D*sizeof(dpu_sum_t));
        mram_write(task_cnt[0], centers_count_mram,
                   K*sizeof(dpu_count_t));
    }
    return 0;
}