/* host_kmeans.c — INT16 */
#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <float.h>
#include <math.h>
#include <time.h>
#include <limits.h>
#include <dpu.h>

/*  data types */
typedef double    feature_t;  
typedef uint64_t  count_t;     /* cluster sizes                               */

typedef int16_t   q_feature_t; /* 16-bit feature sent to DPU & used by CPU ref*/
typedef int32_t   q_sum_t;     /* 32-bit running sums (safe for our ranges)   */

/* argument struct */
typedef struct {
    uint32_t dpu_points;
    uint32_t nfeatures;
    uint32_t nclusters;
} dpu_arguments_t;

/* constants */
#define MAX_NUMBER 99          /* random data range 0…98 */

/* helpers */
static feature_t *
gen_fp_data(unsigned *p_pts, unsigned *p_dim)
{
    unsigned N=*p_pts, D=*p_dim;
    feature_t *a = malloc((size_t)N*D*sizeof(*a));
    if(!a){perror("malloc");exit(1);}
    for(unsigned i=0;i<N*D;++i) a[i]=(rand()%MAX_NUMBER);
    return a;
}

/* INT-CPU reference (mirrors DPU) */
static unsigned
kmeans_int16(const q_feature_t *pts,
             q_feature_t       *c,
             unsigned N, unsigned D, unsigned K,
             double   thr,                /* convergence threshold   */
             unsigned max_iter)           /* hard upper bound        */
{
    q_sum_t *sum = calloc((size_t)K*D, sizeof *sum);
    count_t *cnt = calloc(K, sizeof *cnt);
    if (!sum || !cnt) { perror("calloc"); exit(1); }

    q_feature_t *prev = malloc((size_t)K*D * sizeof *prev);

    unsigned it = 0;
    double shift = thr + 1.0;

    while (it < max_iter && shift > thr) {
        memcpy(prev, c, (size_t)K*D*sizeof *prev);
        memset(sum, 0, (size_t)K*D*sizeof *sum);
        memset(cnt, 0, K*sizeof *cnt);

        // asign
        for (unsigned i = 0; i < N; ++i) {
            int64_t best = INT64_MAX; unsigned bestk = 0;
            for (unsigned k = 0; k < K; ++k) {
                int64_t dsq = 0;
                for (unsigned f = 0; f < D; ++f) {
                    int32_t d = (int32_t)pts[i*D+f] - c[k*D+f];
                    dsq += (int64_t)d * d;
                }
                if (dsq < best) { best = dsq; bestk = k; }
            }
            cnt[bestk]++;
            for (unsigned f = 0; f < D; ++f)
                sum[bestk*D+f] += pts[i*D+f];
        }

        // update
        for (unsigned k = 0; k < K; ++k)
            if (cnt[k])
                for (unsigned f = 0; f < D; ++f)
                    c[k*D+f] = (q_feature_t)(sum[k*D+f] / (int32_t)cnt[k]);

        // shift
        shift = 0.0;
        for (unsigned i = 0; i < K*D; ++i) {
            double diff = (double)c[i] - (double)prev[i];
            shift += diff*diff;
        }
        shift = sqrt(shift);
        ++it;
    }

    free(prev); free(sum); free(cnt);
    return it;               /* <= max_iter */
}

// print
static void print_centroids(const char *lbl,
                            const q_feature_t *c_i16,
                            unsigned K,unsigned D)
{
    printf("%s\n",lbl);
    for(unsigned k=0;k<K;++k){
        printf(" cluster %u ⇒ (",k);
        for(unsigned f=0;f<D;++f){
            printf("%d%s",c_i16[k*D+f],f==D-1?")\n":", ");
        }
    }
}

/* =================================================================== */
int main(int argc,char **argv)
{
    unsigned N=1024,D=2,K=5;
    if(argc==4){ N=atoi(argv[1]); D=atoi(argv[2]); K=atoi(argv[3]); }
    else if(argc!=1){
        fprintf(stderr,"Usage: %s <points> <features> <clusters>\n",argv[0]);
        return 1;
    }
    srand((unsigned)time(NULL));

    /* generate & quantise data */
    feature_t  *pts_fp = gen_fp_data(&N,&D);
    q_feature_t *pts_q = malloc((size_t)N*D*sizeof *pts_q);
    for(unsigned i=0;i<N*D;++i) pts_q[i]=(q_feature_t)lrintf(pts_fp[i]);

    printf("Loaded dataset: %u points, %u features, %u clusters\n",N,D,K);

    /* common centroid seed (integers 0-99) */
    q_feature_t *cent_cpu = malloc((size_t)K*D*sizeof *cent_cpu);
    q_feature_t *cent_dpu = malloc((size_t)K*D*sizeof *cent_dpu);

    for(unsigned i=0;i<K*D;++i)
        cent_cpu[i]=cent_dpu[i]=(rand()%MAX_NUMBER);

    /* ---------------- CPU INT16 reference ---------------- */
    struct timespec t0,t1;
    const double THR = 0.0001;
    const unsigned CPU_MAX = 300;

    clock_gettime(CLOCK_MONOTONIC, &t0);
    unsigned cpu_iters = kmeans_int16(pts_q, cent_cpu, N, D, K, THR, CPU_MAX);
    clock_gettime(CLOCK_MONOTONIC, &t1);

    double cpu_ms = (t1.tv_sec - t0.tv_sec)*1e3 +
                    (t1.tv_nsec - t0.tv_nsec)/1e6;
    char label[64];
    snprintf(label, sizeof label, "CPU-INT16 final (%u iters)", cpu_iters);
    print_centroids(label, cent_cpu, K, D);

    /* ---------------- DPU set-up ---------------- */
    struct timespec s0,s1;
    clock_gettime(CLOCK_MONOTONIC,&s0);

    struct dpu_set_t dpus;
    DPU_ASSERT(dpu_alloc(NR_DPUS,NULL,&dpus));
    DPU_ASSERT(dpu_load(dpus,"./bin/kmeans_dpu",NULL));

    uint32_t NR; DPU_ASSERT(dpu_get_nr_dpus(dpus,&NR));
    printf("Number of DPUs: %u\n",NR);

    typedef struct{uint32_t n,off;} part_t;
    part_t *part=malloc(NR*sizeof *part);

    uint32_t base=N/NR,rem=N%NR,off=0;
    for(uint32_t i=0;i<NR;++i){
        uint32_t n=base+(i<rem);
        part[i]=(part_t){n,off}; off+=n;
    }

    dpu_arguments_t *arg=malloc(NR*sizeof *arg);

    struct dpu_set_t d; uint32_t idx=0;
    DPU_FOREACH(dpus,d,idx){
        arg[idx]=(dpu_arguments_t){part[idx].n,D,K};
        q_feature_t *sub=&pts_q[part[idx].off*D];
        size_t bytes=(size_t)part[idx].n*D*sizeof(q_feature_t);
        DPU_ASSERT(dpu_prepare_xfer(d,sub));
        DPU_ASSERT(dpu_push_xfer(d,DPU_XFER_TO_DPU,
                   "t_features",0,bytes,DPU_XFER_DEFAULT));
    }
    /* push args */
    idx=0; DPU_FOREACH(dpus,d,idx){DPU_ASSERT(dpu_prepare_xfer(d,&arg[idx]));}
    DPU_ASSERT(dpu_push_xfer(dpus,DPU_XFER_TO_DPU,
               "DPU_INPUT_ARGUMENTS",0,sizeof(dpu_arguments_t),
               DPU_XFER_DEFAULT));

    clock_gettime(CLOCK_MONOTONIC,&s1);
    double setup_ms=(s1.tv_sec-s0.tv_sec)*1e3+(s1.tv_nsec-s0.tv_nsec)/1e6;

    /* ---------------- iterative DPU K-means ---------------- */
    const unsigned MAX_IT = cpu_iters;
    double comp_ms=0, read_ms=0;

    struct timespec run0,run1; clock_gettime(CLOCK_MONOTONIC,&run0);
    unsigned it=0;

    q_feature_t *prev = malloc((size_t)K*D*sizeof *prev);

    while(it<MAX_IT){
        memcpy(prev,cent_dpu,(size_t)K*D*sizeof *prev);

        /* ship current centroids */
        size_t cbytes=(size_t)K*D*sizeof(q_feature_t);
        idx=0; DPU_FOREACH(dpus,d,idx){DPU_ASSERT(dpu_prepare_xfer(d,cent_dpu));}
        DPU_ASSERT(dpu_push_xfer(dpus,DPU_XFER_TO_DPU,
                   "c_clusters",0,cbytes,DPU_XFER_DEFAULT));

        /* launch */
        struct timespec l0,l1; clock_gettime(CLOCK_MONOTONIC,&l0);
        DPU_ASSERT(dpu_launch(dpus,DPU_SYNCHRONOUS));
        clock_gettime(CLOCK_MONOTONIC,&l1);
        comp_ms+=(l1.tv_sec-l0.tv_sec)*1e3+(l1.tv_nsec-l0.tv_nsec)/1e6;

        /* gather partial sums */
        q_sum_t *gs = calloc((size_t)K*D,sizeof *gs);
        count_t *gc = calloc(K,sizeof *gc);

        size_t sb=(size_t)K*D*sizeof(q_sum_t);
        size_t cb=K*sizeof(count_t);

        DPU_FOREACH(dpus,d,idx){
            q_sum_t *ls = calloc((size_t)K*D,sizeof *ls);
            count_t *lc = calloc(K,sizeof *lc);

            struct timespec r0,r1; clock_gettime(CLOCK_MONOTONIC,&r0);
            DPU_ASSERT(dpu_prepare_xfer(d,ls));
            DPU_ASSERT(dpu_push_xfer(d,DPU_XFER_FROM_DPU,
                       "centers_sum_mram",0,sb,DPU_XFER_DEFAULT));
            DPU_ASSERT(dpu_prepare_xfer(d,lc));
            DPU_ASSERT(dpu_push_xfer(d,DPU_XFER_FROM_DPU,
                       "centers_count_mram",0,cb,DPU_XFER_DEFAULT));
            clock_gettime(CLOCK_MONOTONIC,&r1);
            read_ms+=(r1.tv_sec-r0.tv_sec)*1e3+(r1.tv_nsec-r0.tv_nsec)/1e6;

            for(unsigned k=0;k<K;++k){
                gc[k]+=lc[k];
                for(unsigned f=0;f<D;++f)
                    gs[k*D+f]+=ls[k*D+f];
            }
            free(ls); free(lc);
        }

        /* host update — **pure integer mean** */
        for(unsigned k=0;k<K;++k)
            if(gc[k])
                for(unsigned f=0;f<D;++f)
                    cent_dpu[k*D+f]=(q_feature_t)(gs[k*D+f]/(int32_t)gc[k]);
        
        free(gs); 
        free(gc);
        it++; 
    }
    clock_gettime(CLOCK_MONOTONIC,&run1);
    double total_ms=(run1.tv_sec-run0.tv_sec)*1e3+(run1.tv_nsec-run0.tv_nsec)/1e6;

    /* ---------------- report ---------------- */
    print_centroids("\nDPU final",cent_dpu,K,D);
    printf("\nTiming (ms):  CPU %6.2f | DPU setup %6.2f  compute %6.2f  read %6.2f  total %6.2f\n",
           cpu_ms,setup_ms,comp_ms,read_ms,total_ms);

    /* ---------------- cleanup -------------- */
    DPU_ASSERT(dpu_free(dpus));
    free(pts_fp); free(pts_q);
    free(cent_cpu); free(cent_dpu); free(prev);
    free(part); free(arg);
    return 0;
}
