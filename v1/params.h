#ifndef _PARAMS_H_
#define _PARAMS_H_

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <assert.h>
#include <getopt.h>

typedef struct {
    unsigned int n_points;
    unsigned int n_features;
    unsigned int n_clusters;
    unsigned int n_warmup;
    unsigned int n_reps;
} Params;

static void usage_kmeans() {
    fprintf(stderr,
        "\nUsage:  ./kmeans_host [options] [data_file]"
        "\n"
        "\nGeneral options:"
        "\n    -h            help"
        "\n    -p <NPOINTS>  number of points (default=8)"
        "\n    -f <NFEAT>    number of features (default=2)"
        "\n    -c <NCLUST>   number of clusters (default=2)"
        "\n    -w <W>        # of warmup iters (default=1)"
        "\n    -r <R>        # of rep iters (default=2)"
        "\n\nIf [data_file] is provided, we read from it instead of using the above parameters."
        "\n");
}

static Params input_params_kmeans(int argc, char** argv, char** data_filename) {
    Params p;
    p.n_points   = 8;
    p.n_features = 2;
    p.n_clusters = 2;
    p.n_warmup   = 1;
    p.n_reps     = 2;

    int opt;
    while ((opt = getopt(argc, argv, "hp:f:c:w:r:")) >= 0) {
        switch(opt) {
            case 'h':
                usage_kmeans();
                exit(0);
            case 'p': p.n_points   = (unsigned int)atoi(optarg); break;
            case 'f': p.n_features = (unsigned int)atoi(optarg); break;
            case 'c': p.n_clusters = (unsigned int)atoi(optarg); break;
            case 'w': p.n_warmup   = (unsigned int)atoi(optarg); break;
            case 'r': p.n_reps     = (unsigned int)atoi(optarg); break;
            default:
                fprintf(stderr,"\nUnrecognized option!\n");
                usage_kmeans();
                exit(1);
        }
    }
    if (optind < argc) {
        *data_filename = argv[optind];
    } else {
        *data_filename = NULL;
    }
    return p;
}

#endif /* _PARAMS_H_ */