#include <stdio.h>
#include <cstdlib>
#include <getopt.h>

void print_cuda_info();
void gen_matrix(unsigned int size);

int main(int argc, char** argv) {
    // print_cuda_info();
    // parse commandline options ////////////////////////////////////////////
    int opt;
    static struct option long_options[] = {
        {"size",  1, 0, 's'},
        {"help", 0, 0, '?'},
        {0 ,0, 0, 0}
    };

    unsigned int size = 512;

    while ((opt = getopt_long(argc, argv, "s:?", long_options, NULL)) != EOF) {
        switch (opt) {
        case 's':
            size = (unsigned int)(atoi(optarg));
            break;
        case '?':
        default:
            break;
        }
    }

    printf("size=%u\n", size);

    gen_matrix(size);

    // gen matrix with (size*size) and vec (1*size) with random fp16

}