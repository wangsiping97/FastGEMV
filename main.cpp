#include <getopt.h>
#include <stdio.h>

#include <cstdlib>

void test_gemv_with_params(unsigned int size, unsigned int iter,
                           unsigned int block_dim_x, unsigned int block_dim_y);

void test_gemv_int8_quantized_with_params(unsigned int size, unsigned int iter,
                                          unsigned int block_dim_x,
                                          unsigned int block_dim_y, float scale,
                                          float zero_point);

void test_gemv_int4_quantized_with_params(unsigned int size, unsigned int iter,
                                          unsigned int block_dim_x,
                                          unsigned int block_dim_y, float scale,
                                          float zero_point);

int main(int argc, char** argv) {
  // parse commandline options
  int opt;
  static struct option long_options[] = {
      {"size", required_argument, 0, 's'},
      {"iter", required_argument, 0, 'i'},
      {"block_x", required_argument, 0, 'x'},
      {"block_y", required_argument, 0, 'y'},
      {"bits", required_argument, 0, 'b'},
      {"scale", required_argument, 0, 'u'},
      {"zero_point", required_argument, 0, 'v'},
      {0, 0, 0, 0}};

  unsigned int size = 512;
  unsigned int iter = 1;
  unsigned int block_dim_x = 32;
  unsigned int block_dim_y = 4;
  unsigned int bits = 16;
  float scale = 0.0625;
  float zero_point = 0.01;

  while ((opt = getopt_long(argc, argv, "s:i:k:x:y:g:b:u:v:", long_options,
                            NULL)) != -1) {
    switch (opt) {
      case 's':
        if (optarg != NULL)
          size = (unsigned int)(atoi(optarg));
        else
          printf("size option requires an argument\n");
        break;
      case 'i':
        if (optarg != NULL)
          iter = (unsigned int)(atoi(optarg));
        else
          printf("iter option requires an argument\n");
        break;
      case 'x':
        if (optarg != NULL)
          block_dim_x = (unsigned int)atoi(optarg);
        else
          printf("block_x option requires an argument\n");
        break;
      case 'y':
        if (optarg != NULL)
          block_dim_y = (unsigned int)atoi(optarg);
        else
          printf("block_y option requires an argument\n");
        break;
      case 'b':
        if (optarg != NULL)
          bits = (unsigned)atoi(optarg);
        else
          printf("bits option requires an argument\n");
        break;
      case 'u':
        if (optarg != NULL)
          scale = atof(optarg);
        else
          printf("scale option requires an argument\n");
        break;
      case 'v':
        if (optarg != NULL)
          zero_point = atof(optarg);
        else
          printf("zero_point option requires an argument\n");
        break;
      default:
        printf(
            "Invalid option. Usage: %s [-s <size> -i <iter> -x <block_x> -y "
            "<block_y> -u <scale> -v <zero_point>]\n",
            argv[0]);
        return -1;
    }
  }

  const unsigned int grid_dim_x = 1;
  printf("size=%u, iter=%u\n", size, iter);
  printf("block_dim\t(%d, %d)\n", block_dim_x, block_dim_y);
  printf("grid_dim\t(%d, %d)\n", grid_dim_x, size / block_dim_y);
  unsigned int num_per_thread = size / (block_dim_x * grid_dim_x);
  printf("num_per_thread=%d\n", num_per_thread);
  if (bits != 16) {
    printf("scale=%f, zero_point=%f\n", scale, zero_point);
  }
  if (bits == 16) test_gemv_with_params(size, iter, block_dim_x, block_dim_y);
  if (bits == 8)
    test_gemv_int8_quantized_with_params(size, iter, block_dim_x, block_dim_y,
                                         scale, zero_point);
  if (bits == 4)
    test_gemv_int4_quantized_with_params(size, iter, block_dim_x, block_dim_y,
                                         scale, zero_point);
  else
    return -1;
}