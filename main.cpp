#include <getopt.h>
#include <stdio.h>

#include <cstdlib>

void test_gemv(unsigned int size, unsigned int iter);
void test_gemv_with_params(unsigned int size, unsigned int iter, int num_kernels, int block_dim_x, int block_dim_y, int grid_dim_x);

int main(int argc, char** argv) {
  // parse commandline options
  int opt;
  static struct option long_options[] = {{"size", required_argument, 0, 's'},
                                         {"iter", required_argument, 0, 'i'},
                                         {"kernels", required_argument, 0, 'k'},
                                         {"block_x", required_argument, 0, 'x'},
                                         {"block_y", required_argument, 0, 'y'},
                                         {"grid_x", required_argument, 0, 'g'},
                                         {0, 0, 0, 0}};

  unsigned int size = 512;
  unsigned int iter = 1;
  int block_dim_x = 32;
  int block_dim_y = 4;
  int grid_dim_x = 1;
  int num_kernels = 1;

  while ((opt = getopt_long(argc, argv, "s:i:k:x:y:g:", long_options, NULL)) != -1) {
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
      case 'k':
        if (optarg != NULL)
          num_kernels = atoi(optarg);
        else
          printf("kernels option requires an argument\n");
        break;
      case 'x':
        if (optarg != NULL)
          block_dim_x = atoi(optarg);
        else
          printf("block_x option requires an argument\n");
        break;
      case 'y':
        if (optarg != NULL)
          block_dim_y = atoi(optarg);
        else
          printf("block_y option requires an argument\n");
        break;
      case 'g':
        if (optarg != NULL)
          grid_dim_x = atoi(optarg);
        else
          printf("grid_x option requires an argument\n");
        break;
      default:
        printf("Invalid option. Usage: %s -s <size> -i <iter> -k <kernels> -x <block_x> -y <block_y> -g <grid_x>\n", argv[0]);
        return -1;
    }
  }

  printf("size=%u, iter=%u\n", size, iter);

  // test_gemv(size, iter);

  printf("block_dim\t(%d, %d)\n", block_dim_x, block_dim_y);
  printf("grid_dim\t(%d, %d)\n", grid_dim_x, size / block_dim_y);
  test_gemv_with_params(size, iter, num_kernels, block_dim_x, block_dim_y, grid_dim_x);
}