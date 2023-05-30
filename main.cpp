#include <getopt.h>
#include <stdio.h>

#include <cstdlib>

void test_gemv(unsigned int size, unsigned int iter);

int main(int argc, char** argv) {
  // parse commandline options
  int opt;
  static struct option long_options[] = {{"size", required_argument, 0, 's'},
                                         {"iter", required_argument, 0, 'i'},
                                         {0, 0, 0, 0}};

  unsigned int size = 512;
  unsigned int iter = 1;

  while ((opt = getopt_long(argc, argv, "s:i:", long_options, NULL)) != -1) {
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
      default:
        printf("Invalid option. Usage: %s -s <size> -i <iter>\n", argv[0]);
        return -1;
    }
  }

  printf("size=%u, iter=%u\n", size, iter);

  test_gemv(size, iter);
}