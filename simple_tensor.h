#ifndef SIMPLE_TENSOR_H_
#define SIMPLE_TENSOR_H_

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <iostream>
#include <cassert>

///////////////////////////// UTILITIES //////////////////////////////

// Define the error checking function
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

inline void check(cudaError_t result, char const* const func, const char* const file,
           int const line) {
  if (result) {
    fprintf(stderr, "CUDA error = %s at %s:%d '%s'\n",
            cudaGetErrorString(result), file, line, func);
    exit(1);
  }
}

///////////////////////////// TENSOR //////////////////////////////
template <typename T>
class SimpleTensor {
 public:
  SimpleTensor(unsigned height, unsigned width)
      : height_(height), width_(width) {
    checkCudaErrors(
        cudaMalloc((void**)&data_, height_ * width_ * sizeof(T)));
  }
  T* device_data() const { return data_; }
  /**
   * @brief generate a height_ * width_ matrix with random fp16 numbers
   */
  void reset();
  /**
   * @brief copy the numbers from device to the host
   */
  void to_host(T* host_data, unsigned n);
  /**
   * @brief move constructor
   */
  SimpleTensor(SimpleTensor&& other) noexcept
      : height_(other.height_), width_(other.width_), data_(other.data_) {
    other.data_ = nullptr;  // Ensure the other object won't delete the data
                            // after being destroyed
  }
  /**
   * @brief overload the assignment operator for move semantics
   */
  SimpleTensor& operator=(SimpleTensor&& other) noexcept {
    if (this != &other) {
      height_ = other.height_;
      width_ = other.width_;

      // Deallocate existing data
      cudaFree(data_);

      // Take ownership of the new data
      data_ = other.data_;
      other.data_ = nullptr;
    }

    return *this;
  }
  ~SimpleTensor() { checkCudaErrors(cudaFree(data_)); }

  unsigned int width_;
  unsigned int height_;
  // device data
  T* data_;
};

template <typename T>
void SimpleTensor<T>::reset() {
  unsigned int total_elements = height_ * width_;
  int threads_per_block = 256;
  int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

  if constexpr (std::is_same<T, half>::value) {
    generate_random_numbers<<<num_blocks, threads_per_block>>>(data_,
                                                            total_elements);
  }
  checkCudaErrors(cudaPeekAtLastError());
}

template <typename T>
void SimpleTensor<T>::to_host(T* host_data, unsigned n) {
  unsigned int total_elements = height_ * width_;
  assert(n <= total_elements);
  checkCudaErrors(
      cudaMemcpy(host_data, data_, n * sizeof(T), cudaMemcpyDeviceToHost));
}

#endif  // SIMPLE_TENSOR_H_
