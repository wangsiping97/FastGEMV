#ifndef SIMPLE_TENSOR_H_
#define SIMPLE_TENSOR_H_

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <iostream>

///////////////////////////// UTILITIES //////////////////////////////
// Define the error checking function
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

void check(cudaError_t result, char const* const func, const char* const file,
           int const line);

void print_cuda_info();

///////////////////////////// SOLVER //////////////////////////////

class SimpleTensor {
 public:
  SimpleTensor(unsigned height, unsigned width)
      : height_(height), width_(width) {
    checkCudaErrors(
        cudaMalloc((void**)&data_, height_ * width_ * sizeof(half)));
  }
  half* device_data() const { return data_; }
  /**
   * @brief generate a height_ * width_ matrix with random fp16 numbers
   */
  void reset();
  /**
   * @brief copy the numbers from device to the host
   */
  void to_host(half* host_data, unsigned n);
  /**
   * @brief compute the dot product
   */
  SimpleTensor solve_gemv(const SimpleTensor& other) const;
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

 private:
  unsigned int width_;
  unsigned int height_;
  // device data
  half* data_;
};

#endif  // SIMPLE_TENSOR_H_
