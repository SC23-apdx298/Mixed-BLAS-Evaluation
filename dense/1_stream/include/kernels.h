/*
  STREAM Benchmark for CUDA
  Author: Mohammed Sourouri <mohammed.sourouri@ntnu.no>
  Date: December 16, 2013
  Updated: June 5, 2017

  Comment: Code uses kernels from the original
           STREAM Benchmark written by John McCalpin.
           https://www.cs.virginia.edu/stream/
*/

#ifndef KERNELS_H_
#define KERNELS_H_

#include "common.h"
#include "common_gpu.h"

extern "C"
{
#include "kernels.h"

  __global__ void initialize(size_t num_elems, real *buffer, const real value)
  {
    const unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < num_elems)
    {
      buffer[idx] = value;
    }
  }

  __global__ void copy(size_t num_elems, real *buffer_b, const real *__restrict__ const buffer_a)
  {
    const unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < num_elems)
    {
      buffer_b[idx] = buffer_a[idx];
    }
  }

  __global__ void scale(size_t num_elems, real *buffer_c, const real *__restrict__ const buffer_b)
  {
    const unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < num_elems)
    {
      buffer_c[idx] = k_scale * buffer_b[idx];
    }
  }

  __global__ void add(size_t num_elems, real *buffer_c, const real *__restrict__ const buffer_a, const real *__restrict__ const buffer_b)
  {
    const unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < num_elems)
    {
      buffer_c[idx] = buffer_a[idx] + buffer_b[idx];
    }
  }

  __global__ void triad(size_t num_elems, real alpha, real *buffer_c, const real *__restrict__ const buffer_a, const real *__restrict__ const buffer_b)
  {
    const unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < num_elems)
    {
      buffer_c[idx] = buffer_a[idx] + alpha * buffer_b[idx];
    }
  }

  extern "C" void initialize_device_buffer(dim3 thread_blocks, dim3 threads_per_block, size_t h_num_elems, real *d_buffer, const real value)
  {
    initialize<<<thread_blocks, threads_per_block>>>(h_num_elems, d_buffer, value);
  }

  extern "C" void stream_copy(dim3 thread_blocks, dim3 threads_per_block, size_t h_num_elems, real *buf_a, real *buf_b, real *buf_c)
  {
    copy<<<thread_blocks, threads_per_block>>>(h_num_elems, buf_b, buf_a);
  }

  extern "C" void stream_scale(dim3 thread_blocks, dim3 threads_per_block, size_t h_num_elems, real *buf_a, real *buf_b, real *buf_c)
  {
    scale<<<thread_blocks, threads_per_block>>>(h_num_elems, buf_c, buf_b);
  }

  extern "C" void stream_add(dim3 thread_blocks, dim3 threads_per_block, size_t h_num_elems, real *buf_a, real *buf_b, real *buf_c)
  {
    add<<<thread_blocks, threads_per_block>>>(h_num_elems, buf_c, buf_a, buf_b);
  }

  extern "C" void stream_triad(dim3 thread_blocks, dim3 threads_per_block, size_t h_num_elems, real *buf_a, real *buf_b, real *buf_c)
  {
    real alpha = 3.0;
    triad<<<thread_blocks, threads_per_block>>>(h_num_elems, alpha, buf_c, buf_a, buf_b);
  }
}

#endif // KERNELS_H_
