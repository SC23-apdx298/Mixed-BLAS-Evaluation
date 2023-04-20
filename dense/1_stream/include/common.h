/*
  STREAM Benchmark for CUDA
  Author: Mohammed Sourouri <mohammed.sourouri@ntnu.no>
  Date: December 16, 2013
  Updated: June 5, 2017

  Comment: Code uses kernels from the original
           STREAM Benchmark written by John McCalpin.
           https://www.cs.virginia.edu/stream/
*/
#pragma once

#ifndef COMMON_H_
#define COMMON_H_

#include <cuda_fp16.h>
#include <fstream>
#include <iostream>

// typedef double real;
// typedef float real;
// typedef __half real;
// constexpr real k_scale = 3.0f;
// real k_scale_h = 3.0;
// real k_scale;
#ifndef real
#define real float
#endif

#define k_scale static_cast<real>(3.0)

// constexpr size_t num_elems = (2 << 25);
// #ifndef num_elems
// #define num_elems (2 << 25)
// #endif

#endif // COMMON_H_

