#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <sys/time.h>

#include "vec_op.h"

#ifndef COMPUTE_TYPE_AX
#define COMPUTE_TYPE_AX CUDA_R_32F
#endif

#ifndef VALUE_TYPE_AX
#define VALUE_TYPE_AX float
#endif

#ifndef COMPUTE_TYPE_Y
#define COMPUTE_TYPE_Y CUDA_R_32F
#endif

#ifndef VALUE_TYPE_Y
#define VALUE_TYPE_Y float
#endif

#ifndef COMPUTE_TYPE
#define COMPUTE_TYPE CUDA_R_32F
#endif

#ifndef ALPHA_TYPE
#define ALPHA_TYPE float
#endif

#ifndef VALUE_TYPE
#define VALUE_TYPE double
#endif
#define STRS(args...) #args

#ifndef WARMUP_TIMES
#define WARMUP_TIMES 10
#endif

#ifndef BENCH_REPEAT
#define BENCH_REPEAT 1000
#endif

#ifndef THRESHOLD_VALUE
#define THRESHOLD_VALUE 2000
#endif

#ifndef WARP_SIZE
#define WARP_SIZE   64
#endif

#ifndef WARP_PER_BLOCK
#define WARP_PER_BLOCK   16
#endif

#define SUBSTITUTION_FORWARD  0
#define SUBSTITUTION_BACKWARD 1

#define OPT_WARP_NNZ   1
#define OPT_WARP_RHS   2
#define OPT_WARP_AUTO  3

#define bool char
#define true 1
#define false 0
