#pragma once

/* This file defines math functions compatible across different gpu
 * platforms (currently CUDA and HIP).
 */
#if defined(__CUDACC__) || defined(__HIPCC__)

#include <c10/macros/Macros.h>

#ifdef __HIPCC__
#define __MATH_FUNCTIONS_DECL__ inline C10_DEVICE
#else /* __HIPCC__ */
#ifdef __CUDACC_RTC__
#define __MATH_FUNCTIONS_DECL__ C10_HOST_DEVICE
#else /* __CUDACC_RTC__ */
#define __MATH_FUNCTIONS_DECL__ static inline C10_HOST_DEVICE
#endif /* __CUDACC_RTC__ */
#endif /* __HIPCC__ */

namespace c10 {
namespace cuda {
namespace compat {

__MATH_FUNCTIONS_DECL__ float abs(float x) {
  return ::fabsf(x);
}
__MATH_FUNCTIONS_DECL__ double abs(double x) {
  return ::fabs(x);
}

__MATH_FUNCTIONS_DECL__ float exp(float x) {
  return ::expf(x);
}
__MATH_FUNCTIONS_DECL__ double exp(double x) {
  return ::exp(x);
}

__MATH_FUNCTIONS_DECL__ float ceil(float x) {
  return ::ceilf(x);
}
__MATH_FUNCTIONS_DECL__ double ceil(double x) {
  return ::ceil(x);
}


__MATH_FUNCTIONS_DECL__ float fp32_from_bits(uint32_t w) {
#if defined(__OPENCL_VERSION__)
  return as_float(w);
#elif defined(__CUDA_ARCH__)
  return __uint_as_float((unsigned int)w);
#elif defined(__INTEL_COMPILER)
  return _castu32_f32(w);
#else
  union {
    uint32_t as_bits;
    float as_value;
  } fp32 = {w};
  return fp32.as_value;
#endif
}

__MATH_FUNCTIONS_DECL__ uint32_t fp32_to_bits(float f) {
#if defined(__OPENCL_VERSION__)
  return as_uint(f);
#elif defined(__CUDA_ARCH__)
  return (uint32_t)__float_as_uint(f);
#elif defined(__INTEL_COMPILER)
  return _castf32_u32(f);
#else
  union {
    float as_value;
    uint32_t as_bits;
  } fp32 = {f};
  return fp32.as_bits;
#endif
}

__MATH_FUNCTIONS_DECL__ double fp64_from_bits(uint64_t w) {
#if defined(__CUDA_ARCH__)
  return __longlong_as_double(w);
#else
  union {
    uint64_t as_bits;
    double as_value;
  } fp64 = {w};
  return fp64.as_value;
#endif
}

__MATH_FUNCTIONS_DECL__ uint64_t fp64_to_bits(double f) {
#if defined(__CUDA_ARCH__)
  return __double_as_longlong(f);
#else
  union {
    double as_value;
    int64_t as_bits;
  } fp64 = {f};
  return fp64.as_bits;
#endif
}

__MATH_FUNCTIONS_DECL__ float copysign(float x, float y) {
  #if (!defined(__aarch64__)) || defined(__clang__) || (__GNUC__ > 8)
   // std::copysign gets ICE/Segfaults with gcc 7/8 on arm64
   // (e.g. Jetson), see PyTorch PR #51834
    return ::copysignf(x, y);
  #else
    return fp32_from_bits(
        (fp32_to_bits(x) & 0x7fffffffu) | (fp32_to_bits(y) & 0x80000000u));
  #endif
}
__MATH_FUNCTIONS_DECL__ double copysign(double x, double y) {
  #if (!defined(__aarch64__)) || defined(__clang__) || (__GNUC__ > 8)
    return ::copysign(x, y);
  #else
    return fp64_from_bits(
        (fp64_to_bits(x) & 0x7fffffffffffffffull) |
        (fp64_to_bits(y) & 0x8000000000000000ull));
  #endif
}

__MATH_FUNCTIONS_DECL__ float floor(float x) {
  return ::floorf(x);
}
__MATH_FUNCTIONS_DECL__ double floor(double x) {
  return ::floor(x);
}

__MATH_FUNCTIONS_DECL__ float log(float x) {
  return ::logf(x);
}
__MATH_FUNCTIONS_DECL__ double log(double x) {
  return ::log(x);
}

__MATH_FUNCTIONS_DECL__ float log1p(float x) {
  return ::log1pf(x);
}

__MATH_FUNCTIONS_DECL__ double log1p(double x) {
  return ::log1p(x);
}

__MATH_FUNCTIONS_DECL__ float max(float x, float y) {
  return ::fmaxf(x, y);
}
__MATH_FUNCTIONS_DECL__ double max(double x, double y) {
  return ::fmax(x, y);
}

__MATH_FUNCTIONS_DECL__ float min(float x, float y) {
  return ::fminf(x, y);
}
__MATH_FUNCTIONS_DECL__ double min(double x, double y) {
  return ::fmin(x, y);
}

__MATH_FUNCTIONS_DECL__ float pow(float x, float y) {
  return ::powf(x, y);
}
__MATH_FUNCTIONS_DECL__ double pow(double x, double y) {
  return ::pow(x, y);
}

__MATH_FUNCTIONS_DECL__ void sincos(float x, float* sptr, float* cptr) {
  return ::sincosf(x, sptr, cptr);
}
__MATH_FUNCTIONS_DECL__ void sincos(double x, double* sptr, double* cptr) {
  return ::sincos(x, sptr, cptr);
}

__MATH_FUNCTIONS_DECL__ float sqrt(float x) {
  return ::sqrtf(x);
}
__MATH_FUNCTIONS_DECL__ double sqrt(double x) {
  return ::sqrt(x);
}

__MATH_FUNCTIONS_DECL__ float rsqrt(float x) {
  return ::rsqrtf(x);
}
__MATH_FUNCTIONS_DECL__ double rsqrt(double x) {
  return ::rsqrt(x);
}

__MATH_FUNCTIONS_DECL__ float tan(float x) {
  return ::tanf(x);
}
__MATH_FUNCTIONS_DECL__ double tan(double x) {
  return ::tan(x);
}

__MATH_FUNCTIONS_DECL__ float tanh(float x) {
  return ::tanhf(x);
}
__MATH_FUNCTIONS_DECL__ double tanh(double x) {
  return ::tanh(x);
}

__MATH_FUNCTIONS_DECL__ float normcdf(float x) {
  return ::normcdff(x);
}
__MATH_FUNCTIONS_DECL__ double normcdf(double x) {
  return ::normcdf(x);
}

} // namespace compat
} // namespace cuda
} // namespace c10

#endif
