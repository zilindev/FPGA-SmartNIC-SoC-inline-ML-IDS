// kernel.cu  (EE533 Lab 7)
// Required kernels: vec_add_i16, vec_sub_i16, bf16_mul, bf16_fma, relu_i16
// Each CUDA thread processes 4 elements to match: 4 packed elements per 64-bit register.

#include <stdint.h>
#include <cuda_bf16.h>

// -------------------------
// Helpers
// -------------------------
__device__ __forceinline__ int16_t relu_i16_scalar(int16_t x) {
    return (x < 0) ? (int16_t)0 : x;
}

__device__ __forceinline__ __nv_bfloat16 relu_bf16_scalar(__nv_bfloat16 x) {
    // Keep it simple and explicit: compare in float, return bf16.
    float fx = __bfloat162float(x);
    return (fx < 0.0f) ? __float2bfloat16(0.0f) : x;
}

// ---------------------------------------------------------------------
// INT16 kernels (4 elements per "thread" => one 64-bit packed register)
// ---------------------------------------------------------------------

extern "C" __global__
void vec_add_i16(const int16_t* __restrict__ a,
                 const int16_t* __restrict__ b,
                 int16_t* __restrict__ out,
                 int n_elements)
{
    int tid  = (int)threadIdx.x;
    int base = tid * 4;

    // Single bounds check for a full 4-lane SIMD word
    if (base + 3 >= n_elements) return;

    out[base + 0] = (int16_t)(a[base + 0] + b[base + 0]);
    out[base + 1] = (int16_t)(a[base + 1] + b[base + 1]);
    out[base + 2] = (int16_t)(a[base + 2] + b[base + 2]);
    out[base + 3] = (int16_t)(a[base + 3] + b[base + 3]);
}

extern "C" __global__
void vec_sub_i16(const int16_t* __restrict__ a,
                 const int16_t* __restrict__ b,
                 int16_t* __restrict__ out,
                 int n_elements)
{
    int tid  = (int)threadIdx.x;
    int base = tid * 4;

    if (base + 3 >= n_elements) return;

    out[base + 0] = (int16_t)(a[base + 0] - b[base + 0]);
    out[base + 1] = (int16_t)(a[base + 1] - b[base + 1]);
    out[base + 2] = (int16_t)(a[base + 2] - b[base + 2]);
    out[base + 3] = (int16_t)(a[base + 3] - b[base + 3]);
}

extern "C" __global__
void relu_i16(const int16_t* __restrict__ in,
              int16_t* __restrict__ out,
              int n_elements)
{
    int tid  = (int)threadIdx.x;
    int base = tid * 4;

    if (base + 3 >= n_elements) return;

    out[base + 0] = relu_i16_scalar(in[base + 0]);
    out[base + 1] = relu_i16_scalar(in[base + 1]);
    out[base + 2] = relu_i16_scalar(in[base + 2]);
    out[base + 3] = relu_i16_scalar(in[base + 3]);
}

// ---------------------------------------------------------------------
// BF16 kernels (4 elements per "thread" => one 64-bit packed register)
// ---------------------------------------------------------------------

extern "C" __global__
void bf16_mul(const __nv_bfloat16* __restrict__ a,
              const __nv_bfloat16* __restrict__ b,
              __nv_bfloat16* __restrict__ out,
              int n_elements)
{
    int tid  = (int)threadIdx.x;
    int base = tid * 4;

    if (base + 3 >= n_elements) return;

    // Use CUDA bf16 intrinsics (keeps intent clear for PTX parsing)
    out[base + 0] = __hmul(a[base + 0], b[base + 0]);
    out[base + 1] = __hmul(a[base + 1], b[base + 1]);
    out[base + 2] = __hmul(a[base + 2], b[base + 2]);
    out[base + 3] = __hmul(a[base + 3], b[base + 3]);
}

extern "C" __global__
void bf16_fma(const __nv_bfloat16* __restrict__ a,
              const __nv_bfloat16* __restrict__ b,
              const __nv_bfloat16* __restrict__ c,
              __nv_bfloat16* __restrict__ out,
              int n_elements)
{
    int tid  = (int)threadIdx.x;
    int base = tid * 4;

    if (base + 3 >= n_elements) return;

    out[base + 0] = __hfma(a[base + 0], b[base + 0], c[base + 0]);
    out[base + 1] = __hfma(a[base + 1], b[base + 1], c[base + 1]);
    out[base + 2] = __hfma(a[base + 2], b[base + 2], c[base + 2]);
    out[base + 3] = __hfma(a[base + 3], b[base + 3], c[base + 3]);
}
