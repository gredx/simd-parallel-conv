#ifndef _SIMD_H_
#define _SIMD_H_



#include <immintrin.h>

// Each vector contains 8 float numbers.
typedef __m256 vec_t;

#define VEC_SIZE 8

// Perform 8 additions using a single instruction.
inline vec_t vec_add(vec_t a, vec_t b) {
    return _mm256_add_ps(a, b);
}

// Return a vector with 8 float numbers with the same value.
inline vec_t vec_set1_float(float x) {
    return _mm256_set1_ps(x);
}

// Return a vector whose elements are the product of the corresponding float
// numbers.
inline vec_t vec_mul(vec_t a, vec_t b) {
    return _mm256_mul_ps(a, b);
}

// Read a vector from the given address.
inline vec_t vec_load(float const *mem_addr) {
    return _mm256_loadu_ps(mem_addr);
}

// Store a vector to the given memory address.
 inline void vec_store(float *mem_addr, vec_t a) {
    _mm256_storeu_ps(mem_addr, a);
}

#endif


