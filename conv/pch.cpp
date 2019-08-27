// pch.cpp: 与预编译标头对应的源文件；编译成功所必需的

#include "pch.h"
#include <stdio.h>   
#include <immintrin.h>   
#include <stdlib.h>   
#include <algorithm>   
#include <windows.h>   

using namespace std;

const int maxN = 1024;           // magnitude of matrix   
const int T = 64;                // tile size   

float A[maxN][maxN];
float B[maxN][maxN];
float X[maxN][maxN];

long long head, tail, freq;        //timers   

void mul(int n, float a[][maxN], float b[][maxN], float c[][maxN]) {
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			c[i][j] = 0.0;
			for (int k = 0; k < n; ++k) {
				c[i][j] += a[i][k] * b[k][j];
			}
		}
	}
}

void trans_mul(int n, float a[][maxN], float b[][maxN], float c[][maxN]) {
	// transposition   
	for (int i = 0; i < n; ++i) for (int j = 0; j < i; ++j) swap(b[i][j], b[j][i]);
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			c[i][j] = 0.0;
			for (int k = 0; k < n; ++k) {
				c[i][j] += a[i][k] * b[j][k];
			}
		}
	}
	// transposition   
	for (int i = 0; i < n; ++i) for (int j = 0; j < i; ++j) swap(b[i][j], b[j][i]);
}

void sse_mul(int n, float a[][maxN], float b[][maxN], float c[][maxN]) {
	__m128 t1, t2, sum;

	for (int i = 0; i < n; ++i) for (int j = 0; j < i; ++j) swap(b[i][j], b[j][i]);
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			c[i][j] = 0.0;
			sum = _mm_setzero_ps();
			for (int k = n - 4; k >= 0; k -= 4) {     //sum every 4th elements   
				t1 = _mm_loadu_ps(a[i] + k);
				t2 = _mm_loadu_ps(b[j] + k);
				t1 = _mm_mul_ps(t1, t2);
				sum = _mm_add_ps(sum, t1);
			}
			sum = _mm_hadd_ps(sum, sum);
			sum = _mm_hadd_ps(sum, sum);
			_mm_store_ss(c[i] + j, sum);
			for (int k = (n % 4) - 1; k >= 0; --k) {    //handle the last n%4 elements   
				c[i][j] += a[i][k] * b[j][k];
			}
		}
	}
	for (int i = 0; i < n; ++i) for (int j = 0; j < i; ++j) swap(b[i][j], b[j][i]);
}

void avx_mul(int n, float a[][maxN], float b[][maxN], float c[][maxN]) {
	__m256 t1, t2, sum;
	__m128 s1, s2;

	for (int i = 0; i < n; ++i) for (int j = 0; j < i; ++j) swap(b[i][j], b[j][i]);
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			c[i][j] = 0.0;
			sum = _mm256_setzero_ps();
			for (int k = n - 8; k >= 0; k -= 8) {     //sum every 4th elements   
				t1 = _mm256_loadu_ps(a[i] + k);
				t2 = _mm256_loadu_ps(b[j] + k);
				t1 = _mm256_mul_ps(t1, t2);
				sum = _mm256_add_ps(sum, t1);
			}
			s1 = _mm256_extractf128_ps(sum, 0);  // s1=[a0,a1,a2,a3]   
			s2 = _mm256_extractf128_ps(sum, 1);  // s2=[a4,a5,a6,a7]   
			s1 = _mm_hadd_ps(s1, s2);   // s1=[a0+a1,a2+a3,a4+a5,a6+a7]   
			s1 = _mm_hadd_ps(s1, s1); // s1=[a0+a1+a2+a3,a4+a5+a6+a7,a0+a1+a2+a3,a4+a5+a6+a7]   
			s1 = _mm_hadd_ps(s1, s1); // s1=[a0+a1+a2+a3+a4+a5+a6+a7,...]   
			_mm_store_ss(c[i] + j, s1);
			for (int k = (n % 8) - 1; k >= 0; --k) {    //handle the last n%4 elements   
				c[i][j] += a[i][k] * b[j][k];
			}
		}
	}
	for (int i = 0; i < n; ++i) for (int j = 0; j < i; ++j) swap(b[i][j], b[j][i]);
}

void sse_tile(int n, float a[][maxN], float b[][maxN], float c[][maxN])
{
	__m128 t1, t2, sum;
	float t;

	for (int i = 0; i < n; ++i) for (int j = 0; j < i; ++j) swap(b[i][j], b[j][i]);
	for (int r = 0; r < n / T; ++r)  for (int q = 0; q < n / T; ++q) {
		for (int i = 0; i < T; ++i) for (int j = 0; j < T; ++j)  c[r * T + i][q * T + j] = 0.0;

		for (int p = 0; p < n / T; ++p) {
			for (int i = 0; i < T; ++i)   for (int j = 0; j < T; ++j) {
				sum = _mm_setzero_ps();

				for (int k = 0; k < T; k += 4) {     //sum every 4th elements   
					t1 = _mm_loadu_ps(a[r * T + i] + p * T + k);
					t2 = _mm_loadu_ps(b[q * T + j] + p * T + k);
					t1 = _mm_mul_ps(t1, t2);
					sum = _mm_add_ps(sum, t1);
				}
				sum = _mm_hadd_ps(sum, sum);
				sum = _mm_hadd_ps(sum, sum);
				_mm_store_ss(&t, sum);
				c[r * T + i][q * T + j] += t;
			}
		}
	}

	for (int i = 0; i < n; ++i) for (int j = 0; j < i; ++j) swap(b[i][j], b[j][i]);
}

void avx_tile(int n, float a[][maxN], float b[][maxN], float c[][maxN])
{
	__m256 t1, t2, sum;
	__m128 s1, s2;
	float t;

	for (int i = 0; i < n; ++i) for (int j = 0; j < i; ++j) swap(b[i][j], b[j][i]);
	for (int r = 0; r < n / T; ++r)  for (int q = 0; q < n / T; ++q) {
		for (int i = 0; i < T; ++i) for (int j = 0; j < T; ++j)  c[r * T + i][q * T + j] = 0.0;

		for (int p = 0; p < n / T; ++p) {
			for (int i = 0; i < T; ++i)   for (int j = 0; j < T; ++j) {
				sum = _mm256_setzero_ps();

				for (int k = 0; k < T; k += 8) {     //sum every 4th elements   
					t1 = _mm256_loadu_ps(a[r * T + i] + p * T + k);
					t2 = _mm256_loadu_ps(b[q * T + j] + p * T + k);
					t1 = _mm256_mul_ps(t1, t2);
					sum = _mm256_add_ps(sum, t1);
				}
				s1 = _mm256_extractf128_ps(sum, 0);  // s1=[a0,a1,a2,a3]   
				s2 = _mm256_extractf128_ps(sum, 1);  // s2=[a4,a5,a6,a7]   
				s1 = _mm_hadd_ps(s1, s2);   // s1=[a0+a1,a2+a3,a4+a5,a6+a7]   
				s1 = _mm_hadd_ps(s1, s1); // s1=[a0+a1+a2+a3,a4+a5+a6+a7,a0+a1+a2+a3,a4+a5+a6+a7]   
				s1 = _mm_hadd_ps(s1, s1); // s1=[a0+a1+a2+a3+a4+a5+a6+a7,...]   
				_mm_store_ss(&t, s1);
				c[r * T + i][q * T + j] += t;
			}
		}
	}

	for (int i = 0; i < n; ++i) for (int j = 0; j < i; ++j) swap(b[i][j], b[j][i]);
}

void init(int n) {
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			A[i][j] = 0.0;
			X[i][j] = (float)rand();
		}
		A[i][i] = 1.0;
	}
}
/*
int main()
{
	QueryPerformanceFrequency((LARGE_INTEGER *)&freq);

	init(maxN);
	QueryPerformanceCounter((LARGE_INTEGER *)&head);
	mul(maxN, A, X, B);
	QueryPerformanceCounter((LARGE_INTEGER *)&tail);
	printf("Sequential: %lfms.\n", (tail - head) * 1000.0 / freq);

	init(maxN);
	QueryPerformanceCounter((LARGE_INTEGER *)&head);
	trans_mul(maxN, A, X, B);
	QueryPerformanceCounter((LARGE_INTEGER *)&tail);
	printf("Transposition: %lfms.\n", (tail - head) * 1000.0 / freq);

	init(maxN);
	QueryPerformanceCounter((LARGE_INTEGER *)&head);
	sse_mul(maxN, A, X, B);
	QueryPerformanceCounter((LARGE_INTEGER *)&tail);
	printf("SSE: %lfms.\n", (tail - head) * 1000.0 / freq);

	init(maxN);
	QueryPerformanceCounter((LARGE_INTEGER *)&head);
	avx_mul(maxN, A, X, B);
	QueryPerformanceCounter((LARGE_INTEGER *)&tail);
	printf("AVX: %lfms.\n", (tail - head) * 1000.0 / freq);

	init(maxN);
	QueryPerformanceCounter((LARGE_INTEGER *)&head);
	sse_tile(maxN, A, X, B);
	QueryPerformanceCounter((LARGE_INTEGER *)&tail);
	printf("SSE tile: %lfms.\n", (tail - head) * 1000.0 / freq);

	init(maxN);
	QueryPerformanceCounter((LARGE_INTEGER *)&head);
	avx_tile(maxN, A, X, B);
	QueryPerformanceCounter((LARGE_INTEGER *)&tail);
	printf("AVX tile: %lfms.\n", (tail - head) * 1000.0 / freq);

	return 0;
}
*/
// 一般情况下，忽略此文件，但如果你使用的是预编译标头，请保留它。
