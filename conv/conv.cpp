// conv.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include "pch.h"

#include "simd.h"
#include "solve.h"



int seed = 132856;
#define FAST_RAND_MAX 32767

inline int fast_rand() {
	seed = 214013 * seed + 2531011;
	return (seed >> 16) & 0x7FFF;
}

void gen_input(int W, int H, int N, float *input) {
	for (int i = 0; i < (W + N - 1) * (H + N - 1); i++)
		input[i] = fast_rand() / (float)FAST_RAND_MAX;
}

int check_result(float *x, float *y, int n) {
	const float eps = 1e-5;

	for (int i = 0; i < n; i++) {
		if (fabsf(x[i] - y[i]) > eps)
			return 0;
	}

	return 1;
}

void test(int W, int H, int N) {

	float *input = new float[(W + N - 1) * (H + N - 1)];
	float *output = new float[W * H];

	gen_input(W, H, N, input);

	clock_t begin, end, result;

	float *output_baseline = new float[W*H];
	begin = clock();
	solve_parall_vec(W, H, N, input, output_baseline);
	end = clock();
	cout << left << setw(20) << "solve_parall_vec"   << setw(10) << "消耗时间:" << 1.0*(end - begin) / CLOCKS_PER_SEC << endl;

	int correctness = 0;

	begin = clock();
	solve_no_paralle(W, H, N, input, output);
	end = clock();
	correctness = check_result(output_baseline, output, W * H);
	cout << left << setw(20) << "solve_no_paralle" << "正确性 " << correctness << setw(10) << "消耗时间:" << 1.0*(end - begin) / CLOCKS_PER_SEC << endl;
	
	begin = clock();
	solve_paralle(W, H, N, input, output);
	end = clock();
	correctness = check_result(output_baseline, output, W * H);
	cout << left << setw(20) << "solve_paralle" << "正确性 " << correctness << setw(10) << "消耗时间:" << 1.0*(end - begin) / CLOCKS_PER_SEC << endl;
	
	begin = clock();
	solve_vec(W, H, N, input, output);
	end = clock();
	correctness = check_result(output_baseline, output, W * H);
	cout << left << setw(20) << "solve_vec" << "正确性 " << correctness << setw(10) << "消耗时间:" << 1.0*(end - begin) / CLOCKS_PER_SEC << endl;
	
	begin = clock();
	solve_naive(W, H, N, input, output);
	end = clock();
	correctness = check_result(output_baseline, output, W * H);
	cout << left << setw(20) << "solve_naive" << "正确性 " << correctness << setw(10) << "消耗时间:" << 1.0*(end - begin) / CLOCKS_PER_SEC << endl;
		
	begin = clock();
	solve_naive_parallel(W, H, N, input, output);
	end = clock();
	correctness = check_result(output_baseline, output, W * H);
	cout << left << setw(20) << "solve_naive_parallel" << "正确性 " << correctness << setw(10) << "消耗时间:" << 1.0*(end - begin) / CLOCKS_PER_SEC << endl;


	delete input;
	delete output;
	delete output_baseline;

}

int main() {

	int W=10000, H=10000, N=8;

	//cin >> W >> H >> N;
	test(W, H, N);

	return 0;
}
