#include "pch.h"
#include "solve.h"

void solve_no_paralle(int W, int H, int N, float * input, float * output)
{
	const int TRUNK_SIZE = 64;
	for (int i = 0; i < H; i += TRUNK_SIZE) {

		float *tmp = new float[W*(TRUNK_SIZE + N - 1)];

		for (int wj = 0; wj < TRUNK_SIZE + N - 1 && wj + i < H + N - 1; wj++) {
			for (int j = 0; j < W; j++) {
				float tmp_float = 0;
				tmp[wj*W + j] = 0;
				for (int wi = 0; wi < N; wi++) {
					tmp_float += input[(i + wj)*(W + N - 1) + j + wi];
				}
				tmp[wj*W + j] = tmp_float / N;

			}
		}

		// ��������Ľ��
		for (int ii = 0; ii < TRUNK_SIZE&&ii + i < H; ii++) {
			for (int j = 0; j < W; j++) {
				float sum = 0;
				for (int k = 0; k < N; k++) {
					sum += tmp[(ii + k)*W + j];
				}
				output[(i + ii)*W + j] = sum / N;
			}
		}
		delete tmp;

	}
}

void solve_paralle(int W, int H, int N, float * input, float * output)
{
	const int CHUNK_SIZE = 64;

#pragma omp parallel for
	for (int i = 0; i < H; i += CHUNK_SIZE) {

		float *tmp_buf =
			(float *)malloc(sizeof(float) * W * (CHUNK_SIZE + N - 1));

		for (int ii = 0; ii < CHUNK_SIZE + N - 1 && i + ii < H + N - 1; ii++)
			for (int j = 0; j < W; j++) {
				float tmp = 0;
				for (int jj = 0; jj < N; jj++)
					tmp += input[(i + ii) * (W + N - 1) + j + jj];

				tmp_buf[ii * W + j] = tmp / N;
			}

		for (int ii = 0; ii < CHUNK_SIZE && i + ii < H; ii++)
			for (int j = 0; j < W; j++) {
				float tmp = 0;
				for (int jj = 0; jj < N; jj++)
					tmp += tmp_buf[(jj + ii) * W + j];

				output[(i + ii) * W + j] = tmp / N;
			}

		free(tmp_buf);
	}
}


void solve_vec(int W, int H, int N, float * input, float * output)
{
	const int TRUNK_SIZE = 64;
	const int WIDTH_SIZE = 256;	// tmp �Ŀ��,256����������ô��
	vec_t divisor = vec_set1_float(1.0 / N);	//  ����vec_t 1/N  
	for (int i = 0; i < H; i += TRUNK_SIZE) {

		vec_t *tmp = new vec_t[(TRUNK_SIZE + N - 1)*WIDTH_SIZE / VEC_SIZE];		//tmp ��ʱ����Ĵ�С�� WIDTH_SIZE*TRUNK_SIZE ��float
		vec_t *tmp_ptr;		// tmp ��ǰ�е�ָ��
		vec_t sum, avg;
		// ������tmp,����N��ֵ��ƽ��ֵ  
		for (int j = 0; j < W; j += WIDTH_SIZE) {						// �к�
			// һ���� tmp����
			for (int wj = 0; wj < TRUNK_SIZE + N - 1 && wj + i < H + N - 1; wj++) {	// �к�

				tmp_ptr = tmp + wj * WIDTH_SIZE / VEC_SIZE;				
				float *input_ptr = input + (i + wj)*(W + N - 1) + j;	//input �ĵ�ǰλ��
				
				for (int x = 0; x < WIDTH_SIZE &&x + j < W; x+=VEC_SIZE) {
					sum = vec_set1_float(0.0);
					for (int k = 0; k < N; k++) {
						sum = vec_add(sum, vec_load(input_ptr + k));
					}
					avg = vec_mul(sum, divisor);
					vec_store((float*)tmp_ptr, avg);
					tmp_ptr++;
					input_ptr += VEC_SIZE;
				}
			}
			//	tmp �������,����������ƽ��
			tmp_ptr = tmp;

			for (int y = 0; y < TRUNK_SIZE&& i + y < H; y++) {	// j��tmp���к�
				vec_t *out_ptr = (vec_t*)(output + (i + y)*W + j);
				tmp_ptr = tmp + y * WIDTH_SIZE / VEC_SIZE;

				for (int x = 0; x < WIDTH_SIZE&&x + j < W; x += VEC_SIZE) {
					sum = vec_set1_float(0.0);

					for (int k = 0; k < N; k++) {
						sum = vec_add(sum, *(tmp_ptr + k * WIDTH_SIZE / VEC_SIZE));
					}

					avg = vec_mul(sum, divisor);
					vec_store((float*)out_ptr, avg);
					tmp_ptr++;
					out_ptr++;
				}
			}
		}

		delete tmp;

	}
}

void solve_parall_vec(int W, int H, int N, float * input, float * output)
{
	const int TRUNK_SIZE = 64;
	const int WIDTH_SIZE = 256;	// tmp �Ŀ��,256����������ô��
	vec_t divisor = vec_set1_float(1.0 / N);	//  ����vec_t 1/N  
#pragma omp parallel for
	for (int i = 0; i < H; i += TRUNK_SIZE) {

		vec_t *tmp = new vec_t[(TRUNK_SIZE + N - 1)*WIDTH_SIZE / VEC_SIZE];		//tmp ��ʱ����Ĵ�С�� WIDTH_SIZE*TRUNK_SIZE ��float
		vec_t *tmp_ptr;		// tmp ��ǰ�е�ָ��
		vec_t sum, avg;
		// ������tmp,����N��ֵ��ƽ��ֵ  
		for (int j = 0; j < W; j += WIDTH_SIZE) {						// �к�
			// һ���� tmp����
			for (int wj = 0; wj < TRUNK_SIZE + N - 1 && wj + i < H + N - 1; wj++) {	// �к�

				tmp_ptr = tmp + wj * WIDTH_SIZE / VEC_SIZE;
				float *input_ptr = input + (i + wj)*(W + N - 1) + j;	//input �ĵ�ǰλ��

				for (int x = 0; x < WIDTH_SIZE &&x + j < W; x += VEC_SIZE) {
					sum = vec_set1_float(0.0);
					for (int k = 0; k < N; k++) {
						sum = vec_add(sum, vec_load(input_ptr + k));
					}
					avg = vec_mul(sum, divisor);
					vec_store((float*)tmp_ptr, avg);
					tmp_ptr++;
					input_ptr += VEC_SIZE;
				}
			}
			//	tmp �������,����������ƽ��
			tmp_ptr = tmp;

			for (int y = 0; y < TRUNK_SIZE&& i + y < H; y++) {	// j��tmp���к�
				vec_t *out_ptr = (vec_t*)(output + (i + y)*W + j);
				tmp_ptr = tmp + y * WIDTH_SIZE / VEC_SIZE;

				for (int x = 0; x < WIDTH_SIZE&&x + j < W; x += VEC_SIZE) {
					sum = vec_set1_float(0.0);

					for (int k = 0; k < N; k++) {
						sum = vec_add(sum, *(tmp_ptr + k * WIDTH_SIZE / VEC_SIZE));
					}

					avg = vec_mul(sum, divisor);
					vec_store((float*)out_ptr, avg);
					tmp_ptr++;
					out_ptr++;
				}
			}
		}

		delete tmp;

	}
}

void solve_naive(int W, int H, int N, float * input, float * output)
{

	for (int i = 0; i < H; i++) {
		for (int j = 0; j < W; j++) {
			float sum = 0;
			for (int x = 0; x < N; x++) {
				for (int y = 0; y < N; y++) {
					sum += input[(i+x)*(W+N-1)+j+y];
				}
			}
			output[i*W + j] = sum / N / N;
		}
	}
}

void solve_naive_parallel(int W, int H, int N, float * input, float * output)
{
#pragma omp parallel for
	for (int i = 0; i < H; i++) {
		for (int j = 0; j < W; j++) {
			float sum = 0;
			for (int x = 0; x < N; x++) {
				for (int y = 0; y < N; y++) {
					sum += input[(i + x)*(W + N - 1) + j + y];
				}
			}
			output[i*W + j] = sum / N / N;
		}
	}
}
