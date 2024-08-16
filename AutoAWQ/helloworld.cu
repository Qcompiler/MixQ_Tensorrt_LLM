
#include <iostream>
#include <cassert>
#include <cublasLt.h>
#include <curand_kernel.h>
#include <curand_kernel.h>
#include <mma.h>

using namespace nvcuda;

const int M = 16;  // 矩阵 A 的行数
const int N = 16;  // 矩阵 B 的列数
const int K = 16;  // 矩阵 A 的列数（也是矩阵 B 的行数）

const int tileSizeM = 16;
const int tileSizeN = 16;
const int tileSizeK = 16;

__global__ void initMatrix(float *matrix, int rows, int cols) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i < rows && j < cols) {
        int index = i * cols + j;
        matrix[index] = (float)(i * cols + j);
    }
}

int main() {
    // 计算矩阵大小
    int A_size = M * K;
    int B_size = K * N;
    int C_size = M * N;

    // 分配内存并初始化矩阵 A、B、C
    float *A, *B, *C;
    cudaMalloc((void**)&A, A_size * sizeof(float));
    cudaMalloc((void**)&B, B_size * sizeof(float));
    cudaMalloc((void**)&C, C_size * sizeof(float));

    dim3 gridDims((N + tileSizeN - 1) / tileSizeN, (M + tileSizeM - 1) / tileSizeM);
    dim3 blockDims(tileSizeN, tileSizeM);
    
    initMatrix<<<gridDims, blockDims>>>(A, M, K);
    initMatrix<<<gridDims, blockDims>>>(B, K, N);

    // 创建 MMA 操作描述符
    nvcuda::wmma::matrix_a<M, K, nvcuda::wmma::matrix_a::kind::row_major> a_matrix;
    nvcuda::wmma::matrix_b<K, N, nvcuda::wmma::matrix_b::kind::row_major> b_matrix;
    nvcuda::wmma::accumulator<N, M, N, nvcuda::wmma::accumulator::kind::row_major> c_matrix;

    // 初始化 MMA 操作
    for (int i = 0; i < M; i += M) {
        for (int j = 0; j < N; j += N) {
            for (int k = 0; k < K; k += K) {
                nvcuda::wmma::fill_fragment(c_matrix, 0.0f);
                nvcuda::wmma::load_matrix_sync(a_matrix, A + i * K + k, K);
                nvcuda::wmma::load_matrix_sync(b_matrix, B + k * N + j, N);
                nvcuda::wmma::mma_sync(c_matrix, a_matrix, b_matrix, c_matrix);
                nvcuda::wmma::store_matrix_sync(C + i * N + j, c_matrix, N, nvcuda::wmma::mem_row_major);
            }
        }
    }

    // 检查结果
    float *hostC = new float[C_size];
    cudaMemcpy(hostC, C, C_size * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float expected = 0.0f;
            for (int k = 0; k < K; k++) {
                expected += A[i * K + k] * B[k * N + j];
            }
            assert(fabs(hostC[i * N + j] - expected) < 1e-5);
        }
    }

    std::cout << "矩阵乘法结果正确！" << std::endl;

    // 释放内存
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    delete[] hostC;

    return 0;
}

