#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;

#define N 64      // matrix size (64x64)
#define TILE 16   // tensor core tile (16x16x16)

// Each block = 1 warp (32 threads) computes ONE 16x16 output tile of C.
// 64x64 -> 4x4 = 16 output tiles -> 16 blocks.
__global__ void wmma_matmul(const half *A, const half *B, float *C) {
    int tileRow = blockIdx.y;
    int tileCol = blockIdx.x;

    wmma::fragment<wmma::matrix_a, TILE, TILE, TILE, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, TILE, TILE, TILE, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, TILE, TILE, TILE, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    // Walk along K dimension: 4 steps of 16
    for (int k = 0; k < N; k += TILE) {
        const half *aPtr = A + tileRow * TILE * N + k;
        const half *bPtr = B + k * N + tileCol * TILE;
        wmma::load_matrix_sync(a_frag, aPtr, N);
        wmma::load_matrix_sync(b_frag, bPtr, N);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);   // tensor core op
    }

    float *cPtr = C + tileRow * TILE * N + tileCol * TILE;
    wmma::store_matrix_sync(cPtr, c_frag, N, wmma::mem_row_major);
}

int main() {
    size_t nElem = N * N;
    half  *hA = (half*)malloc(nElem * sizeof(half));
    half  *hB = (half*)malloc(nElem * sizeof(half));
    float *fA = (float*)malloc(nElem * sizeof(float));
    float *fB = (float*)malloc(nElem * sizeof(float));
    float *hC = (float*)malloc(nElem * sizeof(float));
    float *ref = (float*)malloc(nElem * sizeof(float));

    srand(42);
    for (size_t i = 0; i < nElem; i++) {
        fA[i] = (rand() % 5) - 2;   // small ints: exact in FP16
        fB[i] = (rand() % 5) - 2;
        hA[i] = __float2half(fA[i]);
        hB[i] = __float2half(fB[i]);
    }

    // CPU reference
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            float s = 0;
            for (int k = 0; k < N; k++) s += fA[i*N+k] * fB[k*N+j];
            ref[i*N+j] = s;
        }

    half *dA, *dB; float *dC;
    cudaMalloc(&dA, nElem * sizeof(half));
    cudaMalloc(&dB, nElem * sizeof(half));
    cudaMalloc(&dC, nElem * sizeof(float));
    cudaMemcpy(dA, hA, nElem * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, nElem * sizeof(half), cudaMemcpyHostToDevice);

    dim3 grid(N / TILE, N / TILE);   // 4 x 4 = 16 tiles
    dim3 block(32);                  // one warp per tile
    wmma_matmul<<<grid, block>>>(dA, dB, dC);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(err)); return 1; }

    cudaMemcpy(hC, dC, nElem * sizeof(float), cudaMemcpyDeviceToHost);

    double maxErr = 0;
    for (size_t i = 0; i < nElem; i++) maxErr = fmax(maxErr, fabs(hC[i] - ref[i]));

    printf("Tiles used: %d (16x16 each)\n", (N/TILE)*(N/TILE));
    printf("Max abs error vs CPU: %g -> %s\n", maxErr, maxErr < 1e-3 ? "PASS" : "FAIL");
    printf("C[0][0..7]: ");
    for (int j = 0; j < 8; j++) printf("%.0f ", hC[j]);
    printf("\n");

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}
