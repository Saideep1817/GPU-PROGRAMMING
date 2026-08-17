#include <stdio.h>
#include <cuda.h>

__device__ volatile int count = 0;

__global__ void barrier(){
    int tid = threadIdx.x;

    printf("Thread %d reached barrier\n", tid);

    // Atomic increment: each thread signals its own arrival.
    atomicAdd((int*)&count, 1);

    while (count < blockDim.x) {
        // busy-wait
    }
    __syncthreads();

    if (tid == 0)
        printf("All threads reached the barrier\n");

    printf("Thread %d passed barrier\n", tid);
}

int main(){
    int threads = 8;

    barrier<<<1, threads>>>();
    cudaDeviceSynchronize();
    return 0;
}