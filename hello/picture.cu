// 20181010
// Yuqiong Li
// an example that uses CUDA to multiply all elements of an array by 2

#include <stdlib.h>
#include <stdio.h>

#define index(i, j, n) ((i) * (n) + (j))

__global__ void pictureKernel(float * a, unsigned int m, unsigned int n);

int main(){
    unsigned int m = 10, n = 3;  // dimensions
    unsigned int size = m * n * sizeof(float);

    float * a;  // declare matrices
    
    a = (float * ) malloc(size);  // a is m by n

    int i = 0, j = 0;
    // initializing a
    for (i = 0; i < m; i++){
        for (j = 0; j < n; j++)
            a[index(i, j, n)] = (i + j) / 1.3;
    }
 
    printf("Now print out the original values: \n");
    for (i = 0; i < m; i++){
        for (j = 0; j < n; j++){
            printf("%.2f\t", a[index(i, j, n)]);
        }
    }

    // 1. allocate device memory for a 
    float * d_a;
    cudaMalloc(& d_a, size);
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);

    // 2. invoke kernel function 
    dim3 blocksPerGrid(ceil(m/16.0), ceil(n/16.0), 1);
    dim3 threadsPerBlock(16, 16, 1);
    pictureKernel<<<blocksPerGrid, threadsPerBlock>>> (d_a, m, n);
 
    // 3. copy the results back 
    cudaMemcpy(a, d_a, size, cudaMemcpyDeviceToHost);

    // check results 
    printf("\nNow print out the changed values: \n");
    for (i = 0; i < m; i++){
        for (j = 0; j < n; j++){
            printf("%.2f\t", a[index(i, j, n)]);
        }
    }
    printf("\n");
    free(a);
    cudaFree(d_a);
    return 0;
}

__global__ void pictureKernel(float * a, unsigned int m, unsigned int n){
    // kernel function to double every element of matrix a
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    if ((row < m) && (col < n))
        a[row * n + col] = 2 * a[row * n + col]; 
}
