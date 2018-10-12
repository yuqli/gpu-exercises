// 20181010
// Yuqiong Li
// Matrix multiplication with CUDA
#include <stdlib.h>
#include <cuda.h>
#include <time.h>
#include <stdio.h>

#define index(i, j, n) ((i) * (n) + (j))

// declare global kernel function 
__global__ void matrixMulKernel(float * a, float * b, float * c, unsigned int m, unsigned int n, unsigned int r);

int main(){
    unsigned int m = 2000, n = 2000, r = 1000;  // dimensions
    float * a, * b, * c, *temp ;  // declare matrices

    a = (float *) malloc(m * n * sizeof(float));  // a is m by n
    b = (float *) malloc(n * r * sizeof(float));  // b is n by r
    c = (float *) calloc(m * r, sizeof(float));  // c is m by r : the result matrix
    temp = (float *) calloc(m * r, sizeof(float));  // to store GPU results
    int i = 0, j = 0;
    // initializing a
    for (i = 0; i < m; i++){
        for (j = 0; j < n; j++)
            a[index(i, j, n)] = i + j;
    }
   // initializing b
    for (i = 0; i < n; i++){
        for (j = 0; j < r; j++)
            b[index(i, j, r)] = i + j + 1;
    }

    double time_taken;
    clock_t start, end;

    // CPU version
    start = clock();    
    int k = 0; 
    for (i = 0; i < m; i++){
        for (j = 0; j < r; j++){
            for (k = 0; k < n; k++)
		c[index(i, j, r)] += a[index(i, k, n)] * b[index(k, j, r)];
        }
    }
    end = clock();
    time_taken = (double) (end - start) / CLOCKS_PER_SEC;
    printf("Time taken for CPU is %.2f.\n", time_taken);

    float val = 0.0;
    for (i = 0; i < m; i++){
        for (j = 0; j < r; j++){
            val += c[index(i, j, r)];
        }
    }
    printf("Check value for CPU: sum is %.2f\n.", val);

    // 1. allocate device memory for cuda variables 
    float * d_a, * d_b, * d_c;
    cudaMalloc((void **) &d_a, m * n * sizeof(float));
    cudaMalloc((void **) &d_b, n * r * sizeof(float));
    cudaMalloc((void **) &d_c, m * r * sizeof(float));

    // copy memory to device 
    cudaMemcpy(d_a, a, m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n * r * sizeof(float), cudaMemcpyHostToDevice);

    // 2. invoke kernel function
    dim3 blocksPerGrid(ceil(m/16.0), ceil(r/16.0), 1);
    dim3 threadsPerBlock(16, 16, 1);
    start = clock();
    matrixMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, m, n, r);    
    end = clock();
    time_taken = (double) (end - start)/ CLOCKS_PER_SEC;
    printf("Time taken for GPU is %.2f\n", time_taken);

    
    // 3. copy results to device 
    cudaMemcpy(temp, d_c, m * r * sizeof(float), cudaMemcpyDeviceToHost);

    val = 0;
    for (i = 0; i < m; i++){
        for (j = 0; j < r; j++){
            val += temp[index(i, j, r)];
        }
    }
    
    printf("Check value for GPU: sum is %.2f\n", val);

    free(a);
    free(b);
    free(c);
    free(temp);
    cudaFree(d_c);
    cudaFree(d_a);
    cudaFree(d_b);
    return 0;
}


__global__ void matrixMulKernel(float * a, float * b, float * c, unsigned int m, unsigned int n, unsigned int r){
    // a function to perform matrix multiplication
    // a is m by n; b is n by r; c is the result m by r
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if ((row < m) && (col < r)){
	float pvalue = 0;
	int k = 0;
	for (k = 0; k < n; k++){
            pvalue += a[index(row, k, n)] * b[index(k, col, r)];
	}
	c[index(row, col, r)] = pvalue;
    }
}

