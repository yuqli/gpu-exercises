// 20181010
// Yuqiong Li
// Implement a convolutional neural net in cpp
#include <cuda.h>
#include <stdio.h>


__global__ void convKernel(float * a, float * mask, float * b, int m, int s);

int main() {
    int m, s;  // m is the length of a, s is the length of mask
    float * a, * mask;
    float * b;  // result array

    m = 5;
    s = 3;

    a = (float *) calloc(m, sizeof(float));
    b = (float *) calloc(m, sizeof(float));
    mask = (float *) calloc(s, sizeof(float));

    // initialize the array
    for (int i = 0; i < m; i++){
        a[i] = (float) i;
        b[i] = 0;
    }

    for (int i = 0; i < s; i++){
        mask[i] = (float) i / 1.34 + 0.32;
    }

    for (int i = 0; i < m; i++)
        printf("%.2f\t", a[i]);
    printf("\n");

    for (int i = 0; i < s; i++)
        printf("%.2f\t", mask[i]);
    printf("\n");

    /* CPU version */
    clock_t start, end;
    double time_taken;
    start = clock();
    for (int i = 0; i < m; i++){
        float buff = 0.0;
        int left = (i-s/2) >= 0 ? (i-s/2) : 0;  // left boundary
        int right = (i+s/2) <= (m-1) ? (i+s/2) : (m-1);   // right boundary
        for (int j = left; j <= right; j++){
            buff += a[j] * mask[j-left];
        }
        b[i] = buff;
    }
    end = clock();
    time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;
  
    printf("Time taken for CPU is %lf\n", time_taken);
  
    printf("Results for CPU :\n");
    for (int i = 0; i < m; i++)
        printf("%.2f\t", b[i]);
    printf("\n");

    // 1. allocate cuda memory
    float * d_a, * d_mask;
    float * temp;  // store results on host
    float * d_temp;

    // initiaze variables
    temp = (float*) calloc(m, sizeof(float));
    unsigned int sizeA = m * sizeof(float);
    unsigned int sizeM = s * sizeof(float);
    cudaMalloc(& d_a, sizeA);
    cudaMalloc(& d_mask, sizeM);
    cudaMalloc(& d_temp, sizeA);

    cudaMemcpy(d_a, a, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, mask, sizeM, cudaMemcpyHostToDevice);

    dim3 blocksPerGrid(ceil(m/16.0), 1, 1);
    dim3 threadsPerBlock(16, 1, 1); 
    start = clock();
    convKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_mask, d_temp, m, s);
    cudaMemcpy(temp, d_temp, sizeA, cudaMemcpyDeviceToHost);
    end = clock();
    time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;
 
 
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if(error!=cudaSuccess)
    {
        fprintf(stderr,"ERROR: %s\n", cudaGetErrorString(error) );
        exit(-1);
    }
 
    printf("Time taken for GPU is %lf\n", time_taken);

    printf("Results for GPU :\n");
    for (int i = 0; i < m; i++)
        printf("%.2f\t", temp[i]);
    printf("\n");

    free(a);
    free(b);
    free(mask);
    cudaFree(d_a);
    cudaFree(d_mask);
    cudaFree(d_temp);
    free(temp);
    return 0;
}


__global__ void convKernel(float * a, float * mask, float * b, int m, int s){
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    int left = i-(s/2) > 0? (i-(s/2)) : 0;
    int right = i+(s/2) < m? (i+(s/2)) : m-1;
    float res = 0.0;
    if (i < m){
        for (int k = left; k <= right; k++)
            res += a[k] * mask[k-left];
        b[i] = res;
    }
}
