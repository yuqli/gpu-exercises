// 20181010
// Yuqiong Li
// a basic CUDA function to familiarize with usage
#include<stdio.h>
#include<cuda.h>

// function declarations 
__global__ void vecAddKernel(float * a, float * b, float * c, unsigned int N);

// main function 
int main()
{   
    int N = 10;    // length of vector 
    float * a, * b, * c;  // a and b are vectors. c is the result
    unsigned int size = N * sizeof(float);  // number of bytes to allocate 
    a = (float *)calloc(N, sizeof(float));
    b = (float *)calloc(N, sizeof(float));

    int i = 0;
    float sum = 0;
    for (i = 0; i < N; i++){
        a[i] = (float)i / 0.23 + 1;
        b[i] = (float)i / 5.89 + 9;
        sum += a[i] + b[i];
    }

    c = (float*) malloc(size);
 
    // 1. allocate memory on CUDA
    float * d_a, * d_b, * d_c;   // device memory 
    cudaMalloc((void **) & d_a, size);
    cudaMalloc((void **) & d_b, size);
    cudaMalloc((void **) & d_c, size);
     
    // copy memory 
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // 2. operate on kernels 
    vecAddKernel<<<ceil(N/256.0), 256>>>(d_a, d_b, d_c, N);

    // 3. copy the results back to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if(error!=cudaSuccess)
    {
        fprintf(stderr,"ERROR: %s\n", cudaGetErrorString(error) );
        exit(-1);
    }
 
    float cuda_res = 0;
    for(i = 0; i < N; i++){
        printf("%.2f\t", c[i]);
        cuda_res += c[i];
    }
 
    printf("Results from host :%.2f\n", sum);
    printf("Results from device:%.2f\n", cuda_res);

    cudaFree(d_a); 
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}

__global__
void vecAddKernel(float * a, float * b, float * c, unsigned int N){
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i<N)  c[i] = a[i] + b[i];
}
