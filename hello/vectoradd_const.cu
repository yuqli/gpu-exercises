// 20181201
// Yuqiong Li
// a basic CUDA function to test working with device constant memory
#include <stdio.h>
#include <cuda.h>

const unsigned int N = 10;    // size of vectors

__constant__ float const_d_a[N * sizeof(float)];  // filter in device const memory

// function declarations
__global__ void vecAddConstantKernel(float * b, float * c, unsigned int n);
__global__ void vecAddConstantKernel2(float * a, float * b, float * c, unsigned int n);

// main function
int main()
{
    float * a, * b, * c;  // a and b are vectors. c is the result
    a = (float *)calloc(N, sizeof(float));
    b = (float *)calloc(N, sizeof(float));

    /**************************** Exp 1: sequential ***************************/
    int i;
    int size = N * sizeof(float);
    float sum = 0;
    for (i = 0; i < N; i++){
        a[i] = (float)i / 0.23 + 1;
        b[i] = (float)i / 5.89 + 9;
        sum += a[i] + b[i];
    }

    c = (float*) malloc(size);
    printf("Results from host :%.2f\n", sum);

    /********************** Exp 2: CUDA w/o const mem *************************/
    // 1. allocate memory on CUDA
    float *  d_b, * d_c;   // device memory
    cudaError_t err2 = cudaMalloc((void **) & d_b, size);
    cudaError_t err3 = cudaMalloc((void **) & d_c, size);

    if (err2 != cudaSuccess){
        printf("%s in %s at line %d\n", cudaGetErrorString(err2), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
    if (err3 != cudaSuccess){
        printf("%s in %s at line %d\n", cudaGetErrorString(err3), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    // copy memory
    cudaError_t err4 = cudaMemcpyToSymbol(const_d_a, a, size);
    if (err4 != cudaSuccess){
        printf("%s in %s at line %d\n", cudaGetErrorString(err4), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // 2. operate on kernels
    vecAddConstantKernel<<<ceil(N/256.0), 256>>>(d_b, d_c, N);  // no need to pass const_d_a as a parameter as it's global

    // 3. copy the results back to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    float cuda_res = 0;
    for(i = 0; i < N; i++){
    //    printf("%.2f\t", c[i]);
        cuda_res += c[i];
    }

    printf("Results from device :%.2f\n", cuda_res);

    // 2. do it again but passing constant variable as a parameter
    float * d_c1;   // device memory
    cudaError_t err5 = cudaMalloc((void **) & d_c1, size);

    if (err5 != cudaSuccess){
        printf("%s in %s at line %d\n", cudaGetErrorString(err5), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    vecAddConstantKernel2<<<ceil(N/256.0), 256>>>(const_d_a, d_b, d_c1, N);  // no need to pass const_d_a as a parameter as it's global

    // 3. copy the results back to host
    cudaMemcpy(c, d_c1, size, cudaMemcpyDeviceToHost);

    cuda_res = 0;
    for(i = 0; i < N; i++){
    //    printf("%.2f\t", c[i]);
        cuda_res += c[i];
    }

    printf("Results from host but pass const var as parameter:%.2f\n", cuda_res);

    cudaFree(d_b);
    cudaFree(d_c);

    free(a);
    free(b);
    free(c);

    return 0;
}


__global__  void vecAddConstantKernel(float * b, float * c, unsigned int N){
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i<N)  c[i] = const_d_a[i] + b[i];
}

__global__  void vecAddConstantKernel2(float * a, float * b, float * c, unsigned int N){
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i<N)  c[i] = a[i] + b[i];
}
