// 20181201
// Yuqiong Li
// a basic CUDA function to test working with device constant memory
#include <stdio.h>
#include <cuda.h>

const unsigned int N = 10;    // size of vectors

__constant__ float const_d_a[N];  // filter in device const memory

int main()
{
    float * a, * b;  // a and b are vectors. c is the result
    a = (float *)calloc(N, sizeof(float));
    b = (float *)calloc(N, sizeof(float));

    /**************************** Exp 1: sequential ***************************/
    int i;
    int size = N * sizeof(float);
    for (i = 0; i < N; i++){
        a[i] = (float)i / 0.23 + 1;
    }


    // 1. copy a to constant memory
    cudaError_t err = cudaMemcpyToSymbol(const_d_a, a, size);
    if (err != cudaSuccess){
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    cudaError_t err2 = cudaMemcpyFromSymbol(b, const_d_a, size);
    if (err2 != cudaSuccess){
        printf("%s in %s at line %d\n", cudaGetErrorString(err2), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    double checksum0, checksum1;
    for (i = 0; i < N; i++){
        checksum0 += a[i];
        checksum1 += b[i];
    }

    printf("Checksum for elements in host memory is %f\n.", checksum0);
    printf("Checksum for elements in constant memory is %f\n.", checksum1);

    return 0;
}
