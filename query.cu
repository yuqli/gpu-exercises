// a toy program to get device property

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

int main(){
    int dev_count;
    cudaGetDeviceCount(&dev_count);
    printf("%d\n", dev_count);
    
    cudaDeviceProp prop;
    int i;
    for (i = 0; i < dev_count; i++){
        cudaGetDeviceProperties(&prop,i);
        printf("Name: %s\n", prop.name);
        printf("SM count: %d\n", prop.multiProcessorCount);
        printf("Max threads per SM: %d\n", prop.maxThreadsPerBlock);
        printf("Max threads dim x: %d\n", prop.maxThreadsDim[0]);
        printf("Max threads dim y: %d\n", prop.maxThreadsDim[1]);
        printf("Max threads dim z: %d\n", prop.maxThreadsDim[2]);
        printf("Number of warps: %d\n", prop.warpSize);
    }
    cudaDeviceSynchronize();

    cudaError_t error = cudaGetLastError();
    if(error!=cudaSuccess)
    {
        fprintf(stderr,"ERROR: %s\n", cudaGetErrorString(error) );
        exit(-1);
    }
    return 0;
}
