// a toy program to get device property

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

int main(){
    int dev_count;
    cudaGetDeviceCount(&dev_count);
    printf("Number of CUDA devices: %d\n", dev_count);

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
        printf("Number of threads in a warp: %d\n", prop.warpSize);
        printf("Max memory (GB) on this device: %d\n", (int)(prop.totalGlobalMem * pow(10, -9)));
        printf("Max shared memory (KB) per block: %d\n", (int)(prop.sharedMemPerBlock * pow(10, -3)));
        printf("Total constant memory (KB): %d\n", (int) (prop.totalConstMem * pow(10, -3)));
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
