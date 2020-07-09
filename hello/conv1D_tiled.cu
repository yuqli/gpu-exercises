// 20181130
// Yuqiong Li
// Implement a convolutional neural net in cpp
// Optimized for const memory and tilting
#include <cuda.h>
#include <stdio.h>

const unsigned int S = 3;    // mask size
__constant__ float d_mask[S];

__global__ void convKernel(float * d_a, float * ds_a_tile, float * d_mask, float * b, int m, int s, int tw);

int main() {
    int m, s;  // m is the length of a, s is the length of mask
    float * a, * mask;
    float * b;  // result array

    m = 6;
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
    float * d_a;
    float * temp;  // store results on host
    float * d_temp;  // store results on the device

    // initiaze variables
    temp = (float*) calloc(m, sizeof(float));
    unsigned int sizeA = m * sizeof(float);
    unsigned int sizeM = s * sizeof(float);
    cudaMalloc(& d_a, sizeA);
    cudaMalloc(& d_temp, sizeA);

    cudaMemcpy(d_a, a, sizeA, cudaMemcpyHostToDevice);
    // optimization 1: copy the mast to const memory
    cudaMemcpyToSymbol(d_mask, mask, sizeM);

    // optimization 2: tilting algorithms for convolution
    int tw = 2;   // tile width
    int tile_size = tw + s - 1;
    __shared__ float * ds_a_tile;
    cudaMalloc(& ds_a_tile, tile_size * sizeof(float));    // the tile size

    dim3 blocksPerGrid(ceil(m/(float)tw), 1, 1);  // block size = tile size which is 2
    dim3 threadsPerBlock(tw, 1, 1);

    start = clock();

    convKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, ds_a_tile, d_mask, d_temp, m, s, tw);

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


__global__ void convKernel(float * d_a, float * ds_a_tile, float * d_mask, float * d_b, int m, int s, int tw){
    /* Params:
    @ d_a: the original array. in device global memory
    @ ds_a_tile: the array tile that resides in the shared memory
    @ d_mask: the mast that resides in device constant memory
    @ d_b: the places to store results in the device global memory
    @ m: the data size
    @ s: the mask size
    @ tw: the tilding width
    rvalue: void. This function will populate the d_b array
    */
    int bid = blockIdx.x;  // block id
    int tid = threadIdx.x;  // thread id
    int bdim = blockDim.x;

    int arm = s/2;    // arm size of the mask

    // Now begin moving data. Three types of data: left halo, internal elements, right halo
    // 1. move left halo. Use the leftmost threads to move these elements
    int left_halo_data_begin = bid * bdim + 0 - arm;
    int left_halo_tile_begin = 0;
    if (tid < arm){
        if (left_halo_data_begin + tid < 0)
            ds_a_tile[left_halo_tile_begin + tid] = 0;    // the ghost elements
        else
            ds_a_tile[left_halo_tile_begin + tid] = d_a[left_halo_data_begin + tid];  // halo elements
    }

    // 2. move the internal elements. This should match exactly the number of threads and everyone should get one
    int left_internal_data_begin = left_halo_data_begin + arm;
    int left_internal_tile_begin = left_halo_tile_begin + arm;
    ds_a_tile[left_internal_tile_begin + tid] = d_a[left_internal_data_begin + tid];

    // 3. move the right halo elements. use the rightmost threads to move them so branch divergence is minimized
    int right_halo_data_begin = bid * bdim + bdim;
    int right_halo_tile_begin = arm + bdim;
    if (tid > bdim - 1 - arm){
        if (tid + right_halo_data_begin > m-1)
            ds_a_tile[right_halo_tile_begin + tid] = 0;    // ghost element
        else
            ds_a_tile[right_halo_tile_begin + tid] = d_a[right_halo_data_begin + tid];
    }

    __syncthreads();

    // Finished copying all elements. Now do convolution
    float res = 0.0;
    for (int k = 0; k < s; k++)
        res += ds_a_tile[tid + k] * d_mask[k];

    // Finished calculating convolution results. copy to the result vector
    int oid = bid * bdim + tid;  // this thread's position in the output vector
    d_b[oid] = res;
}
