/*
 * conv_derivative.cu
 *
 *  Created on: Dec 02, 2018
 *      Author: yuqiong
 */

#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

/* To index element (c, x, y) of a 3D array stored as 1D */
// x is row index, y is column index, c is channel index
#define index3(c, x, y, H, W) ((c)*(H)*(W)) + (x)*(W) + (y)
#define index4(k, c, i, j, C, H, W) ((k)*(C)*(H)*(W)) + ((c)*(H)*(W)) + ((i)*(W))+ (j)
#define index6(k, x, y, c, i, j, H, W, C, FH, FW) ((k)*(H)*(W)*(C)*(FH)*(FW)) + \
                                                  ((x)*(W)*(C)*(FH)*(FW)) + ((y)*(C)*(FH)*(FW)) + \
												  ((c)*(FH)*(FW)) + ((i)*(FW)) + (j)

// global variables
const unsigned int H = 4096;
const unsigned int W = 4096;
const unsigned int C = 3;
const unsigned int FW = 3;
const unsigned int FH = 3;
const unsigned int K = 10;
const unsigned int P = 1;
const unsigned int H0 = H + 2 * P;
const unsigned int W0 = W + 2 * P;

const unsigned int BDIM = 32;    // block dimension
const unsigned int TW = BDIM + FW - 1;  // tile width. Assume filter width is odd

__constant__ double D_F[K * C * FW * FH];  // filter in device const memory

__global__ void convKernel(double * img, double * f, double * o);
__global__ void convDevKernel(double * o, double * f, double * de);

__host__ __device__ double checksum3(double * i, unsigned int cnl, unsigned int row, unsigned int col){
	// @i: image which is a 3D image represented as an array
	// @cnl: number of channels
	// @row: number of rows
	// @col: number of columns
	// rvalue: the checksum of a 3D image
	unsigned int c, x, y;
	double sum = 0.0;
	for (c = 0; c < cnl; c++){
		for (x = 0; x < row; x++){
			for (y = 0; y < col; y++)
			    sum += i[index3(c, x, y, row, col)];
		}
	}
	return sum;
}

__host__ __device__ double checksum4(double * f, unsigned int num, unsigned int cnl, unsigned int row, unsigned int col){
	// @f: f which is a 4D set of filters represented as an 1D array
	// @num: number of filters
	// @cnl: number of channels
	// @row: number of rows
	// @col: number of columns
	// rvalue: the checksum of a 3D image
	unsigned int k, c, i, j;
	double sum = 0.0;
	for (k = 0; k < num; k++){
		for (c = 0; c < cnl; c++){
			for (i = 0; i < row; i++){
				for (j = 0; j < col; j++)
				    sum += f[index4(k, c, i, j, cnl, row, col)];
			}
		}
	}
	return sum;
}


int main()
{
	// initialize a set of filters
	double * filter;
	unsigned int num_f = K * C * FH * FW;  // number of elements in the filter
	filter = (double*)calloc(num_f, sizeof(double));
	if( !filter )
	{
	    fprintf(stderr, " Cannot allocate the set of filters!\n");
	    exit(1);
	}

	int k, c, i, j;
	for (k = 0; k < K; k++){
		for (c = 0; c < C; c++){
			for (i = 0; i < FH; i++){
				for (j = 0; j < FW; j++){
					/*
					printf("%d\t%d\n", k, K);
					printf("%d\t%d\n", c, C);
					printf("%d\t%d\n", i, FH);
					printf("%d\t%d\n", j, FW);
					printf("%f\n", (double) ((c+k)*(i+j)));
					*/
					filter[index4(k, c, i, j, C, FH, FW)] = (double) ((c+k)*(i+j));
					// printf("%f\n", filter[index4(k, c, i, j, C, FH, FW)]);
				}
			}
		}
	}


	// initialize padded image
	double * img0;
	unsigned int num_i = C * H0 * W0;  // number of elements in the image
	img0 = (double*) calloc(num_i, sizeof(double));
	if( !img0 )
	{
	    fprintf(stderr, " Cannot allocate the image array\n");
	    exit(1);
	}

	int x, y;
	for (c = 0; c < C; c++){  // reuse previous indexer c
		// 1. initialize four padded vectors
		for (y = 0; y < W0; y++){
		    img0[index3(c, 0, y, H0, W0)] = 0;
			img0[index3(c, H0-1, y, H0, W0)] = 0;
		}
		for (x = 1; x < H0-1; x++){
		    img0[index3(c, x, 0, H0, W0)] = 0;
		    img0[index3(c, x, W0-1, H0, W0)] = 0;
		}
		// 2. initialize the matrix in the middle
		for (x = 1; x < H0-1; x++){
			for (y = 1; y < W0-1; y++){
			    img0[index3(c, x, y, H0, W0)] = c * (x-1 + y-1);
	            // printf("image pixel at (%d, %d, %d) is %.2f\n", c, x, y, img0[index3(c, x, y, H0, W0)]);
			}
		}
	}

/*
	double checksum_i = checksum3(img0, C, H0, W0);
	printf("checksum image %.2f\n", checksum_i);

	printf("the size of the image is %d\n.", sizeof(img0));
	for (c = 0; c < C; c++){
		for (x = 0; x < H0; x++){
			for (y = 0; y < W0; y++)
			    printf("Image position(%d, %d, %d) is %.2f\n", c, x, y, img0[index3(c, x, y, H0, W0)]);
		}
	}
*/

	double * out;
	unsigned int num_o = K * H * W;  // number of elements in the output
	out = (double*)calloc(num_o, sizeof(double));   // num_o is calculated when assigning kernels

	// assigning variables on cuda
	double * d_i, * d_f, * d_o;    // image, filter and output

	cudaMalloc(& d_i, num_i * sizeof(double));
	cudaMalloc(& d_f, num_f * sizeof(double));
	cudaMalloc(& d_o, num_o * sizeof(double));

	cudaMemcpy(d_i, img0, num_i * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_f, filter, num_f * sizeof(double), cudaMemcpyHostToDevice);

	// assigning grids and blocks
	dim3 blocksPerGrid(ceil(H/(double)(BDIM)), ceil(W/(double)(BDIM)), K);
	dim3 threadsPerBlock(BDIM, BDIM, 1);    // 1024 threads per block and square, single output channel

    clock_t start, end;
    double time_taken;
    start = clock();  // begin timer

	convKernel<<<blocksPerGrid, threadsPerBlock>>>(d_i, d_f, d_o);  // calculation goes to CUDA

	end = clock();  // end timer
    time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;
    // printf("Time taken for vanilla convolution is %lf seconds.\n", time_taken);

	// check errors
	cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if(error!=cudaSuccess)
    {
       fprintf(stderr,"ERROR: %s\n", cudaGetErrorString(error) );
       exit(-1);
    }
	// get results back from cuda
	cudaMemcpy(out, d_o, num_o * sizeof(double), cudaMemcpyDeviceToHost);

	// get checksum
	double checksum_o = checksum3(out, K, H, W);
	printf("%4.6lf, %4.6lf\n", checksum_o, time_taken);


	free(filter);
	free(img0);
	free(out);
	cudaFree(d_i);
	cudaFree(d_f);
	cudaFree(d_o);
	return 0;
}


__global__ void convKernel(double * img, double * f, double * o){
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;  // column
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;  // row
	unsigned int k = blockIdx.z * blockDim.z + threadIdx.z;  // filter id

	unsigned int c, i, j;
	for (c = 0; c < C; c++){
		for (j = 0; j < FH; j++){
			for (i = 0; i < FW; i++)
	            o[index3(k, x, y, H, W)] +=
				    f[index4(k, c, FW-1-i, FH-1-j, C, FH, FW)] *
					img[index3(c, x+i, y+j, H0, W0)];  // convolution step
		}
	}
}


__global__ void convDevKernel(double * o, double * f, double * de){
	// calculate derivatives for a output w.r.t filter weights
	// rvalue: de is a 6D matrix. first two dimension is w.r.t output, second three is w.r.t. filer dimension

    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;  // column
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;  // row
	unsigned int k = blockIdx.z * blockDim.z + threadIdx.z;  // filter id

	unsigned int c, i, j;
	for (c = 0; c < C; c++){
		for (j = 0; j < FH; j++){
			for (i = 0; i < FW; i++)
	            de[index6(k, x, y, c, i, j, H, W, C, FH, FW)] =
					img[index3(c, x+i, y+j, H0, W0)];  // convolution step
		}
	}
}
