/*
 * conv_vanilla.cu
 *
 *  Created on: Nov 29, 2018
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

__constant__ double D_F[ K * C * FW * FH * sizeof(double)];  // filter in device const memory

__global__ void convKernel(double * img, double * f, double * o);
__global__ void convTiledKernel(double * img, double * f, double * o);

double checksum3(double * i, unsigned int cnl, unsigned int row, unsigned int col){
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

double checksum4(double * f, unsigned int num, unsigned int cnl, unsigned int row, unsigned int col){
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
	/*----------------------- Exp 1: vanilla --------------------------------*/
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

	/*
	double checksum_f = checksum4(filter, K, C, FH, FW);
	printf("checksum filter %.2f\t", checksum_f);
	*/

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
	printf("checksum image %.2f\t", checksum_i);
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

/*--------------------- Exp 2 : tiling + const memory ------------------------*/

	double * out_t;  // tiled output
	out_t = (double*)calloc(num_o, sizeof(double));   // num_o is calculated when assigning kernels

	// assigning variables on cuda
	double * d_o_t;  // output on device, tiled
	cudaMalloc(& d_o_t, num_o * sizeof(double));

    // optimization 1: copy the filter from host to the const memory
    cudaMemcpyToSymbol(D_F, filter, num_f * sizeof(double));

    start = clock();  // begin timer

	// optimization 2: use tilting algorithms for convolution
	convTiledKernel<<<blocksPerGrid, threadsPerBlock>>>(d_i, D_F, d_o_t);  // calculation goes to CUDA

	end = clock();  // end timer
    time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;
//    printf("Time taken for tiled convolution is %lf seconds.\n", time_taken);

	// check errors
	cudaDeviceSynchronize();
    error = cudaGetLastError();
    if(error!=cudaSuccess)
    {
       fprintf(stderr,"ERROR: %s\n", cudaGetErrorString(error) );
       exit(-1);
    }
	// get results back from cuda
	cudaMemcpy(out_t, d_o_t, num_o * sizeof(double), cudaMemcpyDeviceToHost);

	// get checksum
	double checksum_o_t = checksum3(out, K, H, W);
//	printf("checksum output for tiled convolution %.2f\n", checksum_o_t);


	free(filter);
	free(img0);
	free(out);
	free(out_t);
	cudaFree(d_i);
	cudaFree(d_f);
	cudaFree(d_o);
	cudaFree(d_o_t);
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


// Kernel for tiled convolution
__global__ void convTiledKernel(double * img, double * f, double * o){
	// save all coordinates to save global memory access in future calculation
	unsigned int bidx = blockIdx.x;
	unsigned int bidy = blockIdx.y;
	unsigned int bidz = blockIdx.z;

	unsigned int bdimx = blockDim.x;
	unsigned int bdimy = blockDim.y;
	unsigned int bdimz = blockDim.z;

    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;
    unsigned int tz = threadIdx.z;

	unsigned int c, i, j;    // indices in the data tile

	// 1. move data into the tile. Employ 32 * 4 + 4 workers, everyone move three elements
	__shared__ double  ds_tile[TW * TW * C];

	// locate the tile in the data coordinate. Used to indexing in data
	unsigned int tile_begin_x_in_data_crd = bidx * bdimx - P + 1;
	unsigned int tile_begin_y_in_data_crd = bidy * bdimy - P + 1;
	unsigned int tile_begin_z_in_data_crd = 0;

	// Three types of data: halo boundaries, halo edges, and internal data
	// 1.1 move left halo boundaries, use the first column of threads
	if (ty == 0){
		// use all 32 threads in the first column to move data
		for (c = 0; c < C; c++){
			ds_tile[index3(c, tx+1, 0, TW, TW)] = img[index3(c, tile_begin_x_in_data_crd + tx, tile_begin_y_in_data_crd + 0, H0, W0)];
		}
	}

	// 1.2 move right halo boundaries, use the second column of threads
	if (ty == 1){
		for (c = 0; c < C; c++){
			ds_tile[index3(c, tx+1, TW-1, TW, TW)] = img[index3(c, tile_begin_x_in_data_crd + tx, tile_begin_y_in_data_crd + TW-1, H0, W0)];
		}
	}

	// 1.3 move upper halo boundaries, use the third column of threads
	if (ty == 2){
		for (c = 0; c < C; c++){
			ds_tile[index3(c, 0, tx+1, TW, TW)] = img[index3(c, tile_begin_x_in_data_crd + 0, tile_begin_y_in_data_crd + tx, H0, W0)];
		}
	}

	// 1.4 move lower halo boundaries, use the forth column of threas
	if (ty == 3){
		for (c = 0; c < C; c++){
			ds_tile[index3(c,TW-1, tx+1, TW, TW)] = img[index3(c, tile_begin_x_in_data_crd + TW - 1, tile_begin_y_in_data_crd + tx, H0, W0)];
		}
	}

	// 1.5 move all the edges
	if (ty == 4){
		// 1.5.1 the first thread in the fifth column deals with up left edge channels
		if (tx == 0){
			for (c = 0; c < C; c++){
				ds_tile[index3(c, 0, 0, TW, TW)] = img[index3(c, tile_begin_x_in_data_crd, tile_begin_y_in_data_crd, H0, W0)];
			}
		}
		// 1.5.2 the second thread in the fifth column deals with lower left edge channels
		if (tx == 1){
			for (c = 0; c < C; c++){
				ds_tile[index3(c, TW-1, 0, TW, TW)] = img[index3(c, tile_begin_x_in_data_crd + TW - 1, tile_begin_y_in_data_crd, H0, W0)];
			}
		}
		// 1.5.3 the third thread in the fifth column deals with up right edge channels
		if (tx == 2){
			for (c = 0; c < C; c++){
				ds_tile[index3(c, 0, TW-1, TW, TW)] = img[index3(c, tile_begin_x_in_data_crd, tile_begin_y_in_data_crd + TW - 1, H0, W0)];
			}
		}
		// 1.5.4 the forth thread in the fifth column deals with the lower right edge channels
		if (tx == 3){
			for (c = 0; c < C; c++){
				ds_tile[index3(c, TW-1, TW-1, TW, TW)] = img[index3(c, tile_begin_x_in_data_crd + TW - 1, tile_begin_y_in_data_crd + TW - 1, H0, W0)];
			}
		}
	}

	// 1.6 move all the internal elements... perfect match!
	for (c = 0; c < C; c++){
		ds_tile[index3(c, tx, ty, TW, TW)] = img[index3(c, tile_begin_x_in_data_crd + tx, tile_begin_y_in_data_crd + ty, H0, W0)];
	}

	__syncthreads();

	// 2. Perform convolution.
    unsigned int y = bidy * bdimy + ty;  // column
	unsigned int x = bidx * bdimx + tx;  // row
	unsigned int k = bidz * bdimz + tz;  // filter id

	for (c = 0; c < C; c++){
		for (j = 0; j < FH; j++){
			for (i = 0; i < FW; i++)
	            o[index3(k, x, y, H, W)] +=
				    f[index4(k, c, FW-1-i, FH-1-j, C, FH, FW)] *
					ds_tile[index3(c, tx + i, ty + j, TW, TW)];  // convolution step on the tile
		}
	}
}
