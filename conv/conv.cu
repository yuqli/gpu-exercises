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
#include <cudnn.h>
#include <iostream>

/* To index element (c, x, y) of a 3D array stored as 1D */
// x is row index, y is column index, c is channel index
#define index3(c, x, y, H, W) ((c)*(H)*(W)) + (x)*(W) + (y)
#define index4(k, c, i, j, C, H, W) ((k)*(C)*(H)*(W)) + ((c)*(H)*(W)) + ((i)*(W))+ (j)

// a macro to check cudnn routine status
#define checkCUDNN(expression)                                  	\
  {																	\
	  cudnnStatus_t status = (expression);							\
	  if (status != CUDNN_STATUS_SUCCESS){							\
		  std::cerr << "Error on line " << __LINE__ << ": " 		\
			   		<< cudnnGetErrorString(status) << std::endl;  	\
		  std::exit(EXIT_FAILURE);									\
	  }																\
  }																	\

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
__global__ void convTiledKernel(double * img,  double * o);  // constant memory variable is no longer passed as argument as it's global

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


	// initialize original image to be used by cuDNN
	double * img;
	unsigned int num_i_no_padding = C * H * W;  // number of elements in the image
	img = (double*) calloc(num_i_no_padding, sizeof(double));
	if( !img)
	{
	    fprintf(stderr, " Cannot allocate the image array\n");
	    exit(1);
	}

	int x, y;
	for (c = 0; c < C; c++){  // reuse previous indexer c
		// 1. initialize the matrix in the middle
		for (x = 0; x < H; x++){
			for (y = 0; y < W; y++){
			    img[index3(c, x, y, H, W)] = c * (x + y);
	            // printf("image pixel at (%d, %d, %d) is %.2f\n", c, x, y, img0[index3(c, x, y, H0, W0)]);
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
	convTiledKernel<<<blocksPerGrid, threadsPerBlock>>>(d_i, d_o_t);  // calculation goes to CUDA

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
	double checksum_o_t = checksum3(out_t, K, H, W);
	printf("%4.6lf, %4.6lf\n", checksum_o_t, time_taken);


/*---------------------------- Exp 3 : cuDNN ---------------------------------*/

    cudnnHandle_t cudnn;
	checkCUDNN(cudnnCreate(&cudnn));

	cudnnTensorDescriptor_t input_descriptor;
	checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
	checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor, CUDNN_TENSOR_NCHW, \
										 CUDNN_DATA_DOUBLE, 1, C, H, W));

	cudnnTensorDescriptor_t output_descriptor;
	checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
	checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor, CUDNN_TENSOR_NCHW, \
										 CUDNN_DATA_DOUBLE, 1, K, H, W));

	cudnnFilterDescriptor_t kernel_descriptor;
	checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
	checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor, CUDNN_DATA_DOUBLE, \
										  CUDNN_TENSOR_NCHW, K, C, FH, FW));

    cudnnConvolutionDescriptor_t convolution_descriptor;
	checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
	checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor,
	                                           P, P, 1, 1, 1, 1, \
										       CUDNN_CONVOLUTION, CUDNN_DATA_DOUBLE));

	cudnnConvolutionFwdAlgo_t convolution_algorithm;
	checkCUDNN(cudnnGetConvolutionForwardAlgorithm(cudnn,
		                                           input_descriptor,
											       kernel_descriptor,
										           convolution_descriptor,
											       output_descriptor,
											       CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
											       0,
											       &convolution_algorithm));

	size_t workspace_bytes = 0;
	checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
	                                                   input_descriptor,
												       kernel_descriptor,
												       convolution_descriptor,
												       output_descriptor,
												       convolution_algorithm,
												       &workspace_bytes));
//    std::cerr << "Workspace size: " << (workspace_bytes / 1048576.0) << "MB\n";

	size_t * d_workspace;
	cudaMalloc(&d_workspace, workspace_bytes);

    // allocate input
	double * d_input;
	cudaMalloc(&d_input, num_i_no_padding * sizeof(double));
	cudaMemcpy(d_input, img, num_i_no_padding * sizeof(double), cudaMemcpyHostToDevice);   // copy image to device

    error = cudaGetLastError();
    if(error!=cudaSuccess)
    {
        fprintf(stderr,"ERROR: %s\n", cudaGetErrorString(error) );
        exit(-1);
    }

//    double checksum_img = checksum3(img, C, H, W);
//    printf("checksum for image is %.4f\n", checksum_img);

	double * check_img;
	check_img = (double*)calloc(num_i_no_padding, sizeof(double));   // num_o is calculated when assigning kernels
	cudaMemcpy(check_img, d_input, num_i_no_padding * sizeof(double), cudaMemcpyDeviceToHost);

    error = cudaGetLastError();
    if(error!=cudaSuccess)
    {
        fprintf(stderr,"ERROR: %s\n", cudaGetErrorString(error) );
        exit(-1);
    }

	double checksum_img_o = checksum3(check_img, C, H, W);
//	printf("%4.6lf\n", checksum_img_o);

	double * check_d_f;
	check_d_f = (double*)calloc(num_f, sizeof(double));   // num_o is calculated when assigning kernels
	cudaMemcpy(check_d_f, d_f, num_f * sizeof(double), cudaMemcpyDeviceToHost);

    error = cudaGetLastError();
    if(error!=cudaSuccess)
    {
        fprintf(stderr,"ERROR: %s\n", cudaGetErrorString(error) );
        exit(-1);
    }

//	double checksum_d_f = checksum4(check_d_f, K, C, FH, FW);
//	printf("%4.6lf\n", checksum_d_f);

    // allocate output
	double * d_output;
	cudaMalloc(&d_output, num_o * sizeof(double));
	cudaMemset(d_output, 0, num_o* sizeof(double));

    error = cudaGetLastError();
    if(error!=cudaSuccess)
    {
        fprintf(stderr,"ERROR: %s\n", cudaGetErrorString(error) );
        exit(-1);
    }

	// kernel is already in memoery d_f
	const double alpha = 1, beta = 0;
    start = clock();  // begin timer
	checkCUDNN(cudnnConvolutionForward(cudnn,
	                                   &alpha,
								       input_descriptor,
								       d_input,
								       kernel_descriptor,
								       d_f,
								       convolution_descriptor,
								       convolution_algorithm,
								       d_workspace,
								       workspace_bytes,
								       &beta,
								       output_descriptor,
								       d_output));

	end = clock();  // end timer
    time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;

	double * out_cudnn;
	out_cudnn = (double*)calloc(num_o, sizeof(double));   // num_o is calculated when assigning kernels
	cudaMemcpy(out_cudnn, d_output, num_o * sizeof(double), cudaMemcpyDeviceToHost);

    error = cudaGetLastError();
    if(error!=cudaSuccess)
    {
        fprintf(stderr,"ERROR: %s\n", cudaGetErrorString(error) );
        exit(-1);
    }

	double checksum_o_cudnn = checksum3(out_cudnn, K, H, W);
	printf("%4.6lf, %4.6lf\n", checksum_o_cudnn, time_taken);


	free(filter);
	free(img0);
	free(out);
	free(out_t);

	cudaFree(d_i);
	cudaFree(d_f);
	cudaFree(d_o);
	cudaFree(d_o_t);

	cudaFree(d_input);
	cudaFree(d_output);
	cudaFree(d_workspace);

	cudnnDestroyTensorDescriptor(input_descriptor);
	cudnnDestroyTensorDescriptor(output_descriptor);
	cudnnDestroyFilterDescriptor(kernel_descriptor);
	cudnnDestroyConvolutionDescriptor(convolution_descriptor);

	cudnnDestroy(cudnn);

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
__global__ void convTiledKernel(double * img,  double * o){
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
	__shared__ double ds_tile[C * TW * TW];

	// locate the tile in the data coordinate. Used to indexing in data
	unsigned int tile_begin_x_in_data_crd = bidx * bdimx - P + 1;
	unsigned int tile_begin_y_in_data_crd = bidy * bdimy - P + 1;

	// Three types of data: halo boundaries, halo edges, and internal data
	// 1.1 move left halo boundaries, use the first column of threads
	if (ty == 0){
		// use all 32 threads in the first column to move data
		for (c = 0; c < C; c++){
			ds_tile[index3(c, tx+1, 0, TW, TW)] = img[index3(c, tile_begin_x_in_data_crd + tx + 1, tile_begin_y_in_data_crd + 0, H0, W0)];
//		    printf("Tiled image position(%d, %d, %d) is %.2f\n", c, tx+1, 0, ds_tile[index3(c, tx+1, 0, TW, TW)]);
		}
	}
//	printf("Left halo complete.\n");

	// 1.2 move right halo boundaries, use the second column of threads
	if (ty == 1){
		for (c = 0; c < C; c++){
			ds_tile[index3(c, tx+1, TW-1, TW, TW)] = img[index3(c, tile_begin_x_in_data_crd + tx + 1, tile_begin_y_in_data_crd + TW-1, H0, W0)];
//		    printf("Tiled image position(%d, %d, %d) is %.2f\n", c, tx+1, TW-1, ds_tile[index3(c, tx+1, TW-1, TW, TW)]);
		}
	}
//	printf("Right halo complete.\n");

	// 1.3 move upper halo boundaries, use the third column of threads
	if (ty == 2){
		for (c = 0; c < C; c++){
			ds_tile[index3(c, 0, tx+1, TW, TW)] = img[index3(c, tile_begin_x_in_data_crd + 0, tile_begin_y_in_data_crd + tx + 1, H0, W0)];
//		    printf("Tiled image position(%d, %d, %d) is %.2f\n", c, 0, tx+1, ds_tile[index3(c, 0, tx+1, TW, TW)]);
		}
	}
//	printf("Upper halo complete.\n");

	// 1.4 move lower halo boundaries, use the forth column of threas
	if (ty == 3){
		for (c = 0; c < C; c++){
			ds_tile[index3(c,TW-1, tx+1, TW, TW)] = img[index3(c, tile_begin_x_in_data_crd + TW - 1, tile_begin_y_in_data_crd + tx + 1, H0, W0)];
//		    printf("Tiled image position(%d, %d, %d) is %.2f\n", c, TW-1, tx+1, ds_tile[index3(c, TW-1, tx+1, TW, TW)]);
		}
	}
//	printf("Lower halo complete.\n");

	// 1.5 move all the edges
	if (ty == 4){
		// 1.5.1 the first thread in the fifth column deals with up left edge channels
		if (tx == 0){
			for (c = 0; c < C; c++){
				ds_tile[index3(c, 0, 0, TW, TW)] = img[index3(c, tile_begin_x_in_data_crd, tile_begin_y_in_data_crd, H0, W0)];
//		    	printf("Tiled image position(%d, %d, %d) is %.2f\n", c, 0, 0, ds_tile[index3(c, 0, 0, TW, TW)]);
			}
		}
//		printf("Upper left complete.\n");
		// 1.5.2 the second thread in the fifth column deals with lower left edge channels
		if (tx == 1){
			for (c = 0; c < C; c++){
				ds_tile[index3(c, TW-1, 0, TW, TW)] = img[index3(c, tile_begin_x_in_data_crd + TW - 1, tile_begin_y_in_data_crd, H0, W0)];
//		    	printf("Tiled image position(%d, %d, %d) is %.2f\n", c, TW-1, 0, ds_tile[index3(c, TW-1, 0, TW, TW)]);
			}
		}
//		printf("Lower left complete.\n");
		// 1.5.3 the third thread in the fifth column deals with up right edge channels
		if (tx == 2){
			for (c = 0; c < C; c++){
				ds_tile[index3(c, 0, TW-1, TW, TW)] = img[index3(c, tile_begin_x_in_data_crd, tile_begin_y_in_data_crd + TW - 1, H0, W0)];
//		    	printf("Tiled image position(%d, %d, %d) is %.2f\n", c, 0, TW-1, ds_tile[index3(c, 0, TW-1, TW, TW)]);
			}
		}
//		printf("upper right complete.\n");
		// 1.5.4 the forth thread in the fifth column deals with the lower right edge channels
		if (tx == 3){
			for (c = 0; c < C; c++){
				ds_tile[index3(c, TW-1, TW-1, TW, TW)] = img[index3(c, tile_begin_x_in_data_crd + TW - 1, tile_begin_y_in_data_crd + TW - 1, H0, W0)];
//		    	printf("Tiled image position(%d, %d, %d) is %.2f\n", c, TW-1, TW-1, ds_tile[index3(c, TW-1, TW-1, TW, TW)]);
			}
		}
//		printf("Lower right complete.\n");
	}

	// 1.6 move all the internal elements... perfect match!
	for (c = 0; c < C; c++){
//		printf("Now copying (%d, %d, %d) in the tile.\n", c, tx, ty);
		ds_tile[index3(c, tx+1, ty+1, TW, TW)] = img[index3(c, tile_begin_x_in_data_crd + tx + 1, tile_begin_y_in_data_crd + ty + 1, H0, W0)];
    	// printf("Tiled image position(%d, %d, %d) is %.2f\n", c, tx+1, ty+1, ds_tile[index3(c, tx+1, ty+1, TW, TW)]);
	}

	__syncthreads();


    // now check if the copying is successful...
//	double checksum_tile = checksum3(ds_tile, C, TW, TW);
//	printf("The checksum for tile is %f\n", checksum_tile);
//	printf("The size of the images is %d.\n", sizeof(ds_tile));

	// 2. Perform convolution.
//	printf("The dimension of data tile is (%d, %d, %d).\n", C, TW, TW);
//	printf("The number of elements in data tile is (%d).\n", sizeof(ds_tile)/sizeof(double));
	unsigned int x = bidx * bdimx + tx;  // row
    unsigned int y = bidy * bdimy + ty;  // column
	unsigned int k = bidz * bdimz + tz;  // filter id

	for (c = 0; c < C; c++){
		for (j = 0; j < FW; j++){
			for (i = 0; i < FH; i++){
//				printf("The element at filter position (%d, %d, %d, %d) is %d.\n",
//				        k, c, FW-1-i, FW-1-j,
//				        D_F[index4(k, c, FW-1-i, FH-1-j, C, FH, FW)]);
//				printf("The element at tile position (%d, %d, %d) is %d.\n", c, tx + i, ty + j,
//			           ds_tile[index3(c, tx + i, ty + j, TW, TW)]);
	            o[index3(k, x, y, H, W)] +=
				    D_F[index4(k, c, FH-1-i, FW-1-j, C, FH, FW)] *
					ds_tile[index3(c, tx + i, ty + j, TW, TW)];  // convolution step on the tile
//				printf("The element at output position (%d, %d, %d) is %d.\n", k, x, y, o[index3(k, x, y, H, W)]);
			}
		}
	}
}
