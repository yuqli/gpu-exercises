/*
 * Assume image is already loaded
 */

#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <fstream>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

/* To index element (c, x, y) of a 3D array stored as 1D */
// x is row index, y is column index, c is channel index
#define index3(c, x, y, H, W) ((c)*(H)*(W)) + (x)*(W) + (y)
#define index4(k, c, i, j, C, H, W) ((k)*(C)*(H)*(W)) + ((c)*(H)*(W)) + ((i)*(W))+ (j)
#define index6(k, x, y, c, i, j, H, W, C, FH, FW) ((k)*(H)*(W)*(C)*(FH)*(FW)) + \
                                               ((x)*(W)*(C)*(FH)*(FW)) + ((y)*(C)*(FH)*(FW)) + \
												  ((c)*(FH)*(FW)) + ((i)*(FW)) + (j)

// global variables
const unsigned int H = 32;
const unsigned int W = 32;
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


__host__ double * fc(double * w, double * o, double * b, unsigned int num, unsigned int row, unsigned int col){
    // implement a fully connected layer
    // @ w: weights. same number of elements as the output and can be organized as 3D
    // @ o: output of convolution layer
    // @ b: bias. scalar
    // @ num: number of filters. First dimension of o
    // @ row: number of rows. Second dimension of o
    // @ col: number of columns. Third dimension of o
    // rvalue: a scalar
    unsigned int k, x, y;
    double sum = 0.0;
    for (k = 0; k < num; k++){
        for (x = 0; x < row; x++){
            for (y = 0; y < col; y++)
                sum += w[index3(k, x, y, row, col)] * o[index3(k, x, y, row, col)];
        }
    }
    sum += *b;
    return sum;
}

__host__ double * fc_d_w(double * w, double * o, double * b, unsigned int num, unsigned int row, unsigned int col){
    // calculate the derivatives of fully connected layer resuls w.r.t. the weight matrix
    // @ w: weights. same number of elements as the output and can be organized as 3D
    // @ o: output of convolution layer
    // @ b: bias. scalar
    // @ num: number of filters. First dimension of o
    // @ row: number of rows. Second dimension of o
    // @ col: number of columns. Third dimension of o
    return o;
}

__host__ double fc_d_b(double * w, double * o, double * b, unsigned int num, unsigned int row, unsigned int col){
    // calculate the derivatives of fully connected layer resuls w.r.t. the bias
    // @ w: weights. same number of elements as the output and can be organized as 3D
    // @ o: output of convolution layer
    // @ b: bias. scalar
    // @ num: number of filters. First dimension of o
    // @ row: number of rows. Second dimension of o
    // @ col: number of columns. Third dimension of o
    return 1.0;
}

__host__ double sigmoid(double x){
    // calculate the result of a sigmoid layer
    // @ x: input
    return 1 / (1 + exp(x));
}

__host__ double sigmoid_d(double x){
    // calculate the result of a sigmoid layer
    // @ x: input
    return exp(-x) / pow((1 + exp(-x)), 2);
}

__host__ double cross_entropy_loss(double p_true, double p_pred){
    // calculate the cross entropy between two scalar p_true and p_pred
    // @ p_true: scalar
    // @ p_pred: scalar
    return p_true * log(p_pred) + (1-p_true) * log(1-p_pred);
}

__host__ double cross_entropy_loss_d(double p_true, double p_pred){
    // return the cross entropy loss derivative w.r.t. p_pred
    return -(1-p_true)/(1-p_pred) + p_true/p_pred;
}

__host__ void * populate_filter(double * filter){
	// populate a set of filters
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
}

__host__ void populate_img(double * img0, double * img){
	int x, y;
	for (int c = 0; c < C; c++){  // reuse previous indexer c for channels
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
			    img0[index3(c, x, y, H0, W0)] = img[index3(c, x-1, y-1, H, W)];
	            // printf("image pixel at (%d, %d, %d) is %.2f\n", c, x, y, img0[index3(c, x, y, H0, W0)]);
			}
		}
	}
}


__host__ void populate_weights(double * w){
    // randomly populate the weight matrix for fully connected layer
    int k, x, y;
    for (k = 0; k < K; k++) {
        for (x = 0; x < H; x++) {
            for (y = 0; y < W; y++)
                w[index3(k, x, y, H, W)] = k + x + y;
        }
    }
}


void read_batch(string filename, vector<Mat> &vec, vector<int> &label){
    // Read a bunch of images to a vector of Mat
    ifstream file (filename.c_str(), ios::binary);
    if (file.is_open())
    {
        int number_of_images = 10000;
        int n_rows = 32;
        int n_cols = 32;
        for(int i = 0; i < number_of_images; ++i)
        {
            unsigned char tplabel = 0;
            file.read((char*) &tplabel, sizeof(tplabel));
            vector<Mat> channels;
            Mat fin_img = Mat::zeros(n_rows, n_cols, CV_8UC3);
            for(int ch = 0; ch < 3; ++ch){
                Mat tp = Mat::zeros(n_rows, n_cols, CV_8UC1);
                for(int r = 0; r < n_rows; ++r){
                    for(int c = 0; c < n_cols; ++c){
                        unsigned char temp = 0;
                        file.read((char*) &temp, sizeof(temp));
                        tp.at<uchar>(r, c) = (int) temp;
                    }
                }
                channels.push_back(tp);
            }
            merge(channels, fin_img);
            vec.push_back(fin_img);
            label.push_back((int)tplabel);
        }
    }
}


void append_batch(vector<Mat> & containerX, vector<int> & containerY, vector<Mat> & batchX, vector<int> & batchY){
    // append batch training data and labels to the containers
    containerX.insert( containerX.end(), batchX.begin(), batchX.end() );
    containerY.insert( containerY.end(), batchY.begin(), batchY.end() );
}


void read_CIFAR10(vector<Mat> &trainX, vector<Mat> &testX, vector<int> &trainY, vector<int> &testY){

    string base_path = "/media/yuqiong/DATA/alexnet/data/cifar-10-batches-bin/";

    string filename;
    filename = base_path + "data_batch_1.bin";
    vector<Mat> batch1;
    vector<int> label1;
    read_batch(filename, batch1, label1);
    append_batch(trainX, trainY, batch1, label1);

    filename = base_path + "data_batch_2.bin";
    vector<Mat> batch2;
    vector<int> label2;
    read_batch(filename, batch2, label2);
    append_batch(trainX, trainY, batch2, label2);

    filename = base_path + "data_batch_3.bin";
    vector<Mat> batch3;
    vector<int> label3;
    read_batch(filename, batch3, label3);
    append_batch(trainX, trainY, batch3, label3);

    filename = base_path + "data_batch_4.bin";
    vector<Mat> batch4;
    vector<int> label4;
    read_batch(filename, batch4, label4);
    append_batch(trainX, trainY, batch4, label4);

    filename = base_path + "data_batch_5.bin";
    vector<Mat> batch5;
    vector<int> label5;
    read_batch(filename, batch5, label5);
    append_batch(trainX, trainY, batch5, label5);

    filename = base_path + "test_batch.bin";
    vector<Mat> batcht;
    vector<int> labelt;
    read_batch(filename, batcht, labelt);
    append_batch(testX, testY, batcht, labelt);
}


int main() {

    vector<Mat> trainX, testX;
    vector<int> trainY, testY;

    read_CIFAR10(trainX, testX, trainY, testY);    // make filter

    Mat img = trainX[0];
    int rows = img.rows;
    int cols = img.cols;
    int n = rows * cols;
    cout << "Rows " << rows << " cols " << cols << endl;

    uchar * arr = (uchar *) calloc(n, sizeof(uchar));
    arr = img.data;
    double * img_arr = (double *) calloc(n, sizeof(double));
    for (int i = 0; i < n; i++){
        img_arr[i] = static_cast<double>(arr[i]);
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

    populate_img(img0, img_arr);

	double * filter;
	unsigned int num_f = K * C * FH * FW;  // number of elements in the filter
	filter = (double*)calloc(num_f, sizeof(double));
	if( !filter )
	{
	    fprintf(stderr, " Cannot allocate the set of filters!\n");
	    exit(1);
	}

    populate_filter(filter);

    int y_true = trainY[0];  // label

	double checksum_i = checksum3(img0, C, H0, W0);
	printf("checksum image %.2f\n", checksum_i);

	printf("the size of the image is %d\n.", sizeof(img0));
	for (int c = 0; c < C; c++){
		for (int x = 0; x < H0; x++){
			for (int y = 0; y < W0; y++)
			    printf("Image position(%d, %d, %d) is %.2f\n", c, x, y, img0[index3(c, x, y, H0, W0)]);
		}
	}

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

    // check errorsS
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

    // Now pass the output to fully connected layer
    double * w;
	unsigned int num_w = K * H * W;  // number of elements in the image
	w = (double*) calloc(num_w, sizeof(double));
	if( !w)
	{
	    fprintf(stderr, "Cannot allocate the weight array\n");
	    exit(1);
	}

    populate_weights(w);
    double b = 1.03;    // bias

    double fc_out = fc(w, out, b, K, H, W);
    double y_pred = sigmoid(fc_out);
    double loss = cross_entropy_loss(y_true, y_pred);

    // Backward Step !!!
    // now immediately calculate derivatives from CUDA
    double * de;  // save derivative
    de = (double*)calloc(num_o, sizeof(double));

    double * d_de;  // save derivative from device
    cudaMalloc(& d_de, num_o * sizeof(double));

    convDevKernel<<<blocksPerGrid, threadsPerBlock>>>(d_o, d_f, d_de);
    cudaMemcpy(de, d_de, num_o * sizeof(double), cudaMemcpyDeviceToHost);

    double * fc_d_f;
    fc_d_f = (double*) calloc(num_f, sizeof(double));
    // now loop through w and cumulate all partial derivative cubes
    int k, x, y;
    for (k = 0; k < K; c++){
        for (x = 0; x < H; x++){
            for (y = 0; y < W; y++){
                double curr_w = w[index3(k, x, y, H, W)];
                // amplify all elements in the partial derivative cube by this curr_w
                int c, i, j;
                for (c = 0; c < C; c++){
                    for (i = 0; i < FH; i++){
                        for (j = 0; j < FW; j++)
                            fc_d_f[index3(c, i, j, FH, FW)] += curr_w * de[index6(k, x, y, c, i, j, H, W, C, FH, FW))];
                    }
                }
            }
        }
    }

    // now fc_d_f is 3D cube
    double y_pred_d_fc = sigmoid_d(fc_out);
    // chain rule
    double * y_pred_d_f;
    y_pred_d_f = (double*) calloc(num_f, sizeof(double));
    for (c = 0; c < C; c++){
        for (i = 0; i < FH; i++){
            for (j = 0; j < FW; j++)
                y_pred_d_f[index3(c, i, j, FH, FW)] = y_pred_d_fc * fc_d_f[index3(c, i, j, FH, FW)];
        }
    }

    double loss_d_y_pred = cross_entropy_loss_d(y_true, y_pred);
    // chain rule
    double * loss_d_f;
    loss_d_f = (double*) calloc(num_f, sizeof(double));
    for (c = 0; c < C; c++){
        for (i = 0; i < FH; i++){
            for (j = 0; j < FW; j++)
                loss_d_f[index3(c, i, j, FH, FW)] = loss_d_y_pred * y_pred_d_f[index3(c, i, j, FH, FW)];
        }
    }

    double lr = 0.1;  // learning rate
    for (c = 0; c < C; c++){
        for (i = 0; i < FH; i++){
            for (j = 0; j < FW; j++)
                filter[index3(c, i, j, FH, FW)] -= lr * loss_d_f[index3(c, i, j, FH, FW)];
        }
    }

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
