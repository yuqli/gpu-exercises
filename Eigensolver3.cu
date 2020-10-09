// Batched Eigensolver for 2x2 matrices. Analytical solution. Implementation only.

#include "cuda_kernels.h"
#include <chrono>
typedef std::chrono::high_resolution_clock Clock;


#define CUDA_CHECK_ERROR( err ) (cudaCheckError( err, __FILE__, __LINE__ ))

inline void cudaCheckError( cudaError_t err, const char *file, int line )
{
	// CUDA error handeling from the "CUDA by example" book
	if (err != cudaSuccess)
    {
		printf( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
		exit( EXIT_FAILURE );
	}
}


// Get index offset in a 2D matrix at position (i, j)
// row-major memory layout
__host__ __device__ int offset2D(int i, int j, int cols) {
    return i * cols + j;
}


// in-place normalize a vector so its length == 1
__host__ __device__ void normalizeVector(float &a, float &b) {
    float tmp = sqrt(a * a + b * b);
    if (tmp < 1e-3) return;  // avoid division by zero
    a /= tmp;
    b /= tmp;
} 


// Analytical solution of get the dominant eigen vector of a 2x2 symmetric matrix
// | a, c | 
// | c, d |
// eigen_values : float[2]
// eigen_vectors: float[4], column major !
__host__ __device__ void getDominantVectorOptimized(float curr_dx, float curr_dy, float *dominant_u, float *dominant_v, float *eigen_val1, float *eigen_val2) {
    float a = curr_dx * curr_dx;
    float b = curr_dx * curr_dy;
    float d = curr_dy * curr_dy;

    float det = a * d - b * b;  // determinant
    float tr = a + d;  // trace
    float tmp = sqrt(tr * tr - 4 * det);

    // first eigen vector
    float dominant_eigen_values = (tr + tmp) / 2;
    *dominant_u = dominant_eigen_values - d;
    *dominant_v = b;
    normalizeVector(*dominant_u, *dominant_v);

    // store eigen values
    *eigen_val1 = (tr + tmp) / 2;
    *eigen_val2 = (tr - tmp) / 2;

    // adjust for signs
    float sign = curr_dx * (*dominant_u) + curr_dy * (*dominant_v);
    if (fabs(sign) < 1e-3) {
        *dominant_u = 0;
        *dominant_v = 0;
    } 
    else if (sign < -1e-3) {
        *dominant_u *= -1;
        *dominant_v *= -1;
    }
}


// param[in] dxx: dx * dx 
// param[in] dxy: dx * dy 
// param[in] dyy: dy * dy 
__global__ void dominantVectorKernel(float *dx, float *dy, float *dominant_u, float *dominant_v, float *eigen_val1, float *eigen_val2, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // row id 
    int j = blockIdx.y * blockDim.y + threadIdx.y;  // col id

    if (i < rows && j < cols) {
        int offset = offset2D(i, j, cols);
        float curr_dx = dx[offset];
        float curr_dy = dy[offset];

        // TODO (yuqli) : add Gaussian smoothing here!
        float curr_dominant_u, curr_dominant_v;  // put on cuda register to save global memory access
        float curr_eigen1, curr_eigen2;  
        getDominantVectorOptimized(curr_dx, curr_dy, &curr_dominant_u, &curr_dominant_v, &curr_eigen1, &curr_eigen2);

        dominant_u[offset] = curr_dominant_u;
        dominant_v[offset] = curr_dominant_v;
        eigen_val1[offset] = curr_eigen1;
        eigen_val2[offset] = curr_eigen2;
    }
}


void cudaInit() {
    cudaFree(0);
}


// wrapper for the dominantVectorKernel
void cudaGetDominantVector(const cv::Mat & grad_x, const cv::Mat & grad_y, int rows, int cols,
                           std::vector<cv::Mat> & dominant_vector, std::vector<cv::Mat> & eigen_values) {

    int n = rows * cols; 

    // Set up inputs on device
    float *d_grad_x;
    float *d_grad_y;
    CUDA_CHECK_ERROR(cudaMalloc((void **) &d_grad_x, sizeof(float) * n));
    CUDA_CHECK_ERROR(cudaMalloc((void **) &d_grad_y, sizeof(float) * n));
    CUDA_CHECK_ERROR(cudaMemcpy(d_grad_x, (float*)grad_x.data, sizeof(float) * n, cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(d_grad_y, (float*)grad_y.data, sizeof(float) * n, cudaMemcpyHostToDevice));

    // Set up outputs on device
    float *d_dominant_u;
    float *d_dominant_v;
    CUDA_CHECK_ERROR(cudaMalloc((void **)& d_dominant_u, sizeof(float) * n));
    CUDA_CHECK_ERROR(cudaMalloc((void **)& d_dominant_v, sizeof(float) * n));

    float *d_eigen_val1;
    float *d_eigen_val2;
    CUDA_CHECK_ERROR(cudaMalloc((void **)& d_eigen_val1, sizeof(float) * n));
    CUDA_CHECK_ERROR(cudaMalloc((void **)& d_eigen_val2, sizeof(float) * n));

    // Run kernel
    dim3 blocksPerGrid(ceil(rows/32.0), ceil(cols/32.0), 1);  // !! blockId.x is rows, blockId.y is cols
    dim3 threadsPerBlock(32, 32, 1);
    dominantVectorKernel<<<blocksPerGrid, threadsPerBlock>>> (d_grad_x, d_grad_y, d_dominant_u, d_dominant_v, d_eigen_val1, d_eigen_val2, rows, cols);

    // Set up outputs buffer on the host 
    float *h_dominant_u = new float[n]();
    float *h_dominant_v = new float[n]();
    CUDA_CHECK_ERROR(cudaMemcpy(h_dominant_u, d_dominant_u, sizeof(float) * n, cudaMemcpyDeviceToHost));
    CUDA_CHECK_ERROR(cudaMemcpy(h_dominant_v, d_dominant_v, sizeof(float) * n, cudaMemcpyDeviceToHost));

    float *h_eigen_val1 = new float[n]();
    float *h_eigen_val2 = new float[n]();
    CUDA_CHECK_ERROR(cudaMemcpy(h_eigen_val1, d_eigen_val1, sizeof(float) * n, cudaMemcpyDeviceToHost));
    CUDA_CHECK_ERROR(cudaMemcpy(h_eigen_val2, d_eigen_val2, sizeof(float) * n, cudaMemcpyDeviceToHost));

    cv::Mat dominant_u = cv::Mat(rows, cols, CV_32F, h_dominant_u);
    cv::Mat dominant_v = cv::Mat(rows, cols, CV_32F, h_dominant_v);
    cv::Mat eigen_val1 = cv::Mat(rows, cols, CV_32F, h_eigen_val1);
    cv::Mat eigen_val2 = cv::Mat(rows, cols, CV_32F, h_eigen_val2);

    // Free device memory
    cudaFree(d_dominant_u);
    cudaFree(d_dominant_v);

    cudaFree(d_eigen_val1);
    cudaFree(d_eigen_val2);

    dominant_vector.push_back(dominant_u);
    dominant_vector.push_back(dominant_v);
    eigen_values.push_back(eigen_val1);
    eigen_values.push_back(eigen_val2);
}


