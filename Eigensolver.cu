// CUDA kernels for image ridgeness 
// Yuqiong Li
// VinSmart 
// 09/25/2020

// http://eigen.tuxfamily.org/dox-devel/TopicCUDA.html
// Note: This is not working with Eigen 3.3 and NVCC 10.1 ! Since the functionality is experimental, it might not have been ported yet.

#include "cuda_kernels.h"

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
__host__ __device__ int offset2D(int i, int j, int col) {
    return i * col + j;
}


__global__ void dominantVectorKernel(Eigen::Matrix2f *tensor, float *dominant_u, float *dominant_v, int rows, int cols) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;  // row id 
    int j = blockIdx.x * blockDim.x + threadIdx.x;  // col id

    if (i < rows && j < cols) {
        int offset = offset2D(i, j, cols);

        // get eigen values
        Eigen::EigenSolver<Eigen::Matrix2f> eigen_solver;
        eigen_solver.compute(tensor[offset]);
        Eigen::Vector2f eigen_values = eigen_solver.eigenvalues().real();
        Eigen::Matrix2f eigen_vectors = eigen_solver.eigenvectors().real();

        // get dominant vector
        int index = eigen_values(0) > eigen_values(1) ? 0 : 1;
        auto dominant_vector = eigen_vectors.col(index);  

        // assign results to output buffer
        dominant_u[offset] = dominant_vector(0);
        dominant_v[offset] = dominant_vector(1);
    }

    return;
}


// wrapper for the dominantVectorKernel
std::vector<cv::Mat> cudaGetDominantVector(const std::vector<Eigen::Matrix2f> & tensor, int rows, int cols) {        

    int n = rows * cols; 

    // Allocate host arrays 
    float *h_dominant_u = new float[n]();
    float *h_dominant_v = new float[n]();

    // Allocate device arrays
    Eigen::Matrix2f *d_tensor ;
    CUDA_CHECK_ERROR(cudaMalloc((void **)&d_tensor, sizeof(Eigen::Matrix2f) * n));

    float *d_dominant_u;
    float *d_dominant_v;
    CUDA_CHECK_ERROR(cudaMalloc((void **)&d_dominant_u, sizeof(float) * n));
    CUDA_CHECK_ERROR(cudaMalloc((void **)&d_dominant_v, sizeof(float) * n));

    // Copy to device
    CUDA_CHECK_ERROR(cudaMemcpy(d_tensor, tensor.data(), sizeof(Eigen::Matrix2f) * n, cudaMemcpyHostToDevice));

    // Run kernel
    dim3 blocksPerGrid(ceil(rows/32.0), ceil(cols/32.0), 1);
    dim3 threadsPerBlock(32, 32, 1);
    dominantVectorKernel<<<blocksPerGrid, threadsPerBlock>>> (d_tensor, d_dominant_u, d_dominant_v, rows, cols);
    
    // Copy to host
    CUDA_CHECK_ERROR(cudaMemcpy(d_dominant_u, h_dominant_u, sizeof(float) * n, cudaMemcpyDeviceToHost));
    CUDA_CHECK_ERROR(cudaMemcpy(d_dominant_v, h_dominant_v, sizeof(float) * n, cudaMemcpyDeviceToHost));

    cv::Mat dominant_u = cv::Mat(rows, cols, CV_32F, h_dominant_u);
    cv::Mat dominant_v = cv::Mat(rows, cols, CV_32F, h_dominant_v);

    // Free device memory
    cudaFree(d_dominant_u);
    cudaFree(d_dominant_v);

    return {dominant_u, dominant_v};
}
