/*
 * How to compile (assume cuda is installed at /usr/local/cuda/)
 *   nvcc -c -I/usr/local/cuda/include syevd_example.cpp 
 *   g++ -o a.out syevd_example.o -L/usr/local/cuda/lib64 -lcudart -lcusolver
 *
 */

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#define CUDA_CHECK_ERROR( err ) (cudaCheckError( err, __FILE__, __LINE__ ))
#define CUDA_SOLVER_CHECK_ERROR( err ) (cudaSolverCheckError( err, __FILE__, __LINE__ ))

inline void cudaCheckError( cudaError_t err, const char *file, int line )
{
	// CUDA error handeling from the "CUDA by example" book
	if (err != cudaSuccess)
    {
		printf( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
		exit( EXIT_FAILURE );
	}
}

inline void cudaSolverCheckError(cusolverStatus_t err, const char *file, int line )
{
    // cuSolver erro handling from official document examples
	if (err != CUSOLVER_STATUS_SUCCESS)
    {
		printf( "%d in %s at line %d\n", err, file, line );
		exit( EXIT_FAILURE );
	}
}


void printMatrix(int m, int n, const double*matrix, int lda, const char* name)
{
    for(int row = 0 ; row < m ; row++){
        for(int col = 0 ; col < n ; col++){
            double Areg = matrix[row + col*lda];
            printf("%s(%d,%d) = %f\n", name, row+1, col+1, Areg);
        }
    }
}


// Calls cuSolver to get eigen values. Wrapper around the cuda kernel
// https://docs.nvidia.com/cuda/cusolver/index.html#syevd-example1
// @in: matrix : m x m symmetrix. double[mxm], row major layout
// @out: eigen_values, double[m]
// @out: eigeh_vectors, double[m]
void eigenDecomposition(double *matrix, double *eigen_values, double *eigen_vectors, int m) {

    cusolverDnHandle_t cusolverH = NULL;

    double *d_matrix = NULL;
    double *d_eigen_values = NULL;

    int *devInfo = NULL;
    double *d_work = NULL;
    int  lwork = 0;
    int info_gpu = 0;

    // step 1: create cusolver/cublas handle
    CUDA_SOLVER_CHECK_ERROR(cusolverDnCreate(&cusolverH));

    // step 2: create device data buffers
    CUDA_CHECK_ERROR(cudaMalloc ((void**)&d_matrix, sizeof(double) * m * m));
    CUDA_CHECK_ERROR(cudaMalloc ((void**)&d_eigen_values, sizeof(double) * m));
    CUDA_CHECK_ERROR(cudaMalloc ((void**)&devInfo, sizeof(int)));

    CUDA_CHECK_ERROR(cudaMemcpy(d_matrix, matrix, sizeof(double) * m * m, cudaMemcpyHostToDevice));

    // step 3: query working space of syevd
    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvalues and eigenvectors.
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
    CUDA_SOLVER_CHECK_ERROR(cusolverDnDsyevd_bufferSize(
        cusolverH,
        jobz,
        uplo,
        m,
        d_matrix,
        m,
        d_eigen_values,
        &lwork));

    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_work, sizeof(double)*lwork));

    // step 4: compute spectrum
    CUDA_SOLVER_CHECK_ERROR(cusolverDnDsyevd(
        cusolverH,
        jobz,
        uplo,
        m,
        d_matrix,
        m,
        d_eigen_values,
        d_work,
        lwork,
        devInfo));

    CUDA_CHECK_ERROR(cudaDeviceSynchronize());

    CUDA_CHECK_ERROR(cudaMemcpy(eigen_values, d_eigen_values, sizeof(double)*m, cudaMemcpyDeviceToHost));
    CUDA_CHECK_ERROR(cudaMemcpy(eigen_vectors, d_matrix, sizeof(double)*m*m, cudaMemcpyDeviceToHost));  // in-place computation on device
    CUDA_CHECK_ERROR(cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost));

    // check solver status
    std::cout << "after syevd: info_gpu = " << info_gpu << std::endl;
    assert(0 == info_gpu);

     
    if (d_matrix) cudaFree(d_matrix);
    if (d_eigen_values) cudaFree(d_eigen_values);
    if (devInfo) cudaFree(devInfo);
    if (d_work ) cudaFree(d_work);
    if (cusolverH) cusolverDnDestroy(cusolverH);

    cudaDeviceReset();
} 


int main(int argc, char*argv[])
{

    /*       | 3.5 0.5 0 |
    *   matrix = | 0.5 3.5 0 |
    *       | 0   0   2 |
    *
    */
    int m = 3;
    double matrix[m*m] = { 3.5, 0.5, 0, 0.5, 3.5, 0, 0, 0, 2.0};
    double lambda[m] = { 2.0, 3.0, 4.0};

    double V[m*m]; // eigenvectors
    double W[m]; // eigenvalues

    std::cout << "matrix = (matlab base-1)\n";
    printMatrix(m, m, matrix, m, "matrix");
    printf("=====\n");

    // call eigen value function 
    eigenDecomposition(matrix, W, V, m);


    printf("eigenvalue = (matlab base-1), ascending order\n");
    for(int i = 0 ; i < m ; i++){
        printf("W[%d] = %E\n", i+1, W[i]);
    }

    printf("V = (matlab base-1)\n");
    printMatrix(m, m, V, m, "V");
    printf("=====\n");


    // step 4: check eigenvalues
    double lambda_sup = 0;
    for(int i = 0 ; i < m ; i++){
        double error = fabs( lambda[i] - W[i]);
        lambda_sup = (lambda_sup > error)? lambda_sup : error;
    }
    printf("|lambda - W| = %E\n", lambda_sup);

    return 0;
}



