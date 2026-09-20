#include <matrixMultiplyGPU.cuh>
#define STUDENTID 48448239 //DO NOT REMOVE
/**
* @brief Implements an NxN matrix multiply C=A*B
*				 			 	    	 		   			 	      
* @param[in] N : dimension of square matrix (NxN)
* @param[in] A : pointer to input NxN matrix
* @param[in] B : pointer to input NxN matrix
* @param[out] C : pointer to output NxN matrix
* @param[in] flags : pointer to array of integers which can be used for debugging and performance tweaks. Optional. If unused, set to zero
* @param[in] flagCount : the length of the flags array
* @return : your student ID
*				 			 	    	 		   			 	      
* */
__host__ int matrixMultiply_GPU(int N, const floatTypeCUDA* A, const floatTypeCUDA* B, floatTypeCUDA* C, int* flags, int flagCount){				 			 	    	 		   			 	      
if (N<=0) { return STUDENTID;}//Your code must be able to deal with N=0 scenario without crashing.				 			 	    	 		   			 	      

//WRITE YOUR CODE HERE
	int flag0 = (flagCount > 0) ? flags[0] : 0;
    	int flag1 = (flagCount > 1) ? flags[1] : 0;
    	int flag2 = (flagCount > 2) ? flags[2] : 0;

	size_t bytes = (size_t)N * N * sizeof(floatTypeCUDA);
	floatTypeCUDA *d_A, *d_B, *d_C;
	
	cudaMalloc(&d_A, bytes);
	cudaMalloc(&d_B, bytes);
	cudaMalloc(&d_C, bytes);

	cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, bytes, cudaMemcpyHostToDevice);

	dim3 threadsPerBlock(16, 16);
	dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

	matrixMultiplyKernel_GPU<<<numBlocks, threadsPerBlock>>>(N, d_A, d_B, d_C, flag0, flag1, flag2);
	
	cudaDeviceSynchronize();
	cudaMemcpy(C, d_C, bytes, cudaMemcpyDeviceToHost);
	cudaFree(d_A);
    	cudaFree(d_B);
    	cudaFree(d_C);
return STUDENTID;				 			 	    	 		   			 	      

}				 			 	    	 		   			 	      

//The kernel (device code) parameters have been setup almost the same as the host code, except the flags are passed in individually rather than as a pointer. This is done just so you don't have to copy the parameters to GPU memory first, you'll be able to pass in up to 3 on the function call.				 			 	    	 		   			 	      
__global__ void matrixMultiplyKernel_GPU(int N, const floatTypeCUDA* A, const floatTypeCUDA* B, floatTypeCUDA* C, int flag0, int flag1, int flag2){				 			 	    	 		   			 	      
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	
	if (row < N && col < N) {
		floatTypeCUDA sum = make_cuFloatComplex(0.0f, 0.0f);;
		
		for (int k = 0; k < N; k++) {
			sum = cuCaddf(sum, cuCmulf(A[k*N+row], B[col*N+k]));
		}
		
		C[col * N + row] = sum;
	}
}				 			 	    	 		   			 	      
