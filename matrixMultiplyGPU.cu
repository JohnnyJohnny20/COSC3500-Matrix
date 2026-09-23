#include <matrixMultiplyGPU.cuh>
#define STUDENTID 48448239 //DO NOT REMOVE

#define TILE_DIM 32
#define BLOCK_ROWS 8

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

	dim3 threadsPerBlock(32, 8);
	dim3 numBlocks((N + TILE_DIM - 1) / TILE_DIM, (N + TILE_DIM - 1) / TILE_DIM);

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
	
	__shared__ floatTypeCUDA s_A[TILE_DIM][TILE_DIM];
	__shared__ floatTypeCUDA s_B[TILE_DIM][TILE_DIM];
	
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int row = blockIdx.x * TILE_DIM + tx;
	int col_start = blockIdx.y * TILE_DIM + ty;
	
	floatTypeCUDA sum[4];

	#pragma unroll
	for (int i = 0; i < 4; ++i) {
        	sum[i] = make_cuFloatComplex(0.0f, 0.0f);
    	}
	
	int numPhases = (N + TILE_DIM - 1) / TILE_DIM;
	for (int phase = 0; phase < numPhases; ++phase) {
		
		#pragma unroll
		for (int i = 0; i < 4 ; ++i) {
			int load_ty = ty + i * BLOCK_ROWS;
			int k_A = phase * TILE_DIM + load_ty;
	        	int k_B = phase * TILE_DIM + tx;
			
			
			// 3. Collaborative Load A with bounds checking
		        if (row < N && k_A < N) {
		            s_A[load_ty][tx] = __ldg(&A[k_A * N + row]);
		        } else {
		            s_A[load_ty][tx] = make_cuFloatComplex(0.0f, 0.0f);
		        }
			
			int load_col = blockIdx.y * TILE_DIM + load_ty;
		        // Collaborative Load B with bounds checking
		        if (k_B < N && load_col < N) {
		            s_B[load_ty][tx] = __ldg(&B[load_col * N + k_B]);
		        } else {
		            s_B[load_ty][tx] = make_cuFloatComplex(0.0f, 0.0f);
		        }
		}
		
		__syncthreads();
		
		#pragma unroll
		for (int k = 0; k < TILE_DIM; ++k) {
			floatTypeCUDA a_val = s_A[k][tx];
				
			#pragma unroll
			for (int i = 0; i < 4; ++i) {
				floatTypeCUDA b_val = s_B[ty + i * BLOCK_ROWS][k];
				sum[i] = cuCaddf(sum[i], cuCmulf(a_val, b_val));
			}
		}
		
		__syncthreads();
	}
	
	#pragma unroll
	for (int i = 0; i < 4; ++i) {
		int current_col = col_start + i * BLOCK_ROWS;
		if (row < N && current_col < N) {
        		C[current_col * N + row] = sum[i];
    		}
	}
}				 			 	    	 		   			 	      
