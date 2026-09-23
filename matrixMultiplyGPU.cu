#include <matrixMultiplyGPU.cuh>
#define STUDENTID 48448239 //DO NOT REMOVE

#define TILE_DIM 32
#define BLOCK_ROWS 8

#define BM 64
#define BN 64
#define BK 8
#define TM 4
#define TN 4

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

	static floatTypeCUDA *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
	static bool initialized = false;
	size_t bytes = (size_t)N * N * sizeof(floatTypeCUDA);

	if (!initialized) {
		cudaMalloc(&d_A, bytes);
	        cudaMalloc(&d_B, bytes);
	        cudaMalloc(&d_C, bytes);
	        initialized = true;
	}

	// Transpose
	cudaMemcpy(d_A, B, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, A, bytes, cudaMemcpyHostToDevice);

	dim3 threadsPerBlock(256);
	dim3 numBlocks(N / BM, N / BN);

	matrixMultiplyKernel_GPU<<<numBlocks, threadsPerBlock>>>(N, d_A, d_B, d_C, flag0, flag1, flag2);
	
	cudaDeviceSynchronize();
	cudaMemcpy(C, d_C, bytes, cudaMemcpyDeviceToHost);

return STUDENTID;				 			 	    	 		   			 	      

}				 			 	    	 		   			 	      

//The kernel (device code) parameters have been setup almost the same as the host code, except the flags are passed in individually rather than as a pointer. This is done just so you don't have to copy the parameters to GPU memory first, you'll be able to pass in up to 3 on the function call.				 			 	    	 		   			 	      
__global__ void matrixMultiplyKernel_GPU(int N, const floatTypeCUDA* A, const floatTypeCUDA* B, floatTypeCUDA* C, int flag0, int flag1, int flag2){				 			 	    	 		   			 	      
	// Block mappings (Row-Major)
    const uint cCol = blockIdx.x;
    const uint cRow = blockIdx.y;

    // 256 threads map to a 16x16 grid of outputs (each computing 4x4)
    const uint threadCol = threadIdx.x % (BN / TN); // 0 to 15
    const uint threadRow = threadIdx.x / (BN / TN); // 0 to 15

    // Flattened SMEM caches (Kernel 5 style)
    __shared__ floatTypeCUDA As[BM * BK];
    __shared__ floatTypeCUDA Bs[BK * BN];

    // Thread loading strides
    const uint innerRowA = threadIdx.x / BK;
    const uint innerColA = threadIdx.x % BK;
    const uint strideA = 256 / BK; // 32

    const uint innerRowB = threadIdx.x / BN;
    const uint innerColB = threadIdx.x % BN;
    const uint strideB = 256 / BN; // 4

    // Thread-local register cache
    floatTypeCUDA threadResults[TM * TN];
    #pragma unroll
    for (int i = 0; i < TM * TN; ++i) {
        threadResults[i] = make_cuFloatComplex(0.0f, 0.0f);
    }

    floatTypeCUDA regM[TM];
    floatTypeCUDA regN[TN];

    int numPhases = N / BK;
    for (uint phase = 0; phase < numPhases; ++phase) {
        
        // 1. GMEM -> SMEM Load Phase (Coalesced via Transpose Trick)
        #pragma unroll
        for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
            int g_row = cRow * BM + innerRowA + loadOffset;
            int g_col = phase * BK + innerColA;
            As[(innerRowA + loadOffset) * BK + innerColA] = __ldg(&A[g_row * N + g_col]);
        }

        #pragma unroll
        for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
            int g_row = phase * BK + innerRowB + loadOffset;
            int g_col = cCol * BN + innerColB;
            Bs[(innerRowB + loadOffset) * BN + innerColB] = __ldg(&B[g_row * N + g_col]);
        }

        __syncthreads();

        // 2. Math Phase: Kernel 5 2D Blocktiling
        #pragma unroll
        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
            
            // Block into registers
            #pragma unroll
            for (uint i = 0; i < TM; ++i) {
                regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
            }

            #pragma unroll
            for (uint i = 0; i < TN; ++i) {
                regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
            }

            // Standard cuCaddf/cuCmulf math
            #pragma unroll
            for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
                #pragma unroll
                for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
                    threadResults[resIdxM * TN + resIdxN] = cuCaddf(
                        threadResults[resIdxM * TN + resIdxN],
                        cuCmulf(regM[resIdxM], regN[resIdxN])
                    );
                }
            }
        }
        __syncthreads();
    }

    // 3. SMEM -> GMEM Write Phase
    #pragma unroll
    for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
        #pragma unroll
        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
            int g_row = cRow * BM + threadRow * TM + resIdxM;
            int g_col = cCol * BN + threadCol * TN + resIdxN;
            C[g_row * N + g_col] = threadResults[resIdxM * TN + resIdxN];
        }
    }
}				 			 	    	 		   			 	      
