#include <matrixMultiply.h>
#define STUDENTID 48448239 //DO NOT REMOVE


static inline __m256 complexMul(__m256 a, __m256 b) {
	__m256 aRe = _mm256_moveldup_ps(a);
	__m256 aIm = _mm256_movehdup_ps(a);
	__m256 bSwap = _mm256_permute_ps(b, 0xB1);
	return _mm256_addsub_ps(_mm256_mul_ps(aRe, b), _mm256_mul_ps(aIm, bSwap));
}

/**
* @brief Implements an NxN matrix multiply C=A*B
*				 			 	    	 		   			 	      
* @param[in] N : dimension of square matrix (NxN)
* @param[in] A : pointer to input NxN matrix
* @param[in] B : pointer to input NxN matrix
* @param[out] C : pointer to output NxN matrix
* @param[in] args : pointer to array of integers which can be used for debugging and performance tweaks. Optional. If unused, set to zero
* @param[in] argCount : the length of the flags array
* @return : your student ID
*				 			 	    	 		   			 	      
* */
int matrixMultiply(int N, const floatType* A, const floatType* B, floatType* C, int* args, int argCount) {		
if (N<=0) { return STUDENTID;}//Your code must be able to deal with N=0 scenario without crashing.				 			 	    	 		   			 	      
//WRITE YOUR CODE HERE
	memset(C, 0, N * N * sizeof(floatType));
	#pragma omp parallel for num_threads(4)
	for (int j = 0; j < N; j++) {
		for (int k = 0; k < N; k++) {
			floatType b = B[j * N + k];
			__m256d bBroadcast = _mm256_broadcast_sd(reinterpret_cast<const double*>(&b));
			__m256 bVec = _mm256_castpd_ps(bBroadcast);
			for (int i = 0; i < N; i+= 4) {
				__m256 aVec = _mm256_loadu_ps(reinterpret_cast<const float*>(A + k*N + i));
				__m256 cVec = _mm256_loadu_ps(reinterpret_cast<const float*>(C + j*N + i));
				cVec = _mm256_add_ps(cVec, complexMul(aVec, bVec));
				_mm256_storeu_ps(reinterpret_cast<float*>(C + j*N + i), cVec);
			}
		}
	}

return STUDENTID;				 			 	    	 		   			 	      

}

				 			 	    	 		   			 	      
