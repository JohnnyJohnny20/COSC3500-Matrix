#include <matrixMultiply.h>
#define STUDENTID 48448239 //DO NOT REMOVE
#pragma GCC target("avx2,fma")

static inline __m256 complexMul(__m256 a, __m256 b) {
	__m256 aRe = _mm256_moveldup_ps(a);
	__m256 aIm = _mm256_movehdup_ps(a);
	__m256 bSwap = _mm256_permute_ps(b, 0xB1);
	return _mm256_addsub_ps(_mm256_mul_ps(aRe, b), _mm256_mul_ps(aIm, bSwap));
}

static inline __m256 complexMulPrecomputed(__m256 aRe, __m256 aIm, __m256 b, __m256 bSw) {
	return _mm256_addsub_ps(_mm256_mul_ps(aRe, b), _mm256_mul_ps(aIm, bSw));
}

static inline __m256 broadcastComplex(floatType v) {
	__m256d d = _mm256_broadcast_sd(reinterpret_cast<const double*>(&v));
    	return _mm256_castpd_ps(d);
}

static inline __m256 fmaAddComplex(__m256 c, __m256 aRe, __m256 aIm, __m256 b, __m256 bSw) {
        __m256 c_new = _mm256_fmadd_ps(aRe, b, c); // c = c + aRe*b
        return _mm256_addsub_ps(c_new, _mm256_mul_ps(aIm, bSw)); // c_new +/- aIm*bSw
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
	
	const floatType* X = B;
	const floatType* Y = A;
	
	int iStart = 0;
	int iEndLimit = N;
	
	if (argCount == 2) {
        	iStart = args[0];
        	iEndLimit = args[1];
    	}
		
	#pragma omp parallel for num_threads(4)
	for (int i = iStart; i < iEndLimit ; i++) {
		memset(C + i * N, 0, N * sizeof(floatType));
	}

	const int blockJ = 32, blockK = 32;
	const int blockI = 32;
	
	#pragma omp parallel for num_threads(4)
	
	for (int ii = iStart; ii < iEndLimit; ii += blockI) {
		int iEnd = ii + blockI < iEndLimit ? ii + blockI : iEndLimit;

        	for (int kk = 0; kk < N; kk += blockK) {
                	int kEnd = kk + blockK < N ? kk + blockK : N;
	
				for (int i = ii; i + 4 <= iEnd; i += 4) {          // widened to 4 at a time
	                        	for (int k = kk; k < kEnd; k+=4) {      
	                                __m256 bVec0 = broadcastComplex(X[(i+0)*N+k]);
	                                __m256 bVec1 = broadcastComplex(X[(i+1)*N+k]);
	                                __m256 bVec2 = broadcastComplex(X[(i+2)*N+k]);
	                                __m256 bVec3 = broadcastComplex(X[(i+3)*N+k]);
					__m256 bSw0 = _mm256_permute_ps(bVec0, 0xB1);
					__m256 bSw1 = _mm256_permute_ps(bVec1, 0xB1);
					__m256 bSw2 = _mm256_permute_ps(bVec2, 0xB1);
					__m256 bSw3 = _mm256_permute_ps(bVec3, 0xB1);

					__m256 bVec4 = broadcastComplex(X[(i+0)*N+k+1]);
	                                __m256 bVec5 = broadcastComplex(X[(i+1)*N+k+1]);
	                                __m256 bVec6 = broadcastComplex(X[(i+2)*N+k+1]);
	                                __m256 bVec7 = broadcastComplex(X[(i+3)*N+k+1]);
	                                __m256 bSw4 = _mm256_permute_ps(bVec4, 0xB1);
	                                __m256 bSw5 = _mm256_permute_ps(bVec5, 0xB1);
	                                __m256 bSw6 = _mm256_permute_ps(bVec6, 0xB1);
	                                __m256 bSw7 = _mm256_permute_ps(bVec7, 0xB1);
						
					// --- Precompute B for step k+2 ---
	                                __m256 bVec8 = broadcastComplex(X[(i+0)*N+k+2]);
	                                __m256 bVec9 = broadcastComplex(X[(i+1)*N+k+2]);
	                                __m256 bVec10 = broadcastComplex(X[(i+2)*N+k+2]);
	                                __m256 bVec11 = broadcastComplex(X[(i+3)*N+k+2]);
	                                __m256 bSw8 = _mm256_permute_ps(bVec8, 0xB1);
	                                __m256 bSw9 = _mm256_permute_ps(bVec9, 0xB1);
	                                __m256 bSw10 = _mm256_permute_ps(bVec10, 0xB1);
	                                __m256 bSw11 = _mm256_permute_ps(bVec11, 0xB1);

	                                // --- Precompute B for step k+3 ---
	                                __m256 bVec12 = broadcastComplex(X[(i+0)*N+k+3]);
	                                __m256 bVec13 = broadcastComplex(X[(i+1)*N+k+3]);
	                                __m256 bVec14 = broadcastComplex(X[(i+2)*N+k+3]);
	                                __m256 bVec15 = broadcastComplex(X[(i+3)*N+k+3]);
	                                __m256 bSw12 = _mm256_permute_ps(bVec12, 0xB1);
	                                __m256 bSw13 = _mm256_permute_ps(bVec13, 0xB1);
	                                __m256 bSw14 = _mm256_permute_ps(bVec14, 0xB1);
	                                __m256 bSw15 = _mm256_permute_ps(bVec15, 0xB1);
					
                                	 for (int jj = 0; jj < N; jj += blockJ) {
                                	int jEnd = jj + blockJ < N ? jj + blockJ : N;
					for (int j = jj; j + 4 <= jEnd; j += 4) {
	                                        
						__m256 c0 = _mm256_loadu_ps(reinterpret_cast<const float*>(C + (i+0)*N + j));
                                                __m256 c1 = _mm256_loadu_ps(reinterpret_cast<const float*>(C + (i+1)*N + j));
                                                __m256 c2 = _mm256_loadu_ps(reinterpret_cast<const float*>(C + (i+2)*N + j));
                                                __m256 c3 = _mm256_loadu_ps(reinterpret_cast<const float*>(C + (i+3)*N + j));
						
						__m256 aVec, aRe, aIm;

                                                aVec = _mm256_loadu_ps(reinterpret_cast<const float*>(Y + k*N + j)); // contiguous, loaded once, used 4x
						aRe = _mm256_moveldup_ps(aVec);
        					aIm = _mm256_movehdup_ps(aVec);

                                                c0 = fmaAddComplex(c0, aRe, aIm, bVec0, bSw0);
                                                c1 = fmaAddComplex(c1, aRe, aIm, bVec1, bSw1);
                                                c2 = fmaAddComplex(c2, aRe, aIm, bVec2, bSw2);
                                                c3 = fmaAddComplex(c3, aRe, aIm, bVec3, bSw3);

						aVec = _mm256_loadu_ps(reinterpret_cast<const float*>(Y + (k+1)*N + j)); // contiguous, loaded once, used 4x
                                                aRe = _mm256_moveldup_ps(aVec);
                                                aIm = _mm256_movehdup_ps(aVec);
                                        	
						c0 = fmaAddComplex(c0, aRe, aIm, bVec4, bSw4);
                                                c1 = fmaAddComplex(c1, aRe, aIm, bVec5, bSw5);
                                                c2 = fmaAddComplex(c2, aRe, aIm, bVec6, bSw6);
                                                c3 = fmaAddComplex(c3, aRe, aIm, bVec7, bSw7);

						aVec = _mm256_loadu_ps(reinterpret_cast<const float*>(Y + (k+2)*N + j)); // contiguous, loaded once, used 4x
                                                aRe = _mm256_moveldup_ps(aVec);
                                                aIm = _mm256_movehdup_ps(aVec);

                                                c0 = fmaAddComplex(c0, aRe, aIm, bVec8, bSw8);
                                                c1 = fmaAddComplex(c1, aRe, aIm, bVec9, bSw9);
                                                c2 = fmaAddComplex(c2, aRe, aIm, bVec10, bSw10);
                                                c3 = fmaAddComplex(c3, aRe, aIm, bVec11, bSw11);

						aVec = _mm256_loadu_ps(reinterpret_cast<const float*>(Y + (k+3)*N + j)); // contiguous, loaded once, used 4x
                                                aRe = _mm256_moveldup_ps(aVec);
                                                aIm = _mm256_movehdup_ps(aVec);

                                                c0 = fmaAddComplex(c0, aRe, aIm, bVec12, bSw12);
                                                c1 = fmaAddComplex(c1, aRe, aIm, bVec13, bSw13);
                                                c2 = fmaAddComplex(c2, aRe, aIm, bVec14, bSw14);
                                                c3 = fmaAddComplex(c3, aRe, aIm, bVec15, bSw15);

						_mm256_storeu_ps(reinterpret_cast<float*>(C + (i+0)*N + j), c0);
                                                _mm256_storeu_ps(reinterpret_cast<float*>(C + (i+1)*N + j), c1);
                                                _mm256_storeu_ps(reinterpret_cast<float*>(C + (i+2)*N + j), c2);
                                                _mm256_storeu_ps(reinterpret_cast<float*>(C + (i+3)*N + j), c3);
					}
                                }
                        }
                }
        }
}	

                       
	



return STUDENTID;				 			 	    	 		   			 	      

}

				 			 	    	 		   			 	      
