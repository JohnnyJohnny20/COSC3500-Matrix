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

static inline void loadBRow(const floatType* B, int N, int row, int col, __m256& bVec, __m256& bSw) {
	bVec = broadcastComplex(B[row * N + col]);
    	bSw  = _mm256_permute_ps(bVec, 0xB1);
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

					__m256 bVec0, bVec1,bVec2, bVec3, bSw0, bSw1, bSw2, bSw3;
	                                loadBRow(X, N, i+0, k, bVec0, bSw0);
					loadBRow(X, N, i+1, k, bVec1, bSw1);
					loadBRow(X, N, i+2, k, bVec2, bSw2);
					loadBRow(X, N, i+3, k, bVec3, bSw3);

					__m256 bVec4, bVec5, bVec6, bVec7, bSw4, bSw5, bSw6, bSw7;
					loadBRow(X, N, i+0, k+1, bVec4, bSw4);
                                        loadBRow(X, N, i+1, k+1, bVec5, bSw5);
                                        loadBRow(X, N, i+2, k+1, bVec6, bSw6);
                                        loadBRow(X, N, i+3, k+1, bVec7, bSw7);
					
					__m256 bVec8, bVec9, bVec10, bVec11, bSw8, bSw9, bSw10, bSw11;
                                        loadBRow(X, N, i+0, k+2, bVec8, bSw8);
                                        loadBRow(X, N, i+1, k+2, bVec9, bSw9);
                                        loadBRow(X, N, i+2, k+2, bVec10, bSw10);
                                        loadBRow(X, N, i+3, k+2, bVec11, bSw11);

					__m256 bVec12, bVec13, bVec14, bVec15, bSw12, bSw13, bSw14, bSw15;
                                        loadBRow(X, N, i+0, k+3, bVec12, bSw12);
                                        loadBRow(X, N, i+1, k+3, bVec13, bSw13);
                                        loadBRow(X, N, i+2, k+3, bVec14, bSw14);
                                        loadBRow(X, N, i+3, k+3, bVec15, bSw15);

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

				 			 	    	 		   			 	      
