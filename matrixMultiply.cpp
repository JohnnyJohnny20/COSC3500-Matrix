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

// Refactored helper functions by Claude
static inline void loadBRow(const floatType* Xp, int rows, int rLocal, int cLocal, __m256& bVec, __m256& bSw) {
	bVec = broadcastComplex(Xp[cLocal * rows + rLocal]);
    	bSw  = _mm256_permute_ps(bVec, 0xB1);
}

// Pack scratch buffer
static void packXPanel(const floatType* X, int N, int ii, int rows, int kk, int cols, floatType* Xp) {
	for (int c = 0; c < cols; c++) {
		for (int r = 0; r < rows; r++) {
			Xp[c * rows + r] = X[(ii + r) * N + (kk + c)];
		}
	}
}

static void packYPanel(const floatType* Y, int N, int kk, int cols, floatType* Yp) {
	for (int kL = 0; kL < cols; kL++) {
		memcpy(Yp + kL * N, Y + (kk + kL) * N, N * sizeof(floatType));
	}
}

static inline void microKernel4x4(const floatType* Y, int N, int j, int k, int i, floatType* C, const __m256 bVec[16], const __m256 bSw[16]) {
	__m256 c0 = _mm256_loadu_ps(reinterpret_cast<const float*>(C + (i+0)*N + j));
        __m256 c1 = _mm256_loadu_ps(reinterpret_cast<const float*>(C + (i+1)*N + j));
        __m256 c2 = _mm256_loadu_ps(reinterpret_cast<const float*>(C + (i+2)*N + j));
        __m256 c3 = _mm256_loadu_ps(reinterpret_cast<const float*>(C + (i+3)*N + j));

        __m256 aVec, aRe, aIm;

        aVec = _mm256_loadu_ps(reinterpret_cast<const float*>(Y + k*N + j)); // contiguous, loaded once, used 4x
        aRe = _mm256_moveldup_ps(aVec);
        aIm = _mm256_movehdup_ps(aVec);

        c0 = fmaAddComplex(c0, aRe, aIm, bVec[0], bSw[0]);
        c1 = fmaAddComplex(c1, aRe, aIm, bVec[1], bSw[1]);
        c2 = fmaAddComplex(c2, aRe, aIm, bVec[2], bSw[2]);
        c3 = fmaAddComplex(c3, aRe, aIm, bVec[3], bSw[3]);

        aVec = _mm256_loadu_ps(reinterpret_cast<const float*>(Y + (k+1)*N + j)); // contiguous, loaded once, used 4x
        aRe = _mm256_moveldup_ps(aVec);
        aIm = _mm256_movehdup_ps(aVec);

        c0 = fmaAddComplex(c0, aRe, aIm, bVec[4], bSw[4]);
        c1 = fmaAddComplex(c1, aRe, aIm, bVec[5], bSw[5]);
        c2 = fmaAddComplex(c2, aRe, aIm, bVec[6], bSw[6]);
        c3 = fmaAddComplex(c3, aRe, aIm, bVec[7], bSw[7]);

        aVec = _mm256_loadu_ps(reinterpret_cast<const float*>(Y + (k+2)*N + j)); // contiguous, loaded once, used 4x
        aRe = _mm256_moveldup_ps(aVec);
        aIm = _mm256_movehdup_ps(aVec);

        c0 = fmaAddComplex(c0, aRe, aIm, bVec[8], bSw[8]);
        c1 = fmaAddComplex(c1, aRe, aIm, bVec[9], bSw[9]);
        c2 = fmaAddComplex(c2, aRe, aIm, bVec[10], bSw[10]);
        c3 = fmaAddComplex(c3, aRe, aIm, bVec[11], bSw[11]);

        aVec = _mm256_loadu_ps(reinterpret_cast<const float*>(Y + (k+3)*N + j)); // contiguous, loaded once, used 4x
        aRe = _mm256_moveldup_ps(aVec);
        aIm = _mm256_movehdup_ps(aVec);

        c0 = fmaAddComplex(c0, aRe, aIm, bVec[12], bSw[12]);
        c1 = fmaAddComplex(c1, aRe, aIm, bVec[13], bSw[13]);
        c2 = fmaAddComplex(c2, aRe, aIm, bVec[14], bSw[14]);
        c3 = fmaAddComplex(c3, aRe, aIm, bVec[15], bSw[15]);

        _mm256_storeu_ps(reinterpret_cast<float*>(C + (i+0)*N + j), c0);
        _mm256_storeu_ps(reinterpret_cast<float*>(C + (i+1)*N + j), c1);
        _mm256_storeu_ps(reinterpret_cast<float*>(C + (i+2)*N + j), c2);
        _mm256_storeu_ps(reinterpret_cast<float*>(C + (i+3)*N + j), c3);
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

	floatType* Yp = (floatType*)malloc(blockK * N * sizeof(floatType));
	for (int kk = 0; kk < N; kk += blockK) {
        	int kEnd = kk + blockK < N ? kk + blockK : N;
                int cols = kEnd - kk;
		packYPanel(Y, N, kk, cols, Yp);

	#pragma omp parallel for num_threads(4)
	
	for (int ii = iStart; ii < iEndLimit; ii += blockI) {
		int iEnd = ii + blockI < iEndLimit ? ii + blockI : iEndLimit;
		int rows = iEnd - ii;

			floatType Xp[blockI * blockK];
			packXPanel(X, N, ii, rows, kk, cols, Xp);

			
				for (int i = ii; i + 4 <= iEnd; i += 4) {          // widened to 4 at a time
	                        	for (int k = kk; k < kEnd; k+=4) {
						int iL = i - ii, kL = k - kk;
						__m256 bVec[16], bSw[16];
		                                loadBRow(Xp, rows, iL+0, kL, bVec[0], bSw[0]);
						loadBRow(Xp, rows, iL+1, kL, bVec[1], bSw[1]);
						loadBRow(Xp, rows, iL+2, kL, bVec[2], bSw[2]);
						loadBRow(Xp, rows, iL+3, kL, bVec[3], bSw[3]);

						loadBRow(Xp, rows, iL+0, kL+1, bVec[4], bSw[4]);
	                                        loadBRow(Xp, rows, iL+1, kL+1, bVec[5], bSw[5]);
	                                        loadBRow(Xp, rows, iL+2, kL+1, bVec[6], bSw[6]);
	                                        loadBRow(Xp, rows, iL+3, kL+1, bVec[7], bSw[7]);

	                                        loadBRow(Xp, rows, iL+0, kL+2, bVec[8], bSw[8]);
	                                        loadBRow(Xp, rows, iL+1, kL+2, bVec[9], bSw[9]);
	                                        loadBRow(Xp, rows, iL+2, kL+2, bVec[10], bSw[10]);
	                                        loadBRow(Xp, rows, iL+3, kL+2, bVec[11], bSw[11]);

	                                        loadBRow(Xp, rows, iL+0, kL+3, bVec[12], bSw[12]);
	                                        loadBRow(Xp, rows, iL+1, kL+3, bVec[13], bSw[13]);
	                                        loadBRow(Xp, rows, iL+2, kL+3, bVec[14], bSw[14]);
	                                        loadBRow(Xp, rows, iL+3, kL+3, bVec[15], bSw[15]);

	                                	for (int jj = 0; jj < N; jj += blockJ) {
	                                		int jEnd = jj + blockJ < N ? jj + blockJ : N;
								for (int j = jj; j + 4 <= jEnd; j += 4) {
				                                        microKernel4x4(Yp, N, j, kL, i, C, bVec, bSw);
								}
                                }
                        }
                }
        }
}	

                       
	

free(Yp);

return STUDENTID;				 			 	    	 		   			 	      

}

				 			 	    	 		   			 	      
