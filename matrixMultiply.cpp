#include <matrixMultiply.h>
#define STUDENTID 48448239 //DO NOT REMOVE
#pragma GCC target("avx2,fma")

#define MR 4
#define NR 8
#define MC 128
#define KC 256
#define NC 512

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

static inline void interleave_and_add(__m256 cr, __m256 ci, float* C) {
        __m256 lo = _mm256_unpacklo_ps(cr, ci);
        __m256 hi = _mm256_unpackhi_ps(cr, ci);
        __m256 o0 = _mm256_permute2f128_ps(lo, hi, 0x20);
        __m256 o1 = _mm256_permute2f128_ps(lo, hi, 0x31);
        o0 = _mm256_add_ps(o0, _mm256_loadu_ps(C));
        o1 = _mm256_add_ps(o1, _mm256_loadu_ps(C + 8));
        _mm256_storeu_ps(C, o0);
        _mm256_storeu_ps(C + 8, o1);
}

// Pack scratch buffer
static void packX_4xK(const floatType* X, int N, int i, int kk, int cols, float* Xp) {
	const float* Xf = reinterpret_cast<const float*>(X);
	for (int c = 0; c < cols; c++) {
		for (int r = 0; r < 4; r++) {
			Xp[c * 8 + r] = Xf[((i + r) * N + (kk + c)) * 2];
			Xp[c * 8 + r + 4] = Xf[((i + r) * N + (kk + c)) * 2 + 1];
		}
	}
}

static void packY_Kx8(const floatType* Y, int N, int kk, int cols, int j, float* Yp) {
	const float* Yf = reinterpret_cast<const float*>(Y);
	for (int k = 0; k < cols; k++) {
                for (int c = 0; c < 8; c++) {
                        Yp[k * 16 + c]     = Yf[((kk + k) * N + (j + c)) * 2];     // Real
                        Yp[k * 16 + c + 8] = Yf[((kk + k) * N + (j + c)) * 2 + 1]; // Imag
                }
        }
}

static inline void microKernel4x8(floatType* C, int N, int i, int j, const float* Xp, const float* Yp) {

	__m256 c0Re = _mm256_setzero_ps(), c0Im = _mm256_setzero_ps();
        __m256 c1Re = _mm256_setzero_ps(), c1Im = _mm256_setzero_ps();
        __m256 c2Re = _mm256_setzero_ps(), c2Im = _mm256_setzero_ps();
        __m256 c3Re = _mm256_setzero_ps(), c3Im = _mm256_setzero_ps();
	
	for (int k = 0; k < KC; k++) {
		// Load 8 contiguous Reals and 8 contiguous Imags from Y
                __m256 yRe = _mm256_loadu_ps(Yp + k * 16);
                __m256 yIm = _mm256_loadu_ps(Yp + k * 16 + 8);

                // Broadcast 1 Real and 1 Imag from X, then pure FMA math
                __m256 x0Re = _mm256_broadcast_ss(Xp + k * 8 + 0);
                __m256 x0Im = _mm256_broadcast_ss(Xp + k * 8 + 4);
                c0Re = _mm256_fmadd_ps(x0Re, yRe, c0Re); c0Re = _mm256_fnmadd_ps(x0Im, yIm, c0Re);
                c0Im = _mm256_fmadd_ps(x0Re, yIm, c0Im); c0Im = _mm256_fmadd_ps(x0Im, yRe, c0Im);

                __m256 x1Re = _mm256_broadcast_ss(Xp + k * 8 + 1);
                __m256 x1Im = _mm256_broadcast_ss(Xp + k * 8 + 5);
                c1Re = _mm256_fmadd_ps(x1Re, yRe, c1Re); c1Re = _mm256_fnmadd_ps(x1Im, yIm, c1Re);
                c1Im = _mm256_fmadd_ps(x1Re, yIm, c1Im); c1Im = _mm256_fmadd_ps(x1Im, yRe, c1Im);

                __m256 x2Re = _mm256_broadcast_ss(Xp + k * 8 + 2);
                __m256 x2Im = _mm256_broadcast_ss(Xp + k * 8 + 6);
                c2Re = _mm256_fmadd_ps(x2Re, yRe, c2Re); c2Re = _mm256_fnmadd_ps(x2Im, yIm, c2Re);
                c2Im = _mm256_fmadd_ps(x2Re, yIm, c2Im); c2Im = _mm256_fmadd_ps(x2Im, yRe, c2Im);

                __m256 x3Re = _mm256_broadcast_ss(Xp + k * 8 + 3);
                __m256 x3Im = _mm256_broadcast_ss(Xp + k * 8 + 7);
                c3Re = _mm256_fmadd_ps(x3Re, yRe, c3Re); c3Re = _mm256_fnmadd_ps(x3Im, yIm, c3Re);
                c3Im = _mm256_fmadd_ps(x3Re, yIm, c3Im); c3Im = _mm256_fmadd_ps(x3Im, yRe, c3Im);
	}
	interleave_and_add(c0Re, c0Im, reinterpret_cast<float*>(C + (i+0)*N + j));
	interleave_and_add(c1Re, c1Im, reinterpret_cast<float*>(C + (i+1)*N + j));
	interleave_and_add(c2Re, c2Im, reinterpret_cast<float*>(C + (i+2)*N + j));
	interleave_and_add(c3Re, c3Im, reinterpret_cast<float*>(C + (i+3)*N + j));
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

	float* YpGlobal = (float*)_mm_malloc((NC / NR) * KC * 16 * sizeof(float), 64);


	for (int jj = 0; jj < N; jj += NC) {
                for (int kk = 0; kk < N; kk += KC) {
		
			// Pack Y globally using the helper function
                        #pragma omp parallel for num_threads(4)
                        for (int j = 0; j < NC; j += NR) {
                                float* YpLocal = YpGlobal + (j / NR) * KC * 16;
                                packY_Kx8(Y, N, kk, KC, jj + j, YpLocal);
                        }
		
                	#pragma omp parallel for num_threads(4)
                        for (int ii = iStart; ii < iEndLimit; ii += MC) {
                                float Xp[MC * KC * 2];

                                // Pack X locally using the helper function
                                for (int i = 0; i < MC; i += MR) {
                                        float* XpLocal = Xp + (i / MR) * KC * 8;
                                        packX_4xK(X, N, ii + i, kk, KC, XpLocal);
                                }

                                for (int j = 0; j < NC; j += NR) {
                                        const float* YpLocal = YpGlobal + (j / NR) * KC * 16;
                
                                for (int i = 0; i < MC; i += MR) {
                                        const float* XpLocal = Xp + (i / MR) * KC * 8;
                                        microKernel4x8(C, N, ii + i, jj + j, XpLocal, YpLocal);
				}
                        }
                }
        }
}


_mm_free(YpGlobal);
return STUDENTID;				 			 	    	 		   			 	      

}

				 			 	    	 		   			 	      
