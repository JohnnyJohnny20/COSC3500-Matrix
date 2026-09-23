/**
 * @file matrixMultiply.cpp
 * @brief Assignment - CPU version
 *
 * @author Johnny Hoang s48448239
 *
 * @section AI Usage
 * 1. Google Gemini was used to debug, refactor, calculate pointer arithmetic
 * and aid in implementing GotoBLAS/Blis architecture.
 *
 * 2. Claude was used to find sources, debug, refactor, conceptual explanations,
 * and review code.
 *
 * @section References
 * 1. "BLISlab: A Sandbox for Optimizing GEMM" GitHub
 *    Link: https://github.com/flame/blislab/tree/master/
 */


#include <matrixMultiply.h>
#define STUDENTID 48448239 //DO NOT REMOVE
#pragma GCC target("avx2,fma")

#define MR 4
#define NR 8
#define MC 128
#define KC 256
#define NC 512

/**
* @brief: Interleaves seperate real and imaginary SIMD vectors, accumulates into
* 	  main memory C and writes back.
*
* @note: Implemented with direct Gemini assitance to find available methods
* 	 and sequence to recombine de-interleaved AVX2 registers
*/
static inline void interleave_and_add(__m256 cr, __m256 ci, float* C) {
        __m256 lo = _mm256_unpacklo_ps(cr, ci); // interleave
        __m256 hi = _mm256_unpackhi_ps(cr, ci);
        __m256 o0 = _mm256_permute2f128_ps(lo, hi, 0x20); // reorganize
        __m256 o1 = _mm256_permute2f128_ps(lo, hi, 0x31);
        o0 = _mm256_add_ps(o0, _mm256_loadu_ps(C));
        o1 = _mm256_add_ps(o1, _mm256_loadu_ps(C + 8));
        _mm256_storeu_ps(C, o0);
        _mm256_storeu_ps(C + 8, o1);
}


/**
* @brief: Packs 4 rows of X into a contiguous L2 cache buffer, de-interleaving
*         complex values into separate real and imaginary blocks.
*
* @note: Conceptual panel packing adapted from BLISlab tutorial.pdf Section 4.1 (Ac buffer)
*	 AI assisted in pointer arithmetic.
*/
static void packX_4xK(const floatType* X, int N, int i, int kk, int cols, float* Xp) {
	const float* Xf = reinterpret_cast<const float*>(X);
	for (int c = 0; c < cols; c++) {
		for (int r = 0; r < 4; r++) {
			Xp[c * 8 + r] = Xf[((i + r) * N + (kk + c)) * 2];
			Xp[c * 8 + r + 4] = Xf[((i + r) * N + (kk + c)) * 2 + 1];
		}
	}
}

/**
* @brief: Packs 8 rows of Y into a contiguous L2 cache buffer, de-interleaving
*         complex values into separate real and imaginary blocks.
*
* @note: Conceptual panel packing adapted from BLISlab tutorial.pdf Section 4.1 (Bc buffer)
*	 AI assited in pointer arithmetic.
*/
static void packY_Kx8(const floatType* Y, int N, int kk, int cols, int j, float* Yp) {
	const float* Yf = reinterpret_cast<const float*>(Y);
	for (int k = 0; k < cols; k++) {
                for (int c = 0; c < 8; c++) {
                        Yp[k * 16 + c]     = Yf[((kk + k) * N + (j + c)) * 2];     // Real
                        Yp[k * 16 + c + 8] = Yf[((kk + k) * N + (j + c)) * 2 + 1]; // Imag
                }
        }
}

/**
 * @brief 4x8 microkernel executing the inner KC reduction loop.
 *
 * @note Corresponds to the micro-kernel loop layer in BLISlab tutorial.pdf Section 4.1.
 *       Uses the rank-1 update broadcast strategy detailed in Section 4.3
 *       (Advanced techniques), adapted for de-interleaved complex arithmetic
 *       with direct AI assistance.
 */
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
* @note Implements the 5-loop cache-blocking hierarchy (NC -> KC -> MC -> NR -> MR)
*       described in BLISlab Section tutorial.pdf 4.1 (Step 3: Blocking for Multiple Levels
*       of Cache), parallelized using OpenMP as outlined in Section 5
*
* Notation follows BLISlab tutorial.pdf Section 4.1, Figure 3:
*   NC, KC, MC : cache-blocking sizes for the 3 outer loops (L3/L2 panel sizes)
*   NR, MR     : register-blocking sizes for the microkernel (accumulator tile)
*   jc, pc, ic : outer loop indices (Loop 5, 4, 3 in Figure 3)
*   jr, ir     : microkernel loop indices (Loop 2, 1)
*/
int matrixMultiply(int N, const floatType* A, const floatType* B, floatType* C, int* args, int argCount) {		
if (N<=0) { return STUDENTID;}//Your code must be able to deal with N=0 scenario without crashing.				 			 	    	 		   			 	      
//WRITE YOUR CODE HERE
	
	const floatType* X = B; // transpose
	const floatType* Y = A;
	
	int iStart = 0; // for MPI
	int iEndLimit = N;
	
	if (argCount == 2) {
        	iStart = args[0];
        	iEndLimit = args[1];
    	}

	// Zero C
	#pragma omp parallel for num_threads(4)
	for (int i = iStart; i < iEndLimit ; i++) {
		memset(C + i * N, 0, N * sizeof(floatType));
	}

	float* YpGlobal = (float*)_mm_malloc((NC / NR) * KC * 16 * sizeof(float), 64);

	// GotoBLAS/Blis hierarchy structure for optimal cache hits
	// See BLISlab tutorial.pdf section 4.1 figure 3
	for (int jc = 0; jc < N; jc += NC) {
                for (int pc = 0; pc < N; pc += KC) {
		
			// Pack Y globally using the helper function
                        #pragma omp parallel for num_threads(4)
                        for (int j = 0; j < NC; j += NR) {
                                float* YpLocal = YpGlobal + (j / NR) * KC * 16;
                                packY_Kx8(Y, N, pc, KC, jc + j, YpLocal);
                        }
		
                	#pragma omp parallel for num_threads(4)
                        for (int ic = iStart; ic < iEndLimit; ic += MC) {
                                float Xp[MC * KC * 2];

                                // Pack X locally using the helper function
                                for (int i = 0; i < MC; i += MR) {
                                        float* XpLocal = Xp + (i / MR) * KC * 8;
                                        packX_4xK(X, N, ic + i, pc, KC, XpLocal);
                                }
				
                                for (int jr = 0; jr < NC; jr += NR) {
                                        const float* YpLocal = YpGlobal + (jr / NR) * KC * 16;
                
	                                for (int ir = 0; ir < MC; ir += MR) {
	                                        const float* XpLocal = Xp + (ir / MR) * KC * 8;
	                                        microKernel4x8(C, N, ic + ir, jc + jr, XpLocal, YpLocal);
					}
                        	}
                	}
        	}
	}


_mm_free(YpGlobal);
return STUDENTID;				 			 	    	 		   			 	      

}

				 			 	    	 		   			 	      
