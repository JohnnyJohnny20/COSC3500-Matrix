#include <matrixMultiply.h>
#define STUDENTID 48448239 //DO NOT REMOVE
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

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			floatType sum = 0;
			for (int  k = 0; k < N; k++) {
				sum += A[i * N + k] * B[k * N + j];
			}
			C[i * N + j] = sum;
		}
	}


return STUDENTID;				 			 	    	 		   			 	      

}
				 			 	    	 		   			 	      
