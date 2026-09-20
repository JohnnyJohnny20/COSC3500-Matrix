#include <matrixMultiplyMPI.h>
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
int matrixMultiply_MPI(int N, const floatType* A, const floatType* B, floatType* C, int* flags, int flagCount){				 			 	    	 		   			 	      
if (N<=0) { return STUDENTID;}//Your code must be able to deal with N=0 scenario without crashing.				 			 	    	 		   			 	      

//WRITE YOUR CODE HERE

	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    	MPI_Comm_size(MPI_COMM_WORLD, &size);

	int colsPerProc = N / size;
	int colStart = rank * colsPerProc;
	int colEnd = colStart + colsPerProc;
	
	int bounds[2] = {colStart, colEnd};
	
	matrixMultiply(N, A, B, C, bounds, 2);
	
	int* recvcounts = new int[size];
	int* displs = new int[size];

	int bytesPerProc = colsPerProc * 2048 * sizeof(floatType);
    
    	for (int p = 0; p < size; p++) {
        	recvcounts[p] = bytesPerProc;
        	displs[p] = p * bytesPerProc;
    	}

    	// Gather the contiguous blocks of bytes into the full matrix C
    	MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, 
                   C, recvcounts, displs, 
                   MPI_BYTE, MPI_COMM_WORLD);
	delete[] recvcounts;
	delete[] displs;
return STUDENTID;				 			 	    	 		   			 	      

}				 			 	    	 		   			 	      
