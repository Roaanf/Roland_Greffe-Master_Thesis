/*
 * GigaVoxels - GigaSpace
 *
 * Website: http://gigavoxels.inrialpes.fr/
 *
 * Contributors: GigaVoxels Team
 *
 * BSD 3-Clause License:
 *
 * Copyright (C) 2007-2015 INRIA - LJK (CNRS - Grenoble University), All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the organization nor the names  of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL COPYRIGHT HOLDER BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/** 
 * @version 1.0
 */

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// Project
#include "Kernel.h"

// Cuda
#include <cuda_runtime.h>

// System
#include <cstring>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <sstream>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/


/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ***************************** METHOD DEFINITION ******************************
 ******************************************************************************/

/******************************************************************************
 * Main entry program
 *
 * @param pArgc number of arguments
 * @param pArgv list of arguments
 *
 * @return exit code
 ******************************************************************************/
int main( int pArgc, char** pArgv )
{
	// CUDA variables used for benchmark (events, timers)
	cudaError_t cudaResult;
	cudaEvent_t startEvent;
	cudaEvent_t stopEvent;
	cudaEventCreate( &startEvent );
	cudaEventCreate( &stopEvent );

	// Declare variables
	unsigned char* h_inputData = NULL;
	unsigned char* d_inputData = NULL;
	unsigned char* d_outputData = NULL;
	const unsigned int cNbElements = 1000000;
	const unsigned int cNbIterations = 25;
	float elapsedTime = 0.0f;
	float totalElapsedTime = 0.0f;

	// Allocate data
	h_inputData = new unsigned char[ cNbElements ];
	cudaMalloc( (void**)&d_inputData, cNbElements * sizeof( unsigned char ) );
	cudaMalloc( (void**)&d_outputData, cNbElements * sizeof( unsigned char ) );

	// Initialize input data
	for ( unsigned int i = 0; i < cNbElements; i++ )
	{
		h_inputData[ i ] = i / 255;
	}
	// Copy data on device
	cudaMemcpy( d_inputData, h_inputData, cNbElements * sizeof( unsigned char ), cudaMemcpyHostToDevice );

	// Setup kernel execution parameters
	dim3 gridSize( cNbElements / 256 + 1, 1, 1 );
	dim3 blockSize( 256, 1, 1 );

	// Benchmark
	//
	// - standard kernel
	for ( unsigned int i = 0; i < cNbIterations; i++ )
	{
		// Start event
		cudaEventRecord( startEvent, 0 );

		// Launch standard kernel
		Kernel_StandardAlgorithm<<< gridSize, blockSize >>>( cNbElements, d_inputData, d_outputData );
	
		// Stop event and record elapsed time
		cudaEventRecord( stopEvent, 0 );
		cudaEventSynchronize( stopEvent );
		cudaEventElapsedTime( &elapsedTime, startEvent, stopEvent );
		totalElapsedTime += elapsedTime;

		// Checking errors
		cudaResult = cudaGetLastError();
	}
	std::cout << "Kernel_StandardAlgorithm : " << ( totalElapsedTime / cNbIterations ) << " ms" << std::endl;

	// Benchmark
	//
	// - kernel with call to device function pointer benchmark
	totalElapsedTime = 0.0f;
	for ( unsigned int i = 0; i < cNbIterations; i++ )
	{
		// Start event
		cudaEventRecord( startEvent, 0 );

		// Launch kernel with device function pointer call
		Kernel_AlgorithmWithFunctionPointer<<< gridSize, blockSize >>>( cNbElements, d_inputData, d_outputData );
	
		// Stop event and record elapsed time
		cudaEventRecord( stopEvent, 0 );
		cudaEventSynchronize( stopEvent );
		cudaEventElapsedTime( &elapsedTime, startEvent, stopEvent );
		totalElapsedTime += elapsedTime;

		// Checking errors
		cudaResult = cudaGetLastError();
	}
	std::cout << "Kernel_AlgorithmWithFunctionPointer : " << ( totalElapsedTime / cNbIterations ) << " ms" << std::endl;

	// Benchmark
	//
	// - standard kernel
	totalElapsedTime = 0.0f;
	for ( unsigned int i = 0; i < cNbIterations; i++ )
	{
		// Start event
		cudaEventRecord( startEvent, 0 );

		// Launch standard kernel
		Kernel_StandardAlgorithm_2<<< gridSize, blockSize >>>( cNbElements, d_inputData, d_outputData );
	
		// Stop event and record elapsed time
		cudaEventRecord( stopEvent, 0 );
		cudaEventSynchronize( stopEvent );
		cudaEventElapsedTime( &elapsedTime, startEvent, stopEvent );
		totalElapsedTime += elapsedTime;

		// Checking errors
		cudaResult = cudaGetLastError();
	}
	std::cout << "Kernel_StandardAlgorithm with 2 functions : " << ( totalElapsedTime / cNbIterations ) << " ms" << std::endl;

	// Benchmark
	//
	// - standard kernel
	totalElapsedTime = 0.0f;
	for ( unsigned int i = 0; i < cNbIterations; i++ )
	{
		// Start event
		cudaEventRecord( startEvent, 0 );

		// Launch standard kernel
		Kernel_AlgorithmWithFunctionPointer_2Functions<<< gridSize, blockSize >>>( cNbElements, d_inputData, d_outputData );
	
		// Stop event and record elapsed time
		cudaEventRecord( stopEvent, 0 );
		cudaEventSynchronize( stopEvent );
		cudaEventElapsedTime( &elapsedTime, startEvent, stopEvent );
		totalElapsedTime += elapsedTime;

		// Checking errors
		cudaResult = cudaGetLastError();
	}
	std::cout << "Kernel_AlgorithmWithFunctionPointer with 2 functions : " << ( totalElapsedTime / cNbIterations ) << " ms" << std::endl;

	// Clean up to ensure correct profiling
	cudaResult = cudaDeviceReset();

	return 0;
}
