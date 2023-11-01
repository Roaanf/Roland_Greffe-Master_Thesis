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
#include "RLECompressor.h"
#include "RLECompressorKernel.h"

// Cuda
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

// System
#include <cstring>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <sstream>

#include "macros.h"

#include <sys/time.h>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

//#define TEST_OTHER_PASSTHROUGH_KERNEL

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
	// CUDA flags to use memory mapped feature
	cudaSetDevice( 0 );
	cudaSetDeviceFlags( cudaDeviceMapHost );
	cudaProfilerStart();
	cudaError_t cudaResult;

	// CUDA events to time the execution time of the kernels
	cudaEvent_t start, stop;
	cudaEventCreate( &start );
	cudaEventCreate( &stop );

	// Initialize RLE encoder
	RLECompressor* compressor = new RLECompressor();

	// Input
	unsigned int* inputHost;
	unsigned int* inputDevice;
	cudaHostAlloc( (void**)&inputHost, STR_SIZE * sizeof( unsigned int ), cudaHostAllocMapped | cudaHostAllocWriteCombined );
	cudaHostGetDevicePointer( (void**)&inputDevice, (void*)inputHost, 0 );
	unsigned int *input = static_cast< unsigned int* >( malloc( STR_SIZE * sizeof( unsigned int )));

	// Compressed input
	unsigned int *bricksEndsHost, *bricksEndsDevice;
	unsigned int *plateausValuesHost, *plateausValuesDevice;
	unsigned char *plateausStartsHost, *plateausStartsDevice; 
	cudaHostAlloc( (void**)&bricksEndsHost, N_BRICKS * sizeof( unsigned int ), cudaHostAllocMapped | cudaHostAllocWriteCombined );
	cudaHostGetDevicePointer( (void**)&bricksEndsDevice, (void*)bricksEndsHost, 0 );
	cudaHostAlloc( (void**)&plateausValuesHost, STR_SIZE * sizeof( unsigned int ), cudaHostAllocMapped | cudaHostAllocWriteCombined );
	cudaHostGetDevicePointer( (void**)&plateausValuesDevice, (void*)plateausValuesHost, 0 );
	cudaHostAlloc( (void**)&plateausStartsHost, STR_SIZE * sizeof( unsigned char ), cudaHostAllocMapped | cudaHostAllocWriteCombined );
	cudaHostGetDevicePointer( (void**)&plateausStartsDevice, (void*)plateausStartsHost, 0 );

	// Output
	unsigned int* outputHost = static_cast< unsigned int * >( malloc( sizeof( unsigned int ) * STR_SIZE ));
	unsigned int* outputDevice;
	cudaMalloc( (void**)&outputDevice, STR_SIZE * sizeof( unsigned int ) );

	// Setting up the log infos
	unsigned int it = 10; // Number of time each kernel will be executed for each compression rate.
	float elapsedTime, totalTime = 0;

    //FILE * f = fopen("results.txt","w");

    // Kernel execution parameters
	dim3 gridPassThrough( N_BRICKS, 1, 1 );
	dim3 blockSizePassThrough( 192 , 1, 1 );

	dim3 gridRLE( GRID_SIZE, 1, 1 );
	dim3 blockSizeRLE( N_THREADS_PER_BLOCS, 1, 1 );

	// Loop on the compression rate
    for ( unsigned int compressionRate = 1; compressionRate <= MAX_COMPRESSION; ++compressionRate )
    {
		// Fill input array
        for ( unsigned int i = 0; i < STR_SIZE; ++i )
        {
            input[i] = static_cast< unsigned int > ( UINT_MAX - i / compressionRate );
		}

		// Copy input array to mapped memory
		memcpy( inputHost, input, STR_SIZE * sizeof( unsigned int ));

		// RLE compression
		struct timeval tv;
        gettimeofday( &tv, NULL );
		unsigned long timeStampBeforeCompression = 1000000 * tv.tv_sec + tv.tv_usec;

        // ...
		compressor->compressionPrefixSum( input, N_BRICKS, bricksEndsHost, plateausValuesHost, plateausStartsHost );

        gettimeofday( &tv, NULL );
		unsigned long timeStampAfterCompression = ( 1000000 * tv.tv_sec + tv.tv_usec );

		float timeCompression = (float)( timeStampAfterCompression - timeStampBeforeCompression ) / 1000.f;

        totalTime = 0;

		// ---------------------------------------------------------------------------------------------
		// COMPARAISON VALUES : cudaMemcpy
		for ( unsigned int iteration = 0; iteration < it; ++iteration ) {

			// Reinitialise output array
			cudaMemsetAsync( outputDevice, 0, STR_SIZE * sizeof( unsigned int ) );

			// Start event
			cudaEventRecord( start, 0 );

			// Simple recopy in the output array
			cudaMemcpy( outputDevice, input, STR_SIZE * sizeof( unsigned int ), cudaMemcpyHostToDevice );

			cudaEventRecord( stop, 0 );
			cudaEventSynchronize( stop );
			cudaEventElapsedTime( &elapsedTime, start, stop );

			// Checking errors
			cudaResult = cudaGetLastError();
			if ( cudaResult != cudaSuccess ) {
				std::cerr << "Error in the test:\n" << std::endl;
				std::cerr << cudaGetErrorString( cudaResult ) << std::endl;
			}

			// Stop event and record elapsed time
			totalTime += elapsedTime;

			// Check that the copy succeeded
			// Copy result from Device to Host
			cudaMemcpyAsync( outputHost, outputDevice, STR_SIZE * sizeof( unsigned int ), cudaMemcpyDeviceToHost );
			unsigned int index = 0;
			while ( index < STR_SIZE  ) {
				if( input[index] != outputHost[index] ) {
					break;
				}
				++index;
			}
			if ( index != STR_SIZE ) {
				std::cerr << "test: error line " << index << " for the compression rate " <<
					compressionRate << "; " << input[index] << " != " << outputHost[index] << std::endl;
				exit(1);
			}
		}

		float memCopy = totalTime / it;
		totalTime = 0;

		// ---------------------------------------------------------------------------------------------

		// ---------------------------------------------------------------------------------------------
		// COMPARAISON VALUES : cudaMemcpy + pinned memory
		for ( unsigned int iteration = 0; iteration < it; ++iteration ) {

			// Reinitialise output array
			cudaMemsetAsync( outputDevice, 0, STR_SIZE * sizeof( unsigned int ) );

			// Start event
			cudaEventRecord( start, 0 );

			// Simple recopy in the output array
			cudaMemcpy( outputDevice, inputHost, STR_SIZE * sizeof( unsigned int ), cudaMemcpyHostToDevice );

			cudaEventRecord( stop, 0 );
			cudaEventSynchronize( stop );
			cudaEventElapsedTime( &elapsedTime, start, stop );

			// Checking errors
			cudaResult = cudaGetLastError();
			if ( cudaResult != cudaSuccess ) {
				std::cerr << "Error in the test:\n" << std::endl;
				std::cerr << cudaGetErrorString( cudaResult ) << std::endl;
			}

			// Stop event and record elapsed time
			totalTime += elapsedTime;

			// Check that the copy succeeded
			// Copy result from Device to Host
			cudaMemcpyAsync( outputHost, outputDevice, STR_SIZE * sizeof( unsigned int ), cudaMemcpyDeviceToHost );
			unsigned int index = 0;
			while ( index < STR_SIZE  ) {
				if( input[index] != outputHost[index] ) {
					break;
				}
				++index;
			}
			if ( index != STR_SIZE ) {
				std::cerr << "test: error line " << index << " for the compression rate " <<
					compressionRate << "; " << input[index] << " != " << outputHost[index] << std::endl;
				exit(1);
			}
		}

		float pinnedMemCopy = totalTime / it;
		totalTime = 0;

		// ---------------------------------------------------------------------------------------------
		// FIRST METHOD : brute force copy in the mapped memory
		for ( unsigned int iteration = 0; iteration < it; ++iteration ) {

			// Reinitialise output array
			cudaMemsetAsync( outputDevice, 0, STR_SIZE * sizeof( unsigned int ) );

			// Start event
			cudaEventRecord( start, 0 );

			// Simple recopy in the output array
			Kernel_PassThrough<<< gridPassThrough, blockSizePassThrough >>>( STR_SIZE, inputDevice, outputDevice );

			cudaEventRecord( stop, 0 );
			cudaEventSynchronize( stop );
			cudaEventElapsedTime( &elapsedTime, start, stop );

			// Checking errors
			cudaResult = cudaGetLastError();
			if ( cudaResult != cudaSuccess ) {
				std::cerr << "Error in the Pass-Through kernel:\n" << std::endl;
				std::cerr << cudaGetErrorString( cudaResult ) << std::endl;
			}

			// Stop event and record elapsed time
			totalTime += elapsedTime;

			// Check that the copy succeeded
			// Copy result from Device to Host
			cudaMemcpyAsync( outputHost, outputDevice, STR_SIZE * sizeof( unsigned int ), cudaMemcpyDeviceToHost );
			unsigned int index = 0;
			while ( index < STR_SIZE  ) {
				if( input[index] != outputHost[index] ) {
					break;
				}
				++index;
			}
			if ( index != STR_SIZE ) {
				std::cerr << "Pass-through: error line " << index << " for the compression rate " <<
					compressionRate << "; " << input[index] << " != " << outputHost[index] << std::endl;
				exit(1);
			}
		}

		float passThrough = totalTime / it;

        // ---------------------------------------------------------------------------------------------

#ifdef TEST_OTHER_PASSTHROUGH_KERNEL

		// Reset total elapsed time
		totalTime = 0.f;

        // FIRST METHOD : brute force copy in the mapped memory
        for ( unsigned int iteration = 0; iteration < it; ++iteration ) {

            // Reinitialise output array
            cudaMemsetAsync( outputDevice, 0, STR_SIZE * sizeof( unsigned int ) );

            // Start event
            cudaEventRecord( start, 0 );

            // Kernel launch configuration
            const unsigned int nbThreads = /*N_THREADS_PER_BLOCS*/64; // 192
            const unsigned int nbBlocks = STR_SIZE / nbThreads + 1; // 500000 / 192 = 2604.16666667
            dim3 blockSizePassThrough_Pascal( nbThreads, 1, 1 );
            dim3 gridPassThrough_Pascal( nbBlocks, 1, 1 );

            // Simple recopy in the output array
            Kernel_PassThrough_Pascal<<< gridPassThrough_Pascal, blockSizePassThrough_Pascal >>>( STR_SIZE, inputDevice, outputDevice );

            cudaEventRecord( stop, 0 );
            cudaEventSynchronize( stop );
            cudaEventElapsedTime( &elapsedTime, start, stop );

            // Checking errors
            cudaResult = cudaGetLastError();
            if ( cudaResult != cudaSuccess ) {
                std::cerr << "Error in the Pass-Through kernel:\n" << std::endl;
                std::cerr << cudaGetErrorString( cudaResult ) << std::endl;
            }

            // Stop event and record elapsed time
            totalTime += elapsedTime;

            // Check that the copy succeeded
            // Copy result from Device to Host
            cudaMemcpyAsync( outputHost, outputDevice, STR_SIZE * sizeof( unsigned int ), cudaMemcpyDeviceToHost );
            unsigned int index = 0;
            while ( index < STR_SIZE  ) {
                if( input[index] != outputHost[index] ) {
                    break;
                }
                ++index;
            }
            if ( index != STR_SIZE ) {
                std::cerr << "Pass-through Pascal: error line " << index << " for the compression rate " <<
                    compressionRate << "; " << input[index] << " != " << outputHost[index] << std::endl;
                exit(1);
            }
        }

        float passThrough_Pascal = totalTime / it;

#endif // TEST_OTHER_PASSTHROUGH_KERNEL

        // ---------------------------------------------------------------------------------------------
        // Reset total elapsed time
        totalTime = 0.f;

		//// ---------------------------------------------------------------------------------------------
		//// SECOND METHOD : simple adding

		// Execute the kernel 'it' times.
		for ( unsigned int i = 0; i < it; ++i ) {
			// Reinitialise output array
			cudaMemsetAsync( outputDevice, 0, STR_SIZE * sizeof( unsigned char ) );
		
			// Start event
			cudaEventRecord( start, 0 );
		
			// Decompressing on the GPU
			simpleRleKernel<<< gridRLE, blockSizeRLE >>>( 
					bricksEndsDevice,
				   	plateausValuesDevice, 
					plateausStartsDevice,
					outputDevice );

			// Stop event and record elapsed time
			cudaEventRecord( stop, 0 );
			cudaEventSynchronize( stop );
			cudaEventElapsedTime( &elapsedTime, start, stop );
			cudaResult = cudaGetLastError();
			// Error check
			if ( cudaResult != cudaSuccess ) {
				std::cerr << "Error in the RLE kernel:\n" << std::endl;
				std::cerr << cudaGetErrorString( cudaResult ) << std::endl;
			}
		
			totalTime += elapsedTime;

			// Check that the copy succeeded
			// Copy result from Device to Host
			cudaMemcpyAsync( outputHost, outputDevice, STR_SIZE * sizeof( unsigned int ), cudaMemcpyDeviceToHost );

			unsigned int index = 0;
			while ( index < STR_SIZE  ) {
				if( input[index] != outputHost[index] ) {
					break;
				}
				++index;
			}
			if ( index != STR_SIZE ) {
				std::cerr << "RLEPrefix: error line " << index << " for the compression rate " <<
					compressionRate << "; " << (unsigned int) input[index] 
					<< " != " << (unsigned int) outputHost[index] << std::endl;
				exit( 1 );
			}
		}

		float simpleRLE = totalTime / it;

		// ---------------------------------------------------------------------------------------------


		// Reset total elapsed time
		totalTime = 0.f;
		// ---------------------------------------------------------------------------------------------
		// THIRD METHOD : simple adding with prefix sum
		// Execute the kernel 'it' times.
		for ( unsigned int i = 0; i < it; ++i ) {
			// Reinitialise output array
			cudaMemsetAsync( outputDevice, 0, STR_SIZE * sizeof( unsigned char ) );
		
			// Start event
			cudaEventRecord( start, 0 );
		
			// Decompressing on the GPU
			dichoRleKernel<<< gridRLE, blockSizeRLE >>>( 
					bricksEndsDevice,
				   	plateausValuesDevice, 
					plateausStartsDevice,
					outputDevice );

			// Stop event and record elapsed time
			cudaEventRecord( stop, 0 );
			cudaEventSynchronize( stop );
			cudaEventElapsedTime( &elapsedTime, start, stop );
			cudaResult = cudaGetLastError();
			// Error check
			if ( cudaResult != cudaSuccess ) {
				std::cerr << "Error in the RLE kernel:\n" << std::endl;
				std::cerr << cudaGetErrorString( cudaResult ) << std::endl;
			}
		
			totalTime += elapsedTime;

			// Check that the copy succeeded
			// Copy result from Device to Host
			cudaMemcpyAsync( outputHost, outputDevice, STR_SIZE * sizeof( unsigned int ), cudaMemcpyDeviceToHost );

			unsigned int index = 0;
			while ( index < STR_SIZE  ) {
				if( input[index] != outputHost[index] ) {
					break;
				}
				++index;
			}
			if ( index != STR_SIZE ) {
				std::cerr << "RLEPrefix: error line " << index << " for the compression rate " <<
					compressionRate << "; " << (unsigned int) input[index] 
					<< " != " << (unsigned int) outputHost[index] << std::endl;
				exit( 1 );
			}
		}

		float dichoRLE = totalTime / it;

		// ---------------------------------------------------------------------------------------------

		// Reset total elapsed time
		totalTime = 0.f;

		// ---------------------------------------------------------------------------------------------
		// FIFTH METHOD : simple adding with prefix sum
		// Execute the kernel 'it' times.
		for ( unsigned int i = 0; i < it; ++i ) {
			// Reinitialise output array
			cudaMemsetAsync( outputDevice, 0, STR_SIZE * sizeof( unsigned char ) );
		
			// Start event
			cudaEventRecord( start, 0 );
		
			// Decompressing on the GPU
			combinedRleKernel<<< gridRLE, blockSizeRLE >>>( 
					bricksEndsDevice,
				   	plateausValuesDevice, 
					plateausStartsDevice,
					outputDevice );

			// Stop event and record elapsed time
			cudaEventRecord( stop, 0 );
			cudaEventSynchronize( stop );
			cudaEventElapsedTime( &elapsedTime, start, stop );
			cudaResult = cudaGetLastError();
			// Error check
			if ( cudaResult != cudaSuccess ) {
				std::cerr << "Error in the RLE kernel:\n" << std::endl;
				std::cerr << cudaGetErrorString( cudaResult ) << std::endl;
			}
		
			totalTime += elapsedTime;

			// Check that the copy succeeded
			// Copy result from Device to Host
			cudaMemcpyAsync( outputHost, outputDevice, STR_SIZE * sizeof( unsigned int ), cudaMemcpyDeviceToHost );

			unsigned int index = 0;
			while ( index < STR_SIZE  ) {
				if( input[index] != outputHost[index] ) {
					break;
				}
				++index;
			}
			if ( index != STR_SIZE ) {
				std::cerr << "RLEPrefix: error line " << index << " for the compression rate " <<
					compressionRate << "; " << (unsigned int) input[index] 
					<< " != " << (unsigned int) outputHost[index] << std::endl;
				exit( 1 );
			}
		}

		float combinedRLE = totalTime / it;

		// Display results
		std::cout << compressionRate 
			<< ";" << memCopy
			<< ";" << pinnedMemCopy
#ifndef TEST_OTHER_PASSTHROUGH_KERNEL
            << ";" << passThrough
            << ";" << simpleRLE
#else TEST_OTHER_PASSTHROUGH_KERNEL
            << ";----" << passThrough
            << "----;----" << passThrough_Pascal
            << (( passThrough_Pascal < passThrough ) ? " FASTER " : "")
            << "----;" << simpleRLE
#endif
			<< ";" << dichoRLE 
			<< ";" << combinedRLE 
			<< ";" << timeCompression
			<< std::endl;
	}

	// Free memory
	free( input );
	cudaFreeHost( inputHost );

	free( outputHost );
	cudaFree( outputDevice );

	cudaFreeHost( plateausStartsHost );
	cudaFreeHost( plateausValuesHost );
	cudaFreeHost( bricksEndsHost );
	cudaEventDestroy( start );
	cudaEventDestroy( stop );

	//fclose(f);
	cudaProfilerStop();

	return 0;
}
