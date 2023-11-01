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

//#include <cudaProfiler.h>
#include <cuda_profiler_api.h>

namespace GvCollision {

/******************************************************************************
 * TODO
 ******************************************************************************/
template< class TVolTreeKernelType >
bool collision_Point_VolTree( TVolTreeKernelType pVolumeTree,
		    float3 pPoint,
		    float pPrecision ) {

	collision_Point_VolTree_Kernel< TVolTreeKernelType > <<< 1,1 >>>( 
				pVolumeTree,
				pPoint,
				pPrecision
			);

	bool ret;
	GS_CUDA_SAFE_CALL( cudaMemcpyFromSymbol( &ret, GvCollision::collision, sizeof( ret ), 0, cudaMemcpyDeviceToHost ) );

	return ret;
}

/******************************************************************************
 * TODO
 ******************************************************************************/
template< class TVolTreeKernelType >
void collision_BBOX_VolTree(
			const TVolTreeKernelType &volumeTree,
		    const std::vector< unsigned int > &precisions,
	   		const std::vector< float3 > &positions,
			const std::vector< float3 > &extents,
			const std::vector< float4x4 > &basis,
	   		std::vector< float > &results )
{
	//CUresult error;
	cudaError_t error;
	
	//error = cuProfilerStart();
	//error = cudaProfilerStart();
	//if ( error == cudaErrorProfilerNotInitialized )
	//if ( error != cudaSuccess )
	//{
	//	//std::cout << "ERROR : cuProfilerStart() is called without initializing profiler" << std::endl;
	//	std::cout << "ERROR : cudaProfilerStart() is called without initializing profiler" << std::endl;
	//}

	// Events to time the collision time
	cudaEvent_t startCollision, stopCollision;
	cudaEventCreate( &startCollision );
	cudaEventCreate( &stopCollision );

	uint arraysSize = precisions.size();

	// Enforce arrays size
	assert( positions.size() == arraysSize );
	assert( extents.size() == arraysSize );
	assert( basis.size() == arraysSize );

	// Allocate memory on the device side and copy arrays to said memory
	// TODO gestion des erreurs
	unsigned int *devicePrecisions;
	float3 *devicePositions;
	float3 *deviceExtents;
	float4x4 *deviceBasis;
	float *deviceResults;

	cudaMalloc(( void ** )&devicePrecisions, arraysSize * sizeof( unsigned int ));
	cudaMalloc(( void ** )&devicePositions, arraysSize * sizeof( float3 ));
	cudaMalloc(( void ** )&deviceExtents, arraysSize * sizeof( float3 ));
	cudaMalloc(( void ** )&deviceBasis, arraysSize * sizeof( float4x4 ));
	cudaMalloc(( void ** )&deviceResults, arraysSize * sizeof( float ));

	GV_CHECK_CUDA_ERROR( "collision_BBOX_VolTree : allocation" );

	// Copy arrays to device
	// TODO gestion des erreurs
	cudaMemcpy( devicePrecisions, &precisions[0], arraysSize * sizeof( unsigned int ), cudaMemcpyHostToDevice );
	cudaMemcpy( devicePositions, &positions[0], arraysSize * sizeof( float3 ), cudaMemcpyHostToDevice );
	cudaMemcpy( deviceExtents, &extents[0], arraysSize * sizeof( float3 ), cudaMemcpyHostToDevice );
	cudaMemcpy( deviceBasis, &basis[0], arraysSize * sizeof( float4x4 ), cudaMemcpyHostToDevice );

	GV_CHECK_CUDA_ERROR( "collision_BBOX_VolTree : copy parameters" );

	// Call the kernel
	// TODO : nb threads/blocs
	cudaEventRecord( startCollision, 0 );
	collision_BBOX_VolTree_Kernel< TVolTreeKernelType > <<< arraysSize, 1 >>>( 
				volumeTree,
				devicePrecisions,
				devicePositions,
				deviceExtents,
				deviceBasis,
				deviceResults,
				arraysSize
			);
	cudaEventRecord( stopCollision, 0 );
	cudaEventSynchronize( stopCollision );

	float elapsedTime;
	cudaEventElapsedTime( &elapsedTime, startCollision, stopCollision );

	std::cout << "Time : " << elapsedTime << "ms" << std::endl;
	
	results.resize( arraysSize );

	GV_CHECK_CUDA_ERROR( "collision_BBOX_VolTree : kernel call" );

	// Copy the results back
	cudaMemcpy( &results[0], deviceResults, arraysSize * sizeof( float ), cudaMemcpyDeviceToHost );
	GV_CHECK_CUDA_ERROR( "collision_BBOX_VolTree : copy results" );

	// Free the memory.
	cudaFree( devicePrecisions );
	cudaFree( devicePositions );
	cudaFree( deviceExtents );
	cudaFree( deviceBasis );
	cudaFree( deviceResults );

	GV_CHECK_CUDA_ERROR( "collision_BBOX_VolTree : free memory" );

	cudaEventDestroy( startCollision );
	cudaEventDestroy( stopCollision );
	GV_CHECK_CUDA_ERROR( "collision_BBOX_VolTree : events" );

	//error = cuProfilerStop();
	//error = cudaProfilerStop();
	//if ( error == cudaErrorProfilerNotInitialized )
	//if ( error != cudaSuccess )
	//{
	//	//std::cout << "ERROR : cuProfilerStop() is called without initializing profiler" << std::endl;
	//	std::cout << "ERROR : cudaProfilerStop() is called without initializing profiler" << std::endl;
	//}
}

}; // GvCollision
