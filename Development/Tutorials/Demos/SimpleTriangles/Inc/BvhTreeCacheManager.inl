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

// GigaVoxels
#include <GvCache/GsCacheManagerResources.h>

/******************************************************************************
 ****************************** INLINE DEFINITION *****************************
 ******************************************************************************/

/******************************************************************************
 * Constructor
 *
 * @param cacheSize ...
 * @param elemSize ...
 ******************************************************************************/
template< unsigned int TId, typename ElementRes >
GPUCacheManager< TId, ElementRes >::GPUCacheManager( uint3 cacheSize, uint3 elemSize )
:	_cacheSize( cacheSize )
{
	_elemsCacheSize = _cacheSize / elemSize;

	d_TimeStampArray = new GvCore::GsLinearMemory< uint >( _elemsCacheSize );
	d_TimeStampArray->fill( 0 );

	uint numElements = _elemsCacheSize.x * _elemsCacheSize.y * _elemsCacheSize.z - cNumLockedElements;

	d_elemAddressList		= new GvCore::GsLinearMemory< uint >( make_uint3( numElements, 1, 1 ) );
	d_elemAddressListTmp	= new GvCore::GsLinearMemory< uint >( make_uint3( numElements, 1, 1 ) );

	d_TempMaskList			= new thrust::device_vector< uint >( numElements );
	d_TempMaskList2			= new thrust::device_vector< uint >( numElements );

	// Initialize
	thrust::fill( d_TempMaskList->begin(), d_TempMaskList->end(), 0 );
	thrust::fill( d_TempMaskList2->begin(), d_TempMaskList2->end(), 0 );

	// TODO
	/*uint3 pageTableRes=d_pageTableArray->getResolution();
	uint pageTableResLinear=pageTableRes.x*pageTableRes.y*pageTableRes.z;*/
	//uint pageTableResLinear = 4000000;//BVH_VERTEX_POOL_SIZE;
	uint pageTableResLinear = 4147426;//BVH_NODE_POOL_SIZE

	//d_TempUpdateMaskList = GPUCacheManagerResources::getTempUsageMask1(pageTableRes.x*pageTableRes.y*pageTableRes.z); 
	d_TempUpdateMaskList	= new thrust::device_vector< uint >( pageTableResLinear );
	d_UpdateCompactList		= new GvCore::GsLinearMemory< uint >( make_uint3( pageTableResLinear, 1, 1 ) );

	// Initialization
	GvCore::Array3D< uint > tmpelemaddress( make_uint3( _elemsCacheSize.x, 1, 1 ) );
	for ( uint pos = 0; pos < _elemsCacheSize.x; pos++ )	// TODO : only 1D, is that OK ?
	{
		//tmpelemaddress.push_back( pos );
		tmpelemaddress.get( pos ) = pos;
	}
	// Dont use element zero !
	GvCore::memcpyArray( d_elemAddressList, tmpelemaddress.getPointer() + cNumLockedElements, numElements );

	uint cudppNumElem = std::max( pageTableResLinear, numElements );
	scanplan = GvCache::GsCacheManagerResources::getScanPlan( cudppNumElem );
	GS_CUDA_SAFE_CALL( cudaMalloc( (void**) &d_numElementsPtr, sizeof( size_t ) ) );
}

/******************************************************************************
 * Handle requests
 *
 * @param updateList input list of node addresses that have been updated with "subdivision" or "load/produce" requests
 * @param numUpdateElems maximum number of elements to process
 * @param updateMask a unique given type of requests to take into account
 * @param maxNumElems ...
 * @param numValidNodes ...
 * @param gpuPool associated pool (nodes or bricks)
 *
 * @return ...
 ******************************************************************************/
template< unsigned int TId, typename ElementRes >
template< typename GPUPoolType, typename TProducerType >
uint GPUCacheManager< TId, ElementRes >
::genericWrite( uint* updateList, uint numUpdateElems, uint updateMask,
			   uint maxNumElems, uint numValidNodes, const GPUPoolType& gpuPool, TProducerType* pProducer )
{
	assert( pProducer != NULL );
	// TO DO : check pProducer at run-time

	uint numElems = 0;
	//uint providerId = GPUProviderType::ProviderId::value;

	if ( numUpdateElems > 0 )
	{
		// [ 1 ] - Create the list of nodes that will be concerned by the data production management
		//
		// Only nodes whose request correponds to the given "updateMask" will be selected.
		//
		// The resulting list will be placed in [ d_UpdateCompactList ]

		//CUDAPM_START_EVENT_CHANNEL( 0, providerId, gpucache_nodes_manageUpdates );
		numElems = createUpdateList( updateList, numUpdateElems, updateMask );
		//CUDAPM_STOP_EVENT_CHANNEL( 0, providerId, gpucache_nodes_manageUpdates );

		// Prevent loading more than the cache size
		numElems = std::min( numElems, getNumElements() );
		
		//-----------------------------------
		// QUESTION : à quoi sert le test ?
		// - ça arrive sur la "simple sphere" quand on augmente trop le depth
		//-----------------------------------
		//if (numElems > numElemsNotUsed)
		//{
		//	std::cout << "CacheManager<" << providerId << ">: Warning: "
		//		<< numElemsNotUsed << " slots available!" << std::endl;
		//}

		///numElems = std::min(numElems, numElemsNotUsed);	// Prevent replacing elements in use
		///numElems = std::min(numElems, maxNumElems);		// Smooth loading

		if ( numElems > 0 )
		{
			//std::cout << "CacheManager<" << providerId << ">: " << numElems << " requests" << std::endl;

			// Invalidation phase
			//totalNumLoads += numElems;
			//lastNumLoads = numElems;

			/*CUDAPM_START_EVENT_CHANNEL( 1, providerId, gpucache_bricks_bricksInvalidation );*/
			//invalidateElements( numElems, numValidNodes );
			/*CUDAPM_STOP_EVENT_CHANNEL( 1, providerId, gpucache_bricks_bricksInvalidation );*/

			//CUDAPM_START_EVENT_CHANNEL( 0, providerId, gpucache_nodes_subdivKernel );
			//CUDAPM_START_EVENT_CHANNEL( 1, providerId, gpucache_bricks_gpuFetchBricks );
			//CUDAPM_EVENT_NUMELEMS_CHANNEL( 1, providerId, gpucache_bricks_gpuFetchBricks, numElems );

			// Write new elements into the cache
			GvCore::GsLinearMemory< uint >* nodesAddressCompactList = d_UpdateCompactList;	// list of nodes to produce
			GvCore::GsLinearMemory< uint >* elemsAddressCompactList = d_elemAddressList;		// ...

#if CUDAPERFMON_CACHE_INFO==1
			{
				dim3 blockSize(64, 1, 1);
				uint numBlocks=iDivUp(numElems, blockSize.x);
				dim3 gridSize=dim3( std::min( numBlocks, 32768U) , iDivUp(numBlocks,32768U), 1);

				SyntheticInfo_Update_DataWrite< ElementRes, AddressType ><<<gridSize, blockSize, 0>>>(
					d_CacheStateBufferArray->getPointer(), numElems,
					thrust::raw_pointer_cast(&(*elemsAddressCompactList)[0]),
					elemsCacheSize);

				GV_CHECK_CUDA_ERROR("SyntheticInfo_Update_DataWrite");
			}

			numPagesWrited = numElems;
#endif
			// Ask the HOST producer to generate its data
			pProducer->produceData( numElems, nodesAddressCompactList, elemsAddressCompactList, Id() );

			//CUDAPM_STOP_EVENT_CHANNEL( 0, providerId, gpucache_nodes_subdivKernel );
			//CUDAPM_STOP_EVENT_CHANNEL( 1, providerId, gpucache_bricks_gpuFetchBricks );
		}
	}

	return numElems;
}

/******************************************************************************
 * Create the list of nodes that will be concerned by the data production management
 *
 * @param inputList input list of node addresses that have been updated with "subdivision" or "load/produce" requests
 * @param inputNumElem maximum number of elements to process
 * @param testFlag a unique given type of requests to take into account
 *
 * @return the number of requests that the manager will have to handle
 ******************************************************************************/
template< unsigned int TId, typename ElementRes >
uint GPUCacheManager< TId, ElementRes >
::createUpdateList( uint* inputList, uint inputNumElem, uint pTestFlag )
{
	// ---- [ 1 ] ---- 1st step
	//
	// Fill the buffer of masks of valid elements whose attribute is equal to "pTestFlag"
	// Result is placed in "_d_TempUpdateMaskList"

	CUDAPM_START_EVENT( gpucachemgr_createUpdateList_createMask );

	// Set kernel execution configuration
	dim3 blockSize( 64, 1, 1 );
	uint numBlocks = iDivUp( inputNumElem, blockSize.x );
	dim3 gridSize = dim3( std::min( numBlocks, 65535U ) , iDivUp( numBlocks,65535U ), 1 );

	// Given a flag indicating a type of request (subdivision or load), and the updated global buffer of requests,
	// it fills a resulting mask buffer.
	// In fact, this mask indicates, for each element of the usage list, if it was used in the current rendering pass
	CacheManagerCreateUpdateMask<<< gridSize, blockSize, 0 >>>( inputNumElem, inputList, /*output*/(uint*)thrust::raw_pointer_cast( &(*d_TempUpdateMaskList)[ 0 ] ), pTestFlag );
	GV_CHECK_CUDA_ERROR( "UpdateCreateSubdivMask" );

	CUDAPM_STOP_EVENT( gpucachemgr_createUpdateList_createMask );

	// ---- [ 2 ] ---- 2nd step
	//
	// Concatenate only previous valid elements from input data in "inputList" into the buffer of requests
	// Result is placed in "_d_UpdateCompactList"

	CUDAPM_START_EVENT( gpucachemgr_createUpdateList_elemsReduction );

	// Stream compaction
	cudppCompact( scanplan,
		/*output*/d_UpdateCompactList->getPointer( 0 ), /*nbValidElements*/d_numElementsPtr,
		/*input*/inputList, /*isValid*/(uint*)thrust::raw_pointer_cast( &(*d_TempUpdateMaskList)[ 0 ] ),
		/*nbElements*/inputNumElem );
	GV_CHECK_CUDA_ERROR( "cudppCompact" );

	// Get number of elements
	size_t numElems;
	GS_CUDA_SAFE_CALL( cudaMemcpy( &numElems, d_numElementsPtr, sizeof( size_t ), cudaMemcpyDeviceToHost ) );

	CUDAPM_STOP_EVENT( gpucachemgr_createUpdateList_elemsReduction );

	return static_cast< uint >( numElems );
}
