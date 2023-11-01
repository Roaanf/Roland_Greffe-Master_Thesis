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

// System
#include <cassert>

/******************************************************************************
 ****************************** INLINE DEFINITION *****************************
 ******************************************************************************/

/******************************************************************************
 * Constructor
 ******************************************************************************/
template< typename TDataStructureType, typename TDataProductionManager >
Producer< TDataStructureType, TDataProductionManager >
::Producer()
:	GvUtils::GvSimpleHostProducer< ProducerKernel< TDataStructureType >, TDataStructureType, TDataProductionManager >()
{
	_maxNumRequest =1;
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
template< typename TDataStructureType, typename TDataProductionManager >
Producer< TDataStructureType, TDataProductionManager >
::~Producer()
{
	// Finalize the producer and its particle system
	finalize();
}

/******************************************************************************
 * Initialize producer and generate particles
 ******************************************************************************/
template< typename TDataStructureType, typename TDataProductionManager >
inline void Producer< TDataStructureType, TDataProductionManager >
::initialize( TDataStructureType* pDataStructure, TDataProductionManager* pDataProductionManager )
{
	// Call parent class
	GvUtils::GvSimpleHostProducer< ProducerKernel< TDataStructureType >, TDataStructureType, TDataProductionManager >::initialize( pDataStructure, pDataProductionManager );
	
}

/******************************************************************************
 * Finalize the producer and its particle system
 ******************************************************************************/
template< typename TDataStructureType, typename TDataProductionManager >
inline void Producer< TDataStructureType, TDataProductionManager >
::finalize()
{
	// TO DO
	// Check if there are special things to do here... ?
	// ...

}


/******************************************************************************
 * Implement the produceData method for the channel 0 (nodes).
 * This method is called by the cache manager when you have to produce data for a given pool.
 *
 * @param pNumElems the number of elements you have to produce.
 * @param nodeAddressCompactList a list containing the addresses of the numElems nodes concerned.
 * @param pElemAddressCompactList a list containing numElems addresses where you need to store the result.
 * @param gpuPool the pool for which we need to produce elements.
 * @param pageTable the page table associated to the pool
 * @param Loki::Int2Type< 0 > id of the channel
 ******************************************************************************/
template< typename TDataStructureType, typename TDataProductionManager >
inline void Producer< TDataStructureType, TDataProductionManager >
::produceData( uint pNumElems,
				thrust::device_vector< uint >* pNodesAddressCompactList,
				thrust::device_vector< uint >* pElemAddressCompactList,
				Loki::Int2Type< 0 > )
{
	/*printf( "---- GPUCacheManager< 0 > : %d requests ----\n", pNumElems );

	for ( size_t i = 0; i < pNumElems; ++i )
	{
		uint nodeAddressEnc = (*pNodesAddressCompactList)[ i ];
		uint3 nodeAddress = GvStructure::GvNode::unpackNodeAddress( nodeAddressEnc );
		printf( " request %d: 0x%x (%d, %d, %d)\n", i, nodeAddressEnc, nodeAddress.x, nodeAddress.y, nodeAddress.z );
	}*/

	// Initialize the device-side producer
	GvCore::GvIProviderKernel< 0, TKernelProducerType > kernelProvider( _kernelProducer );

	// Define kernel block size
	const uint3 kernelBlockSize = TKernelProducerType::NodesKernelBlockSize::get();
	const dim3 blockSize( kernelBlockSize.x, kernelBlockSize.y, kernelBlockSize.z );

	// Retrieve updated addresses
	uint* nodesAddressList = thrust::raw_pointer_cast( &(*pNodesAddressCompactList)[ 0 ] );
	uint* elemAddressList = thrust::raw_pointer_cast( &(*pElemAddressCompactList)[ 0 ] );

	// Call cache helper to write into cache
	//
	// - this function call encapsulates a kernel launch to produce data on device
	// - i.e. the associated device-side producer will call its device function "ProducerKernel::produceData< 0 >()"
	NodePoolType* pool = this->_nodePool;
	NodePageTableType* pageTable = _nodePageTable;
	
	_cacheHelper.template genericWriteIntoCache< NodeTileResLinear >( pNumElems, nodesAddressList, elemAddressList, pool, kernelProvider, pageTable, blockSize );

}

/******************************************************************************
 * Implement the produceData method for the channel 1 (bricks).
 * This method is called by the cache manager when you have to produce data for a given pool.
 *
 * @param pNumElems the number of elements you have to produce.
 * @param pNodesAddressCompactList a list containing the addresses of the numElems nodes concerned.
 * @param pElemAddressCompactList a list containing numElems addresses where you need to store the result.
 * @param gpuPool the pool for which we need to produce elements.
 * @param pageTable the page table associated to the pool
 * @param Loki::Int2Type< 1 > id of the channel
 ******************************************************************************/
template< typename TDataStructureType, typename TDataProductionManager >
inline void Producer< TDataStructureType, TDataProductionManager >
::produceData( uint pNumElems,
				thrust::device_vector< uint >* pNodesAddressCompactList,
				thrust::device_vector< uint >* pElemAddressCompactList,
				Loki::Int2Type< 1 > )
{
	/*printf( "---- GPUCacheManager< 1 > : %d requests ----\n", pNumElems );

	for ( size_t i = 0; i < pNumElems; ++i )
	{
		uint nodeAddressEnc = (*pNodesAddressCompactList)[ i ];
		uint3 nodeAddress = GvStructure::GvNode::unpackNodeAddress( nodeAddressEnc );
		printf( " request %d: 0x%x (%d, %d, %d)\n", i, nodeAddressEnc, nodeAddress.x, nodeAddress.y, nodeAddress.z );
	}*/

	// Initialize the device-side producer
	GvCore::GvIProviderKernel< 1, TKernelProducerType > kernelProvider( _kernelProducer );

	// Define kernel block size
	const uint3 kernelBlockSize = TKernelProducerType::BricksKernelBlockSize::get();
	const dim3 blockSize( kernelBlockSize.x, kernelBlockSize.y, kernelBlockSize.z );

	// Retrieve updated addresses
	uint* nodesAddressList = thrust::raw_pointer_cast( &(*pNodesAddressCompactList)[ 0 ] );
	uint* elemAddressList = thrust::raw_pointer_cast( &(*pElemAddressCompactList)[ 0 ] );

	// Call cache helper to write into cache
	//
	// - this function call encapsulates a kernel launch to produce data on device
	// - i.e. the associated device-side producer will call its device function "ProducerKernel::produceData< 1 >()"
	DataPoolType* pool = this->_dataPool;
	DataPageTableType* pageTable = _dataPageTable;
	float elapsedTime = 0.f;
	cudaEvent_t start, stop;
	cudaEventCreate( &start );
	cudaEventCreate( &stop );
	FILE * f;
	if (pNumElems>=_maxNumRequest)
	{

		pNumElems = _maxNumRequest;
		_maxNumRequest++;
		printf("Benchmarking production of %u bricks.........",pNumElems);
		// Start event
		cudaEventRecord( start, 0 );
		_cacheHelper.template genericWriteIntoCache< BrickFullRes >( pNumElems, nodesAddressList, elemAddressList, pool, kernelProvider, pageTable, blockSize );
		cudaEventRecord( stop, 0 );
		cudaEventSynchronize( stop );
		cudaEventElapsedTime( &elapsedTime, start, stop );
		printf("done ! \n");
		f=fopen("results.txt","a");
		fprintf(f,"%u %f\n",pNumElems,elapsedTime);
		fclose(f);
	} else {
		printf("Not enough brick to produce , need %u bricks , do whatever you want but give me %u bricks ! !\n",_maxNumRequest,_maxNumRequest);
		_cacheHelper.template genericWriteIntoCache< BrickFullRes >( pNumElems, nodesAddressList, elemAddressList, pool, kernelProvider, pageTable, blockSize );
		
	}
		
	
}