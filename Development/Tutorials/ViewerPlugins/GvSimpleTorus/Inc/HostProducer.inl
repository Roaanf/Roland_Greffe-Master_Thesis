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

// Cuda
#include <driver_types.h>
#include <cuda_runtime.h>

// STL
#include <iostream>

/******************************************************************************
 ****************************** INLINE DEFINITION *****************************
 ******************************************************************************/

/******************************************************************************
 * Constructor
 *
 * @param pDataStructure data structure
 ******************************************************************************/
template< typename TKernelProducerType, typename TDataStructureType, typename TDataProductionManager >
inline HostProducer< TKernelProducerType, TDataStructureType, TDataProductionManager >
::HostProducer()
:	GvUtils::GvSimpleHostProducer< TKernelProducerType, TDataStructureType, TDataProductionManager >()
{
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
template< typename TKernelProducerType, typename TDataStructureType, typename TDataProductionManager >
inline HostProducer< TKernelProducerType, TDataStructureType, TDataProductionManager >
::~HostProducer()
{
	finalize();
}

/******************************************************************************
 * Initialize
 *
 * @param pDataStructure data structure
 * @param pDataProductionManager data production manager
 ******************************************************************************/
template< typename TKernelProducerType, typename TDataStructureType, typename TDataProductionManager >
inline void HostProducer< TKernelProducerType, TDataStructureType, TDataProductionManager >
::initialize( TDataStructureType* pDataStructure, TDataProductionManager* pDataProductionManager )
{
	// Call parent class
	GvUtils::GvSimpleHostProducer< TKernelProducerType, TDataStructureType, TDataProductionManager >::initialize( pDataStructure, pDataProductionManager );
}

/******************************************************************************
 * Finalize
 ******************************************************************************/
template< typename TKernelProducerType, typename TDataStructureType, typename TDataProductionManager >
inline void HostProducer< TKernelProducerType, TDataStructureType, TDataProductionManager >
::finalize()
{
}

/******************************************************************************
 * This method is called by the cache manager when you have to produce data for a given pool.
 * Implement the produceData method for the channel 0 (nodes)
 *
 * @param pNumElems the number of elements you have to produce.
 * @param pNodeAddressCompactList a list containing the addresses of the numElems nodes concerned.
 * @param pElemAddressCompactList a list containing numElems addresses where you need to store the result.
 * @param pGpuPool the pool for which we need to produce elements.
 * @param pPageTable the page table associated to the pool
 * @param Loki::Int2Type< 0 > corresponds to the index of the node pool
 ******************************************************************************/
template< typename TKernelProducerType, typename TDataStructureType, typename TDataProductionManager >
inline void HostProducer< TKernelProducerType, TDataStructureType, TDataProductionManager >
::produceData( uint pNumElems,
				thrust::device_vector< uint >* pNodesAddressCompactList,
				thrust::device_vector< uint >* pElemAddressCompactList,
				Loki::Int2Type< 0 > )
{
	// NOTE : the call to taht function doesn't seem to work (compilation pb due to LOKI...)
	//GvUtils::GvSimpleHostProducer< TKernelProducerType, TDataStructureType, TDataProductionManager >::produceData( pNumElems, pNodesAddressCompactList, pElemAddressCompactList, 0 );

	cudaEvent_t start, stop;

	cudaEventCreate( &start );
	cudaEventCreate( &stop );

	// Initialize the device-side producer
	GvCore::GvIProviderKernel< 0, TKernelProducerType > kernelProvider( this->_kernelProducer );

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
	NodePageTableType* pageTable = this->_nodePageTable;

	cudaEventRecord( start, 0 );

	this->_cacheHelper.template genericWriteIntoCache< NodeTileResLinear >( pNumElems, nodesAddressList, elemAddressList, pool, kernelProvider, pageTable, blockSize );
	
	cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop );

	float elapsedTime;
	cudaEventElapsedTime( &elapsedTime, start, stop );

	std::cout << "Elapsed time : " << elapsedTime << std::endl;

	cudaEventDestroy( start );
	cudaEventDestroy( stop );
}

/******************************************************************************
 * This method is called by the cache manager when you have to produce data for a given pool.
 * Implement the produceData method for the channel 1 (bricks)
 *
 * @param pNumElems the number of elements you have to produce.
 * @param pNodeAddressCompactList a list containing the addresses of the numElems nodes concerned.
 * @param pElemAddressCompactList a list containing numElems addresses where you need to store the result.
 * @param pGpuPool the pool for which we need to produce elements.
 * @param pPageTable the page table associated to the pool
 * @param Loki::Int2Type< 1 > corresponds to the index of the brick pool
 ******************************************************************************/
template< typename TKernelProducerType, typename TDataStructureType, typename TDataProductionManager >
inline void HostProducer< TKernelProducerType, TDataStructureType, TDataProductionManager >
::produceData( uint pNumElems,
				thrust::device_vector< uint >* pNodesAddressCompactList,
				thrust::device_vector< uint >* pElemAddressCompactList,
				Loki::Int2Type< 1 > )
{
	// NOTE : the call to taht function doesn't seem to work (compilation pb due to LOKI...)
	//GvUtils::GvSimpleHostProducer< TKernelProducerType, TDataStructureType, TDataProductionManager >::produceData( pNumElems, pNodesAddressCompactList, pElemAddressCompactList, Loki::Int2Type< 1 > );

	// Initialize the device-side producer
	GvCore::GvIProviderKernel< 1, TKernelProducerType > kernelProvider( this->_kernelProducer );

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
	DataPageTableType* pageTable = this->_dataPageTable;
	this->_cacheHelper.template genericWriteIntoCache< BrickFullRes >( pNumElems, nodesAddressList, elemAddressList, pool, kernelProvider, pageTable, blockSize );
}
