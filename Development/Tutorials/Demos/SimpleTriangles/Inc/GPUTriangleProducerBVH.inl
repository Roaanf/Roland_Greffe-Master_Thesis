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

// STL
#include <string>

/******************************************************************************
 ****************************** INLINE DEFINITION *****************************
 ******************************************************************************/

/******************************************************************************
 * Constructor
 *
 * @param baseFileName ...
 ******************************************************************************/
template< typename TDataStructureType, uint DataPageSize, typename TDataProductionManager >
GPUTriangleProducerBVH< TDataStructureType, DataPageSize, TDataProductionManager >
::GPUTriangleProducerBVH()
:	GvCore::GsProvider< TDataStructureType, TDataProductionManager >()
,	 _nodesBuffer( NULL )
,	 _dataBuffer( NULL )
,	_kernelProducer()
,	_bvhTrianglesManager( NULL )
,	_cacheHelper()
,	_filename()
{
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
template< typename TDataStructureType, uint DataPageSize, typename TDataProductionManager >
GPUTriangleProducerBVH< TDataStructureType, DataPageSize, TDataProductionManager >
::~GPUTriangleProducerBVH()
{
}

/******************************************************************************
 * Initialize
 *
 * @param pDataStructure data structure
 * @param pDataProductionManager data production manager
 ******************************************************************************/
template< typename TDataStructureType, uint DataPageSize, typename TDataProductionManager >
inline void GPUTriangleProducerBVH< TDataStructureType, DataPageSize, TDataProductionManager >
::initialize( GvStructure::GsIDataStructure* pDataStructure, GvStructure::GsIDataProductionManager* pDataProductionManager )
{
	assert( ! _filename.empty() );

	// Call parent class
	GvCore::GsProvider< TDataStructureType, TDataProductionManager >::initialize( pDataStructure, pDataProductionManager );

	// Triangles manager's initialization
	_bvhTrianglesManager = new BVHTrianglesManager< DataTypeList, DataPageSize >();

#if 1
	//_bvhTrianglesManager->loadPowerPlant( baseFileName );
	_bvhTrianglesManager->loadMesh( _filename );
	//------------
	/*std::string path = GsEnvironment::getDataDir( GsEnvironment::e3DModelsDir );
	path += std::string( "/" );
	path += std::string( "PowerPlant" );
	path += std::string( "/" );
	path += std::string( "complete" );
	_bvhTrianglesManager->loadPowerPlant( path );*/
	//------------
	//_bvhTrianglesManager->saveRawMesh( baseFileName );
#else
	_bvhTrianglesManager->loadRawMesh( baseFileName );
#endif

	_bvhTrianglesManager->generateBuffers( 2 );
	_nodesBuffer = _bvhTrianglesManager->getNodesBuffer();
	_dataBuffer = _bvhTrianglesManager->getDataBuffer();

	// Producer's device_side associated object initialization
	_kernelProducer.init( (VolTreeBVHNodeStorage*)_nodesBuffer->getGPUMappedPointer(), _dataBuffer->getKernelPool() );
}

/******************************************************************************
 * Finalize
 ******************************************************************************/
template< typename TDataStructureType, uint DataPageSize, typename TDataProductionManager >
inline void GPUTriangleProducerBVH< TDataStructureType, DataPageSize, TDataProductionManager >
::finalize()
{
}

/******************************************************************************
 * Get the triangles manager
 *
 * @return the triangles manager
 ******************************************************************************/
template< typename TDataStructureType, uint DataPageSize, typename TDataProductionManager >
BVHTrianglesManager< typename GPUTriangleProducerBVH< TDataStructureType, DataPageSize, TDataProductionManager >::DataTypeList, DataPageSize >* GPUTriangleProducerBVH< TDataStructureType, DataPageSize, TDataProductionManager >
::getBVHTrianglesManager()
{
	return _bvhTrianglesManager;
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
template< typename TDataStructureType, uint DataPageSize, typename TDataProductionManager >
inline void GPUTriangleProducerBVH< TDataStructureType, DataPageSize, TDataProductionManager >
::produceData( uint pNumElems,
				GvCore::GsLinearMemory< uint >* pNodesAddressCompactList,
				GvCore::GsLinearMemory< uint >* pElemAddressCompactList,
				Loki::Int2Type< 0 > )
{
	// NOT USED ATM
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
template< typename TDataStructureType, uint DataPageSize, typename TDataProductionManager >
inline void GPUTriangleProducerBVH< TDataStructureType, DataPageSize, TDataProductionManager >
::produceData( uint pNumElems,
				GvCore::GsLinearMemory< uint >* pNodesAddressCompactList,
				GvCore::GsLinearMemory< uint >* pElemAddressCompactList,
				Loki::Int2Type< 1 > )
{
	// Initialize the device-side producer
	IBvhTreeProviderKernel< 1, KernelProducerType > kernelProvider( _kernelProducer );

	// Define kernel block size
	dim3 blockSize( BVH_DATA_PAGE_SIZE, 1, 1 );

	// Retrieve updated addresses
	uint* nodesAddressList = pNodesAddressCompactList->getPointer();
	uint* elemAddressList = pElemAddressCompactList->getPointer();

	// Call cache helper to write into cache
	//
	// - this function call encapsulates a kernel launch to produce data on device
	// - i.e. the associated device-side producer will call its device function "ProducerKernel::produceData< 1 >()"
	DataPoolType* pool = this->_dataPool;
	_cacheHelper.template genericWriteIntoCache< typename TDataProductionManager::DataCacheResolution >( pNumElems, nodesAddressList, elemAddressList, this->_dataStructure, pool, kernelProvider, blockSize );
}

/******************************************************************************
 * ...
 ******************************************************************************/
template< typename TDataStructureType, uint DataPageSize, typename TDataProductionManager >
void GPUTriangleProducerBVH< TDataStructureType, DataPageSize, TDataProductionManager >
::renderGL()
{
	_bvhTrianglesManager->renderGL();
}

/******************************************************************************
 * ...
 ******************************************************************************/
template< typename TDataStructureType, uint DataPageSize, typename TDataProductionManager >
void GPUTriangleProducerBVH< TDataStructureType, DataPageSize, TDataProductionManager >
::renderFullGL()
{
	_bvhTrianglesManager->renderFullGL();
}

/******************************************************************************
 * ...
 ******************************************************************************/
template< typename TDataStructureType, uint DataPageSize, typename TDataProductionManager >
void GPUTriangleProducerBVH< TDataStructureType, DataPageSize, TDataProductionManager >
::renderDebugGL()
{
	_bvhTrianglesManager->renderDebugGL();
}
