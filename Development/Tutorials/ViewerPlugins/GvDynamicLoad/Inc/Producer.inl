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
#include <GvCore/GsIProviderKernel.h>
#include <GvCore/GsError.h>
#include <GvUtils/GsDataLoader.h>

// STL
#include <string>

/******************************************************************************
 ****************************** INLINE DEFINITION *****************************
 ******************************************************************************/

/******************************************************************************
 * Constructor
 *
 * @param gpuCacheSize gpu cache size
 * @param nodesCacheSize nodes cache size
 ******************************************************************************/
template< typename TDataStructureType, typename TDataProductionManager >
Producer< TDataStructureType, TDataProductionManager >
::Producer( size_t gpuCacheSize, size_t nodesCacheSize )
:	_dataLoader( NULL )
{
	// Resolution of one brick, without borders
	uint3 brickRes = BrickRes::get();
	// Resolution of one brick, including borders
	uint3 brickResWithBorder = brickRes + make_uint3( BorderSize * 2 );
	// Number of voxels per brick
	//size_t nbVoxelsPerBrick = brickResWithBorder.x * brickResWithBorder.y * brickResWithBorder.z;
	// Number of bytes used by one voxel (=sum of the size of each channel_
	size_t voxelSize = GvCore::DataTotalChannelSize< DataTList >::value;

	// size_t brickSize = voxelSize * nbVoxelsPerBrick;
	size_t brickSize = voxelSize * KernelProducerType::BrickVoxelAlignment;
	this->_nbMaxRequests = gpuCacheSize / brickSize;
	this->_bufferNbVoxels = gpuCacheSize / voxelSize;

	// Allocate caches in mappable pinned memory
	_channelsCachesPool	= new DataCachePool( make_uint3( this->_bufferNbVoxels, 1, 1 ), 2 );

	// Localization info initialization (code and depth)
	// This is the ones that producer will have to produce
	_requestListDepth = new GvCore::GsLocalizationInfo::DepthType[ _nbMaxRequests ];
	_requestListLoc = new GvCore::GsLocalizationInfo::CodeType[ _nbMaxRequests ];
	// DEVICE temporary buffers used to retrieve localization info.
	// Data will then be copied in the previous HOST buffers ( _requestListDepth and _requestListLoc) 
	d_TempLocalizationCodeList	= new thrust::device_vector< GvCore::GsLocalizationInfo::CodeType >( nodesCacheSize );
	d_TempLocalizationDepthList	= new thrust::device_vector< GvCore::GsLocalizationInfo::DepthType >( nodesCacheSize );

	// TODO fix maxRequestNumber * 8
	_h_nodesBuffer = new GvCore::Array3D< uint >( dim3( _nbMaxRequests * 8, 1, 1 ), 2 ); // Allocated mappable pinned memory // TODO : check this size limit

	// Check error
	GV_CHECK_CUDA_ERROR( "GsOracleRegionInfoDynamic:GsOracleRegionInfoDynamic : end" );
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
template< typename TDataStructureType, typename TDataProductionManager >
Producer< TDataStructureType, TDataProductionManager >
::~Producer()
{
	finalize();
}

/******************************************************************************
 * Initialize
 *
 * @param pDataStructure data structure
 * @param pDataProductionManager data production manager
 ******************************************************************************/
template< typename TDataStructureType, typename TDataProductionManager >
inline void Producer< TDataStructureType, TDataProductionManager >
::initialize( GvStructure::GsIDataStructure* pDataStructure, GvStructure::GsIDataProductionManager* pDataProductionManager )
{
	// Call parent class
	GvUtils::GsSimpleHostProducer< ProducerKernel< TDataStructureType >, TDataStructureType, TDataProductionManager >::initialize( pDataStructure, pDataProductionManager );
}

/******************************************************************************
 * Finalize
 ******************************************************************************/
template< typename TDataStructureType, typename TDataProductionManager >
inline void Producer< TDataStructureType, TDataProductionManager >
::finalize()
{
	// Clean data loader resources
	finalizeDataLoader();

	// Clean producer resources
	delete _h_nodesBuffer;
	_h_nodesBuffer = NULL;
	delete _channelsCachesPool;
	_channelsCachesPool = NULL;
	delete _requestListDepth;
	_requestListDepth = NULL;
	delete _requestListLoc;
	_requestListLoc = NULL;
	delete d_TempLocalizationCodeList;
	d_TempLocalizationCodeList = NULL;
	delete d_TempLocalizationDepthList;
	d_TempLocalizationDepthList = NULL;
}

/******************************************************************************
 * Initialize the associated data loader
 *
 * @param pFilename filename
 * @param pBrickResolution resolution of bricks of voxels (in each dimension)
 * @param pBrickBordersize brick border size
 * @param pUseHostCache flag to tell whether or not a cache mechanismn is required when reading files (nodes and bricks)
 *
 * @return the flag telling whether or not it succeeds
 ******************************************************************************/
template< typename TDataStructureType, typename TDataProductionManager >
bool Producer< TDataStructureType, TDataProductionManager >
::initializeDataLoader( const char* pFilename, const uint3& pBrickResolution, int pBrickBordersize, bool pUseHostCache )
{
	assert( _dataLoader == NULL );

	// Initialize data loader
	_dataLoader = new GvUtils::GsDataLoader< DataTList >
	(
		std::string( pFilename ),
		pBrickResolution,
		pBrickBordersize,
		pUseHostCache
	);

	// Initialize producer
	//_dataLoader->setRegionResolution( BrickRes::get() + make_uint3( 2 * BorderSize ) );	// seem to be not used anymore

	// Compute max depth based on channel 0 producer
	float3 featureSize = _dataLoader->getFeaturesSize();
	float minFeatureSize = mincc( featureSize.x, mincc( featureSize.y, featureSize.z ) );
	if ( minFeatureSize > 0.0f )
	{
		uint maxres = static_cast< uint >( ceilf( 1.0f / minFeatureSize ) );

		// Compute the octree level corresponding to a given grid resolution.
		_maxDepth = getResolutionLevel( make_uint3( maxres ) );
	}
	else
	{
		_maxDepth = 9;	// Limitation from nodes hashes
	}

	return true;
}

/******************************************************************************
 * Finalize the associated data loader
 *
 * @return the flag telling whether or not it succeeds
 ******************************************************************************/
template< typename TDataStructureType, typename TDataProductionManager >
bool Producer< TDataStructureType, TDataProductionManager >
::finalizeDataLoader()
{
	delete _dataLoader;
	_dataLoader = NULL;

	return true;
}

/******************************************************************************
 * Get the associated data loader
 *
 * @return the data loader
******************************************************************************/
template< typename TDataStructureType, typename TDataProductionManager >
const GvUtils::GsIDataLoader< typename Producer< TDataStructureType, TDataProductionManager >::DataTList >* Producer< TDataStructureType, TDataProductionManager >
::getDataLoader() const
{
	return _dataLoader;
}

/******************************************************************************
 * Get the associated data loader
 *
 * @return the data loader
******************************************************************************/
template< typename TDataStructureType, typename TDataProductionManager >
GvUtils::GsIDataLoader< typename Producer< TDataStructureType, TDataProductionManager >::DataTList >* Producer< TDataStructureType, TDataProductionManager >
::editDataLoader()
{
	return _dataLoader;
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
template< typename TDataStructureType, typename TDataProductionManager >
inline void Producer< TDataStructureType, TDataProductionManager >
::produceData( uint pNumElems,
				GvCore::GsLinearMemory< uint >* pNodesAddressCompactList,
				GvCore::GsLinearMemory< uint >* pElemAddressCompactList,
				Loki::Int2Type< 0 > )
{
	// Initialize the device-side producer (with the node pool and the brick pool)
	this->_kernelProducer.init( _maxDepth, _h_nodesBuffer->getDeviceArray(), _channelsCachesPool->getKernelPool() );
	GvCore::GsIProviderKernel< 0, KernelProducerType > kernelProvider( this->_kernelProducer );
		
	// Define kernel block size
	// 1D block (warp size)
	dim3 blockSize( 32, 1, 1 );

	// Retrieve localization info
	GvCore::GsLocalizationInfo::CodeType* locCodeList = thrust::raw_pointer_cast( &(*d_TempLocalizationCodeList)[ 0 ] );
	GvCore::GsLocalizationInfo::DepthType* locDepthList = thrust::raw_pointer_cast( &(*d_TempLocalizationDepthList)[ 0 ] );

	// Retrieve elements address lists
	uint* nodesAddressList = pNodesAddressCompactList->getPointer( 0 );
	uint* elemAddressList = pElemAddressCompactList->getPointer( 0 );

	// Iterate through elements (i.e. nodes)
	while ( pNumElems > 0 )
	{
		// Prevent too workload
		uint numRequests = mincc( pNumElems, _nbMaxRequests );

		// Create localization info lists of the node elements to produce (code and depth)
		//
		// Resulting lists are written into the two following buffers :
		// - d_TempLocalizationCodeList
		// - d_TempLocalizationDepthList
		this->_nodePageTable->createLocalizationLists( numRequests, nodesAddressList, d_TempLocalizationCodeList, d_TempLocalizationDepthList );

		// For each node of the lists, thanks to its localization info,
		// an oracle will determine the type the associated 3D region of space
		// (i.e. max depth reached, containing data, etc...)
		//
		// Node info are then written 
		preLoadManagementNodes( numRequests, locDepthList, locCodeList );

		// Call cache helper to write into cache
		//
		// This will then call the associated DEVICE-side producer
		// whose goal is to update the cache
		this->_cacheHelper.template genericWriteIntoCache< NodeTileResLinear >( numRequests, nodesAddressList, elemAddressList, this->_nodePool, kernelProvider, this->_nodePageTable, blockSize );

		// Update loop variables
		pNumElems			-= numRequests;
		nodesAddressList	+= numRequests;
		elemAddressList		+= numRequests;
	}
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
template< typename TDataStructureType, typename TDataProductionManager >
inline void Producer< TDataStructureType, TDataProductionManager >
::produceData( uint pNumElems,
				GvCore::GsLinearMemory< uint >* pNodesAddressCompactList,
				GvCore::GsLinearMemory< uint >* pElemAddressCompactList,
				Loki::Int2Type< 1 > )
{
	// Initialize the device-side producer (with the node pool and the brick pool)
	this->_kernelProducer.init( _maxDepth, _h_nodesBuffer->getDeviceArray(), _channelsCachesPool->getKernelPool() );
	GvCore::GsIProviderKernel< 1, KernelProducerType > kernelProvider( this->_kernelProducer );
	
	// Define kernel block size
	dim3 blockSize( 16, 8, 1 );

	// Retrieve localization info
	GvCore::GsLocalizationInfo::CodeType* locCodeList = thrust::raw_pointer_cast( &(*d_TempLocalizationCodeList)[ 0 ] );
	GvCore::GsLocalizationInfo::DepthType* locDepthList = thrust::raw_pointer_cast( &(*d_TempLocalizationDepthList)[ 0 ] );

	// Retrieve elements address lists
	uint* nodesAddressList = pNodesAddressCompactList->getPointer( 0 );
	uint* elemAddressList = pElemAddressCompactList->getPointer( 0 );

	// Iterate through elements
	while ( pNumElems > 0 )
	{
		uint numRequests = mincc( pNumElems, _nbMaxRequests );

		// Create localization lists (code and depth)
		this->_dataPageTable->createLocalizationLists( numRequests, nodesAddressList, d_TempLocalizationCodeList, d_TempLocalizationDepthList );

		// For each brick of the lists, thanks to its localization info,
		// retrieve the associated brick located in this region of space,
		// and load its data from HOST disk (or retrieve data from HOST cache).
		//
		// Voxels data are then written on the DEVICE
		preLoadManagementData( numRequests, locDepthList, locCodeList );

		// Call cache helper to write into cache
		this->_cacheHelper.template genericWriteIntoCache< BrickFullRes >( numRequests, nodesAddressList, elemAddressList, this->_dataPool, kernelProvider, this->_dataPageTable, blockSize );

		// Update loop variables
		pNumElems			-= numRequests;
		nodesAddressList	+= numRequests;
		elemAddressList		+= numRequests;
	}
}

/******************************************************************************
 * Prepare nodes info for GPU download.
 * Takes a device pointer to the request lists containing depth and localization of the nodes.
 *
 * @param numElements number of elements to process
 * @param d_requestListDepth list of localization depths on the DEVICE
 * @param d_requestListLoc list of localization codes on the DEVICE
 ******************************************************************************/
template< typename TDataStructureType, typename TDataProductionManager >
inline void Producer< TDataStructureType, TDataProductionManager >
::preLoadManagementNodes( uint numElements, GvCore::GsLocalizationInfo::DepthType* d_requestListDepth, GvCore::GsLocalizationInfo::CodeType* d_requestListLoc )
{
	assert( numElements <= _nbMaxRequests );

	// TODO: use cudaMemcpyAsync
	//
	// Fetch on HOST the updated localization info list (code and depth) of all requested elements
	CUDAPM_START_EVENT( gpuProdDynamic_preLoadMgtNodes_fetchRequestList )
	cudaMemcpy( _requestListDepth, d_requestListDepth, numElements * sizeof( GvCore::GsLocalizationInfo::DepthType ), cudaMemcpyDeviceToHost );
	cudaMemcpy( _requestListLoc, d_requestListLoc, numElements * sizeof( GvCore::GsLocalizationInfo::CodeType ), cudaMemcpyDeviceToHost );
	GV_CHECK_CUDA_ERROR( "preLoadManagementNodes : cudaMemcpy" );
	CUDAPM_STOP_EVENT( gpuProdDynamic_preLoadMgtNodes_fetchRequestList )

	CUDAPM_START_EVENT( gpuProdDynamic_preLoadMgtNodes_dataLoad )

	// Nodes constantness is based on the producer of the channel 0
	typedef GvUtils::GsIDataLoader< DataTList > LoaderType;
	LoaderType* loader = _dataLoader;
	if ( loader != NULL )
	{
		uint3 brickResWithBorder = BrickRes::get() + make_uint3( 2 * BorderSize );

		// Iterate through elements (i.e. node tiles)
		for ( uint i = 0; i < numElements; ++i )
		{
			// Get localization info of current element
			GvCore::GvLocalizationDepth::ValueType locDepthElem = this->_requestListDepth[ i ].get();// + 1;
			GvCore::GvLocalizationCode::ValueType locCodeElem = this->_requestListLoc[ i ].get();

			// Localization code of childs is at the next level
			uint locDepth = locDepthElem + 1;//+2;

			// Compute sub nodes info
			// Iterate through each node of the current node tile
			uint3 subNodeOffset;
			uint subNodeOffsetIndex = 0;
			for ( subNodeOffset.z = 0; subNodeOffset.z < NodeRes::z; ++subNodeOffset.z )
			{
				for ( subNodeOffset.y = 0; subNodeOffset.y < NodeRes::y; ++subNodeOffset.y )
				{
					for ( subNodeOffset.x = 0; subNodeOffset.x < NodeRes::x; ++subNodeOffset.x )
					{
						uint3 locCode = locCodeElem * NodeRes::get() + subNodeOffset;

						// Convert localization info to a region of space
						float3 regionPos;
						float3 regionSize;
						this->getRegionFromLocalization( locDepth, locCode * BrickRes::get(), regionPos, regionSize );
					
						// Retrieve the node located in this region of space,
						// and get its information (i.e. address containing its data type region).
						uint encodedNodeInfo = loader->getRegionInfoNew( regionPos, regionSize );

						// Constant values are terminal
						if ( ( encodedNodeInfo & GV_VTBA_BRICK_FLAG ) == 0 )
						{
							encodedNodeInfo |= GV_VTBA_TERMINAL_FLAG;
						}

						// If we reached the maximal depth, set the terminal flag
						if ( locDepth >= _maxDepth )
						{
							encodedNodeInfo |= GV_VTBA_TERMINAL_FLAG;
						}

						// Write produced data
						_h_nodesBuffer->get( i * static_cast< uint >( NodeRes::getNumElements() ) + subNodeOffsetIndex ) = encodedNodeInfo;

						// Increment sub node offset
						subNodeOffsetIndex++;
					}
				}
			}
		}
	}

	CUDAPM_STOP_EVENT( gpuProdDynamic_preLoadMgtNodes_dataLoad )
}

/******************************************************************************
 * Prepare date for GPU download.
 * Takes a device pointer to the request lists containing depth and localization of the nodes.
 *
 * @param numElements ...
 * @param d_requestListDepth ...
 * @param d_requestListLoc ...
 ******************************************************************************/
template< typename TDataStructureType, typename TDataProductionManager >
inline void Producer< TDataStructureType, TDataProductionManager >
::preLoadManagementData( uint numElements, GvCore::GsLocalizationInfo::DepthType* d_requestListDepth, GvCore::GsLocalizationInfo::CodeType* d_requestListLoc )
{
	assert( numElements <= _nbMaxRequests );

	// TODO: use cudaMemcpyAsync
	//
	// Fetch on HOST the updated localization info list (code and depth) of all requested elements
	CUDAPM_START_EVENT( gpuProdDynamic_preLoadMgtData_fetchRequestList )
	cudaMemcpy( _requestListDepth, d_requestListDepth, numElements * sizeof( GvCore::GsLocalizationInfo::DepthType ), cudaMemcpyDeviceToHost );
	cudaMemcpy( _requestListLoc, d_requestListLoc, numElements * sizeof( GvCore::GsLocalizationInfo::CodeType ), cudaMemcpyDeviceToHost );
	GV_CHECK_CUDA_ERROR( "preLoadManagementData : cudaMemcpy" );
	CUDAPM_STOP_EVENT( gpuProdDynamic_preLoadMgtData_fetchRequestList )

	CUDAPM_START_EVENT( gpuProdDynamic_preLoadMgtData_dataLoad )
	
	typedef GvUtils::GsIDataLoader< DataTList > LoaderType;
	LoaderType* loader = _dataLoader;
	if ( loader != NULL )
	{
		// Compute real brick resolution (i.e. with borders)
		uint3 brickResWithBorder = BrickRes::get() + make_uint3( 2 * BorderSize );

		CUDAPM_START_EVENT( gpuProdDynamic_preLoadMgtData_dataLoad_elemLoop )

		// Iterate through elements (i.e. brick of voxels)
		for ( uint i = 0; i < numElements; ++i )
		{
			// XXX: Fixed depth offset
			uint locDepth = _requestListDepth[ i ].get();// + 1;
			uint3 locCode = _requestListLoc[ i ].get();

			// Convert localization info to a region of space
			float3 regionPos;
			float3 regionSize;
			getRegionFromLocalization( locDepth, locCode * BrickRes::get(), regionPos, regionSize );
			
			// Uses statically computed alignment
			uint brickOffset = KernelProducerType::BrickVoxelAlignment;
			
			// Retrieve the node and associated brick located in this region of space,
			// and depending of its type, if it contains data, load it.
			loader->getRegion( regionPos, regionSize, _channelsCachesPool, brickOffset * i );
		}

		CUDAPM_STOP_EVENT( gpuProdDynamic_preLoadMgtData_dataLoad_elemLoop )
	}

	CUDAPM_STOP_EVENT( gpuProdDynamic_preLoadMgtData_dataLoad )
}

/******************************************************************************
 * Compute the resolution of a given octree level.
 *
 * @param level the given level
 *
 * @return the resolution at the given level
 ******************************************************************************/
template< typename TDataStructureType, typename TDataProductionManager >
inline uint3 Producer< TDataStructureType, TDataProductionManager >
::getLevelResolution( uint level )
{
	return make_uint3( 1 << level ) * BrickRes::get();
}

/******************************************************************************
 * Compute the octree level corresponding to a given grid resolution.
 *
 * @param resol the given resolution
 *
 * @return the level at the given resolution
 ******************************************************************************/
template< typename TDataStructureType, typename TDataProductionManager >
inline uint Producer< TDataStructureType, TDataProductionManager >
::getResolutionLevel( uint3 resol )
{
	uint3 brickGridResol = resol / BrickRes::get();

	uint maxBrickGridResol = maxcc( brickGridResol.x, maxcc( brickGridResol.y, brickGridResol.z ) );

	return cclog2( maxBrickGridResol );
}

/******************************************************************************
 * Get the region corresponding to a given localization info (depth and code)
 *
 * @param depth the given localization depth
 * @param locCode the given localization code
 * @param regionPos the returned region position
 * @param regionSize the returned region size
 ******************************************************************************/
template< typename TDataStructureType, typename TDataProductionManager >
inline void Producer< TDataStructureType, TDataProductionManager >
::getRegionFromLocalization( uint depth, const uint3& locCode, float3& regionPos, float3& regionSize )
{
	// Compute the resolution of a given octree level.
	uint3 levelResolution = getLevelResolution( depth );
	// std::cout << "Level res: " << levelResolution << "\n";

	// Retrieve region position
	regionPos = make_float3( locCode ) / make_float3( levelResolution );

	// Retrieve region size
	regionSize = make_float3( BrickRes::get() ) / make_float3( levelResolution );
}
