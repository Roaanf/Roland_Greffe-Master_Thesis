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

#ifndef _PRODUCER_H_
#define _PRODUCER_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include <GvUtils/GsSimpleHostProducer.h>
#include <GvCore/GsArray.h>
#include <GvCore/GsLinearMemory.h>
#include <GvCore/GsPool.h>
#include <GvUtils/GsIDataLoader.h>

// Project
#include "ProducerKernel.h"

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ******************************** CLASS USED **********************************
 ******************************************************************************/

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

/** 
 * @class Producer
 *
 * @brief The Producer class provides the mecanisms to produce data
 * following data requests emitted during N-tree traversal (rendering).
 *
 * It is the main user entry point to produce data from CPU, for instance,
 * loading data from disk or procedurally generating data.
 *
 * This class is implements the mandatory functions of the GsIProvider base class.
 *
 * @param DataTList Data type list
 * @param NodeRes Node tile resolution
 * @param BrickRes Brick resolution
 * @param BorderSize Border size of bricks
 */
template< typename TDataStructureType, typename TDataProductionManager >
class Producer : public GvUtils::GsSimpleHostProducer< ProducerKernel< TDataStructureType >, TDataStructureType, TDataProductionManager >
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/****************************** INNER TYPES *******************************/

	/**
	 * Type definition of the inherited parent class
	 */
	typedef GvUtils::GsSimpleHostProducer< ProducerKernel< TDataStructureType >, TDataStructureType, TDataProductionManager > ParentClassType;

	/**
	 * Type definition of the node tile resolution
	 */
	typedef typename TDataStructureType::NodeTileResolution NodeRes;

	/**
	 * Type definition of the brick resolution
	 */
	typedef typename TDataStructureType::BrickResolution BrickRes;

	/**
	 * Enumeration to define the brick border size
	 */
	enum
	{
		BorderSize = TDataStructureType::BrickBorderSize
	};

	/**
	 * Linear representation of a node tile
	 */
	//typedef typename TDataProductionManager::NodeTileResLinear NodeTileResLinear;
	typedef typename ParentClassType::NodeTileResLinear NodeTileResLinear;

	/**
	 * Type definition of the full brick resolution (i.e. with border)
	 */
	typedef typename TDataProductionManager::BrickFullRes BrickFullRes;
	//typedef typename ParentClassType::BrickFullRes BrickFullRes;

	/**
	 * Defines the data type list
	 */
	typedef typename TDataStructureType::DataTypeList DataTList;

	/**
	 * Type definition of a data cache pool
	 */
	typedef GvCore::GPUPoolHost< GvCore::Array3D, DataTList > DataCachePool;

	/**
	 * Typedef the kernel part of the producer
	 */
	typedef ProducerKernel< TDataStructureType > KernelProducerType;
	//typedef typename ParentClassType::KernelProducer KernelProducerType;

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Constructor
	 *
	 * @param gpuCacheSize gpu cache size
	 * @param nodesCacheSize nodes cache size
	 */
	Producer( size_t gpuCacheSize, size_t nodesCacheSize );

	/**
	 * Destructor
	 */
	virtual ~Producer();

	/**
	 * Initialize
	 *
	 * @param pDataStructure data structure
	 * @param pDataProductionManager data production manager
	 */
	virtual void initialize( GvStructure::GsIDataStructure* pDataStructure, GvStructure::GsIDataProductionManager* pDataProductionManager );

	/**
	 * Finalize
	 */
	virtual void finalize();
	
	/**
	 * This method is called by the cache manager when you have to produce data for a given pool.
	 * Implement the produceData method for the channel 0 (nodes)
	 *
	 * @param pNumElems the number of elements you have to produce.
	 * @param pNodeAddressCompactList a list containing the addresses of the numElems nodes concerned.
	 * @param pElemAddressCompactList a list containing numElems addresses where you need to store the result.
	 * @param pGpuPool the pool for which we need to produce elements.
	 * @param pPageTable the page table associated to the pool
	 * @param Loki::Int2Type< 0 > corresponds to the index of the node pool
	 */
	inline virtual void produceData( uint pNumElems,
									GvCore::GsLinearMemory< uint >* pNodesAddressCompactList,
									GvCore::GsLinearMemory< uint >* pElemAddressCompactList,
									Loki::Int2Type< 0 > );
	
	/**
	 * This method is called by the cache manager when you have to produce data for a given pool.
	 * Implement the produceData method for the channel 1 (bricks)
	 *
	 * @param pNumElems the number of elements you have to produce.
	 * @param pNodeAddressCompactList a list containing the addresses of the numElems nodes concerned.
	 * @param pElemAddressCompactList a list containing numElems addresses where you need to store the result.
	 * @param pGpuPool the pool for which we need to produce elements.
	 * @param pPageTable the page table associated to the pool
	 * @param Loki::Int2Type< 1 > corresponds to the index of the brick pool
	 */
	inline virtual void produceData( uint pNumElems,
									GvCore::GsLinearMemory< uint >* pNodesAddressCompactList,
									GvCore::GsLinearMemory< uint >* pElemAddressCompactList,
									Loki::Int2Type< 1 > );

	/**
	 * Attach a producer to a data channel.
	 *
	 * @param srcProducer producer
	 */
	void attachProducer( GvUtils::GsIDataLoader< DataTList >* srcProducer );

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Number of voxels in the transfer buffer
	 */
	size_t _bufferNbVoxels;

	/**
	 * Maximum NUMBER of requests allowed
	 */
	size_t _nbMaxRequests;

	/**
	 * Max depth
	 */
	uint _maxDepth;

	/**
	 * Localization depth's list of nodes that producer has to produce
	 *
	 * Requests buffer. (note : suppose integer types)
	 */
	GvCore::GsLocalizationInfo::DepthType* _requestListDepth; 

	/**
	 * Localization code's list of nodes that producer has to produce
	 *
	 * Requests buffer. (note : suppose integer types)
	 */
	GvCore::GsLocalizationInfo::CodeType* _requestListLoc;

	/**
	 * Indices cache.
	 * Will be accessed through zero-copy.
	 *
	 * HOST producer store a buffer with nodes address that is used on its associated DEVICE-side object
	 * It corresponds to the childAddress of an GvStructure::GsNode.
	 */
	GvCore::Array3D< uint >* _h_nodesBuffer;

	/**
	 * Channels caches pool
	 *
	 * This is where all data reside for each channel (color, normal, etc...)
	 * HOST producer store a brick pool with voxel data that is used on its associated DEVICE-side object
	 */
	DataCachePool* _channelsCachesPool;

	/**
	 * Channels producers pool
	 */
	GvUtils::GsIDataLoader< DataTList >* _dataLoader;

	/******************************** METHODS *********************************/

	/**
	 * Prepare nodes info for GPU download.
	 * Takes a device pointer to the request lists containing depth and localization of the nodes.
	 *
	 * @param numElements number of elements
	 * @param d_requestListDepth ...
	 * @param d_requestListLoc ...
	 */
	inline void preLoadManagementNodes( uint numElements, GvCore::GsLocalizationInfo::DepthType* d_requestListDepth, GvCore::GsLocalizationInfo::CodeType* d_requestListLoc );

	/**
	 * Prepare date for GPU download.
	 * Takes a device pointer to the request lists containing depth and localization of the nodes.
	 *
	 * @param numElements ...
	 * @param d_requestListDepth ...
	 * @param d_requestListLoc ...
	 */
	inline void preLoadManagementData( uint numElements, GvCore::GsLocalizationInfo::DepthType* d_requestListDepth, GvCore::GsLocalizationInfo::CodeType* d_requestListLoc );

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Helper buffer used to retrieve the list of localization code that the producer has to produce.
	 *
	 * Data in this temporary buffer is then copied in the HOST producer's _requestListLoc member.
	 */
	thrust::device_vector< GvCore::GsLocalizationInfo::CodeType >* d_TempLocalizationCodeList;

	/**
	 * Helper buffer used to retrieve the list of localization depth that the producer has to produce.
	 *
	 * Data in this temporary buffer is then copied in the HOST producer's _requestListDepth member.
	 */
	thrust::device_vector< GvCore::GsLocalizationInfo::DepthType >* d_TempLocalizationDepthList;

	/******************************** METHODS *********************************/

	/**
	 * Compute the resolution of a given octree level.
	 *
	 * @param level the given level
	 *
	 * @return the resolution at the given level
	 */
	inline uint3 getLevelResolution( uint level );

	/**
	 * Compute the octree level corresponding to a given grid resolution.
	 *
	 * @param resol the given resolution
	 *
	 * @return the level at the given resolution
	 */
	inline uint getResolutionLevel( uint3 resol );

	/**
	 * Get the region corresponding to a given localization info (depth and code)
	 *
	 * @param depth the given localization depth
	 * @param locCode the given localization code
	 * @param regionPos the returned region position
	 * @param regionSize the returned region size
	 */
	inline void getRegionFromLocalization( uint depth, const uint3& locCode, float3& regionPos, float3& regionSize );

};

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "Producer.inl"

#endif // !_PRODUCER_H_
