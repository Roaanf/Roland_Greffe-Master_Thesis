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

#ifndef BVHTrianglesGPUCache_H
#define BVHTrianglesGPUCache_H

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// Cuda
#include <vector_types.h>

// Cuda SDK
#include <helper_math.h>

// Thrust
#include <thrust/device_vector.h>
#include <thrust/copy.h>

// GigaVoxels
#include <GvStructure/GsIDataProductionManager.h>
#include <GvCore/GsIProvider.h>
#include <GvCore/GsRendererTypes.h>
#include <GvRendering/GsRendererHelpersKernel.h>
#include <GvCore/GsFunctionalExt.h>

// Project
#include "BvhTreeCache.hcu"
#include "BvhTreeCacheManager.h"

// STL
#include <vector>

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

////////////////////////////////////////////////////////////////////////////////
//! Volume Tree Cache manager
////////////////////////////////////////////////////////////////////////////////

/** 
 * @struct BvhTreeCache
 *
 * @brief The BvhTreeCache struct provides ...
 *
 * @param BvhTreeType ...
 * @param ProducerType ...
 */
template< typename BvhTreeType >
class BvhTreeCache : public GvStructure::GsIDataProductionManager
{
	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/****************************** INNER TYPES *******************************/

	/**
	 * Type definition of the node tile resolution
	 */
	typedef GvCore::GsVec3D< 2, 1, 1 > NodeCacheResolution;

	/**
	 * Type definition of the full brick resolution (i.e. with border)
	 */
	typedef GvCore::GsVec3D< BVH_DATA_PAGE_SIZE, 1, 1 > DataCacheResolution;

	/**
	 * Type definition for nodes cache manager
	 */
	typedef GPUCacheManager< 0, NodeCacheResolution > NodesCacheManager;

	/**
	 * Type definition for bricks cache manager
	 */
	typedef GPUCacheManager< 1, DataCacheResolution > BricksCacheManager;

	/**
	 * Type definition of producers
	 */
	typedef GvCore::GsIProvider ProducerType;

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Node cache manager
	 */
	NodesCacheManager* nodesCacheManager;
	
	/**
	 * Brick cache manager
	 */
	BricksCacheManager* bricksCacheManager;

	/******************************** METHODS *********************************/

	/**
	 * Constructor
	 *
	 * @param bvhTree BVH tree
	 * @param gpuprod producer
	 * @param voltreepoolres ...
	 * @param nodetileres nodetile resolution
	 * @param brickpoolres brick pool resolution
	 * @param brickRes brick resolution
	 */
	BvhTreeCache( BvhTreeType* bvhTree, uint3 voltreepoolres, uint3 nodetileres, uint3 brickpoolres, uint3 brickRes );

	/**
	 * Pre-render pass
	 */
	void preRenderPass();
	
	/**
	 * Post-render pass
	 */
	uint handleRequests();

	/**
	 * Clear cache
	 */
	void clearCache();

#if USE_SYNTHETIC_INFO
	/**
	 * ...
	 */
	void clearSyntheticInfo()
	{
		bricksCacheManager->d_SyntheticCacheStateBufferArray->fill(0);
	}
#endif

	/**
	 * Add a producer
	 *
	 * @param pProducer the producer to add
	 */
	void addProducer( ProducerType* pProducer );

	/**
	 * Remove a producer
	 *
	 * @param pProducer the producer to remove
	 */
	void removeProducer( ProducerType* pProducer );

	/**
	 * This method is called to serialize an object
	 *
	 * @param pStream the stream where to write
	 */
	virtual void write( std::ostream& pStream ) const;

	/**
	 * This method is called deserialize an object
	 *
	 * @param pStream the stream from which to read
	 */
	virtual void read( std::istream& pStream );

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/**
	 * BVH tree (data structure)
	 */
	BvhTreeType* _bvhTree;

	/**
	 * Node pool resolution
	 */
	uint3 _nodePoolRes;
	
	/**
	 * Brick pool resolution
	 */
	uint3 _brickPoolRes;

	/**
	 * Global buffer of requests for each node
	 */
	GvCore::GsLinearMemory< uint >* d_UpdateBufferArray;	// unified path
	
	/**
	 * Buffer of requests containing only valid requests (hence the name compacted)
	 */
	thrust::device_vector< uint >* d_UpdateBufferCompactList;

	/**
	 * ...
	 */
	uint numNodeTilesNotInUse;
	
	/**
	 * ...
	 */
	uint numBricksNotInUse;

	/**
	 * ...
	 */
	uint totalNumBricksLoaded;
	
	// CUDPP
	/**
	 * ...
	 */
	size_t* d_numElementsPtr;
	
	// CUDPP
	/**
	 * ...
	 */
	CUDPPHandle scanplan;

	/**
	 * List of producers
	 */
	std::vector< ProducerType* > _producers;

	/******************************** METHODS *********************************/

	/**
	 * Update all needed symbols in constant memory
	 */
	void updateSymbols();

	/**
	 * Update time stamps
	 */
	void updateTimeStamps();

	/**
	 * Manage updates
	 *
	 * @return ...
	 */
	uint manageUpdates();

	/**
	 * Manage the node subdivision requests
	 *
	 * @param pNumUpdateElems number of elements to process
	 *
	 * @return ...
	 */
	uint manageSubDivisions( uint pNumUpdateElems );
	
	/**
	 * Manage the brick load/produce requests
	 *
	 * @param pNumUpdateElems number of elements to process
	 *
	 * @return ...
	 */
	uint manageDataLoadGPUProd( uint pNumUpdateElems );

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/
	
	/******************************** METHODS *********************************/

	/**
	 * Copy constructor forbidden.
	 */
	BvhTreeCache( const BvhTreeCache& );

	/**
	 * Copy operator forbidden.
	 */
	BvhTreeCache& operator=( const BvhTreeCache& );

};

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "BvhTreeCache.inl"

////////////////////////////////////////////////////////////////////////////////
//! Volume Tree Cache manager
////////////////////////////////////////////////////////////////////////////////

/******************************************************************************
 * Clear cache
 ******************************************************************************/
template< typename BvhTreeType >
void BvhTreeCache< BvhTreeType >
::clearCache()
{
	//volTree->clear();
	//volTree->initCache(gpuProducer->getBVHTrianglesManager());

	//volTreeCacheManager->clearCache();
	//bricksCacheManager->clearCache();
}

#endif // !BVHTrianglesGPUCache_H
