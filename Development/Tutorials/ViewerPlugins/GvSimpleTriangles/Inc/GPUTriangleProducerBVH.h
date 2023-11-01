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

#ifndef _GPU_TRIANGLE_PRODUCER_BVH_H_
#define _GPU_TRIANGLE_PRODUCER_BVH_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include <GvCore/Array3D.h>
//#include <GvCore/Array3DGPULinear.h>
#include <GvCore/GPUPool.h>
//#include <GvCore/IProviderKernel.h>
//#include <gigavoxels/cache/GPUCacheHelper.h>

#include <GvCore/GvProvider.h>

// Project
#include "IBvhTreeProviderKernel.h"
#include "BvhTreeCacheHelper.h"
#include "BVHTrianglesManager.h"
#include "GPUTriangleProducerBVHKernel.h"

// STL
#include <string>

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ******************************** CLASS USED **********************************
 ******************************************************************************/

using namespace gigavoxels; // FIXME

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

/** 
 * @class GPUTriangleProducerBVH
 *
 * @brief The GPUTriangleProducerBVH class provides the mecanisms to produce data
 * following data requests emitted during N-tree traversal (rendering).
 *
 * It is the main user entry point to produce data from CPU, for instance,
 * loading data from disk or procedurally generating data.
 *
 * This class is implements the mandatory functions of the GvIProvider base class.
 */
template< typename TDataStructureType, uint DataPageSize, typename TDataProductionManager >
class GPUTriangleProducerBVH : public GvCore::GvProvider< TDataStructureType, TDataProductionManager >
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/****************************** INNER TYPES *******************************/

	/**
	 * Type definition of the data pool
	 */
	typedef typename TDataStructureType::DataTypeList DataTypeList;

	/**
	 * Type definition of the node pool type
	 */
	typedef typename TDataStructureType::NodePoolType NodePoolType;

	/**
	 * Type definition of the data pool type
	 */
	typedef typename TDataStructureType::DataPoolType DataPoolType;

	/**
	 * Nodes buffer type
	 */
	typedef GvCore::Array3D< VolTreeBVHNode > NodesBufferType;

	/**
	 * Type definition of the data pool
	 */
	typedef GvCore::GPUPoolHost< GvCore::Array3D, DataTypeList > DataBufferType;

	/**
	 * Type definition of the associated device-side object
	 */
	typedef GPUTriangleProducerBVHKernel< TDataStructureType, DataPageSize > KernelProducerType;

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Node pool
	 */
	NodesBufferType* _nodesBuffer;

	/**
	 * Data pool
	 */
	DataBufferType* _dataBuffer;

	/**
	 * Device-side associated object
	 */
	KernelProducerType _kernelProducer;

	/**
	 * Triangles manager used to load mesh files
	 */
	BVHTrianglesManager< DataTypeList, DataPageSize >* _bvhTrianglesManager;

	/**
	 * Mesh/scene filename
	 */
	std::string _filename;

	/******************************** METHODS *********************************/

	/**
	 * Constructor
	 */
	GPUTriangleProducerBVH();

	/**
	 * Destructor
	 */
	virtual ~GPUTriangleProducerBVH();

	/**
	 * Initialize
	 *
	 * @param pDataStructure data structure
	 * @param pDataProductionManager data production manager
	 */
	virtual void initialize( TDataStructureType* pDataStructure, TDataProductionManager* pDataProductionManager );

	/**
	 * Finalize
	 */
	virtual void finalize();

	/**
	 * Get the triangles manager
	 *
	 * @return the triangles manager
	 */
	BVHTrianglesManager< DataTypeList, DataPageSize >* getBVHTrianglesManager();

	/**
	 * This method is called by the cache manager when you have to produce data for a given pool.
	 *
	 * @param pNumElems the number of elements you have to produce.
	 * @param pNodeAddressCompactList a list containing the addresses of the numElems nodes concerned.
	 * @param pElemAddressCompactList a list containing numElems addresses where you need to store the result.
	 */
	inline virtual void produceData( uint pNumElems,
										thrust::device_vector< uint >* pNodesAddressCompactList,
										thrust::device_vector< uint >* pElemAddressCompactList,
										Loki::Int2Type< 0 > );

	/**
	 * This method is called by the cache manager when you have to produce data for a given pool.
	 *
	 * @param pNumElems the number of elements you have to produce.
	 * @param pNodeAddressCompactList a list containing the addresses of the numElems nodes concerned.
	 * @param pElemAddressCompactList a list containing numElems addresses where you need to store the result.
	 */
	inline virtual void produceData( uint pNumElems,
										thrust::device_vector< uint >* pNodesAddressCompactList,
										thrust::device_vector< uint >* pElemAddressCompactList,
										Loki::Int2Type< 1 > );

	/**
	 * ...
	 */
	void renderGL();

	/**
	 * ...
	 */
	void renderFullGL();
	
	/**
	 * ...
	 */
	void renderDebugGL();

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Cache helper
	 */
	BvhTreeCacheHelper _cacheHelper;

	/******************************** METHODS *********************************/

	/**
	 * Copy constructor forbidden.
	 */
	GPUTriangleProducerBVH( const GPUTriangleProducerBVH& );

	/**
	 * Copy operator forbidden.
	 */
	GPUTriangleProducerBVH& operator=( const GPUTriangleProducerBVH& );

};

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "GPUTriangleProducerBVH.inl"

#endif
