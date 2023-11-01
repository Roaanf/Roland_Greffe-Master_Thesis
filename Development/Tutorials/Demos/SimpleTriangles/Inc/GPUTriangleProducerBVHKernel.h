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

#ifndef _GPU_TRIANGLE_PRODUCER_BVH_HCU_
#define _GPU_TRIANGLE_PRODUCER_BVH_HCU_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include <GvCore/GsPool.h>
//#include <GvCore/IntersectionTests.hcu>

//#include "BvhTree.hcu"

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
 * @class GPUTriangleProducerBVHKernel
 *
 * @brief The GPUTriangleProducerBVHKernel class provides ...
 *
 * ...
 *
 * @param TDataTypeList ...
 * @param TDataPageSize ...
 */
template< typename TDataStructureType, uint TDataPageSize >
class GPUTriangleProducerBVHKernel
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/****************************** INNER TYPES *******************************/

	/**
	 * Defines the data type list
	 */
	typedef typename TDataStructureType::DataTypeList DataTypeList;

	///**
	// * Nodes buffer type
	// */
	//typedef GvCore::Array3D< VolTreeBVHNode > NodesBufferType;

	/**
	 * Type definition of the data pool
	 */
	typedef GvCore::GPUPoolKernel< GvCore::GsLinearMemoryKernel, DataTypeList > DataBufferKernelType;

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Nodes buffer in host memory (mapped)
	 */
	/**
	 * Seems to be unused anymore because produceNodeTileData() seems to be unused anymore...
	 */
	VolTreeBVHNodeStorage* _nodesBufferKernel;

	/**
	 * Data pool (position and color)
	 */
	DataBufferKernelType _dataBufferKernel;

	/******************************** METHODS *********************************/

	/**
	 * Initialize
	 *
	 * @param h_nodesbufferarray node buffer
	 * @param h_vertexbufferpool data buffer
	 */
	inline void init( VolTreeBVHNodeStorage* h_nodesbufferarray, GvCore::GPUPoolKernel< GvCore::GsLinearMemoryKernel, DataTypeList > h_vertexbufferpool );

	/**
	 * Produce node tiles
	 *
	 * @param nodePool ...
	 * @param requestID ...
	 * @param processID ...
	 * @param pNewElemAddress ...
	 * @param parentLocInfo ...
	 *
	 * @return ...
	 */
	template< typename GPUPoolKernelType >
	__device__
	inline uint produceData( GPUPoolKernelType& nodePool, uint requestID, uint processID,
							uint3 pNewElemAddress, const VolTreeBVHNodeUser& node, Loki::Int2Type< 0 > );

	/**
	 * Produce bricks of data
	 *
	 * @param dataPool ...
	 * @param requestID ...
	 * @param processID ...
	 * @param pNewElemAddress ...
	 * @param parentLocInfo ...
	 *
	 * @return ...
	 */
	template< typename GPUPoolKernelType >
	__device__
	inline uint produceData( GPUPoolKernelType& dataPool, uint requestID, uint processID,
							uint3 pNewElemAddress, VolTreeBVHNodeUser& pNode, Loki::Int2Type< 1 > );

	/**
	 * Seems to be unused anymore...
	 */
	template< class GPUTreeBVHType >
	__device__
	inline uint produceNodeTileData( GPUTreeBVHType& gpuTreeBVH, uint requestID, uint processID, VolTreeBVHNodeUser& node, uint newNodeTileAddressNode );

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

	/******************************** METHODS *********************************/

};

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "GPUTriangleProducerBVHKernel.inl"

#endif
