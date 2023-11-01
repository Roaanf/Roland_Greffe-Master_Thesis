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

#ifndef _BVH_TREE_H_
#define _BVH_TREE_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// Cuda
#include <vector_types.h>

// Cuda SDK
#include <helper_math.h>

// GigaVoxels
#include <GvCore/GsPool.h>
#include <GvStructure/GsIDataStructure.h>

#include "RendererBVHTrianglesCommon.h"

// FIXME
#include <GvCore/GsRendererTypes.h>

#include "BVHTrianglesManager.h"
#include "BvhTreeKernel.h"

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
 * @struct BvhTree
 *
 * @brief The BvhTree struct provides interface to manage BVHs data structere
 *
 * @param DataTList Data type list provided by the user
 * (exemple with a normal and a color by voxel : typedef Loki::TL::MakeTypelist< uchar4, half4 >::Result DataType;)
 */
template< class DataTList >
struct BvhTree : public GvStructure::GsIDataStructure
{
	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

	/****************************** INNER TYPES *******************************/

	/**
	 * Defines the data type list
	 */
	typedef DataTList DataTypeList;

	/**
	 * Type definition of the node pool type
	 */
	typedef GvCore::GsLinearMemory< VolTreeBVHNodeUser > NodePoolType;

	/**
	 * Type definition of the data pool type
	 */
	typedef GvCore::GPUPoolHost< GvCore::GsLinearMemory, DataTList > DataPoolType;

	/**
	 * Typedef of its associated device-side object
	 */
	typedef BvhTreeKernel< DataTList > BvhTreeKernelType;

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Root node
	 *
	 * - seems to be unused anymore
	 */
	uint _rootNode;

	/**
	 * Node pool
	 */
	NodePoolType* _nodePool;
	
	/**
	 * Data pool
	 */
	DataPoolType* _dataPool;

	/**
	 * Associated device-side object
	 */
	BvhTreeKernelType _kernelObject;

	/******************************** METHODS *********************************/

	/**
	 * Constructor
	 *
	 * @param pNodePoolSize node pool size
	 * @param pVertexPoolSize data pool size
	 */
	BvhTree( uint pNodePoolSize, uint pVertexPoolSize );

	/**
	 * Destructor
	 */
	virtual ~BvhTree();

	/**
	 * Get the associated device-side object
	 */
	BvhTreeKernelType getKernelObject();

	/**
	 * CUDA initialization
	 */
	void cuda_Init();

	/**
	 * Initialize the cache
	 *
	 * @param pBvhTrianglesManager Helper class that store the node and data pools from a mesh
	 */
	void initCache( BVHTrianglesManager< DataTList, BVH_DATA_PAGE_SIZE >* pBvhTrianglesManager );

	/**
	 * Clear
	 */
	void clear();

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

	/******************************** METHODS *********************************/

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
	BvhTree( const BvhTree& );

	/**
	 * Copy operator forbidden.
	 */
	BvhTree& operator=( const BvhTree& );

};

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "BvhTree.inl"

#endif // !_BVH_TREE_H_
