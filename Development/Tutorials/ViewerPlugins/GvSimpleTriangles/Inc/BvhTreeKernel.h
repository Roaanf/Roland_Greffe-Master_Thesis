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

#ifndef _GPU_Tree_BVH_hcu_
#define _GPU_Tree_BVH_hcu_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include <GvCore/Array3DGPULinear.h>
#include <GvCore/GvCUDATexHelpers.h>

#include "BVHTriangles.hcu"

#include <cuda.h>
#include "RendererBVHTrianglesCommon.h"

// Cuda SDK
#include <helper_math.h>

//#include "CUDATexHelpers.h"
//#include "Array3DKernel.h"

#include <loki/Typelist.h>
#include <loki/HierarchyGenerators.h>
#include <loki/TypeManip.h>
#include <loki/NullType.h>

#include "GPUTreeBVHCommon.h"

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/**
 * ...
 */
#define TEXDATAPOOL_BVHTRIANGLES 10

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

////////// DATA ARRAYS ///////////

/**
 * Node buffer
 *
 * Seems to be not used anymore
 */
texture< uint, 1, cudaReadModeElementType > volumeTreeBVHTexLinear;

// FIXME
//GPUPoolTextureReferences(TEXDATAPOOL_BVHTRIANGLES, 4, 1, BVHVertexPosType, cudaReadModeElementType);
//GPUPoolTextureReferences(TEXDATAPOOL_BVHTRIANGLES, 4, 1, BVHVertexColorType, cudaReadModeElementType);

/**
 * 1D texture used to store user data (color)
 */
GPUPoolTextureReferences( TEXDATAPOOL_BVHTRIANGLES, 4, 1, uchar4, cudaReadModeElementType );
/**
 * 1D texture used to store user data (position)
 */
GPUPoolTextureReferences( TEXDATAPOOL_BVHTRIANGLES, 4, 1, float4, cudaReadModeElementType );

/******************************************************************************
 ******************************** CLASS USED **********************************
 ******************************************************************************/

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

/** 
 * @struct BvhTreeKernel
 *
 * @brief The BvhTreeKernel struct provides ...
 *
 * @param TDataTypeList Data type list provided by the user
 * (exemple with a normal and a color by voxel : typedef Loki::TL::MakeTypelist< uchar4, half4 >::Result DataType;)
 *
 */
template< class TDataTypeList >
struct BvhTreeKernel
{
	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

	/****************************** INNER TYPES *******************************/

	/**
	 * Type definition of the data pool
	 */
	typedef typename GvCore::GPUPool_KernelPoolFromHostPool< GvCore::Array3DGPULinear, TDataTypeList >::Result KernelPoolType;

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Node pool
	 */
	GvCore::Array3DKernelLinear< VolTreeBVHNodeStorageUINT > _volumeTreeBVHArray;

	/**
	 * Data pool
	 */
	KernelPoolType _dataPool;

	/******************************** METHODS *********************************/

	/**
	 * ...
	 *
	 * @param ... ...
	 *
	 * @return ...
	 */
	template< int channel >
	__device__
	typename GvCore::DataChannelType< TDataTypeList, channel >::Result getVertexData( uint address );

	/**
	 * ...
	 *
	 * @param ... ...
	 * @param ... ...
	 */
	__device__
	inline void fetchBVHNode( VolTreeBVHNodeUser& resnode, uint address );

	/**
	 * ...
	 *
	 * @param ... ...
	 * @param ... ...
	 * @param ... ...
	 */
	__device__
	inline void parallelFetchBVHNode( uint Pid, VolTreeBVHNodeUser& resnode, uint address );

	/**
	 * ...
	 *
	 * @param ... ...
	 * @param ... ...
	 * @param ... ...
	 */
	__device__
	inline void parallelFetchBVHNodeTile( uint Pid, VolTreeBVHNodeUser* resnodetile, uint address );

	/**
	 * ...
	 *
	 * @param ... ...
	 * @param ... ...
	 */
	__device__
	inline void writeBVHNode( const VolTreeBVHNodeUser& node, uint address );

	/**
	 * ...
	 *
	 * @param ... ...
	 * @param ... ...
	 * @param ... ...
	 */
	__device__
	inline void parallelWriteBVHNode( uint Pid, const VolTreeBVHNodeStorage& node, uint address );

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

#include "BvhTreeKernel.inl"

#endif // !_GPU_Tree_BVH_hcu_
