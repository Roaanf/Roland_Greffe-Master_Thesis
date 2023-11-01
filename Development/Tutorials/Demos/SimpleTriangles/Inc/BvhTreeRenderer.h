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

#ifndef _BVHTREERENDERER_H_
#define _BVHTREERENDERER_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// Gigavoxels
#include <GvCore/GsVector.h>
#include <GvCore/GsPool.h>
#include <GvCore/GsRendererTypes.h>

#include "RendererBVHTrianglesCommon.h"

#include "BvhTree.h"
#include "BvhTreeCache.h"

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

template< typename BvhTreeType, typename BvhTreeCacheType, typename ProducerType >
class BvhTreeRenderer
{
	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/****************************** INNER TYPES *******************************/

	/**
	 * ...
	 */
	typedef GvCore::GsVec3D< NUM_RAYS_PER_BLOCK_X, NUM_RAYS_PER_BLOCK_Y, 1 > RenderBlockResolution;

	// typedef base class types, since there is no unqualified name lookups for templated classes.

	/******************************** METHODS *********************************/

	/**
	 * Constructor
	 *
	 * @param volumeTree ...
	 * @param gpuProd ...
	 * @param nodePoolRes ...
	 * @param brickPoolRes ...
	 */
	BvhTreeRenderer( BvhTreeType* bvhTree, BvhTreeCacheType* bvhTreeCache, ProducerType* gpuProd );

	/**
	 * Destructor
	 */
	~BvhTreeRenderer();

	/**
	 * ...
	 *
	 * @param modelMatrix ...
	 * @param viewMatrix ...
	 * @param projectionMatrix ...
	 * @param viewport ...
	 */
	void renderImpl(const float4x4 &modelMatrix, const float4x4 &viewMatrix, const float4x4 &projectionMatrix, const int4 &viewport);

	/**
	 * ...
	 *
	 * @param colorResource ...
	 */
	void setColorResource(struct cudaGraphicsResource *colorResource);

	/**
	 * ...
	 *
	 * @param depthResource ...
	 */
	void setDepthResource(struct cudaGraphicsResource *depthResource);

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/******************************* ATTRIBUTES *******************************/

	/**
	 * ...
	 */
	BvhTreeType			*_bvhTree;
	/**
	 * ...
	 */
	BvhTreeCacheType	*_bvhTreeCache;
	/**
	 * ...
	 */
	struct cudaGraphicsResource *_colorResource;
	/**
	 * ...
	 */
	struct cudaGraphicsResource *_depthResource;

	/******************************** METHODS *********************************/
	
	/**
	 * ...
	 */
	void cuda_Init();
	/**
	 * ...
	 */
	void cuda_Destroy();

	/**
	 * ...
	 *
	 * @param fs ...
	 */
	void initFrameObjects(const uint2 &fs);
	/**
	 * ...
	 */
	void deleteFrameObjects();

	/**
	 * ...
	 *
	 * @param modelMatrix ...
	 * @param viewMatrix ...
	 * @param projectionMatrix ...
	 */
	virtual void doRender(const float4x4 &modelMatrix, const float4x4 &viewMatrix, const float4x4 &projectionMatrix, const int4& pViewport );

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	uint2 _frameSize;

	ProducerType *gpuProducer;

	bool _dynamicUpdate;
	uint _currentTime;

	float4 _userParam;

	uint _maxVolTreeDepth;

	//uint3 volTreeRootAddressGPU;

	//////////////////////////////////////////
	GvCore::GsLinearMemory<uchar4> *d_inFrameColor;
	GvCore::GsLinearMemory<float> *d_inFrameDepth;
	GvCore::GsLinearMemory<uchar4> *d_outFrameColor;
	GvCore::GsLinearMemory<float> *d_outFrameDepth;

	GvCore::GsLinearMemory<uchar4> *d_rayOutputColor;
	GvCore::GsLinearMemory<float4> *d_rayOutputNormal;

	///////////////////////
	//Restart info
	GvCore::GsLinearMemory<float> *d_rayBufferTmin;		// 1 per ray
	GvCore::GsLinearMemory<float> *d_rayBufferT;			// 1 per ray
	GvCore::GsLinearMemory<int> *d_rayBufferMaskedAt;		// 1 per ray
	GvCore::GsLinearMemory<int> *d_rayBufferStackIndex;	// 1 per tile (i.e. rendering tile)
	GvCore::GsLinearMemory<uint> *d_rayBufferStackVals;	// BVH_TRAVERSAL_STACK_SIZE per tile

public:

	bool debugDisplayTimes;
	//Debug options
	int2 currentDebugRay;

	void clearCache();

	bool &dynamicUpdateState(){
		return _dynamicUpdate;
	}

	void setUserParam(const float4 &up){
		_userParam=up;
	}
	float4 &getUserParam(){
		return _userParam;
	}

	uint getMaxVolTreeDepth(){
		return _maxVolTreeDepth;
	}
	void setMaxVolTreeDepth(uint maxVolTreeDepth){
		_maxVolTreeDepth = maxVolTreeDepth;
	}

	void nextFrame(){
		_currentTime++;
	}
};

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#if 0
#include "BvhTreeRenderer.hcu"
#else
#include "BVHRenderer_kernel.hcu"
#endif
#include "BvhTreeRenderer.inl"

#endif // !_BVHTREERENDERER_H_
