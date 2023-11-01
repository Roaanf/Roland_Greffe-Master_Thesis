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

#ifndef _VOLUMETREERENDERERCUDA_INSTANCING_H_
#define _VOLUMETREERENDERERCUDA_INSTANCING_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// Cuda
#include <cutil_gl_error.h>

// Gigavoxels
#include "GvConfig.h"

#include "GvCore/GPUPool.h"
#include "GvCore/RendererTypes.h"
#include "GvCore/StaticRes3D.h"
#include "GvCore/GvIRenderer.h"
#include "GvStructure/GvVolumeTree.h"
#include "GvStructure/GvVolumeTreeCache.h"
#include "GvRenderer/VolumeTreeRenderer.h"
#include "GvRenderer/VolumeTreeRendererCUDA_instancing.hcu"

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

namespace GvRenderer
{

///
/// GigaVoxels main class
///
/** 
 * @class VolumeTreeRendererCUDA_instancing
 *
 * @brief The VolumeTreeRendererCUDA_instancing class provides...
 *
 * ...
 */
template<
	typename VolumeTreeType, typename VolumeTreeCacheType,
	typename ProducerType, typename SampleShader >
class VolumeTreeRendererCUDA_instancing
	: public GvCore::GvIRenderer< VolumeTreeRendererCUDA_instancing<VolumeTreeType, VolumeTreeCacheType, ProducerType, SampleShader> >
	, public VolumeTreeRenderer< VolumeTreeType, VolumeTreeCacheType, ProducerType >
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/****************************** INNER TYPES *******************************/

	/**
	 * ...
	 */
	typedef GvCore::StaticRes3D<8, 8, 1>		RenderBlockResolution;

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Constructor
	 *
	 * @param volumeTree ...
	 * @param volumeTreeCache ...
	 * @param gpuProd ...
	 */
	VolumeTreeRendererCUDA_instancing(VolumeTreeType *volumeTree, VolumeTreeCacheType *volumeTreeCache, ProducerType *gpuProd);

	/**
	 * Destructor
	 */
	~VolumeTreeRendererCUDA_instancing();

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

	/******************************** METHODS *********************************/

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/******************************* ATTRIBUTES *******************************/

    /**
     * ...
     */
    struct cudaGraphicsResource *_colorResource;
    /**
     * ...
     */
    struct cudaGraphicsResource *_depthResource;

	/**
	 * ...
	 */
	uint2 _frameSize;

	//VolumeTreeType *volumeTree;

	/**
	 * ...
	 */
	bool fastBuildMode;

	//Debug options
	/**
	 * ...
	 */
	int2 currentDebugRay;

	/**
	 * ...
	 */
	cudaStream_t cudaStream[1];

	//////////////////////////////////////////
	/**
	 * ...
	 */
	GvCore::Array3DGPULinear< uchar4 > *d_inFrameColor;
	/**
	 * ...
	 */
	GvCore::Array3DGPULinear< float > *d_inFrameDepth;
	/**
	 * ...
	 */
	GvCore::Array3DGPULinear< uchar4 > *d_outFrameColor;
	/**
	 * ...
	 */
	GvCore::Array3DGPULinear< float > *d_outFrameDepth;

	/**
	 * ...
	 */
	GvCore::Array3DGPULinear< float > *d_rayBufferT;
	/**
	 * ...
	 */
	GvCore::Array3DGPULinear< float > *d_rayBufferTmax;
	/**
	 * ...
	 */
	GvCore::Array3DGPULinear< float4 > *d_rayBufferAccCol;

	///////////////////////

	/**
	 * ...
	 */
	uint frameNumAfterUpdate;
	/**
	 * ...
	 */
	uint numUpdateFrames;	//>=1

	/******************************** METHODS *********************************/

	/**
	 * ...
	 *
	 * @param frameAfterUpdate ...
	 *
	 * @return ...
	 */
	float getVoxelSizeMultiplier(uint frameAfterUpdate){
		float quality;
		if(frameNumAfterUpdate<numUpdateFrames)
			quality=this->_updateQuality+ (this->_generalQuality-this->_updateQuality)* ((float)frameAfterUpdate/(float)(numUpdateFrames)) ;
		else
			quality=this->_generalQuality;

		return 1.0f/quality;
	}

	//////////////////////
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
	void doRender(const float4x4 &modelMatrix, const float4x4 &viewMatrix, const float4x4 &projectionMatrix);

};

} // namespace GvRenderer

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "VolumeTreeRendererCUDA_instancing.inl"

#endif // _VOLUMETREERENDERERCUDA_INSTANCING_H_
