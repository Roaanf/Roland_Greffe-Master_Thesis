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

#ifndef _VOLUME_TREE_RENDERER_CUDA_H_
#define _VOLUME_TREE_RENDERER_CUDA_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// Cuda
#include <driver_types.h>

// GigaVoxels
#include <GvCore/GvCoreConfig.h>
#include <GvCore/GPUPool.h>
#include <GvCore/RendererTypes.h>
#include <GvCore/StaticRes3D.h>
#include <GvCore/vector_types_ext.h>
#include <GvStructure/GvVolumeTree.h>
#include <GvStructure/GvDataProductionManager.h>
#include <GvRendering/GvRenderer.h>
#include <GvRendering/GvGraphicsInteroperabiltyHandler.h>
#include <GvRendering/GvGraphicsInteroperabiltyHandlerKernel.h>

// Project
#include "VolumeTreeRendererCUDAKernel.h"

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
 * @class GvVolumeTreeRendererCUDA
 *
 * @brief The GvVolumeTreeRendererCUDA class provides an implementation of a renderer
 * specialized for CUDA.
 *
 * That is the commun renderer that users may use.
 * It implements the renderImpl() method from GvRenderer::GvIRenderer base class
 * and has access to the data structure, cache and producer through the inherited
 * GvVolumeTreeRenderer base class.
 */
template< typename TVolumeTreeType, typename TVolumeTreeCacheType, typename TSampleShader >
class VolumeTreeRendererCUDA : public GvRendering::GvRenderer< TVolumeTreeType, TVolumeTreeCacheType >
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/****************************** INNER TYPES *******************************/

	/**
	 * CUDA block dimension used during rendering (kernel launch).
	 * Screen is split in 2D blocks of 8 per 8 pixels.
	 */
	typedef GvCore::StaticRes3D< 8, 8, 1 > RenderBlockResolution;

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Constructor
	 *
	 * @param pVolumeTree data structure to render
	 * @param pVolumeTreeCache cache
	 * @param pProducer producer of data
	 */
	VolumeTreeRendererCUDA( TVolumeTreeType* pVolumeTree, TVolumeTreeCacheType* pVolumeTreeCache );

	/**
	 * Destructor
	 */
	virtual ~VolumeTreeRendererCUDA();

	/**
	 * This function is the specific implementation method called
	 * by the parent GvIRenderer::render() method during rendering.
	 *
	 * @param pModelMatrix the current model matrix
	 * @param pViewMatrix the current view matrix
	 * @param pProjectionMatrix the current projection matrix
	 * @param pViewport the viewport configuration
	 */
	virtual void render( const float4x4& pModelMatrix, const float4x4& pViewMatrix, const float4x4& pProjectionMatrix, const int4& pViewport );
	
	/**
	 * Attach an OpenGL buffer object (i.e. a PBO, a VBO, etc...) to an internal graphics resource 
	 * that will be mapped to a color or depth slot used during rendering.
	 *
	 * @param pGraphicsResourceSlot the associated internal graphics resource (color or depth) and its type of access (read, write or read/write)
	 * @param pBuffer the OpenGL buffer
	 *
	 * @return a flag telling wheter or not it succeeds
	 */
	bool connect( GvRendering::GvGraphicsInteroperabiltyHandler::GraphicsResourceSlot pGraphicsResourceSlot, GLuint pBuffer );

	/**
	 * Attach an OpenGL texture or renderbuffer object to an internal graphics resource 
	 * that will be mapped to a color or depth slot used during rendering.
	 *
	 * @param pGraphicsResourceSlot the associated internal graphics resource (color or depth) and its type of access (read, write or read/write)
	 * @param pImage the OpenGL texture or renderbuffer object
	 * @param pTarget the target of the OpenGL texture or renderbuffer object
	 *
	 * @return a flag telling wheter or not it succeeds
	 */
	bool connect( GvRendering::GvGraphicsInteroperabiltyHandler::GraphicsResourceSlot pGraphicsResourceSlot, GLuint pImage, GLenum pTarget );

	/**
	 * Dettach an OpenGL buffer object (i.e. a PBO, a VBO, etc...), texture or renderbuffer object
	 * to its associated internal graphics resource mapped to a color or depth slot used during rendering.
	 *
	 * @param pGraphicsResourceSlot the internal graphics resource slot (color or depth)
	 *
	 * @return a flag telling wheter or not it succeeds
	 */
	bool disconnect( GvRendering::GvGraphicsInteroperabiltyHandler::GraphicsResourceSlot pGraphicsResourceSlot );

	/**
	 * Disconnect all registered graphics resources
	 *
	 * @return a flag telling wheter or not it succeeds
	 */
	bool resetGraphicsResources();

	/**
	 * TEST - optimization
	 *
	 * Launch the post-render phase
	 */
	virtual void doPostRender();
	
	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Frame size
	 */
	uint2 _frameSize;

	/**
	 * Fast build mode flag
	 */
	bool _fastBuildMode;

#ifndef USE_SIMPLE_RENDERER
	/**
	 * Cuda stream
	 */
	cudaStream_t _cudaStream[ 1 ];

	/**
	 * ...
	 */
	GvCore::Array3DGPULinear< float >* d_rayBufferT;

	/**
	 * ...
	 */
	GvCore::Array3DGPULinear< float >* d_rayBufferTmax;

	/**
	 * ...
	 */
	GvCore::Array3DGPULinear< float4 >* d_rayBufferAccCol;

	/**
	 * Use a debug options
	 */
	int2 _currentDebugRay;
#endif

	/**
	 * ...
	 */
	uint _frameNumAfterUpdate;

	/**
	 * ...
	 */
	uint _numUpdateFrames;	// >= 1

	/**
	 * Graphics interoperability handler
	 */
	GvRendering::GvGraphicsInteroperabiltyHandler* _graphicsInteroperabiltyHandler;

	/******************************** METHODS *********************************/

	/**
	 * Get the graphics interoperability handler
	 *
	 * @return the graphics interoperability handler
	 */
	GvRendering::GvGraphicsInteroperabiltyHandler* getGraphicsInteroperabiltyHandler();

	/**
	 * Initialize Cuda objects
	 */
	void initializeCuda();

	/**
	 * Finalize Cuda objects
	 */
	void finalizeCuda();

	/**
	 * Initialize internal buffers used during rendering
	 * (i.e. input/ouput color and depth buffers, ray buffers, etc...).
	 * Buffers size are dependent of the frame size.
	 *
	 * @param pFrameSize the frame size
	 */
	void initFrameObjects( const uint2& pFrameSize );

	/**
	 * Destroy internal buffers used during rendering
	 * (i.e. input/ouput color and depth buffers, ray buffers, etc...)
	 * Buffers size are dependent of the frame size.
	 */
	void deleteFrameObjects();

	/**
	 * Start the rendering process.
	 *
	 * @param pModelMatrix the current model matrix
	 * @param pViewMatrix the current view matrix
	 * @param pProjectionMatrix the current projection matrix
	 */
	void doRender( const float4x4& pModelMatrix, const float4x4& pViewMatrix, const float4x4& pProjectionMatrix );

#ifndef USE_SIMPLE_RENDERER
	/**
	 * ...
	 *
	 * @param pFrameAfterUpdate ...
	 *
	 * @return ...
	 */
	float getVoxelSizeMultiplier( uint pFrameAfterUpdate );
#endif

	/**
	 * Bind all graphics resources used by the GL interop handler during rendering.
	 *
	 * Internally, it binds textures and surfaces to arrays associated to mapped graphics reources.
	 *
	 * NOTE : this method should be in the GvGraphicsInteroperabiltyHandler but it seems that
	 * there are conflicts with textures ans surfaces symbols. The binding succeeds but not the
	 * read/write operations.
	 *
	 * @return a flag telling wheter or not it succeeds
	 */
	bool bindGraphicsResources();

	/**
	 * Unbind all graphics resources used by the GL interop handler during rendering.
	 *
	 * Internally, it unbinds textures and surfaces to arrays associated to mapped graphics reources.
	 *
	 * NOTE : this method should be in the GvGraphicsInteroperabiltyHandler but it seems that
	 * there are conflicts with textures ans surfaces symbols. The binding succeeds but not the
	 * read/write operations.
	 *
	 * @return a flag telling wheter or not it succeeds
	 */
	bool unbindGraphicsResources();

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
	VolumeTreeRendererCUDA( const VolumeTreeRendererCUDA& );

	/**
	 * Copy operator forbidden.
	 */
	VolumeTreeRendererCUDA& operator=( const VolumeTreeRendererCUDA& );

};

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "VolumeTreeRendererCUDA.inl"

#endif // _VOLUME_TREE_RENDERER_CUDA_H_
