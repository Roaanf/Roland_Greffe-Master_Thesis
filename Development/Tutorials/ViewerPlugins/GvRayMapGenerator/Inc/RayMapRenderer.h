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

#ifndef _RAY_MAP_RENDERER_H_
#define _RAY_MAP_RENDERER_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include <GvRendering/GvRendererCUDA.h>

// Project
#include "RayMapRendererKernel.h"

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ******************************** CLASS USED **********************************
 ******************************************************************************/

// GigaSpace
namespace GvRendering
{
	class GvGraphicsResource;
}

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

/** 
 * @class RayMapRenderer
 *
 * @brief The RayMapRenderer class provides an implementation of a renderer
 * specialized for ray map generation.
 *
 * That is the commun renderer that users may use.
 * It implements the renderImpl() method from GvRenderer::GvIRenderer base class
 * and has access to the data structure, cache and producer through the inherited
 * GvVolumeTreeRenderer base class.
 */
template< typename TVolumeTreeType, typename TVolumeTreeCacheType, typename TSampleShader >
class RayMapRenderer : public GvRendering::GvRendererCUDA< TVolumeTreeType, TVolumeTreeCacheType, TSampleShader >
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
	typedef typename GvRendering::GvRendererCUDA< TVolumeTreeType, TVolumeTreeCacheType, TSampleShader >::RenderBlockResolution RenderBlockResolution;

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Constructor
	 *
	 * @param pVolumeTree data structure to render
	 * @param pVolumeTreeCache cache
	 */
	RayMapRenderer( TVolumeTreeType* pVolumeTree, TVolumeTreeCacheType* pVolumeTreeCache );

	/**
	 * Destructor
	 */
	virtual ~RayMapRenderer();

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
	 * ...
	 */
	void setRayMapResource( GvRendering::GvGraphicsResource* pGraphicsResource );
	
	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/
	
protected:

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Start the rendering process.
	 *
	 * @param pModelMatrix the current model matrix
	 * @param pViewMatrix the current view matrix
	 * @param pProjectionMatrix the current projection matrix
	 */
	virtual void doRender( const float4x4& pModelMatrix, const float4x4& pViewMatrix, const float4x4& pProjectionMatrix, const int4& pViewport );

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/******************************* ATTRIBUTES *******************************/

	/**
	  * ...
	  */
	 GvRendering::GvGraphicsResource* _rayMapResource;

	/******************************** METHODS *********************************/

};

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "RayMapRenderer.inl"

#endif // _RAY_MAP_RENDERER_H_
