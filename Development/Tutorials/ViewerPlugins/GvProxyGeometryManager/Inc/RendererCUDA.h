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

#ifndef _RENDERER_CUDA_H_
#define _RENDERER_CUDA_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include <GvRendering/GsRendererCUDA.h>

// Project
#include "RendererCUDAKernel.h"
#include "ProxyGeometry.h"

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
 * @class RendererCUDA
 *
 * @brief The RendererCUDA class provides...
 *
 * ...
 */
template< typename TDataStructureType, typename TDataProductionManagerType, typename TShaderType >
class RendererCUDA : public GvRendering::GsRendererCUDA< TDataStructureType, TDataProductionManagerType, TShaderType >
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/****************************** INNER TYPES *******************************/

	/**
	 * CUDA block dimension used during rendering (kernel launch).
	 * Screen is split in 2D blocks of blockDim.x per blockDim.y pixels.
	 */
	typedef typename GvRendering::GsRendererCUDA< TDataStructureType, TDataProductionManagerType, TShaderType >::RenderBlockResolution RenderBlockResolution;

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Constructor
	 *
	 * @param pDataStructure data structure to render
	 * @param pDataProductionManager data production manager that will handle requests emitted during rendering
	 */
	RendererCUDA( TDataStructureType* pDataStructure, TDataProductionManagerType* pDataProductionManager );

	/**
	 * Destructor
	 */
	virtual ~RendererCUDA();

	/**
	 * This function is called by the user to render a frame.
	 *
	 * @param pModelMatrix the current model matrix
	 * @param pViewMatrix the current view matrix
	 * @param pProjectionMatrix the current projection matrix
	 * @param pViewport the viewport configuration
	 */
	virtual void preRender( const float4x4& pModelMatrix, const float4x4& pViewMatrix, const float4x4& pProjectionMatrix, const int4& pViewport );

	/**
	 * This function is the specific implementation method called
	 * by the parent GsIRenderer::render() method during rendering.
	 *
	 * @param pModelMatrix the current model matrix
	 * @param pViewMatrix the current view matrix
	 * @param pProjectionMatrix the current projection matrix
	 * @param pViewport the viewport configuration
	 */
	virtual void render( const float4x4& pModelMatrix, const float4x4& pViewMatrix, const float4x4& pProjectionMatrix, const int4& pViewport );

	/**
	 * This function is called by the user to render a frame.
	 *
	 * @param pModelMatrix the current model matrix
	 * @param pViewMatrix the current view matrix
	 * @param pProjectionMatrix the current projection matrix
	 * @param pViewport the viewport configuration
	 */
	virtual void postRender( const float4x4& pModelMatrix, const float4x4& pViewMatrix, const float4x4& pProjectionMatrix, const int4& pViewport );
	
	/**
	 * Get the associated proxy geometry
	 *
	 * @return the proxy geometry
	 */
	const ProxyGeometry* getProxyGeometry() const;

	/**
	 * Get the associated proxy geometry
	 *
	 * @return the proxy geometry
	 */
	ProxyGeometry* editProxyGeometry();

	/**
	 * Set the associated proxy geometry
	 *
	 * @param pProxyGeometry the proxy geometry
	 */
	void setProxyGeometry( ProxyGeometry* pProxyGeometry );

	/**
	 * Register the graphics resources associated to proxy geometry
	 *
	 * @return a flag to tell wheter or not it succeeds
	 */
	bool registerProxyGeometryGraphicsResources();
	
	/**
	 * Unregister the graphics resources associated to proxy geometry
	 *
	 * @return a flag to tell wheter or not it succeeds
	 */
	bool unregisterProxyGeometryGraphicsResources();

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
	 * Graphics resource associated to the proxy geomtry minimum depth GL buffer
	 */
	struct cudaGraphicsResource *_rayMinResource;

	/**
	 * Graphics resource associated to the proxy geomtry minimum depth GL buffer
	 */
	struct cudaGraphicsResource *_rayMaxResource;

	/**
	 * Proxy geometry
	 */
	ProxyGeometry* _proxyGeometry;

	/******************************** METHODS *********************************/

	/**
	 * Start the rendering process.
	 *
	 * @param pModelMatrix the current model matrix
	 * @param pViewMatrix the current view matrix
	 * @param pProjectionMatrix the current projection matrix
	 */
	virtual void doRender( const float4x4& pModelMatrix, const float4x4& pViewMatrix, const float4x4& pProjectionMatrix, const int4& pViewport );

	/**
	 * Copy constructor forbidden.
	 */
	RendererCUDA( const RendererCUDA& );

	/**
	 * Copy operator forbidden.
	 */
	RendererCUDA& operator=( const RendererCUDA& );

};

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "RendererCUDA.inl"

#endif // _RENDERER_CUDA_H_
