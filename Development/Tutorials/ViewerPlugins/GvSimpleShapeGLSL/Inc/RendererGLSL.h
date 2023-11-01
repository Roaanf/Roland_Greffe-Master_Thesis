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

#ifndef _RENDERER_GLSL_H_
#define _RENDERER_GLSL_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include <GvRendering/GsRenderer.h>
#include <GvRendering/GsRendererContext.h>

// Cuda
#include <driver_types.h>

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ******************************** CLASS USED **********************************
 ******************************************************************************/

// GigaVoxels
namespace GsGraphics
{
	class GsShaderProgram;
}

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

/** 
 * @class RendererGLSL
 *
 * @brief The RendererGLSL class provides an implementation of a renderer
 * specialized for GLSL.
 *
 * It implements the renderImpl() method from GsRenderer::GsIRenderer base class
 * and has access to the data structure, cache and producer through the inherited
 * VolumeTreeRenderer base class.
 */
template< typename TDataStructureType, typename TVolumeTreeCacheType >
class RendererGLSL : public GvRendering::GsRenderer< TDataStructureType, TVolumeTreeCacheType >
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Constructor
	 *
	 * It initializes all OpenGL-related stuff
	 *
	 * @param pVolumeTree data structure to render
	 * @param pVolumeTreeCache cache
	 * @param pProducer producer of data
	 */
	RendererGLSL( TDataStructureType* pVolumeTree, TVolumeTreeCacheType* pVolumeTreeCache );

	/**
	 * Destructor
	 */
	virtual ~RendererGLSL();

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
	 * Get the associated shader program
	 *
	 * @return the associated shader program
	 */
	GsGraphics::GsShaderProgram* getShaderProgram()
	{
		return _shaderProgram;
	}

	/**
	 * Get the cone aperture scale
	 *
	 * @return the cone aperture scale
	 */
	float getConeApertureScale() const;

	/**
	 * Set the cone aperture scale
	 *
	 * @param pValue the cone aperture scale
	 */
	void setConeApertureScale( float pValue );

	/**
	 * Get the max number of loops during the main GigaSpace pipeline pass (GLSL shader)
	 *
	 * @return the max number of loops
	 */
	unsigned int getMaxNbLoops() const;

	/**
	 * Set the max depth
	 *
	 * @param pValue the max depth
	 */
	void setMaxDepth( unsigned int pValue );

	/**
	 * Set the max number of loops during the main GigaSpace pipeline pass (GLSL shader)
	 *
	 * @param pValue the max number of loops
	 */
	void setMaxNbLoops( unsigned int pValue );

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
	 * This function is called by the user to render a frame.
	 *
	 * @param pModelMatrix the current model matrix
	 * @param pViewMatrix the current view matrix
	 * @param pProjectionMatrix the current projection matrix
	 * @param pViewport the viewport configuration
	 */
	virtual void postRender( const float4x4& pModelMatrix, const float4x4& pViewMatrix, const float4x4& pProjectionMatrix, const int4& pViewport );

	/**
	 * Initialize shader program
	 *
	 * @return a flag telling wheter or not it succeeds
	 */
	bool initializeShaderProgram();

	/**
	 * Finalize shader program
	 *
	 * @return a flag telling wheter or not it succeeds
	 */
	bool finalizeShaderProgram();

	/**
	 * Initialize shader program
	 *
	 * @return a flag telling wheter or not it succeeds
	 */
	bool initializeShaderProgramUniforms();
		
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
	 * Renderer context
	 * - to access to useful variables during rendering (view matrix, model matrix, etc...)
	 */
	GvRendering::GsRendererContext viewContext;

	/**
	 * Shader program
	 */
	GsGraphics::GsShaderProgram* _shaderProgram;

	/**
	 * Buffer of requests
	 */
	GLuint _updateBufferTBO;

	/**
	 * Node time stamps buffer
	 */
	GLuint _nodeTimeStampTBO;

	/**
	 * Brick time stamps buffer
	 */
	GLuint _brickTimeStampTBO;

	/**
	 * Node pool's child array (i.e. encoded data structure [octree, N3-Tree, etc...])
	 */
	GLuint _childArrayTBO;

	/**
	 * Node pool's data array (i.e. addresses of bricks associated to each node in cache)
	 */
	GLuint _dataArrayTBO;

	// Unmap GigaVoxels graphics resources from CUDA environment in order to be used by OpenGL
	cudaGraphicsResource* _graphicsResources[ 7 ];

	/**
	 * Cone aperture scale
	 */
	float _coneApertureScale;

	/**
	 * Max number of loops during the main GigaSpace pipeline pass (GLSL shader)
	 */
	unsigned int _maxNbLoops;

	/**
	 * Viewing System uniform parameters
	 */
	GLint _viewPosLoc;
	GLint _viewPlaneLoc;
	GLint _viewAxisXLoc;
	GLint _viewAxisYLoc;
	GLint _pixelSizeLoc;
	GLint _frustumNearInvLoc;
	// Cone aperture management
	GLint _coneApertureScaleLoc;
	// GigaSpace pipeline uniform parameters
	GLint _maxNbLoopsLoc;

	/**
	 * Locations of uniform variables
	 */
	GLint _nodeBufferLoc;
	GLint _dataBufferLoc;
	GLint _requestBufferLoc;
	GLint _nodeTimestampBufferLoc;
	GLint _brickTimestampBufferLoc;
	GLint _currentTimeLoc;

	/**
	 * Locations of uniform variables
	 */
	GLint _nodePoolResInvLoc;
	GLint _brickPoolResInvLoc;
	GLint _nodeCacheSizeLoc;
	GLint _brickCacheSizeLoc;
	GLint _dataPool_Channel_0_Loc;
	GLint _dataPool_Channel_1_Loc;
	GLint _maxDepthLoc;

	/**
	 * Locations of uniform variables
	 */
	GLint _positionLoc;

	// GigaVoxels/GigaSpace arrays
	GvCore::GsLinearMemory< uint >* _nodeBuffer;
	GvCore::GsLinearMemory< uint >* _dataBuffer;
	GvCore::GsLinearMemory< uint >* _requestBuffer;
	GvCore::GsLinearMemory< uint >* _nodeTimestampBuffer;
	GvCore::GsLinearMemory< uint >* _brickTimestampBuffer;

	/******************************** METHODS *********************************/

	/**
	 * Start the rendering process.
	 *
	 * @param pModelMatrix the current model matrix
	 * @param pViewMatrix the current view matrix
	 * @param pProjectionMatrix the current projection matrix
	 * @param pViewport the viewport configuration
	 */
	virtual void doRender( const float4x4& pModelMatrix, const float4x4& pViewMatrix, const float4x4& pProjectionMatrix, const int4& pViewport );

};

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "RendererGLSL.inl"

#endif // !_RENDERER_GLSL_H_
