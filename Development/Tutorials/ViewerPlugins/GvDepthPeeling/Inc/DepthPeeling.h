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

#ifndef _DEPTH_PEELING_H_
#define _DEPTH_PEELING_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GL
#include <GL/glew.h>

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
namespace GvUtils
{
	class GvShaderProgram;
}

// Assimp
struct aiScene;

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

/** 
 * @class Renderer
 *
 * @brief The Renderer class provides an implementation of a renderer
 * specialized for CUDA.
 *
 * That is the commun renderer that users may use.
 * It implements the renderImpl() method from GvRendering::GvIRenderer base class
 * and has access to the data structure, cache and producer through the inherited
 * GvVolumeTreeRenderer base class.
 */
class DepthPeeling
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
	 * @param pVolumeTree data structure to render
	 * @param pVolumeTreeCache cache
	 * @param pProducer producer of data
	 */
	DepthPeeling();

	/**
	 * Destructor
	 */
	virtual ~DepthPeeling();

	/**
	 * Initialize
	 *
	 * @return a flag telling wheter or not it succeeds
	 */
	bool initialize();

	/**
	 * Finalize
	 *
	 * @return a flag telling wheter or not it succeeds
	 */
	bool finalize();

	/**
	 * This function is the specific implementation method called
	 * by the parent GvIRenderer::render() method during rendering.
	 *
	 * @param pModelMatrix the current model matrix
	 * @param pViewMatrix the current view matrix
	 * @param pProjectionMatrix the current projection matrix
	 * @param pViewport the viewport configuration
	 */
	void render();

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Shader program
	 */
	GvUtils::GvShaderProgram* _meshShaderProgram;

	/**
	 * Shader program
	 */
	GvUtils::GvShaderProgram* _frontToBackPeelingShaderProgram;

	/**
	 * Shader program
	 */
	GvUtils::GvShaderProgram* _blenderShaderProgram;

	/**
	 * Shader program
	 */
	GvUtils::GvShaderProgram* _finalShaderProgram;

	// OpenGL buffer object
	GLuint _frameBuffers[ 2 ];
	GLuint _textures[ 2 ];
	GLuint _depthTextures[ 2 ];
	GLuint _blenderTexture;
	GLuint _blenderFrameBuffer;

	/**
	 * Proxy geometry 3D model is loaded into an Assimp scene
	 */
	const aiScene* _scene;

	/**
	 * VAO for mesh rendering (vertex array object)
	 */
	GLuint _meshVertexArray;
	GLuint _meshVertexBuffer;
	GLuint _meshIndexBuffer;
	
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

#include "DepthPeeling.inl"

#endif // _DEPTH_PEELING_H_
