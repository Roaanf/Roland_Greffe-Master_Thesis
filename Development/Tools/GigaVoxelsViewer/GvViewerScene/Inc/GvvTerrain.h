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

#ifndef _GVV_TERRAIN_H_
#define _GVV_TERRAIN_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvSceneConfig.h"
#include "GvvMesh.h"

// OpenGL
#include <GL/glew.h>

// STL
#include <vector>
#include <string>

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
namespace GsGraphics
{
	class GsShaderProgram;
}

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

namespace GvViewerScene
{

/** 
 * @class GvvTerrain
 *
 * @brief The GvvTerrain class provides an interface to manage terrains.
 *
 * ...
 */
class GVVIEWERSCENE_EXPORT GvvTerrain : public GvvMesh
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
	 */
	GvvTerrain();

	/**
	 * Destructor
	 */
	virtual ~GvvTerrain();

	/**
	 * Initialize
	 *
	 * @param pDataStructure data structure
	 * @param pDataProductionManager data production manager
	 */
	bool initialize();

	/**
	 * Finalize
	 */
	bool finalize();

	/**
	 * Load heightmap
	 *
	 * @param pFilename heightmap file
	 *
	 * @return ...
	 *
	 */
	//bool load( const std::string& pFilename );
	bool load( const std::string& pFilename, const std::string& pLandFilename );

	/**
	 * This function is the specific implementation method called
	 * by the parent GvIRenderer::render() method during rendering.
	 *
	 * @param pModelMatrix the current model matrix
	 * @param pViewMatrix the current view matrix
	 * @param pProjectionMatrix the current projection matrix
	 * @param pViewport the viewport configuration
	 */
	virtual void render( const glm::mat4& pModelViewMatrix, const glm::mat4& pProjectionMatrix, const glm::uvec4& pViewport );

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
	 * Shader program
	 */
	GsGraphics::GsShaderProgram* _shaderProgram;

	/**
	 * Vertex array
	 */
	GLuint _vertexArray;
	GLuint _vertexBuffer;
	//GLuint _indexBuffer;

	/**
	 * Height map texture
	 */
	GLuint _heightmap;
	GLuint _land; // the land texture object

	/******************************** METHODS *********************************/

	/**
	 * Copy constructor forbidden.
	 */
	GvvTerrain( const GvvTerrain& );
	
	/**
	 * Copy operator forbidden.
	 */
	GvvTerrain& operator=( const GvvTerrain& );

};

} // namespace GvViewerScene

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

//#include "Terrain.inl"

#endif // !_GVV_TERRAIN_H_
