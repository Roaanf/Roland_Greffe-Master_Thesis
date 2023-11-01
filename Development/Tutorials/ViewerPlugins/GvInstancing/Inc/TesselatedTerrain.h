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

#ifndef _TESSELATED_TERRAIN_H_
#define _TESSELATED_TERRAIN_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// OpenGL
#include <GL/glew.h>

// GigaVoxels
#include <GvCore/GsVectorTypesExt.h>

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

/** 
 * @class TesselatedTerrain
 *
 * @brief The TesselatedTerrain class provides an interface to manage terrains.
 *
 * ...
 */
class TesselatedTerrain
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
	TesselatedTerrain();

	/**
	 * Destructor
	 */
	virtual ~TesselatedTerrain();

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
	bool load( const std::string& pFilename, const std::string& pLandFilename );

	/**
	 * Render
	 */
	void render( const float4x4& pModelViewMatrix, const float4x4& pProjectionMatrix, const int4& pViewport ) const;

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

	/**
	 * Height map texture
	 */
	GLuint _heightmap;
	GLuint _land; // the land texture object

	/******************************** METHODS *********************************/

	/**
	 * Copy constructor forbidden.
	 */
	TesselatedTerrain( const TesselatedTerrain& );

	/**
	 * Copy operator forbidden.
	 */
	TesselatedTerrain& operator=( const TesselatedTerrain& );

};

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

//#include "TesselatedTerrain.inl"

#endif // !_TESSELATED_TERRAIN_H_
