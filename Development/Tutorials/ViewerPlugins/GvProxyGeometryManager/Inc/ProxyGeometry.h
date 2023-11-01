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

#ifndef _PROXY_GEOMETRY_H_
#define _PROXY_GEOMETRY_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GL
#include <GL/glew.h>

// GigaVoxels
#include <GvCore/GsVectorTypesExt.h>

// STL
#include <string>
#include <vector>
#include <map>

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

// Project
class IMesh;

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

/** 
 * @class ProxyGeometry
 *
 * @brief The ProxyGeometry class provides an interface for proxy geometry management.
 *
 * Proxy geometry are used to provide depth map of front faces and back faces
 */
class ProxyGeometry
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	GLuint _depthMinTex;
	GLuint _depthMaxTex;

	/******************************** METHODS *********************************/

	/**
	 * Constructor
	 */
	ProxyGeometry();

	/**
	 * Destructor
	 */
	virtual ~ProxyGeometry();

	/**
	 * Initialize
	 *
	 * @return a flag telling wheter or not it succeeds
	 */
	virtual bool initialize();

	/**
	 * Finalize
	 *
	 * @return a flag telling wheter or not it succeeds
	 */
	virtual bool finalize();

	/**
	 * This function is the specific implementation method called
	 * by the parent GsIRenderer::render() method during rendering.
	 *
	 * @param pModelMatrix the current model matrix
	 * @param pViewMatrix the current view matrix
	 * @param pProjectionMatrix the current projection matrix
	 * @param pViewport the viewport configuration
	 */
	virtual void render( const float4x4& pModelViewMatrix, const float4x4& pProjectionMatrix, const int4& pViewport );

	/**
	 * Set buffer size
	 *
	 * @param pWidth buffer width
	 * @param pHeight buffer height
	 */
	void setBufferSize( int pWidth, int pHeight );

	/**
	 * Get the 3D model filename to load
	 *
	 * @return the 3D model filename to load
	 */
	const std::string& get3DModelFilename() const;

	/**
	 * Set the 3D model filename to load
	 *
	 * @param pFilename the 3D model filename to load
	 */
	void set3DModelFilename( const std::string& pFilename );

	/**
	 * Tell wheter or not the screen-based criteria is activated
	 *
	 * @return a flag telling wheter or not the screen-based criteria is activated
	 */
	bool getScreenBasedCriteria() const;

	/**
	 * Set the flag telling wheter or not the screen-based criteria is activated
	 *
	 * @param pFlag a flag telling wheter or not the screen-based criteria is activated
	 */
	void setScreenBasedCriteria( bool pFlag );

	/**
	 * Get the screen-based criteria coefficient
	 *
	 * @return the screen-based criteria coefficient
	 */
	float getScreenBasedCriteriaCoefficient() const;

	/**
	 * Set the screen-based criteria coefficient
	 *
	 * @param pValue the screen-based criteria coefficient
	 */
	void setScreenBasedCriteriaCoefficient( float pValue );

	/**
	 * Get the material alpha correction coefficient
	 *
	 * @return the material alpha correction coefficient
	 */
	float getMaterialAlphaCorrectionCoefficient() const;

	/**
	 * Set the material alpha correction coefficient
	 *
	 * @param pValue the material alpha correction coefficient
	 */
	void setMaterialAlphaCorrectionCoefficient( float pValue );

	/**
	 * Get the associated mesh
	 *
	 * @return the associated mesh
	 */
	const IMesh* getMesh() const;

	/**
	 * Get the associated mesh
	 *
	 * @return the associated mesh
	 */
	IMesh* editMesh();
	
	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Mesh
	 */
	IMesh* _mesh;

	/**
	 * Shader program
	 */
	GsGraphics::GsShaderProgram* _shaderProgram;

	/**
	 * Shader program
	 */
	GsGraphics::GsShaderProgram* _meshShaderProgram;

	/**
	 * Shadow map's FBO (frame buffer object)
	 */
	GLuint _frameBuffer;
	/*GLuint _depthMinTex;
	GLuint _depthMaxTex;*/
	GLuint _depthTex;
	GLsizei _bufferWidth;
	GLsizei _bufferHeight;

	/**
	 * 3D model filename
	 */
	std::string _filename;

	/**
	 * Flag telling wheter or not the screen-based criteria is activated
	 */
	bool _screenBasedCriteria;

	/**
	 * the screen-based criteria coefficient
	 */
	float _screenBasedCriteriaCoefficient;

	/**
	 * Material alpha correction coefficient
	 *
	 * - its the traversed distance at which full opacity will be reached inside matter
	 */
	float _materialAlphaCorrectionCoefficient;
	
	/******************************** METHODS *********************************/

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
	ProxyGeometry( const ProxyGeometry& );

	/**
	 * Copy operator forbidden.
	 */
	ProxyGeometry& operator=( const ProxyGeometry& );

};

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "ProxyGeometry.inl"

#endif // _PROXY_GEOMETRY_H_
