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

#ifndef _OBJECT_RECEIVING_SHADOW_H_
#define _OBJECT_RECEIVING_SHADOW_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

#include "ShaderManager.h"
#include "Mesh.h"
namespace GvCore
{
	template<typename T>
	class Array3DGPULinear;
}

/** 
 * @class ObjectReceivingShadow
 *
 * @brief The ObjectReceivingShadow class provides the mecanisms to manage mesh/scene objects
 * that should receive shadows.
 *
 * This class is the base class for all "receiving shadow" objects.
 */
class ObjectReceivingShadow
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * ...
	 */
	ObjectReceivingShadow();

	/**
	 * ...
	 */
	void init();

	/**
	 * ...
	 */
	void render();

	/**
	 * ...
	 */
	void setLightPosition( float x, float y, float z );

	/**
	 * ...
	 */
	~ObjectReceivingShadow();

	/**
	 * ...
	 */
	void setBrickCacheSize( unsigned int x, unsigned int y, unsigned int z );

	/**
	 * ...
	 */
	void setBrickPoolResInv( float x, float y, float z );

	/**
	 * ...
	 */
	void setMaxDepth( unsigned int v );

	/**
	 * ...
	 */
	void setVolTreeChildArray( GvCore::Array3DGPULinear< unsigned int >* v, GLint id );

	/**
	 * ...
	 */
	void setVolTreeDataArray( GvCore::Array3DGPULinear< unsigned int >* v, GLint id );

	/**
	 * ...
	 */
	void setModelMatrix( float m00, float m01, float m02, float m03,
						float m10, float m11, float m12, float m13,
						float m20, float m21, float m22, float m23,
						float m30, float m31, float m32, float m33 );

	/**
	 * ...
	 */
	void setWorldLight( float x, float y, float z );

	/**
	 * ...
	 */
	void setTexBufferName( GLint v );

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
	 * ...
	 */
	bool loadedObject;//true if the user loads an object, false if he creates it manually

	/**
	 * ...
	 */
	//if the user wants to load a file 
	Mesh* object;

	/**
	 * ...
	 */
	//if he wants to define the object manually
	GLuint idVBO;

	/**
	 * ...
	 */
	GLuint idIndices;

	/**
	 * ...
	 */
	GLuint vshader;

	/**
	 * ...
	 */
	GLuint fshader;

	/**
	 * ...
	 */
	GLuint program;

	/**
	 * ...
	 */
	//Light positions
	float lightPos[ 3 ];

	/**
	 * ...
	 */
	float worldLight[ 3 ];

	/**
	 * ...
	 */
	//GigaVoxels object casting shadows stuff 
	unsigned int brickCacheSize[ 3 ];

	/**
	 * ...
	 */
	float brickPoolResInv[ 3 ];

	/**
	 * ...
	 */
	unsigned int maxDepth;

	/**
	 * ...
	 */
	GLuint _childArrayTBO;

	/**
	 * ...
	 */
	GLuint _dataArrayTBO;

	/**
	 * ...
	 */
	GvCore::Array3DGPULinear< unsigned int >* volTreeChildArray;

	/**
	 * ...
	 */
	GvCore::Array3DGPULinear< unsigned int >* volTreeDataArray;

	/**
	 * ...
	 */
	GLint childBufferName;

	/**
	 * ...
	 */
	GLint dataBufferName;

	/**
	 * ...
	 */
	GLint texBufferName;

	/**
	 * ...
	 */
	float modelMatrix[ 16 ];

	/******************************** METHODS *********************************/

};

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

//#include "ObjectReceivingShadow.inl"

#endif // !_OBJECT_RECEIVING_SHADOW_H_
