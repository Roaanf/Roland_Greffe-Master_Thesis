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

// Project
#include "ShaderManager.h"
#include "Mesh.h"

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
namespace GvCore
{
	template< typename T >
	class GsLinearMemory;
}

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

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
	 * Constructor
	 */
	ObjectReceivingShadow();

	/**
	 * Destructor
	 */
	virtual ~ObjectReceivingShadow();

	/**
	 * Initialize
	 */
	void init();

	/**
	 * Render
	 */
	void render();

	/**
	 * Set light position
	 *
	 * @param x ...
	 * @param y ...
	 * @param z ...
	 */
	void setLightPosition( float x, float y, float z );

	/**
	 * Set brick cache size
	 *
	 * @param x ...
	 * @param y ...
	 * @param z ...
	 */
	void setBrickCacheSize( unsigned int x, unsigned int y, unsigned int z );

	/**
	 * Set brick pool resolution inverse
	 *
	 * @param x ...
	 * @param y ...
	 * @param z ...
	 */
	void setBrickPoolResInv( float x, float y, float z );

	/**
	 * Set max depth
	 *
	 * @param x ...
	 */
	void setMaxDepth( unsigned int v );

	/**
	 * Set the data structure's node pool's child array (i.e. the octree)
	 *
	 * @param v ...
	 * @param id ...
	 */
	void setVolTreeChildArray( GvCore::GsLinearMemory< unsigned int >* v, GLint id );

	/**
	 * Set the data structure's node pool's data array (i.e. addresses of brick in cache)
	 *
	 * @param v ...
	 * @param id ...
	 */
	void setVolTreeDataArray( GvCore::GsLinearMemory< unsigned int >* v, GLint id );

	/**
	 * Set the model matrix
	 *
	 * @param m ...
	 * @param m ...
	 * @param m ...
	 * @param m ...
	 * @param m ...
	 * @param m ...
	 * @param m ...
	 * @param m ...
	 * @param m ...
	 * @param m ...
	 * @param m ...
	 * @param m ...
	 * @param m ...
	 * @param m ...
	 * @param m ...
	 * @param m ...
	 */
	void setModelMatrix( float m00, float m01, float m02, float m03,
						float m10, float m11, float m12, float m13,
						float m20, float m21, float m22, float m23,
						float m30, float m31, float m32, float m33 );

	/**
	 * Set light position in world coordinate system
	 *
	 * @param x ...
	 * @param y ...
	 * @param z ...
	 */
	void setWorldLight( float x, float y, float z );

	/**
	 * ...
	 *
	 * @param v ...
	 */
	void setTexBufferName( GLint v );

	/**
	 * Get the 3D model filename to load
	 *
	 * @return the 3D model filename to load
	 */
	const char* get3DModelFilename() const;

	/**
	 * Set the 3D model filename to load
	 *
	 * @param pFilename the 3D model filename to load
	 */
	void set3DModelFilename( const char* pFilename );

	/**
	 * Get the translation
	 *
	 * @param pX the translation on x axis
	 * @param pY the translation on y axis
	 * @param pZ the translation on z axis
	 */
	void getTranslation( float& pX, float& pY, float& pZ ) const;

	/**
	 * Set the translation
	 *
	 * @param pX the translation on x axis
	 * @param pY the translation on y axis
	 * @param pZ the translation on z axis
	 */
	void setTranslation( float pX, float pY, float pZ );

	/**
	 * Get the rotation
	 *
	 * @param pAngle the rotation angle (in degree)
	 * @param pX the x component of the rotation vector
	 * @param pY the y component of the rotation vector
	 * @param pZ the z component of the rotation vector
	 */
	void getRotation( float& pAngle, float& pX, float& pY, float& pZ ) const;

	/**
	 * Set the rotation
	 *
	 * @param pAngle the rotation angle (in degree)
	 * @param pX the x component of the rotation vector
	 * @param pY the y component of the rotation vector
	 * @param pZ the z component of the rotation vector
	 */
	void setRotation( float pAngle, float pX, float pY, float pZ );

	/**
	 * Get the uniform scale
	 *
	 * @param pValue the uniform scale
	 */
	void getScale( float& pValue ) const;

	/**
	 * Set the uniform scale
	 *
	 * @param pValue the uniform scale
	 */
	void setScale( float pValue );

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
	 * Flag to tell wheter or not the user loads an object or creates it manually
	 *
	 * - true if the user loads an object, false if he creates it manually
	 */
	bool _loadedObject;

	/**
	 * Mesh
	 *
	 * - used if users load a mesh file
	 */
	Mesh* _object;

	/**
	 * Vertex buffer
	 *
	 * - used if users define the object manually
	 */
	GLuint _vertexBuffer;

	/**
	 * Index buffer
	 *
	 * - used if users define the object manually
	 */
	GLuint _indexBuffer;

	/**
	 * Shader program
	 */
	GLuint _shaderProgram;

	/**
	 * Vertex shader
	 */
	GLuint _vertexShader;

	/**
	 * Fragment
	 */
	GLuint _fragmentShader;

	/**
	 * Light position
	 */
	float _lightPos[ 3 ];

	/**
	 * Light position in World coordinates system
	 */
	float _worldLight[ 3 ];

	// GigaVoxels object casting shadows stuff

	/**
	 * The GigaSpace brick cache size
	 */
	unsigned int _brickCacheSize[ 3 ];

	/**
	 * The GigaSpace brick pool resolution inverse
	 */
	float _brickPoolResInv[ 3 ];

	/**
	 * The GigaSpace max depth (of the data structure)
	 */
	unsigned int _maxDepth;

	/**
	 * The GigaSpace node pool's child array's texture (i.e. the octree)
	 */
	GLuint _childArrayTexture;

	/**
	 * The GigaSpace node pool's data array's texture (i.e. pointers of bricks of data in cache)
	 */
	GLuint _dataArrayTexture;

	/**
	 * The GigaSpace node pool's child array (i.e. the octree)
	 */
	GvCore::GsLinearMemory< unsigned int >* _nodePoolChildArray;

	/**
	 * The GigaSpace node pool's data array (i.e. pointers of bricks of data in cache)
	 */
	GvCore::GsLinearMemory< unsigned int >* _nodePoolDataArray;

	/**
	 * The GigaSpace node pool's child array's texture buffer (i.e. the octree)
	 */
	GLint _nodePoolChildArrayTextureBuffer;

	/**
	 * The GigaSpace node pool's data array's texture buffer (i.e. pointers of bricks of data in cache)
	 */
	GLint _nodePoolDataArrayTextureBuffer;

	/**
	 * The GigaSpace data pool's texture
	 *
	 * - it corresponds to one of the user-defined channel of the GigaSpace data structure (ex : color, normal, etc...)
	 */
	GLint _dataPoolTexture;

	/**
	 * Model matrix
	 */
	float _modelMatrix[ 16 ];

	/**
	 * 3D model filename (mesh or scene)
	 */
	std::string _modelFilename;

	/**
	 * Translation used to position the GigaVoxels data structure
	 */
	float _translation[ 3 ];

	/**
	 * Rotation used to position the GigaVoxels data structure
	 */
	float _rotation[ 4 ];

	/**
	 * Scale used to transform the GigaVoxels data structure
	 */
	float _scale;

	/******************************** METHODS *********************************/

};

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

//#include "ObjectReceivingShadow.inl"

#endif // !_OBJECT_RECEIVING_SHADOW_H_
