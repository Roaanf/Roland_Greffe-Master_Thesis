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

#ifndef _SCENE_H_
#define _SCENE_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GL
#include <GL/glew.h>

// GigaVoxels
#include <GvCore/GsVectorTypesExt.h>

// Assimp
#include <assimp/scene.h>

// STL
#include <vector>
#include <list>

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/******************************************************************************
 ******************************** CLASS USED **********************************
 ******************************************************************************/

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/**
 * ...
 */
struct node
{
	/**
	 * ...
	 */
	unsigned int first;

	/**
	 * ...
	 */
	int count;
};

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

/** 
 * @class Scene
 *
 * @brief The Scene class provides helper class allowing to extract a scene from a file 
 * and drawing it with openGL
 *
 * 
 */
class Scene
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Constructor
	 */
	Scene( unsigned int pMaxDepth = 5 );

	/**
	 * Destructor
	 */
	~Scene();

	/**
	 * ...
	 */
	void init( const char* pSceneFile );

	/**
	 * Draw the scene
	 */
	void draw() const;

	/**
	 * Draw the scene contained in a specific region of space
	 *
	 * @param depth depth (i.e. level of resolution)
	 */
	void draw( unsigned int depth ) const;

	/**
	 * Draw the scene contained in a specific region of space
	 *
	 * @param depth depth (i.e. level of resolution)
	 * @param locCode localization code
	 */
	void draw( unsigned int depth, const uint3& locCode ) const;

	/**
	 * ...
	 */
	uint intersectMesh( unsigned int depth, const uint3& locCode ) const;

	/**
	 * Initialize a node in the node buffer
	 *
	 * @param pDepth node's depth localization info
	 * @param pCode node's code localization info
	 * @param pFirst start index of node in node buffer
	 * @param pCount number of primitives in node (i.e triangles)
	 */
	void setOctreeNode( unsigned int pDepth, const uint3& pCode, unsigned int pFirst, int pCount );

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
	 * ...
	 */
	const aiScene* _scene;

	/**
	 * ...
	 */
	GLuint mVAO[ 1 ]; 

	/**
	 * ...
	 */
	GLuint mBuffers[ 3 ];

	/**
	 * ...
	 */
	unsigned int mNbTriangle;

	/**
	 * ...
	 */
	node* mOctree;

	/**
	 * Octree max depth
	 */
	unsigned int mDepthMax;

	/**
	 * ...
	 */
	unsigned int mDepthMaxPrecomputed;

	/**
	 * ...
	 */
	unsigned int mIBOLengthMax;

	/******************************** METHODS *********************************/
	
	/**
	 * ...
	 */
	void organizeIBO( std::vector< unsigned int >& IBO, const float* vertices );
	
	/**
	 * ...
	 */
	bool organizeNode( unsigned int d, const uint3& locCode, std::vector< unsigned int >& IBO, const float* vertices );

	/**
	 * ...
	 */
	inline uint3 getFather( const uint3& locCode ) const;

	/**
	 * Compute global index of a node in the node buffer given its depth and code localization info
	 *
	 * TODO : use generic code => only valid for octree...
	 *
	 * @param pDepth node's depth localization info
	 * @param pCode node's code localization info
	 *
	 * return node's global index in the node buffer
	 */
	inline unsigned int getIndex( unsigned int pDepth, const uint3& pCode ) const;
	
	/**
	 * ...
	 */
	inline bool triangleIntersectBick( const float3& brickPos, 
								const float3& brickSize, 
								unsigned int triangleIndex, 
								const std::vector< unsigned int >& IBO, 
								const float* vertices );
	/**
	 * ...
	 */
	inline bool vertexIsInBrick( const float3& brickPos, 
								const float3& brickSize, 
								unsigned int vertexIndex,
								const float* vertices );

	/**
	 * ...
	 */
	void organizeIBOGlsl();

	/**
	 * ...
	 */
	bool organizeNodeGlsl( unsigned int d, const uint3& locCode, GLuint TBO, GLuint triangleCounter );

};

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "Scene.inl"

#endif // !_SCENE_H_
