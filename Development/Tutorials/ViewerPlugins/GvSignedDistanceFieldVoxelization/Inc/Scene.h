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

struct node {
	unsigned int first;
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
	bool init( const char* pSceneFile );

	/**
	 * ...
	 */
	void draw() const;

	/**
	 * ...
	 */
	void draw( unsigned int depth, uint3 locCode ) const;
	
	/**
	 * ...
	 */
	void draw( unsigned int depth ) const;

	/**
	 * ...
	 */
	uint intersectMesh( unsigned int depth, uint3 locCode ) const;

	/**
	 * ...
	 */
	void setOctreeNode( unsigned int depth, uint3 locCode, unsigned int pFirst, int pCount );

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Initialize graphics resources
	 *
	 * @return a flag telling wheter or not process has succeeded
	 */
	bool initializeGraphicsResources();

	/**
	 * Finalize graphics resources
	 *
	 * @return a flag telling wheter or not process has succeeded
	 */
	bool finalizeGraphicsResources();

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/******************************* ATTRIBUTES *******************************/

	/**
	 * 3D scene
	 */
	const aiScene* _scene;

	/**
	 * Vertex array object
	 */
	GLuint _vao; 

	/**
	 * Vertex buffer objects
	 * - vertices, normals and indexes
	 */
	GLuint _buffers[ 3 ];

	unsigned int mNbTriangle;

	node* _octree;
	
	// Octree max depth
	unsigned int mDepthMax;

	unsigned int _depthMaxPrecomputed;

	/******************************** METHODS *********************************/
	
	/**
	 * ...
	 */
	void organizeIBO( std::vector<unsigned int> & IBO, const float *vertices );
	
	/**
	 * ...
	 */
	bool organizeNode( unsigned int d, uint3 locCode, std::vector<unsigned int> & IBO, const float *vertices );

	/**
	 * ...
	 */
	inline uint3 getFather( uint3 locCode ) const;

	/**
	 * Get the index of the brick in the octree
	 */
	inline unsigned int getIndex( unsigned int d, uint3 locCode ) const;
	
	/**
	 * ...
	 */
	inline bool triangleIntersectBick( const float3 & brickPos, 
								const float3 & brickSize, 
								unsigned int triangleIndex, 
								const std::vector<unsigned int> & IBO, 
								const float *vertices );
	/**
	* ...
	*/
	inline bool triangleAabbIntersection2D( const float2 & a, 
										    const float2 & b, 
											const float2 & c,
											const float4 & aabb );

};


/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "Scene.inl"

#endif // !_SCENE_H_
