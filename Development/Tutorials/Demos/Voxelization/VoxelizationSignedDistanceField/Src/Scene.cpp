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

#include "Scene.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

#include <iostream>

// System
#include <cfloat>
#include <limits>

// Assimp
#include <assimp/cimport.h>
#include <assimp/postprocess.h>

//Debugue
#include <iostream>

//Shader
#include <GvUtils/GsShaderManager.h>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

#define BUFFER_OFFSET(i) ((void*)(i))

/**
 * Assimp library object to load 3D model (with a log mechanism)
 */
static aiLogStream stream;


/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ***************************** METHOD DEFINITION ******************************
 ******************************************************************************/

/******************************************************************************
 * Constructor
 ******************************************************************************/
Scene::Scene( unsigned int pMaxDepth )
:   mScene( NULL ),
	mNbTriangle(0),
	mDepthMaxPrecomputed( 5 ), // with 5 can take around 15 seconds
	mDepthMax( pMaxDepth )
{
	// lenght_octree = Somme( 8^i, i = 0:mDepthMax ) + 8 
	unsigned int length = ( powf( 8, mDepthMax + 1 ) - 1 ) / (float)7 ;
	mOctree = new node[ length ];
	for ( unsigned int i = 0; i < length; i++ ) {
		mOctree[i].first = 0;
		mOctree[i].count = -1;
	}
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
Scene::~Scene()
{
	
		glDeleteBuffers(3, &mBuffers[0]);
		glDeleteVertexArrays(1, &mVAO[0]);
		delete[] mOctree;
}

/******************************************************************************
 * ...
 ******************************************************************************/
void Scene::init( const char* pSceneFile )
{
	// Import the geometry
	stream = aiGetPredefinedLogStream( aiDefaultLogStream_STDOUT, NULL );
	aiAttachLogStream( &stream );
	mScene = aiImportFile( pSceneFile, 0 );	// aiProcessPreset_TargetRealtime_Fast );

	// Scale the geometry
	float minx = +std::numeric_limits<float>::max();//FLT_MAX;
	float miny = +std::numeric_limits<float>::max();//FLT_MAX;
	float minz = +std::numeric_limits<float>::max();//FLT_MAX;
	float maxx = -std::numeric_limits<float>::max();//-FLT_MAX;
	float maxy = -std::numeric_limits<float>::max();//-FLT_MAX;
	float maxz = -std::numeric_limits<float>::max();//-FLT_MAX;

	for (unsigned int meshIndex = 0; meshIndex < mScene->mNumMeshes; ++meshIndex)
	{
		const aiMesh *pMesh = mScene->mMeshes[meshIndex];

		for (unsigned int vertexIndex = 0; vertexIndex < pMesh->mNumVertices; ++vertexIndex)
		{
			minx = std::min(minx, pMesh->mVertices[vertexIndex].x);
			miny = std::min(miny, pMesh->mVertices[vertexIndex].y);
			minz = std::min(minz, pMesh->mVertices[vertexIndex].z);
			maxx = std::max(maxx, pMesh->mVertices[vertexIndex].x);
			maxy = std::max(maxy, pMesh->mVertices[vertexIndex].y);
			maxz = std::max(maxz, pMesh->mVertices[vertexIndex].z);
		}
	}

	float scale = 0.95f / std::max(std::max(maxx - minx, maxy - miny), maxz - minz);

	for (unsigned int meshIndex = 0; meshIndex < mScene->mNumMeshes; ++meshIndex)
	{
		const aiMesh *pMesh = mScene->mMeshes[meshIndex];

		for (unsigned int vertexIndex = 0; vertexIndex < pMesh->mNumVertices; ++vertexIndex)
		{
			pMesh->mVertices[vertexIndex].x = (pMesh->mVertices[vertexIndex].x - (maxx + minx) * 0.5f) * scale + 0.5f;
			pMesh->mVertices[vertexIndex].y = (pMesh->mVertices[vertexIndex].y - (maxy + miny) * 0.5f) * scale + 0.5f;
			pMesh->mVertices[vertexIndex].z = (pMesh->mVertices[vertexIndex].z - (maxz + minz) * 0.5f) * scale + 0.5f;
		}
	}
	
	// WARNING : we assume here that faces of the mesh are triangle. Plus we don't take care of scene tree structure...
		
	// Computing number of vertices and triangles:
	unsigned int nbVertices = 0;
	mNbTriangle = 0;
	
	for (unsigned int meshIndex = 0; meshIndex < mScene->mNumMeshes; ++meshIndex) {
		nbVertices += mScene->mMeshes[meshIndex]->mNumVertices;
		mNbTriangle += mScene->mMeshes[meshIndex]->mNumFaces;
	}

	float *vertices = new float[3*nbVertices];
	float *normals = new float[3*nbVertices]();
	unsigned int *count = new unsigned int[nbVertices](); // To count the normals, to average
	
	std::vector<unsigned int> IBO = std::vector<unsigned int>();
	// Reserve place to store the IBO
	// Estimate the length of the IBO 
	IBO.reserve( 3 * 3 * ( ( powf( 2, mDepthMax + 1 ) - 1 )  ) * mNbTriangle );
	// Resize vector to store the depth lvl 0 
	IBO.resize( 3*mNbTriangle );

	unsigned int offsetIBO = 0;
	unsigned int offsetVBO = 0;
	

	// First pass to fill IBO's depth lvl 0, vertices, and normal.
	for (unsigned int meshIndex = 0; meshIndex < mScene->mNumMeshes; ++meshIndex)
		{
			const aiMesh *pMesh = mScene->mMeshes[meshIndex];
			
			for (unsigned int faceIndex = 0; faceIndex < pMesh->mNumFaces; ++faceIndex)
				{
					const aiFace *pFace = &pMesh->mFaces[faceIndex];
					
					// Remark : we can compute different normal for same vertex, but new one overwrites the old one
					for (unsigned int vertIndex = 0; vertIndex < pFace->mNumIndices; ++vertIndex)
						{
							unsigned int index = pFace->mIndices[vertIndex];
							
							float normal[3];

							if (!pMesh->HasNormals()) {
								// We compute normal with cross product :
								
								// retrieve vertex index of the face
								int a = pFace->mIndices[0];
								int b = pFace->mIndices[1];
								int c = pFace->mIndices[2];
								
								float e1[3] = { pMesh->mVertices[b].x - pMesh->mVertices[a].x,
										pMesh->mVertices[b].y - pMesh->mVertices[a].y,
										pMesh->mVertices[b].z - pMesh->mVertices[a].z };
								
								float e2[3] = { pMesh->mVertices[c].x - pMesh->mVertices[a].x,
										pMesh->mVertices[c].y - pMesh->mVertices[a].y,
										pMesh->mVertices[c].z - pMesh->mVertices[a].z };
								
								normals[offsetVBO + 3*index + 0] += e1[1]*e2[2] - e1[2]*e2[1];
								normals[offsetVBO + 3*index + 1] += e1[2]*e2[0] - e1[0]*e2[2];
								normals[offsetVBO + 3*index + 2] += e1[0]*e2[1] - e1[1]*e2[0];

							} else {
								normals[offsetVBO + 3*index + 0] += pMesh->mNormals[index].x;
								normals[offsetVBO + 3*index + 1] += pMesh->mNormals[index].y;
								normals[offsetVBO + 3*index + 2] += pMesh->mNormals[index].z;
							}
							// To average normals
							count[ index ]++;
							
							vertices[offsetVBO + 3*index + 0] = pMesh->mVertices[index].x;
							vertices[offsetVBO + 3*index + 1] = pMesh->mVertices[index].y;
							vertices[offsetVBO + 3*index + 2] = pMesh->mVertices[index].z;
							
							IBO[offsetIBO + vertIndex] = index;							
						}
					offsetIBO += 3;
				}
			offsetVBO +=  mScene->mMeshes[meshIndex]->mNumVertices ;
		}
	
	// We average and normalize the sum of normals
	for ( unsigned int i = 0; i < nbVertices; i++ ) {

		// Average 
		normals[ 3*i + 0 ] /= count[i];
		normals[ 3*i + 1 ] /= count[i];
		normals[ 3*i + 2 ] /= count[i];
		
		// Normalizing the normal 
		float normal = sqrt ( normals[ 3*i + 0 ]*normals[ 3*i + 0 ] + 
		                      normals[ 3*i + 1 ]*normals[ 3*i + 1 ] +
							  normals[ 3*i + 2 ]*normals[ 3*i + 2 ] );
		normals[ 3*i + 0 ] /= normal;
		normals[ 3*i + 1 ] /= normal;
		normals[ 3*i + 2 ] /= normal;
	}

	// Rmk : Assimp seems to create one vertex per triangle so the above average step isn't usefull...


	// Organize the octree 
	organizeIBO( IBO, vertices );

	// create the VAO
	glGenVertexArrays(1, mVAO);
	glBindVertexArray(mVAO[0]);

	// create buffers for our vertex data
	glGenBuffers(3, mBuffers);

	//vertex coordinates buffer
	glBindBuffer(GL_ARRAY_BUFFER, mBuffers[0]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 3 * nbVertices, vertices, GL_STATIC_DRAW);
	glEnableVertexAttribArray( (GLuint)0 );
	glVertexAttribPointer( (GLuint)0, 3, GL_FLOAT, 0, 0, 0);

	//normals buffer
	glBindBuffer(GL_ARRAY_BUFFER, mBuffers[1]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 3 * nbVertices, normals, GL_STATIC_DRAW);
	glEnableVertexAttribArray( (GLuint)1 );
	glVertexAttribPointer( (GLuint)1 , 3, GL_FLOAT, 0, 0, 0);

	//index buffer
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mBuffers[2]);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int) * IBO.size(), IBO.data(), GL_STATIC_DRAW );

	// unbind the VAO
	glBindVertexArray(0);

	// deleting tab used 
	delete[] vertices;
	delete[] normals;

}



/******************************************************************************
 * ...
 ******************************************************************************/
void Scene::draw() const
{
	// render VAO
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mBuffers[2]);
	glBindVertexArray( mVAO[0] );
	glDrawElements(GL_TRIANGLES, mNbTriangle*3, GL_UNSIGNED_INT, 0);
	glBindVertexArray(0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

/******************************************************************************
 * ...
 ******************************************************************************/
void Scene::draw( unsigned int depth ) const
{
	for ( int i = ( powf( 8, depth ) - 1 ) / (float)7; i < ( powf( 8, depth +1 ) - 1 ) / (float)7; i++ ) {
		if ( mOctree[i].count != 0 ) {
			// render VAO	
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mBuffers[2]);
			glBindVertexArray( mVAO[0] );
			glDrawElements( GL_TRIANGLES,  
							mOctree[i].count, 
							GL_UNSIGNED_INT,
							(void*)(sizeof( unsigned int) * mOctree[i].first) );
			glBindVertexArray(0);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
		}
	}
}


void Scene::draw( unsigned int depth, uint3 locNode ) const
{
	// Compute index of the node in the octree
	unsigned int i = getIndex( depth, locNode );
	if ( depth > mDepthMaxPrecomputed ) {
		return draw( depth - 1, getFather( locNode ) );
	} else {
		if ( mOctree[i].count != 0 ) {
			// render VAO	
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mBuffers[2]);
			glBindVertexArray( mVAO[0] );
			glDrawElements( GL_TRIANGLES,  
							mOctree[i].count, 
							GL_UNSIGNED_INT,
							(void*)(sizeof( unsigned int) * mOctree[i].first) );
			glBindVertexArray(0);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
		}
	}
}

/******************************************************************************
 * ...
 ******************************************************************************/
uint Scene::intersectMesh( unsigned int depth, uint3 locCode ) const
{

	// Compute index of the node in the octree
	unsigned int i = getIndex( depth, locCode );
	if ( mOctree[i].count == -1 ) {
		return intersectMesh( depth - 1, getFather( locCode ) );
	} else {
		if ( mOctree[i].count != 0 ) {
			if (depth == mDepthMax ) {
				return  2;
			}
			return 1;
		}
		return  0;
	}
}

/**
* ...
*/
void Scene::organizeIBO( std::vector<unsigned int> & IBO, const float *vertices ) 
{
	std::list< uint3 > *nodes = new std::list< uint3 >();
	std::list< uint3 > *nodesNextDepth = new std::list< uint3 >();
	unsigned int d = 0;

	// Init algorithm with first node at depth 0
	mOctree[0].first = 0;
	mOctree[0].count = mNbTriangle * 3 ;
	// Move on the next depth level
	d++;
	nodes->push_back( make_uint3( 0, 0, 0 ) );
	nodes->push_back( make_uint3( 1, 0, 0 ) );
	nodes->push_back( make_uint3( 0, 1, 0 ) );
	nodes->push_back( make_uint3( 1, 1, 0 ) );
	nodes->push_back( make_uint3( 0, 0, 1 ) );
	nodes->push_back( make_uint3( 1, 0, 1 ) );
	nodes->push_back( make_uint3( 0, 1, 1 ) );
	nodes->push_back( make_uint3( 1, 1, 1 ) );

	while ( d <= mDepthMaxPrecomputed ) {
		
		// For each nodes of this depth we orginize the IBO 
		while ( ! nodes->empty() ) {

			// Retrieve next node to treat it 
			uint3 currentBrick = nodes->front();
			nodes->pop_front();

			if ( organizeNode( d, currentBrick, IBO, vertices ) ) {
				// If there are triangle in this brick we add its child to node list

				nodesNextDepth->push_back( currentBrick * 2 );
				nodesNextDepth->push_back( currentBrick * 2 + make_uint3( 1, 0, 0 ) );
				nodesNextDepth->push_back( currentBrick * 2 + make_uint3( 0, 1, 0 ) );
				nodesNextDepth->push_back( currentBrick * 2 + make_uint3( 1, 1, 0 ) );
				nodesNextDepth->push_back( currentBrick * 2 + make_uint3( 0, 0, 1 ) );
				nodesNextDepth->push_back( currentBrick * 2 + make_uint3( 1, 0, 1 ) );
				nodesNextDepth->push_back( currentBrick * 2 + make_uint3( 0, 1, 1 ) );
				nodesNextDepth->push_back( currentBrick * 2 + make_uint3( 1, 1, 1 ) );

			}
		}

		// We go on the next depth level
		d++;
		std::list< uint3 > *nodesAux = nodes;
		nodes = nodesNextDepth;
		nodesNextDepth = nodesAux ;
	}
}
	

/**
* ...
*/
bool Scene::organizeNode( unsigned int d, uint3 locCode, std::vector<unsigned int> & IBO, const float *vertices )
{
	// Retrieve brick size and real coordinate and father index
	// Rmk : we assume here that the border size is 1.0 / 8.0, but the value is TDataStructureType::BrickBorderSize.
	// We have to use template...
	float3 brickPos = make_float3( locCode ) / make_float3( 1 << d ) - make_float3( 1.0 / 8.0 ) /  make_float3( 1 << d );
	float3 brickSize = make_float3( 1.0 ) / make_float3( 1 << d ) + make_float3( 2.0 / 8.0 ) /  make_float3( 1 << d );

	unsigned int index = getIndex( d, locCode );
	unsigned int indexFather = getIndex( d - 1, getFather( locCode ) ); 
	unsigned int count = 0; // Number of triangle that intersect the brick
	unsigned int begin = IBO.size();

	// We test for each fater's triangle if they intertect this birck
	for ( unsigned int i = mOctree[ indexFather ].first; i < mOctree[ indexFather ].first + mOctree[ indexFather ].count ; i+=3 ) {
		if ( triangleIntersectBick( brickPos, brickSize, i, IBO, vertices ) ) {
			count+=3;
			IBO.push_back( IBO[ i ] );
			IBO.push_back( IBO[ i + 1 ] );
			IBO.push_back( IBO[ i + 2 ] );
		}
	}
	
	mOctree[ index ].first = begin;
	mOctree[ index ].count = count; 

	return ( count > 0 );
}

/**
* ...
*/
void Scene::setOctreeNode( unsigned int depth, uint3 locCode, unsigned int pFirst, int pCount ) {
	// We assume that depth < mDepthMax
	if ( depth > mDepthMaxPrecomputed ) {
		unsigned int i = getIndex( depth, locCode );
		mOctree[i].count = pCount;
		mOctree[i].first = pFirst;
	}
}
