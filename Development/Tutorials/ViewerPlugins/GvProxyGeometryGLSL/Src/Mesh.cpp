/*
 * GigaVoxels is a ray-guided streaming library used for efficient
 * 3D real-time rendering of highly detailed volumetric scenes.
 *
 * Copyright (C) 2011-2013 INRIA <http://www.inria.fr/>
 *
 * Authors : GigaVoxels Team
 *
 * GigaVoxels is distributed under a dual-license scheme.
 * You can obtain a specific license from Inria at gigavoxels-licensing@inria.fr.
 * Otherwise the default license is the GPL version 3.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/** 
 * @version 1.0
 */

#include "Mesh.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// Assimp
#include <assimp/cimport.h>
#include <assimp/postprocess.h>
#include <assimp/config.h>

// Cuda
#include <vector_types.h>

// Qt
#include <QCoreApplication>
#include <QString>
#include <QDir>
#include <QFileInfo>

// STL
#include <iostream>
#include <limits>

// System
#include <cassert>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

// GigaVoxels
using namespace GsGraphics;

// STL
using namespace std;

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/**
 * Assimp library object to load 3D model (with a log mechanism)
 */
static aiLogStream logStream;

/******************************************************************************
 ***************************** METHOD DEFINITION ******************************
 ******************************************************************************/

/******************************************************************************
 * Constructor
 ******************************************************************************/
Mesh::Mesh()
:	IMesh()
,	_scene( NULL )
{
	// Attach stdout to the logging system.
	// Get one of the predefine log streams. This is the quick'n'easy solution to 
	// access Assimp's log system. Attaching a log stream can slightly reduce Assimp's
	// overall import performance.
	logStream = aiGetPredefinedLogStream( aiDefaultLogStream_STDOUT, NULL );

	// Attach a custom log stream to the libraries' logging system.
	aiAttachLogStream( &logStream );
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
Mesh::~Mesh()
{
	finalize();
}

/******************************************************************************
 * Finalize
 *
 * @return a flag telling wheter or not it succeeds
 ******************************************************************************/
bool Mesh::finalize()
{
	// Clean Assimp library ressources
	if ( _scene != NULL )
	{
		//	If the call to aiImportFile() succeeds, the imported data is returned in an aiScene structure. 
		// The data is intended to be read-only, it stays property of the ASSIMP 
		// library and will be stable until aiReleaseImport() is called. After you're 
		// done with it, call aiReleaseImport() to free the resources associated with 
		// this file.
		aiReleaseImport( _scene );
		
		_scene = NULL;
	}
		
	// Detach a custom log stream from the libraries' logging system.
	aiDetachLogStream( &logStream );
	
	return true;
}

/******************************************************************************
 * Render
 ******************************************************************************/
void Mesh::render( const float4x4& pModelViewMatrix, const float4x4& pProjectionMatrix, const int4& pViewport )
{
	// Draw mesh
	glBindVertexArray( _vertexArray );
	if ( _useIndexedRendering )
	{
		glDrawElements( GL_TRIANGLES, _nbFaces * 3, GL_UNSIGNED_INT, NULL );
	}
	else
	{
		glDrawArrays( GL_TRIANGLES, 0, 0 );
	}
	glBindVertexArray( 0 );
}

/******************************************************************************
 * Read mesh data
 ******************************************************************************/
bool Mesh::read( const char* pFilename, std::vector< float3 >& pVertices, std::vector< float3 >& pNormals, std::vector< float2 >& pTexCoords, std::vector< unsigned int >& pIndices )
{
	// Delete the 3D scene if needed
	if ( _scene != NULL )
	{
		aiReleaseImport( _scene );
		_scene = NULL;
	}

	// ---- Load the 3D scene ----
	//aiSetImportPropertyInteger( AI_CONFIG_PP_SBP_REMOVE, aiPrimitiveType_POINT | aiPrimitiveType_LINE );
	//const unsigned int flags = 0;
	//const unsigned int flags = aiProcessPreset_TargetRealtime_Fast;
	//const unsigned int flags = aiProcessPreset_TargetRealtime_Quality;
	const unsigned int flags = aiProcessPreset_TargetRealtime_MaxQuality;
	_scene = aiImportFile( pFilename, flags );
	
	assert( _scene != NULL );

	// Compute mesh bounds
	float minX = +std::numeric_limits< float >::max();
	float minY = +std::numeric_limits< float >::max();
	float minZ = +std::numeric_limits< float >::max();
	float maxX = -std::numeric_limits< float >::max();
	float maxY = -std::numeric_limits< float >::max();
	float maxZ = -std::numeric_limits< float >::max();

	// Iterate through meshes
	for ( unsigned int i = 0; i < _scene->mNumMeshes; ++i )
	{
		// Retrieve current mesh
		const aiMesh* mesh = _scene->mMeshes[ i ];

		// Iterate through vertices
		for ( unsigned int j = 0; j < mesh->mNumVertices; ++j )
		{
			minX = std::min( minX, mesh->mVertices[ j ].x );
			minY = std::min( minY, mesh->mVertices[ j ].y );
			minZ = std::min( minZ, mesh->mVertices[ j ].z );
			maxX = std::max( maxX, mesh->mVertices[ j ].x );
			maxY = std::max( maxY, mesh->mVertices[ j ].y );
			maxZ = std::max( maxZ, mesh->mVertices[ j ].z );
		}
	}
	
	// Update mesh bounds
	_minX = minX;
	_minY = minY;
	_minZ = minZ;
	_maxX = maxX;
	_maxY = maxY;
	_maxZ = maxZ;

	// WARNING : we assume here that faces of the mesh are triangle. Plus we don't take care of scene tree structure...

	// Computing number of vertices and triangles:
	//
	// Iterate through meshes
	for ( unsigned int i = 0; i < _scene->mNumMeshes; ++i )
	{
		_nbVertices += _scene->mMeshes[ i ]->mNumVertices;
		_nbFaces += _scene->mMeshes[ i ]->mNumFaces;
	}
	pVertices.reserve( _nbVertices );
	pNormals.reserve( _nbVertices );
	pTexCoords.reserve( _nbVertices );
	pIndices.reserve( _nbFaces * 3 );

	// Iterate through meshes
	for ( unsigned int i = 0; i < _scene->mNumMeshes; ++i )
	{
		// Retrieve current mesh
		const aiMesh* mesh = _scene->mMeshes[ i ];

		// Iterate through vertices
		//
		// TO DO : extract IF for normal and texture coordinates to speed code
		for ( unsigned int j = 0; j < mesh->mNumVertices; ++j )
		{
			// Retrieve vertex position
			pVertices.push_back( make_float3( mesh->mVertices[ j ].x, mesh->mVertices[ j ].y, mesh->mVertices[ j ].z ) );
			
			// Retrieve vertex normal
			if ( _hasNormals )
			{
				pNormals.push_back( make_float3( mesh->mNormals[ j ].x, mesh->mNormals[ j ].y, mesh->mNormals[ j ].z ) );
			}

			// Retrieve texture coordinates
			if ( _hasTextureCoordinates )
			{
				// ...
			}
		}
	}

	// Retrieve face information for indexed rendering
	if ( _useIndexedRendering )
	{
		// Iterate through meshes
		for ( unsigned int i = 0; i < _scene->mNumMeshes; ++i )
		{
			// Retrieve current mesh
			const aiMesh* mesh = _scene->mMeshes[ i ];

			// Iterate through faces
			for ( unsigned int j = 0; j < mesh->mNumFaces; ++j )
			{
				// Retrieve current face
				const aiFace* face = &mesh->mFaces[ j ];

				// Remark : we can compute different normal for same vertex, but new one overwrites the old one
				for ( unsigned int k = 0; k < face->mNumIndices; ++k )
				{
					pIndices.push_back( face->mIndices[ k ] );
				}
			}
		}
	}
	
	return true;
}
