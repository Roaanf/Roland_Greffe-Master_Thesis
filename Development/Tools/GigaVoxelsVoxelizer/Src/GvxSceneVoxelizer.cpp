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

#include "GvxSceneVoxelizer.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// Project
#include "GvxDataStructureIOHandler.h"
// STL
#include <iostream>
#include <cassert>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

// GvVoxelizer
using namespace Gvx;

// STL
using namespace std;

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ***************************** METHOD DEFINITION ******************************
 ******************************************************************************/

/******************************************************************************
 * Constructor
 ******************************************************************************/
GvxSceneVoxelizer::GvxSceneVoxelizer()
:	_filePath()
,	_fileName()
,	_fileExtension()
,	_maxResolution( 512 )
,	_isGenerateNormalsOn( false )
,	_brickWidth( 8 )
,	_dataType( GvxDataTypeHandler::gvUCHAR4 )
,	_voxelizerEngine()
{
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
GvxSceneVoxelizer::~GvxSceneVoxelizer()
{
}

/******************************************************************************
 * The main method called to voxelize a scene.
 * All settings must have been done previously (i.e. filename, path, etc...)
 ******************************************************************************/
bool GvxSceneVoxelizer::launchVoxelizationProcess()
{
	// Load/import scene
	loadScene();

	// Normalize the scene
	normalizeScene();

	// Voxelize the scene
	voxelizeScene();

	// mipmap();

	return false;
}

/******************************************************************************
 * Load/import the scene
 ******************************************************************************/
bool GvxSceneVoxelizer::loadScene()
{
	return false;
}

/******************************************************************************
 * Normalize the scene.
 * It determines the whole scene bounding box and then modifies vertices
 * to scale the scene.
 ******************************************************************************/
bool GvxSceneVoxelizer::normalizeScene()
{
	return false;
}

/******************************************************************************
 * Voxelize the scene
 ******************************************************************************/
bool GvxSceneVoxelizer::voxelizeScene()
{
	return false;
}

/******************************************************************************
 * Apply the mip-mapping algorithmn.
 * Given a pre-filtered voxel scene at a given level of resolution,
 * it generates a mip-map pyramid hierarchy of coarser levels (until 0).
******************************************************************************/
bool GvxSceneVoxelizer::mipmap()
{
	// The mip-map pyramid hierarchy is built recursively from adjacent levels.
	// Two files/streamers are used :
	// UP is an already pre-filtered scene at resolution [ N ]
	// DOWN is the coarser version to generate at resolution [ N - 1 ]
	GvxDataStructureIOHandler* dataStructureIOHandlerUP = new GvxDataStructureIOHandler( getFileName(), getMaxResolution(), getBrickWidth(), getDataType(), false );
	GvxDataStructureIOHandler* dataStructureIOHandlerDOWN = NULL;

	// Iterate through levels of resolution
	for ( int level = getMaxResolution() - 1; level >= 0; level-- )
	{
		// LOG info
		std::cout << "GvxVoxelizer::mipmap : level : " << level << std::endl;

		// The coarser data handler is allocated dynamically due to memory consumption considerations.
		dataStructureIOHandlerDOWN = new GvxDataStructureIOHandler( getFileName(), getMaxResolution(), getBrickWidth(), getDataType(), true );

		// Iterate through nodes of the structure
		unsigned int nodePos[ 3 ];
		for ( nodePos[ 2 ] = 0; nodePos[ 2 ] < dataStructureIOHandlerUP->_nodeGridSize; nodePos[ 2 ]++ )
		for ( nodePos[ 1 ] = 0; nodePos[ 1 ] < dataStructureIOHandlerUP->_nodeGridSize; nodePos[ 1 ]++ )
		for ( nodePos[ 0 ] = 0; nodePos[ 0 ] < dataStructureIOHandlerUP->_nodeGridSize; nodePos[ 0 ]++ )
		{
			// Retrieve the current node info
			unsigned int node = dataStructureIOHandlerUP->getNode( nodePos );

			// If node is empty, go to next node
			if ( GvxDataStructureIOHandler::isEmpty( node ) )
			{
				continue;
			}

			// Iterate through voxels of the current node
			unsigned int voxelPos[3];
			for ( voxelPos[ 2 ] = _brickWidth * nodePos[ 2 ]; voxelPos[ 2 ] < dataStructureIOHandlerUP->_brickWidth * ( nodePos[ 2 ] + 1 ); voxelPos[ 2 ] += 2 )
			for ( voxelPos[ 1 ] = _brickWidth * nodePos[ 1 ]; voxelPos[ 1 ] < dataStructureIOHandlerUP->_brickWidth * ( nodePos[ 1 ] + 1 ); voxelPos[ 1 ] += 2 )
			for ( voxelPos[ 0 ] = _brickWidth * nodePos[ 0 ]; voxelPos[ 0 ] < dataStructureIOHandlerUP->_brickWidth * ( nodePos[ 0 ] + 1 ); voxelPos[ 0 ] += 2 )
			{
				float voxelDataDOWNf[ 4 ] = { 0.f, 0.f, 0.f, 0.f };
				float voxelDataDOWNf2[ 4 ] = { 0.f, 0.f, 0.f, 0.f };

				// As the underlying structure is an octree,
				// to compute data at coaser level,
				// we need to iterate through 8 voxels and take the mean value.
				for ( unsigned int z = 0; z < 2; z++ )
				for ( unsigned int y = 0; y < 2; y++ )
				for ( unsigned int x = 0; x < 2; x++ )
				{
					// Retrieve position of voxel in the UP resolution version
					unsigned int voxelPosUP[ 3 ];
					voxelPosUP[ 0 ] = voxelPos[ 0 ] + x;
					voxelPosUP[ 1 ] = voxelPos[ 1 ] + y;
					voxelPosUP[ 2 ] = voxelPos[ 2 ] + z;

					// Get associated data (in the UP resolution version)
					unsigned char voxelDataUP[ 4 ];
					dataStructureIOHandlerUP->getVoxel( voxelPosUP, voxelDataUP, 0 );
					voxelDataDOWNf[ 0 ] += voxelDataUP[ 0 ];
					voxelDataDOWNf[ 1 ] += voxelDataUP[ 1 ];
					voxelDataDOWNf[ 2 ] += voxelDataUP[ 2 ];
					voxelDataDOWNf[ 3 ] += voxelDataUP[ 3 ];
				}

				// Coarser voxel is scaled from current UP voxel (2 times smaller for octree)
				unsigned int voxelPosDOWN[ 3 ];
				voxelPosDOWN[ 0 ] = voxelPos[ 0 ] / 2;
				voxelPosDOWN[ 1 ] = voxelPos[ 1 ] / 2;
				voxelPosDOWN[ 2 ] = voxelPos[ 2 ] / 2;

				// Set data in coarser voxel (take alpha into account)
				unsigned char vd[ 4 ];	// "vd" stands for "voxel data"
				float alpha = voxelDataDOWNf[ 3 ] / 8.f;
				if ( alpha > 0 )
				{
					float R = voxelDataDOWNf[ 0 ] / 8.f / alpha;
					float G = voxelDataDOWNf[ 1 ] / 8.f / alpha;
					float B = voxelDataDOWNf[ 2 ] / 8.f / alpha;

					alpha = min< float >( 2.f * alpha, 255.f );

					vd[ 0 ] = static_cast< unsigned char >( R * alpha );
					vd[ 1 ] = static_cast< unsigned char >( G * alpha );
					vd[ 2 ] = static_cast< unsigned char >( B * alpha );
					vd[ 3 ] = static_cast< unsigned char >( alpha );
				}
				else
				{
					vd[ 0 ] = 0;
					vd[ 1 ] = 0;
					vd[ 2 ] = 0;
					vd[ 3 ] = 0;
				}
				dataStructureIOHandlerDOWN->setVoxel( voxelPosDOWN, vd, 0 );
			}
		}

		// Generate the border data of the coarser scene
		dataStructureIOHandlerDOWN->computeBorders();

		// Destroy the coarser data handler (due to memory consumption considerations)
		delete dataStructureIOHandlerUP;

		// The mip-map pyramid hierarchy is built recursively from adjacent levels.
		// Now that the coarser version has been generated, a coarser one need to be generated from it.
		// So, the coarser one is the UP version.
		dataStructureIOHandlerUP = dataStructureIOHandlerDOWN;
	}

	// Free memory
	delete dataStructureIOHandlerDOWN;

	return true;
}

/******************************************************************************
 * Set the filter type
 *
 * 0 = mean
 * 1 = gaussian
 * 2 = laplacian
 ******************************************************************************/
void GvxSceneVoxelizer::setFilterType( int filterType )
{
	_voxelizerEngine.setFilterType( filterType );
}

/******************************************************************************
 * Set the number of application of the filter
 ******************************************************************************/
void GvxSceneVoxelizer::setFilterIterations( int nbFilterOperation )
{
	_voxelizerEngine.setNbFilterApplications( nbFilterOperation );
}

/******************************************************************************
 * Set whether or not we generate the normal field
 ******************************************************************************/
void GvxSceneVoxelizer::setNormals( bool normals )
{
	_voxelizerEngine.setNormals( normals );
}
