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

#include "GvVoxelizer/GsDataStructureMipmapGenerator.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include "GvVoxelizer/GsDataTypeHandler.h"
#include "GvVoxelizer/GsDataStructureIOHandler.h"

// STL
#include <vector>
#include <iostream>

// System
#include <cmath>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

// GvVoxelizer
using namespace GvVoxelizer;

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
GsDataStructureMipmapGenerator::GsDataStructureMipmapGenerator()
{
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
GsDataStructureMipmapGenerator::~GsDataStructureMipmapGenerator()
{
}

/******************************************************************************
 * Apply the mip-mapping algorithm.
 * Given a pre-filtered voxel scene at a given level of resolution,
 * it generates a mip-map pyramid hierarchy of coarser levels (until 0).
 *
 * @param pFilename 3D model file name
 * @param pDataResolution Data resolution
 ******************************************************************************/
bool GsDataStructureMipmapGenerator::generateMipmapPyramid( const std::string& pFileName, unsigned int pDataResolution, const std::vector< GsDataTypeHandler::VoxelDataType >& pDataTypes, GsDataStructureIOHandler* up, unsigned int brickSize)
{
	bool result = false;

	unsigned int dataResolution = pDataResolution;
	unsigned int levelOfResolution = static_cast< unsigned int >( log( static_cast< float >( dataResolution / brickSize) ) / log( static_cast< float >( 2 ) ) );
	unsigned int brickWidth = brickSize; // TO DO : template different size
	//GsDataTypeHandler::VoxelDataType dataType = GsDataTypeHandler::gvUCHAR4;	// TO DO : template differents type
	//GsDataTypeHandler::VoxelDataType dataType = GsDataTypeHandler::gvUCHAR;
	//GsDataTypeHandler::VoxelDataType dataType = GsDataTypeHandler::gvUSHORT;
	//std::vector< GsDataTypeHandler::VoxelDataType > dataTypes;
	//dataTypes.push_back( dataType );
	std::vector< GsDataTypeHandler::VoxelDataType > dataTypes = pDataTypes;

	// TO DO
	// Check parameters
	//...
	
	// The mip-map pyramid hierarchy is built recursively from adjacent levels.
	// Two files/streamers are used :
	// UP is an already pre-filtered scene at resolution [ N ]
	// DOWN is the coarser version to generate at resolution [ N - 1 ]
	GsDataStructureIOHandler* dataStructureIOHandlerUP = up;
	GsDataStructureIOHandler* dataStructureIOHandlerDOWN = NULL;

	// Iterate through levels of resolution
	for ( int level = levelOfResolution - 1; level >= 0; level-- )
	{
		// LOG info
		std::cout << "GvVoxelizerEngine::mipmap : level : " << level << std::endl;

		unsigned int * nodeData = dataStructureIOHandlerUP->_nodeData;
		unsigned short * brickData = dataStructureIOHandlerUP->_brickData;
		size_t _nodeGridSize = dataStructureIOHandlerUP->_nodeGridSize;
		unsigned short* upRangeData = dataStructureIOHandlerUP->_rangeData;

		// The coarser data handler was allocated dynamically due to memory consumption considerations.
		// here i pre allocate it to make it way faster
		size_t nbOfvalues = (1 << level);
		dataStructureIOHandlerDOWN = new GsDataStructureIOHandler( pFileName, level, brickWidth, dataTypes[0], true, dataStructureIOHandlerUP->getBufferSize()/2);

		// Iterate through nodes of the structure (UP structure ???)
		size_t nodePos[ 3 ];
		for ( nodePos[2] = 0; nodePos[2] < dataStructureIOHandlerUP->_nodeGridSize; nodePos[2]++ )
		for ( nodePos[1] = 0; nodePos[1] < dataStructureIOHandlerUP->_nodeGridSize; nodePos[1]++ )
		{
		for ( nodePos[ 0 ] = 0; nodePos[ 0 ] < dataStructureIOHandlerUP->_nodeGridSize; nodePos[ 0 ]++ )
		{
			// Retrieve the current node info
			unsigned int node = nodeData[ nodePos[0] + _nodeGridSize*(nodePos[1] + _nodeGridSize*nodePos[2]) ];
			
			// If node is empty, go to next node
			if ( GsDataStructureIOHandler::isEmpty( node ) )
			{
				continue;
			}

			unsigned int brickoffset = ( node & 0x3fffffff );
			unsigned short min = upRangeData[brickoffset * 2];
			unsigned short max = upRangeData[brickoffset * 2 + 1];
			// Iterate through voxels of the current node
			size_t voxelPos[ 3 ];
			// Le += 2 c'est pcq on fait un mipmap au niveau sup donc on a 8 fois moins de voxels (/2 par chaque dimension)
			for ( voxelPos[ 2 ] = brickWidth * nodePos[ 2 ]; voxelPos[ 2 ] < dataStructureIOHandlerUP->_brickWidth * ( nodePos[ 2 ] + 1 ); voxelPos[ 2 ] +=2 )
			for ( voxelPos[ 1 ] = brickWidth * nodePos[ 1 ]; voxelPos[ 1 ] < dataStructureIOHandlerUP->_brickWidth * ( nodePos[ 1 ] + 1 ); voxelPos[ 1 ] +=2 )
			for ( voxelPos[ 0 ] = brickWidth * nodePos[ 0 ]; voxelPos[ 0 ] < dataStructureIOHandlerUP->_brickWidth * ( nodePos[ 0 ] + 1 ); voxelPos[ 0 ] +=2 )
			{
				double voxelDataDOWNf = 0;
				float voxelDataDOWNf2[ 4 ] = { 0.f, 0.f, 0.f, 0.f };

				// As the underlying structure is an octree, to compute data at coarser level,
				// we need to iterate through 8 voxels and take the mean value.
				for (size_t z = 0; z < 2; z++ )
				for (size_t y = 0; y < 2; y++ )
				for (size_t x = 0; x < 2; x++ )
				{
					// Retrieve position of voxel in the UP resolution version
					size_t voxelPosUP[ 3 ];
					voxelPosUP[ 0 ] = voxelPos[ 0 ] + x;
					voxelPosUP[ 1 ] = voxelPos[ 1 ] + y;
					voxelPosUP[ 2 ] = voxelPos[ 2 ] + z;

					// Get associated data (in the UP resolution version)
					unsigned short voxelDataUP;
					//unsigned char voxelDataUP[ 1 ];
					//unsigned short voxelDataUP[ 1 ];
					size_t brickWidth = dataStructureIOHandlerUP->_brickWidth;
					size_t voxelPosInBrick[ 3 ];
					voxelPosInBrick[ 0 ] = voxelPosUP[ 0 ] % brickWidth;
					voxelPosInBrick[ 1 ] = voxelPosUP[ 1 ] % brickWidth;
					voxelPosInBrick[ 2 ] = voxelPosUP[ 2 ] % brickWidth;
					size_t index = voxelPosInBrick[ 0 ] + ( brickWidth + 2 ) * ( voxelPosInBrick[ 1 ] + ( brickWidth + 2 ) * voxelPosInBrick[ 2 ] );
					voxelDataUP = brickData[ brickoffset * dataStructureIOHandlerUP->_brickSize + index ];
					voxelDataDOWNf += (double) voxelDataUP;
										
#ifdef NORMALS
					// Get associated normal (in the UP resolution version)
					dataStructureIOHandlerUP->getVoxel( voxelPosUP, voxelDataUP, 1 );
					voxelDataDOWNf2[ 0 ] += 2.f * voxelDataUP[ 0 ] - 1.f;
					voxelDataDOWNf2[ 1 ] += 2.f * voxelDataUP[ 1 ] - 1.f;
					voxelDataDOWNf2[ 2 ] += 2.f * voxelDataUP[ 2 ] - 1.f;
					voxelDataDOWNf2[ 3 ] += 0.f;
#endif
				}

				// Coarser voxel is scaled from current UP voxel (2 times smaller for octree)ssssssssssssssssss
				size_t voxelPosDOWN[3];
				voxelPosDOWN[ 0 ] = voxelPos[ 0 ] / 2;
				voxelPosDOWN[ 1 ] = voxelPos[ 1 ] / 2;
				voxelPosDOWN[ 2 ] = voxelPos[ 2 ] / 2;

				// Set data in coarser voxel
				unsigned short vd;		// "vd" stands for "voxel data"
				//unsigned char vd[ 1 ];		// "vd" stands for "voxel data"
				//unsigned short vd[ 1 ];		// "vd" stands for "voxel data"
				vd = static_cast< unsigned short >( voxelDataDOWNf / 8.0 );
				dataStructureIOHandlerDOWN->setVoxel_buffered( voxelPosDOWN, &vd, 0 );
				
				/*
				Not needed since setVoxelBuffered already min max on vd !
				if (vd < min) {
					min = vd;qqqqqssssssssssssssss
				}
				if (vd > max) {
					max = vd;
				}
				*/
				dataStructureIOHandlerDOWN->updateRange( voxelPosDOWN, min, max);

#ifdef NORMALS
				// Set normal in coarser voxel
				float norm = sqrtf( voxelDataDOWNf2[ 0 ] * voxelDataDOWNf2[ 0 ] + voxelDataDOWNf2[ 1 ] * voxelDataDOWNf2[ 1 ] + voxelDataDOWNf2[ 2 ] * voxelDataDOWNf2[ 2 ] );
				vd[ 0 ] = static_cast< unsigned char >( ( 0.5f * ( voxelDataDOWNf2[ 0 ] / norm ) + 0.5f ) * 255.f );
				vd[ 1 ] = static_cast< unsigned char >( ( 0.5f * ( voxelDataDOWNf2[ 1 ] / norm ) + 0.5f ) * 255.f );
				vd[ 2 ] = static_cast< unsigned char >( ( 0.5f * ( voxelDataDOWNf2[ 2 ] / norm ) + 0.5f ) * 255.f );
				vd[ 3 ] = 0;
				dataStructureIOHandlerDOWN->setVoxel( voxelPosDOWN, vd, 1 );
#endif
			}
		}
		}

		// Generate the border data of the coarser scene
		
		dataStructureIOHandlerDOWN->computeBorders();

		dataStructureIOHandlerDOWN->writeFiles();
		
		// Destroy the coarser data handler (due to memory consumption considerations)
		delete dataStructureIOHandlerUP;
		
		// The mip-map pyramid hierarchy is built recursively from adjacent levels.
		// Now that the coarser version has been generated, a coarser one need to be generated from it.
		// So, the coarser one is the UP version.
		dataStructureIOHandlerUP = dataStructureIOHandlerDOWN;
	}

	// Free memory
	delete dataStructureIOHandlerDOWN;

	result = true;

	return result;
}
