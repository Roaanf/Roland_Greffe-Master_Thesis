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

#include "GvVoxelizer/GsDataStructureIOHandler.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// STL
#include <sstream>
#include <iostream>

#include <cstdio>
#include <cstring>

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

/**
 * Empty node flag
 */
const unsigned int GsDataStructureIOHandler::_cEmptyNodeFlag = 0;

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ***************************** METHOD DEFINITION ******************************
 ******************************************************************************/

/******************************************************************************
 * Constructor
 *
 * @param pName Name of the data (.i.e. sponza, dragon, sibenik, etc...)
 * @param pLevel level of resolution of the data structure
 * @param pBrickWidth width of bricks in the data structure
 * @param pDataType type of voxel data (i.e. uchar4, float, float4, etc...)
 * @param pNewFiles a flag telling whether or not "new files" are used
 ******************************************************************************/
GsDataStructureIOHandler::GsDataStructureIOHandler( const std::string& pName, 
							size_t pLevel,
							size_t pBrickWidth,
							GsDataTypeHandler::VoxelDataType pDataType,
							bool pNewFiles,
							size_t pTrueNbOfValues)
// TO DO
// attention � l'ordre des initializations...
:	_nodeFile( NULL )
,	_level( pLevel )
,	_brickWidth( pBrickWidth )
//,	_brickSize( ( pBrickWidth) * ( pBrickWidth) * ( pBrickWidth) )
,	_brickSize( ( pBrickWidth + 2 ) * ( pBrickWidth + 2 ) * ( pBrickWidth + 2 ) )
,	_isBufferLoaded( false )
,	_brickNumber( 0 )
,	_nodeGridSize( (size_t)1 << pLevel )
,	_voxelGridSize( _nodeGridSize * pBrickWidth )
,   _trueNbOfValues(pTrueNbOfValues)
{
	// Store the voxel data type
	_dataTypes.push_back( pDataType );

	// Initialize all the files that will be generated.
	// Note : Associated brick buffer(s) will be created/initialized.

	openFiles( pName, pNewFiles );

	std::cout << "GsDataStructureIOHandler::GsDataStructureIOHandler" << std::endl;

	std::cout << (size_t)_nodeGridSize * (size_t)_nodeGridSize * (size_t)_nodeGridSize * (size_t)_brickSize << std::endl;
	std::cout << (size_t)_nodeGridSize * (size_t)_nodeGridSize * (size_t)_nodeGridSize << std::endl;

	size_t sizeBufferCalloc = (_trueNbOfValues * 1.5); // TEMP FIX !!!
	std::cout << sizeBufferCalloc << std::endl;
	std::cout << sizeBufferCalloc * sizeof(unsigned short) << std::endl;
	
	// Initialize buffers
	_brickData = (unsigned short *) calloc( sizeBufferCalloc, sizeof( unsigned short ) );
	
	if (_brickData == NULL) {
		std::cout << "Couldn't allocate the brickdata buffer" << std::endl;
	}

	_nodeData = (unsigned int *) calloc( (size_t)_nodeGridSize * (size_t)_nodeGridSize * (size_t)_nodeGridSize, sizeof( unsigned int ) );

	if (_nodeData == NULL) {
		std::cout << "Couldn't allocate the nodedata buffer" << std::endl;
	}
}

/******************************************************************************
 * Constructor
 *
 * @param pName Name of the data (.i.e. sponza, dragon, sibenik, etc...)
 * @param pLevel level of resolution of the data structure
 * @param pBrickWidth width of bricks in the data structure
 * @param pDataTypes types of voxel data (i.e. uchar4, float, float4, etc...)
 * @param pNewFiles a flag telling whether or not "new files" are used
 ******************************************************************************/
GsDataStructureIOHandler::GsDataStructureIOHandler( const std::string& pName, 
							size_t pLevel,
							size_t pBrickWidth,
							const vector< GsDataTypeHandler::VoxelDataType >& pDataTypes,
							bool pNewFiles)
// TO DO
// attention � l'ordre des initializations...
:	_nodeFile( NULL )
,	_level( pLevel )
,	_brickWidth( pBrickWidth )
//,	_brickSize( (pBrickWidth) * ( pBrickWidth) * (pBrickWidth) )
,	_brickSize( ( pBrickWidth + 2 ) * ( pBrickWidth + 2 ) * ( pBrickWidth + 2 ) )
,	_dataTypes( pDataTypes )
,	_isBufferLoaded( false )
,	_brickNumber( 0 )
,	_nodeGridSize((size_t)1 << pLevel )
,	_voxelGridSize( _nodeGridSize * pBrickWidth )
,   _trueNbOfValues(0)
{
	// Store the voxel data type
	_dataTypes = pDataTypes;

	// Initialize all the files that will be generated.
	// Initialize buffers
	_brickData = (unsigned short *) calloc( (size_t)_nodeGridSize * (size_t)_nodeGridSize * (size_t)_nodeGridSize * (size_t)_brickSize, sizeof( unsigned short ) );

	_nodeData = (unsigned int *) calloc( (size_t)_nodeGridSize * (size_t)_nodeGridSize * (size_t)_nodeGridSize, sizeof( unsigned int ) );
	// Note : Associated brick buffer(s) will be created/initialized.
	openFiles( pName, pNewFiles );

	std::cout << "GsDataStructureIOHandler::GsDataStructureIOHandler" << std::endl;
}

/******************************************************************************
 * ...
 ******************************************************************************/
GsDataStructureIOHandler::~GsDataStructureIOHandler()
{
	if ( _isBufferLoaded )
	{
		saveNodeandBrick_buffered();
	}

	fclose( _nodeFile );

	free(_nodeData);
	free(_brickData);

	for ( unsigned int c = 0; c < _dataTypes.size(); ++c )
	{
//		delete [] _brickBuffers[ c ];
		operator delete( _brickBuffers[ c ] );
		fclose( _brickFiles[ c ] );
	}
}

/******************************************************************************
 * Get the current brick number
 *
 * @return the current brick number
 ******************************************************************************/
unsigned int GsDataStructureIOHandler::getBrickNumber() const
{
	return _brickNumber;
}

/******************************************************************************
 * Get the voxel size at current level of resolution
 *
 * @return the voxel size
 ******************************************************************************/
float GsDataStructureIOHandler::getVoxelSize() const
{
	return 1.f / static_cast< float >( _voxelGridSize );
}


/******************************************************************************
 * Set data in a voxel at given data channel
 *
 * @param pVoxelPos voxel position
 * @param pVoxelData voxel data
 * @param pDataChannel data channel index
 * @param pInputDataSize size of the input data (in bytes), it may be smaller 
 * than the type of the channel. If 0, use sizeof( channel ) instead.
 ******************************************************************************/
void GsDataStructureIOHandler::setVoxel_buffered( size_t pVoxelPos[ 3 ], const void* pVoxelData, unsigned int pDataChannel, unsigned int pInputDataSize )
{
	// Retrieve the associated node position in which the voxel resides
	size_t nodePos[ 3 ];
	nodePos[ 0 ] = pVoxelPos[ 0 ] / _brickWidth;
	nodePos[ 1 ] = pVoxelPos[ 1 ] / _brickWidth;
	nodePos[ 2 ] = pVoxelPos[ 2 ] / _brickWidth;

	unsigned int node = _nodeData[nodePos[0] + _nodeGridSize * (nodePos[1] + _nodeGridSize * nodePos[2])];

	// If node is empty, as we set a voxel data, the node information needs to be updated
	if( node == _cEmptyNodeFlag)
	{
		// Update node info
		// Mark the node as a region containing data (i.e. 0x40000000u flag) and add the associated brick index
		node = 0x40000000 | _brickNumber;

		// Append 0 to brick files -> no need I calloced the brickData
		//for ( unsigned int i = 0; i < _brickSize; ++i ) {
		//	_brickData[_brickNumber * _brickSize + i] = 0;
		//}

		// Update the bricks counter
		_brickNumber++;
		_nodeData[nodePos[0] + _nodeGridSize * (nodePos[1] + _nodeGridSize * nodePos[2])] = node;
	}
 
	// Retrieve the voxel position in the current brick
	// (take into account the border)

	unsigned int offset = getBrickOffset(node);

	size_t voxelPosInBrick[ 3 ];
	/*
	voxelPosInBrick[ 0 ] = pVoxelPos[ 0 ] % _brickWidth;
	voxelPosInBrick[ 1 ] = pVoxelPos[ 1 ] % _brickWidth;
	voxelPosInBrick[ 2 ] = pVoxelPos[ 2 ] % _brickWidth;
	*/

	voxelPosInBrick[ 0 ] = pVoxelPos[ 0 ] % _brickWidth + 1;
	voxelPosInBrick[ 1 ] = pVoxelPos[ 1 ] % _brickWidth + 1;
	voxelPosInBrick[ 2 ] = pVoxelPos[ 2 ] % _brickWidth + 1;

	// Write voxel data
	unsigned short* voxelData = (unsigned short *) pVoxelData;
	_brickData[offset * _brickSize + voxelPosInBrick[0] + (_brickWidth + 2) * (voxelPosInBrick[1] + (_brickWidth + 2) * voxelPosInBrick[2])] = *voxelData;
}

/******************************************************************************
 * Get data in a voxel at given data channel
 *
 * @param pVoxelPos voxel position
 * @param pVoxelData voxel data
 * @param pDataChannel data channel index
 ******************************************************************************/
void GsDataStructureIOHandler::getVoxel_buffered( size_t pVoxelPos[ 3 ], void* voxelData, unsigned int pDataChannel )
{
	// Retrieve the associated node position in which the voxel resides
	size_t nodePos[ 3 ];
	nodePos[ 0 ] = pVoxelPos[ 0 ] / _brickWidth;
	nodePos[ 1 ] = pVoxelPos[ 1 ] / _brickWidth;
	nodePos[ 2 ] = pVoxelPos[ 2 ] / _brickWidth;
	
	// Retrieve the voxel position in the current brick
	size_t voxelPosInBrick[ 3 ];
	voxelPosInBrick[ 0 ] = pVoxelPos[ 0 ] % _brickWidth;
	voxelPosInBrick[ 1 ] = pVoxelPos[ 1 ] % _brickWidth;
	voxelPosInBrick[ 2 ] = pVoxelPos[ 2 ] % _brickWidth;

	// Load data (eventually from cache)
	loadNodeandBrick_buffered( nodePos );

	// Copy data from memory
	memcpy( voxelData,
			GsDataTypeHandler::getAddress( _dataTypes[ pDataChannel ], _brickBuffers[ pDataChannel ], voxelPosInBrick[ 0 ] + ( _brickWidth + 2 ) * ( voxelPosInBrick[ 1 ] + ( _brickWidth + 2 ) * voxelPosInBrick[ 2 ] ) ),		
			GsDataTypeHandler::canalByteSize( _dataTypes[ pDataChannel ] ) );
}


/******************************************************************************
 * Get brick data in a node at given data channel
 *
 * @param pNodePos node position
 * @param pBrickData brick data
 * @param pDataChannel data channel index
 ******************************************************************************/
void GsDataStructureIOHandler::getBrick_buffered( size_t pNodePos[ 3 ], void* pBrickData, unsigned int pDataChannel )
{
	// Retrieve node info and associated brick data
	unsigned int node = _nodeData[ pNodePos[0] + _nodeGridSize*(pNodePos[1] + _nodeGridSize*pNodePos[2]) ];
	unsigned int offset = getBrickOffset(node);

	// Read data from memory
	memcpy( pBrickData, _brickData + offset*_brickSize , _brickSize * GsDataTypeHandler::canalByteSize( _dataTypes[ pDataChannel ] ) );
}

/******************************************************************************
 * Set brick data in a node at given data channel
 *
 * @param pNodePos node position
 * @param pBrickData brick data
 * @param pDataChannel data channel index
 ******************************************************************************/
void GsDataStructureIOHandler::setBrick_buffered( size_t pNodePos[ 3 ], void* pBrickData, unsigned int pDataChannel )
{
	// Retrieve node info and associated brick data
	unsigned int node = _nodeData[ pNodePos[0] + _nodeGridSize*(pNodePos[1] + _nodeGridSize*pNodePos[2]) ];
	unsigned int offset = getBrickOffset(node);
	// Write data in memory
	memcpy( _brickData + offset*_brickSize , pBrickData, _brickSize * GsDataTypeHandler::canalByteSize( _dataTypes[ pDataChannel ] ) );
}

/******************************************************************************
 * Convert a normalized node position to its indexed node position
 *
 * @param pNormalizedNodePos normalized node position
 * @param pNodePos indexed node position
  ******************************************************************************/
void GsDataStructureIOHandler::getNodePosition( float pNormalizedNodePos[ 3 ], size_t pNodePos[ 3 ] )
{
	pNodePos[ 0 ] = static_cast< size_t >( pNormalizedNodePos[ 0 ] * static_cast< float >( _nodeGridSize ) );
	pNodePos[ 1 ] = static_cast< size_t >( pNormalizedNodePos[ 1 ] * static_cast< float >( _nodeGridSize ) );
	pNodePos[ 2 ] = static_cast< size_t >( pNormalizedNodePos[ 2 ] * static_cast< float >( _nodeGridSize ) );
}

/******************************************************************************
 * Convert a normalized voxel position to its indexed voxel position
 *
 * @param pNormalizedVoxelPos normalized voxel position
 * @param pVoxelPos indexed voxel position
 ******************************************************************************/
void GsDataStructureIOHandler::getVoxelPosition( float pNormalizedVoxelPos[ 3 ], size_t pVoxelPos[ 3 ] )
{
	pVoxelPos[ 0 ] = static_cast< size_t >( pNormalizedVoxelPos[ 0 ] * static_cast< float >( _voxelGridSize ) );
	pVoxelPos[ 1 ] = static_cast< size_t >( pNormalizedVoxelPos[ 1 ] * static_cast< float >( _voxelGridSize ) );
	pVoxelPos[ 2 ] = static_cast< size_t >( pNormalizedVoxelPos[ 2 ] * static_cast< float >( _voxelGridSize ) );
}

/******************************************************************************
 * Convert a normalized voxel position to its indexed voxel position in its associated brick
 *
 * @param pNormalizedVoxelPos normalized voxel position
 * @param pVoxelPosInBrick indexed voxel position in its associated brick
 ******************************************************************************/
void GsDataStructureIOHandler::getVoxelPositionInBrick( float pNormalizedVoxelPos[ 3 ], size_t pVoxelPosInBrick[ 3 ] )
{
	pVoxelPosInBrick[ 0 ] = ( ( static_cast< size_t >( pNormalizedVoxelPos[ 0 ] * static_cast< float >( _voxelGridSize ) ) ) % _brickWidth ) + 1;
	pVoxelPosInBrick[ 1 ] = ( ( static_cast< size_t >( pNormalizedVoxelPos[ 1 ] * static_cast< float >( _voxelGridSize ) ) ) % _brickWidth ) + 1;
	pVoxelPosInBrick[ 2 ] = ( ( static_cast< size_t >( pNormalizedVoxelPos[ 2 ] * static_cast< float >( _voxelGridSize ) ) ) % _brickWidth ) + 1;
}

/******************************************************************************
 * Get the node info associated to an indexed node position
 *
 * @param pNodePos an indexed node position
 *
 * @return node info (address + brick index)
 ******************************************************************************/
unsigned int GsDataStructureIOHandler::getNode_buffered( size_t pNodePos[ 3 ] )
{
	// Retrieve node info and associated brick data
	loadNodeandBrick_buffered( pNodePos );

	// Return the associated node info
	return _nodeBuffer;
}

/******************************************************************************
 * Retrieve node info and brick data associated to a node position.
 * Data is retrieved from disk if not already in cache, otherwise exit.
 *
 * Data is stored in buffers if not yet in cache.
 *
 * Note : Data is written on disk a previous node have been processed
 * and a new one is requested.
 *
 * @param pNodePos node position
 ******************************************************************************/
void GsDataStructureIOHandler::loadNodeandBrick_buffered( size_t pNodePos[ 3 ] )
{
	// Try to find node in cache.
	// If yes, exit.
	if ( ( _isBufferLoaded && 
		pNodePos[ 0 ] == _nodeBufferPos[ 0 ] && 
		pNodePos[ 1 ] == _nodeBufferPos[ 1 ] && 
		pNodePos[ 2 ] == _nodeBufferPos[ 2 ] ) )
	{
		return;
	}

	// If a previous node has been processed and a new one is requested,
	// This means that the previous data need to be written on disk.
	if ( _isBufferLoaded )
	{
		saveNodeandBrick_buffered();
	}

	// Update current node position in buffer
	_nodeBufferPos[ 0 ] = pNodePos[ 0 ];
	_nodeBufferPos[ 1 ] = pNodePos[ 1 ];
	_nodeBufferPos[ 2 ] = pNodePos[ 2 ];

	// Read node info (address+brick index) in node buffer

	_nodeBuffer =_nodeData[ _nodeBufferPos[0] + _nodeGridSize*(_nodeBufferPos[1] + _nodeGridSize*_nodeBufferPos[2]) ];

	// Iterate through data channels
	for ( unsigned int c = 0; c < _dataTypes.size(); ++c )
	{
		// If node is empty, set 0 in buffer of brick data
		if ( isEmpty( _nodeBuffer ) )
		{
			memset( _brickBuffers[ c ], 0, _brickSize * GsDataTypeHandler::canalByteSize( _dataTypes[ c ] ) );
		}
		else
		{
			// Retrieve the brick offset in brick file
			size_t brickOffset = getBrickOffset( _nodeBuffer );

			// Read brick data and store it in buffer
			unsigned short * casted_brickBuffer = (unsigned short *) _brickBuffers[ c ];
			for (size_t i = 0; i < _brickSize; ++i) {
				casted_brickBuffer[i] = _brickData[ brickOffset * _brickSize + i ];
			}
		}
	}

	// Update the flag telling whether or not the current node and brick have been loaded in memory (and stored in buffers)
	_isBufferLoaded = true;
}

/******************************************************************************
 * Save node info and brick data associated to current node position on disk.
 ******************************************************************************/
void GsDataStructureIOHandler::saveNodeandBrick_buffered()
{
	// f
	// Check the flag telling whether or not the current node and brick have been loaded in memory (and stored in buffers)
	if ( _isBufferLoaded ) {
		// Write current node info (address+brick index)
		_nodeData[ _nodeBufferPos[0] + _nodeGridSize*(_nodeBufferPos[1] + _nodeGridSize*_nodeBufferPos[2]) ] = _nodeBuffer;

		// Retrieve the brick offset in brick file
		size_t brickOffset = getBrickOffset( _nodeBuffer );
		// Iterate through data channels (we only have 1 channel in our case)
		unsigned short * casted_brickBuffer = (unsigned short *) _brickBuffers[ 0 ];
		for ( size_t c = 0; c < _brickSize; ++c ) {
			// Write current brick data
			_brickData[ (size_t)brickOffset * (size_t)_brickSize + c ] = casted_brickBuffer[c];
		}
	}

	// Update the flag telling whether or not the current node and brick have been loaded in memory (and stored in buffers)
	_isBufferLoaded = false;
}

/******************************************************************************
 * Fill all brick borders of the data structure with data.
 ******************************************************************************/
void GsDataStructureIOHandler::computeBorders()
{
	// LOG message
	std::cout << "GsDataStructureIOHandler::computeBorders()" << std::endl;
	
	// Iterate trough data ytpes
	// On en a toujours qu'un seul donc GigaOsef
	for ( size_t c = 0; c < _dataTypes.size(); ++c )
	{
		//Instead of using buffers we'll deal with pointers to the brick data to avoid copying memory
		size_t brick = 0;
		size_t brick2 = 0;

		// Iterate trough nodes of the data structure
		for ( size_t k = 0; k < _nodeGridSize; ++k ){
			std::cout << "k = " << k << " / " << _nodeGridSize << std::endl;
			for ( size_t j = 0; j < _nodeGridSize; ++j ){
				for ( size_t i = 0; i < _nodeGridSize; ++i ){
					// Store current node position
					size_t nodePos[ 3 ];
					nodePos[ 0 ] = i;
					nodePos[ 1 ] = j;
					nodePos[ 2 ] = k;

					// Retrieve the associated node info
					unsigned int node = _nodeData[ nodePos[0] + _nodeGridSize*(nodePos[1] + _nodeGridSize*nodePos[2]) ];

					// If node is empty, there is nothing to do, so go to next node
					if ( isEmpty( node ) )
					{
						// TODO c'est faux, il faut v�rifier que le bord du noeud voisin est aussi vide 
						// (sinon il faut remplir la bordure des noeuds vides et leur associer une brique)
						continue; 
					}

					// Retrieve the brick data of the current node (here brick is just a pointer to the data inside the brickData array)
					brick = (size_t)getBrickOffset( node ) * (size_t)_brickSize;

					// Iterate through each neighbor nodes (in 3D, there are 26 (c'est 6 non ???) neighbors)
					for ( int k2 =- 1; k2 <= 1; ++k2 )
					for ( int j2 =- 1; j2 <= 1; ++j2 )
					for ( int i2 =- 1; i2 <= 1; ++i2 )
					{
						// Check the current neighbor node position.
						// If it is outside the data structure bounds,
						// there is nothing to do, so go to next neighbor node.
						if ( i + i2 < 0 || i + i2 >= _nodeGridSize ||
							j + j2 < 0 || j + j2 >= _nodeGridSize ||
							k + k2 < 0 || k + k2 >= _nodeGridSize )
						{
							continue;
						}

						// Store current neighbor node position
						size_t nodePos2[ 3 ];
						nodePos2[ 0 ] = i + i2;
						nodePos2[ 1 ] = j + j2;
						nodePos2[ 2 ] = k + k2;

						// Retrieve the associated neighbor node info
						unsigned int node2 = _nodeData[ nodePos2[0] + _nodeGridSize*(nodePos2[1] + _nodeGridSize*nodePos2[2]) ];

						// If node is empty :
						// there is nothing to do, so go to next neighbor node
						if ( isEmpty( node2 ) )
						{
							continue;
						}

						// Retrieve the associated neighbor brick data
						brick2 = (size_t)getBrickOffset( node2 ) * (size_t)_brickSize;

						// In this part, the goal is to copy voxel data from original brick to voxel data
						// of the current neighbor brick if it is on a border.

						// Iterate through voxels of the brick (the first one ?)
						// Pourquoi ça commence pas en 0

						// 2DO : A la place de faire une boucle de golmon sur tous les voxels pourquoi pas dirctement prendre les bordures et les copier entre chaque briques vu que 
						// de toute façon c'est déjà hardcodé donc quel est l'intéret de faire la boucle

						// C'est effectivement plus compliqué que ce que je pensais
						// Je vais faire des gros IF pour chaques cas possible 
						// C'est horible mais je jure je repasse dessus un jour

						if ((i2 == -1) && (j2 == -1) && (k2 == -1)) {
							size_t x = 1;
							size_t y = 1;
							size_t z = 1;
							size_t x2 = (-i2) * (_brickWidth)+x;
							size_t y2 = (-j2) * (_brickWidth)+y;
							size_t z2 = (-k2) * (_brickWidth)+z;
							_brickData[brick2 + x2 + (_brickWidth + 2) * (y2 + (_brickWidth + 2) * z2)] =
								_brickData[brick + x + (_brickWidth + 2) * (y + (_brickWidth + 2) * z)];
						} else if ((i2 == 1) && (j2 == 1) && (k2 == 1)) {
							size_t x = _brickWidth;
							size_t y = _brickWidth;
							size_t z = _brickWidth;
							size_t x2 = (-i2) * (_brickWidth)+x;
							size_t y2 = (-j2) * (_brickWidth)+y;
							size_t z2 = (-k2) * (_brickWidth)+z;
							_brickData[brick2 + x2 + (_brickWidth + 2) * (y2 + (_brickWidth + 2) * z2)] =
								_brickData[brick + x + (_brickWidth + 2) * (y + (_brickWidth + 2) * z)];
						}
						else if ((i2 == -1) && (j2 == 1) && (k2 == 1)) {
							size_t x = 1;
							size_t y = _brickWidth;
							size_t z = _brickWidth;
							size_t x2 = (-i2) * (_brickWidth)+x;
							size_t y2 = (-j2) * (_brickWidth)+y;
							size_t z2 = (-k2) * (_brickWidth)+z;
							_brickData[brick2 + x2 + (_brickWidth + 2) * (y2 + (_brickWidth + 2) * z2)] =
								_brickData[brick + x + (_brickWidth + 2) * (y + (_brickWidth + 2) * z)];
						}
						else if ((i2 == -1) && (j2 == -1) && (k2 == 1)) {
							size_t x = 1;
							size_t y = 1;
							size_t z = _brickWidth;
							size_t x2 = (-i2) * (_brickWidth)+x;
							size_t y2 = (-j2) * (_brickWidth)+y;
							size_t z2 = (-k2) * (_brickWidth)+z;
							_brickData[brick2 + x2 + (_brickWidth + 2) * (y2 + (_brickWidth + 2) * z2)] =
								_brickData[brick + x + (_brickWidth + 2) * (y + (_brickWidth + 2) * z)];
						}
						else if ((i2 == 1) && (j2 == -1) && (k2 == 1)) {
							size_t x = _brickWidth;
							size_t y = 1;
							size_t z = _brickWidth;
							size_t x2 = (-i2) * (_brickWidth)+x;
							size_t y2 = (-j2) * (_brickWidth)+y;
							size_t z2 = (-k2) * (_brickWidth)+z;
							_brickData[brick2 + x2 + (_brickWidth + 2) * (y2 + (_brickWidth + 2) * z2)] =
								_brickData[brick + x + (_brickWidth + 2) * (y + (_brickWidth + 2) * z)];
						}
						else if ((i2 == 1) && (j2 == 1) && (k2 == -1)) {
							size_t x = _brickWidth;
							size_t y = _brickWidth;
							size_t z = 1;
							size_t x2 = (-i2) * (_brickWidth)+x;
							size_t y2 = (-j2) * (_brickWidth)+y;
							size_t z2 = (-k2) * (_brickWidth)+z;
							_brickData[brick2 + x2 + (_brickWidth + 2) * (y2 + (_brickWidth + 2) * z2)] =
								_brickData[brick + x + (_brickWidth + 2) * (y + (_brickWidth + 2) * z)];
						}
						else if ((i2 == 1) && (j2 == -1) && (k2 == -1)) {
							size_t x = _brickWidth;
							size_t y = 1;
							size_t z = 1;
							size_t x2 = (-i2) * (_brickWidth)+x;
							size_t y2 = (-j2) * (_brickWidth)+y;
							size_t z2 = (-k2) * (_brickWidth)+z;
							_brickData[brick2 + x2 + (_brickWidth + 2) * (y2 + (_brickWidth + 2) * z2)] =
								_brickData[brick + x + (_brickWidth + 2) * (y + (_brickWidth + 2) * z)];
						}
						else if ((i2 == -1) && (j2 == 1) && (k2 == -1)) {
							size_t x = 1;
							size_t y = _brickWidth;
							size_t z = 1;
							size_t x2 = (-i2) * (_brickWidth)+x;
							size_t y2 = (-j2) * (_brickWidth)+y;
							size_t z2 = (-k2) * (_brickWidth)+z;
							_brickData[brick2 + x2 + (_brickWidth + 2) * (y2 + (_brickWidth + 2) * z2)] =
								_brickData[brick + x + (_brickWidth + 2) * (y + (_brickWidth + 2) * z)];
						}
						else if ((i2 == -1) && (j2 == 0) && (k2 == 0)) {
							size_t x = 1;
							size_t x2 = (-i2) * (_brickWidth)+x;
							for (size_t z = 1; z < _brickWidth + 1; ++z) {
								for (size_t y = 1; y < _brickWidth + 1; ++y) {
									size_t y2 = (-j2) * (_brickWidth)+y;
									size_t z2 = (-k2) * (_brickWidth)+z;
									_brickData[brick2 + x2 + (_brickWidth + 2) * (y2 + (_brickWidth + 2) * z2)] =
										_brickData[brick + x + (_brickWidth + 2) * (y + (_brickWidth + 2) * z)];
								}
							}
						}
						else if ((i2 == 0) && (j2 == -1) && (k2 == 0)) {
							size_t y = 1;
							size_t y2 = (-j2) * (_brickWidth)+y;
							for (size_t z = 1; z < _brickWidth + 1; ++z) {
								for (size_t x = 1; x < _brickWidth + 1; ++x) {
									size_t x2 = (-i2) * (_brickWidth)+x;
									size_t z2 = (-k2) * (_brickWidth)+z;
									_brickData[brick2 + x2 + (_brickWidth + 2) * (y2 + (_brickWidth + 2) * z2)] =
										_brickData[brick + x + (_brickWidth + 2) * (y + (_brickWidth + 2) * z)];
								}
							}
						}
						else if ((i2 == 0) && (j2 == 0) && (k2 == -1)) {
							size_t z = 1;
							size_t z2 = (-k2) * (_brickWidth)+z;
							for (size_t y = 1; y < _brickWidth + 1; ++y) {
								for (size_t x = 1; x < _brickWidth + 1; ++x) {
									size_t x2 = (-i2) * (_brickWidth)+x;
									size_t y2 = (-j2) * (_brickWidth)+y;
									_brickData[brick2 + x2 + (_brickWidth + 2) * (y2 + (_brickWidth + 2) * z2)] =
										_brickData[brick + x + (_brickWidth + 2) * (y + (_brickWidth + 2) * z)];
								}
							}
						}
						else if ((i2 == 1) && (j2 == 0) && (k2 == 0)) {
							size_t x = _brickWidth;
							size_t x2 = (-i2) * (_brickWidth)+x;
							for (size_t z = 1; z < _brickWidth + 1; ++z) {
								for (size_t y = 1; y < _brickWidth + 1; ++y) {
									size_t y2 = (-j2) * (_brickWidth)+y;
									size_t z2 = (-k2) * (_brickWidth)+z;
									_brickData[brick2 + x2 + (_brickWidth + 2) * (y2 + (_brickWidth + 2) * z2)] =
										_brickData[brick + x + (_brickWidth + 2) * (y + (_brickWidth + 2) * z)];
								}
							}
						}
						else if ((i2 == 0) && (j2 == 1) && (k2 == 0)) {
							size_t y = _brickWidth;
							size_t y2 = (-j2) * (_brickWidth)+y;
							for (size_t z = 1; z < _brickWidth + 1; ++z) {
								for (size_t x = 1; x < _brickWidth + 1; ++x) {
									size_t x2 = (-i2) * (_brickWidth)+x;
									size_t z2 = (-k2) * (_brickWidth)+z;
									_brickData[brick2 + x2 + (_brickWidth + 2) * (y2 + (_brickWidth + 2) * z2)] =
										_brickData[brick + x + (_brickWidth + 2) * (y + (_brickWidth + 2) * z)];
								}
							}
						}
						else if ((i2 == 0) && (j2 == 0) && (k2 == 1)) {
							size_t z = _brickWidth;
							size_t z2 = (-k2) * (_brickWidth)+z;
							for (size_t y = 1; y < _brickWidth + 1; ++y) {
								for (size_t x = 1; x < _brickWidth + 1; ++x) {
									size_t x2 = (-i2) * (_brickWidth)+x;
									size_t y2 = (-j2) * (_brickWidth)+y;
									_brickData[brick2 + x2 + (_brickWidth + 2) * (y2 + (_brickWidth + 2) * z2)] =
										_brickData[brick + x + (_brickWidth + 2) * (y + (_brickWidth + 2) * z)];
								}
							}
						}
						else if ((i2 == -1) && (j2 == -1) && (k2 == 0)) {
							size_t x = 1;
							size_t y = 1;
							size_t x2 = (-i2) * (_brickWidth)+x;
							size_t y2 = (-j2) * (_brickWidth)+y;
							for (size_t z = 1; z < _brickWidth + 1; ++z) {
								size_t z2 = (-k2) * (_brickWidth)+z;
								_brickData[brick2 + x2 + (_brickWidth + 2) * (y2 + (_brickWidth + 2) * z2)] =
									_brickData[brick + x + (_brickWidth + 2) * (y + (_brickWidth + 2) * z)];
							}
						}
						else if ((i2 == -1) && (j2 == 0) && (k2 == -1)) {
							size_t x = 1;
							size_t z = 1;
							size_t x2 = (-i2) * (_brickWidth)+x;
							size_t z2 = (-k2) * (_brickWidth)+z;
							for (size_t y = 1; y < _brickWidth + 1; ++y) {
								size_t y2 = (-j2) * (_brickWidth)+y;
								_brickData[brick2 + x2 + (_brickWidth + 2) * (y2 + (_brickWidth + 2) * z2)] =
									_brickData[brick + x + (_brickWidth + 2) * (y + (_brickWidth + 2) * z)];
							}
						}
						else if ((i2 == 0) && (j2 == -1) && (k2 == -1)) {
							size_t y = 1;
							size_t z = 1;
							size_t y2 = (-j2) * (_brickWidth)+y;
							size_t z2 = (-k2) * (_brickWidth)+z;
							for (size_t x = 1; x < _brickWidth + 1; ++x) {
								size_t x2 = (-i2) * (_brickWidth)+x;
								_brickData[brick2 + x2 + (_brickWidth + 2) * (y2 + (_brickWidth + 2) * z2)] =
									_brickData[brick + x + (_brickWidth + 2) * (y + (_brickWidth + 2) * z)];
							}
						}
						else if ((i2 == 1) && (j2 == 1) && (k2 == 0)) {
							size_t x = _brickWidth;
							size_t y = _brickWidth;
							size_t x2 = (-i2) * (_brickWidth)+x;
							size_t y2 = (-j2) * (_brickWidth)+y;
							for (size_t z = 1; z < _brickWidth + 1; ++z) {
								size_t z2 = (-k2) * (_brickWidth)+z;
								_brickData[brick2 + x2 + (_brickWidth + 2) * (y2 + (_brickWidth + 2) * z2)] =
									_brickData[brick + x + (_brickWidth + 2) * (y + (_brickWidth + 2) * z)];
							}
						}
						else if ((i2 == 1) && (j2 == 0) && (k2 == 1)) {
							size_t x = _brickWidth;
							size_t z = _brickWidth;
							size_t x2 = (-i2) * (_brickWidth)+x;
							size_t z2 = (-k2) * (_brickWidth)+z;
							for (size_t y = 1; y < _brickWidth + 1; ++y) {
								size_t y2 = (-j2) * (_brickWidth)+y;
								_brickData[brick2 + x2 + (_brickWidth + 2) * (y2 + (_brickWidth + 2) * z2)] =
									_brickData[brick + x + (_brickWidth + 2) * (y + (_brickWidth + 2) * z)];
							}
						}
						else if ((i2 == 0) && (j2 == 1) && (k2 == 1)) {
							size_t y = _brickWidth;
							size_t z = _brickWidth;
							size_t y2 = (-j2) * (_brickWidth)+y;
							size_t z2 = (-k2) * (_brickWidth)+z;
							for (size_t x = 1; x < _brickWidth + 1; ++x) {
								size_t x2 = (-i2) * (_brickWidth)+x;
								_brickData[brick2 + x2 + (_brickWidth + 2) * (y2 + (_brickWidth + 2) * z2)] =
									_brickData[brick + x + (_brickWidth + 2) * (y + (_brickWidth + 2) * z)];
							}
						}
						else if ((i2 == -1) && (j2 == 1) && (k2 == 0)) {
							size_t x = 1;
							size_t y = _brickWidth;
							size_t x2 = (-i2) * (_brickWidth)+x;
							size_t y2 = (-j2) * (_brickWidth)+y;
							for (size_t z = 1; z < _brickWidth + 1; ++z) {
								size_t z2 = (-k2) * (_brickWidth)+z;
								_brickData[brick2 + x2 + (_brickWidth + 2) * (y2 + (_brickWidth + 2) * z2)] =
									_brickData[brick + x + (_brickWidth + 2) * (y + (_brickWidth + 2) * z)];
							}
						}
						else if ((i2 == -1) && (j2 == 0) && (k2 == 1)) {
							size_t x = 1;
							size_t z = _brickWidth;
							size_t x2 = (-i2) * (_brickWidth)+x;
							size_t z2 = (-k2) * (_brickWidth)+z;
							for (size_t y = 1; y < _brickWidth + 1; ++y) {
								size_t y2 = (-j2) * (_brickWidth)+y;
								_brickData[brick2 + x2 + (_brickWidth + 2) * (y2 + (_brickWidth + 2) * z2)] =
									_brickData[brick + x + (_brickWidth + 2) * (y + (_brickWidth + 2) * z)];
							}
						}
						else if ((i2 == 0) && (j2 == -1) && (k2 == 1)) {
							size_t y = 1;
							size_t z = _brickWidth;
							size_t y2 = (-j2) * (_brickWidth)+y;
							size_t z2 = (-k2) * (_brickWidth)+z;
							for (size_t x = 1; x < _brickWidth + 1; ++x) {
								size_t x2 = (-i2) * (_brickWidth)+x;
								_brickData[brick2 + x2 + (_brickWidth + 2) * (y2 + (_brickWidth + 2) * z2)] =
									_brickData[brick + x + (_brickWidth + 2) * (y + (_brickWidth + 2) * z)];
							}
						}
						else if ((i2 == 1) && (j2 == -1) && (k2 == 0)) {
							size_t x = _brickWidth;
							size_t y = 1;
							size_t x2 = (-i2) * (_brickWidth)+x;
							size_t y2 = (-j2) * (_brickWidth)+y;
							for (size_t z = 1; z < _brickWidth + 1; ++z) {
								size_t z2 = (-k2) * (_brickWidth)+z;
								_brickData[brick2 + x2 + (_brickWidth + 2) * (y2 + (_brickWidth + 2) * z2)] =
									_brickData[brick + x + (_brickWidth + 2) * (y + (_brickWidth + 2) * z)];
							}
						}
						else if ((i2 == 1) && (j2 == 0) && (k2 == -1)) {
							size_t x = _brickWidth;
							size_t z = 1;
							size_t x2 = (-i2) * (_brickWidth)+x;
							size_t z2 = (-k2) * (_brickWidth)+z;
							for (size_t y = 1; y < _brickWidth + 1; ++y) {
								size_t y2 = (-j2) * (_brickWidth)+y;
								_brickData[brick2 + x2 + (_brickWidth + 2) * (y2 + (_brickWidth + 2) * z2)] =
									_brickData[brick + x + (_brickWidth + 2) * (y + (_brickWidth + 2) * z)];
							}
						}
						else if ((i2 == 0) && (j2 == 1) && (k2 == -1)) {
							size_t y = _brickWidth;
							size_t z = 1;
							size_t y2 = (-j2) * (_brickWidth)+y;
							size_t z2 = (-k2) * (_brickWidth)+z;
							for (size_t x = 1; x < _brickWidth + 1; ++x) {
								size_t x2 = (-i2) * (_brickWidth)+x;
								_brickData[brick2 + x2 + (_brickWidth + 2) * (y2 + (_brickWidth + 2) * z2)] =
									_brickData[brick + x + (_brickWidth + 2) * (y + (_brickWidth + 2) * z)];
							}
						}
					}
				}
			}
		}
	}
}

/******************************************************************************
 * Initialize all the files that will be generated.
 * 
 * Note : Associated brick buffer(s) will be created/initialized.
 *
 * @param pName name of the data (i.e. sponza, sibenik, dragon, etc...)
 * @param pNewFiles a flag telling whether or not new files are used
 ******************************************************************************/
void GsDataStructureIOHandler::openFiles( const string& pName, bool pNewFiles )
{
	// [ 1 ] - Handle node file - [ 1 ]

	// Retrieve the node file name
	_fileNameNode = getFileNameNode( pName, _level, _brickWidth );

	// Handle case where no "new files" are requested
	if ( ! pNewFiles )
	{
		// Open a file for update both reading and writing. The file must exist
		_nodeFile = fopen( _fileNameNode.data(), "rb+" );

		if ( _nodeFile )
		{
			// Iterate through nodes of the data structure
			unsigned int nodeData;
			for ( size_t node = 0; node < _nodeGridSize * _nodeGridSize * _nodeGridSize ; ++node )
			{
				// Read current node info and update brick number if not empty
				fread( &nodeData, 1, sizeof( unsigned int ), _nodeFile );
				if ( ! isEmpty( nodeData ) )
				{
					// Update brick counter
					_brickNumber++;
				}
				_nodeData[ node ] = nodeData;
			}
		}
	}

	// Handle case where "new files" are requested or if "_nodeFile" has not been initialized
	if ( ! _nodeFile || pNewFiles )
	{
		// Create an empty file for both reading and writing.
		// If a file with the same name already exists its content is erased and the file is treated as a new empty file.
		_nodeFile = fopen( _fileNameNode.data(), "wb+" );
		
		// Iterate through nodes of the data structure
		//for ( unsigned int node = 0; node < _nodeGridSize * _nodeGridSize * _nodeGridSize; ++node )
		//{
			// Fill node file with the "empty node flag"
		//	fwrite( &_cEmptyNodeFlag, 1, sizeof( unsigned int ), _nodeFile );
		//}
		//fflush( _nodeFile );
	}

	// [ 2 ] - Handle brick file(s) - [ 2 ]

	// Iterate through data channels (i.e. data types)
	FILE *brickFile = NULL;
	if ( ! pNewFiles )
	{
		// Open a file for update both reading and writing. The file must exist.
		free(_brickData);
		_brickData = (unsigned short *) calloc( (size_t)_brickNumber * (size_t)_brickSize, sizeof( unsigned short ) );
		brickFile = fopen( getFileNameBrick( pName, _level, _brickWidth, 0, GsDataTypeHandler::getTypeName( _dataTypes[ 0 ] ) ).data(), "rb+" );
		if ( brickFile ) {
			if (_brickData) {
				fread(_brickData, sizeof(unsigned short), _brickNumber * _brickSize, brickFile);
				fclose(brickFile);
			}
		}
	}
	for ( int c = 0; c < _dataTypes.size(); ++c )
	{
		// Retrieve the brick file name associated to current data channel and store it
		_fileNamesBrick.push_back( getFileNameBrick( pName, _level, _brickWidth, c, GsDataTypeHandler::getTypeName( _dataTypes[ c ] ) ) );

		// Handle case where no "new files" are requested

		// Handle case where "new files" are requested or if "brickFile" has not been initialized
		if ( ! brickFile || pNewFiles )
		{
			// Create an empty file for both reading and writing.
			// If a file with the same name already exists its content is erased and the file is treated as a new empty file.
			brickFile = fopen( _fileNamesBrick[ c ].data(), "wb+" );
		}
		
		// Store the opened brick file handler
		_brickFiles.push_back( brickFile );

		// Create a brick buffer for the current data type and initialize it whith 0
		_brickBuffers.push_back( GsDataTypeHandler::allocateVoxels( _dataTypes[ c ], _brickSize ) );
		memset( _brickBuffers[ c ], 0, _brickSize * GsDataTypeHandler::canalByteSize( _dataTypes[ c ] ) );
	}
}

/******************************************************************************
 * Retrieve the node file name.
 * An example of GigaVoxels node file could be : "fux_BR8_B1_L0.nodes"
 * where "fux" is the name, "8" is the brick width BR, "1" is the brick border size B,
 * "0" is the level of resolution L and "nodes" is the file extension.
 *
 * @param pName name of the data file
 * @param pLevel data structure level of resolution
 * @param pBrickWidth width of bricks
 *
 * @return the node file name in GigaVoxels format.
 ******************************************************************************/
string GsDataStructureIOHandler::getFileNameNode( const std::string& pName, size_t pLevel, size_t pBrickWidth )
{
	std::ostringstream oss;
	
	//oss << pName << "_BR" << pBrickWidth << "_B0_L" << pLevel << ".nodes";
	oss << pName << "_BR" << pBrickWidth << "_B1_L" << pLevel << ".nodes";

	return oss.str();
}

/******************************************************************************
 * Retrieve the brick file name.
 * An example of GigaVoxels brick file could be : "fux_BR8_B1_L0_C0_uchar4.bricks"
 * where "fux" is the name, "8" is the brick width BR, "1" is the brick border size B,
 * "0" is the level of resolution L, "0" is the data channel index C,
 * "uchar4" the data type name and "bricks" is the file extension.
 *
 * @param pName name of the data file
 * @param pLevel data structure level of resolution
 * @param pBrickWidth width of bricks
 * @param pDataChannelIndex data channel index
 * @param pDataTypeName data type name
 *
 * @return the brick file name in GigaVoxels format.
 ******************************************************************************/
string GsDataStructureIOHandler::getFileNameBrick( const string& pName, size_t pLevel, size_t pBrickWidth, unsigned int pDataChannelIndex, const string& pDataTypeName )
{
	std::ostringstream oss;

	//oss << pName << "_BR" << pBrickWidth << "_B0_L" << pLevel << "_C" << pDataChannelIndex << "_" << pDataTypeName << ".bricks";
	oss << pName << "_BR" << pBrickWidth << "_B1_L" << pLevel << "_C" << pDataChannelIndex << "_" << pDataTypeName << ".bricks";

	return oss.str();
}

/******************************************************************************
 * Tell whether or not a node is empty given its node info.
 *
 * @param pNode a node info
 *
 * @return a flag telling whether or not a node is empty
 ******************************************************************************/
bool GsDataStructureIOHandler::isEmpty( unsigned int pNode )
{
	return pNode == _cEmptyNodeFlag;
}

/******************************************************************************
 * Create a brick node info (address + brick index)
 *
 * @param pBrickNumber a brick index
 *
 * @return a brick node info
 ******************************************************************************/
unsigned int createBrickNode( unsigned int pBrickNumber )
{
	return ( pBrickNumber | 0x40000000 );
}

/******************************************************************************
 * Retrieve the brick offset of a brick given a node info.
 *
 * @param pNode a node info
 *
 * @return the brick offset
 ******************************************************************************/
unsigned int GsDataStructureIOHandler::getBrickOffset( unsigned int pNode )
{
	return ( pNode & 0x3fffffff );
}

void GsDataStructureIOHandler::writeFiles(){
	
	//Write node file
	fwrite( _nodeData, sizeof( unsigned int ), (size_t)_nodeGridSize * (size_t)_nodeGridSize * (size_t)_nodeGridSize, _nodeFile );
	fflush( _nodeFile );

	//Write brick file in chunks ?
	for ( size_t i = 0; i < _brickNumber; ++i ) {
		fwrite( _brickData + i * _brickSize, sizeof( unsigned short ), _brickSize, _brickFiles[ 0 ] );
	}
	fflush( _brickFiles[ 0 ] );
}

size_t GsDataStructureIOHandler::getBufferSize() {
	return (this->_trueNbOfValues * 1.5);
}
