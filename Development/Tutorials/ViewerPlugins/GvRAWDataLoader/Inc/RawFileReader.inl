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

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include "GvVoxelizer/GsDataStructureIOHandler.h"
#include "GvVoxelizer/GsDataTypeHandler.h"

// System
#include <cassert>
#include <cmath>

// STL
#include <iostream>
#include <fstream>
#include <limits>

/******************************************************************************
 ****************************** INLINE DEFINITION *****************************
 ******************************************************************************/

/******************************************************************************
 * Constructor
 ******************************************************************************/
template< typename TType >
RawFileReader< TType >::RawFileReader()
:	GvVoxelizer::GsIRAWFileReader()
,	_minDataValue( std::numeric_limits< TType >::max() )
,	_maxDataValue( std::numeric_limits< TType >::min() )
{
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
template< typename TType >
RawFileReader< TType >::~RawFileReader()
{
}

/******************************************************************************
 * Get the min data value
 *
 * @return the min data value
 ******************************************************************************/
template< typename TType >
TType RawFileReader< TType >::getMinDataValue() const
{
	return _minDataValue;
}

/******************************************************************************
 * Get the max data value
 *
 * @return the max data value
 ******************************************************************************/
template< typename TType >
TType RawFileReader< TType >::getMaxDataValue() const
{
	return _maxDataValue;
}

/******************************************************************************
 * Load/import the scene the scene
 ******************************************************************************/
template< typename TType >
bool RawFileReader< TType >::readData(const size_t brickWidth, const size_t trueX, const size_t trueY, const size_t trueZ)
{
	bool result = false;

	// Read data and create the GigaSpace mip-map pyramid of files
	//result = bruteForceReadData();
	result = optimizedReadData(brickWidth, trueX, trueY, trueZ);
	
	return result;
}

/******************************************************************************
 * ...
 ******************************************************************************/
template< typename TType >
bool RawFileReader< TType >::optimizedReadData(const size_t brickWidth, const size_t trueX_in, const size_t trueY_in, const size_t trueZ_in)
{
	std::string dataFilename = getFilename() + ".raw";

	// LOG
	std::cout << "- read file : " << dataFilename << std::endl;

	// Read file
	if ( getMode() == eBinary )
	{
		// Allocate a buffer to store all data
		// - BEWARE : only works if _dataResolution is maximum 1024 (2048 will fail due to max "unsigned int" limit)
		// Thats an issue in 32bit but since we are in 64 we should be good to do way more
		const size_t nbValues = (size_t)_dataResolution * (size_t)_dataResolution * (size_t)_dataResolution;
		std::cout << "Nb values : " << nbValues << std::endl;
		// Hardcoded RN because try to see if it works
		// TODO : use the .mhd file that should always be there
		size_t trueX = trueX_in;
		size_t trueY = trueY_in;
		size_t trueZ = trueZ_in;
		std::cout << "True Sizes : " << trueX << " / " << trueY << " / " << trueZ << std::endl;
		//test 

		const size_t trueNbValues = trueX * trueY * trueZ;

		// Read data file
		// - open file
		std::ifstream file;
		file.open( dataFilename.c_str(), std::ios::in | std::ios::binary );

		if ( ! file.is_open() )
		{
			// LOG
			std::cerr << "Unable to open the file : " << dataFilename << std::endl;

			return false;
		}
		// Potential fix for large files not beeing fully read ?
		size_t bufferSize = trueX * trueY * brickWidth;
		TType* bufferBrick = new TType[bufferSize];

		/*
		
		*/

		// Write equivalent GigaSpace voxels file
		// - create a file/streamer handler to read/write GigaVoxels data
		const size_t levelOfResolution = static_cast<size_t>( log( static_cast< float >( getDataResolution() / brickWidth ) ) / log( static_cast< float >( 2 ) ) );
		std::cout << "Level of resolution : " << levelOfResolution << std::endl;
		
		_dataStructureIOHandler = new GvVoxelizer::GsDataStructureIOHandler( getFilename(), levelOfResolution, brickWidth, getDataType(), true, trueNbValues );
		TType voxelData;
		size_t voxelPosition[ 3 ];
		
		/*Old method
		size_t index = 0;
		std::cout << "Entering the loop" << std::endl;
		for ( unsigned int z = 0; z < _dataResolution; z++ ) // Is trueZ an issue here ?
		{
			if (z >= trueZ)
				break;

			for ( unsigned int y = 0; y < _dataResolution; y++ )
			{
				if (y >= trueY) {
					break;
				}

				for ( unsigned int x = 0; x < _dataResolution; x++ )
				{
					// Retrieve data at current position
					// Bandaid fix :(
					if (x >= trueX)
						break;

					voxelData = _data[ index ];
					
					// Update min/max values
					if ( voxelData < _minDataValue )
					{
						_minDataValue = voxelData;
					}
					if ( voxelData > _maxDataValue )
					{
						_maxDataValue = voxelData;
					}

					if (voxelData == 0) {
						index++;
						continue;
					}

					// Threshold management:
					// - it could be better to import data as is,
					//   and rely on the thresholds provided at real-time (shader)
					//   or when reading data (producer).
					//	if ( voxelData >= userThreshold )
					//{
					// Write voxel data (in channel 0)
					voxelPosition[ 0 ] = x;
					voxelPosition[ 1 ] = y;
					voxelPosition[ 2 ] = z;
					
					_dataStructureIOHandler->setVoxel_buffered( voxelPosition, &voxelData, 0 );
					//}

					// Update counter
					index++;
				}
			}
		}
		*/
		std::cout << "Entering the loop" << std::endl;
		// Try to do it brick by brick -> speedup de fou !
		size_t nodeSize = (1 << levelOfResolution);
		/*
			We iterate over the nodes, and for each node we iterate over each voxel of that node
			It goes faster than just iterating over the voxels, because we don't have to change the current noce/brick buffer as often in the GsDataStructureIOHandler
			But we might have read order issues since the raw file in encoded in the order slice, row, column
		*/
		for (size_t node_z = 0; node_z < nodeSize; node_z++) {
			file.read((char*)bufferBrick, bufferSize * sizeof(TType));
			for (size_t node_y = 0; node_y < nodeSize; node_y++) {
				for (size_t node_x = 0; node_x < nodeSize; node_x++) {	
					for (size_t z_brick = 0; z_brick < brickWidth; z_brick++) {
						for (size_t y_brick = 0; y_brick < brickWidth; y_brick++) {
							for (size_t x_brick = 0; x_brick < brickWidth; x_brick++) {
								size_t true_x = node_x * brickWidth + x_brick;
								if (true_x >= trueX) {
									break;
								}
								size_t true_y = node_y * brickWidth + y_brick;
								if (true_y >= trueY) {
									break;
								}
								size_t true_z = node_z * brickWidth + z_brick;
								if (true_z >= trueZ) {
									break;
								}

								size_t index = true_x + true_y * trueX + z_brick * trueX * trueY;

								voxelData = bufferBrick[index];

								// Update min/max values
								if (voxelData < _minDataValue)
								{
									_minDataValue = voxelData;
								}
								if (voxelData > _maxDataValue)
								{
									_maxDataValue = voxelData;
								}

								if (voxelData == 0) {
									continue;
								}

								voxelPosition[0] = true_x;
								voxelPosition[1] = true_y;
								voxelPosition[2] = true_z;

								_dataStructureIOHandler->setVoxel_buffered(voxelPosition, &voxelData, 0);

							}
						}
					}
				}
			}
		}

		/*
		for ( size_t node_z = 0; node_z < nodeSize ; node_z++ ){
			for (size_t node_y = 0; node_y < nodeSize; node_y++) {
				for (size_t node_x = 0; node_x < nodeSize; node_x++) {
					file.read((char*)bufferBrick, bufferSize * sizeof(TType));
					for (size_t z_brick = 0; z_brick < brickWidth; z_brick++) {
						for (size_t y_brick = 0; y_brick < brickWidth; y_brick++) {
							for (size_t x_brick = 0; x_brick < brickWidth; x_brick++) {
								size_t true_x = node_x * brickWidth + x_brick;
								if (true_x >= trueX) {
									continue;
								}
								size_t true_y = node_y * brickWidth + y_brick;
								if (true_y >= trueY) {
									continue;
								}
								size_t true_z = node_z * brickWidth + z_brick;
								if (true_z >= trueZ) {
									continue;
								}

								size_t index = x_brick + y_brick * brickWidth + z_brick * brickWidth * brickWidth;
								
								voxelData = bufferBrick[index];
					
								// Update min/max values
								if ( voxelData < _minDataValue )
								{
									_minDataValue = voxelData;
								}
								if ( voxelData > _maxDataValue )
								{
									_maxDataValue = voxelData;
								}

								if (voxelData == 0) {
									continue;
								}

								voxelPosition[ 0 ] = true_x;
								voxelPosition[ 1 ] = true_y;
								voxelPosition[ 2 ] = true_z;
					
								_dataStructureIOHandler->setVoxel_buffered( voxelPosition, &voxelData, 0 );
							
							}
						}
					}
				}
			}
		}
		*/

		// Free resources
		delete[] bufferBrick;
		file.close();

	}
	else
	{
		// LOG
		std::cout << "Reading RAW files not working... : " << dataFilename << std::endl;
		
		assert( false );
	}

	return true;
}
