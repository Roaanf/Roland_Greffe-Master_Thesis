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
bool RawFileReader< TType >::readData()
{
	bool result = false;

	// Read data and create the GigaSpace mip-map pyramid of files
	//result = bruteForceReadData();
	result = optimizedReadData();
	
	return result;
}

/******************************************************************************
 * ...
 ******************************************************************************/
template< typename TType >
bool RawFileReader< TType >::bruteForceReadData()
{
	bool result = false;

	std::string dataFilename = getFilename() + ".raw";

	// LOG
	std::cout << "- read file : " << dataFilename << std::endl;

	// Create a file/streamer handler to read/write GigaVoxels data
	const unsigned int levelOfResolution = static_cast< unsigned int >( log( static_cast< float >( getDataResolution() / 8/*<== if 8 voxels by bricks*/ ) ) / log( static_cast< float >( 2 ) ) );
	const unsigned int brickWidth = 8;
	_dataStructureIOHandler = new GvVoxelizer::GsDataStructureIOHandler( getFilename(), levelOfResolution, brickWidth, getDataType(), true );
		
	// Read file
	if ( getMode() == eBinary )
	{
		// Open file
		FILE* file = fopen( dataFilename.c_str(), "rb" );
		if ( file != NULL )
		{
			// TODO
			// - optimizations => reduce IO read/write overheads
			// ---- read file with 1 unique fread()
			// ---- reorganize data in memory to be able to work with bricks and not voxels
			// ---- then try to rely on setBrick() instead of setVoxel()

			// Set voxel data
			TType voxelData;
			unsigned int voxelPosition[ 3 ];
			for ( unsigned int z = 0; z < _dataResolution; z++ )
			{
				for ( unsigned int y = 0; y < _dataResolution; y++ )
				{
					for ( unsigned int x = 0; x < _dataResolution; x++ )
					{
						// Read voxel data
						fread( &voxelData, sizeof( TType ), 1, file );

						// Update min/max values
						if ( voxelData < _minDataValue )
						{
							_minDataValue = voxelData;
						}
						if ( voxelData > _maxDataValue )
						{
							_maxDataValue = voxelData;
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
							_dataStructureIOHandler->setVoxel( voxelPosition, &voxelData, 0 );
						//}
					}
				}
			}
		}
		else
		{
			// LOG
			std::cout << "- Error : unable to open file : " << dataFilename << std::endl;

			assert( false );
		}
	}
	else if ( _mode == eASCII )
	{
		// LOG
		std::cout << "ASCII files are not yet handled... : " << dataFilename << std::endl;

		// TO DO
		// Add ASCII mode
		// ...

		/*FILE* file = fopen( filename.c_str(), "r" );
		if ( file != NULL )
		{
		}
		else
		{*/
			assert( false );
		//}
	}
	else
	{
		// LOG
		std::cout << "Reading RAW files not working... : " << dataFilename << std::endl;
		
		assert( false );
	}

	return result;
}

/******************************************************************************
 * ...
 ******************************************************************************/
template< typename TType >
bool RawFileReader< TType >::optimizedReadData()
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
		const unsigned int nbValues = _dataResolution * _dataResolution * _dataResolution;
		// Hardcoded RN because try to see if it works
		// TODO : find an elegant solution to the problem
		unsigned int trueX = 840;
		unsigned int trueY = 1103;
		unsigned int trueZ = 840;

		const unsigned int trueNbValues = trueX * trueY * trueZ;

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
		file.seekg(0, std::ios_base::end);
		size_t size = file.tellg();
		TType* _data = new TType[ size ];
		file.seekg(0);
		// - read all data
		file.read( reinterpret_cast< char* >( _data ), size );
		// - close the file
		file.close();

		// Write equivalent GigaSpace voxels file
		// - create a file/streamer handler to read/write GigaVoxels data
		const unsigned int brickWidth = 8;
		const unsigned int levelOfResolution = static_cast< unsigned int >( log( static_cast< float >( getDataResolution() / brickWidth ) ) / log( static_cast< float >( 2 ) ) );
		
		_dataStructureIOHandler = new GvVoxelizer::GsDataStructureIOHandler( getFilename(), levelOfResolution, brickWidth, getDataType(), true );
		TType voxelData;
		unsigned int voxelPosition[ 3 ];
		unsigned int index = 0;
		// The issue is probably because of the thing machin truc pfffffff
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
					_dataStructureIOHandler->setVoxel( voxelPosition, &voxelData, 0 );
					//}

					// Update counter
					index++;
				}
			}
		}

		// Free resources
		delete[] _data;
	}
	else if ( _mode == eASCII )
	{
		// LOG
		std::cout << "ASCII files are not yet handled... : " << dataFilename << std::endl;

		// TO DO
		// Add ASCII mode
		// ...

		/*FILE* file = fopen( filename.c_str(), "r" );
		if ( file != NULL )
		{
		}
		else
		{*/
			assert( false );
		//}
	}
	else
	{
		// LOG
		std::cout << "Reading RAW files not working... : " << dataFilename << std::endl;
		
		assert( false );
	}

	return true;
}
