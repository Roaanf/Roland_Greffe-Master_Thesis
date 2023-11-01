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

// STL
#include <iostream>
#include <string>
#include <memory>

// Cuda
#include <vector_types.h>

// GigaVoxels
#include <GvCore/GsVectorTypesExt.h>
#include <GvCore/GsTypeHelpers.h>
#include <GvVoxelizer/GsDataStructureIOHandler.h>
#include <GvVoxelizer/GsDataStructureMipmapGenerator.h>
#include <GvVoxelizer/GsDataTypeHandler.h>
#include <GvStructure/GsWriter.h>

// Dcmtk
// Force dcmtk to use the new syntax for including standard library
// (iostream instead of iostream.h)
#ifdef WIN32
#else
	#define HAVE_CONFIG_H
#endif
#include <dcmtk/dcmimgle/dcmimage.h>
#include <dcmtk/dcmjpeg/djdecode.h>
#include <dcmtk/dcmimage/diregist.h>
#include <dcmtk/dcmdata/dcfilefo.h>
#ifdef WIN32
#else
	#undef HAVE_CONFIG_H
#endif

// Qt
#include <QFileInfo>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

// Constant
uint brickSize = 8;

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ***************************** METHOD DEFINITION ******************************
 ******************************************************************************/

/******************************************************************************
 * Display a progress in text mode
 *
 * @param message ...
 * @param percent ...
 ******************************************************************************/
void DrawProgressBar( const std::string& message, double percent )
{
	// Length of the console
	const uint len = 80;

	// Erase the entire current line.
	std::cout << "\x1B[2K";

   	// Move to the beginning of the current line.
	std::cout << "\x1B[0E"; 

	// Generate string
	std::string progress;
	for ( uint i = 0; i < len; ++i )
	{
		if ( i < static_cast< uint >( len * percent ) )
		{
			progress += "=";
		}
		else
		{
			progress += " ";
		}
	}
	std::cout << "[" << progress << "] " <<( static_cast< int >( 100 * percent ) ) << "%";
	if ( !message.empty() )
	{
		std::cout << " (" << message << ")";
	}

	// Force display
	if ( percent >= 1.f )
	{
		std::cout << std::endl;
	}
	else
	{
		flush( std::cout );
	}
}

/******************************************************************************
 * Display a progress in text mode
 *
 * @param fileName ...
 *
 * @return ...
 ******************************************************************************/
std::auto_ptr< DicomImage > loadImage( const std::string& fileName )
{
	std::auto_ptr< DicomImage > image( new DicomImage( fileName.c_str() ));

	// Check status information
	if ( image->getStatus() != EIS_Normal )
	{
		std::cerr << "Error: cannot load DICOM image (" << DicomImage::getString( image->getStatus() ) << ")" << std::endl;
	
		// TODO Throw exception
		throw 0;
	}

	// Check data
	const void* pixelData = image->getOutputData( 0/*to use "bits stored in the image"*/ );
	if ( pixelData == NULL )
	{
		// LOG
		std::cerr << "Error: no pixel data in " << fileName << std::endl;
		
		// TODO Throw exception
		throw 0;
	}

	// Get image dimension
	uint3 imageDim;
	imageDim.x = image->getWidth();
	imageDim.y = image->getHeight();
	imageDim.z = image->getFrameCount();
	std::cout << "\timage dimension : " << imageDim.x << "x" << imageDim.y << "x" << imageDim.z << std::endl;

	if ( image->getInterData() != NULL )
	{
		const EP_Representation representation = image->getInterData()->getRepresentation();
		switch ( representation )
		{
			case EPR_Uint8:
				std::cout << "\tdata type : uchar" << std::endl;
				break;

			case EPR_Sint8:
				std::cout << "\tdata type : char" << std::endl;
				break;

			case EPR_Uint16:
				std::cout << "\tdata type : ushort" << std::endl;
				break;

			case EPR_Sint16:
				std::cout << "\tdata type : short" << std::endl;
				break;

			case EPR_Uint32:
				std::cout << "\tdata type : uint" << std::endl;
				break;

			case EPR_Sint32:
				std::cout << "\tdata type : int" << std::endl;
				break;

			default:
				std::cerr << "\tERROR : unknown data type..." << std::endl;
				break;
		}
	}

	// LOG
	double min;
	double max;
	int result = image->getMinMaxValues( min, max );
	if ( result )
	{
		std::cout << "\tmin-max : [ " << min << " ; " << max << " ]" << std::endl;
	}

	return image;
}

/******************************************************************************
 * Load data
 *
 * @param fileName ...
 * @param brickSize ...
 * @param resolution ...
 * @param level ...
 *
 * @return ...
 ******************************************************************************/
GvVoxelizer::GsDataStructureIOHandler* loadData( const std::string& fileName, uint brickSize, unsigned int& resolution, unsigned int& level )
{
	// Start loading
	std::cout << "Load DICOM file:" << std::endl;

	// Open Dicom image
	std::auto_ptr< DicomImage > image;
	image = loadImage( fileName );

	// Get image dimension
	uint3 imageDim;
	imageDim.x = image->getWidth();
	imageDim.y = image->getHeight();
	imageDim.z = image->getFrameCount();

	// Fit the shape in a cube no smaller than a brick and whose sides are a power of two.
	uint dimCube; // number of voxel in each dimension
	dimCube = std::max( imageDim.x, std::max( imageDim.y, imageDim.z ) );
	dimCube = static_cast< uint >( pow( 2, ceil( log2( (float)dimCube ) ) ) );
	dimCube = std::max( brickSize, dimCube );

	// Number of bricks per dimension
	uint3 poolDim = dimCube / make_uint3( brickSize );

	// Most detailed level
	level = static_cast< unsigned int >( log2( static_cast< float >( poolDim.x ) ) );

	// Display info on loading
	std::cout << "GigaVoxels-GigaSpace data structure" << std::endl;
	std::cout << "\tnb levels : " << level << std::endl;
	std::cout << "\tresolution : " << dimCube << "x" << dimCube << "x" << dimCube << std::endl;
	resolution = dimCube;
	DrawProgressBar( "", 0.f );

	// Allocate storage for the data
	GvVoxelizer::GsDataStructureIOHandler* brickPool = new GvVoxelizer::GsDataStructureIOHandler( fileName, level, brickSize, GvVoxelizer::GsDataTypeHandler::gvUSHORT, true );

	// Find the number of bytes by pixel in the image, and thus the size of the type we will need
	uint dataSize = static_cast< uint >( ceil( static_cast< float >( image->getDepth() / 8 ) ) );
	assert( dataSize <= 4 );

	// Clear all filter that may be present on the image
	image->deleteDisplayLUT();
	image->setNoVoiTransformation();

	// Fill the brick array
	// Load each frame as a new depth layer
	uint voxelCoordinates[ 3 ];
	for ( uint z = 0; z < imageDim.z; ++z )
	{
		// Retrieve data at given frame (i.e depth layer)
		const char* pixelMap = static_cast< const char* >( image->getOutputData( 0/*to use "bits stored in the image"*/, z ) );

		// Check data
		if ( pixelMap == NULL )
		{
			throw( 0 ); // TODO
		}

		// Iterate through height
		for ( uint y = 0; y < dimCube; ++y )
		{
			// Iterate through width
			for ( uint x = 0; x < dimCube; ++x )
			{
				// Check DICOM image bounds
				if ( x < imageDim.x && y < imageDim.y )
				{
					// Voxel position
					voxelCoordinates[ 0 ] = x;
					voxelCoordinates[ 1 ] = y;
					voxelCoordinates[ 2 ] = z;
										
					// Write voxel data
					uint pixelIndex = ( x + y * imageDim.x ) * dataSize;
					brickPool->setVoxel( voxelCoordinates, 
							static_cast< const void* >( &pixelMap[ pixelIndex ] ),
						   	0/*index of GigaSpace user data channel*/,
						   	dataSize );
				} 			
			}
		}

		// LOG
		DrawProgressBar( "", static_cast< float >( z ) / static_cast< float >( imageDim.z ) );
	}

	// LOG
	DrawProgressBar( "", 1.0f );

	return brickPool;
}

/******************************************************************************
 * Main entry program
 *
 * @param pArgc number of arguments
 * @param pArgv list of arguments
 *
 * @return exit code
 ******************************************************************************/
int main( int pArgc, char* pArgv[] ) 
{
	// Check command line arguments
	if ( pArgc <= 1 )
	{
		std::cerr << "Error:\nUsage:\nFor several files forming a volume: " << pArgv[ 0 ] << " [dicom_files]" << std::endl;
		std::cerr << "For a single file containing several layer: " << pArgv[ 0 ] << " [dicom_file]" << std::endl;
		
		return EXIT_FAILURE;
	}

	GvVoxelizer::GsDataStructureIOHandler* brickPool;

	//if( pArgc > 2 ) {
	//	// The data are in different files
	//	// Convert arguments to vector of strings
	//	std::vector< std::string > fileNames( &( pArgv[ 1 ] ), &( pArgv[ pArgc ] ) );

	//	// Load data
	//	brickPool = loadData( fileNames );
	//} else {
	//	// All the data are in the same file
	//	brickPool = loadData( std::string( pArgv[ 1 ] ));
	//}

	// LOG
	std::cout << "- [step 1 / 4] - Read data and write voxels..." << std::endl;
	std::string fileName( pArgv[ 1 ] );
	unsigned int resolution, level;
	brickPool = loadData( fileName, brickSize, resolution, level );

	// LOG
	std::cout << "- [step 2 / 4] - Update borders..." << std::endl;
	brickPool->computeBorders();

	// LOG
	std::cout << "- [step 3 / 4] - Mipmap pyramid generation..." << std::endl;
	std::vector< GvVoxelizer::GsDataTypeHandler::VoxelDataType > dataTypes;
	dataTypes.push_back( GvVoxelizer::GsDataTypeHandler::gvUSHORT );
	GvVoxelizer::GsDataStructureMipmapGenerator::generateMipmapPyramid( fileName, resolution, dataTypes );

	// LOG
	std::cout << "- [step 4 / 4] - Generate GigaVoxels/GigaSpace meta-data file..." << std::endl;
	// Use GigaSpace Meta Data file writer
	GvStructure::GsWriter xmlWriter;
	// - filename
	QFileInfo dataFileInfo( fileName.c_str() );
	// - exporter configuration
	xmlWriter.setModelDirectory( dataFileInfo.absolutePath().toLatin1().constData() );
	xmlWriter.setModelName( dataFileInfo.completeBaseName().toLatin1().constData() );
	xmlWriter.setModelMaxResolution( level );
	xmlWriter.setBrickWidth( brickSize );
	xmlWriter.setNbDataChannels( 1 );
	std::vector< std::string > typesNames;
	typesNames.push_back( "ushort" ); // TODO : try to be generic...
	xmlWriter.setDataTypeNames( typesNames );
	if ( !xmlWriter.write() )
	{
		// LOG
		std::cout << "ERROR : enable to write the GigaVoxels/GigaSpace meta-data file." << std::endl;

		// TODO error 
		return false;
	}

	// Free resources
	delete brickPool;

	return EXIT_SUCCESS;
}
