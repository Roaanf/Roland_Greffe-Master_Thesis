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
 ********************************* INFORMATION ********************************
 ******************************************************************************/

//-----------------------------------------------------------------------------
// Conversion from OLD to NEW format
//-----------------------------------------------------------------------------
//
// Nodes files extension:
// - OLD : .ngbodi
// - NEW : .nodes
//
// Bricks files extension:
// - OLD : .ngbod
// - NEW : .bricks
//
// Level of details indices:
// - OLD : max level of resolution starts at "max defined resolution"
// - NEW : min level of resolution starts at 0
//
// IMPORTANT : an "Out" directory MUST be in the same directory that the data
//
//-----------------------------------------------------------------------------
// TASKS:
//-----------------------------------------------------------------------------
//
// (1) - Rename NODES files:
// --------- change ".ngbodi" extensions to ".nodes" by reversing LOD number (max resolution becomes lower)
// ------------- ex : glomerolus-mouse_1024x1024x238_8bit_LI_BR32_B1_L5.ngbodi => glomerolus-mouse_1024x1024x238_8bit_LI_BR32_B1_L0.nodes
//
// (2) - Rename BRICKS files:
// --------- change ".ngbod" extensions to ".bricks" by reversing LOD number (max resolution becomes lower)
// --------- ex : glomerolus-mouse_1024x1024x238_8bit_LI_BR32_B1_L5.ngbod => glomerolus-mouse_1024x1024x238_8bit_LI_BR32_B1_L5.bricks
//
// (3) - Add an "Out" directory in the same directory

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/**
 * Helper function used to renames OLD files (.ngbodi and .ngbod) with NEW extensions (.nodes and .bricks)
 */
//#define GS_RENAME_FILES

/**
 * Activate this definition to rename OLD files instead of copying them
 */
//#define GS_USE_RENAME_OLD_FILES

#ifndef _CRT_SECURE_NO_WARNINGS
	#define _CRT_SECURE_NO_WARNINGS
#endif

// Type definition
#ifdef WIN32
	typedef unsigned __int64 uint64;
#else
	typedef unsigned long long uint64;
#endif

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// System
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cassert>

// STL
#include <string>
#include <sstream>
#include <iostream>

// Qt
#include <QFile>
#include <QDir>
#include <QFileInfoList>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/**
 * Number of data channel (i.e. used defined data : color, density, normals, etc...)
 */
const size_t cNbChannels = 1;

/**
 * Max number of levels of resolution
 */
const size_t cNbMipmapLevels = 8;

/**
 * Prefix used to created filename (i.e. data repository)
 */
//const std::string cPrefixDirectory = "./";
//const std::string cPrefixDirectory = "lerma1024_BR8_B1/";
//const std::string cPrefixDirectory = "fux512_BR8_B1/";
//const std::string cPrefixDirectory = "F:/Projects/GigaVoxels/GigaSpace/Data/Voxels/VoxelData/3DCoat/";
const std::string cPrefixDirectory = "F:/Projects/GigaVoxels/GigaSpace/Data/Voxels/glomerolus-mouse/";


/**
 * Name of the output filename
 */
//const std::string cModelName = "out69_BR8_B1";
//const std::string cModelName = "fux_BR8_B1";
const std::string cModelName = "Coral1024_BR8_B1";

/**
 * Global variable for bricks counter
 */
static unsigned int sNbBricks = 0;

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/**
 * Type definition for unsigned char
 */
typedef unsigned char uchar;

/**
 * Type definition for uchar4
 */
struct uchar4
{
	uchar x;
	uchar y;
	uchar z;
	uchar w;
};

/**
 * Type definition for float4
 */
struct float4
{
	float x;
	float y;
	float z;
	float w;
};

/******************************************************************************
 ***************************** METHOD DEFINITION ******************************
 ******************************************************************************/

/******************************************************************************
 * Compare two values.
 * Internally, it checks if their difference is superior to a threshold.
 *
 * @param a a value to compare
 * @param b an other value to compare
 *
 * @return a flag to tell wheter or not their difference if greater than a threshold (component wise)
 ******************************************************************************/
int fcmpf( float4 a, float4 b )
{
	const float eps = 1.0f;
	
	// Compute the difference of the two values
	float4 n;
	n.x = a.x - b.x;
	n.y = a.y - b.y;
	n.z = a.z - b.z;
	n.w = a.w - b.w;

	// Check if the difference if greater than a threshold (component wise)
	if ( fabsf( n.x ) > eps || fabsf( n.y ) > eps ||
		fabsf( n.z ) > eps || fabsf( n.w ) > eps )
	{
		return 1;
	}

	return 0;
}

/******************************************************************************
 * Generate node a filename given its level of resolution
 *
 * @param pLevel level of resolution
 *
 * @return the node filename
 ******************************************************************************/
std::string createNodeFileName( size_t pLevel )
{
	std::stringstream ss;

	ss << cModelName << "_L" << pLevel << ".nodes";

	return ss.str();
}

/******************************************************************************
 * Generate a brick filename given its level of resolution and its data channel
 *
 * @param pLevel level of resolution
 * @param pChannel data channel index (i.e. color, density, normal, etc...)
 *
 * @return the brick filename
 ******************************************************************************/
std::string createBrickFileName( size_t pLevel, size_t pChannel )
{
	std::stringstream ss;

	ss << cModelName << "_L" << pLevel << "_C" << pChannel << "_uchar4.bricks";

	return ss.str();
}

/******************************************************************************
 * Compute the given brick's average value
 *
 * @param pBrickFilePtr bricks file pointer (current level)
 * @param pBrickOffset brick offset to desired brick in the associated brick file
 * @param pAvgValue the returned brick average value
 ******************************************************************************/
void averageValue( FILE* pBrickFilePtr, /*size_t*/uint64 pBrickOffset, float4* pAvgValue )
{
	// Position pointer in brick file at given brick offset
#ifdef WIN32
	if ( _fseeki64( pBrickFilePtr, pBrickOffset , SEEK_SET ) )
#else
	if ( fseeko( pBrickFilePtr, pBrickOffset , SEEK_SET ) )
#endif
	{
		fprintf( stderr, "fatal: cannot seek through brick element\n" );

		exit( EXIT_FAILURE );
	}

	// Read brick data
	//
	// Note : 1000 is here because we have brick of resolution 8x8x8 brick + 1 border at each side
	uchar4 buf[ 1000 ];
	if ( fread( buf, sizeof( uchar4 ), 1000, pBrickFilePtr ) != 1000 )
	{
		fprintf( stderr, "fatal: cannot read brick element\n" );

		exit( EXIT_FAILURE );
	}

	// Iterate through brick data and compute its average value
	//
	// Note : 1000 is here because we have brick of resolution 8x8x8 brick + 1 border at each side
	float4 avg;
	avg.x = 0.f;
	avg.y = 0.f;
	avg.z = 0.f;
	avg.w = 0.f;
	float scale = 1.0f / 1000.0f;
	for ( size_t i = 0; i < 1000; i++ )
	{
		avg.x += static_cast< float >( buf[ i ].x ) * scale;
		avg.y += static_cast< float >( buf[ i ].y ) * scale;
		avg.z += static_cast< float >( buf[ i ].z ) * scale;
		avg.w += static_cast< float >( buf[ i ].w ) * scale;
	}

	// Update (return) the average brick data
	*pAvgValue = avg;
}

/******************************************************************************
 * Retrieve the node info (address)
 *
 * @param pNodeFilePtr nodes file pointer (at current level)
 * @param pLevelSize current level size (.i.e. resolution is number of nodes in each dimension)
 * @param pX current node's x spatial index position
 * @param pY current node's y spatial index position
 * @param pZ current node's z spatial index position
 *
 * @return the node info (address)
 ******************************************************************************/
/*size_t*/unsigned int getIndexValue( FILE* pNodeFilePtr, size_t pLevelSize, size_t pX, size_t pY, size_t pZ )
{
	// Retrieve the node offset in node file
	/*size_t*/uint64 indexpos = ((uint64)pX + (uint64)pLevelSize*((uint64)pY + (uint64)pZ*(uint64)pLevelSize))*(uint64)sizeof(unsigned int);

	// Position pointer in node file at given node offset
#ifdef WIN32
	if ( _fseeki64( pNodeFilePtr, indexpos, SEEK_SET ) )
#else
	if ( fseeko( pNodeFilePtr, indexpos, SEEK_SET ) )
#endif
	{
		fprintf( stderr, "fatal: cannot seek through node element\n" );

		exit( EXIT_FAILURE );
	}

	// Read node info (address)
	/*size_t*/unsigned int indexval = 0;
	if ( fread( &indexval, sizeof( indexval ), 1, pNodeFilePtr ) != 1 )
	{
		fprintf( stderr, "fatal: cannot read node element\n" );

		exit( EXIT_FAILURE );
	}

	return indexval;
}

/******************************************************************************
 * Compute the brick average value inside the associated node (given its spatial index coordinates)
 *
 * @param pNodeFilePtr nodes file pointer (at current level)
 * @param pBrickFilePtr bricks file pointer (current level)
 * @param pLevelSize current level size (.i.e. resolution is number of nodes in each dimension)
 * @param pX current node's x spatial index position
 * @param pY current node's y spatial index position
 * @param pZ current node's z spatial index position
 * @param pAvgValue the returned brick average value
 *
 * @return a flag telling wheter or not it succeeded
 ******************************************************************************/
bool getBrickAverageValue( FILE* pNodeFilePtr, FILE* pBrickFilePtr, size_t pLevelSize, size_t pX, size_t pY, size_t pZ, float4* pAvgValue )
{
	// Read current node info (address + brick index) in node file
	/*size_t*/unsigned int indexval = getIndexValue( pNodeFilePtr, pLevelSize, pX, pY, pZ );

	// Check if node has terminal flag
	if ( indexval & 0x80000000U )
	{
		// Retrieve the brick offset in brick file
		//
		// Note : 1000 is here because we have brick of resolution 8x8x8 brick + 1 border at each side
		/*size_t*/uint64 brickOffset = (uint64)( ( indexval & 0x7FFFFFFF ) * 1000 * sizeof( uchar4 ) );
		
		// Compute the brick average value
		averageValue( pBrickFilePtr, brickOffset, pAvgValue );

		// The node is terminal
		return true;
	}

	// The node is NOT terminal
	return false;
}

/******************************************************************************
 * Traverse each valid node of the current level
 *
 * @param pNodeFilePtr nodes file pointer (current level)
 * @param pBrickFilePtr bricks file pointer (current level)
 * @param pChildNodeFilePtr nodes file pointer (current level)
 * @param pChildBrickFilePtr bricks file pointer (current level)
 * @param pOutNodeFilePtr output file
 * @param pLevel current traversed level
 ******************************************************************************/
void traverseLevel( FILE* pNodeFilePtr, FILE* pBrickFilePtr, FILE* pChildNodeFilePtr, FILE* pChildBrickFilePtr, FILE* pOutNodeFilePtr, size_t pLevel )
{
	// Iterate through nodes at given level of resolution
	size_t levelSize = 1 << pLevel;
	for ( size_t z = 0; z < levelSize; z++ )
	for ( size_t y = 0; y < levelSize; y++ )
	for ( size_t x = 0; x < levelSize; x++ )
	{
		/*size_t*/unsigned int indexValue = 0;

		// Compute the brick average value inside the associated current node.
		//
		// Note : it returns a flag telling wheter or not it succeeded
		float4 avgValue;
		if ( getBrickAverageValue( pNodeFilePtr, pBrickFilePtr, levelSize, x, y, z, &avgValue ) )
		{
			bool hasAlreadySuccedded = false;
			bool hasAlreadyFailed = false;

			// Iterate through child nodes of current node
			//
			// Note : 8 is here because we have an octree data structure
			float4 childAvgValue[ 8 ];
			size_t childId = 0;
			for ( size_t dz = 0; dz < 2; dz++ )
			for ( size_t dy = 0; dy < 2; dy++ )
			for ( size_t dx = 0; dx < 2; dx++ )
			{
				// Compute spatial index coordinates of current child node
				size_t nx = x * 2 + dx;
				size_t ny = y * 2 + dy;
				size_t nz = z * 2 + dz;

				// Compute the child brick average value inside the associated current child node.
				//
				// Note : it returns a flag telling wheter or not it succeeded
				if ( getBrickAverageValue( pChildNodeFilePtr, pChildBrickFilePtr, levelSize * 2, nx, ny, nz, &childAvgValue[ childId ] ) )
				{
					hasAlreadySuccedded = true;
				}
				else
				{
					hasAlreadyFailed = true;
				}

				childId++;
			}

			bool allSuccedded	= hasAlreadySuccedded && !hasAlreadyFailed;
			bool allFailed		= !hasAlreadySuccedded && hasAlreadyFailed;

			bool isTerminal = allSuccedded || allFailed;

			// Special process if all succeeded
			if ( allSuccedded )
			{
				// Iterate through child nodes of current node
				//
				// Note : 8 is here because we have an octree data structure
				for ( size_t i = 0; i < 8; i++ )
				{
					if ( fcmpf( childAvgValue[ i ], avgValue ) )
					{
						isTerminal = false;

						// Exit loop
						break;
					}
				}
			}

			// Convert the old value to the new format
			indexValue = 0x40000000 | ( getIndexValue( pNodeFilePtr, levelSize, x, y, z ) & 0x3FFFFFFF );

			// Check if node is terminal
			if ( isTerminal == true )
			{
				fprintf( stdout, "found a terminal brick. level %zd: brick %zd, %zd, %zd.\n", pLevel, x, y, z );
				
				// Add terminal flag to node
				indexValue |= 0x80000000;

				// Update bricks number
				sNbBricks++;
			}
		}

		// Write the node info in output node file
		fwrite( &indexValue, sizeof( indexValue ), 1, pOutNodeFilePtr );
	}
}

/******************************************************************************
 * Helper function used to renames OLD files (.ngbodi and .ngbod) with NEW extensions (.nodes and .bricks)
 *
 * @return a flag telling whether or not it succeeds
 ******************************************************************************/
bool renameFiles()
{
	// Data directory
	QDir dir( cPrefixDirectory.c_str() );

	// Retrieve Node filenames
	QStringList nodeFilters;
	nodeFilters << "*.ngbodi";
	dir.setNameFilters( nodeFilters );
	QStringList nodeFilenames = dir.entryList( QDir::Files );
	QFileInfoList nodeEntryInfoList = dir.entryInfoList( QDir::Files );

	// Retrieve Brick filenames
	QStringList brickFilters;
	brickFilters << "*.ngbod";
	dir.setNameFilters( brickFilters );
	QStringList brickFilenames = dir.entryList( QDir::Files );
	QFileInfoList brickEntryInfoList = dir.entryInfoList( QDir::Files );

	// Check
	assert( nodeFilenames.size() != 0 );
	assert( brickFilenames.size() != 0 );
	assert( nodeFilenames.size() == brickFilenames.size() );

#ifdef _DEBUG
	// LOG
	// - Node files
	std::cout << "\nNODE Filenames:" << std::endl;
	for ( int i = 0; i < nodeFilenames.size(); i++ )
	{
		std::cout << "\t" << nodeFilenames[ i ].toLatin1().constData() << std::endl;
	}
	// - Brick files
	std::cout << "\nBRICK Filenames:" << std::endl;
	for ( int i = 0; i < brickFilenames.size(); i++ )
	{
		std::cout << "\t" << brickFilenames[ i ].toLatin1().constData() << std::endl;
	}
#endif

	// Rename files
#ifdef _DEBUG
	// LOG
	std::cout << "\nNEW Filenames:" << std::endl;
#endif
	// - Node files
	std::cout << "\nNODE Filenames:" << std::endl;
	for ( int i = 0; i < nodeFilenames.size(); i++ )
	{
		const QFileInfo& nodeFileInfo = nodeEntryInfoList[ i ];
		const QString oldNodeFilename = nodeFileInfo.canonicalFilePath();
		const QString newNodeFilename = nodeFileInfo.canonicalPath() + QDir::separator() + nodeFileInfo.completeBaseName() + ".nodes";
		
		// Rename or copy files with NEW extension
#if GS_USE_RENAME_OLD_FILES
		bool result = QFile::rename( oldNodeFilename, newNodeFilename );
#else
		bool result = QFile::copy( oldNodeFilename, newNodeFilename );
#endif

#ifdef _DEBUG
		// LOG
		std::cout << "\tOLD : " << oldNodeFilename.toLatin1().constData() << std::endl;
		std::cout << "\tNEW : " << newNodeFilename.toLatin1().constData() << std::endl;
		std::cout << "\t\tresult : " << ( result ? "TRUE" : "FALSE" ) << std::endl;
#endif
	}
	// - Brick files
	std::cout << "\nBRICK Filenames:" << std::endl;
	for ( int i = 0; i < brickFilenames.size(); i++ )
	{
		const QFileInfo& brickFileInfo = brickEntryInfoList[ i ];
		const QString oldBrickFilename = brickFileInfo.canonicalFilePath();
		const QString newBrickFilename = brickFileInfo.canonicalPath() + QDir::separator() + brickFileInfo.completeBaseName() + ".bricks";
		
		// Rename or copy files with NEW extension
#if GS_USE_RENAME_OLD_FILES
		bool result = QFile::rename( oldBrickFilename, newBrickFilename );
#else
		bool result = QFile::copy( oldBrickFilename, newBrickFilename );
#endif

#ifdef _DEBUG
		// LOG
		std::cout << "\tOLD : " << oldBrickFilename.toLatin1().constData() << std::endl;
		std::cout << "\tNEW : " << newBrickFilename.toLatin1().constData() << std::endl;
		std::cout << "\t\tresult : " << ( result ? "TRUE" : "FALSE" ) << std::endl;
#endif
	}

	return true;
}

/******************************************************************************
 * Main entry program
 *
 * @param pArgc number of arguments
 * @param pArgv list of arguments
 *
 * @return exit code
 ******************************************************************************/
int main( int pArgc, char** pArgv )
{
	// Helper function used to renames OLD files (.ngbodi and .ngbod) with NEW extensions (.nodes and .bricks)
#if GS_RENAME_FILES
	renameFiles();
#endif

	// Iterate through levels of resolution
	for ( size_t level = 0; level < cNbMipmapLevels - 1; level++ )
	{
		std::string nodeFileName = createNodeFileName( level );

		// Open files
		FILE* nodeFilePtr = fopen( ( cPrefixDirectory + nodeFileName ).c_str(), "rb" );
		FILE* outNodeFilePtr = fopen( ( cPrefixDirectory + "/Out/" + nodeFileName ).c_str(), "wb+" );	// Beware : this directory must exist
		FILE* childNodeFilePtr = fopen( ( cPrefixDirectory + createNodeFileName( level + 1 ) ).c_str(), "rb" );

		// Check opening status
		if ( nodeFilePtr == NULL || childNodeFilePtr == NULL )
		{
			fprintf( stderr, "fatal: cannot open node files (%s)\n", ( cPrefixDirectory + nodeFileName ).c_str() );

			return EXIT_FAILURE;
		}

		// Iterate through data channels (i.e. used defined data : color, density, normals, etc...)
		for ( size_t channel = 0; channel < cNbChannels; channel++ )
		{
			// Open files
			FILE* brickFilePtr = fopen( ( cPrefixDirectory + createBrickFileName( level, channel ) ).c_str(), "rb" );
			FILE* childBrickFilePtr = fopen( ( cPrefixDirectory + createBrickFileName( level + 1, channel ) ).c_str(), "rb" );

			// Check opening status
			if ( brickFilePtr == NULL || childBrickFilePtr == NULL )
			{
				fprintf( stderr, "fatal: cannot open brick files\n" );

				return EXIT_FAILURE;
			}

			// Traverse each valid node of the current level
			traverseLevel( nodeFilePtr, brickFilePtr, childNodeFilePtr, childBrickFilePtr, outNodeFilePtr, level );

			// Close files
			fclose( brickFilePtr );
			fclose( childBrickFilePtr );
		}

		// Close files
		fclose( nodeFilePtr );
		fclose( outNodeFilePtr );
		fclose( childNodeFilePtr );
	}

	// Last level
	std::string nodeFileName = createNodeFileName( cNbMipmapLevels - 1 );

	// Open files
	FILE* nodeFilePtr = fopen( ( cPrefixDirectory + nodeFileName ).c_str(), "rb" );
	FILE* outNodeFilePtr = fopen( ( cPrefixDirectory + "/Out/" + nodeFileName ).c_str(), "wb+" );

	// Check opening status
	if ( nodeFilePtr == NULL )
	{
		fprintf( stderr, "fatal: cannot open node files\n" );

		return EXIT_FAILURE;
	}

	// Iterate through nodes
	size_t levelSize = 1 << ( cNbMipmapLevels - 1 );
	for ( size_t z = 0; z < levelSize; z++ )
	for ( size_t y = 0; y < levelSize; y++ )
	for ( size_t x = 0; x < levelSize; x++ )
	{
		/*size_t*/unsigned int indexValue = 0;
		
		// Retrieve current node info
		/*size_t*/unsigned int tempValue = getIndexValue( nodeFilePtr, levelSize, x, y, z );
		if ( tempValue & 0x80000000 )
		{
			// Convert the old value to the new format
			indexValue = 0x40000000 | ( tempValue & 0x3FFFFFFF );
		}

		// Write the node info in output node file
		fwrite( &indexValue, sizeof( indexValue ), 1, outNodeFilePtr );
	}

	// Close files
	fclose( nodeFilePtr );
	fclose( outNodeFilePtr );

	// Statistics
	fprintf( stdout, "%d bricks have been marked with the terminal flag.\n", sNbBricks );

	return EXIT_SUCCESS;
}
