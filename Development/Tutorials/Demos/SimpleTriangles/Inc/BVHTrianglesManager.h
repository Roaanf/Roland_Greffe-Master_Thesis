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

#ifndef _BVH_TRIANGLES_MANAGER_H_
#define _BVH_TRIANGLES_MANAGER_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include <GvCore/GsArray.h>
#include <GvCore/GsPool.h>
#include <GvCore/GsVectorTypesExt.h>

// Project
#include "GPUTreeBVHCommon.h"
#include "BVHTriangles.hcu"

// STL
#include <string>
#include <vector>
#include <sstream>
#include <iostream>
#include <fstream>

// Assimp
#include <assimp/scene.h>
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>

#include <sys/types.h>
#include <sys/stat.h>
///#include <unistd.h>

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ******************************** CLASS USED **********************************
 ******************************************************************************/

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

using namespace GvCore; // FIXME

/******************************************************************************
 * ...
 ******************************************************************************/
template< class T >
void writeStdVector( const std::string& fileName, std::vector< T >& vec )
{
	std::ofstream dataOut( fileName.c_str(), std::ios::out | std::ios::binary );

	dataOut.write( (const char *)&( vec[ 0 ] ), vec.size() * sizeof( T ) );

	dataOut.close();
}

/******************************************************************************
 * ...
 ******************************************************************************/
template< class T >
void readStdVector( const std::string& fileName, std::vector< T >& vec )
{
	struct stat results;
	if ( stat( fileName.c_str(), &results ) == 0 )
	{
		// results.st_size

		uint numElem = results.st_size / sizeof( T );

		vec.resize( numElem );

		std::ifstream dataIn( fileName.c_str(), std::ios::in | std::ios::binary );
		
		dataIn.read( (char *)&( vec[ 0 ] ), vec.size() * sizeof( T ) );
		
		dataIn.close();
	}
	else
	{
		std::cout << "Unable to open file: " << fileName << "\n";
	}
}

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

/** 
 * @class BVHTrianglesManager
 *
 * @brief The BVHTrianglesManager class provides ...
 *
 * The BVHTrianglesManager class ...
 *
 * @param TDataTypeList data type list choosen by user (ex : (float4, uchar4) for position and color )
 * @param DataPageSize data page size in vertices (size is given by BVH_DATA_PAGE_SIZE)
 */
template< class TDataTypeList, uint DataPageSize >
class BVHTrianglesManager 
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/****************************** INNER TYPES *******************************/

	//typedef unsigned long long PosHasType;
	typedef uint PosHasType;

	/**
	 * Type definition of a custom data pool (array mapped on GPU)
	 */
	typedef GvCore::GPUPoolHost< GvCore::Array3D, TDataTypeList > DataBufferType;

	// Advanced

	/** 
	 * @struct BBoxInfo
	 *
	 * @brief The BBoxInfo struct provides ...
	 *
	 * The BBoxInfo struct ...
	 */
	struct BBoxInfo
	{

		/******************************* ATTRIBUTES *******************************/

		/**
		 * Object ID
		 */
		uint objectID;

		/**
		 * ...
		 */
		PosHasType posHash;

		/**
		 * ...
		 */
		//uchar surfaceHash;
		
		/******************************** METHODS *********************************/

		/**
		 * Constructor
		 */
		BBoxInfo( uint oid, AABB& bbox/*, float2 minMaxSurf*/ );
		
		/**
		 * Constructor
		 */
		BBoxInfo( uint oid, float3 v0, float3 v1, float3 v2 );
		
		/**
		 * ...
		 */
		bool operator<( const BBoxInfo& b ) const;

	};	// end of struct BBoxInfo

	/******************************* ATTRIBUTES *******************************/

	///// MESH /////

	/** 
	 * @struct Triangle
	 *
	 * @brief The Triangle struct provides ...
	 *
	 * The Triangle struct ...
	 */
	struct Triangle
	{
		uint vertexID[ 3 ];
	};

	/**
	 * Buffer of vertex positions
	 */
	std::vector< float3 > meshVertexPositionList;
	
	/**
	 * Buffer of vertex colors
	 */
	std::vector< float4 > meshVertexColorList;

	/**
	 * Buffer of triangles
	 */
	std::vector< Triangle > meshTriangleList;

	// If field added, think of updating splitting and buffer filling.
	
	/**
	 * Node buffer
	 */
	GvCore::Array3D< VolTreeBVHNode >* _nodesBuffer;
	
	/**
	 * Data pool
	 */
	DataBufferType* _dataBuffer;
	
	/**
	 * ...
	 */
	uint numDataNodesCounter;

	/******************************** METHODS *********************************/

	/**
	 * Constructor
	 */
	BVHTrianglesManager();

	/**
	 * Destructor
	 */
	~BVHTrianglesManager();
	
	/**
	 * Helper function to stringify a value
	 */
	inline std::string stringify( int x );

	/**
	 * Iterate through triangles and split them if required (depending on a size criteria)
	 *
	 * @param criticalEdgeLength max "triangle edge length" criteria beyond which a split must occur
	 *
	 * @return flag to tell wheter or not a split has happend
	 */
	bool splitTrianges( float criticalEdgeLength );

	/**
	 * ...
	 */
	std::string getBaseFileName( const std::string& fileName );

	/**
	 * ...
	 */
	void loadPowerPlant( const std::string& baseFileName );

	/**
	 * ...
	 */
	void loadMesh( const std::string& meshFileName );

	/**
	 * ...
	 */
	void saveRawMesh( const std::string& fileName );

	/**
	 * ...
	 */
	void loadRawMesh( const std::string& fileName );

	/**
	 * Generate buffers
	 */
	void generateBuffers( uint arrayFlag );

	// TODO: case where data is linked
	/**
	 * Fill nodes buffer
	 */
	VolTreeBVHNode fillNodesBuffer( std::vector< VolTreeBVHNode >& bvhNodesList, std::vector< BBoxInfo >& bboxInfoList,
									int level, uint2 curInterval, uint& nextNodeBufferOffset,
									std::vector< VolTreeBVHNode >& bvhNodesBufferBuildingList );

	/**
	 * ...
	 */
	void recursiveAddEscapeIdx( uint nodeAddress, uint escapeTo );

	/////////////

	/**
	 * ...
	 */
	void loadPowerPlantDirectoryStructure( const std::string& baseFileName );

	/**
	 * ...
	 */
	void addMeshFile( const std::string& meshFileName );

	/**
	 * ...
	 */
	void renderGL();

	/**
	 * ...
	 */
	void renderDebugGL();

	/**
	 * ...
	 */
	void recursiveRenderDebugGL( uint curNodeIdx );

	/**
	 * ...
	 */
	void displayTriangles( uint index );

	/**
	 * ...
	 */
	void renderFullGL();

	/**
	 * ...
	 */
	void recursiveRenderFullGL( uint curNodeIdx );

	/**
	 * Get the node pool
	 */
	GvCore::Array3D< VolTreeBVHNode >* getNodesBuffer();

	/**
	 * Get the data pool
	 */
	DataBufferType* getDataBuffer();

protected:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

private:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Copy constructor forbidden.
	 */
	BVHTrianglesManager( const BVHTrianglesManager& );

	/**
	 * Copy operator forbidden.
	 */
	BVHTrianglesManager& operator=( const BVHTrianglesManager& );

};

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "BVHTrianglesManager.inl"

#endif
