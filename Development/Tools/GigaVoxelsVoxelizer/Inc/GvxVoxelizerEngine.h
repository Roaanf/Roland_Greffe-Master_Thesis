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

#ifndef _GVX_VOXELIZER_ENGINE_H_
#define _GVX_VOXELIZER_ENGINE_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// Project
#include "GvxDataStructureIOHandler.h"
#include "GvxDataTypeHandler.h"

// STL
#include <vector>

// CImg
#define cimg_use_magick	// Beware, this definition must be placed before including CImg.h
#include <CImg.h>

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

namespace Gvx
{

/** 
 * @class GvxVoxelizerEngine
 *
 * @brief The GvxVoxelizerEngine class provides a client interface to voxelize a mesh.
 *
 * It is the core class that, given data at triangle (vertices, normals, textures, etc...),
 * generate voxel data in the GigaVoxels data structure (i.e. octree).
 */
class GvxVoxelizerEngine
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Flag to tell wheter or not to handle textures
	 */
	bool _useTexture;

	/******************************** METHODS *********************************/

	/**
	 * Constructor
	 */
	GvxVoxelizerEngine();

	/**
	 * Destructor
	 */
	~GvxVoxelizerEngine();

	/**
	 * Initialize the voxelizer
	 *
	 * Call before voxelization
	 *
	 * @param pLevel Max level of resolution
	 * @param pBrickWidth Width a brick
	 * @param pName Filename to be processed
	 * @param pDataType Data type that will be processed
	 */
	void init( int pLevel, int pBrickWidth, const std::string& pName, GvxDataTypeHandler::VoxelDataType pDataType );
	
	/**
	 * Finalize the voxelizer
	 *
	 * Call after voxelization
	 */
	void end();

	/**
	  * Voxelize a triangle.
	 *
	 * Given vertex attributes previously set for a triangle (positions, normals,
	 * colors and texture coordinates), it voxelizes triangle (by writing data). 
	 */
	void voxelizeTriangle();

	/**
	 * Store a 3D position in the vertex buffer.
	 * During voxelization, each triangle attribute is stored.
	 * Due to kind of circular buffer technique, calling setVertex() method on each vertex
	 * of a triangle, register each position internally.
	 *
	 * @param pX x coordinate
	 * @param pY y coordinate
	 * @param pZ z coordinate
	 */
	void setVertex( float pX, float pY, float pZ );

	/**
	 * Store a nomal in the buffer of normals.
	 * During voxelization, each triangle attribute is stored.
	 * Due to kind of circular buffer technique, calling setNormal() method on each vertex
	 * of a triangle, register each normal internally.
	 *
	 * @param pX x normal component
	 * @param pY y normal component
	 * @param pZ z normal component
	 */
	void setNormal( float pX, float pY, float pZ );

	/**
	 * Store a color in the color buffer.
	 * During voxelization, each triangle attribute is stored.
	 * Due to kind of circular buffer technique, calling setColor() method on each vertex
	 * of a triangle, register each color internally.
	 *
	 * @param pR red color component
	 * @param pG green color component
	 * @param pB blue color component
	 */
	void setColor( float pR, float pG, float pB );

	/**
	 * Store a texture coordinates in the texture coordinates buffer.
	 * During voxelization, each triangle attribute is stored.
	 * Due to kind of circular buffer technique, calling setTexCoord() method on each vertex
	 * of a triangle, register each texture coordinate internally.
	 *
	 * @param pR r texture coordinate
	 * @param pS s texture coordinate
	 */
	void setTexCoord( float pR, float pS );

	/**
	 * Construct image from reading an image file.
	 *
	 * @param pFilename the image filename
	 */
	void setTexture( const std::string& pFilename );

	/**
	 * Set the number of times we apply the filter
	 *
	 * @param pValue the number of filter passes
	 */
	void setNbFilterApplications( int pValue );

	/**
	 * Set the type of the filter 
	 *
	 * @param pValue (0 = mean, 1 = gaussian, 2 = laplacian)
	 */
	void setFilterType( int pValue );

	/**
	 * Set the flag telling whether or not to use normals
	 *
	 * @aparam pFlag a flag telling whether or not to use normals
	 */
	void setNormals( bool pFlag );
	
	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/******************************* ATTRIBUTES *******************************/
	
	/**
	 * List of types that will be used during voxelization (i.e. ucahr4, float, float4, etc...)
	 */
	std::vector< GvxDataTypeHandler::VoxelDataType > _dataTypes;

	/**
	 * File/stream handler.
	 * It ios used to read and/ or write to GigaVoxels files (internal format).
	 */
	GvxDataStructureIOHandler* _dataStructureIOHandler;
	
	/**
	 * Filename to be processed.
	 * Currently, this is just a name like "sponza", not a real path+filename.
	 */
	std::string _fileName;
	
	/**
	 * Brick width
	 */
	int _brickWidth;
	
	/**
	 * Max level of resolution
	 */
	int _level;

	/**
	 * The number of times we apply the filter
	 */
	int _nbFilterApplications;

	/**
	 * the type of the filter 
	 * 0 = mean
	 * 1 = gaussian
	 * 2 = laplacian
	 */
	int _filterType;

	/**
	 * bool that says whether or not we produce normals
	 */
	bool _normals;
	
	// primitives

	/**
	 * Vertices of a triangle
	 */
	float _v1[ 3 ];
	float _v2[ 3 ];
	float _v3[ 3 ];

	/**
	 * Normals of a triangle
	 */
	float _n1[ 3 ];
	float _n2[ 3 ];
	float _n3[ 3 ];

	/**
	 * Colors of a triangle
	 */
	float _c1[ 3 ];
	float _c2[ 3 ];
	float _c3[ 3 ];

	/**
	 * Texture coordinates of a triangle
	 */
	float _t1[ 2 ];
	float _t2[ 2 ];
	float _t3[ 2 ];

	/**
	 * Class representing an image (up to 4 dimensions wide), each pixel being of type T, i.e. "float".
	 * This is the main class of the CImg Library.
	 * It declares and constructs an image, allows access to its pixel values, and is able to perform various image operations.
	 */
	cimg_library::CImg< float > _texture;

	/******************************** METHODS *********************************/

	/**
	 * Apply the update borders algorithmn.
	 * Fill borders with data.
	 */
	void updateBorders();

	/**
	 * Apply the normalize algorithmn
	 */
	void normalize();

	/**
	 * Apply the filtering algorithm
	 */
	void applyFilter();

	/**
	 * Apply the mip-mapping algorithmn.
	 * Given a pre-filtered voxel scene at a given level of resolution,
	 * it generates a mip-map pyramid hierarchy of coarser levels (until 0).
	 */
	void mipmap();

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Copy constructor forbidden.
	 */
	GvxVoxelizerEngine( const GvxVoxelizerEngine& );

	/**
	 * Copy operator forbidden.
	 */
	GvxVoxelizerEngine& operator=( const GvxVoxelizerEngine& );

};

}

#endif
