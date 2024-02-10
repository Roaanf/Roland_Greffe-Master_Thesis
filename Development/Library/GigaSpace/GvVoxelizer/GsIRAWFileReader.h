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

#ifndef _GS_I_RAW_FILE_READER_H_
#define _GS_I_RAW_FILE_READER_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include "GvCore/GsCoreConfig.h"
#include "GvVoxelizer/GsDataStructureIOHandler.h"
#include "GvVoxelizer/GsDataTypeHandler.h"

// STL
#include <string>

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

namespace GvVoxelizer
{

/** 
 * @class GsIRAWFileReader
 *
 * @brief The GsIRAWFileReader class provides an implementation
 * of a scene voxelizer with ASSIMP, the Open Asset Import Library.
 *
 * It is used to load a 3D scene in memory and traverse it to retrieve
 * useful data needed during voxelization (i.e. vertices, faces, normals,
 * materials, textures, etc...)
 *
 * TO DO : add support to multi-data type ==> GvDataStructureMipmapGemerator is dependent of type...
 */
class GIGASPACE_EXPORT GsIRAWFileReader
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Enumeration of reading mode
	 */
	enum Mode
	{
		eUndefinedMode,
		eASCII,
		eBinary
	};

	/******************************** METHODS *********************************/

	/**
	 * Constructor
	 */
	GsIRAWFileReader();

	/**
	 * Destructor
	 */
	virtual ~GsIRAWFileReader();

	/**
	 * Load/import the scene
	 */
	virtual bool read(const size_t brickWidth, const size_t trueX, const size_t trueY, const size_t trueZ);

	/**
	 * 3D model file name
	 */
	const std::string& getFilename() const;

	/**
	 * 3D model file name
	 */
	void setFilename( const std::string& pName );

	/**
	 * Data resolution
	 */
	unsigned int getDataResolution() const;

	/**
	 * Data resolution
	 */
	void setDataResolution( unsigned int pValue );

	/**
	 * Mode (binary or ascii)
	 */
	Mode getMode() const;

	/**
	 * Mode (binary or ascii)
	 */
	void setMode( Mode pMode );

	/**
	 * Get the data type
	 *
	 * @return the data type
	 */
	GsDataTypeHandler::VoxelDataType getDataType() const;

	/**
	 * Set the data type
	 *
	 * @param pType the data type
	 */
	void setDataType( GsDataTypeHandler::VoxelDataType pType );

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/******************************* ATTRIBUTES *******************************/

	/**
	 * 3D model file name
	 */
#if defined _MSC_VER
#pragma warning( push )
#pragma warning( disable:4251 )
#endif
	std::string _filename;
#if defined _MSC_VER
#pragma warning( pop )
#endif

	/**
	 * Data resolution
	 */
	unsigned int _dataResolution;

	/**
	 * Mode (binary or ascii)
	 */
	Mode _mode;

	/**
	 * Data type
	 */
	GsDataTypeHandler::VoxelDataType _dataType;

	/**
	 * File/stream handler.
	 * It ios used to read and/ or write to GigaVoxels files (internal format).
	 */
	GsDataStructureIOHandler* _dataStructureIOHandler;
	
	/******************************** METHODS *********************************/

	/**
	 * Load/import the scene
	 */
	virtual bool readData(const size_t brickWidth, const size_t trueX, const size_t trueY, const size_t trueZ) = 0;

	/**
	 * Apply the mip-mapping algorithmn.
	 * Given a pre-filtered voxel scene at a given level of resolution,
	 * it generates a mip-map pyramid hierarchy of coarser levels (until 0).
	 */
	virtual bool generateMipmapPyramid(GsDataStructureIOHandler* up, unsigned int brickSize);

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Copy constructor forbidden.
	 */
	GsIRAWFileReader( const GsIRAWFileReader& );

	/**
	 * Copy operator forbidden.
	 */
	GsIRAWFileReader& operator=( const GsIRAWFileReader& );

};

}

#endif
