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

#ifndef _GVX_RAW_READER_H_
#define _GVX_RAW_READER_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// Project
#include "GvxDataStructureIOHandler.h"
#include "GvxDataTypeHandler.h"

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

namespace Gvx
{

/** 
 * @class GvxRAWReader
 *
 * @brief The GvxRAWReader class provides an implementation
 * of a scene voxelizer with ASSIMP, the Open Asset Import Library.
 *
 * It is used to load a 3D scene in memory and traverse it to retrieve
 * useful data needed during voxelization (i.e. vertices, faces, normals,
 * materials, textures, etc...)
 */
class GvxRAWReader
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
		eASCII,
		eBinary
	};

	/******************************** METHODS *********************************/

	/**
	 * Constructor
	 */
	GvxRAWReader();

	/**
	 * Destructor
	 */
	virtual ~GvxRAWReader();

	/**
	 * Load/import the scene
	 */
	virtual bool read();

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/******************************* ATTRIBUTES *******************************/

	/**
	 * 3D model file path.
	 * Path must be terminated by the specific Operationg System directory seperator (/, \, //, etc...).
	 */
	std::string _filePath;
	
	/**
	 * 3D model file name
	 */
	std::string _fileName;
	
	/**
	 * 3D model file extension
	 */
	std::string _fileExtension;

	/**
	 * Data resolution
	 */
	unsigned int _dataResolution;

	/**
	 * Mode (binary or ascii)
	 */
	Mode _mode;

	/**
	 * File/stream handler.
	 * It ios used to read and/ or write to GigaVoxels files (internal format).
	 */
	GvxDataStructureIOHandler* _dataStructureIOHandler;
	
	/******************************** METHODS *********************************/

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Copy constructor forbidden.
	 */
	GvxRAWReader( const GvxRAWReader& );

	/**
	 * Copy operator forbidden.
	 */
	GvxRAWReader& operator=( const GvxRAWReader& );

};

}

#endif
