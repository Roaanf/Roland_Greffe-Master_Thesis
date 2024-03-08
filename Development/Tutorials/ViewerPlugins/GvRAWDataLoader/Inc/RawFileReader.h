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

#ifndef _RAW_FILE_READER_H_
#define _RAW_FILE_READER_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include "GvVoxelizer/GsIRAWFileReader.h"

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

/** 
 * @class RawFileReader
 *
 * @brief The RawFileReader class provides an implementation
 * of a scene voxelizer with ASSIMP, the Open Asset Import Library.
 *
 * It is used to load a 3D scene in memory and traverse it to retrieve
 * useful data needed during voxelization (i.e. vertices, faces, normals,
 * materials, textures, etc...)
 */
template< typename TType >
class RawFileReader : public GvVoxelizer::GsIRAWFileReader
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Constructor
	 */
	RawFileReader();

	/**
	 * Destructor
	 */
	virtual ~RawFileReader();

	/**
	 * Get the min data value
	 *
	 * @return the min data value
	 */
	TType getMinDataValue() const;

	/**
	 * Get the max data value
	 *
	 * @return the max data value
	 */
	TType getMaxDataValue() const;

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Minimum data value
	 */
	TType _minDataValue;

	/**
	 * Maximum data value
	 */
	TType _maxDataValue;

	/******************************** METHODS *********************************/

	/**
	 * Load/import the scene
	 */
	virtual bool readData(const size_t brickWidth, const size_t trueX, const size_t trueY, const size_t trueZ, const unsigned int radius);

	bool optimizedReadData(const size_t brickWidth, const size_t trueX, const size_t trueY, const size_t trueZ, const unsigned int radius);

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Copy constructor forbidden.
	 */
	RawFileReader( const RawFileReader& );

	/**
	 * Copy operator forbidden.
	 */
	RawFileReader& operator=( const RawFileReader& );

};

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "RawFileReader.inl"

/******************************************************************************
 ************************** INSTANTIATION SECTION *****************************
 ******************************************************************************/

/**
 * Instantiates the RawFileReader< xxx > classes.
 */
//template class RawFileReader< char >;
template class RawFileReader< unsigned char >;
//template class RawFileReader< short >;
template class RawFileReader< unsigned short >;
//template class RawFileReader< int >;
//template class RawFileReader< unsigned int >;
template class RawFileReader< float >;
//typedef RawFileReader< char > RawFileReaderc;
typedef RawFileReader< unsigned char > RawFileReaderuc;
//typedef RawFileReader< short > RawFileReaders;
typedef RawFileReader< unsigned short > RawFileReaderus;
//typedef RawFileReader< int > RawFileReaderi;
//typedef RawFileReader< unsigned int > RawFileReaderui;
typedef RawFileReader< float > RawFileReader_f;

#endif
