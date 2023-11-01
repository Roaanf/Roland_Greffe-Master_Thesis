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

#ifndef _VOLUME_DATA_CUBE_3D_4_
#define _VOLUME_DATA_CUBE_3D_4_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// Cuda
#include <driver_types.h>
#include <texture_types.h>

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

namespace VolumeData
{

/** 
 * @class vdCube3D4
 *
 * @brief The vdCube3D4 class provides interface to handles 3D data file
 * of format vdCube3D4.
 *
 * Format of xxx.vdCube3D4 files is 3D binary data with data size
 * followed by 4 components float data.
 *
 * Is is used here to load 3D model of a bunny with 4 components
 * float data organized as :
 * - normal (3 components)
 * - distance (1 component)
 */
class vdCube3D4
{
	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Constructor
	 *
	 * Open a 3D data file and store data in a buffer.
	 * Format of xxx.vdCube3D4 is binary data with data size
	 * followed by 4 components float data.
	 *
	 * @param pSize resolution of data (4 components 3D float data) [size is identic on each dimension]
	 */
	vdCube3D4( int pSize );

	/**
	 * Constructor
	 *
	 * Open a 3D data file and store data in a buffer.
	 * Format of xxx.vdCube3D4 is binary data with data size
	 * followed by 4 components float data.
	 *
	 * @param pFilename name of the file
	 */
	vdCube3D4( const std::string& pFilename );

	/**
	 * Destructor
	 */
	~vdCube3D4();

	/**
	 * Initialize
	 *
	 * @return flag to tell wheter or not it succeeded
	 */
	bool initialize();

	/**
	 * Get the size of the data.
	 * Resolution of data (4 components 3D float data) [size is identic on each dimension]
	 *
	 * @return the data size
	 */
	inline int getSize() const;

	/**
	 * Get the buffer of data
	 *
	 * @return the pointer on the data buffer
	 */
	inline float* getData();

	/**
	 * Get the data at given 3D indexed position and component in the buffer of data
	 *
	 * @param pX x component of 3D indexed position of data
	 * @param pY y component of 3D indexed position of data
	 * @param pZ z component of 3D indexed position of data
	 * @param pComponent component of the data (data has 4 components)
	 *
	 * @return a reference on the data
	 */
	inline float& get( int pX, int pY, int pZ, int pComponent );
	
	/**
	 * Save data in vdCube3D4 format on disk.
	 *
	 * Format of xxx.vdCube3D4 is binary data with data size
	 * followed by 4 components float data.
	 *
	 * @param pFilename name of the file to write data
	 */
	void save( const std::string& pFilename );

	/**
	 * Bind the internal data to a specified texture
	 * that can be used to fetch data on device.
	 *
	 * @param pTexRefName name of the texture reference to bind
	 * @param pNormalizedAccess indicates whether texture reads are normalized or not
	 * @param pFilterMode type of texture filter mode
	 * @param pAddressMode type of texture access mode
	 */
	void bindToTextureReference( const void* pTextureReferenceSymbol, const char* pTexRefName, bool pNormalizedAccess, cudaTextureFilterMode pFilterMode, cudaTextureAddressMode pAddressMode );

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Size of the data.
	 * Resolution of data (4 components 3D float data) [size is identic on each dimension]
	 */
	int _size;

	/**
	 * Buffer of data
	 */
	float* _data;

	/**
	 * Data in CUDA memory space
	 */
	cudaArray* _dataArray;

	/**
	 * Channel format descriptor
	 */
	cudaChannelFormatDesc _channelFormatDesc;

	/******************************** METHODS *********************************/

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/
};

}

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "vdCube3D4.inl"

#endif
