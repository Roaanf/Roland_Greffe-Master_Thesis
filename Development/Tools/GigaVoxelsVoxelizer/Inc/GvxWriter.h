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

#ifndef _GVX_WRITER_H_
#define _GVX_WRITER_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// Project
#include "GvxIWriter.h"

// STL
#include <string>
#include <vector>

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
 * @class GvxWriter
 *
 * @brief The GvxWriter class provides an interface to write meta data from a voxelization process.
 *
 * XML file format is used.
 */
class GvxWriter : public GvxIWriter
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
	 */
	GvxWriter();

	/**
	 * Destructor
	 */
	virtual ~GvxWriter();

	/**
	 * Export meta data of a volexization process
	 *
	 * @return a flag telling whether ot not it succeeds
	 */
	virtual bool write();

	/**
	 * Set the model directory
	 *
	 * @param pName model directory
	 */
	void setModelDirectory( const char* pName );

	/**
	 * Set the model name
	 *
	 * @param pName model name
	 */
	void setModelName( const char* pName );

	/**
	 * Set the model max resolution
	 *
	 * @param pValue model max resolution
	 */
	void setModelMaxResolution( unsigned int pValue );

	/**
	 * Set the brick width
	 *
	 * @param pValue brick width
	 */
	void setBrickWidth( unsigned int pValue );

	/**
	 * Set the number of data channels
	 *
	 * @param pValue number of data channels
	 */
	void setNbDataChannels( unsigned int pValue );

	/**
	 * Set the list of data type names
	 *
	 * @param pNames list of data type names
	 */
	void setDataTypeNames( const std::vector< std::string >& pNames );

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Model directory
	 */
	std::string _modelDirectory;

	/**
	 * Model name
	 */
	std::string _modelName;

	/**
	 * Number of level of details
	 */
	unsigned int _nbModelMaxResolution;

	/**
	 * Brick width
	 */
	unsigned int _brickWidth;

	/**
	 * Number of data channels
	 */
	unsigned int _nbDataChannels;

	/**
	 * List of data type names
	 */
	std::vector< std::string > _dataTypeNames;

	/******************************** METHODS *********************************/

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Copy constructor forbidden.
	 */
	GvxWriter( const GvxWriter& );

	/**
	 * Copy operator forbidden.
	 */
	GvxWriter& operator=( const GvxWriter& );

};

}

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

//#include "GvxWriter.inl"

#endif
