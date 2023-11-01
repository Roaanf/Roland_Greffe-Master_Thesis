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

#ifndef _GVV_DATA_TYPE_H_
#define _GVV_DATA_TYPE_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvCoreConfig.h"

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

namespace GvViewerCore
{

/** 
 * @class GvvDataType
 *
 * @brief The GvvDataType class provides info on voxels data types
 * stored in a GigaVoxels data structure.
 *
 * ...
 */
class GVVIEWERCORE_EXPORT GvvDataType
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
	GvvDataType();

	/**
	 * Destructor
	 */
	virtual ~GvvDataType();

	/**
	 * Get the data type list used to store voxels in the data structure
	 *
	 * @return the data type list of voxels
	 */
	const std::vector< std::string >& getTypes() const;

	/**
	 * Set the data type list used to store voxels in the data structure
	 *
	 * @param pTypeList the data type list of voxels
	 */
	void setTypes( const std::vector< std::string >& pTypeList );

	/**
	 * Get the name of data type list used to store voxels in the data structure
	 *
	 * @return the names of data type list of voxels
	 */
	const std::vector< std::string >& getNames() const;

	/**
	 * Set the name of data type list used to store voxels in the data structure
	 *
	 * @param pNameList the name of the data type list of voxels
	 */
	void setNames( const std::vector< std::string >& pNameList );

	/**
	 * Get the info of data type list used to store voxels in the data structure
	 *
	 * @return the info of data type list of voxels
	 */
	const std::vector< std::string >& getInfo() const;

	/**
	 * Set the info of the data type list used to store voxels in the data structure
	 *
	 * @param pInfoList the info of the data type list of voxels
	 */
	void setInfo( const std::vector< std::string >& pInfoList );

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Data type list used to store voxels in the data structure
	 */
#if defined _MSC_VER
#pragma warning( push )
#pragma warning( disable:4251 )
#endif
	std::vector< std::string > _typeList;
#if defined _MSC_VER
#pragma warning( pop )
#endif

	/**
	 * Name of data type list used to store voxels in the data structure
	 */
#if defined _MSC_VER
#pragma warning( push )
#pragma warning( disable:4251 )
#endif
	std::vector< std::string > _nameList;
#if defined _MSC_VER
#pragma warning( pop )
#endif

	/**
	 * Info on data type list used to store voxels in the data structure
	 */
#if defined _MSC_VER
#pragma warning( push )
#pragma warning( disable:4251 )
#endif
	std::vector< std::string > _infoList;
#if defined _MSC_VER
#pragma warning( pop )
#endif

	/******************************** METHODS *********************************/

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

};

} // namespace GvViewerCore

#endif // !_GVV_DATA_TYPE_H_
