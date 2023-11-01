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

#include "GvvDataType.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

// GvViewer
using namespace GvViewerCore;

// STL
using namespace std;

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ***************************** METHOD DEFINITION ******************************
 ******************************************************************************/

/******************************************************************************
 * Constructor
 ******************************************************************************/
GvvDataType::GvvDataType()
:	_typeList()
,	_nameList()
,	_infoList()
{
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
GvvDataType::~GvvDataType()
{
}

/******************************************************************************
 * Get the data type list used to store voxels in the data structure
 *
 * @return the data type list of voxels
 ******************************************************************************/
const vector< string >& GvvDataType::getTypes() const
{
	return _typeList;
}

/******************************************************************************
 * Set the data type list used to store voxels in the data structure
 *
 * @return the data type list of voxels
 ******************************************************************************/
void GvvDataType::setTypes( const vector< string >& pTypeList )
{
	_typeList = pTypeList;
}

/******************************************************************************
 * Get the name of data type list used to store voxels in the data structure
 *
 * @return the names of data type list of voxels
 ******************************************************************************/
const vector< string >& GvvDataType::getNames() const
{
	return _nameList;
}

/******************************************************************************
 * Set the name of data type list used to store voxels in the data structure
 *
 * @param pNameList the name of the data type list of voxels
 ******************************************************************************/
void GvvDataType::setNames( const vector< string >& pNameList )
{
	_nameList = pNameList;
}

/******************************************************************************
 * Get the info of data type list used to store voxels in the data structure
 *
 * @return the info of data type list of voxels
 ******************************************************************************/
const vector< string >& GvvDataType::getInfo() const
{
	return _infoList;
}

/******************************************************************************
 * Set the info of the data type list used to store voxels in the data structure
 *
 * @param pInfoList the info of the data type list of voxels
 ******************************************************************************/
void GvvDataType::setInfo( const vector< string >& pInfoList )
{
	_infoList = pInfoList;
}
