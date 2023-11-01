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

#include "GvvTransferFunctionInterface.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

// GvViewer
using namespace GvViewerCore;

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/**
 * Tag name identifying a space profile element
 */
const char* GvvTransferFunctionInterface::cTypeName = "TransferFunction";

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ***************************** METHOD DEFINITION ******************************
 ******************************************************************************/

/******************************************************************************
 * Constructor
 ******************************************************************************/
GvvTransferFunctionInterface::GvvTransferFunctionInterface()
:	GvvBrowsable()
,	_filename()
{
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
GvvTransferFunctionInterface::~GvvTransferFunctionInterface()
{
}

/******************************************************************************
 * Returns the type of this browsable. The type is used for retrieving
 * the context menu or when requested or assigning an icon to the
 * corresponding item
 *
 * @return the type name of this browsable
 ******************************************************************************/
const char* GvvTransferFunctionInterface::getTypeName() const
{
	//return "PIPELINE";
	return cTypeName;
}

/******************************************************************************
 * Gets the name of this browsable
 *
 * @return the name of this browsable
 ******************************************************************************/
const char* GvvTransferFunctionInterface::getName() const
{
	return "TransferFunction";
}

/******************************************************************************
 * Initialize
 ******************************************************************************/
bool GvvTransferFunctionInterface::initialize()
{
	return false;
}

/******************************************************************************
 * Initialize
 ******************************************************************************/
bool GvvTransferFunctionInterface::finalize()
{
	return false;
}

/******************************************************************************
 * Get the associated filename
 *
 * @return the associated filename
 ******************************************************************************/
const char* GvvTransferFunctionInterface::getFilename() const
{
	return _filename.c_str();
}

/******************************************************************************
 * Set the associated filename
 *
 * @param pFilename the associated filename
 ******************************************************************************/
void GvvTransferFunctionInterface::setFilename( const char* pFilename )
{
	_filename = std::string( pFilename );
}


/******************************************************************************
 * Update the associated transfer function
 *
 * @param the new transfer function data
 * @param the size of the transfer function
 ******************************************************************************/
void GvvTransferFunctionInterface::update( float* pData, unsigned int pSize )
{
}
