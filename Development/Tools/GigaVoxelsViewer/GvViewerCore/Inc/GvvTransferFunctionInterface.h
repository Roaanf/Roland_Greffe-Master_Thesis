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

#ifndef GVVTRANSFERFUNCTIONINTERFACE_H
#define GVVTRANSFERFUNCTIONINTERFACE_H

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvCoreConfig.h"
#include "GvvBrowsable.h"

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

namespace GvViewerCore
{

/** 
 * @class GvvTransferFunctionInterface
 *
 * @brief The GvvTransferFunctionInterface class provides...
 *
 * ...
 */
class GVVIEWERCORE_EXPORT GvvTransferFunctionInterface : public GvvBrowsable
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Type name
	 */
	static const char* cTypeName;

	/******************************** METHODS *********************************/

	/**
	 * Constructor
	 */
	GvvTransferFunctionInterface();

	/**
	 * Destructor
	 */
	virtual ~GvvTransferFunctionInterface();

	/**
	 * Returns the type of this browsable. The type is used for retrieving
	 * the context menu or when requested or assigning an icon to the
	 * corresponding item
	 *
	 * @return the type name of this browsable
	 */
	virtual const char* getTypeName() const;

	/**
     * Gets the name of this browsable
     *
     * @return the name of this browsable
     */
	virtual const char* getName() const;

	/**
	 * Initialize
	 *
	 * @return ...
	 */
	virtual bool initialize();

	/**
	 * Finalize
	 *
	 * @return ...
	 */
	virtual bool finalize();

	/**
	 * Get the associated filename
	 *
	 * @return the associated filename
	 */
	const char* getFilename() const;

	/**
	 * Set the associated filename
	 *
	 * @param pFilename the associated filename
	 */
	void setFilename( const char* pFilename );

	/**
	 * Update the associated transfer function
	 *
	 * @param the new transfer function data
	 * @param the size of the transfer function
	 */
	virtual void update( float* pData, unsigned int pSize );

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/******************************* ATTRIBUTES *******************************/

	/**
	 * File name associated to transfer function
	 */
#if defined _MSC_VER
#pragma warning( push )
#pragma warning( disable:4251 )
#endif
	std::string _filename;
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

#endif // !GVVPIPELINEINTERFACE_H
