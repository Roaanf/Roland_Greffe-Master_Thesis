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

#ifndef _GVV_PROGRAMMABLE_SHADER_INTERFACE_H_
#define _GVV_PROGRAMMABLE_SHADER_INTERFACE_H_

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
 * @class GvvDeviceInterface
 *
 * @brief The GvvDeviceInterface class provides info on a device.
 *
 * ...
 */
class GVVIEWERCORE_EXPORT GvvProgrammableShaderInterface : public GvvBrowsable
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/******************************* ATTRIBUTES *******************************/

	/**
	 * ...
	 */
	static const char* cTypeName;

	/******************************** METHODS *********************************/

	/**
	 * Constructor
	 */
	GvvProgrammableShaderInterface();

	/**
	 * Destructor
	 */
	virtual ~GvvProgrammableShaderInterface();

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

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/******************************* ATTRIBUTES *******************************/

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
	GvvProgrammableShaderInterface( const GvvProgrammableShaderInterface& );
	
	/**
	 * Copy operator forbidden.
	 */
	GvvProgrammableShaderInterface& operator=( const GvvProgrammableShaderInterface& );

};

} // namespace GvViewerCore

#endif // !_GVV_PROGRAMMABLE_SHADER_INTERFACE_H_
