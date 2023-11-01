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

#include "GvvDeviceInterface.h"

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

/**
 * Tag name identifying a space profile element
 */
const char* GvvDeviceInterface::cTypeName = "Device";

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ***************************** METHOD DEFINITION ******************************
 ******************************************************************************/

/******************************************************************************
 * Constructor
 ******************************************************************************/
GvvDeviceInterface::GvvDeviceInterface()
:	GvvBrowsable()
,	_deviceName()
,	_computeCapabilityMajor( 0 )
,	_computeCapabilityMinor( 0 )
{
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
GvvDeviceInterface::~GvvDeviceInterface()
{
}

/******************************************************************************
 * Returns the type of this browsable. The type is used for retrieving
 * the context menu or when requested or assigning an icon to the
 * corresponding item
 *
 * @return the type name of this browsable
 ******************************************************************************/
const char* GvvDeviceInterface::getTypeName() const
{
	return cTypeName;
}

/******************************************************************************
 * Gets the name of this browsable
 *
 * @return the name of this browsable
 ******************************************************************************/
const char* GvvDeviceInterface::getName() const
{
	return "Device";
}

/******************************************************************************
 * Get device name
 *
 * @return device name
 ******************************************************************************/
const string& GvvDeviceInterface::getDeviceName() const
{
	return _deviceName;
}

/******************************************************************************
 * Set device name
 *
 * @param pName device name
 ******************************************************************************/
void GvvDeviceInterface::setDeviceName( const string& pName )
{
	_deviceName = pName;
}

/******************************************************************************
 * Get compute capability (major number)
 *
 * @return major compute capability 
 ******************************************************************************/
int GvvDeviceInterface::getComputeCapabilityMajor() const
{
	return _computeCapabilityMajor;
}

/******************************************************************************
 * Set compute capability (major number)
 *
 * @param pValue major compute capability 
 ******************************************************************************/
void GvvDeviceInterface::setComputeCapabilityMajor( int pValue )
{
	_computeCapabilityMajor = pValue;
}

/******************************************************************************
 * Get compute capability (minor number)
 *
 * @return minor compute capability 
 ******************************************************************************/
int GvvDeviceInterface::getComputeCapabilityMinor() const
{
	return _computeCapabilityMinor;
}

/******************************************************************************
 * Set compute capability (minor number)
 *
 * @param pValue minor compute capability 
 ******************************************************************************/
void GvvDeviceInterface::setComputeCapabilityMinor( int pValue )
{
	_computeCapabilityMinor = pValue;
}
