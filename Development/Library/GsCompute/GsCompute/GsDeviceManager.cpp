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

#include "GsCompute/GsDeviceManager.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaSpace
#include "GsCompute/GsDevice.h"

// CUDA toolkit
#include <cuda_runtime.h>

// System
#include <cassert>
#include <iostream>
#include <cstdio>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

// Gigavoxels
using namespace GsCompute;

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/**
 * The unique device manager
 */
GsDeviceManager* GsDeviceManager::msInstance = NULL;

/**
 * Required compute capability
 */
#define GV_REQUIRED_COMPUTE_CAPABILITY_MAJOR 2
#define GV_REQUIRED_COMPUTE_CAPABILITY_MINOR 0

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ***************************** METHOD DEFINITION ******************************
 ******************************************************************************/

/******************************************************************************
 * Get the device manager.
 *
 * @return the device manager
 ******************************************************************************/
GsDeviceManager& GsDeviceManager::get()
{
	if ( msInstance == NULL )
	{
		msInstance = new GsDeviceManager();
	}
	assert( msInstance != NULL );
	return *msInstance;
}

/******************************************************************************
 * Constructor.
 ******************************************************************************/
GsDeviceManager::GsDeviceManager()
:	_devices()
,	_currentDevice( NULL )
,	_isInitialized( false )
,	_hasCompliantHardware( false )
{
}

/******************************************************************************
 * Destructor.
 ******************************************************************************/
GsDeviceManager::~GsDeviceManager()
{
	finalize();
}

/******************************************************************************
 * Initialize the device manager
 ******************************************************************************/
bool GsDeviceManager::initialize()
{
	// Check the flag to tell wheter or not the device manager is initialized
	if ( _isInitialized )
	{
		return _hasCompliantHardware;
	}
	
	// Retrieve number of devices
	int nbDevices;
	cudaGetDeviceCount( &nbDevices );

	// Iterate through devices
	for ( int i = 0; i < nbDevices; i++ )
	{
		// Iterate device properties
		cudaDeviceProp cudaDeviceProperties;
		cudaGetDeviceProperties( &cudaDeviceProperties, i );

		// TO DO
		// - modify this, cause the value could be different from devices ?
		// - in practice, it should not occur
		// ...
		// Register warp size
		GsDevice::_warpSize = cudaDeviceProperties.warpSize;

		// Create GigaSpace device and fill its properties
		GsDevice* device = new GsDevice();
		if ( device != NULL )
		{
			device->_index = i;

			// Retrieve device property
			GsDeviceProperties& deviceProperties = device->mProperties;
			device->_name = cudaDeviceProperties.name;
			deviceProperties._computeCapabilityMajor = cudaDeviceProperties.major;
			deviceProperties._computeCapabilityMinor = cudaDeviceProperties.minor;
			deviceProperties._warpSize = cudaDeviceProperties.warpSize;
			
			// Store the GigaSpace device
			_devices.push_back( device );

			// TEST the architecture
			// The GigaSpace Engine require devices with at least compute capability
			if ( cudaDeviceProperties.major >= GV_REQUIRED_COMPUTE_CAPABILITY_MAJOR &&
				 cudaDeviceProperties.minor >= GV_REQUIRED_COMPUTE_CAPABILITY_MINOR )
			{
				// Update the flag to tell wheter or not the device manager has found at least
				// one compliant hardware
				_hasCompliantHardware = true;
			}
		}
	}

	// Architecture(s) report
	std::cout << "\nThe GigaVoxels-GigaSpace Engine requires devices with at least compute capability " << GV_REQUIRED_COMPUTE_CAPABILITY_MAJOR << "." << GV_REQUIRED_COMPUTE_CAPABILITY_MINOR << std::endl;
	for ( int i = 0; i < getNbDevices(); i++ )
	{
		getDevice( i )->printInfo();
	}
	if ( _hasCompliantHardware )
	{
		std::cout << "OK" << std::endl;
	}
	else
	{
		// Test failed
		// 
		// TO DO : exit program ?
		std::cout << "ERROR : " << "There is no compliant devices" << std::endl;
	}

	// Update the flag to tell wheter or not the device manager is initialized
	_isInitialized = true;

	return _hasCompliantHardware;
}

/******************************************************************************
 * Finalize the device manager
 ******************************************************************************/
void GsDeviceManager::finalize()
{
	for ( int i = 0; i < _devices.size(); i++ )
	{
		delete _devices[ i ];
		_devices[ i ] = NULL;
	}
	_devices.clear();

	// Update the flag to tell wheter or not the device manager is initialized
	_isInitialized = false;

	// Update the flag to tell wheter or not the device manager has found at least
	// one compliant hardware
	_hasCompliantHardware = false;
}

/******************************************************************************
 * Get the number of devices
 *
 * @return the number of devices
 ******************************************************************************/
size_t GsDeviceManager::getNbDevices() const
{
	return _devices.size();
}

/******************************************************************************
 * Get the device given its index
 *
 * @param the index of the requested device
 *
 * @return the requested device
 ******************************************************************************/
const GsDevice* GsDeviceManager::getDevice( int pIndex ) const
{
	assert( pIndex < _devices.size() );
	return _devices[ pIndex ];
}

/******************************************************************************
 * Get the device given its index
 *
 * @param the index of the requested device
 *
 * @return the requested device
 ******************************************************************************/
GsDevice* GsDeviceManager::editDevice( int pIndex )
{
	assert( pIndex < _devices.size() );
	return _devices[ pIndex ];
}

/******************************************************************************
 * Get the current used device if set
 *
 * @return the current device
 *******************************************************************************/
const GsDevice* GsDeviceManager::getCurrentDevice() const
{
	return _currentDevice;
}

/******************************************************************************
 * Get the current used device if set
 *
 * @return the current device
 ******************************************************************************/
GsDevice* GsDeviceManager::editCurrentDevice()
{
	return _currentDevice;
}

/******************************************************************************
 * Set the current device
 *
 * @param pDevice the device
 ******************************************************************************/
void GsDeviceManager::setCurrentDevice( GsDevice* pDevice )
{
	_currentDevice = pDevice;
}
