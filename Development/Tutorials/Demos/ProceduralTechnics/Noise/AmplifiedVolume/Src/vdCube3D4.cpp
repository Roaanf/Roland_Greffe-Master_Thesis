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

#include "vdCube3D4.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// STL
#include <iostream>
#include <fstream>

// Cuda
#include <cuda_runtime.h>
#include <driver_types.h>
#include <driver_functions.h>
#include <channel_descriptor.h>

// GigaVoxels
#include <GvCore/GsError.h>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

// Project
using namespace VolumeData;

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
 *
 * Open a 3D data file and store data in a buffer.
 * Format of xxx.vdCube3D4 is binary data with data size
 * followed by 4 components float data.
 *
 * @param pSize resolution of data (4 components 3D float data) [size is identic on each dimension]
 ******************************************************************************/
vdCube3D4::vdCube3D4( int pSize )
:	_size( pSize )
,	_data( NULL )
,	_dataArray( NULL )
{
	// Allocate buffer ( 3D data with 4 components )
	_data = new float[ 4 * pSize * pSize * pSize ];
}

/******************************************************************************
 * Constructor
 *
 * Open a 3D data file and store data in a buffer.
 * Format of xxx.vdCube3D4 is binary data with data size
 * followed by 4 components float data.
 *
 * @param pFilename name of the file
 ******************************************************************************/
vdCube3D4::vdCube3D4( const string& pFilename )
:	_size( 0 )
,	_data( NULL )
,	_dataArray( NULL )
{
	// Try to open file in read mode
	ifstream file( pFilename.data(), ios::in | ios::binary );
	if ( ! file )
	{       
		cerr << "Unable to open the file : " << pFilename << endl;	
		return;    
    }

	// Read data size
	file.read( reinterpret_cast< char* >( &_size ), sizeof( int ) );

	// Allocate buffer ( 3D data with 4 components )
	_data = new float[ 4 * _size * _size * _size ];

	// Read all data
	file.read( reinterpret_cast< char* >( _data ), 4 * _size * _size * _size * sizeof( float ) );

	// Close the file
	file.close();	
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
vdCube3D4::~vdCube3D4()
{
	// Free memory
	delete [] _data;

	// Free device memory
	GS_CUDA_SAFE_CALL( cudaFreeArray( _dataArray ) );
}

/******************************************************************************
 * Initialize
 *
 * @return flag to tell wheter or not it succeeded
 ******************************************************************************/
bool vdCube3D4::initialize()
{
	// Create 3D array
	size_t volumeSize = _size * _size * _size;
	cudaExtent volumeExtent = make_cudaExtent( _size, _size, _size );

    _channelFormatDesc = cudaCreateChannelDesc< float4 >();
    GS_CUDA_SAFE_CALL( cudaMalloc3DArray( &_dataArray, &_channelFormatDesc, volumeExtent ) );

    // Copy data to 3D array on device
    cudaMemcpy3DParms copyParams = { 0 };
	copyParams.srcPtr = make_cudaPitchedPtr( static_cast< void* >( getData() ), getSize() * sizeof( float4 ), getSize(), getSize() );
    copyParams.dstArray = _dataArray;
    copyParams.extent = volumeExtent;
    copyParams.kind = cudaMemcpyHostToDevice;
    GS_CUDA_SAFE_CALL( cudaMemcpy3D( &copyParams ) );

	return true;
}

/******************************************************************************
 * Save data in vdCube3D4 format on disk.
 *
 * Format of xxx.vdCube3D4 is binary data with data size
 * followed by 4 components float data.
 *
 * @param pFilename name of the file to write data
 ******************************************************************************/
void vdCube3D4::save( const string& pFilename )
{
	// Try to open file in write mode
	ofstream file( pFilename.data(), ios::out | ios::trunc | ios::binary );
	if ( ! file )
	{
		cerr << "Unable to open the file : " << pFilename << endl;	
		return;    
    }

	// Write data size
	file.write( reinterpret_cast< char* >( &_size ), sizeof( int ) );

	// Write all data
	file.write( reinterpret_cast< char* >( _data ), 4 * _size * _size * _size * sizeof( float ) );
	
	// Close the file
	file.close();
}

/******************************************************************************
 * Bind the internal data to a specified texture
 * that can be used to fetch data on device.
 *
 * @param pTexRefName name of the texture reference to bind
 * @param pNormalizedAccess indicates whether texture reads are normalized or not
 * @param pFilterMode type of texture filter mode
 * @param pAddressMode type of texture access mode
 ******************************************************************************/
void vdCube3D4::bindToTextureReference( const void* pTextureReferenceSymbol, const char* pTexRefName, bool pNormalizedAccess, cudaTextureFilterMode pFilterMode, cudaTextureAddressMode pAddressMode )
{
	std::cout << "bindToTextureReference : " << pTexRefName << std::endl;

	textureReference* texRefPtr;
	GS_CUDA_SAFE_CALL( cudaGetTextureReference( (const textureReference **)&texRefPtr, pTextureReferenceSymbol ) );

	// Set texture parameters
	texRefPtr->normalized = pNormalizedAccess;
	texRefPtr->filterMode = pFilterMode;
	texRefPtr->addressMode[ 0 ] = pAddressMode;
	texRefPtr->addressMode[ 1 ] = pAddressMode;
	texRefPtr->addressMode[ 2 ] = pAddressMode;

	// Bind array to 3D texture
	GS_CUDA_SAFE_CALL( cudaBindTextureToArray( (const textureReference *)texRefPtr, _dataArray, &_channelFormatDesc ) );
}
