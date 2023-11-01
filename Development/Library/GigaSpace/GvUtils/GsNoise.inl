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

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include "GvCore/GsError.h"
#include "GvUtils/GsNoiseKernel.h"

// Cuda
#include <cuda_runtime.h>

/******************************************************************************
 ****************************** INLINE DEFINITION *****************************
 ******************************************************************************/

namespace GvUtils
{

/******************************************************************************
 * Constructor
 ******************************************************************************/
GsNoise::GsNoise()
:	_dataArray( NULL )
,	_gradientCUDAArray( NULL )
{
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
GsNoise::~GsNoise()
{
	// Free device memory
	GS_CUDA_SAFE_CALL( cudaFreeArray( _dataArray ) );
	GS_CUDA_SAFE_CALL( cudaFreeArray( _gradientCUDAArray ) );
}

/******************************************************************************
 * Initialize the noise
 *
 * @return a flag telling wheter or not it succeeds
 ******************************************************************************/
inline bool GsNoise::initialize()
{
	/**
	 * Perlin noise permutation table
	 */
	const uchar permutationTable[] =
	{
		151,160,137, 91, 90, 15,131, 13,201, 95, 96, 53,194,233,  7,225,
		140, 36,103, 30, 69,142,  8, 99, 37,240, 21, 10, 23,190,  6,148,
		247,120,234, 75,  0, 26,197, 62, 94,252,219,203,117, 35, 11, 32,
		 57,177, 33, 88,237,149, 56, 87,174, 20,125,136,171,168, 68,175,
		 74,165, 71,134,139, 48, 27,166, 77,146,158,231, 83,111,229,122,
		 60,211,133,230,220,105, 92, 41, 55, 46,245, 40,244,102,143, 54,
		 65, 25, 63,161,  1,216, 80, 73,209, 76,132,187,208, 89, 18,169,
		200,196,135,130,116,188,159, 86,164,100,109,198,173,186,  3, 64,
		 52,217,226,250,124,123,  5,202, 38,147,118,126,255, 82, 85,212,
		207,206, 59,227, 47, 16, 58, 17,182,189, 28, 42,223,183,170,213,
		119,248,152,  2, 44,154,163, 70,221,153,101,155,167, 43,172,  9,
		129, 22, 39,253, 19, 98,108,110, 79,113,224,232,178,185,112,104,
		218,246, 97,228,251, 34,242,193,238,210,144, 12,191,179,162,241,
		 81, 51,145,235,249, 14,239,107, 49,192,214, 31,181,199,106,157,
		184, 84,204,176,115,121, 50, 45,127,  4,150,254,138,236,205, 93,
		222,114, 67, 29, 24, 72,243,141,128,195, 78, 66,215, 61,156,180,
	};

	{
		// Create texture containing the lookup table.
		const unsigned int nElem = sizeof( permutationTable ) / sizeof( permutationTable[ 0 ] );
		const unsigned int lookupTableNElem = nElem * nElem;

		// Pre-compute table
		uchar4* tmpTable = static_cast< uchar4* >( malloc( lookupTableNElem * sizeof( uchar4 ) ) );

		for ( unsigned int y = 0; y < nElem; ++y )
		{
			unsigned int col = y * nElem;
			for ( unsigned int x = 0; x < nElem; ++x )
			{
				tmpTable[ x + col ].x = permutationTable[ ( permutationTable[ x ] + y ) % 256 ];
				tmpTable[ x + col ].y = permutationTable[ ( permutationTable[ x ] + y + 1 ) % 256 ];
				tmpTable[ x + col ].z = permutationTable[ ( permutationTable[ ( x + 1 ) % 256 ] + y ) % 256 ];
				tmpTable[ x + col ].w = permutationTable[ ( permutationTable[ ( x + 1 ) % 256 ] + y + 1 ) % 256 ];
			}
		}

		// Allocate CUDA array in device memory
		cudaChannelFormatDesc channelFormatDesc = gs_permutationTableTexture.channelDesc;
		GS_CUDA_SAFE_CALL( cudaMallocArray( &_dataArray, &channelFormatDesc, nElem, nElem ) );

		// Copy the lookup table to device memory
		GS_CUDA_SAFE_CALL( cudaMemcpy2DToArrayAsync( _dataArray, 0, 0, tmpTable, nElem * sizeof( uchar4 ), nElem * sizeof( uchar4 ), nElem, cudaMemcpyHostToDevice ));

		// Bind the array to a texture
		
		GS_CUDA_SAFE_CALL( cudaBindTextureToArray( gs_permutationTableTexture, _dataArray ) );
		
		// Change texture properties
		gs_permutationTableTexture.normalized = true;
		gs_permutationTableTexture.filterMode = cudaFilterModePoint;
		gs_permutationTableTexture.addressMode[ 0 ] = cudaAddressModeWrap; // Wrap texture coordinates
		gs_permutationTableTexture.addressMode[ 1 ] = cudaAddressModeWrap;
		gs_permutationTableTexture.addressMode[ 2 ] = cudaAddressModeWrap;
		
		free( tmpTable );
	}

	GV_CHECK_CUDA_ERROR( "GvUtils::GsNoise::initialize" );

	{
		/**
		 * Gradient for the Perlin noise.
		 * The table will be normalized and not denormalized, so 127 = 1.f and -128 = -1.f
		 */
		const char4 gradientTable[] =
		{
			{127,127,0,0}, {-128,127,0,0}, {127,-128,0,0}, {-128,-128,0,0},
			{127,0,127,0}, {-128,0,127,0}, {127,0,-128,0}, {-128,0,-128,0},
			{0,127,127,0}, {0,-128,127,0}, {0,127,-128,0}, {0,-128,-128,0},
			{127,127,0,0}, {0,-128,127,0}, {-128,127,0,0}, {0,-128,-128,0},
		};

		// Create texture containing the gradient
		// Allocate CUDA array in device memory
		const unsigned int gradientTableNElem = sizeof( permutationTable ) / sizeof( permutationTable[ 0 ] );
		cudaChannelFormatDesc channelFormatDesc = gs_gradientTexture.channelDesc;
		GS_CUDA_SAFE_CALL( cudaMallocArray( &_gradientCUDAArray, &channelFormatDesc, gradientTableNElem ) );

		// Create table
		char4* tmpTable = static_cast< char4* >( malloc( gradientTableNElem * sizeof( char4 ) ) );
		for ( unsigned int i = 0; i < gradientTableNElem; ++i )
		{
			tmpTable[ i ] = gradientTable[ permutationTable[ i ] % 16 ];
		}

		// Copy the table to device memory
		cudaMemcpyToArrayAsync( _gradientCUDAArray, 0, 0, tmpTable, gradientTableNElem * sizeof( char4 ), cudaMemcpyHostToDevice );

		gs_gradientTexture.normalized = true;
		gs_gradientTexture.filterMode = cudaFilterModePoint;
		gs_gradientTexture.addressMode[ 0 ] = cudaAddressModeWrap; // Wrap texture coordinates
		gs_gradientTexture.addressMode[ 1 ] = cudaAddressModeWrap;
		gs_gradientTexture.addressMode[ 2 ] = cudaAddressModeWrap;

		GS_CUDA_SAFE_CALL( cudaBindTextureToArray( gs_gradientTexture, _gradientCUDAArray ) );

		free( tmpTable );
	}

	GV_CHECK_CUDA_ERROR( "GvUtils::GsNoise::initialize" );

	return true;
}

} // namespace GvUtils
