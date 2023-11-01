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

/******************************************************************************
 ****************************** KERNEL DEFINITION *****************************
 ******************************************************************************/

/******************************************************************************
 * ...
 *
 * @param pSize number of elements to process
 * @param pInput input array
 * @param pOutput output array
 ******************************************************************************/
__global__
void Kernel_StandardAlgorithm( const unsigned int pSize, const unsigned char* pInput, unsigned char* pOutput )
{
	// Retrieve global data index
	const unsigned int index = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x;

	// Check bound
	if ( index < pSize )
	{
		// Compute value
		const unsigned char value = algorithmFunction( index, pInput );

		// Write value
		pOutput[ index ] = value;
	}
}

/******************************************************************************
 * ...
 *
 * @param pSize number of elements to process
 * @param pInput input array
 * @param pOutput output array
 ******************************************************************************/
__global__
void Kernel_StandardAlgorithm_2( const unsigned int pSize, const unsigned char* pInput, unsigned char* pOutput )
{
	// Retrieve global data index
	const unsigned int index = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x;

	// Check bound
	if ( index < pSize )
	{
		// Compute value
		const unsigned char value = algorithmFunction( index, pInput );

		// Compute value
		const unsigned char value_2 = algorithmFunction_2( index, pInput );

		// Write value
		pOutput[ index ] = value + value_2;
	}
}

/******************************************************************************
 * ...
 *
 * @param pSize number of elements to process
 * @param pInput input array
 * @param pOutput output array
 ******************************************************************************/
__global__
void Kernel_AlgorithmWithFunctionPointer( const unsigned int pSize, const unsigned char* pInput, unsigned char* pOutput )
{
	// Retrieve global data index
	const unsigned int index = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x;

	// Check bound
	if ( index < pSize )
	{
		// Compute value
		const unsigned char value = _d_algorithmFunction( pSize, pInput );

		// Write value
		pOutput[ index ] = value;
	}
}

/******************************************************************************
 * ...
 *
 * @param pSize number of elements to process
 * @param pInput input array
 * @param pOutput output array
 ******************************************************************************/
__global__
void Kernel_AlgorithmWithFunctionPointer_2Functions( const unsigned int pSize, const unsigned char* pInput, unsigned char* pOutput )
{
	// Retrieve global data index
	const unsigned int index = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x;

	// Check bound
	if ( index < pSize )
	{
		// Compute value
		const unsigned char value = _d_algorithmFunction( pSize, pInput );

		// Compute value
		const unsigned char value_2 = _d_algorithmFunction_2( pSize, pInput );

		// Write value
		pOutput[ index ] = value + value_2;
	}
}

/******************************************************************************
 * Algorithm function
 *
 * @param pIndex index of the element to process
 * @param pInput input array
 ******************************************************************************/
__device__
unsigned char algorithmFunction( const unsigned int pIndex, const unsigned char* pInput )
{
	unsigned char value = static_cast< unsigned char >( ( sinf( pInput[ pIndex ] * 3.141592f / 180.f ) * 0.5f + 0.5f ) * 255.f );

	return value;
}

/******************************************************************************
 * Algorithm function
 *
 * @param pIndex index of the element to process
 * @param pInput input array
 ******************************************************************************/
__device__
unsigned char algorithmFunction_2( const unsigned int pIndex, const unsigned char* pInput )
{
	unsigned char value = static_cast< unsigned char >( ( cosf( pInput[ pIndex ] * 3.141592f / 180.f ) * 0.5f + 0.5f ) * 255.f );

	return value;
}
