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
#include <GvRendering/GsRendererContext.h>

/******************************************************************************
 ****************************** INLINE DEFINITION *****************************
 ******************************************************************************/

/******************************************************************************
 ****************************** KERNEL DEFINITION *****************************
 ******************************************************************************/


/******************************************************************************
 * Map a distance to a color (with a transfer function)
 *
 * @param dist the distance to map
 *
 * @return The color corresponding to the distance
 ******************************************************************************/
__device__
inline float4 densityToColor( float pValue )
{
	return tex1D( transferFunctionTexture, pValue );
}

/******************************************************************************
 * This method is called for each sample. For example, shading or secondary rays
 * should be done here.
 *
 * @param pBrickSampler brick sampler
 * @param pSamplePosScene position of the sample in the scene
 * @param pRayDir ray direction
 * @param pRayStep ray step
 * @param pConeAperture cone aperture
 ******************************************************************************/
template< typename BrickSamplerType >
__device__
inline void ShaderKernel::runImpl( const BrickSamplerType& brickSampler, const float3 samplePosScene,
					const float3 rayDir, float& rayStep, const float coneAperture )
{
	// Sample data
	float4 col = brickSampler.template getValue< 0 >( coneAperture );

	// Threshold data
	if ( col.w < cShaderThreshold )
	{
		//col.w = 0.f;
		
		// Exit
		return;
	}

	// Use transfer function to color density
	// - you can choose where to place the Transfer Function
	// - if you place it here and use ALPHA channel editor, you can stop the ray-marching
	//col = densityToColor( col.w );

	// Apply shading
	// TODO : for scientific visualisation, maybe no need to have special lighting
	if ( col.w > 0.0f )
	{
		// Use transfer function to color density
		col = densityToColor( col.w );

		// Due to alpha pre-multiplication
		//
		// Note : inspecting device SASS code, assembly langage seemed to compute (1.f / color.w) each time, that's why we use a constant and multiply
		const float alphaPremultiplyConstant = 1.f / col.w;
		col.x = col.x * alphaPremultiplyConstant;
		col.y = col.y * alphaPremultiplyConstant;
		col.z = col.z * alphaPremultiplyConstant;
		
		// -- [ Opacity correction ] --
		// The standard equation :
		//		_accColor = _accColor + ( 1.0f - _accColor.w ) * color;
		// must take alpha correction into account
		// NOTE : if ( color.w == 0 ) then alphaCorrection equals 0.f
		const float alphaCorrection = ( 1.0f -_accColor.w ) * ( 1.0f - __powf( 1.0f - col.w, rayStep * cFullOpacityDistance ) );

		// Accumulate the color
		_accColor.x += alphaCorrection * col.x;
		_accColor.y += alphaCorrection * col.y;
		_accColor.z += alphaCorrection * col.z;
		_accColor.w += alphaCorrection;
	}
}
