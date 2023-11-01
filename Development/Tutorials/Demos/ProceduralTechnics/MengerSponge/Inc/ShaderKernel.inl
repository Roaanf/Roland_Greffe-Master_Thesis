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
 ****************************** KERNEL DEFINITION *****************************
 ******************************************************************************/

/******************************************************************************
 * ...
 *
 * @param materialColor ...
 * @param normalVec ...
 * @param lightVec ...
 * @param eyeVec ...
 * @param ambientTerm ...
 * @param diffuseTerm ...
 * @param specularTerm ...
 *
 * @return ...
 ******************************************************************************/
__device__
inline float3 ShaderKernel::shadePointLight( const float3 materialColor, const float3 normalVec, const float3 lightVec, const float3 eyeVec,
	const float3 ambientTerm, const float3 diffuseTerm, const float3 specularTerm )
{
	// Ambient component
	float3 final_color = materialColor * ambientTerm;

	//float lightDist = length( lightVec );
	float3 lightVecNorm = lightVec;
	float lambertTerm = dot( normalVec, lightVecNorm );

	if ( lambertTerm > 0.0f )
	{
		// Diffuse component
		final_color += materialColor * diffuseTerm * lambertTerm;

		float3 halfVec = normalize( lightVecNorm + eyeVec );	// * 0.5f;
		float specular = __powf( max( dot( normalVec, halfVec ), 0.0f ), 64.0f );

		// Specular component
		final_color += make_float3( specular ) * specularTerm;
	}

	return final_color;
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
template< typename SamplerType >
__device__
inline void ShaderKernel::runImpl( const SamplerType& brickSampler, const float3 samplePosScene,
										const float3 rayDir, float& rayStep, const float coneAperture )
{
	// Retrieve first channel element : color
	float4 color = brickSampler.template getValue< 0 >( coneAperture );

	// Retrieve second channel element : normal
	//float4 normal = brickSampler.template getValue< 1 >( coneAperture );

	if ( color.w > 0.0f )
	{
		float3 grad = make_float3( 0.0f );

		float gradStep = rayStep;	// * 0.5f;

		float4 v0, v1;

		v0 = brickSampler.template getValue< 0 >( coneAperture, make_float3(  gradStep, 0.0f, 0.0f ) );
		v1 = brickSampler.template getValue< 0 >( coneAperture, make_float3( -gradStep, 0.0f, 0.0f ) );
		grad.x = v0.w - v1.w;

		v0 = brickSampler.template getValue< 0 >( coneAperture, make_float3( 0.0f,  gradStep, 0.0f ) );
		v1 = brickSampler.template getValue< 0 >( coneAperture, make_float3( 0.0f, -gradStep, 0.0f ) );
		grad.y = v0.w - v1.w;

		v0 = brickSampler.template getValue< 0 >( coneAperture, make_float3( 0.0f, 0.0f,  gradStep ) );
		v1 = brickSampler.template getValue< 0 >( coneAperture, make_float3( 0.0f, 0.0f, -gradStep ) );
		grad.z = v0.w - v1.w;

		if ( length( grad ) > 0.0f )
		{
			grad =- grad;
			grad = normalize( grad );
			//float3 grad = normalize( make_float3( normal.x, normal.y, normal.z ) );

			//float vis = 1.0f;

			float3 lightVec = normalize( cLightPosition - samplePosScene );
			float3 viewVec = ( -1.0f * rayDir );

			float3 rgb;
			rgb.x = color.x;
			rgb.y = color.y;
			rgb.z = color.z;
			rgb = shadePointLight( rgb, grad, lightVec, viewVec, make_float3( 0.2f ), make_float3( 0.8f ), make_float3( 0.9f ) );
			
			// Due to alpha pre-multiplication
			//
			// Note : inspecting device SASS code, assembly langage seemed to compute (1.f / color.w) each time, that's why we use a constant and multiply
			const float alphaPremultiplyConstant = 1.f / color.w;
			color.x = rgb.x * alphaPremultiplyConstant;
			color.y = rgb.y * alphaPremultiplyConstant;
			color.z = rgb.z * alphaPremultiplyConstant;
		}
		else
		{
			// Problem !
			// In this case, the normal generated by the gradient is null...
			// That generates visual artefacts...
			//col = make_float4( 0.0f );
			//color = make_float4( 1.0, 0.f, 0.f, 1.0f );
			// Ambient : no shading
			//float3 final_color = materialColor * ambientTerm;
			float vis = 1.0f;
			float3 rgb;
			rgb.x = color.x; rgb.y = color.y; rgb.z = color.z;
			float3 final_color = rgb * make_float3( 0.2f * vis );

			// Due to alpha pre-multiplication
			//
			// Note : inspecting device SASS code, assembly langage seemed to compute (1.f / color.w) each time, that's why we use a constant and multiply
			const float alphaPremultiplyConstant = 1.f / color.w;
			color.x = final_color.x * alphaPremultiplyConstant;
			color.y = final_color.y * alphaPremultiplyConstant;
			color.z = final_color.z * alphaPremultiplyConstant;
		}

		// -- [ Opacity correction ] --
		// The standard equation :
		//		_accColor = _accColor + ( 1.0f - _accColor.w ) * color;
		// must take alpha correction into account
		// NOTE : if ( color.w == 0 ) then alphaCorrection equals 0.f
		const float alphaCorrection = ( 1.0f -_accColor.w ) * ( 1.0f - __powf( 1.0f - color.w, rayStep * 512.f ) );

		// Accumulate the color
		_accColor.x += alphaCorrection * color.x;
		_accColor.y += alphaCorrection * color.y;
		_accColor.z += alphaCorrection * color.z;
		_accColor.w += alphaCorrection;
	}
}
