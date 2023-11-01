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

/**
 * Transfer function texture
 */
texture< float4, cudaTextureType1D, cudaReadModeElementType > transferFunctionTexture;

/******************************************************************************
 * Map a distance to a color (with a transfer function)
 *
 * @param pDistance the distance to map
 *
 * @return The color corresponding to the distance
 ******************************************************************************/
__device__
inline float4 distToColor( float pDistance )
{
	// Fetch data from transfer function
	return tex1D( transferFunctionTexture, pDistance );
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
	//__shared__
	float brickSize;
	//__shared__
	float voxelSize;
	//__shared__
	float brickRes;
	//__shared__
	float levelRes;

	//printf("%f\n",color.x);
	// Test opacity
	if ( color.x > 0.0f )
	{
		float4 voxelNormalAndDist = tex3D( volumeTex, samplePosScene.x, samplePosScene.y, samplePosScene.z );
		//if (voxelNormalAndDist.w<0)
		{
			// Type definition for the noise
			typedef GvUtils::GsNoiseKernel Noise;

			// Retrieving the noise first frequency set with the sample viewer
			float noise_first_frequency = cNoiseParameters.x;

			// Size of the texture in cache (not in 3D world)
			brickSize = brickSampler._volumeTree->brickSizeInCacheNormalized.x;
			voxelSize = brickSampler._volumeTree->brickCacheResINV.x;

			brickRes = brickSize/voxelSize;

			// Calculating the level resolution
			levelRes = 1.f / brickSampler._nodeSizeTree *  brickRes;

			float noise_shell_width = cNoiseParameters.y;

			float noise_first_amplitude ;
			noise_first_amplitude = ( noise_shell_width / ( 2.f - 1.f / ( log2f( levelRes ) ) ) ) ;
			if ( noise_first_amplitude <= 0.f ) {
				noise_first_amplitude = 0.f ;
			}

			// Calculating the brick resolution

			float3 normalVec = normalize( make_float3( voxelNormalAndDist.x, voxelNormalAndDist.y, voxelNormalAndDist.z ) );

			color = distToColor( clamp( 0.5f + 0.5f * voxelNormalAndDist.w * noise_first_frequency, 0.f, 1.f ) );
			// Due to alpha pre-multiplication
			if ( color.w > 0.f )
			{
				// De multiply color with alpha because transfer function data has been pre-multiplied when generated
				color.x /= color.w;
				color.y /= color.w;
				color.z /= color.w;
			}

			// Compute noise
			float dist_noise = 0.0f;
			float amplitude = noise_first_amplitude;
			for ( float frequency = noise_first_frequency; frequency < levelRes; frequency *= 2.f )
			{
				dist_noise += amplitude * Noise::getValue( frequency * ( samplePosScene - voxelNormalAndDist.w * normalVec ) );
				amplitude = amplitude/2.f;
			}

			// Compute alpha
			color.w = clamp( 0.5f - 0.5f * ( voxelNormalAndDist.w + dist_noise ) * static_cast< float >( levelRes ), 0.f, 1.f );
			{
				// De multiply color with alpha because transfer function data has been pre-multiplied when generated
				color.x *= color.w;
				color.y *= color.w;
				color.z *= color.w;
			}

			// Retrieve second channel element : normal
			// float4 normal = brickSampler.template getValue< 1 >( coneAperture );

			// float3 normalVec = normalize( make_float3( voxelNormal.x, voxelNormal.y, voxelNormal.z ) );

			float eps = 0.5f / static_cast< float >( levelRes );
			amplitude = noise_first_amplitude;
			// Compute symetric gradient noise
			float3 grad_noise = make_float3( 0.0f );
			for ( float frequency = noise_first_frequency; frequency < levelRes ; frequency *= 2.f )
			{

				grad_noise.x +=  amplitude * Noise::getValue( frequency * ( samplePosScene + make_float3( eps, 0.0f, 0.0f ) - voxelNormalAndDist.w * normalVec ) )
								-amplitude * Noise::getValue( frequency * ( samplePosScene - make_float3( eps, 0.0f, 0.0f ) - voxelNormalAndDist.w * normalVec ) );

				grad_noise.y +=  amplitude * Noise::getValue( frequency * ( samplePosScene + make_float3( 0.0f, eps, 0.0f ) - voxelNormalAndDist.w * normalVec ) )
								-amplitude * Noise::getValue( frequency * ( samplePosScene - make_float3( 0.0f, eps, 0.0f ) - voxelNormalAndDist.w * normalVec ) );

				grad_noise.z +=  amplitude * Noise::getValue( frequency * ( samplePosScene + make_float3( 0.0f, 0.0f, eps ) - voxelNormalAndDist.w * normalVec ) )
								-amplitude * Noise::getValue( frequency * ( samplePosScene - make_float3( 0.0f, 0.0f, eps ) - voxelNormalAndDist.w * normalVec ) );
				amplitude = amplitude/2.f;
			}

			grad_noise *= 0.5f / eps;
			//grad_noise = normalize(grad_noise);
			normalVec = normalize( normalVec + grad_noise - dot( grad_noise, normalVec ) * normalVec );

			float3 lightVec = normalize( cLightPosition );

			// Lambertian lighting
			float3 rgb = make_float3( color.x, color.y, color.z ) * max( 0.f, dot( normalVec, lightVec ) );

			if ( color.w > 0.f )
			{
				color.x = rgb.x / color.w;
				color.y = rgb.y / color.w;
				color.z = rgb.z / color.w;
			}
			
			// -- [ Opacity correction ] --
			// The standard equation :
			//		_accColor = _accColor + ( 1.0f - _accColor.w ) * color;
			// must take alpha correction into account
			// NOTE : if ( color.w == 0 ) then alphaCorrection equals 0.f
			float alphaCorrection = ( 1.0f -_accColor.w ) * ( 1.0f - powf( 1.0f - color.w, rayStep * 512.f ) );

			// Accumulate the color
			_accColor.x += alphaCorrection * color.x;
			_accColor.y += alphaCorrection * color.y;
			_accColor.z += alphaCorrection * color.z;
			_accColor.w += alphaCorrection;
		}
	}
}
