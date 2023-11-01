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

/**
 * ...
 *
 * @param ... ...
 *
 * @return ...
 */
typedef float ( *func )( float x );

/**
 * ...
 *
 * @param ... ...
 *
 * @return ...
 */
__device__
float identity( float value )
{
	return value;
}

/**
 * ...
 *
 * @param ... ...
 *
 * @return ...
 */
__device__
float absolute( float value )
{
	return 2.f * fabs( value ) - 1.f;
}

/**
 * ...
 *
 * @param ... ...
 *
 * @return ...
 */
__device__
float minusAbsolute( float value )
{
	return 2.f * ( 1.f - fabs( value ) ) - 1.f;
}

/**
 * ...
 *
 * @param ... ...
 *
 * @return ...
 */
__device__
float innerShell( float value )
{
	return -1.f;
}

/**
 * ...
 *
 * @param ... ...
 *
 * @return ...
 */
__device__
float outerShell( float value )
{
	return 1.f;
}

/**
 * ...
 */
__device__ func pfunc0 = identity;
__device__ func pfunc1 = absolute;
__device__ func pfunc2 = minusAbsolute;
__device__ func pfunc3 = innerShell;
__device__ func pfunc4 = outerShell;

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
 * This method is called just before the cast of a ray. Use it to initialize any data
 *  you may need. You may also want to modify the initial distance along the ray (tTree).
 *
 * @param pRayStartTree the starting position of the ray in octree's space.
 * @param pRayDirTree the direction of the ray in octree's space.
 * @param pTTree the distance along the ray's direction we start from.
 ******************************************************************************/
__device__
inline void ShaderKernel::preShadeImpl( const float3& rayStartTree, const float3& rayDirTree, float& tTree )
{
	_accColor = make_float4( 0.f );
}

/******************************************************************************
 * This method is called after the ray stopped or left the bounding
 * volume. You may want to do some post-treatment of the color.
 ******************************************************************************/
__device__
inline void ShaderKernel::postShadeImpl()
{
	if ( _accColor.w >= cOpacityStep )
	{
		_accColor.w = 1.f;
	}
}

/******************************************************************************
 * This method returns the cone aperture for a given distance.
 *
 * @param pTTree the current distance along the ray's direction.
 *
 * @return the cone aperture
 ******************************************************************************/
__device__
inline float ShaderKernel::getConeApertureImpl( const float tTree ) const
{
	// Overestimate to avoid aliasing
	//const float scaleFactor = 1.333f;
	//return k_renderViewContext.pixelSize.x * tTree * scaleFactor * k_renderViewContext.frustumNearINV;

	return k_renderViewContext.pixelSize.x * tTree * GvRendering::k_voxelSizeMultiplier * k_renderViewContext.frustumNearINV;
}

/******************************************************************************
 * This method returns the final rgba color that will be written to the color buffer.
 *
 * @return the final rgba color.
 ******************************************************************************/
__device__
inline float4 ShaderKernel::getColorImpl() const
{
	return _accColor;
}

/******************************************************************************
 * This method is called before each sampling to check whether or not the ray should stop.
 *
 * @param pRayPosInWorld the current ray's position in world space.
 *
 * @return true if you want to continue the ray. false otherwise.
 ******************************************************************************/
__device__
inline bool ShaderKernel::stopCriterionImpl( const float3& rayPosInWorld ) const
{
	return ( _accColor.w >= cOpacityStep );
}

/******************************************************************************
 * This method is called to know if we should stop at the current octree's level.
 *
 * @param pVoxelSize the voxel's size in the current octree level.
 *
 * @return false if you want to stop at the current octree's level. true otherwise.
 ******************************************************************************/
__device__
inline bool ShaderKernel::descentCriterionImpl( const float voxelSize ) const
{
	return true;
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
	//printf( "%f\n", color.x );

	//__shared__
	float brickSize;
	//__shared__
	float voxelSize;
	//__shared__
	float brickRes;
	//__shared__
	float levelRes;

	// TO DO
	// - move this elsewhere, because it's constant during the kernel...
	float ( *noiseFunctionModifier )( float );
	switch ( cNoiseFunctionModifier )
	{
		case 0:
			noiseFunctionModifier = pfunc0;
			break;

		case 1:
			noiseFunctionModifier = pfunc1;
			break;

		case 2:
			noiseFunctionModifier = pfunc2;
			break;

		case 3:
			noiseFunctionModifier = pfunc3;
			break;

		case 4:
			noiseFunctionModifier = pfunc4;
			break;

		default:
			break;
	}
		
	// Test opacity
	if ( color.x > 0.0f )
	{
		float4 voxelNormalAndDist = tex3D( volumeTex, samplePosScene.x, samplePosScene.y, samplePosScene.z );
		//if ( voxelNormalAndDist.w < 0.0f )
		{
			// Type definition for the noise
			typedef GvUtils::GsNoiseKernel Noise;

			// Retrieving the noise first frequency set with the sample viewer
			//float noise_first_frequency = cNoiseFirstFrequency;

			// Size of the texture in cache (not in 3D world)
			brickSize = brickSampler._volumeTree->brickSizeInCacheNormalized.x;
			voxelSize = brickSampler._volumeTree->brickCacheResINV.x;

			// Calculating the brick resolution
			brickRes = brickSize / voxelSize;

			// Calculating the level resolution
			levelRes = 1.f / brickSampler._nodeSizeTree * brickRes;

			//float noise_shell_width = cNoiseShellWidth;

			// Compute an upper bound to the Noise Dist (a is the first noise amplitude)
			//     Noise = sum[ n = 1:N ] { ( a / ( 2^n ) ) } = ( a ) * sum[ n = 1:N ] { ( 1 / ( 2^n ) ) }
			//
			// A fine Upper bound of this sum is :
			//     ( a ) * ( 2 - 1/N )
			//
			// As we double the frequency until it reaches levelRes, we can deduce that N = log2( levelRes )
			// an other upper bound is :
			//     ( a ) * 2
			const float noise_first_amplitude = ( ( cNoiseShellWidth / 2.f ) / 2.f );
			//const float noise_first_amplitude = ( noise_shell_width / ( 2.f - 1.f / ( log2f( levelRes ) ) ) );
			
			float3 normalVec = normalize( make_float3( voxelNormalAndDist.x, voxelNormalAndDist.y, voxelNormalAndDist.z ) );

			// Compute noise
			float dist_noise = 0.0f;
			float amplitude = noise_first_amplitude;
			/*for ( float frequency = noise_first_frequency; frequency < levelRes; frequency *= 2.f )
			{
				if ( cAnimation )
				{
					dist_noise += amplitude * noiseFunctionModifier( Noise::getValue( frequency * sinf( 0.001f * k_currentTime ) * ( samplePosScene - voxelNormalAndDist.w * normalVec ) ) );
				}
				else
				{
					dist_noise += amplitude * noiseFunctionModifier( Noise::getValue( frequency * ( samplePosScene - voxelNormalAndDist.w * normalVec ) ) );
				}
				amplitude = amplitude * 0.5f;
			}*/
			// TO DO
			// - sinf( 0.001f * k_currentTime ) is common to all thread, and stay identity during loop => put it in shared of constante memory ?
			float animationCoeff = 1.f;
			if ( cAnimation )
			{
				animationCoeff = sinf( 0.001f * k_currentTime );
			}

			for ( float frequency = cNoiseFirstFrequency; frequency < levelRes; frequency *= 2.f )
			{
				dist_noise += amplitude * noiseFunctionModifier( Noise::getValue( animationCoeff * frequency * ( samplePosScene - voxelNormalAndDist.w * normalVec ) ) );
				amplitude = amplitude * 0.5f;
			}
			
			// Compute color and opacity
				
			if ( cNoiseDisplacement == 0 )
			{
				// Color
				const float a = 10.f;
				const float f = 1.f;
				dist_noise = sinf( f * ( samplePosScene.x + a * dist_noise ) );
				color = distToColor( clamp( 0.5f - 0.5f * dist_noise, 0.f, 1.f ) );
			}
			else
			{
				// Color
				color = distToColor( clamp( 0.5f - 0.5f * ( voxelNormalAndDist.w + dist_noise ) / ( noise_first_amplitude ), 0.f, 1.f ) );

				// Normal
				const float eps = 0.5f / static_cast< float >( levelRes );
				// - compute symetric gradient noise
				amplitude = noise_first_amplitude;
				float3 grad_noise = make_float3( 0.0f );
				for ( float frequency = cNoiseFirstFrequency; frequency < levelRes ; frequency *= 2.f )
				{
					grad_noise.x +=  amplitude * noiseFunctionModifier( Noise::getValue( frequency * ( samplePosScene + make_float3( eps, 0.0f, 0.0f ) - voxelNormalAndDist.w * normalVec ) ) )
									-amplitude * noiseFunctionModifier( Noise::getValue( frequency * ( samplePosScene - make_float3( eps, 0.0f, 0.0f ) - voxelNormalAndDist.w * normalVec ) ) );

					grad_noise.y +=  amplitude * noiseFunctionModifier( Noise::getValue( frequency * ( samplePosScene + make_float3( 0.0f, eps, 0.0f ) - voxelNormalAndDist.w * normalVec ) ) )
									-amplitude * noiseFunctionModifier( Noise::getValue( frequency * ( samplePosScene - make_float3( 0.0f, eps, 0.0f ) - voxelNormalAndDist.w * normalVec ) ) );

					grad_noise.z +=  amplitude * noiseFunctionModifier( Noise::getValue( frequency * ( samplePosScene + make_float3( 0.0f, 0.0f, eps ) - voxelNormalAndDist.w * normalVec ) ) )
									-amplitude * noiseFunctionModifier( Noise::getValue( frequency * ( samplePosScene - make_float3( 0.0f, 0.0f, eps ) - voxelNormalAndDist.w * normalVec ) ) );

					amplitude = amplitude * 0.5f;
				}

				grad_noise *= 0.5f / eps;
				//grad_noise = normalize( grad_noise );
				normalVec = normalize( normalVec + grad_noise - dot( grad_noise, normalVec ) * normalVec );
			}

			// Lambertian lighting
			const float3 lightVec = normalize( cLightPosition );
			const float3 rgb = make_float3( color.x, color.y, color.z ) * max( 0.f, dot( normalVec, lightVec ) );

			// Due to alpha pre-multiplication
			if ( color.w > 0.0f )
			{
				// Note : inspecting device SASS code, assembly langage seemed to compute (1.f / color.w) each time, that's why we use a constant and multiply
				const float alphaPremultiplyConstant = 1.f / color.w;
				color.x = rgb.x * alphaPremultiplyConstant;
				color.y = rgb.y * alphaPremultiplyConstant;
				color.z = rgb.z * alphaPremultiplyConstant;
			}

			// -- [ Opacity correction ] --
			// The standard equation :
			//		_accColor = _accColor + ( 1.0f - _accColor.w ) * color;
			// must take alpha correction into account
			// NOTE : if ( color.w == 0 ) then alphaCorrection equals 0.f
			const float alphaCorrection = ( 1.0f -_accColor.w ) * ( 1.0f - __powf( 1.0f - color.w, rayStep * 8.f ) );

			// Accumulate the color
			_accColor.x += alphaCorrection * color.x;
			_accColor.y += alphaCorrection * color.y;
			_accColor.z += alphaCorrection * color.z;
			_accColor.w += alphaCorrection;
		}
	}
}
