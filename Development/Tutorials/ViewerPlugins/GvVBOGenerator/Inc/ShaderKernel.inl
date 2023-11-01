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
 * This method is called just before the cast of a ray. Use it to initialize any data
 *  you may need. You may also want to modify the initial distance along the ray (tTree).
 *
 * @param pRayStartTree the starting position of the ray in octree's space.
 * @param pRayDirTree the direction of the ray in octree's space.
 * @param pTTree the distance along the ray's direction we start from.
 ******************************************************************************/
__device__
inline void ShaderKernel::preShadeImpl( const float3& pRayStartTree, const float3& pRayDirTree, float& pTTree )
{
	_accColor = make_float4( 0.f );
}

/******************************************************************************
 * This method is called after the ray stopped or left the bounding
 * volume. You may want to do some post-treatment of the color.
 ******************************************************************************/
__device__
inline void ShaderKernel::postShadeImpl( /*int pCounter*/ )
{
	if ( _accColor.w >= cOpacityThreshold )
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
inline float ShaderKernel::getConeApertureImpl( const float pTTree ) const
{
	// Overestimate to avoid aliasing
	const float scaleFactor = 1.333f;

	// It is an estimation of the size of a voxel at given distance from the camera.
	// It is based on THALES theorem. Its computation is rotation invariant.
	return k_renderViewContext.pixelSize.x * pTTree * ( scaleFactor * k_renderViewContext.frustumNearINV );
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
inline bool ShaderKernel::stopCriterionImpl( const float3& pRayPosInWorld ) const
{
	return ( _accColor.w >= cOpacityThreshold );
}

/******************************************************************************
 * This method is called to know if we should stop at the current octree's level.
 *
 * @param pVoxelSize the voxel's size in the current octree level.
 *
 * @return false if you want to stop at the current octree's level. true otherwise.
 ******************************************************************************/
//__device__
//inline bool ShaderKernel::descentCriterionImpl( const float pVoxelSize ) const
//inline bool ShaderKernel::descentCriterionImpl( const float pVoxelSize, const float pNodeSize, const float pConeAperture ) const
//{
//	if ( ! cApparentMinSizeCriteria )
//	{
//		return true;
//	}
//	else
//	{
//		return ( pNodeSize > ( cApparentMinSize * pConeAperture ) );
//	}
//	return true;
//}

/******************************************************************************
 * This method is called to know if we should stop at the current octree's level.

 * @param pElementSize the desired element size in the current octree level.
 *
 * @param pConeAperture the ConeAperture at the considered point
 *
 * @return false if you want to stop at the current octree's level. true otherwise.
 ******************************************************************************/
__device__
inline bool ShaderKernel::descentCriterionImpl( const float pVoxelSize, const float pNodeSize, const float pConeAperture ) const
{
	// In this example, the "spatial oracle" says that "there is data everywhere".
	//
	// - the renderer has the min/max scene (i.e. mesh) depths information to only renderer between that region.
	// - an optimization criteria is added in the Node Visitor to stop node refinement based on screen-based criteria.

	if( cApparentMinSizeCriteria )
	{
		return ( pNodeSize > ( cApparentMinSize * pConeAperture ) );
	}
	
	return ( pVoxelSize > pConeAperture );
	//return ( pNodeSize > pConeAperture );
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
inline void ShaderKernel::runImpl( const SamplerType& pBrickSampler, const float3 pSamplePosScene, const float3 pRayDir, float& pRayStep, const float pConeAperture )
{
	// TO DO : remove that as shader is un-used

	// Here, we are in a node and we know that there is at least one sphere
	// due to the producer's oracle.

	// brickInfo contient la dimension de la brique dans les 3 premiers parametres puis le nombre de spheres dans la brique en 4eme parametre
	const float4 brickInfo = pBrickSampler._volumeTree->template getSampleValueTriLinear< 0 >( pBrickSampler._brickChildPosInPool - pBrickSampler._volumeTree->brickCacheResINV,
																0.5f * pBrickSampler._volumeTree->brickCacheResINV );

	// brickGeoData contient la position de la brique dans le cube gigavoxel [0; 1] puis la taille de la brique
	const float4 brickData = pBrickSampler._volumeTree->template getSampleValueTriLinear< 0 >( pBrickSampler._brickChildPosInPool - pBrickSampler._volumeTree->brickCacheResINV,
																0.5f * pBrickSampler._volumeTree->brickCacheResINV + make_float3( pBrickSampler._volumeTree->brickCacheResINV.x, 0.f, 0.f ) );
	
	const float3 eyeToBrickVector = make_float3( brickData.x, brickData.y, brickData.z ) - k_renderViewContext.viewCenterTP;

	// Iterate through spheres
	for ( int i = 2; i < brickInfo.w + 2 ; ++i )
	{
		// Retrieve sphere index
		uint3 index3D;
		index3D.x = i % static_cast< uint >( brickInfo.x );
		index3D.y = ( i / static_cast< uint >( brickInfo.x ) ) % static_cast< uint >( brickInfo.y );
		index3D.z = i / static_cast< uint >( brickInfo.x * brickInfo.y );
		
		// Sample data structrure to retrieve sphere data (position and radius)
		const float4 sphereData = pBrickSampler._volumeTree->template getSampleValueTriLinear< 0 >
																	( pBrickSampler._brickChildPosInPool - pBrickSampler._volumeTree->brickCacheResINV,
																	0.5f * pBrickSampler._volumeTree->brickCacheResINV +  make_float3( index3D.x * pBrickSampler._volumeTree->brickCacheResINV.x,
																	index3D.y * pBrickSampler._volumeTree->brickCacheResINV.y,
																	index3D.z * pBrickSampler._volumeTree->brickCacheResINV.z) );
		
		float pointSize = sphereData.w;
		float3 spherePosition = make_float3( sphereData.x, sphereData.y, sphereData.z );

		// Animation
		if ( cShaderAnimation )
		{
			// Animate sphere radius
			pointSize = pointSize - /*scale*/0.25f * ( 0.5f * ( 1.0f + cosf( 2 * 3.141592f * 0.004f * k_currentTime + /* phase */10.0f * sphereData.x ) ) ) * sphereData.w;

			// Animate sphere position
			spherePosition = make_float3( spherePosition.x + brickData.x, spherePosition.y + brickData.y, spherePosition.z + brickData.z );
			spherePosition = spherePosition - /*scale*/0.25f * ( 0.5f * ( 1.0f + cosf( 2 * 3.141592f * 0.004f * k_currentTime + /* phase */7.0f * sphereData.x + /* phase */823.0f * sphereData.y - /* phase */100.0f * sphereData.z ) ) ) * sphereData.w;
			// TO TEST :
			// Utiliser poids faible de "x"
		}

		// Sphere ray-Tracing with anti-aliasing

//
//                 __ __ __ __ __ __ __ __ 
//                |                       |
//                |            (sphere 1) |
//                |               X       |
//   (eye)        |               |  D    |             Ray
//    X---->------|---------------|-------|--------------->
//     -->        |                       |
//     d          |       (sphere 2)      |
//                |            X          |
//                |__ __ __ __ __ __ __ __|
//                X 
//                (brick position)
//

		const float3 eyeToSphere = eyeToBrickVector + spherePosition;
		const float distanceFromEyeToProjectedSphereOnRay = dot( eyeToSphere, pRayDir );
		const float coneAperture = getConeAperture( distanceFromEyeToProjectedSphereOnRay );

		// Compute distance D between sphere center and the ray
		//const float D = sqrtf( max( 0.f , dot( eyeToSphere, eyeToSphere ) - distanceFromEyeToProjectedSphereOnRay * distanceFromEyeToProjectedSphereOnRay ) );
		const float D = sqrtf( dot( eyeToSphere, eyeToSphere ) - distanceFromEyeToProjectedSphereOnRay * distanceFromEyeToProjectedSphereOnRay + 1e-6f/*in case of float error precision*/ );
	
		// Cone aperture
		const float Rv = coneAperture * 0.5f;

		// Opacity
        float alpha = 0.0f;

		// Handle ray-sphere intersection
		if ( D - Rv - pointSize < 0.0f )
		{
			float4 color;

			if ( cShaderUseUniformColor )
			{
				color = cShaderUniformColor;
			}
			else
			{
				// Voxel is in the sphere
				if ( D + Rv < pointSize )
				{
					alpha = 1.0f;
				}
				// Sphere is in the voxel
				else if ( D + pointSize < Rv )
				{
					// Evaluate ratio surface of 2D discs ( pi * r² )
					alpha = ( pointSize * pointSize ) / ( Rv * Rv );
				}
				else
				{
					// Approximation sphere is bigger than voxel
					//
					// Evaluate the 1D ratio of distance of intersection
					alpha = ( pointSize + Rv - D ) / ( 2 * Rv );

					// Clamp value
					__saturatef( alpha );
				}
			
				color = make_float4( ( brickData.x + sphereData.x ) * alpha, ( brickData.y + sphereData.y ) * alpha, ( brickData.z + sphereData.z ) * alpha, alpha );
			}

			_accColor = _accColor + ( 1.0f - _accColor.w ) * color;
		}

		// Stop criterion
		if ( _accColor.w >= cOpacityThreshold )	// stopCriterionImpl() method
		{
			return;
		}
	}
}
