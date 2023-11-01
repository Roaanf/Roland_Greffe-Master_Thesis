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
#include "GvStructure/GvNode.h"

/******************************************************************************
 ****************************** INLINE DEFINITION *****************************
 ******************************************************************************/

/******************************************************************************
 * CUDA kernel
 * This is the main GigaVoxels KERNEL
 * It is in charge of casting rays and found the color and depth values at pixels.
 *
 * @param pVolumeTree data structure
 * @param pCache cache
 ******************************************************************************/
template<	class TBlockResolution, bool TFastUpdateMode, bool TPriorityOnBrick, 
			class TSampleShaderType, class TVolTreeKernelType, class TCacheType >
__global__
void RayMapRenderKernel( TVolTreeKernelType pVolumeTree, TCacheType pCache )
{
	// Per-pixel shader instance
	typename TSampleShaderType::KernelType sampleShader;

	// Shared memory
	//__shared__ float3 rayStartInWorld;
	__shared__ float3 rayStartInTree;

	CUDAPM_KERNEL_DEFINE_EVENT( 0 );
	CUDAPM_KERNEL_DEFINE_EVENT( 1 );

	// Compute thread ID
	uint Pid = threadIdx.x + threadIdx.y * TBlockResolution::x;

	// Retrieve current processed pixel position
	// This function modifies the pixel accessing pattern (i.e. z-curve)
	uint2 pixelCoords;
	/*uint2 blockPos;*/ // NOTE : this "block position" parameter seemed not used anymore
	GvRendering::GvRendererKernel::initPixelCoords< TBlockResolution >( Pid, /*blockPos,*/ pixelCoords );

	CUDAPM_KERNEL_START_EVENT( pixelCoords, 0 );

	// Check if were are inside the frame (window or viewport ?)
	bool outOfFrame = ( pixelCoords.x >= k_renderViewContext.frameSize.x ) || ( pixelCoords.y >= k_renderViewContext.frameSize.y );
	// FUTUR optimization
	//
	//bool outOfFrame = ( ( pixelCoords.x >= /*projectedBBoxSize*/k_renderViewContext._projectedBBox.z ) || ( pixelCoords.y >= /*projectedBBoxSize*/k_renderViewContext._projectedBBox.w ) );
	//bool outOfFrame = ( ( pixelCoords.x > /*projectedBBoxSize*/k_renderViewContext._projectedBBox.z ) || ( pixelCoords.y > /*projectedBBoxSize*/k_renderViewContext._projectedBBox.w ) );
	if ( ! outOfFrame )
	//bool inFrame = ( ( pixelCoords.x < k_renderViewContext._projectedBBox.z ) || ( pixelCoords.y < k_renderViewContext._projectedBBox.w ) );
	//if ( inFrame )
	{
		// Read depth from the input depth buffer.
		// Depth buffer contains the Zwindow (distance to camera plane) which is different from Zeye (distance to camera)
		// Zwindow is between 0.0 and 1.0
		// The depth buffer doesn't contain distance values from the camera.
		// The depth values are the perpendicular distance to the plane of the camera.
		float frameDepth = GvRendering::getInputDepth( pixelCoords );		// TO DO : this read memory could be placed before to avoid cycles waiting...

		// FUTUR optimization
		//
		//// Add offset of the projected BBox bottom left corner
		//pixelCoords.x += /*projectedBBoxBottomLeft*/k_renderViewContext._projectedBBox.x;
		//pixelCoords.y += /*projectedBBoxBottomLeft*/k_renderViewContext._projectedBBox.y;

		//// Calculate eye ray in world space

		//float3 pixelVecWP = k_renderViewContext.viewPlaneDirWP
		//					+ k_renderViewContext.viewPlaneXAxisWP * static_cast< float >( pixelCoords.x )
		//					+ k_renderViewContext.viewPlaneYAxisWP * static_cast< float >( pixelCoords.y );

		//rayStartInWorld = k_renderViewContext.viewCenterWP;
		//float3 rayDirInWorld = normalize( pixelVecWP );

		//// Transform the ray from World to Tree Space
		//rayStartInTree = mul( k_renderViewContext.invModelMatrix, rayStartInWorld );	// ce terme pourrait/devrait être calculé sur le HOST car il est constant...
		//
		//// Beware, vectors and transformed by inverse transpose...
		//// TO DO : do correction
		//float3 rayDirInTree = normalize( mulRot( k_renderViewContext.invModelMatrix, rayDirInWorld ) );

		//---------------------------------------
		// TEST
		// Calculate eye ray in tree space
		//
		// Apply the inverse set of transformations to the ray to produce an "inverse transformed ray"
	/*	float3 rayDirInTree = k_renderViewContext.viewPlaneDirTP
							+ k_renderViewContext.viewPlaneXAxisTP * static_cast< float >( pixelCoords.x )
							+ k_renderViewContext.viewPlaneYAxisTP * static_cast< float >( pixelCoords.y );*/
		
		float4 rayDirection = tex2D( rayMapTexture, pixelCoords.x, pixelCoords.y );

		//-------------------
		//rayDirection.z = k_renderViewContext.viewPlaneDirTP.z;
		//-------------------
		//if ( ( rayDirection.x < 0.f ) && ( rayDirection.y < 0.f ) )
		//{
		//	// Write color in color buffer
		//	GvRendering::setOutputColor( pixelCoords, make_uchar4( 0.0f, 255.0f, 0.0f, 255.0f ) );
		//	return;
		//}

		//float4 tmp = mul( k_renderViewContext.invModelMatrix, rayDirection );

		//float3 rayDirInTree = normalize( mulRot( k_renderViewContext.invViewMatrix, /*normalize( */make_float3( rayDirection.x, rayDirection.y, rayDirection.z ) /*)*/ ) );
		float3 rayDirInTree = normalize( make_float3( rayDirection.x, rayDirection.y, rayDirection.z ) );
		
		/*float3*/ rayStartInTree = k_renderViewContext.viewCenterTP;
		// + PASCAL

		//rayDirInTree = normalize( rayDirInTree );

		//float3 rayDirInTree = normalize( make_float3( rayDirection.x, rayDirection.y, rayDirection.z ) );
		//float3 rayDirInTree = normalize( mulRot( k_renderViewContext.invModelMatrix, normalize( make_float3( rayDirection.x, rayDirection.y, rayDirection.z ) ) ) );
		//---------------------------------------
			
		// Intersect the inverse transformed ray with the untransformed object
		float boxInterMin = 0.0f;
		float boxInterMax = 10000.0f;
		int hit = GvRendering::intersectBox( rayStartInTree, rayDirInTree, make_float3( 0.001f ), make_float3( 0.999f ), boxInterMin, boxInterMax );
		bool masked = ! ( hit && ( boxInterMax > 0.0f ) );
		
		// Set closest hit point
		boxInterMin = maxcc( boxInterMin, k_renderViewContext.frustumNear );	// TO DO : attention, c'est faux => frustumNear est en "espace camera" !!
		float t = boxInterMin + sampleShader.getConeAperture( boxInterMin );
		
		// Set farthest hit point
		float tMax = boxInterMax;
		if ( frameDepth < 1.0f )
		{
			// Retrieve the view-space depth from the depth buffer. Only works if w was 1.0f.
			// See: http://www.opengl.org/discussion_boards/ubbthreads.php?ubb=showflat&Number=304624&page=2
			float clipZ = 2.0f * frameDepth - 1.0f;
			float frameT = k_renderViewContext.frustumD / ( -clipZ - k_renderViewContext.frustumC );
			frameT = -frameT;

			tMax = mincc( frameT, boxInterMax );
			//tMax = boxInterMax;
		}

		if ( t == 0.0f || t >= tMax )
		{
			masked = true;
		}

		if ( ! masked )
		{
			// Read color from the input color buffer
			uchar4 frameColor = GvRendering::getInputColor( pixelCoords );

			// Launch N3-tree traversal and rendering
			CUDAPM_KERNEL_START_EVENT( pixelCoords, 1 );
			GvRendering::GvRendererKernel::render< TFastUpdateMode, TPriorityOnBrick >( pVolumeTree, sampleShader, pCache, pixelCoords, rayStartInTree, rayDirInTree, tMax, t );
			CUDAPM_KERNEL_STOP_EVENT( pixelCoords, 1 );

			// Retrieve the accumulated color along the ray
			float4 accCol = sampleShader.getColor();

			// Convert color from uchar [ 0 ; 255 ] to float [ 0.0 ; 1.0 ]
			float4 scenePixelColorF = make_float4( (float)frameColor.x / 255.0f, (float)frameColor.y / 255.0f, (float)frameColor.z / 255.0f, (float)frameColor.w / 255.0f );
			
			// Blend colors (ray and framebuffer)
			float4 pixelColorF = accCol + scenePixelColorF * ( 1.0f - accCol.w );

			// Clamp color to be within the interval [+0.0, 1.0]
			pixelColorF.x = __saturatef( pixelColorF.x );
			pixelColorF.y = __saturatef( pixelColorF.y );
			pixelColorF.z = __saturatef( pixelColorF.z );
			pixelColorF.w = 1.0f;		// <== why 1.0f and not __saturatef( pixelColorF.w ) ?	// Pour éviter une opération OpenGL de ROP ? Car ça a été penser pour rendre en dernier au départ ?
			//pixelColorF.w = __saturatef( pixelColorF.w );
			
			// Convert color from float [ 0.0 ; 1.0 ] to uchar [ 0 ; 255 ]
			frameColor = make_uchar4( (uchar)( pixelColorF.x * 255.0f ), (uchar)( pixelColorF.y * 255.0f ), (uchar)( pixelColorF.z * 255.0f ), (uchar)( pixelColorF.w * 255.0f ) );
			
			// Project the depth and check against the current one
			float pixDepth = 1.0f;

			if ( accCol.w > cOpacityStep )
			{
				float VP = -fabsf( t * rayDirInTree.z );
				//http://www.opengl.org/discussion_boards/ubbthreads.php?ubb=showflat&Number=234519&page=2
				float clipZ = ( VP * k_renderViewContext.frustumC + k_renderViewContext.frustumD ) / -VP;
				
				//pixDepth = clamp( ( clipZ + 1.0f ) / 2.0f, 0.0f, 1.0f );		// TO DO : use __saturatef instead !!
				pixDepth = __saturatef( ( clipZ + 1.0f ) / 2.0f );		// TO DO : use __saturatef instead !!	=====> ( [ x 0.5f ] instead ) ??
			}

			//frameDepth = getFrameDepthIn( pixelCoords );
			frameDepth = min( frameDepth, pixDepth );

			// Write color in color buffer
			GvRendering::setOutputColor( pixelCoords, frameColor );
			
			// Write depth in depth buffer
			GvRendering::setOutputDepth( pixelCoords, frameDepth );
		}
	} // !outOfFrame

	CUDAPM_KERNEL_STOP_EVENT( pixelCoords, 0 );
}
