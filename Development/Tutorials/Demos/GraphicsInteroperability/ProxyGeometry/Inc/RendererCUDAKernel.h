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

#ifndef _RENDERER_CUDA_KERNEL_H_
#define _RENDERER_CUDA_KERNEL_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// Cuda
#include <helper_math.h>

// Gigavoxels
#include <GvCore/GsPool.h>
#include <GvCore/GsRendererTypes.h>
#include <GvRendering/GsRendererHelpersKernel.h>
#include <GvRendering/GsSamplerKernel.h>
#include <GvRendering/GsNodeVisitorKernel.h>
#include <GvRendering/GsBrickVisitorKernel.h>
#include <GvRendering/GsRendererContext.h>
#include <GvPerfMon/GsPerformanceMonitor.h>
#include <GvRendering/GsRendererCUDAKernel.h>

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

texture< float, cudaTextureType2D, cudaReadModeElementType > rayMinTex;
texture< float, cudaTextureType2D, cudaReadModeElementType > rayMaxTex;

/******************************************************************************
 ****************************** INLINE DEFINITION *****************************
 ******************************************************************************/

/******************************************************************************
 * CUDA kernel ...
 *
 * @param volumeTree ...
 * @param gpuCache ...
 ******************************************************************************/
template<	class BlockResolution, bool fastUpdateMode, bool priorityOnBrick, 
			class SampleShaderType, class VolTreeKernelType, class GPUCacheType>
__global__
void RenderKernel( VolTreeKernelType volumeTree, GPUCacheType gpuCache )
{
	// per-pixel shader instance
	typename SampleShaderType::KernelType sampleShader;

	//__shared__ float3 rayStartInWorld;
	__shared__ float3 rayStartInTree;

	//float3 rayDirInWorld;
	//float3 rayDirInTree;

	CUDAPM_KERNEL_DEFINE_EVENT(0);
	CUDAPM_KERNEL_DEFINE_EVENT(1);

	//Compute thread ID
	uint Pid=threadIdx.x+threadIdx.y*BlockResolution::x;

	// pixel position
	uint2 pixelCoords;
	uint2 blockPos;
	GvRendering::GsRendererKernel::initPixelCoords<BlockResolution>(Pid, /*blockPos,*/ pixelCoords);

	CUDAPM_KERNEL_START_EVENT(pixelCoords, 0);

	bool outOfFrame=(pixelCoords.x >= k_renderViewContext.frameSize.x) || (pixelCoords.y >= k_renderViewContext.frameSize.y);

	if(!outOfFrame)
	{
		//// calculate eye ray in world space

		//float3 pixelVecWP = k_renderViewContext.viewPlaneDirWP
		//					+ k_renderViewContext.viewPlaneXAxisWP*(float)pixelCoords.x
		//					+ k_renderViewContext.viewPlaneYAxisWP*(float)pixelCoords.y;

		//rayStartInWorld = k_renderViewContext.viewCenterWP;
		//rayDirInWorld = normalize(pixelVecWP);

		//// transform the ray from world to tree space
		//rayStartInTree = mul(k_renderViewContext.invModelMatrix, rayStartInWorld);

		//rayDirInTree = normalize(mulRot(k_renderViewContext.invModelMatrix, rayDirInWorld));

		//---------------------------------------
		// TEST
		// Calculate eye ray in tree space
		float3 rayDirInTree = k_renderViewContext.viewPlaneDirTP
							+ k_renderViewContext.viewPlaneXAxisTP * static_cast< float >( pixelCoords.x )
							+ k_renderViewContext.viewPlaneYAxisTP * static_cast< float >( pixelCoords.y );
		/*float3*/ rayStartInTree = k_renderViewContext.viewCenterTP;
		// + PASCAL
		rayDirInTree = normalize( rayDirInTree );
		//---------------------------------------
		
		bool masked=false;

		//float t = 0.0f;
		//float tMax;

		/*float boxInterMin = 0.0f; float boxInterMax = 10000.0f;
		int hit = GvRendering::intersectBox( rayStartInTree, rayDirInTree, make_float3(0.001f), make_float3(0.999f), boxInterMin, boxInterMax );
		hit = hit && boxInterMax > 0.0f;*/

		////
		//if ( hit )
		//{
		//	printf( "test : intersectBox" );
		//}
		////

		// Beware !!!!!!!
		// BEFORE : z-eye from OpenGL
		// NOW : in z-Window
		float boxInterMin = tex2D( rayMinTex, pixelCoords.x, pixelCoords.y );
		float boxInterMax = tex2D( rayMaxTex, pixelCoords.x, pixelCoords.y );

		if ( boxInterMin < 0.f )
		{
			//GvRendering::setOutputColor( pixelCoords, make_uchar4( 255, 0, 255, 255 ) );
			//GvRendering::setOutputDepth( pixelCoords, 1.f );
			return;
		}
		if ( boxInterMax < 0.f )
		{
			//GvRendering::setOutputColor( pixelCoords, make_uchar4( 255, 0, 255, 255 ) );
			//GvRendering::setOutputDepth( pixelCoords, 1.f );
			return;
		}
		
		int hit = boxInterMax > 0.f;
		masked = masked || ( ! hit );
		boxInterMin = maxcc( boxInterMin, k_renderViewContext.frustumNear );

		//t = boxInterMin - sampleShader.getConeAperture(boxInterMin);

		// Read color and depth from the input buffers
		//uchar4 frameColor = k_renderViewContext.inFrameColor.get(pixelCoords);
		uchar4 frameColor = GvRendering::getInputColor( pixelCoords );
		//float frameDepth = GvCore::getFrameDepthIn(pixelCoords);
		float frameDepth = GvRendering::getInputDepth( pixelCoords );

		//// Retrieve the view-space depth from the depth buffer. Only works if w was 1.0f.
		//// See: http://www.opengl.org/discussion_boards/ubbthreads.php?ubb=showflat&Number=304624&page=2
		//float frameT;
		//float clipZ;
		//clipZ = 2.0f * frameDepth - 1.0f;
		//frameT = k_renderViewContext.frustumD / ( -clipZ - k_renderViewContext.frustumC );
		//frameT = -frameT;
		//tMax = mincc( frameT, boxInterMax + sampleShader.getConeAperture( boxInterMax ) );
		////tMax = boxInterMax;

		//float t = boxInterMin + sampleShader.getConeAperture( boxInterMin );
		float t = boxInterMin - sampleShader.getConeAperture( boxInterMin );
		float tMax = boxInterMax;
		if ( frameDepth < 1.0f )
		{
			// Retrieve the view-space depth from the depth buffer. Only works if w was 1.0f.
			// See: http://www.opengl.org/discussion_boards/ubbthreads.php?ubb=showflat&Number=304624&page=2
			float clipZ = 2.0f * frameDepth - 1.0f;
			float frameT = k_renderViewContext.frustumD / ( -clipZ - k_renderViewContext.frustumC );
			frameT = -frameT;

			//tMax = mincc( frameT, boxInterMax );
			tMax = mincc( frameT, boxInterMax )	+ sampleShader.getConeAperture( boxInterMax );
			//tMax = boxInterMax;
		}
		
		if ( t == 0.0f || t >= tMax )
		{
			masked = true;
			//
			//printf( "\ntest : masked = %f", t );
			//
		}

		if ( ! masked )
		{
			//
			//printf( "test : renderVolTree_Std" );
			//

			CUDAPM_KERNEL_START_EVENT(pixelCoords, 1);

			GvRendering::GsRendererKernel::render< fastUpdateMode, priorityOnBrick >(volumeTree, sampleShader, gpuCache, pixelCoords, rayStartInTree, rayDirInTree, tMax, t);

			CUDAPM_KERNEL_STOP_EVENT(pixelCoords, 1);

			// Get the accumulated color
			float4 accCol = sampleShader.getColor();

			// Update color
			float4 scenePixelColorF = make_float4((float)frameColor.x/255.0f, (float)frameColor.y/255.0f, (float)frameColor.z/255.0f, (float)frameColor.w/255.0f);

			float4 pixelColorF = accCol + scenePixelColorF * (1.0f - accCol.w);
			pixelColorF.x = __saturatef(pixelColorF.x);
			pixelColorF.y = __saturatef(pixelColorF.y);
			pixelColorF.z = __saturatef(pixelColorF.z);
			pixelColorF.w = 1.0f;

			frameColor = make_uchar4((uchar)(pixelColorF.x*255.0f), (uchar)(pixelColorF.y*255.0f), (uchar)(pixelColorF.z*255.0f), (uchar)(pixelColorF.w*255.0f));
			//frameColor = make_uchar4((uchar)(boxInterMax - boxInterMin) * 255.f, 0, 0, 255);
			//frameColor = make_uchar4((uchar)(boxInterMin * 255.f), 0, 0, 255);

			// project the depth and check against the current one
			float pixDepth = 1.0f;
			float VP = -fabsf(t * rayDirInTree.z);
			//http://www.opengl.org/discussion_boards/ubbthreads.php?ubb=showflat&Number=234519&page=2
			//clipZ = (VP * k_renderViewContext.frustumC + k_renderViewContext.frustumD) / -VP;
			float clipZ = (VP * k_renderViewContext.frustumC + k_renderViewContext.frustumD) / -VP;

			if ( accCol.w > cOpacityDepthThreshold )
			{
				pixDepth=clamp((clipZ+1.0f)/2.0f, 0.0f, 1.0f);
			}

			//frameDepth = GvCore::getFrameDepthIn(pixelCoords);
			frameDepth = GvRendering::getInputDepth( pixelCoords );
			//frameDepth=min(frameDepth, pixDepth);
		}

		// TODO: Check if it's really working
		// write color + depth
		//k_renderViewContext.outFrameColor.set(pixelCoords, frameColor);
		GvRendering::setOutputColor( pixelCoords, frameColor );
		//GvCore::setFrameDepthOut(pixelCoords, frameDepth);
		GvRendering::setOutputDepth( pixelCoords, frameDepth );

	} //!outOfFrame

	CUDAPM_KERNEL_STOP_EVENT(pixelCoords, 0);
}

#endif // !_RENDERER_CUDA_KERNEL_H_
