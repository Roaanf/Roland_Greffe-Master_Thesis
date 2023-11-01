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

// GigaVoxels
#include <GvCore/GsPool.h>
#include <GvCore/GsRendererTypes.h>
#include <GvRendering/GsRendererHelpersKernel.h>
#include <GvRendering/GsSamplerKernel.h>
#include <GvRendering/GsBrickVisitorKernel.h>
#include <GvRendering/GsRendererContext.h>
#include <GvPerfMon/GsPerformanceMonitor.h>

#include <GvRendering/GsRendererCUDAKernel.h>

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/**
 * Proxy Geometry textures used to read depth map of closest and farthest faces of a mesh
 */
texture< float, cudaTextureType2D, cudaReadModeElementType > rayMinTex;
texture< float, cudaTextureType2D, cudaReadModeElementType > rayMaxTex;

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ******************************** CLASS USED **********************************
 ******************************************************************************/

/******************************************************************************
 ****************************** KERNEL DEFINITION *****************************
 ******************************************************************************/
 
/**
 * CUDA kernel
 * This is the main GigaVoxels KERNEL
 * It is in charge of casting rays and found the color and depth values at pixels.
 *
 * @param pVolumeTree data structure
 * @param pCache cache
 */
template
<
	class TBlockResolution, bool TFastUpdateMode, bool TPriorityOnBrick, 
	class TSampleShaderType, class TVolTreeKernelType, class TCacheType
>
__global__
void CustomizedRenderKernel( TVolTreeKernelType pVolumeTree, TCacheType pCache );

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

/** 
 * @class GsRendererKernel
 *
 * @brief The GsRendererKernel class provides ...
 *
 * ...
 */
class RendererKernel
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/
	
	/**
	 * This function is used to :
	 * - traverse the data structure (and emit requests if necessary)
	 * - render bricks
	 *
	 * @param pDataStructure data structure
	 * @param pShader shader
	 * @param pCache cache
	 * @param pPixelCoords pixel coordinates in window
	 * @param pRayStartTree ray start point
	 * @param pRayDirTree ray direction
	 * @param ptMaxTree max distance from camera found after box intersection test and comparing with input z (from the scene)
	 * @param ptTree the distance from camera found after box intersection test and comparing with input z (from the scene)
	 */
	template< bool TFastUpdateMode, bool TPriorityOnBrick, class VolTreeKernelType, class SampleShaderType, class TCacheType >
	__device__
	static __forceinline__ void render( VolTreeKernelType& pDataStructure, SampleShaderType& pShader, TCacheType& pCache,
										uint2 pPixelCoords, const float3 pRayStartTree, const float3 pRayDirTree, const float ptMaxTree, float& ptTree );
	
	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

};

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "RendererCUDAKernel.inl"

#endif // !_RENDERER_CUDA_KERNEL_H_
