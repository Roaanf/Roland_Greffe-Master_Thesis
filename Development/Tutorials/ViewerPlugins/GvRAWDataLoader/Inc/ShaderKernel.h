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

#ifndef _SHADER_KERNEL_H_
#define _SHADER_KERNEL_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include <GvRendering/GsIRenderShader.h>
#include <GvUtils/GsCommonShaderKernel.h>

// Cuda
#include <cuda_runtime.h>
#include <cuda_texture_types.h>

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/**
 * Light position
 */
__constant__ float3 cLightPosition;

/**
 * Shader threshold
 */
__constant__ float cShaderThresholdLow;

/**
 * Shader threshold
 */
__constant__ float cShaderThresholdHigh;

/**
 * Full opacity distance
 */
__constant__ float cFullOpacityDistance;

/**
 * Gradient step
 */
__constant__ float cGradientStep;

/*
0 = Default	
1 = Isosurface
2 = Max
3 = Mean
4 = XRay
5 = XRay Gradient
6 = XRay Color
*/
__constant__ int cRenderMode;

__constant__ int cDataResolution;

__constant__ float cXRayConst;


/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ******************************** CLASS USED **********************************
 ******************************************************************************/

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

/** 
 * @class ShaderLoadKernel
 *
 * @brief The ShaderLoadKernel struct provides...
 *
 * ...
 */
class ShaderKernel
:	public GvUtils::GsCommonShaderKernel
,	public GvRendering::GsIRenderShader< ShaderKernel >
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
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
	 */
	__device__
	inline float3 shadePointLight( float3 materialColor, float3 normalVec, float3 lightVec, float3 eyeVec,
		float3 ambientTerm, float3 diffuseTerm, float3 specularTerm );

	/**
	 * ...
	 *
	 * @param brickSampler ...
	 * @param samplePosScene ...
	 * @param rayDir ...
	 * @param rayStep ...
	 * @param coneAperture ...
	 */
	template< typename BrickSamplerType >
	__device__
	inline void runImpl( const BrickSamplerType& brickSampler, const float3 samplePosScene,
		const float3 rayDir, float& rayStep, const float coneAperture );

	
	__device__
	inline bool stopCriterionImpl(const float3& pRayPosInWorld) const;

	void resetValues();

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	int firstTouch = 2;

	float maxCol = 0;

	float meanCol = 0;

	uint sampleCount = 0;

	int currMode = 0;

	float currXRayIntensity = 1.0f;

};

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "ShaderKernel.inl"

#endif // !_SHADER_KERNEL_H_
